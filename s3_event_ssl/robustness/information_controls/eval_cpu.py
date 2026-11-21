#!/usr/bin/env python3

from __future__ import annotations
import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

GPU_SEEDS=(2026,666,606,37,4669)
VARIANTS=('aug','clean')


def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def objhash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def npy_sha(value):
    import numpy as np
    buf=io.BytesIO();np.save(buf,value,allow_pickle=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()

def _load_api(code_root):
    root=Path(code_root).resolve()
    sys.path.insert(0,str(root))
    modules={name:importlib.import_module(name) for name in ('ssl_common','ssl_data','ssl_neural','ssl_statistics','ssl_pipeline')}
    for name,module in modules.items():require(Path(module.__file__).resolve().parent==root,f'Wrong imported module: {name}')
    return SimpleNamespace(common=modules['ssl_common'],data=modules['ssl_data'],neural=modules['ssl_neural'],
                           statistics=modules['ssl_statistics'],pipeline=modules['ssl_pipeline'])

class EvalContext:

    def data(self,seed,split):
        require(seed==self.seed,'Cross-seed data access in one evaluator')
        key=(seed,split)
        if key not in self._data:self._data[key]=self.api.data.Prepared(self.input_root,split)
        return self._data[key]
    def verify_inputs(self):
        for record in self.records:self.api.common.verify_record(record)

def build_context(args,api):
    import numpy as np
    import torch
    c=api.common
    require(args.seed in GPU_SEEDS,'Seed42 is read-only; only five newly GPU-trained seeds are allowed')
    require(args.threads==8,'Frozen CPU evaluation requires 8 intra-op threads')
    require(len(args.contract)==64 and all(x in '0123456789abcdef' for x in args.contract),'Contract must be a lower-case SHA256')
    history=Path(args.history_root).resolve();out=Path(args.output_root).resolve()
    require(out==(history/c.RUN_NAME).resolve(),'Output must be the existing declared run root; writes are restricted to the five seed subtrees')
    old=read_json(out/'RUN_CONTRACT.json')
    cpu_contract=old['contract'];frozen_code=c.code_digest()
    require(args.contract!=cpu_contract,'GPU training and CPU preparation require distinct contracts')
    require(old['contract']==cpu_contract and old['code']==frozen_code and not old['smoke'],'Original S0/RUN identity mismatch')
    execution=read_json(out/'S3_EXECUTION_CONTRACT.json')
    require(execution['contract']==args.contract and execution['old_contract']==cpu_contract,'Wrong GPU execution contract')
    require(execution['cpu_seeds']==[42] and set(map(int,execution['gpu_assignment']))==set(GPU_SEEDS),'Unexpected mixed-device seed scope')
    require(execution['scripts_sha256']['eval_cpu.py']==sha(__file__),'Evaluator differs from the frozen extension bundle')
    c.verify_committed(out/'S0',cpu_contract)
    s0=out/'S0';s0run=read_json(s0/'run_contract.json')
    require(s0run['run_contract']==cpu_contract and s0run['code']==frozen_code,'S0 inner identity mismatch')
    paths=read_json(s0/'path_resolution.json');bindings=read_json(s0/'resolved_source_bindings.json')
    require(Path(bindings['history_root']).resolve()==history,'History path is not the S0-resolved source')
    root=Path(paths[str(args.seed)]['input_root']).resolve()
    require(root==Path(bindings['seeds'][str(args.seed)]['input_root']).resolve(),'Resolved input root differs from S0 binding')
    source={str(Path(r['path']).resolve()):r for r in read_json(s0/'source_records.json')}
    selected=set(root.glob('metadata/*.json'))
    for split in ('A_train','A_val','B_confirm'):
        selected.update((root/split).glob('*.mmap'));selected.add(root/split/'sequence_offsets.npy')

    root42=Path(paths['42']['input_root']).resolve()
    selected.update([root42/'metadata/stage4_input_manifest.json',root42/'metadata/normalizer.json'])
    selected.update(Path(p) for p in bindings['partition_files'])
    template=Path(paths[str(args.seed)]['full']).resolve();selected.add(template)
    require(bindings['reference'] is not None,'Frozen C3 reference missing')
    selected.update(Path(bindings['reference'][k]) for k in ('rows','models','freeze','complete'))
    records=[]
    for p in sorted({p.resolve() for p in selected}):
        require(str(p) in source,f'Input was not part of the verified original S0: {p}')
        record=source[str(p)];c.verify_record(record,full=True);records.append(record)
    manifest=read_json(root/'metadata/stage4_input_manifest.json');m42=read_json(root42/'metadata/stage4_input_manifest.json')
    for key in ('numeric_feature_names_after_expansion','categorical_input_source_columns','categorical_hash_buckets'):
        require(manifest[key]==m42[key],f'Seed input schema differs: {key}')
    normal=read_json(root/'metadata/normalizer.json')
    require(normal==read_json(root42/'metadata/normalizer.json') and normal['fitted_on']=='A_train only','Original A_train normalizer differs')
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:require(torch.get_num_interop_threads()==1,'CPU interop thread setting differs')
    ctx=EvalContext();ctx.api=api;ctx.seed=args.seed;ctx.out=out;ctx.history=history;ctx.contract=args.contract
    ctx.paths=paths;ctx.input_root=root;ctx.records=records;ctx.source_bindings=bindings;ctx._data={}
    ctx.execution=execution;ctx.original_run=old;ctx.original_s0_run=s0run
    ctx.cpu_contract=cpu_contract;ctx.frozen_code=frozen_code
    ctx.template_path=template;ctx.template_sha=source[str(template)]['sha256']
    ctx.ref=read_json(s0/'fourterm_reference.json');ctx.reference_identity=read_json(s0/'reference_identity.json')
    require(ctx.reference_identity==old['reference'],'Reference identity differs between S0 and root receipt')
    ctx.part=read_json(s0/'frozen_partition.json')
    for key in ('mean','scale','standardized_centers','centers'):ctx.part[key]=np.asarray(ctx.part[key],float)
    ctx.chunk=512;ctx.infer_batch=16;ctx.s3=read_json(Path(args.code_root)/'protocol.json')['s3']
    integrity={};users=[]
    for split in ('A_train','A_val','B_confirm'):
        data=ctx.data(args.seed,split);integrity[split]=data.validate();users.append(np.unique(data.values('uid')))
    for i in range(3):
        for j in range(i):require(not np.intersect1d(users[i],users[j]).size,'Learner leakage between splits')
    tr=ctx.data(args.seed,'A_train');idx=api.neural.selected_indices(tr.n,ctx.s3['probe_rows'],42)
    expected=read_json(out/'S3/seed_42/aug/probe/_SUCCESS.json')
    require(expected['contract']==cpu_contract,'Seed42 probe has not retained its original CPU identity')
    c.verify_committed(out/'S3/seed_42/aug/probe',cpu_contract)
    sampled={'sample/indices.npy':npy_sha(idx),
             'sample/keys.npy':npy_sha(np.column_stack([tr.values('uid',idx),tr.values('step',idx)])),
             'sample/X.npy':npy_sha(tr.values('X',idx)),'sample/Y.npy':npy_sha(tr.values('Y',idx))}
    for key,value in sampled.items():require(value==expected['outputs'][key],f'Probe sample/anchor differs from original seed42: {key}')
    ctx.sample_hashes=sampled;ctx.input_integrity=integrity
    ctx.eval_identity={'module_sha256':sha(__file__),'frozen_code':frozen_code,'original_cpu_contract':cpu_contract,
                       'new_training_contract':args.contract,'seed':args.seed,'input_root':str(root),
                       'original_S0_receipt':c.receipt(s0),'sample_hashes':sampled,
                       'CPU_threads':8,'CPU_interop_threads':1,'chunk':512,'sequence_batch':16,
                       'probe_rows':300000,'probe_seed':42,'alpha':1.,'bootstrap_requested':False}
    ctx.eval_spec=objhash(ctx.eval_identity)
    return ctx

def validate_training_pair(ctx):

    import numpy as np
    require(ctx.seed in GPU_SEEDS,'Seed42 is read-only; no evaluation commits permitted')
    c=ctx.api.common;mask,_=ctx.api.data.alignment_mask(ctx.data(ctx.seed,'A_train').manifest)
    contracts={};paths={}
    for variant in VARIANTS:
        p=ctx.out/f'S3/seed_{ctx.seed}/{variant}/training';c.verify_committed(p,ctx.contract)
        tc=read_json(p/'training_contract.json');stage_record=read_json(p/'_SUCCESS.json')
        ck=c.torch_load(p/'fixed_epoch_model.pt');cfg=ck['config'];inputs=stage_record['inputs']
        require(tc['inputs']==inputs and tc['stage_contract']==ctx.contract and not tc['smoke'],'GPU training receipt/config inputs disagree')
        require(inputs['source_S0']==c.receipt(ctx.out/'S0') and inputs['source_S0_contract']==ctx.cpu_contract,'Training used a different S0')
        require(inputs['source_signature']==ctx.original_s0_run['sources_signature'],'Training source set differs')
        require(Path(inputs['input_root']).resolve()==ctx.input_root and inputs['seed']==ctx.seed and inputs['variant']==variant,'Training input root/seed/variant differs')
        require(inputs['template']==ctx.template_sha and Path(inputs['template_path']).resolve()==ctx.template_path,'Training architecture template differs')
        require(inputs['frozen_code']==ctx.frozen_code and inputs['settings']==ctx.s3 and inputs['mask']==mask.tolist(),'Training science contract differs')
        require(inputs['gpu_driver_sha256']==ctx.execution['scripts_sha256']['train_gpu.py'],'GPU training driver identity differs')
        require(inputs['train_rows']==ctx.data(ctx.seed,'A_train').n and inputs['validation_rows']==ctx.data(ctx.seed,'A_val').n and not inputs['smoke'],'Training cohort or formal/smoke identity differs')
        require(tc['variant']==variant and ck.get('variant')==variant,'GPU variant identity mismatch')
        require(cfg['seed']==ctx.seed and cfg['model_kind']=='pure_ssl','GPU checkpoint seed/kind mismatch')
        require(cfg['epochs']==ck['epoch']==8 and ck.get('checkpoint_selection')=='fixed epoch','Training did not reach the fixed endpoint')
        require(cfg['lambda_state']==cfg['lambda_closure']==0 and cfg['lambda_future']==1,'Unexpected encoder supervision')
        require(ck.get('training_backend')=='cuda_fp32' and cfg['amp_dtype']=='none_gpu_fp32','New seed is not a GPU-FP32 training endpoint')
        runtime=tc['numerical_runtime']
        require(runtime==ck['numerical_runtime']==inputs['numerical_runtime'],'Training runtime records disagree')
        require(runtime['precision']=='GPU float32' and not runtime['AMP'] and not runtime['TF32'] and not runtime['compile'],'GPU precision contract differs')
        require(runtime['loader_workers']==0 and runtime['deterministic_algorithms'],'GPU loader/determinism contract differs')
        require(ck.get('future_only') and ck.get('macro_labels_in_loss') is False,'Pure future-gradient checkpoint contract missing')
        require(tc['config']==cfg and tc['selection']=='fixed epoch, never M/Psi validation','Training config/selection receipt mismatch')
        for key in ('epochs','seq_len','stride','min_seq_len','warmup_steps','batch_size','lr','weight_decay','future_steps'):
            require(cfg[key]==ctx.s3[key],f'Frozen training setting changed: {key}')
        require(np.array_equal(np.asarray(ck['clean_mask'],bool),mask),'Checkpoint feature lineage differs')
        removed=np.flatnonzero(mask).tolist() if variant=='clean' else []
        require(tc['features_excluded_from_target_denominator']==removed,'Wrong future-target exclusion')
        history=read_json(p/'training_history.json')
        require([r['epoch'] for r in history]==list(range(1,9)),'Incomplete epoch history')
        require(all(r['selection']=='none; fixed epoch' for r in history),'Validation selection changed')
        require(all(r['batches']==tc['original_data_windows']//ctx.s3['batch_size'] for r in history),'Training update budget differs')
        contracts[variant]=tc;paths[variant]=p
        del ck
    require(contracts['aug']['initial_state_sha256']==contracts['clean']['initial_state_sha256'],'Aug/clean initial states differ')
    require(contracts['aug']['config']==contracts['clean']['config'],'Aug/clean training config/device differs')
    require(contracts['aug']['numerical_runtime']==contracts['clean']['numerical_runtime'],'Aug/clean backend/precision runtime differs')
    require(contracts['aug']['original_data_windows']==contracts['clean']['original_data_windows'],'Training windows differ')
    return paths,mask,contracts

def evaluate_seed(ctx):

    import gc
    api=ctx.api;c=api.common;n=api.neural
    paths,mask,contracts=validate_training_pair(ctx)
    tr=ctx.data(ctx.seed,'A_train');outputs={}
    for variant in VARIANTS:
        tp=paths[variant]
        def probe_stage(dst,tp=tp,variant=variant):
            model,ck=n.load_model(tp/'fixed_epoch_model.pt',tr)
            idx=n.selected_indices(tr.n,ctx.s3['probe_rows'],42)
            n.collect_features(model,tr,idx,dst/'sample',ctx.chunk,ctx.infer_batch,mask if variant=='clean' else None)
            for key,value in ctx.sample_hashes.items():require(c.sha(dst/key)==value,f'Saved probe sample differs: {key}')
            n.fit_probe(dst/'sample',dst)
        pp=c.stage(ctx.out,f'S3/seed_{ctx.seed}/{variant}/probe',ctx.contract,
                   {'encoder':c.receipt(tp),'alpha':1.,'sample_seed':42,'sample_rows':ctx.s3['probe_rows'],
                    'CPU_evaluation_spec':ctx.eval_spec},probe_stage)
        outputs[variant]={'training':c.receipt(tp),'probe':c.receipt(pp)}
        for split in ('A_val','B_confirm'):
            data=ctx.data(ctx.seed,split)
            def predict(dst,data=data,tp=tp,pp=pp,variant=variant):
                model,ck=n.load_model(tp/'fixed_epoch_model.pt',data);probe=c.load_json(pp/'probe.json')
                p=n.predict_native(model,data,dst,ctx.chunk,ctx.infer_batch,mask if variant=='clean' else None,
                                   probe,future_common_keep=~mask)
                api.statistics.evaluate_predictions(data,{variant:p[:,:2]},{variant:p[:,2:4]},ctx.part,dst/'evaluation')
                api.pipeline.bridge_metrics(ctx,data,{variant:p[:,2:4]},dst/'bridge')
            ep=c.stage(ctx.out,f'S3/seed_{ctx.seed}/{variant}/{split}',ctx.contract,
                       {'encoder':c.receipt(tp),'probe':c.receipt(pp),'split':split,
                        'reference':api.pipeline.reference_stage_input(ctx),'CPU_evaluation_spec':ctx.eval_spec},predict)
            outputs[variant][split]=c.receipt(ep)
        gc.collect()
    ctx.verify_inputs()
    result={'seed':ctx.seed,'contract':ctx.contract,'evaluation_identity':ctx.eval_identity,
            'input_integrity':ctx.input_integrity,'outputs':outputs,'bootstrap_run':False,
            'CPU42_bootstrap':'untouched; applies only to the previously completed CPU seed42 pair',
            'scientific_direction_not_an_acceptance_gate':True}
    c.save_json(result,ctx.out/f'S3/seed_{ctx.seed}/CPU_EVALUATION_COMPLETE.json')
    return result

def main(argv=None):
    ap=argparse.ArgumentParser(description='Evaluate paired GPU checkpoints with the frozen CPU probe and field estimands.')
    for key in ('code-root','history-root','output-root'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--seed',type=int,choices=GPU_SEEDS,required=True)
    ap.add_argument('--contract',required=True);ap.add_argument('--threads',type=int,default=8)
    args=ap.parse_args(argv)

    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=str(args.threads);os.environ['MKL_NUM_THREADS']=str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS']=str(args.threads);os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    api=_load_api(args.code_root)

    require(args.seed in GPU_SEEDS,'Read-only seed requested')
    expected=(args.history_root.resolve()/api.common.RUN_NAME).resolve()
    require(args.output_root.resolve()==expected,'Wrong output root')
    import fcntl
    seed_root=expected/f'S3/seed_{args.seed}';require(seed_root.is_dir(),'Missing completed GPU seed directory')
    require(seed_root.resolve()==seed_root,'A seed directory must not redirect evaluation writes through a symlink')
    with (seed_root/'CPU_EVALUATION.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another evaluator owns this seed') from exc
        try:
            ctx=build_context(args,api);result=evaluate_seed(ctx)
            print(json.dumps({'seed':args.seed,'complete':True,'bootstrap_run':False,'contract':result['contract']}),flush=True)
        finally:fcntl.flock(lock,fcntl.LOCK_UN)
    return 0

if __name__=='__main__':raise SystemExit(main())
