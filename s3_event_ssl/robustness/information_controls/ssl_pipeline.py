
from __future__ import annotations
import copy, gc, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ssl_common import *
from ssl_data import *
from ssl_neural import *
from ssl_statistics import evaluate_predictions,paired_bootstrap,decomposition,field,field_metrics


def check_history(history,strict=True):
    h=Path(history).resolve();paths=resolve_paths(h);base_manifest=None;reports=[]
    for seed,entry in paths.items():
        root=Path(entry['input_root']);m=load_json(root/'metadata/stage4_input_manifest.json')
        module('train_event_ssl.py').validate_input_contract(m)
        mask,lineage=alignment_mask(m);normal=load_json(root/'metadata/normalizer.json')
        require(normal['fitted_on']=='A_train only','Normalizer is not A_train-only')
        require(normal['numeric_feature_names']==m['numeric_feature_names_after_expansion'],'Normalizer feature order mismatch')
        require(np.isfinite(normal['mean']).all() and (np.array(normal['std'])>0).all(),'Invalid normalizer')
        schema={k:m[k] for k in ['numeric_feature_names_after_expansion','categorical_input_source_columns','categorical_hash_buckets']}
        schema['normalizer']=normal
        if base_manifest is None:base_manifest=schema
        else:require(schema==base_manifest,'Seeds have different ordered input schema/normalizer; do not silently combine')
        for kind in ('full','shuffled'):
            ck=torch_load(entry[kind]);cfg=ck['config'];sh=ck['model_shapes']
            require(int(cfg['seed'])==int(seed),f'Checkpoint seed identity mismatch: {kind} {seed}')
            require(int(sh['n_num'])==len(mask),'Checkpoint numeric width mismatch')
            require(tuple(map(int,cfg['future_steps']))==(1,2,4),'Original future horizons differ from protocol')
            em=Path(entry[kind+'_evaluation'])/'metadata/stage4_event_ssl_evaluation_manifest.json'
            require(em.exists(),f'Historical evaluation manifest missing {em}')
            ev=load_json(em)
            require(ev.get('checkpoint_sha256')==sha(entry[kind]),'Historical checkpoint reference mismatch')
            require(ev.get('train_script_sha256')==load_json(ROOT/'source_contract.json')['files']['train_event_ssl.py'],
                    'Original training source differs from supplied source; exact modified SSL source is required')
            for split in ('A_val','B_confirm'):
                table_path(Path(entry[kind+'_evaluation'])/'predictions'/f'stage4_event_ssl_predictions_{split}')
            reports.append({'seed':int(seed),'kind':kind,'checkpoint':entry[kind],'best_epoch':ck.get('best_epoch'),
                     'shape':sh,'config':cfg,'historical_amp':cfg.get('amp_dtype')})
    part=load_partition(h);ref=discover_reference(h)
    if strict:require(ref is not None,'Completed four-term per-row export not discovered under output root. Preserve the actual formal outputs here, or pass --fourterm-root; do not refit C3.')
    return paths,part,ref,reports

class Context:
    def __init__(self,history,smoke=False,threads=8,loader_workers=0,infer_batch=16,chunk=512,fourterm_root=None):
        self.history=Path(history).resolve();self.smoke=smoke;self.threads=threads;self.loader_workers=loader_workers
        self.infer_batch=infer_batch;self.chunk=chunk
        self.out=self.history/run_name(smoke)
        self.out.mkdir(parents=True,exist_ok=True)
        self.paths,self.part,self.ref,self.modelinfo=check_history(history,strict=fourterm_root is None)
        if fourterm_root:
            self.ref=discover_reference(Path(fourterm_root))
            require(self.ref is not None,'Explicit four-term root contains no completed prediction export')
        self.seeds=(42,) if smoke else ALL_SEEDS;self.newseeds=(42,) if smoke else NEW_SEEDS
        self.maxusers=32 if smoke else None;self.nboot=20 if smoke else 2000
        self.protocol=load_json(ROOT/'protocol.json')
        self.s2=copy.deepcopy(self.protocol['s2']);self.s3=copy.deepcopy(self.protocol['s3'])
        if smoke:
            self.s2.update(epochs=1,batch_size=32,train_max_rows=512,val_max_rows=256)
            self.s3.update(epochs=1,batch_size=16,probe_rows=512)
        self._data={}
        torch.set_num_threads(threads)

        torch.use_deterministic_algorithms(True)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:pass
        self.code=code_digest();self.signature=None;self.contract=None
    def data(self,seed,split):
        k=(seed,split)
        if k not in self._data:self._data[k]=Prepared(self.paths[str(seed)]['input_root'],split,self.maxusers)
        return self._data[k]
    def verify_inputs(self):
        if hasattr(self,'records'):
            for r in self.records:verify_record(r)


def reference_paths(ref):
    if ref is None:return None
    return {'run':str(Path(ref['run']).resolve()),'rows':str(Path(ref['rows']).resolve()),
            **{k:str(Path(ref[k]['path']).resolve()) for k in ('models','freeze','complete')}}


def resolved_source_selection(ctx):

    paths=set();selected={}
    for seed,e in ctx.paths.items():
        root=Path(e['input_root']).resolve()
        selected[str(seed)]={'input_root':str(root)}
        paths.update(root.glob('metadata/*.json'))
        for split in SPLITS:
            paths.update((root/split).glob('*.mmap'));paths.add(root/split/'sequence_offsets.npy')
        for kind in ('full','shuffled'):
            checkpoint=Path(e[kind]).resolve();er=Path(e[kind+'_evaluation']).resolve()
            selected[str(seed)][kind]=str(checkpoint);selected[str(seed)][kind+'_evaluation']=str(er)
            paths.add(checkpoint);paths.update(er.glob('metadata/*.json'))
            paths.update(er.glob('tables/stage4_event_ssl_structural_metrics_*.csv'))
            for split in ('A_val','B_confirm'):
                paths.add(table_path(er/'predictions'/f'stage4_event_ssl_predictions_{split}'))
    partition=sorted(str(Path(p).resolve()) for p in ctx.part['source_files'])
    paths.update(map(Path,partition));ref=reference_paths(ctx.ref)
    if ref is not None:paths.update(Path(ref[k]) for k in ('rows','models','freeze','complete'))
    return paths,{'history_root':str(ctx.history.resolve()),'seeds':selected,
                  'partition_files':partition,'reference':ref}


def reference_stage_input(ctx):

    require(reference_paths(ctx.ref)==ctx.source_bindings['reference'],
            'Selected four-term reference changed after S0')
    ctx.verify_inputs()
    return ctx.reference_identity


def s0(ctx):


    paths,bindings=resolved_source_selection(ctx)
    records,ctx.signature=fingerprint_sources(paths,bindings)
    ctx.records=records;ctx.source_bindings=bindings
    bypath={r['path']:r for r in records};ref=bindings['reference']
    ctx.reference_identity=None if ref is None else {
        'run':ref['run'],**{k:{f:bypath[ref[k]][f] for f in ('path','bytes','sha256')}
                           for k in ('rows','models','freeze','complete')}}

    ctx.contract=objhash({'version':VERSION,'code':ctx.code,'sources':ctx.signature,'smoke':ctx.smoke,
                         'protocol':ctx.protocol,'threads':ctx.threads,'loader_workers':ctx.loader_workers,
                         'infer_batch':ctx.infer_batch,'chunk':ctx.chunk})
    def perform(dst):
        rows=[];allusers=[]
        for seed in ctx.seeds:
            per=[]
            for split in SPLITS:
                d=ctx.data(seed,split);audit=d.validate();audit.update(seed=seed,split=split);rows.append(audit)
                per.append(np.unique(d.values('uid')))
            for i in range(3):
                for j in range(i):require(np.intersect1d(per[i],per[j]).size==0,'User leakage between splits')
            if seed==42:allusers=per
            else:
                for i in range(3):require(np.array_equal(allusers[i],per[i]),'Different cohort users across seeds')
        mask,lineage=alignment_mask(ctx.data(42,'A_train').manifest)
        write_csv(lineage,dst/'feature_lineage.csv');write_csv(rows,dst/'input_integrity.csv')
        save_json(records,dst/'source_records.json');save_json(ctx.paths,dst/'path_resolution.json')
        save_json(bindings,dst/'resolved_source_bindings.json')
        save_json(ctx.reference_identity,dst/'reference_identity.json')
        save_json(ctx.part,dst/'frozen_partition.json');save_json(ctx.modelinfo,dst/'checkpoint_contracts.json')
        save_json(existing_results(ctx.history),dst/'existing_evidence_inventory.json')
        save_json(ctx.ref,dst/'fourterm_reference.json')
        save_json({'macro_targets_recomputed':False,'shuffle_recomputed':['row order','step index','successor labels'],
          'shuffle_not_recomputed':['S/E/D','alignment/neutral fields','time gaps'],
          'interpretation':'record-level shuffled-to-ordered transfer; not isolated temporal-order effect'},dst/'shuffle_scope.json')
        save_json({'code':ctx.code,'sources_signature':ctx.signature,'run_contract':ctx.contract,
                   'smoke':ctx.smoke,'all_seeds':ctx.seeds,'new_seeds':ctx.newseeds,
                   'CPU':True,'synthetic':(ctx.history/'_SYNTHETIC_FIXTURE.json').exists()},dst/'run_contract.json')
        save_json({'S1':'same frozen readout before and after each event',
                   'S2':'paired state-only and state-plus-event prediction heads',
                   'S3':'joint removal of alignment-derived inputs and future targets',
                   'completion':'independent of effect direction or benchmark advantage'},dst/'analysis_scope.json')
    p=stage(ctx.out,'S0',ctx.contract,{'sources':ctx.signature,'reference':ctx.reference_identity},perform)
    save_json({'contract':ctx.contract,'code':ctx.code,'sources':ctx.signature,'smoke':ctx.smoke,
               'reference':ctx.reference_identity,'run_name':run_name(ctx.smoke)},ctx.out/'RUN_CONTRACT.json')
    return p


def bridge_metrics(ctx,data,predictions,dst):
    if data.split!='B_confirm':return
    reference_stage_input(ctx)
    rb=load_reference_rows(ctx.ref,data,dst)
    if rb is None:return
    ii=np.load(Path(dst)/'reference_rows.npy')
    if len(ii)==0:return
    r=np.asarray(rb[ii]);X=r[:,:2];Y=r[:,2:4];uid=data.values('uid',ii)
    _,ui,c=np.unique(uid,return_inverse=True,return_counts=True);w=1./c[ui];emp=field(X,Y-X,w)
    baseline={'four_term':r[:,4:6],'C3':r[:,6:8]}
    baseline.update({n:np.asarray(a[ii],float) for n,a in predictions.items()})
    grid=[];metrics=[];fs={}
    for name,N in baseline.items():
        f=field(X,N-X,w);fs[name]=f
        d=field_metrics(emp['field'],f['field'],emp['mass'],emp['support'])
        metrics.append({'model':name,'view':'exact_mechanism_export_anchor_intersection','rows':len(ii),
                        'quantity':'endpoint N-X','target':'Y-X','not_actual_velocity':True,**d})
    for name in predictions:
        S=emp['support'];a=emp['field']-fs['C3']['field'];b=fs[name]['field']-fs['C3']['field']
        d=field_metrics(a,b,emp['mass'],S);metrics.append({'model':name,'view':'C3_correction_descriptive','rows':len(ii),
            'quantity':'endpoint N-X','target':'Y-X','not_actual_velocity':True,
            'reference_operation':'subtract C3 endpoint field from both endpoint fields',**d})
    write_csv(metrics,Path(dst)/'C3_bridge_metrics.csv')


def run_s1(ctx):
    for seed in ctx.seeds:
        for kind in ('full','shuffled'):
            e=ctx.paths[str(seed)];checkpoint=e[kind]
            for split in ('A_val','B_confirm'):
                data=ctx.data(seed,split)
                def perform(dst):
                    m,ck=load_model(checkpoint,data)
                    p=predict_native(m,data,dst,ctx.chunk,ctx.infer_batch)
                    archive_bridge(data,e[kind+'_evaluation'],p,dst,checkpoint)
                    cur={'proposal':p[:,:2],'same_readout':p[:,:2]}
                    nxt={'proposal':p[:,4:6],'same_readout':p[:,2:4]}
                    evaluate_predictions(data,cur,nxt,ctx.part,dst/'evaluation')
                    historical=np.load(dst/'historical_S_P.npy',mmap_mode='r')
                    archived_metric_identity(data,e[kind+'_evaluation'],historical,ctx.part,dst)
                    evaluate_predictions(data,{'archived_proposal':historical[:,:2]},
                        {'archived_proposal':historical[:,2:4]},ctx.part,dst/'historical_view')
                    decomposition(data,p,dst/'diagnostics')
                    bridge_metrics(ctx,data,nxt,dst/'bridge')
                    if seed==42 and kind=='full' and split=='B_confirm':
                        comp=[('same_readout_minus_proposal_next_MSE','same_readout','proposal','next_MSE'),
                              ('same_readout_minus_proposal_velocity_D2','same_readout','proposal','velocity_D2')]
                        paired_bootstrap(data,cur,nxt,comp,dst/'inference',ctx.nboot)
                    del m,p;gc.collect()
                ctx.verify_inputs()
                stage(ctx.out,f'S1/{kind}/seed_{seed}/{split}',ctx.contract,
                      {'S0':receipt(ctx.out/'S0'),'checkpoint':sha(checkpoint),'split':split,
                       'reference':reference_stage_input(ctx)},perform)


def run_s2(ctx):
    for seed in ctx.newseeds:
        entry=ctx.paths[str(seed)];checkpoint=entry['full'];data=ctx.data(seed,'A_train')
        model,ck=load_model(checkpoint,data)
        samples={}
        for split,cap in [('A_train',ctx.s2['train_max_rows']),('A_val',ctx.s2['val_max_rows'])]:
            d=ctx.data(seed,split);idx=selected_indices(d.n,cap,42)
            samples[split]=stage(ctx.out,f'S2/seed_{seed}/sample_{split}',ctx.contract,
                 {'checkpoint':sha(checkpoint),'split':split,'selected_indices_hash':objhash(idx)},
                 lambda dst,d=d,idx=idx:collect_features(model,d,idx,dst,ctx.chunk,ctx.infer_batch))
        heads={};headpaths={}
        for mode in ('H','HZ'):
            hp=stage(ctx.out,f'S2/seed_{seed}/head_{mode}',ctx.contract,
                     {'train':receipt(samples['A_train']),'val':receipt(samples['A_val']),'settings':ctx.s2},
                     lambda dst,mode=mode:fit_head(model,samples['A_train'],samples['A_val'],mode,seed,dst,ctx.s2,ctx.loader_workers))
            head=initialized_head(model,seed);head.load_state_dict(torch_load(hp/'best_head.pt')['state_dict']);head.eval()
            heads[mode]=head;headpaths[mode]=hp
        for split in ('A_val','B_confirm'):
            d=ctx.data(seed,split)
            def perform(dst,d=d):
                p=predict_heads(model,heads,d,dst,ctx.chunk,ctx.infer_batch)
                cur={'H':p[:,:2],'HZ':p[:,:2]};nxt={'H':p[:,2:4],'HZ':p[:,4:6]}
                evaluate_predictions(d,cur,nxt,ctx.part,dst/'evaluation');bridge_metrics(ctx,d,nxt,dst/'bridge')
                if seed==42 and d.split=='B_confirm':
                    paired_bootstrap(d,cur,nxt,[('H_minus_HZ_next_MSE','H','HZ','next_MSE'),
                      ('H_minus_HZ_endpoint_D2','H','HZ','endpoint_D2')],dst/'inference',ctx.nboot)
            stage(ctx.out,f'S2/seed_{seed}/{split}',ctx.contract,
                   {'heads':{k:receipt(p) for k,p in headpaths.items()},'split':split,
                    'reference':reference_stage_input(ctx)},perform)
        del model,heads;gc.collect()


def run_s3(ctx):
    for seed in ctx.newseeds:
        tr=ctx.data(seed,'A_train');va=ctx.data(seed,'A_val');mask,_=alignment_mask(tr.manifest)
        template=torch_load(ctx.paths[str(seed)]['full']);paths={};probes={};evalpaths={}
        for variant in ('aug','clean'):
            def train(dst,variant=variant):
                train_pure(tr,va,template,seed,variant,mask,dst,ctx.s3,ctx.loader_workers,ctx.smoke)
            tp=stage(ctx.out,f'S3/seed_{seed}/{variant}/training',ctx.contract,
                {'template':sha(ctx.paths[str(seed)]['full']),'settings':ctx.s3,'mask':mask.tolist(),
                 'reference_policy':'CPU-matched fixed-epoch replay; original best-only GPU checkpoints not substituted'},train)
            paths[variant]=tp
            def probe_stage(dst,tp=tp,variant=variant):
                model,ck=load_model(tp/'fixed_epoch_model.pt',tr)
                idx=selected_indices(tr.n,ctx.s3['probe_rows'],42)
                collect_features(model,tr,idx,dst/'sample',ctx.chunk,ctx.infer_batch,
                                 mask if variant=='clean' else None)
                fit_probe(dst/'sample',dst)
            pp=stage(ctx.out,f'S3/seed_{seed}/{variant}/probe',ctx.contract,
                {'encoder':receipt(tp),'alpha':1.,'sample_seed':42,'sample_rows':ctx.s3['probe_rows']},probe_stage)
            probes[variant]=pp
            for split in ('A_val','B_confirm'):
                d=ctx.data(seed,split)
                def pred(dst,d=d,tp=tp,pp=pp,variant=variant):
                    model,ck=load_model(tp/'fixed_epoch_model.pt',d);probe=load_json(pp/'probe.json')
                    p=predict_native(model,d,dst,ctx.chunk,ctx.infer_batch,mask if variant=='clean' else None,probe,future_common_keep=~mask)
                    evaluate_predictions(d,{variant:p[:,:2]},{variant:p[:,2:4]},ctx.part,dst/'evaluation')
                    bridge_metrics(ctx,d,{variant:p[:,2:4]},dst/'bridge')
                ep=stage(ctx.out,f'S3/seed_{seed}/{variant}/{split}',ctx.contract,
                     {'encoder':receipt(tp),'probe':receipt(pp),'split':split,
                      'reference':reference_stage_input(ctx)},pred)
                evalpaths[(variant,split)]=ep

        if seed==42:
            def infer(dst):
                d=ctx.data(seed,'B_confirm');p={v:np.load(evalpaths[(v,'B_confirm')]/'trajectory.npy',mmap_mode='r') for v in ('aug','clean')}
                paired_bootstrap(d,{v:a[:,:2] for v,a in p.items()},{v:a[:,2:4] for v,a in p.items()},
                    [('clean_minus_aug_current_MSE','clean','aug','current_MSE'),
                     ('clean_minus_aug_velocity_D2','clean','aug','velocity_D2')],dst,ctx.nboot)
            stage(ctx.out,f'S3/seed_{seed}/inference',ctx.contract,
                  {'predictions':{v:receipt(evalpaths[(v,'B_confirm')]) for v in ('aug','clean')}},infer)
        gc.collect()


def assert_complete(ctx):
    paths=[ctx.out/'S0']
    for seed in ctx.seeds:
        for k in ('full','shuffled'):
            for split in ('A_val','B_confirm'):paths.append(ctx.out/f'S1/{k}/seed_{seed}/{split}')
    for seed in ctx.newseeds:
        for split in ('A_train','A_val'):paths.append(ctx.out/f'S2/seed_{seed}/sample_{split}')
        for mode in ('H','HZ'):paths.append(ctx.out/f'S2/seed_{seed}/head_{mode}')
        for split in ('A_val','B_confirm'):paths.append(ctx.out/f'S2/seed_{seed}/{split}')
        for v in ('aug','clean'):
            for step in ('training','probe','A_val','B_confirm'):paths.append(ctx.out/f'S3/seed_{seed}/{v}/{step}')
        h=[load_json(ctx.out/f'S2/seed_{seed}/head_{m}/head_contract.json') for m in ('H','HZ')]
        for key in ('initial_state_sha256','train_keys_sha256','val_keys_sha256','settings'):
            require(h[0][key]==h[1][key],f'Paired head contract differs: {key}')
        t=[load_json(ctx.out/f'S3/seed_{seed}/{v}/training/training_contract.json') for v in ('aug','clean')]
        require(t[0]['initial_state_sha256']==t[1]['initial_state_sha256'],'Pure matched initial states differ')
        if seed==42:paths.append(ctx.out/f'S3/seed_{seed}/inference')
    ctx.verify_inputs()
    return {str(p.relative_to(ctx.out)):verify_committed(p,ctx.contract) for p in paths}
