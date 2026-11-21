#!/usr/bin/env python3

from __future__ import annotations
import argparse, gzip, hashlib, json, os, platform, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from cn_common import *


def input_inventory(data):
    paths=[]
    for channel in ('bundle','question','study'):paths.extend(data[channel].values())
    paths.extend(str(find_table(Path(data['contents_root'])/n)) for n in ('questions_clean','bundles_clean','lectures_clean','explanations_clean'))
    return {p:dict(size=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns) for p in sorted(set(paths))}


def check_inventory(p):
    for f,item in p['input_inventory'].items():
        s=Path(f).stat()
        if s.st_size!=item['size'] or s.st_mtime_ns!=item['mtime_ns']:raise RuntimeError(f'Preprocessed input changed since protocol freeze: {f}')


def choose_work(output,explicit=None,smoke=False):
    output=Path(output).resolve();work=Path(explicit).resolve() if explicit else output/(NEW_DIR+('_smoke' if smoke else ''))
    if work==output or work==output/'stage1' or (output/'stage1') in work.parents:raise ValueError('Refusing to write to an existing Stage-1 tree.')
    protected=['stage1_construction_matched_null','stage1_construction_matched_null_confirm','semantic_specificity_control','null_referenced_downstream_recovery','six_seed_null_referenced_recovery']
    if any(work==output/x or output/x in work.parents for x in protected):raise ValueError('Refusing to overwrite an existing analysis tree.')
    return work


def initialize(args):
    out=Path(args.output_root).resolve();work=choose_work(out,args.work_root,args.smoke)
    work.mkdir(parents=True,exist_ok=True)
    with lock(work/'locks/init.lock'):
        if (work/'protocol.json').exists():
            p=load_protocol(work)
            if p['replicates']!=args.replicates or p['seed']!=args.seed or p['smoke']!=args.smoke or p['smoke_users']!=(args.smoke_users if args.smoke else 0):raise RuntimeError('Existing protocol differs; no silent overwrite.')
            print(f'[init] frozen protocol already exists: {work}',flush=True);return work
        if not args.smoke and args.replicates!=999:raise ValueError('Formal protocol fixes 999 replicates per null and split; reduced counts are smoke-only.')
        if args.smoke and (args.replicates<2 or args.smoke_users<2):raise ValueError('Smoke needs >=2 users and >=2 replicates.')
        sources=source_paths(args.repo_root);data=resolve_data(out,args.data_root)
        source_hashes={k:sha256(sources[k]) for k in ('stage1','semantic','original_null')}
        frozen={}
        for s in SPLITS:frozen[str(find_table(out/'stage1/splits'/f'{s}_users'))]=''
        for name in ('A_train_question_itemEB_correctness_priors','A_train_tag_correctness_priors'):
            frozen[str(find_table(out/'stage1/metadata'/name))]=''
        for f in ('stage1/metadata/stage1_empirical_v3_manifest.json','stage1/metadata/bundle_chunk_user_completeness_audit.json','stage1/splits/split_manifest.json','stage1/dynamics/candidate_regions/MR_PsiA/A_train_primary_convergence_core_mask.npy','stage1/dynamics/candidate_regions/MR_PsiA/training_convergence_thresholds.json'):
            frozen[str(out/f)]=''
        for coord in ('psi','phi'):
            for s in SPLITS:

                frozen[str(find_table(archived_field_base(out,s,coord)))]=''

            if coord=='phi':
                for name in ('A_train_primary_convergence_core_mask.npy','training_convergence_thresholds.json'):
                    f=out/'semantic_specificity_control/coordinate/stage1/dynamics/candidate_regions/MR_PhiAI'/name
                    if f.exists():frozen[str(f)]=''
        for name in ('questions_clean','bundles_clean','lectures_clean','explanations_clean'):
            frozen[str(find_table(Path(data['contents_root'])/name))]=''
        for f in frozen:frozen[f]=sha256(f)
        if not read_json(out/'stage1/metadata/bundle_chunk_user_completeness_audit.json').get('gate_passed',False):raise RuntimeError('Original bundle user-completeness audit did not pass.')
        p=dict(version=VERSION,output_root=str(out),work_root=str(work),sources=sources,source_hashes=source_hashes,code_hashes=code_hashes(),data=data,input_inventory=input_inventory(data),frozen_files=frozen,seed=args.seed,replicates=args.replicates,smoke=args.smoke,smoke_users=args.smoke_users if args.smoke else 0,modes=['null_a','null_b'],coordinates=['psi','phi'],parameters=CONTRACT,
            inferential_support='A_train drift support intersected with all observed-plus-simulated drift supports; symmetric in all R+1 realizations',
            descriptive_support_frequencies=[.90,.95,.99],min_common_cells=1,
            field_distance='Symmetric leave-one-realization-out distance; each realization uses its own full-panel learner-normalised occupancy weights on the same common support',
            basin_statistics='Original negative-divergence occupancy and flow-weighted frozen-shell/core metrics; common support/stencil; no replicate deletion',
            calibration='Training-only empirical shrunken residual-score random intercept sensitivity, 256 equiprobable quantile support points; exactly same discrete distribution in calibration and simulation',
            limitations=['fixed recorded opportunities, not a counterfactual platform policy','plug-in training item baselines; baseline estimation uncertainty not integrated','Null B is a stable-heterogeneity sensitivity, not a fitted GLMM','post hoc analyses; not prespecified original experiments'])
        p['protocol_id']=digest(p)

        m=source_module(p);users,content,priors=frozen_inputs(m)
        p['split_sizes']={s:len(v) for s,v in users.items()};p['protocol_id']=digest(protocol_payload(p))
        snap=work/'source_snapshot';snap.mkdir(exist_ok=True)
        for k in ('stage1','semantic','original_null'):shutil.copy2(sources[k],snap/Path(sources[k]).name)
        atomic_json(work/'path_resolution.json',dict(data=data,sources=sources,stage1_root=str(out/'stage1'),raw_panel_root=str(out/'stage1/dynamics/raw_panels'),existing_null_validation=str(out/'stage1_construction_matched_null'),existing_null_confirmation=str(out/'stage1_construction_matched_null_confirm')))
        atomic_json(work/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,executable=sys.executable))
        atomic_json(work/'protocol.json',p)
        print(f'[init] protocol {p["protocol_id"]}: {work}',flush=True)
    return work


def fit_null_b(work):
    work=Path(work);p=load_protocol(work);check_inventory(p)
    with lock(work/'locks/calibration.lock'):
        ready=work/'calibration_complete.json'
        if ready.exists():
            validate_calibration(work,p)
            return
        m=source_module(p);users,content,priors=frozen_inputs(m);builder=m.UserEmpiricalDynamicsBuilder(content,priors)
        train=users['A_train']
        if p['smoke']:
            g=np.random.default_rng(p['seed']+99001);u=np.array(sorted(train));train=set(g.choice(u,min(max(p['smoke_users'],32),len(u)),replace=False))
        scores={};infos={};counts={}
        qprior={qid:builder._question_prior(qid,content.question_tag_idx.get(qid)) for qid in set(content.question_tag_idx)|set(priors.question_priors)}
        for name in p['data']['question'].values():
            q=m.read_path_table(Path(name),columns=Q_REQ);q=q[q.user_id.isin(train)&q.is_correct.notna()].copy()
            if q.empty:continue
            q=q[q.question_id.astype(str).isin([k for k,v in content.question_tag_idx.items() if len(v)>0])]
            if q.empty:continue
            prob=probabilities(q,content,builder);y=q.is_correct.to_numpy(float)
            if np.any(~np.isin(y,[0.,1.])):raise ValueError('Nonbinary training correctness')
            t=pd.DataFrame(dict(u=q.user_id.to_numpy('int64'),score=y-prob,info=prob*(1-prob),n=np.ones(len(q))))
            for uid,r in t.groupby('u').sum().iterrows():
                u=int(uid);scores[u]=scores.get(u,0.)+r.score;infos[u]=infos.get(u,0.)+r['info'];counts[u]=counts.get(u,0.)+r.n
        if len(counts)<(2 if p['smoke'] else 100):raise RuntimeError('Insufficient training users for the declared Null-B sensitivity.')
        effects=np.asarray([np.clip(scores[u]/max(infos[u],1e-12)*counts[u]/(counts[u]+20.),-2.5,2.5) for u in sorted(counts)])
        effects-=effects.mean()
        support=np.quantile(effects,(np.arange(256)+.5)/256);support-=support.mean()
        pg=np.unique(np.asarray(list(qprior.values())+[priors.global_prior],float));alpha=calibrated_intercepts(pg,support)
        maxerr=max(float(np.max(np.abs(expit(alpha[i:i+256,None]+support[None,:]).mean(axis=1)-pg[i:i+256]))) for i in range(0,len(pg),256))
        if maxerr>1e-11:raise RuntimeError(f'Null-B marginal calibration failed: {maxerr}')
        atomic_npz(work/'null_b_calibration.npz',p=pg,alpha=alpha,effects=support)
        atomic_json(ready,dict(protocol_id=p['protocol_id'],sha256=sha256(work/'null_b_calibration.npz'),fit_split='A_train',n_train_users=len(counts),effect_mean=float(support.mean()),effect_sd=float(support.std()),effect_points=len(support),max_item_marginal_error=maxerr,smoke=p['smoke'],description=p['calibration']))
        print(f'[calibration] Null B frozen; {len(counts)} training users; max marginal error={maxerr:.3g}',flush=True)


def raw_chunk(m,p,ci,selected):
    bpath=Path(p['data']['bundle'][str(ci)])
    b=m.read_path_table_with_optional(bpath,B_REQ,B_OPT);b=b[b.user_id.isin(selected)].copy()
    if b.empty:return b,{},{}
    u=set(b.user_id.astype('int64'))
    qp=p['data']['question'].get(str(ci));sp=p['data']['study'].get(str(ci))
    q=m.read_path_table(Path(qp),columns=Q_REQ) if qp else pd.DataFrame(columns=Q_REQ)
    s=m.read_path_table_with_optional(Path(sp),S_REQ,S_OPT) if sp else pd.DataFrame(columns=S_REQ+S_OPT)
    q=q[q.user_id.isin(u)];s=s[s.user_id.isin(u)]
    return b,{int(k):v.copy() for k,v in q.groupby('user_id',sort=False)},{int(k):v.copy() for k,v in s.groupby('user_id',sort=False)}


def prepare_split(work,split):
    work=Path(work);p=load_protocol(work);check_inventory(p)
    if split=='B_confirm':require_confirm(work,p)
    with lock(work/f'locks/prepare_{split}.lock'):
        done=work/'cache'/split/'complete.json'
        if done.exists():
            validate_prepared_split(work,p,split)
            return
        m=source_module(p);sem=semantic_module(p);users,content,priors=frozen_inputs(m);builder=m.UserEmpiricalDynamicsBuilder(content,priors)
        selected=users[split]
        if p['smoke']:
            if split!='A_val':raise RuntimeError('Smoke has no confirmation analysis.')
            rng=np.random.default_rng(p['seed']+6001);a=np.array(sorted(selected));selected=set(rng.choice(a,min(p['smoke_users'],len(a)),replace=False))
        accs=new_accumulators(m,sem);seen=set();blocks=[];identity=[];rawfiles=chunks(m.RAW_ROOT,'raw_panel')
        if not rawfiles:raise FileNotFoundError('Archived raw_panel_chunk_* are required for source-replay identity checks; no Stage-1 rerun.')
        started=time.time()
        for ci in map(int,p['data']['bundle']):
            b,qg,sg=raw_chunk(m,p,ci,selected)
            if b.empty:continue
            rebuilt=[];cache=[]
            for uid,bu in b.groupby('user_id',sort=False):
                uid=int(uid)
                if uid in seen:raise RuntimeError(f'User {uid} occurs in multiple raw chunks; cannot reset history.')
                seen.add(uid);qu=qg.get(uid,pd.DataFrame(columns=Q_REQ));su=sg.get(uid,pd.DataFrame(columns=S_REQ+S_OPT))
                rows=builder.build_user(uid,bu,qu,su,split)
                if not rows:continue
                rebuilt.extend(rows)
                pr=probabilities(qu,content,builder)
                y=pd.to_numeric(qu.is_correct,errors='raise').to_numpy(float)
                if np.any(np.isfinite(y)&~np.isin(y,[0.,1.])):raise ValueError('Nonbinary held-out correctness')
                qt=qu.copy();qt['is_correct']=np.where(np.isfinite(y),0.,np.nan)
                bt=bu.copy();bt['accuracy']=np.nan
                cache.append((uid,bt,qt,su,pr,len(rows)))
            if not rebuilt:continue
            if ci not in rawfiles:raise FileNotFoundError(f'Archived raw panel chunk missing: {ci}')
            saved=m.read_path_table(rawfiles[ci],columns=KEYS+IDENTITY_COLS)
            saved=saved[saved.user_id.isin({v[0] for v in cache})]
            proof=replay_identity(pd.DataFrame(rebuilt),saved);proof.update(chunk=ci,source_raw_panel=str(rawfiles[ci]),source_sha256=sha256(rawfiles[ci]))
            identity.append(proof)
            add_fields(m,sem,accs,rebuilt)


            for start in range(0,len(cache),128):
                part=cache[start:start+128];path=work/'cache'/split/f'chunk_{ci:04d}_block_{start//128:04d}.pkl.gz'
                atomic_pickle(path,part)
                blocks.append(dict(path=str(path.relative_to(work)),sha256=sha256(path),users=len(part),rows=sum(x[-1] for x in part)))
            print(f'[prepare {split}] source_chunk={ci}; users={accs["psi"].users}; rows={accs["psi"].rows}',flush=True)
        if not blocks:raise RuntimeError(f'No complete users for {split}')
        fields=pack_fields(accs);audit={}
        if not p['smoke']:
            for c in ('psi','phi'):
                audit[c]=audit_field(unpack_fields(fields,c),archived_field_base(p['output_root'],split,c))
        else:audit={'scope':'smoke subset; full publication-grid identity is deliberately not claimed'}
        op=work/'observed'/f'{split}.npz';atomic_npz(op,**fields)
        ip=work/'observed'/f'{split}_identity.json'
        atomic_json(ip,dict(protocol_id=p['protocol_id'],split=split,smoke=p['smoke'],source_replay=identity,publication_field=audit,elapsed_seconds=time.time()-started,passed=True))
        atomic_json(done,dict(protocol_id=p['protocol_id'],split=split,blocks=blocks,observed_file=str(op.relative_to(work)),observed_sha256=sha256(op),identity_sha256=sha256(ip),users=accs['psi'].users,rows=accs['psi'].rows,smoke=p['smoke']))


def load_calibration(work,p):
    validate_calibration(work,p);f=Path(work)/'null_b_calibration.npz'
    with np.load(f,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def simulate(work,split,mode,rid):
    work=Path(work);p=load_protocol(work)
    if rid<0 or rid>=p['replicates']:raise ValueError('Replicate index outside frozen range')
    if split=='B_confirm':require_confirm(work,p)
    root=work/'simulations'/split/mode;out=root/f'replicate_{rid:04d}.npz';meta=out.with_suffix('.json')
    with lock(work/f'locks/{split}_{mode}_{rid}.lock'):
        complete=read_json(work/'cache'/split/'complete.json')
        _metadata_expected=dict(protocol_id=p['protocol_id'],split=split,smoke=p['smoke'])
        if any(complete.get(k)!=v for k,v in _metadata_expected.items()):raise RuntimeError('Cache metadata mismatch')
        cache_manifest_sha256=sha256(work/'cache'/split/'complete.json')
        calibration_meta=validate_calibration(work,p) if mode=='null_b' else None
        if meta.exists():
            validate_replicate(work,p,split,mode,rid,complete,calibration_meta)
            print(f'[simulate] verified completed {split}/{mode}/{rid}',flush=True);return
        m=source_module(p);sem=semantic_module(p);_,content,priors=frozen_inputs(m);builder=m.UserEmpiricalDynamicsBuilder(content,priors)
        calibration=load_calibration(work,p) if mode=='null_b' else None
        accs=new_accumulators(m,sem);ca=np.zeros(4);started=time.time()

        for i,block in enumerate(complete['blocks']):
            f=work/block['path']
            if sha256(f)!=block['sha256']:raise RuntimeError(f'Corrupt cache {f}')
            users=read_pickle(f);rows=[]
            for uid,bu,qt,su,pr,n in users:
                qs,d=simulate_questions(qt,pr,p['seed'],split,mode,rid,uid,calibration);ca+=d
                rr=builder.build_user(uid,bu,qs,su,split)
                if len(rr)!=n:raise RuntimeError('Simulated submitted-bundle count changed')
                rows.extend(rr)
            add_fields(m,sem,accs,rows)
            if i%25==0:print(f'[{split}/{mode}/{rid}] blocks {i+1}/{len(complete["blocks"])}; rows={accs["psi"].rows}',flush=True)
        if accs['psi'].rows!=complete['rows'] or accs['psi'].users!=complete['users']:raise RuntimeError('Simulated cohort/interval coverage differs from audited original')
        fields=pack_fields(accs)
        if not all(np.all(np.isfinite(fields[k])) for k in fields if k.endswith(('_u','_v','_w','_occ'))):raise RuntimeError('Non-finite supported field sufficient statistics')

        z=(ca[1]-ca[2])/np.sqrt(ca[3]) if ca[3]>0 else None
        atomic_npz(out,**fields)
        atomic_json(meta,dict(protocol_id=p['protocol_id'],split=split,mode=mode,replicate=rid,sha256=sha256(out),seed=p['seed'],rows=accs['psi'].rows,users=accs['psi'].users,smoke=p['smoke'],cache_manifest_sha256=cache_manifest_sha256,null_b_calibration_sha256=calibration_meta['sha256'] if calibration_meta else None,calibration=dict(n=ca[0],correct=ca[1],sum_probability=ca[2],sum_conditional_variance=ca[3],standardized_residual=z),elapsed_seconds=time.time()-started))
        print(f'[done] {out}',flush=True)


def smoke_finish(work):
    work=Path(work);p=load_protocol(work)
    if not p['smoke']:raise RuntimeError('Not a smoke directory')
    marker=work/'SMOKE_OK.json'

    if marker.exists():marker.unlink()
    refs={}
    for mode in p['modes']:
        validate_realizations(work,p,'A_val',mode)
        for r in range(p['replicates']):
            for ext in ('npz','json'):
                f=work/'simulations/A_val'/mode/f'replicate_{r:04d}.{ext}';refs[str(f.relative_to(work))]=sha256(f)
    for rel in ('cache/A_val/complete.json','observed/A_val.npz','observed/A_val_identity.json','calibration_complete.json','null_b_calibration.npz'):
        refs[rel]=sha256(work/rel)
    atomic_json(marker,dict(protocol_id=p['protocol_id'],code_hashes=p['code_hashes'],source_hashes=p['source_hashes'],passed=True,schema_version=1,evidence=refs,scope='real-data subset replay and both simulation paths only; not formal inference'))


def main():
    a=argparse.ArgumentParser(description='Prepare frozen inputs and simulate joint response-generating construction controls.')
    a.add_argument('command',choices=['init','calibrate','prepare','simulate','smoke-finish'])
    a.add_argument('--output-root',default=DEFAULT_OUTPUT);a.add_argument('--work-root');a.add_argument('--repo-root');a.add_argument('--data-root')
    a.add_argument('--seed',type=int,default=42);a.add_argument('--replicates',type=int,default=999)
    a.add_argument('--smoke',action='store_true');a.add_argument('--smoke-users',type=int,default=200)
    a.add_argument('--split',choices=['A_val','B_confirm']);a.add_argument('--mode',choices=['null_a','null_b']);a.add_argument('--replicate',type=int)
    args=a.parse_args()
    work=choose_work(args.output_root,args.work_root,args.smoke)
    if args.command=='init':initialize(args)
    elif args.command=='calibrate':fit_null_b(work)
    elif args.command=='prepare':
        if not args.split:a.error('--split required')
        prepare_split(work,args.split)
    elif args.command=='simulate':
        rid=args.replicate if args.replicate is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID','-1'))
        if not args.split or not args.mode:a.error('--split and --mode required')
        simulate(work,args.split,args.mode,rid)
    else:smoke_finish(work)

if __name__=='__main__':main()
