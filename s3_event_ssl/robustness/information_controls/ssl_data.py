
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ssl_common import *

ALIGNMENT_SOURCES=(
 'response_alignment_to_pre_demand','response_aligned_mass_interval',
 'response_off_target_mass_interval','response_neutral_mass_interval',
 'support_alignment_to_pre_demand_or_current_bundle','support_aligned_mass_interval',
 'support_off_target_mass_interval','support_neutral_mass_interval')


def alignment_mask(manifest):
    prep=module('prepare_event_ssl_inputs.py')
    sources=manifest['numeric_input_source_columns']
    expected=[]
    for col in sources:
        require(col in prep.NUMERIC_PRIMITIVE_CANDIDATES,f'Unsupported source feature/alias {col}; audit lineage before use.')
        require(not prep.is_forbidden_feature(col),f'Forbidden macrostate feature {col}')
        expected.append(col)
        if any(t in col for t in prep.LOG1P_NUMERIC_TOKENS): expected.append('log1p_'+col)
    names=manifest['numeric_feature_names_after_expansion']
    require(names==expected,'Numeric feature expansion/order differs from fixed preparation source')
    missing=set(ALIGNMENT_SOURCES)-set(sources)
    require(not missing,'Alignment source fields missing; cannot silently redefine the planned clean control: '+str(sorted(missing)))
    mask=np.array([n.removeprefix('log1p_') in ALIGNMENT_SOURCES for n in names],bool)
    require(mask.any() and (~mask).any(),'Invalid clean feature mask')
    require(all(x in prep.CATEGORICAL_CANDIDATES for x in manifest['categorical_input_source_columns']),
            'Unsupported categorical alias; possible derived-information side channel')
    return mask,[{'index':i,'expanded_feature':n,'source':n.removeprefix('log1p_'),
                  'clean_encoder_input':'zero_after_original_normalization' if mask[i] else 'retained',
                  'clean_future_target':'excluded_from_loss_and_denominator' if mask[i] else 'retained'}
                 for i,n in enumerate(names)]

class Prepared:

    def __init__(self,root,split,max_users=None):
        self.root=Path(root);self.split=split
        self.manifest=load_json(self.root/'metadata/stage4_input_manifest.json')
        module('train_event_ssl.py').validate_input_contract(self.manifest)
        self.summary=self.manifest['split_summaries'][split]
        n=int(self.summary['rows']);self.n_num=int(self.summary['numeric_shape'][1]);self.n_cat=int(self.summary['categorical_shape'][1])
        self.original_n=n;d=self.root/split;self.files=[]
        def mm(name,dtype,shape):
            p=d/name;require(p.is_file(),f'Missing prepared file {p}')
            require(p.stat().st_size==int(np.prod(shape))*np.dtype(dtype).itemsize,f'Memmap byte count mismatch {p}')
            self.files.append(p)
            return np.memmap(p,mode='r',dtype=dtype,shape=shape) if np.prod(shape)>0 else np.empty(shape,dtype)
        self.num=mm('x_num.float32.mmap',np.float32,(n,self.n_num))
        self.cat=mm('x_cat.int64.mmap',np.int64,(n,self.n_cat))
        self.X=mm('y_current.float32.mmap',np.float32,(n,2))
        self.Y=mm('y_next.float32.mmap',np.float32,(n,2))
        self.uid=mm('user_id.int64.mmap',np.int64,(n,))
        self.step=mm('bundle_step_index.int64.mmap',np.int64,(n,))
        self.offsets=np.load(d/'sequence_offsets.npy',allow_pickle=False);self.files.append(d/'sequence_offsets.npy')
        require(n>0 and len(self.offsets)>1 and self.offsets[0]==0 and self.offsets[-1]==n and np.all(np.diff(self.offsets)>0),'Invalid sequence offsets')
        require(np.issubdtype(self.offsets.dtype,np.integer),'Non-integer sequence offsets')
        self.indices=None
        if max_users:
            users=np.unique(self.uid)
            if len(users)>max_users:
                rng=np.random.default_rng(42+SPLITS.index(split));users=np.sort(rng.choice(users,max_users,replace=False))
            self.indices=np.flatnonzero(np.isin(self.uid,users))
            require(len(self.indices)>0,'No smoke rows')
            u=np.asarray(self.uid[self.indices]);s=np.asarray(self.step[self.indices])
            self.offsets=module('prepare_event_ssl_inputs.py').sequence_offsets(u,s)
        self.n=n if self.indices is None else len(self.indices)
    def values(self,name,index=slice(None)):
        x=getattr(self,name)
        return np.asarray(x[index] if self.indices is None else x[self.indices[index]])
    def sequence(self,i): return int(self.offsets[i]),int(self.offsets[i+1])
    def keys(self): return np.column_stack([self.values('uid'),self.values('step')])
    def as_original_arrays(self):
        return {'x_num':self.values('num'),'x_cat':self.values('cat'),'y':self.values('X'),
                'y_next':self.values('Y'),'user_id':self.values('uid'),'step':self.values('step'),
                'offsets':self.offsets,'n':self.n}
    def validate(self):
        u=self.values('uid');s=self.values('step')
        require(np.all(u[1:]>=u[:-1]),'Prepared users not sorted')
        same=u[1:]==u[:-1]
        require(np.all(s[1:][same]>s[:-1][same]),'Duplicate or unordered bundle keys')
        expected=module('prepare_event_ssl_inputs.py').sequence_offsets(u,s)
        require(np.array_equal(expected,self.offsets),'Offset boundaries do not match user/step continuity')
        for i in range(0,self.n,200000):
            sl=slice(i,min(i+200000,self.n))
            for name in ['num','X','Y']:
                a=self.values(name,sl);require(np.isfinite(a).all(),f'Nonfinite prepared {name}: {self.split}')
            for name in ['X','Y']:
                require((np.abs(self.values(name,sl))<=1.000001).all(),f'Target coordinate outside domain: {name}')
            a=self.values('cat',sl)
            require(((a>=0)&(a<int(self.manifest['categorical_hash_buckets']))).all(),'Categorical bucket out of range')

        mx=0.
        for i in range(0,self.n-1,200000):
            e=min(i+200000,self.n-1);ok=(u[i+1:e+1]==u[i:e])&(s[i+1:e+1]==s[i:e]+1)
            if ok.any(): mx=max(mx,float(np.max(np.abs(self.values('Y',slice(i,e))[ok]-self.values('X',slice(i+1,e+1))[ok]))))
        require(mx<2e-6,f'Next-state/current-state continuity mismatch {mx:g}')
        return {'rows':self.n,'users':len(np.unique(u)),'sequences':len(self.offsets)-1,
                'max_target_continuity_error':mx,'whole_users_smoke':self.indices is not None}


def resolve_paths(history):
    h=Path(history).resolve();require(h.is_dir(),f'History root missing: {h}')
    result={}
    for seed in ALL_SEEDS:
        base=h if seed==42 else h/f'random_seed_experiments/seed_{seed}'
        inp=base/'stage4_event_ssl/prepared_inputs'
        full=base/'stage4_event_ssl/models/predictive_state/best_model.pt'
        shuffle=base/'stage4_event_ssl_time_shuffle_control/model/best_model.pt'
        require((inp/'metadata/stage4_input_manifest.json').exists(),f'Missing original input manifest: {inp}')
        for p in (full,shuffle):require(p.is_file(),f'Missing frozen model: {p}')
        result[str(seed)]={'input_root':str(inp),'full':str(full),'shuffled':str(shuffle),
           'full_evaluation':str(base/'stage4_event_ssl/evaluation_predictive_state'),
           'shuffled_evaluation':str(base/'stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs'),
           'pure':str(base/'stage4_event_ssl/models/pure_ssl/best_model.pt')}
    return result


def load_partition(history):
    h=Path(history);p=h/'stage1/dynamics/fixed_k6_mesostates'
    mp=p/'fixed_k6_model_metadata.json';m=load_json(mp)
    require(m.get('macrostate_k')==6 and m.get('fit_split')=='A_train' and m.get('macrostate_k_rule')=='fixed a priori','K6 frozen contract missing')

    export=h/'stage4_event_ssl/evaluation_predictive_state/tables/stage4_event_ssl_fixed_k6_macrostate_centers.csv'
    cp=export if export.is_file() else table_path(p/'fixed_k6_centers')
    c=read_table(cp).sort_values('macrostate')
    require(np.array_equal(c['macrostate'],np.arange(6)),'K6 label IDs changed')
    mean=np.asarray(m['scaler_mean'],float);scale=np.asarray(m['scaler_scale'],float)
    std=c[['center_M_standardized','center_Psi_standardized']].to_numpy(float)
    raw=c[['center_M','center_Psi']].to_numpy(float)
    require(mean.shape==(2,) and scale.shape==(2,) and np.all(scale>0) and np.isfinite(std).all(),'Invalid frozen partition')

    require(np.allclose((raw-mean)/scale,std,atol=2e-6,rtol=2e-6),'Frozen center standardization inconsistent')
    return {'mean':mean,'scale':scale,'standardized_centers':std,'centers':raw,
            'source_files':[str(mp),str(cp)],'historical_export_preferred':cp==export}


def labels(x,part):
    x=np.asarray(x,float);require(x.ndim==2 and x.shape[1]==2 and np.isfinite(x).all(),'Invalid states for K6')
    return np.argmin(((((x-part['mean'])/part['scale'])[:,None,:]-part['standardized_centers'][None,:,:])**2).sum(axis=2),axis=1)


def key_positions(keys,query):
    keys=np.asarray(keys,dtype=np.int64);query=np.asarray(query,dtype=np.int64)
    dt=np.dtype([('uid','<i8'),('step','<i8')])
    a=np.ascontiguousarray(keys).view(dt).ravel();b=np.ascontiguousarray(query).view(dt).ravel()
    ii=np.searchsorted(a,b);ok=ii<len(a);safe=np.minimum(ii,len(a)-1)
    ok &= a[safe]==b
    return ii,ok


def archive_bridge(data,evalroot,predictions,dest,checkpoint):

    root=Path(evalroot);mp=root/'metadata/stage4_event_ssl_evaluation_manifest.json'
    require(mp.is_file(),f'Historical evaluation manifest missing {mp}')
    m=load_json(mp)
    require(m.get('checkpoint_sha256')==sha(checkpoint),'Historical evaluation checkpoint hash mismatch')
    reviewed=load_json(ROOT/'source_contract.json')['files']['train_event_ssl.py']
    require(m.get('train_script_sha256')==reviewed,'Historical network implementation differs from fixed source; cannot silently import a guessed model')
    p=table_path(root/'predictions'/f'stage4_event_ssl_predictions_{data.split}')
    columns=['user_id','bundle_step_index','M','Psi','target_M_next','target_Psi_next','pred_M','pred_Psi','pred_next_M','pred_next_Psi']
    keys=data.keys();seen=np.zeros(data.n,bool);mx=np.zeros(4);se=np.zeros(4);count=0
    legacy=np.lib.format.open_memmap(Path(dest)/'historical_S_P.npy',mode='w+',dtype=np.float32,shape=(data.n,4))
    for df in iter_table(p,columns):
        idx,ok=key_positions(keys,df[['user_id','bundle_step_index']].to_numpy(np.int64))
        if data.indices is None: require(ok.all(),'Historical prediction has unexpected keys')
        else:
            selected=np.isin(df.user_id,np.unique(data.values('uid')))
            require(ok[selected].all(),'Historical prediction has unexpected keys for smoke users')
        df=df.loc[ok];idx=idx[ok]
        if not len(idx):continue
        require(len(np.unique(idx))==len(idx) and not seen[idx].any(),'Duplicate historical prediction keys')
        seen[idx]=True
        assert_close(data.values('X',idx),df[['M','Psi']].to_numpy(),'Historical origin identity',atol=2e-7,rtol=0)
        assert_close(data.values('Y',idx),df[['target_M_next','target_Psi_next']].to_numpy(),'Historical target identity',atol=2e-7,rtol=0)
        v=df[['pred_M','pred_Psi','pred_next_M','pred_next_Psi']].to_numpy(np.float32)
        require(np.isfinite(v).all(),'Nonfinite historical predictions')
        legacy[idx]=v
        now=np.concatenate([np.asarray(predictions[idx,:2]),np.asarray(predictions[idx,4:6])],axis=1)
        diff=now.astype(float)-v
        mx=np.maximum(mx,np.max(np.abs(diff),axis=0));se+=(diff**2).sum(axis=0);count+=len(df)
    require(seen.all(),'Historical prediction missing rows')
    legacy.flush()
    report={'checkpoint_sha256':sha(checkpoint),'archive':file_record(p),'evaluation_manifest':file_record(mp),
      'rows':count,'all_keys_targets_verified':True,'columns':['S_M','S_Psi','P_M','P_Psi'],
      'max_absolute_cpu_minus_archive':mx,'rmse_cpu_minus_archive':np.sqrt(se/max(count,1)),
      'replay_precision':'CPU float32; historical GPU autocast/precision may differ',
      'status':'numeric_identity_within_2e-5' if mx.max()<=2e-5 else 'distinct_CPU_precision_view',
      'policy':'Archived metrics remain immutable; CPU recomputation is separately named. No outcome-based recentering or tolerance expansion.'}
    save_json(report,Path(dest)/'S0_archive_bridge.json')
    return report


def existing_results(history):

    h=Path(history);names={'final_models.json','FIT_FROZEN.json','PROTOCOL.json',
      'reviewer_evidence_table.csv','construction_null_statistics.csv','construction_null_report.md','three_way_point_metrics_by_seed.csv',
      'fixed_support_improvement_addendum.json','Table11_same_method_cross_comparison.csv',
      'C3_referenced_cross_model_by_seed.csv','coordinate_recovery_certificate.json'}
    rows=[]
    for p in sorted(h.rglob('*')):
        if p.is_file() and p.name in names and not any(a.startswith(('ssl_core_revision',RUN_NAME)) for a in p.relative_to(h).parts):
            if p.stat().st_size<10_000_000: rows.append(file_record(p))
    return rows


def discover_reference(history):

    h=Path(history);found=[]
    for p in h.rglob('final_models.json'):
        parts=p.relative_to(h).parts
        if any(a.startswith(('ssl_core_revision',RUN_NAME)) or 'smoke' in a.lower() for a in parts):continue
        if p.parent.name!='models':continue
        run=p.parent.parent
        row=table_path(run/'confirmation/row_predictions',required=False)
        freeze=p.parent/'FIT_FROZEN.json';complete=run/'final/COMPLETE.json'
        if row and freeze.exists() and complete.exists():
            finished=load_json(complete)
            if finished.get('smoke',False) or finished.get('smoke_only',False):continue
            f=load_json(freeze)
            require(f.get('model_payload_sha256')==sha(p),'Existing C3 frozen payload hash mismatch')
            found.append((run,row,p,freeze))
    if not found:return None
    hashes={sha(q[2]) for q in found}
    require(len(hashes)==1,'Multiple distinct frozen C3 model packages found. Place different model versions in separate directories outside the history root and pass one completed run with --fourterm-root.')
    run,row,p,freeze=found[0]
    return {'run':str(run),'rows':str(row),'models':file_record(p),'freeze':file_record(freeze),
            'complete':file_record(run/'final/COMPLETE.json'),'read_only':True,'fit_performed':False}


def load_reference_rows(ref,data,dest):

    Path(dest).mkdir(parents=True,exist_ok=True)
    if ref is None:
        save_json({'status':'not_available','no_refit_or_imputation':True},Path(dest)/'C3_bridge_status.json');return None
    cols=['user_id','bundle_step_index','M','Psi','target_M_next','target_Psi_next',
          'four_term_next_M','four_term_next_Psi','C3_next_M','C3_next_Psi']
    keys=data.keys();seen=np.zeros(data.n,bool)
    v=np.lib.format.open_memmap(Path(dest)/'frozen_reference.npy',mode='w+',dtype=np.float64,shape=(data.n,8));v[:]=np.nan
    for df in iter_table(ref['rows'],cols):
        idx,ok=key_positions(keys,df[['user_id','bundle_step_index']].to_numpy(np.int64));df=df.loc[ok];idx=idx[ok]
        require(len(np.unique(idx))==len(idx) and not seen[idx].any(),'Duplicate frozen C3 export keys')
        seen[idx]=True
        if len(idx):
            arr=df[cols[2:]].to_numpy(float);require(np.isfinite(arr).all(),'Nonfinite frozen reference')
            assert_close(arr[:,:2],data.values('X',idx),'C3 bridge coordinate agreement',atol=2e-7,rtol=0)
            assert_close(arr[:,2:4],data.values('Y',idx),'C3 bridge target agreement',atol=2e-7,rtol=0)
            v[idx]=arr
    v.flush();np.save(Path(dest)/'reference_rows.npy',np.flatnonzero(seen))
    save_json({'status':'complete' if seen.all() else 'intersection_only','matched_rows':int(seen.sum()),
               'SSL_rows':data.n,'fraction':float(seen.mean()),'source':file_record(ref['rows']),
               'origin_contract':'exact historical mechanism export on key intersection; not a replacement for primary SSL memmap anchors',
               'models_refitted':False},Path(dest)/'C3_bridge_status.json')
    return v


def archived_metric_identity(data,evalroot,legacy,part,dest):

    er=Path(evalroot);ev=module('evaluate_event_ssl_structure.py')
    mp=er/'metadata/stage4_event_ssl_evaluation_manifest.json';m=load_json(mp)
    cm=m.get('convergence_reference',{})
    conv=(float(cm.get('reference_M',0)),float(cm.get('reference_Psi',0)))
    partition=ev.MacroPartition(part['mean'],part['scale'],part['centers'],part['standardized_centers'],6,{})
    arrays={'y':data.values('X'),'y_next':data.values('Y'),'user_id':data.values('uid')}
    metrics,_=ev.metrics_for_predictions(arrays,np.asarray(legacy[:,:2]),np.asarray(legacy[:,2:4]),partition,conv)
    write_csv([metrics],Path(dest)/'historical_original_evaluator_metrics.csv')
    target=er/'tables'/f'stage4_event_ssl_structural_metrics_{data.split}.csv'
    require(target.exists(),f'Original metric table missing: {target}')
    archived=pd.read_csv(target).iloc[0]
    fields=['coordinate_rmse_M','coordinate_rmse_Psi','coordinate_corr_M','coordinate_corr_Psi',
      'one_step_rmse_M','one_step_rmse_Psi','one_step_corr_M','one_step_corr_Psi',
      'current_state_occupancy_js','next_state_occupancy_js','anchor_drift_vector_corr',
      'learned_plane_drift_vector_corr','anchor_transition_mean_row_tv','learned_plane_transition_mean_row_tv']
    audit=[]
    for name in fields:
        if name not in archived and name not in metrics:
            audit.append({'metric':name,'archived':np.nan,'original_evaluator_recomputed':np.nan,
                          'difference':np.nan,'status':'both_undefined','tolerance':'source evaluator omitted unsupported metric'})
            continue
        require(name in archived,f'Original metric column missing: {name}')
        a=float(archived[name]);b=float(metrics.get(name,np.nan))
        if data.indices is not None:
            status='smoke_subset_not_compared_to_full_population'
        elif not np.isfinite(a) and not np.isfinite(b):status='both_undefined'
        else:
            require(np.isfinite(a) and np.isfinite(b) and np.isclose(a,b,atol=2e-6,rtol=2e-6),
              f'Archived metric identity mismatch {name}: archived={a}, recomputed={b}. Investigate exact frozen coordinates/partition; do not snap all zeros.')
            status='passed'
        audit.append({'metric':name,'archived':a,'original_evaluator_recomputed':b,
          'difference':b-a,'status':status,'tolerance':'2e-6 absolute/relative for archived float32 outputs'})
    write_csv(audit,Path(dest)/'S0_historical_metric_identity.csv')
    return audit
