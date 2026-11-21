#!/usr/bin/env python3

from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from cn_common import *

SUMMARY_SCHEMA_VERSION = 1
METRICS = ('full_field_distance', 'negative_divergence_occupancy_fraction',
           'flow_weighted_shell_fraction_inward', 'flow_core_to_shell_speed_ratio')


def analysis_role(mode, coord, metric):
    if coord == 'phi':return 'secondary Phi comparator; no between-coordinate test'
    if mode == 'null_b':return 'Null B stable-heterogeneity sensitivity'
    return 'primary Null A/Psi field test' if metric == 'full_field_distance' else 'Null A/Psi three-basin-metric family'


def scope_coverage(occ,w,observed_state,observed_drift,common_state,common_drift,core,radius):

    def domains(state,drift):
        stencil=np.zeros_like(drift,dtype=bool);m=state&drift
        stencil[1:-1,1:-1]=m[1:-1,1:-1]&m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:]
        sc=np.zeros_like(drift,dtype=bool);shell=sc.copy()
        if core is not None and np.any(core):
            edges=np.linspace(-1.,1.,41);centres=(edges[:-1]+edges[1:])/2
            spacing=(float(np.mean(np.diff(centres))),)*2
            dist=distance_transform_edt(~core,sampling=spacing)
            sc=core&drift;shell=drift&~core&(dist>0)&(dist<=radius)
        return dict(field=drift,interior_stencil=stencil,core=sc,shell=shell)
    original=domains(observed_state,observed_drift);common=domains(common_state,common_drift)
    flow=np.sqrt(np.maximum(w,0.));rows=[]
    for name in original:
        a=original[name];b=common[name]
        if np.any(b&~a):raise RuntimeError('Common reporting domain is not a subset of observed domain')
        weight=occ if name in ('field','interior_stencil') else flow
        den=float(weight[a].sum());num=float(weight[b].sum())
        rows.append(dict(domain=name,observed_domain_cells=int(a.sum()),common_domain_cells=int(b.sum()),
                         weight_kind='observed occupancy' if name in ('field','interior_stencil') else 'sqrt(observed drift weight)',
                         observed_domain_weight=den,common_domain_weight=num,retained_weight_fraction=num/den if den>0 else None,
                         scope='common-supported subset of the original observed domain; descriptive coverage'))
    return rows


def artifact_manifest(out,p,split=None,mode=None):
    out=Path(out)
    paths=sorted(f for f in out.iterdir() if f.is_file() and f.name not in {'summary_artifacts.json','final_artifacts.json','COMPLETE.json'} and '.tmp.' not in f.name)
    return dict(schema_version=SUMMARY_SCHEMA_VERSION,protocol_id=p['protocol_id'],split=split,mode=mode,
                replicates=p['replicates'],seed=p['seed'],artifacts={f.name:sha256(f) for f in paths})


def validate_summary_evidence(work,p,split,mode,validate_inputs=True):

    work=Path(work);out=work/'summary'/split/mode
    if p.get('smoke') or p.get('replicates')!=999:raise RuntimeError('Final evidence requires the formal 999-replicate protocol')
    if split not in ('A_val','B_confirm') or mode not in ('null_a','null_b'):raise RuntimeError('Invalid summary identity')
    summary=read_json(out/'summary.json');manifest=read_json(out/'summary_artifacts.json')
    expected=dict(schema_version=SUMMARY_SCHEMA_VERSION,protocol_id=p['protocol_id'],split=split,mode=mode,replicates=p['replicates'],seed=p['seed'])
    for key,value in expected.items():
        if summary.get(key)!=value or manifest.get(key)!=value:raise RuntimeError(f'Summary/manifest identity mismatch: {split}/{mode}/{key}')
    if summary.get('technical_complete') is not True or summary.get('artifacts_manifest')!='summary_artifacts.json':raise RuntimeError('Summary is not technically complete')
    if set(summary.get('coordinates',{}))!={'psi','phi'} or set(p.get('coordinates',[]))!={'psi','phi'}:raise RuntimeError('Summary coordinate schema differs')
    artifacts=manifest.get('artifacts',{})
    required={'summary.json','numeric_report.md','response_calibration.csv'}
    for coord in ('psi','phi'):
        required.update(f'{coord}_{suffix}' for suffix in ('field_comparison.npz','tests.csv','scope_coverage.csv','realization_statistics.csv','field_grid.csv','support_sensitivity.csv'))
    if not isinstance(artifacts,dict) or not required.issubset(artifacts):raise RuntimeError('Incomplete summary artifact manifest')
    for relative,expected_hash in artifacts.items():
        file=(out/relative).resolve()
        if file.parent!=out.resolve() or not file.is_file() or sha256(file)!=expected_hash:raise RuntimeError(f'Summary artifact missing/changed: {relative}')
    for coord,result in summary['coordinates'].items():
        if result.get('coordinate')!=coord:raise RuntimeError('Coordinate result identity mismatch')
        rows=result.get('tests',[])
        if len(rows)!=len(METRICS) or {row.get('metric') for row in rows}!=set(METRICS):raise RuntimeError('Summary metric schema differs')
        if {r.get('domain') for r in result.get('scope_coverage',[])}!={'field','interior_stencil','core','shell'}:raise RuntimeError('Missing metric-specific coverage')
        for row in rows:
            pv=row.get('p_mc');status=row.get('status')
            if status not in ('estimable','not_estimable_no_replica_deleted'):raise RuntimeError('Unknown scientific estimability status')
            if status=='estimable' and (pv is None or not np.isfinite(pv) or not 0<pv<=1):raise RuntimeError('Invalid estimable p value')
            if status!='estimable' and pv is not None:raise RuntimeError('Non-estimable result has a p value')
            for name in ('observed','null_mean','null_q025','null_q975','observed_minus_null_mean','p_mc','q_bh_three_basin_metrics'):
                value=row.get(name)
                if value is not None and (not isinstance(value,(int,float)) or not np.isfinite(value)):raise RuntimeError(f'Invalid metric value: {name}')
            if row.get('scope')!='common-supported domain only; not the full original observed domain':raise RuntimeError('Missing inference scope')
        with np.load(out/f'{coord}_field_comparison.npz',allow_pickle=False) as z:
            for key in ('observed_u','observed_v','null_mean_u','null_mean_v','excess_u','excess_v','common_support','interior_common','observed_occupancy','simulation_support_frequency'):
                if key not in z or z[key].shape!=(40,40):raise RuntimeError('Comparison artifact schema differs')
            mask=z['common_support']
            if mask.dtype!=np.bool_ or int(mask.sum())!=result.get('common_cells'):raise RuntimeError('Comparison support differs from summary')
            for key in ('observed_u','observed_v','null_mean_u','null_mean_v','excess_u','excess_v','observed_occupancy'):
                if not np.isfinite(z[key][mask]).all():raise RuntimeError('Nonfinite comparison evidence on common support')
    if validate_inputs:read_realizations(work,p,split,mode)
    return summary,manifest


def invalidate_final(work):
    for name in ('COMPLETE.json','final_artifacts.json'):
        path=Path(work)/'final'/name
        if path.exists():path.unlink()


def conservative_rank_p(values,direction='greater',index=0):
    v=np.asarray(values,float)
    if not np.isfinite(v).all() or len(v)<3:return None
    tol=100*np.finfo(float).eps*max(abs(float(v[index])),1.)

    return float(np.mean(v>=v[index]-tol)) if direction=='greater' else float(np.mean(v<=v[index]+tol))


def bh_three(pvalues):
    p=np.asarray([1. if v is None else v for v in pvalues],float)
    order=np.argsort(p);q=np.empty(3);q[order]=np.minimum(1.,np.minimum.accumulate((p[order]*3/np.arange(1,4))[::-1])[::-1])
    return [None if v is None else float(q[i]) for i,v in enumerate(pvalues)]


def symmetric_distances(u,v,occ,mask):

    n=len(u)
    if n<3 or not np.any(mask):return np.full(n,np.nan)
    uu=u[:,mask];vv=v[:,mask];w=occ[:,mask]
    if not np.isfinite(uu).all() or not np.isfinite(vv).all() or np.any(w.sum(axis=1)<=0):return np.full(n,np.nan)
    others_u=(uu.sum(axis=0,keepdims=True)-uu)/(n-1)
    others_v=(vv.sum(axis=0,keepdims=True)-vv)/(n-1)
    w=w/w.sum(axis=1,keepdims=True)
    return np.sqrt(np.sum(w*((uu-others_u)**2+(vv-others_v)**2),axis=1))


def vector_diagnostics(a,b,w):
    ok=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)&np.isfinite(w)&(w>0);a=a[ok];b=b[ok];w=w[ok]
    if len(a)<2:return dict(cells=len(a),vector_r=None,local_cosine=None)
    r=float(np.corrcoef(a.ravel(),b.ravel())[0,1]) if a.std()>0 and b.std()>0 else None
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);valid=den>1e-12
    c=float(np.sum(w[valid]*np.sum(a[valid]*b[valid],axis=1)/den[valid])/w[valid].sum()) if valid.any() else None
    return dict(cells=len(a),vector_r=r,local_cosine=c)


def metric_bank(u,v,occ,w,state_common,drift_common,core,radius):

    n=len(u);shape=u.shape[1:];edges=np.linspace(-1.,1.,41);xc=(edges[:-1]+edges[1:])/2;yc=xc.copy()
    if shape!=(40,40):raise ValueError('Formal grid must be 40x40')
    stencil=np.zeros(shape,bool)
    m=state_common&drift_common
    stencil[1:-1,1:-1]=m[1:-1,1:-1]&m[:-2,1:-1]&m[2:,1:-1]&m[1:-1,:-2]&m[1:-1,2:]
    div=np.full_like(u,np.nan,dtype=float)
    dv=(u[:,2:,1:-1]-u[:,:-2,1:-1])/(xc[2:]-xc[:-2])[None,:,None]+(v[:,1:-1,2:]-v[:,1:-1,:-2])/(yc[2:]-yc[:-2])[None,None,:]
    div[:,1:-1,1:-1]=np.where(stencil[None,1:-1,1:-1],dv,np.nan)
    def wmean(values,weights,mask):
        den=weights[:,mask].sum(axis=1);num=(values[:,mask]*weights[:,mask]).sum(axis=1)
        return np.divide(num,den,out=np.full(n,np.nan),where=den>0)
    neg=wmean((div<0).astype(float),occ,stencil)
    speed=np.hypot(u,v);fw=np.sqrt(np.maximum(w,0.))
    if core is None or not np.any(core):
        return dict(negative_divergence_occupancy_fraction=neg,flow_weighted_shell_fraction_inward=np.full(n,np.nan),flow_core_to_shell_speed_ratio=np.full(n,np.nan)),dict(interior_cells=int(stencil.sum()),core_cells=0,shell_cells=0),stencil
    spacing=(float(np.mean(np.diff(xc))),float(np.mean(np.diff(yc))))
    dist,near=distance_transform_edt(~core,sampling=spacing,return_indices=True)
    X,Y=np.meshgrid(xc,yc,indexing='ij');tx=X[near[0],near[1]]-X;ty=Y[near[0],near[1]]-Y
    shell=drift_common&~core&(dist>0)&(dist<=radius);sc=core&drift_common
    positive=((u*tx+v*ty)>0)&(speed>1e-12)
    inward=wmean(positive.astype(float),fw,shell)
    cs=wmean(speed,fw,sc);ss=wmean(speed,fw,shell)
    ratio=np.divide(cs,ss,out=np.full(n,np.nan),where=ss>0)
    return dict(negative_divergence_occupancy_fraction=neg,flow_weighted_shell_fraction_inward=inward,flow_core_to_shell_speed_ratio=ratio),dict(interior_cells=int(stencil.sum()),core_cells=int(sc.sum()),original_core_cells=int(core.sum()),shell_cells=int(shell.sum())),stencil


def dist_record(values,name,direction='greater'):
    v=np.asarray(values,float);p=conservative_rank_p(v,direction)
    finite=np.isfinite(v[1:]);good=v[1:][finite]
    return dict(metric=name,direction=direction,observed=float(v[0]) if np.isfinite(v[0]) else None,null_mean=float(good.mean()) if len(good) else None,null_q025=float(np.quantile(good,.025)) if len(good) else None,null_q975=float(np.quantile(good,.975)) if len(good) else None,finite_null_replicates=int(finite.sum()),p_mc=p,status='estimable' if p is not None else 'not_estimable_no_replica_deleted')


def read_realizations(work,p,split,mode):
    return validate_realizations(Path(work),p,split,mode)


def frozen_geometry(p,coord):
    out=Path(p['output_root'])
    root=out/'stage1' if coord=='psi' else out/'semantic_specificity_control/coordinate/stage1'
    name='MR_PsiA' if coord=='psi' else 'MR_PhiAI'
    folder=root/'dynamics/candidate_regions'/name
    cf=folder/'A_train_primary_convergence_core_mask.npy';tf=folder/'training_convergence_thresholds.json'
    if not cf.exists() or not tf.exists():
        if coord=='psi':raise FileNotFoundError('Primary frozen geometry missing')
        return None,.35
    core=np.load(cf,allow_pickle=False).astype(bool);th=read_json(tf)
    if core.shape!=(40,40):raise RuntimeError('Frozen core shape mismatch')
    return core,float(th['shell_radius'])


def write_figures(out,coord,result,bank,arrays,common,stencil,core):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fields=[unpack_fields(x,coord) for x in arrays]
    u=np.stack([x['u'] for x in fields]);v=np.stack([x['v'] for x in fields])
    support=np.stack([x['drift_mask'] for x in fields]);freq=support[1:].mean(axis=0)
    X,Y=np.meshgrid(np.linspace(-.975,.975,40),np.linspace(-.975,.975,40),indexing='ij')
    ylab='Exposure alignment Ψ' if coord=='psi' else 'Activity–idle balance Φ'
    def save(fig,name):
        for ext in ('pdf','svg','png'):fig.savefig(out/f'{coord}_{name}.{ext}',dpi=350,bbox_inches='tight')
        plt.close(fig)

    for name,uu,vv in [('observed',u[0],v[0]),('null_mean',u[1:].mean(axis=0),v[1:].mean(axis=0)),('excess',u[0]-u[1:].mean(axis=0),v[0]-v[1:].mean(axis=0))]:
        fig,ax=plt.subplots(figsize=(5.2,4.6));mask=common
        if mask.any():
            qu=ax.quiver(X[mask],Y[mask],uu[mask],vv[mask],angles='xy',scale_units='xy',scale=1.,width=.003)
            ax.quiverkey(qu,.69,1.035,.1,'0.1 / step',labelpos='E',coordinates='axes',fontproperties={'size':8})
        else:ax.text(.5,.5,'No common supported cells',ha='center',transform=ax.transAxes)
        if core is not None and core.any() and not core.all():ax.contour(X,Y,core.astype(int),levels=[.5],linewidths=.8)
        ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='Response evidence M',ylabel=ylab);ax.set_title(name.replace('_',' ').title()+' (common support)',loc='left',pad=13);ax.set_xticks(np.linspace(-1,1,5));ax.set_yticks(np.linspace(-1,1,5));ax.set_aspect('equal');save(fig,name+'_field')
    fig,ax=plt.subplots(figsize=(5.2,4.6));im=ax.pcolormesh(np.linspace(-1,1,41),np.linspace(-1,1,41),freq.T,vmin=0,vmax=1,shading='flat')
    fig.colorbar(im,ax=ax,label='Fraction of simulated fields with ≥30 transitions');ax.set(xlabel='Response evidence M',ylabel=ylab,title='Simulation support, not statistical significance');ax.set_xticks(np.linspace(-1,1,5));ax.set_yticks(np.linspace(-1,1,5));save(fig,'support')
    labels={'full_field_distance':('Common-support field distance (state units / step)',1.),'negative_divergence_occupancy_fraction':('Common-stencil negative-divergence occupancy (%)',100.),'flow_weighted_shell_fraction_inward':('Supported frozen-shell inward fraction (%)',100.),'flow_core_to_shell_speed_ratio':('Supported core / shell drift-speed ratio',1.)}
    for row in result['tests']:
        label,factor=labels[row['metric']]
        vals=np.asarray(bank[row['metric']],float)*factor;good=vals[1:][np.isfinite(vals[1:])]
        fig,ax=plt.subplots(figsize=(5.8,3.7))
        if len(good):ax.hist(good,bins=min(40,max(3,int(np.sqrt(len(good))))),label='Simulated realizations')
        if np.isfinite(vals[0]):ax.axvline(vals[0],linestyle='--',linewidth=1.5,label='Observed')
        ax.set(xlabel=label,ylabel='Simulation count',title='One-sided comparison; '+('p='+format(row['p_mc'],'.4g') if row['p_mc'] is not None else 'not estimable'))
        if len(good) or np.isfinite(vals[0]):ax.legend(frameon=False)
        save(fig,row['metric'])


def analyze_coordinate(work,p,split,mode,coord,arrays,out):
    fields=[unpack_fields(x,coord) for x in arrays]
    u=np.stack([x['u'] for x in fields]);v=np.stack([x['v'] for x in fields]);occ=np.stack([x['occ'] for x in fields]);w=np.stack([x['w'] for x in fields])
    dm=np.stack([x['drift_mask'].astype(bool) for x in fields]);sm=np.stack([x['state_mask'].astype(bool) for x in fields])
    train=read_table(archived_field_base(p['output_root'],'A_train',coord)).sort_values(['x_bin','y_bin'])
    tm=train.drift_supported.to_numpy(bool).reshape(40,40);ts=train.state_supported.to_numpy(bool).reshape(40,40)

    common=tm&np.all(dm,axis=0);state_common=ts&np.all(sm,axis=0)
    core,radius=frozen_geometry(p,coord)
    mincells=1 if p['smoke'] else p['min_common_cells'];estimable=int(common.sum())>=mincells
    distance=symmetric_distances(u,v,occ,common) if estimable else np.full(len(arrays),np.nan)
    metrics,geometry,stencil=metric_bank(u,v,occ,w,state_common,common,core,radius)
    if not estimable:metrics={k:np.full(len(arrays),np.nan) for k in metrics}
    bank={'full_field_distance':distance,**metrics}
    rows=[dist_record(distance,'full_field_distance')]
    basin=[dist_record(metrics[k],k,d) for k,d in [('negative_divergence_occupancy_fraction','greater'),('flow_weighted_shell_fraction_inward','greater'),('flow_core_to_shell_speed_ratio','less')]]
    for row,q in zip(basin,bh_three([r['p_mc'] for r in basin])):row['q_bh_three_basin_metrics']=q
    rows.extend(basin)


    original_metrics,_,_=metric_bank(u[:1],v[:1],occ[:1],w[:1],sm[0],dm[0],core,radius)
    coverage=scope_coverage(occ[0],w[0],sm[0],dm[0],state_common,common,core,radius)
    coverage_by_domain={r['domain']:r for r in coverage}
    for row in rows:
        row['interpretation']=analysis_role(mode,coord,row['metric'])
        row['cohort_role']='primary analysis cohort' if split=='A_val' else 'held-out replication cohort (post hoc analysis)'
        row['scope']='common-supported domain only; not the full original observed domain'
        row['observed_minus_null_mean']=row['observed']-row['null_mean'] if row['observed'] is not None and row['null_mean'] is not None else None
        original_value=original_metrics.get(row['metric'],np.array([np.nan]))[0]
        row['observed_original_domain']=float(original_value) if np.isfinite(original_value) else None
        for domain,item in coverage_by_domain.items():
            row[f'{domain}_observed_cells']=item['observed_domain_cells'];row[f'{domain}_common_cells']=item['common_domain_cells']
            row[f'{domain}_retained_weight_fraction']=item['retained_weight_fraction']
    om=occ[0];original=tm&dm[0]
    mass=float(om[common].sum()/max(om[original].sum(),1e-12))
    rms=float(np.sqrt(np.sum(om[common]*(u[0,common]**2+v[0,common]**2))/om[common].sum())) if common.any() and om[common].sum()>0 else None
    descriptive=[];freq=dm[1:].mean(axis=0)
    count=dm[1:].sum(axis=0);nu=np.sum(np.where(dm[1:],u[1:],0.),axis=0)/np.maximum(count,1);nv=np.sum(np.where(dm[1:],v[1:],0.),axis=0)/np.maximum(count,1)
    for f in p['descriptive_support_frequencies']:
        mask=tm&dm[0]&(freq>=f);weight=om[mask];den=weight.sum()
        d=float(np.sqrt(np.sum(weight*((u[0,mask]-nu[mask])**2+(v[0,mask]-nv[mask])**2))/den)) if den>0 else None
        base=float(np.sqrt(np.sum(weight*(u[0,mask]**2+v[0,mask]**2))/den)) if den>0 else None
        descriptive.append(dict(support_frequency=f,cells=int(mask.sum()),retained_occupancy=float(den/max(om[original].sum(),1e-12)),distance=d,normalized_departure=d/base if d is not None and base and base>0 else None,p_mc=None,role='support sensitivity only; no deletion-conditioned tail test'))
    result=dict(coordinate=coord,common_cells=int(common.sum()),training_observed_intersection_cells=int(original.sum()),retained_observed_occupancy=mass,min_common_cells=mincells,full_field_estimable=estimable,restricted_support_warning=(mass<.90),geometry=geometry,observed_field_rms=rms,normalized_departure=float(distance[0]/rms) if rms and np.isfinite(distance[0]) and rms>0 else None,tests=rows,descriptive_support=descriptive,
                scope_coverage=coverage,retained_occupancy_denominator='training-supported intersection with observed drift support (legacy field retained)',
                scope='Inference is restricted to common-supported domains; metric-specific coverage uses the original observed domain.')
    atomic_npz(out/f'{coord}_field_comparison.npz',observed_u=u[0],observed_v=v[0],null_mean_u=u[1:].mean(axis=0),null_mean_v=v[1:].mean(axis=0),excess_u=u[0]-u[1:].mean(axis=0),excess_v=v[0]-v[1:].mean(axis=0),common_support=common,interior_common=stencil,observed_occupancy=om,simulation_support_frequency=freq)
    pd.DataFrame({'realization':np.arange(len(arrays))-1,**bank}).to_csv(out/f'{coord}_realization_statistics.csv',index=False)
    pd.DataFrame(rows).to_csv(out/f'{coord}_tests.csv',index=False);pd.DataFrame(descriptive).to_csv(out/f'{coord}_support_sensitivity.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/f'{coord}_scope_coverage.csv',index=False)
    grid=pd.DataFrame(dict(x_bin=np.repeat(np.arange(40),40),y_bin=np.tile(np.arange(40),40),observed_M=u[0].ravel(),observed_second=v[0].ravel(),null_M=u[1:].mean(axis=0).ravel(),null_second=v[1:].mean(axis=0).ravel(),common_support=common.ravel(),sim_support_frequency=freq.ravel()))
    grid.to_csv(out/f'{coord}_field_grid.csv',index=False)
    write_figures(out,coord,result,bank,arrays,common,stencil,core)
    return result


def summarize(work,split,mode):
    work=Path(work);p=load_protocol(work)
    if p['smoke']:raise RuntimeError('Smoke data cannot be used for inferential summaries.')
    if split=='B_confirm':require_confirm(work,p)
    out=work/'summary'/split/mode
    with lock(work/f'locks/summary_{split}_{mode}.lock'):
        out.mkdir(parents=True,exist_ok=True);invalidate_final(work)
        manifest_path=out/'summary_artifacts.json'
        if manifest_path.exists():manifest_path.unlink()
        if split=='A_val' and (work/'A_val_release.json').exists():(work/'A_val_release.json').unlink()
        arrays,metrics=read_realizations(work,p,split,mode)
        results={c:analyze_coordinate(work,p,split,mode,c,arrays,out) for c in p['coordinates']}
        pd.DataFrame([dict(replicate=x['replicate'],elapsed_seconds=x['elapsed_seconds'],**x['calibration']) for x in metrics]).to_csv(out/'response_calibration.csv',index=False)
        summary=dict(schema_version=SUMMARY_SCHEMA_VERSION,artifacts_manifest='summary_artifacts.json',protocol_id=p['protocol_id'],split=split,mode=mode,replicates=p['replicates'],seed=p['seed'],method='symmetric R+1 realization ranking, conservative ties, all completed replicates retained',coordinates=results,limitations=p['limitations'],technical_complete=True,scientific_success_not_a_gate=True)
        atomic_json(out/'summary.json',summary)
        lines=[f'# {split} / {mode}: jointly reconstructed response null','',f'Protocol: `{p["protocol_id"]}`',f'Replicates: {p["replicates"]}. No simulated realizations were deleted.','',
            'The original Stage-1 builder was used from empty per-user accumulators. Item outcomes, response evidence, unresolved demand, activity alignment and both coordinates were rebuilt jointly. The recorded opportunity schedule was fixed.','',
            'These post hoc analyses use plug-in training baselines and do not identify a policy counterfactual or autonomous state dependence.','']
        for c,r in results.items():
            lines += [f'## {c}',f'Common cells: {r["common_cells"]}; retained training×observed-support occupancy: {r["retained_observed_occupancy"]:.2%}.',f'Normalized departure: {r["normalized_departure"]}.','', '| Metric | Observed on common domain | Observed on original domain | Null mean | Observed minus null mean | pMC | BH q | Status |','|---|---:|---:|---:|---:|---:|---:|---|']
            for t in r['tests']:lines.append('| '+' | '.join(str(t.get(k,'')) for k in ['metric','observed','observed_original_domain','null_mean','observed_minus_null_mean','p_mc','q_bh_three_basin_metrics','status'])+' |')
            lines += ['', '| Domain | Original observed cells | Common cells | Weight | Retained weight fraction |','|---|---:|---:|---|---:|']
            for item in r['scope_coverage']:lines.append('| '+' | '.join(str(item[k]) for k in ['domain','observed_domain_cells','common_domain_cells','weight_kind','retained_weight_fraction'])+' |')
            lines += ['', 'Insufficient common support is reported as not estimable, not as evidence for or against the null. Low retained mass limits every conclusion to the supported overlap.','']
        (out/'numeric_report.md').write_text('\n'.join(lines),encoding='utf-8')
        atomic_json(manifest_path,artifact_manifest(out,p,split,mode))
        validate_summary_evidence(work,p,split,mode,validate_inputs=False)
        print(f'[summary] {out}',flush=True)


def release_validation(work):
    work=Path(work).resolve()
    with lock(work/'locks/release.lock'):
        if (work/'A_val_release.json').exists():(work/'A_val_release.json').unlink()
        invalidate_final(work);p=load_protocol(work)
        if p['smoke'] or set(p['modes'])!={'null_a','null_b'} or len(p['modes'])!=2:raise RuntimeError('Invalid formal release protocol')
        refs={};manifests={}
        for mode in p['modes']:
            validate_summary_evidence(work,p,'A_val',mode)
            f=work/'summary/A_val'/mode/'summary.json';mf=f.with_name('summary_artifacts.json')
            refs[str(f)]=sha256(f);manifests[str(mf)]=sha256(mf)
        atomic_json(work/'A_val_release.json',dict(schema_version=SUMMARY_SCHEMA_VERSION,protocol_id=p['protocol_id'],summaries=refs,manifests=manifests,criterion='technical completeness only; significance and positive findings are not requirements'))


def final_report(work):
    work=Path(work).resolve()
    with lock(work/'locks/final.lock'):
        invalidate_final(work)
        return _final_report(work)


def _final_report(work):
    p=load_protocol(work);rows=[];reports={};evidence_hashes={}
    if p.get('smoke') or p.get('replicates')!=999 or set(p.get('modes',[]))!={'null_a','null_b'} or len(p['modes'])!=2:raise RuntimeError('Final report requires both formal nulls and 999 replicates')
    require_confirm(work,p)
    for split in ('A_val','B_confirm'):
        for mode in p['modes']:
            s,_=validate_summary_evidence(work,p,split,mode)
            folder=work/'summary'/split/mode
            for name in ('summary.json','summary_artifacts.json','psi_field_comparison.npz','phi_field_comparison.npz'):
                file=folder/name;evidence_hashes[str(file.relative_to(work))]=sha256(file)
            reports[(split,mode)]=s
            for coord,r in s['coordinates'].items():
                for t in r['tests']:rows.append(dict(split=split,mode=mode,coordinate=coord,common_cells=r['common_cells'],retained_occupancy=r['retained_observed_occupancy'],normalized_departure=r['normalized_departure'],**t))
    out=work/'final';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'construction_null_statistics.csv',index=False)
    legacy=[];legacy_sources=[]
    for sp,rootname in [('A_val','stage1_construction_matched_null'),('B_confirm','stage1_construction_matched_null_confirm')]:
        base=Path(p['output_root'])/rootname/'tables'/f'{sp}_construction_null_summary'
        try:
            file=find_table(base);table=read_table(file)
            table['legacy_source_file']=str(file)
            table['reference_scope']='original conditional innovation-reassignment null; not the generative null'
            legacy.append(table);legacy_sources.append(dict(path=str(file),sha256=sha256(file)))
        except FileNotFoundError:
            legacy_sources.append(dict(path=str(base),status='unavailable'))
    if legacy:pd.concat(legacy,ignore_index=True).to_csv(out/'legacy_innovation_reassignment_tests.csv',index=False)
    elif (out/'legacy_innovation_reassignment_tests.csv').exists():(out/'legacy_innovation_reassignment_tests.csv').unlink()
    atomic_json(out/'legacy_sources.json',legacy_sources)
    replication=[]
    for mode in p['modes']:
        for coord in p['coordinates']:
            with np.load(work/'summary/A_val'/mode/f'{coord}_field_comparison.npz',allow_pickle=False) as a, np.load(work/'summary/B_confirm'/mode/f'{coord}_field_comparison.npz',allow_pickle=False) as b:
                mask=a['common_support']&b['common_support'];va=np.column_stack([a['excess_u'][mask],a['excess_v'][mask]]);vb=np.column_stack([b['excess_u'][mask],b['excess_v'][mask]])
                wt=np.sqrt(a['observed_occupancy'][mask]*b['observed_occupancy'][mask])
                replication.append(dict(mode=mode,coordinate=coord,scope='descriptive common-overlap excess fields; shared baseline error is not removed',**vector_diagnostics(va,vb,wt)))
    pd.DataFrame(replication).to_csv(out/'excess_field_replication_descriptive.csv',index=False)
    report_lines=['# Generative construction controls','',
        '## Joint reconstruction',
        'The original conditional innovation-reassignment reference does not preserve all innovation-denominator dependence. The response-generating controls use frozen training item baselines, the recorded submitted-bundle schedule and the original state builder. Signed and absolute evidence, unresolved demand, response/support alignment and both macrostates are reconstructed jointly. Numerical results for each cohort and generating reference are in construction_null_statistics.csv.','',
        '## Generating references and coordinate comparison',
        'Null A draws nonmissing final-response correctness from frozen item probabilities and preserves the joint signed/absolute residual relationship. Missing responses are not imputed. The activity-idle comparator is evaluated on the same eligible trajectories under its own frozen training support and core; normalized-departure contrasts are descriptive and do not test coordinate optimality. Null B adds a training-only stable-heterogeneity sensitivity with one user effect per trajectory; its calibrated marginal and simulated effect distributions are identical. It is not a complete model of evolving ability or item selection.','',
        '## Field and convergence diagnostics',
        'Field departure, negative-divergence occupancy, inward flow and core slowing are reported separately. Field departure does not substitute for unsupported slowing. The core is never selected again using simulated or held-out outcomes. Each statistic uses its supported subset; core, shell and interior-stencil coverage are reported separately.','',
        '## Interpretation',
        'Inference is conditional on the recorded opportunity schedule, which may encode adaptive policy and prior responses. Neither rejection nor nonrejection alone identifies a causal platform mechanism. Item baselines are plug-in training estimates; their estimation uncertainty is not integrated. The validation and confirmation cohorts are within-platform held-out cohorts used in the original study; these post hoc checks are not a prospective replication on previously untouched data.','',
        'Model-recovery results relative to the original innovation-reassignment reference remain separate from these response-generating controls. No downstream model is fitted by this workflow.','',
        '## Scope and endpoint roles',
        'Null A/Psi is primary; Null B assesses stable heterogeneity and Phi is a secondary coordinate comparison. A_val and B_confirm retain their analysis and replication roles. The BH adjustment covers the three convergence metrics within each generating reference, cohort and coordinate; it is not a correction across all reported endpoints.',
        'All inferential results concern the common-supported domain, not the full original observed domain. Field and stencil coverage use observed occupancy; core and shell coverage use the square root of observed drift weight. Original-domain values are descriptive and are not substituted into common-support tests. An observed-minus-null-mean effect is a signed difference, not a confidence interval.','',
        '## Results','| Split | Null | Coordinate | Metric | Role | Common cells | Original-field mass retained | Observed | Null mean | Difference | pMC | BH q | Status |','|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:report_lines.append('| '+' | '.join(str(r.get(k,'')) for k in ['split','mode','coordinate','metric','interpretation','common_cells','field_retained_weight_fraction','observed','null_mean','observed_minus_null_mean','p_mc','q_bh_three_basin_metrics','status'])+' |')
    report_lines += ['', '## Metric-specific supported domains', '| Split | Null | Coordinate | Domain | Original observed cells | Common cells | Retained observed weight | Weight definition |','|---|---|---|---|---:|---:|---:|---|']
    for (split,mode),report in reports.items():
        for coord,result in report['coordinates'].items():
            for item in result['scope_coverage']:
                report_lines.append('| '+' | '.join(str(x) for x in (split,mode,coord,item['domain'],item['observed_domain_cells'],item['common_domain_cells'],item['retained_weight_fraction'],item['weight_kind']))+' |')
    (out/'construction_null_report.md').write_text('\n'.join(report_lines),encoding='utf-8')

    for relative,h in evidence_hashes.items():
        if sha256(work/relative)!=h:raise RuntimeError(f'Evidence changed during final assembly: {relative}')
    final_manifest=artifact_manifest(out,p);final_manifest['evidence_hashes']=evidence_hashes
    atomic_json(out/'final_artifacts.json',final_manifest)
    atomic_json(out/'COMPLETE.json',dict(schema_version=SUMMARY_SCHEMA_VERSION,protocol_id=p['protocol_id'],technical_complete=True,replicates_per_mode_split=p['replicates'],evidence_hashes=evidence_hashes,final_artifacts_sha256=sha256(out/'final_artifacts.json'),scientific_conclusions_require_reading_evidence_table=True))


def main():
    a=argparse.ArgumentParser();a.add_argument('command',choices=['summarize','release','final']);a.add_argument('--work-root',required=True);a.add_argument('--split',choices=['A_val','B_confirm']);a.add_argument('--mode',choices=['null_a','null_b']);x=a.parse_args()
    if x.command=='summarize':
        if not x.split or not x.mode:a.error('--split and --mode required')
        summarize(x.work_root,x.split,x.mode)
    elif x.command=='release':release_validation(x.work_root)
    else:final_report(x.work_root)

if __name__=='__main__':main()
