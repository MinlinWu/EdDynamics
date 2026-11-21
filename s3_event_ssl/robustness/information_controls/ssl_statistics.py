
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from ssl_common import *
from ssl_data import labels

BINS=np.linspace(-1.,1.,41);NCELL=1600;MIN_COUNT=30

def cells(x):
    x=np.asarray(x,float);require(x.ndim==2 and x.shape[1]==2 and np.isfinite(x).all(),'Invalid field origin')
    require((np.abs(x)<=1.000001).all(),'Field origin outside signed domain')

    z=np.clip(x,-1,1);z=np.where(z==1,np.nextafter(1.,-1.),z)
    i=np.digitize(z[:,0],BINS)-1;j=np.digitize(z[:,1],BINS)-1
    return (40*i+j).astype(np.int64)

def corr(a,b):
    a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel()
    if len(a)<3 or len(a)!=len(b) or not(np.isfinite(a).all() and np.isfinite(b).all()):return np.nan
    a=a-a.mean();b=b-b.mean();d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d>EPS else np.nan

def field(origin,delta,w):
    c=cells(origin);count=np.bincount(c,minlength=NCELL);W=np.bincount(c,weights=w,minlength=NCELL)
    ss=np.column_stack([np.bincount(c,weights=w*delta[:,a],minlength=NCELL) for a in range(2)])
    v=np.divide(ss,W[:,None],out=np.full_like(ss,np.nan),where=W[:,None]>0)
    return {'count':count,'mass':W,'field':v,'support':count>=MIN_COUNT}

def field_metrics(a,b,W,S):
    positive=S&(W>0);den=W[positive].sum()
    if den<=EPS:return {'status':'not_estimable','D2':np.nan,'D2_M':np.nan,'D2_Psi':np.nan,'cells':int(S.sum())}
    e=np.sum(W[positive,None]*(b[positive]-a[positive])**2,axis=0)/den
    complete=np.isfinite(a[S]).all() and np.isfinite(b[S]).all()
    r=corr(a[S],b[S]) if complete else np.nan
    speeda=np.linalg.norm(a,axis=1);speedb=np.linalg.norm(b,axis=1)
    good=positive&(speeda*speedb>EPS)
    cosine=float(np.sum(W[good]*np.sum(a[good]*b[good],axis=1)/(speeda[good]*speedb[good]))/W[good].sum()) if good.any() else np.nan
    energy=float(np.sum(W[positive,None]*a[positive]**2)/den)
    slope=float(np.sum(W[positive,None]*a[positive]*b[positive])/(den*energy)) if energy>EPS else np.nan
    return {'status':'complete' if complete else 'weighted_error_only','D2':float(e.sum()),'D2_M':float(e[0]),'D2_Psi':float(e[1]),
       'vector_r':r,'speed_r':corr(speeda[S],speedb[S]) if complete else np.nan,
       'local_cosine':cosine,'cosine_mass_fraction':float(W[good].sum()/den),
       'amplitude_slope':slope,'reference_energy':energy,'reference_relative_skill':1-float(e.sum())/energy if energy>EPS else np.nan,
       'cells':int(S.sum()),'support_mass':float(den),'support_mass_fraction':float(den/max(W.sum(),EPS))}

def transition(X,Y,C,N,part):
    arr=[X,Y,C,N];labs=[]
    for x in arr:
        lab=np.empty(len(x),np.int64)
        for s in range(0,len(x),200000):lab[s:s+200000]=labels(x[s:s+200000],part)
        labs.append(lab)
    ce=np.bincount(labs[0]*6+labs[1],minlength=36).reshape(6,6)
    ca=np.bincount(labs[0]*6+labs[3],minlength=36).reshape(6,6)
    cl=np.bincount(labs[2]*6+labs[3],minlength=36).reshape(6,6)
    def normalize(c):return np.divide(c,c.sum(axis=1,keepdims=True),out=np.full(c.shape,np.nan,float),where=c.sum(axis=1,keepdims=True)>0)
    ep=normalize(ce);ap=normalize(ca);lp=normalize(cl)
    info={};rows=[]
    for view,p,counts in [('anchor',ap,ca),('readout',lp,cl)]:
        tv=.5*np.sum(np.abs(p-ep),axis=1)
        info[view+'_transition_mean_row_TV']=float(np.mean(tv)) if np.isfinite(tv).all() else np.nan
        info[view+'_self_transition_r_descriptive']=corr(np.diag(ep),np.diag(p))
        for a in range(6):
            for b in range(6):rows.append({'view':view,'origin':a,'destination':b,'empirical_count':ce[a,b],
             'model_count':counts[a,b],'empirical_probability':ep[a,b],'model_probability':p[a,b],
             'row_TV':tv[a],'Pii_difference':p[a,a]-ep[a,a]})
    return info,rows


def evaluate_predictions(data,currents,nexts,part,dest):

    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    X=np.asarray(data.values('X'),float);Y=np.asarray(data.values('Y'),float);uid=data.values('uid')
    users,ui,nrows=np.unique(uid,return_inverse=True,return_counts=True);w=1./nrows[ui]
    empirical=field(X,Y-X,w);rows=[];grid=[];trs=[]
    for name in nexts:
        C=np.asarray(currents[name],float);N=np.asarray(nexts[name],float)
        require(C.shape==X.shape and N.shape==Y.shape and np.isfinite(C).all() and np.isfinite(N).all(),'Prediction rows/shapes/nonfinites changed')
        if (np.abs(C)>1.000001).any() or (np.abs(N)>1.000001).any():raise AuditError('Prediction outside fixed coordinate range')
        dcur=C-X;dn=N-Y;dv=(N-C)-(Y-X)
        row={'model':name,'split':data.split,'rows':len(X),'learners':len(users),
          'current_MSE':float(np.mean(dcur**2)),'next_MSE':float(np.mean(dn**2)),
          'interval_velocity_MSE':float(np.mean(dv**2)),
          'error_difference_identity_max':float(np.max(np.abs(dv-(dn-dcur))))}
        for j,axis in enumerate(('M','Psi')):
            row.update({f'current_rmse_{axis}':float(np.sqrt(np.mean(dcur[:,j]**2))),
              f'one_step_rmse_{axis}':float(np.sqrt(np.mean(dn[:,j]**2))),
              f'current_r_{axis}':corr(C[:,j],X[:,j]),f'next_r_{axis}':corr(N[:,j],Y[:,j]),
              f'user_equal_next_rmse_{axis}':float(np.sqrt(np.sum(w*dn[:,j]**2)/len(users)))})
        ef=empirical
        views={'endpoint':(ef,field(X,N-X,w)), 'velocity':(ef,field(X,N-C,w)),
               'learned_plane':(ef,field(C,N-C,w)),
               'matched_origin':(field(C,Y-C,w),field(C,N-C,w))}
        for view,(e,p) in views.items():
            S=e['support']&p['support'];fm=field_metrics(e['field'],p['field'],e['mass'],S)
            row.update({view+'_'+k:v for k,v in fm.items()})
            for g in range(NCELL):grid.append({'model':name,'view':view,'cell':g,'count_emp':e['count'][g],
              'count_model':p['count'][g],'mass':e['mass'][g],'supported':bool(S[g]),
              'emp_M':e['field'][g,0],'emp_Psi':e['field'][g,1],
              'model_M':p['field'][g,0],'model_Psi':p['field'][g,1]})
        tm,tr=transition(X,Y,C,N,part);row.update(tm)
        for t in tr:t['model']=name;t['split']=data.split
        trs.extend(tr);rows.append(row)
    write_csv(rows,dest/'metrics.csv');write_csv(grid,dest/'field_grids.csv.gz');write_csv(trs,dest/'statewise_transitions.csv')
    save_json({'rows':len(X),'users':len(users),'field_weight':'1/n_valid_rows per learner',
      'interval_weight':'one per interval','transition_weight':'one per interval then row normalization',
      'primary_anchor':'exact original SSL prepared current coordinates',
      'gauges':{'endpoint':'N-X vs Y-X on X','velocity':'N-C vs Y-X on X',
      'learned_plane':'N-C on C vs Y-X on X','matched_origin':'N-C vs Y-C on C'},
      'grid':[40,40,-1,1],'minimum_count':30,'not_a_policy_or_autonomous_test':True},dest/'evaluation_contract.json')
    return pd.DataFrame(rows)


def basic_interval(v,point,level):
    v=np.asarray(v,float);finite=np.isfinite(v);n=int(finite.sum())
    out={'point':point,'level':level,'requested_replicates':len(v),'finite_replicates':n}
    if n!=len(v) or n<2 or not np.isfinite(point):
        return {**out,'status':'not_estimable',**{k:np.nan for k in ['basic_low','basic_high','percentile_low','percentile_high','se_low','se_high','bootstrap_mean','bootstrap_bias','bootstrap_sd','bias_over_sd','method_sign_conflict']}}
    q=np.quantile(v,[(1-level)/2,1-(1-level)/2]);sd=float(v.std(ddof=1));bias=float(v.mean()-point);se=norm.ppf(.5+level/2)*sd
    out.update(status='complete',basic_low=2*point-q[1],basic_high=2*point-q[0],percentile_low=q[0],percentile_high=q[1],
               se_low=point-se,se_high=point+se,bootstrap_bias=bias,bootstrap_sd=sd,bias_over_sd=bias/sd if sd>EPS else np.nan)
    sign=lambda a,b:1 if a>0 else (-1 if b<0 else 0)
    out['method_sign_conflict']=len({sign(out[a],out[b]) for a,b in [('basic_low','basic_high'),('percentile_low','percentile_high'),('se_low','se_high')]})>1

    out['bootstrap_mean']=float(v.mean())
    return out


def interval_metadata(support,level,family_size):

    return {'primary_analysis':support=='fixed' and level==.975,
        'primary_support':'fixed','support_role':'primary_support' if support=='fixed' else 'support_sensitivity',
        'family_scope':'prespecified contrasts within this block','family_size':family_size,
        'inference_condition':'seed42; frozen fitted models/heads/probes; whole-learner resampling',
        'coverage_status':'not_verified','availability_status_is_not_coverage':True}


class UserSufficient:

    def __init__(self,data,currents,nexts):
        X=np.asarray(data.values('X'),float);Y=np.asarray(data.values('Y'),float);uid=data.values('uid')
        self.names=list(nexts);self.users,ui,ct=np.unique(uid,return_inverse=True,return_counts=True)
        U=len(ct);w=1./ct[ui];c=cells(X)
        pair=ui.astype(np.int64)*NCELL+c;unique,inverse=np.unique(pair,return_inverse=True);ru=unique//NCELL;co=unique%NCELL
        def uc(v):
            b=np.bincount(inverse,weights=v,minlength=len(unique));return sparse.coo_matrix((b,(ru,co)),shape=(U,NCELL)).tocsr()
        mats=[uc(np.ones(len(X))),uc(w)]
        for j in range(2):mats.append(uc(w*(Y[:,j]-X[:,j])))
        small=[ct.astype(float)]
        for name in self.names:
            N=np.asarray(nexts[name],float);C=np.asarray(currents[name],float)
            for residual in (N-Y,(N-C)-(Y-X)):
                for j in range(2):mats.append(uc(w*residual[:,j]))
            for err in (C-X,N-Y):
                for j in range(2):small.append(np.bincount(ui,weights=err[:,j]**2,minlength=U))
        mats.append(sparse.csr_matrix(np.column_stack(small)))
        self.matrix=sparse.hstack(mats,format='csr');self.matrix.eliminate_zeros()
        self.point=np.asarray(self.matrix.sum(axis=0)).ravel();self.fixed=self.point[:NCELL]>=MIN_COUNT
        self.scalar_start=(4+4*len(self.names))*NCELL
    def summarize(self,v,support):
        count=v[:NCELL];W=v[NCELL:2*NCELL];S=self.fixed if support=='fixed' else count>=MIN_COUNT
        pos=S&(W>0);den=W[pos].sum();res={};off=4*NCELL
        small=v[self.scalar_start:];N=small[0]
        for k,name in enumerate(self.names):
            for view in ('endpoint','velocity'):
                r=np.column_stack([v[off:off+NCELL],v[off+NCELL:off+2*NCELL]]);off+=2*NCELL
                axes=np.sum(r[pos]**2/W[pos,None],axis=0)/den if den>EPS else [np.nan,np.nan]
                res[(name,view+'_D2')]=float(np.sum(axes))
                res[(name,view+'_D2_M')]=axes[0];res[(name,view+'_D2_Psi')]=axes[1]
            res[(name,'current_MSE')]=float(np.sum(small[1+4*k:3+4*k])/(2*N)) if N>0 else np.nan
            res[(name,'next_MSE')]=float(np.sum(small[3+4*k:5+4*k])/(2*N)) if N>0 else np.nan
        meta={'empty_support_cells':int(np.sum(S&(W<=0))),'support_cells':int(S.sum()),
              'support_mass':float(den),'support_mass_fraction':float(den/W.sum()) if W.sum()>0 else np.nan,
              'support_jaccard':float((S&self.fixed).sum()/max((S|self.fixed).sum(),1))}
        return res,meta


def paired_bootstrap(data,currents,nexts,comparisons,dest,nboot=2000,seed=20261117):

    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);stats=UserSufficient(data,currents,nexts)
    points={};records=[]
    for support in ('fixed','reselected'):
        p,meta=stats.summarize(stats.point,support)
        points[support]={label:p[(a,metric)]-p[(b,metric)] for label,a,b,metric in comparisons}

    for first in range(0,nboot,16):
        ids=range(first,min(first+16,nboot));mult=[]
        for r in ids:
            rng=np.random.default_rng(np.random.SeedSequence([seed,r]));sample=rng.integers(len(stats.users),size=len(stats.users))
            mult.append(np.bincount(sample,minlength=len(stats.users)).astype(float))
        V=np.asarray(stats.matrix.T@np.stack(mult,axis=1))
        for j,r in enumerate(ids):
            for support in ('fixed','reselected'):
                values,meta=stats.summarize(V[:,j],support)
                row={'replicate':r,'support_contract':support,**meta}
                for label,a,b,metric in comparisons:row[label]=values[(a,metric)]-values[(b,metric)]
                records.append(row)
        if first%128==0:log(f'Paired user bootstrap: {min(first+16,nboot)}/{nboot}')
    df=pd.DataFrame(records);write_csv(df,dest/'paired_bootstrap_replicates.csv.gz')
    intervals=[]
    for support in ('fixed','reselected'):
        rows=df[df.support_contract==support]
        require(len(rows)==nboot and rows.replicate.nunique()==nboot,'Missing or duplicate bootstrap draw')
        for label,a,b,metric in comparisons:
            for level in (.95,.975):
                intervals.append({'contrast':label,'model_a':a,'model_b':b,'metric':metric,'support_contract':support,
                 'direction':'positive = model_a has larger error','primary_level':level==.975,
                 **interval_metadata(support,level,len(comparisons)),
                 **basic_interval(rows[label],points[support][label],level)})
    write_csv(intervals,dest/'paired_intervals.csv')
    save_json({'replicates':nboot,'cluster':'whole learner','model_refit':False,'seed':seed,
       'same_draws_all_compared_models':True,'primary':'fixed-support nominal 97.5% basic per two prespecified contrasts; 95% and reselected-support diagnostics',
       'primary_support':'fixed','family_scope':'within this block only; not across blocks or support rules',
       'family_size':len(comparisons),'coverage_status':'not_verified',
       'finite_or_small_bias_does_not_certify_coverage':True,
       'conditional_on':'frozen fitted models/heads/probes and declared data processing; no refitting',
       'zero_mass':'zero measure for weighted squared errors, never an imputed drift vector',
       'no_nonfinite_draws_deleted':True,'support_contracts_not_exchangeable':True,
       'seed_variation_not_pooled_with_user_uncertainty':True},dest/'bootstrap_contract.json')
    return pd.DataFrame(intervals)


def decomposition(data,p,dest):
    X=np.asarray(data.values('X'),float);Y=np.asarray(data.values('Y'),float)
    S=np.asarray(p[:,:2],float);T=np.asarray(p[:,2:4],float);P=np.asarray(p[:,4:6],float);P0=np.tanh(S)
    parts={'current_readout_error':S-X,'zero_residual_head_contraction':P0-S,'residual_after_outer_tanh':P-P0,
           'proposal_minus_same_readout':P-T,'same_readout_velocity_error':(T-S)-(Y-X)}
    error=float(np.max(np.abs(P-X-sum(parts[n] for n in list(parts)[:3]))))
    require(error<1e-12,'Proposal decomposition does not add back')
    _,ui,ct=np.unique(data.values('uid'),return_inverse=True,return_counts=True);w=1./ct[ui]
    rows=[];grids=[]
    for name,d in parts.items():
        for j,axis in enumerate(('M','Psi')):rows.append({'component':name,'axis':axis,
             'row_mean':float(d[:,j].mean()),'learner_balanced_mean':float(np.sum(w*d[:,j])/w.sum()),
             'row_rms':float(np.sqrt(np.mean(d[:,j]**2)))})
        f=field(X,d,w)
        for g in range(NCELL):grids.append({'component':name,'cell':g,'supported':f['support'][g],'mass':f['mass'][g],
                   'M':f['field'][g,0],'Psi':f['field'][g,1]})
    write_csv(rows,Path(dest)/'head_decomposition.csv');write_csv(grids,Path(dest)/'head_decomposition_fields.csv.gz')
    save_json({'max_addback_error':error,'P0_is_not_statistical_null':True,
       'component_squared_errors_not_additive_causal_contributions':True},Path(dest)/'decomposition_contract.json')
