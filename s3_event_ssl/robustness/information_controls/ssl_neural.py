
from __future__ import annotations
import copy, gc, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from ssl_common import *
from ssl_data import Prepared


def make_model(cfg,shapes):
    cls=module('train_event_ssl.py').PredictiveStateEventSSL
    return cls(n_num=int(shapes['n_num']),n_cat=int(shapes['n_cat']),hash_buckets=int(shapes['hash_buckets']),
       hidden_dim=int(cfg['hidden_dim']),input_dim=int(cfg['input_dim']),num_layers=int(cfg['num_layers']),
       dropout=float(cfg['dropout']),categorical_emb_dim=int(cfg['categorical_emb_dim']),
       future_steps=tuple(map(int,cfg['future_steps'])),delta_scale=float(cfg['delta_scale'])).cpu()


def load_model(checkpoint,data=None):
    ck=torch_load(checkpoint);require({'config','model_shapes','model_state_dict'}<=set(ck),'Unknown checkpoint schema')
    cfg=ck['config'];shapes=ck['model_shapes']
    if data is not None:
        require(shapes['n_num']==data.n_num and shapes['n_cat']==data.n_cat,'Checkpoint input dimensions mismatch')
        require(shapes['hash_buckets']==data.manifest['categorical_hash_buckets'],'Categorical hashing mismatch')
    m=make_model(cfg,shapes);m.load_state_dict(ck['model_state_dict'],strict=True);m.eval()
    return m,ck


def selected_indices(n,maximum,seed=42):
    if maximum>=n:return np.arange(n,dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n,maximum,replace=False))

@torch.inference_mode()
def trajectory_chunks(model,data,chunk=512,batch_sequences=16,clean_mask=None):

    model.eval();require(next(model.parameters()).device.type=='cpu','CPU contract violated')
    offsets=data.offsets

    order=np.argsort(-np.diff(offsets),kind='stable')
    for first in range(0,len(offsets)-1,batch_sequences):
        stops=[];starts=[]
        for si in order[first:min(first+batch_sequences,len(offsets)-1)]:
            s,e=data.sequence(si);starts.append(s);stops.append(e)
        starts=np.array(starts);stops=np.array(stops);pos=starts.copy();B=len(starts)
        hidden=None;previous=None
        while (pos<stops).any():
            length=np.minimum(stops-pos,chunk);T=int(length.max())
            xn=np.zeros((B,T,data.n_num),np.float32);xc=np.zeros((B,T,data.n_cat),np.int64)
            for b,L in enumerate(length):
                if L:
                    xn[b,:L]=data.values('num',slice(pos[b],pos[b]+L))
                    xc[b,:L]=data.values('cat',slice(pos[b],pos[b]+L))
            if clean_mask is not None:xn[...,clean_mask]=0
            z=model.embed_inputs(torch.from_numpy(xn),torch.from_numpy(xc))
            ha,hidden=model.rnn(z,hidden)
            initial=torch.zeros_like(ha[:,:1]) if previous is None else previous
            hb=torch.cat([initial,ha[:,:-1]],dim=1)
            S=model.state_head(hb);Tstate=model.state_head(ha)
            P=torch.tanh(S+model.delta_scale*model.delta_head(torch.cat([hb,z],dim=-1)))
            for b,L in enumerate(length):
                if L:
                    yield (np.arange(pos[b],pos[b]+L,dtype=np.int64),
                        hb[b,:L].float().numpy(),ha[b,:L].float().numpy(),z[b,:L].float().numpy(),
                        S[b,:L].float().numpy(),Tstate[b,:L].float().numpy(),P[b,:L].float().numpy())
            previous=ha[:,-1:,:].detach();hidden=hidden.detach();pos+=length


def predict_native(model,data,dest,chunk=512,batch_sequences=16,clean_mask=None,probe=None,future_common_keep=None):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)


    p=np.lib.format.open_memmap(dest/'trajectory.npy',mode='w+',dtype=np.float32,shape=(data.n,6))
    seen=np.zeros(data.n,bool);last=time.time();n=0;future_sums={}
    for idx,hb,ha,z,S,T,P in trajectory_chunks(model,data,chunk,batch_sequences,clean_mask):
        require(not seen[idx].any(),'Duplicate trajectory rows');seen[idx]=True
        if future_common_keep is not None:
            sequence_end=int(data.offsets[np.searchsorted(data.offsets,idx[0],side='right')])
            own=np.ones(data.n_num,bool) if clean_mask is None else ~np.asarray(clean_mask,bool)
            with torch.inference_mode():
                for horizon,head in model.future_heads.items():
                    k=int(horizon);valid=idx+k<sequence_end
                    if not valid.any():continue
                    target=data.values('num',idx[valid]+k).astype(float)
                    estimate=head(torch.from_numpy(ha[valid])).float().numpy().astype(float)
                    for scope,keep in [('own_training_targets',own),('common_retained_targets',future_common_keep)]:
                        keep=np.asarray(keep,bool);key=(k,scope)
                        se=float(np.sum((estimate[:,keep]-target[:,keep])**2));npairs=int(valid.sum())
                        old=future_sums.get(key,[0.,0,0]);future_sums[key]=[old[0]+se,old[1]+npairs,int(keep.sum())]
        if probe is not None:S=probe_predict(hb,probe);T=probe_predict(ha,probe);P=T
        p[idx]=np.column_stack([S,T,P]);n+=len(idx)
        if time.time()-last>30:log(f'CPU forward {data.split}: {n:,}/{data.n:,} rows');last=time.time()
    require(seen.all() and np.isfinite(p).all(),'Incomplete/nonfinite trajectory')
    maximum=0.;nlinks=0
    for si in range(len(data.offsets)-1):
        s,e=data.sequence(si)
        if e-s>1:
            maximum=max(maximum,float(np.max(np.abs(p[s:e-1,2:4]-p[s+1:e,:2]))));nlinks+=e-s-1
    require(maximum<3e-6,f'Same-readout continuity failed: {maximum:g}')
    p.flush()
    if future_common_keep is not None:
        write_csv([{'horizon':k,'target_scope':scope,'squared_error_sum':v[0],'observed_pairs':v[1],
                    'target_dimensions':v[2],'MSE':v[0]/(v[1]*v[2]),
                    'scope':'continuous heldout sequences; no padding or cross-boundary pairs; not used for checkpoint selection'}
                   for (k,scope),v in sorted(future_sums.items())],dest/'future_prediction_heldout.csv')
    save_json({'rows':data.n,'continuous_links':nlinks,'same_readout_max_error':maximum,
       'chunk':chunk,'sequence_batch':batch_sequences,'precision':'CPU float32',
       'user_or_noncontiguous_step_reset':True,'inference_chunk_reset':False,
       'readout':'one frozen ridge applied before/after' if probe is not None else 'one native frozen state_head',
       'proposal_column_is_surrogate_same_readout_for_probe':probe is not None,
       'row_order_source':str(data.root/data.split),'row_order':'exact prepared row order after whole-user smoke selection if any'},dest/'trajectory_audit.json')
    return p


def collect_features(model,data,indices,dest,chunk,batch_sequences,clean_mask=None,need_after=False):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);idx=np.asarray(indices,np.int64)
    require(np.array_equal(idx,np.unique(idx)) and idx.size and idx[-1]<data.n,'Invalid deterministic sample')
    H=int(model.rnn.hidden_size);Z=int(model.rnn.input_size)
    arrays={'hb':np.lib.format.open_memmap(dest/'hb.npy',mode='w+',dtype=np.float32,shape=(len(idx),H)),
            'z':np.lib.format.open_memmap(dest/'z.npy',mode='w+',dtype=np.float32,shape=(len(idx),Z)),
            'S':np.lib.format.open_memmap(dest/'S.npy',mode='w+',dtype=np.float32,shape=(len(idx),2))}
    if need_after:arrays['ha']=np.lib.format.open_memmap(dest/'ha.npy',mode='w+',dtype=np.float32,shape=(len(idx),H))
    seen=np.zeros(len(idx),bool)
    for rows,hb,ha,z,S,T,P in trajectory_chunks(model,data,chunk,batch_sequences,clean_mask):
        a=np.searchsorted(idx,rows[0]);b=np.searchsorted(idx,rows[-1],side='right')
        if a==b:continue
        j=idx[a:b]-rows[0];arrays['hb'][a:b]=hb[j];arrays['z'][a:b]=z[j];arrays['S'][a:b]=S[j]
        if need_after:arrays['ha'][a:b]=ha[j]
        seen[a:b]=True
    require(seen.all(),'Sample extraction missed a row')
    for a in arrays.values():a.flush()
    np.save(dest/'indices.npy',idx);np.save(dest/'keys.npy',data.keys()[idx])
    np.save(dest/'X.npy',data.values('X',idx));np.save(dest/'Y.npy',data.values('Y',idx))
    save_json({'rows':len(idx),'scope':data.split,'sample_seed':42,'sample_depends_on_targets':False,
               'history':'complete contiguous histories replayed before selecting hidden rows'},dest/'sample.json')
    return arrays


def fit_probe(features,dest):

    x=np.asarray(np.load(Path(features)/'hb.npy',mmap_mode='r'),np.float64)
    y=np.asarray(np.load(Path(features)/'X.npy'),np.float64)
    require(len(x)==len(y) and len(x)>1,'Insufficient probe sample')
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-12]=1.
    z=(x-mean)/scale;zm=z.mean(axis=0);ym=y.mean(axis=0)
    z-=zm;gram=z.T@z;cross=z.T@(y-ym)
    coef=np.linalg.solve(gram+np.eye(x.shape[1]),cross)
    out={'mean':mean,'scale':scale,'coef':coef,'intercept':ym-zm@coef,
         'alpha':1.,'fit_scope':'A_train only','same_map_before_after':True,'rows':len(x),
         'macro_labels_after_encoder_freeze_only':True,'clip':[-1,1]}
    save_json(out,Path(dest)/'probe.json');return out


def probe_predict(h,probe):
    x=(np.asarray(h,np.float64)-np.asarray(probe['mean']))/np.asarray(probe['scale'])
    return np.clip(x@np.asarray(probe['coef'])+np.asarray(probe['intercept']),-1,1).astype(np.float32)


class HeadDataset(Dataset):
    def __init__(self,root):
        p=Path(root);self.hb=np.load(p/'hb.npy',mmap_mode='r');self.z=np.load(p/'z.npy',mmap_mode='r')
        self.S=np.load(p/'S.npy',mmap_mode='r');self.Y=np.load(p/'Y.npy',mmap_mode='r')
    def __len__(self):return len(self.Y)
    def __getitem__(self,i):return tuple(torch.from_numpy(np.array(x[i],copy=True)) for x in (self.hb,self.z,self.S,self.Y))


def initialized_head(model,seed):
    torch.manual_seed(seed+7103);head=copy.deepcopy(model.delta_head).cpu()
    for m in head.modules():
        if isinstance(m,nn.Linear):m.reset_parameters()
    return head


def head_loss(head,h,z,S,Y,mode,scale):
    if mode=='H':z=torch.zeros_like(z)
    pred=torch.tanh(S+scale*head(torch.cat([h,z],dim=-1)))
    return torch.nn.functional.smooth_l1_loss(pred,Y,beta=.05),pred


def fit_head(model,trainpath,valpath,mode,seed,dest,settings,loader_workers=0):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    td=HeadDataset(trainpath);vd=HeadDataset(valpath)
    head=initialized_head(model,seed);initial_digest=tensor_digest(head.state_dict())
    opt=torch.optim.AdamW(head.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    gen=torch.Generator().manual_seed(seed+911)
    tr=DataLoader(td,batch_size=settings['batch_size'],shuffle=True,generator=gen,num_workers=loader_workers,drop_last=False)
    va=DataLoader(vd,batch_size=settings['batch_size'],shuffle=False,num_workers=loader_workers)
    history=[];best=float('inf');start=1
    cp=dest/'last_epoch.pt';identity=objhash({'mode':mode,'seed':seed,'settings':settings,'train':sha(Path(trainpath)/'keys.npy'),'val':sha(Path(valpath)/'keys.npy')})
    if cp.exists():
        ck=torch_load(cp);require(ck['identity']==identity,'Head resume contract mismatch')
        head.load_state_dict(ck['state']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);gen.set_state(ck['loader_rng'])
        history=ck['history'];best=ck['best'];start=ck['epoch']+1
    for ep in range(start,settings['epochs']+1):
        head.train();tot=0.;n=0
        for h,z,S,Y in tr:
            opt.zero_grad(set_to_none=True);loss,_=head_loss(head,h,z,S,Y,mode,model.delta_scale)
            require(torch.isfinite(loss).item(),'Nonfinite head training loss');loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            tot+=float(loss.detach())*len(Y);n+=len(Y)
        head.eval();vl=0.;vn=0
        with torch.no_grad():
            for h,z,S,Y in va:
                loss,_=head_loss(head,h,z,S,Y,mode,model.delta_scale);vl+=float(loss)*len(Y);vn+=len(Y)
        metric=vl/max(vn,1);require(np.isfinite(metric),'Nonfinite head validation')
        history.append({'epoch':ep,'train_huber':tot/max(n,1),'validation_huber':metric,'rows_train':n,'rows_val':vn})
        if metric<best:
            best=metric;torch_save({'state_dict':head.state_dict(),'mode':mode,'seed':seed,'epoch':ep,'identity':identity,
                 'selection':'minimum A_val one-step Huber; no confirmation access'},dest/'best_head.pt')
        torch_save({'state':head.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),
           'loader_rng':gen.get_state(),'history':history,'best':best,'epoch':ep,'identity':identity},cp)
        save_json(history,dest/'history.json');log(f'Head {mode}, seed {seed}, epoch {ep}: val={metric:.6g}')
    save_json({'seed':seed,'mode':mode,'initial_state_sha256':initial_digest,
      'train_keys_sha256':sha(Path(trainpath)/'keys.npy'),'val_keys_sha256':sha(Path(valpath)/'keys.npy'),
      'settings':settings,'frozen_encoder_and_state_head':True,'CPU_float32':True},dest/'head_contract.json')
    head.load_state_dict(torch_load(dest/'best_head.pt')['state_dict']);head.eval();return head

@torch.inference_mode()
def predict_heads(model,heads,data,dest,chunk,batch_sequences):
    dest=Path(dest);p=np.lib.format.open_memmap(dest/'heads.npy',mode='w+',dtype=np.float32,shape=(data.n,6))
    for idx,hb,ha,z,S,T,P in trajectory_chunks(model,data,chunk,batch_sequences):
        h=torch.from_numpy(hb);zz=torch.from_numpy(z);ss=torch.from_numpy(S)
        hpred=torch.tanh(ss+model.delta_scale*heads['H'](torch.cat([h,torch.zeros_like(zz)],dim=-1)))
        zpred=torch.tanh(ss+model.delta_scale*heads['HZ'](torch.cat([h,zz],dim=-1)))
        p[idx]=np.column_stack([S,hpred.numpy(),zpred.numpy()])
    p.flush();require(np.isfinite(p).all(),'Nonfinite head predictions');return p


class FutureWindows(Dataset):

    def __init__(self,data,seq_len=256,stride=128,min_seq_len=3):
        self.data=data;self.seq_len=seq_len;self.windows=[]
        for a,b in zip(data.offsets[:-1],data.offsets[1:]):
            s,e=int(a),int(b)
            if e-s<min_seq_len:continue
            first=min(s+seq_len,e);self.windows.append((s,first))
            if first>=e:continue
            pos=s+max(1,stride)
            while pos+min_seq_len<=e:
                end=min(pos+seq_len,e);self.windows.append((pos,end))
                if end==e:break
                pos+=max(1,stride)
        require(len(self.windows)>0,'No future training windows')
    def __len__(self):return len(self.windows)
    def __getitem__(self,i):
        s,e=self.windows[i];L=e-s
        x=np.zeros((self.seq_len,self.data.n_num),np.float32);c=np.zeros((self.seq_len,self.data.n_cat),np.int64);m=np.zeros(self.seq_len,np.float32)
        x[:L]=self.data.values('num',slice(s,e));c[:L]=self.data.values('cat',slice(s,e));m[:L]=1
        return {'x':torch.from_numpy(x),'cat':torch.from_numpy(c),'mask':torch.from_numpy(m)}


def future_objective(model,x,cat,observed,removed=None,score_keep=None):
    tr=module('train_event_ssl.py')
    if removed is not None:
        x=x.clone();x[...,removed]=0
    out=model(x,cat)
    keep=np.ones(model.n_num,bool) if removed is None else ~np.asarray(removed,bool)
    keep=torch.as_tensor(keep,dtype=torch.bool)
    total=torch.zeros((),dtype=x.dtype);common=torch.zeros((),dtype=x.dtype);nh=0
    commonkeep=keep if score_keep is None else torch.as_tensor(score_keep,dtype=torch.bool)
    for k,p in out['future'].items():
        target,valid=tr.shifted_future_target(x,observed,k)
        if not bool((valid>0).any()):continue
        total=total+tr.masked_mse(p[...,keep],target[...,keep],valid)
        common=common+tr.masked_mse(p[...,commonkeep],target[...,commonkeep],valid);nh+=1
    require(nh>0,'No real source/target future pair')
    return total/nh,common/nh


def train_pure(data_train,data_val,template,seed,variant,removed,dest,settings,loader_workers=0,smoke=False):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    cfg=dict(template['config']);shapes=template['model_shapes']
    cfg.update(model_kind='pure_ssl',seed=seed,epochs=settings['epochs'],lambda_future=1.,lambda_state=0.,lambda_closure=0.,
               seq_len=settings['seq_len'],stride=settings['stride'],min_seq_len=settings['min_seq_len'],
               warmup_steps=settings['warmup_steps'],batch_size=settings['batch_size'],lr=settings['lr'],weight_decay=settings['weight_decay'],
               future_steps=settings['future_steps'],use_compile=False,amp_dtype='none_cpu_fp32',allow_truncated_supervision=False)
    torch.manual_seed(seed);np.random.seed(seed)
    model=make_model(cfg,shapes);initial_digest=tensor_digest(model.state_dict())
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(seed+101)
    ds=FutureWindows(data_train,settings['seq_len'],settings['stride'],settings['min_seq_len'])
    vs=FutureWindows(data_val,settings['seq_len'],settings['stride'],settings['min_seq_len'])
    require(smoke or len(ds)>=settings['batch_size'],'Fewer than one complete original-size training batch')
    dl=DataLoader(ds,batch_size=settings['batch_size'],shuffle=True,generator=gen,num_workers=loader_workers,
                  drop_last=not smoke,persistent_workers=False)
    vl=DataLoader(vs,batch_size=settings['batch_size'],shuffle=False,num_workers=loader_workers,persistent_workers=False)
    identity=objhash({'config':cfg,'variant':variant,'feature_mask':removed,'code':code_digest(),
                      'input_root':str(data_train.root),'rows':data_train.n,'smoke':smoke})
    start=1;history=[];cp=dest/'last_epoch.pt'
    if cp.exists():
        ck=torch_load(cp);require(ck['identity']==identity,'Pure replay resume identity changed')
        model.load_state_dict(ck['model_state_dict']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);gen.set_state(ck['loader_rng'])
        start=ck['epoch']+1;history=ck['history']
    mask=removed if variant=='clean' else None
    for ep in range(start,settings['epochs']+1):
        model.train();tot=common=n=0;last=time.time()
        for b in dl:
            opt.zero_grad(set_to_none=True)
            loss,co=future_objective(model,b['x'],b['cat'],b['mask'],mask,~removed)
            require(torch.isfinite(loss).item(),'Nonfinite pure future objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            tot+=float(loss.detach());common+=float(co.detach());n+=1
            if time.time()-last>30:log(f'Pure {variant} seed={seed} epoch={ep}: {n}/{len(dl)} CPU batches');last=time.time()
        require(n>0,'No optimizer updates')
        model.eval();vt=vc=nv=0
        with torch.no_grad():
            for b in vl:
                loss,co=future_objective(model,b['x'],b['cat'],b['mask'],mask,~removed)
                vt+=float(loss);vc+=float(co);nv+=1
        history.append({'epoch':ep,'train_future_loss':tot/n,'train_common_target_loss':common/n,
          'val_future_loss':vt/max(nv,1),'val_common_target_loss':vc/max(nv,1),'batches':n,'selection':'none; fixed epoch'})
        ck={'model_state_dict':model.state_dict(),'config':cfg,'model_shapes':shapes,'epoch':ep,'best_epoch':ep,
            'identity':identity,'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'loader_rng':gen.get_state(),
            'history':history,'variant':variant,'clean_mask':removed,'future_only':True,
            'macro_labels_in_loss':False,'checkpoint_selection':'fixed epoch','CPU_precision':'float32',
            'original_source_sha256':load_json(ROOT/'source_contract.json')['files']['train_event_ssl.py']}
        torch_save(ck,cp);save_json(history,dest/'training_history.json')
        log(f'Pure {variant} seed {seed}, epoch {ep} done; validation is recorded, not selected')
    require(len(history)==settings['epochs'],'Pure training did not reach fixed endpoint')
    final=torch_load(cp)
    for k in ('optimizer','rng','loader_rng'): final.pop(k,None)
    torch_save(final,dest/'fixed_epoch_model.pt')
    save_json({'identity':identity,'config':cfg,'variant':variant,'original_data_windows':len(ds),
      'initial_state_sha256':initial_digest,
      'train_sample_filter':'original finite macrostate eligibility inherited; not label-free sample selection',
      'gradient':'future loss only; direct macrostate losses absent','selection':'fixed epoch, never M/Psi validation',
      'parameter_width_unchanged':True,'features_excluded_from_target_denominator':np.flatnonzero(removed).tolist() if variant=='clean' else [],
      'reference':'CPU-matched replay; not silently substituted for the historical best_model'},dest/'training_contract.json')
    model.eval();return model
