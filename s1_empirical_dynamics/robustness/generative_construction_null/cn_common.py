from __future__ import annotations
import contextlib, dataclasses, gzip, hashlib, importlib.util, json, os, pickle, re, sys, uuid
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

VERSION = 'generative-construction-null-3.0'
DEFAULT_OUTPUT = '/path/to/outputs_KT4'
NEW_DIR = 'generative_construction_null'
SPLITS = ('A_train', 'A_val', 'B_confirm')
SOURCE_NAME = 'build_effective_dynamics_kt4_stage1_empirical.py'
B_REQ = ['user_id','bundle_attempt_index','bundle_id','enter_ts','submit_ts','source_code','platform_code','accuracy','total_response_count','total_response_change_count','duration_ms']
B_OPT = ['answered_question_count','total_erase_count','total_undo_erase_count','total_choice_process_count','total_text_enter_count','access_status_aux','payment_coupon_window_aux','access_pay_event_count','access_refund_event_count','access_coupon_event_count']
Q_REQ = ['user_id','bundle_attempt_index','question_id','is_correct']
S_REQ = ['user_id','study_episode_index','item_id','item_type_code','enter_ts','quit_ts','dwell_ms']
S_OPT = ['dwell_ratio','video_length_ms','related_bundle_id','media_event_count','media_elapsed_ms','media_pair_anomaly_count']
CONTRACT = dict(TAU_RESPONSE_DAYS=10., TAU_ACTIVITY_DAYS=10., EVIDENCE_MATURITY_SCALE=20., RESPONSE_DURATION_HALF_SAT_MIN=3., EXPLANATION_HALF_SAT_MIN=2.5, LECTURE_HALF_SAT_MIN=4., IDLE_HALF_SAT_DAYS=1., MAX_SUPPORT_EPISODE_ACTIVE=1., MIN_STATE_BIN_COUNT=50, MIN_DRIFT_BIN_COUNT=30, MIN_CELL_USERS=5, CONVERGENCE_SHELL_RADIUS=.35)
KEYS = ['user_id','bundle_step_index','bundle_attempt_index']
STATE_BASE = ['M_response_prebalanced','response_evidence_mass','unresolved_response_demand_mass','activity_alignment_order_Psi','activity_active_mass','activity_aligned_mass','activity_off_target_mass','activity_neutral_mass','activity_idle_mass']
IDENTITY_COLS = [f'{x}_{phase}' for phase in ('pre','resp','support','post') for x in STATE_BASE] + ['response_active_mass_interval','response_alignment_to_pre_demand','support_active_total_interval','support_alignment_to_pre_demand_or_current_bundle','idle_mass_interval']


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.bool_,bool)): return bool(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, Path): return str(x)
    return x


def canonical(x): return json.dumps(clean(x),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()


def atomic_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
    try:
        t.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)
    finally:
        if t.exists(): t.unlink()


def atomic_npz(path,**x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with t.open('wb') as f: np.savez_compressed(f,**x)
        os.replace(t,p)
    finally:
        if t.exists():t.unlink()


def atomic_pickle(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with gzip.open(t,'wb',compresslevel=1) as f:pickle.dump(x,f,protocol=5)
        os.replace(t,p)
    finally:
        if t.exists():t.unlink()


def read_pickle(p):

    with gzip.open(p,'rb') as f:return pickle.load(f)


@contextlib.contextmanager
def lock(p):
    import fcntl
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f'Another process owns {p}')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def find_table(base):
    base=Path(base)
    if base.is_file():return base
    for ext in ('.parquet','.csv.gz','.csv'):
        p=Path(str(base)+ext)
        if p.is_file():return p
    raise FileNotFoundError(f'No table: {base} [parquet/csv.gz/csv]')


def read_table(base,columns=None):
    p=find_table(base)
    if p.suffix=='.parquet':return pd.read_parquet(p,columns=columns)
    return pd.read_csv(p,usecols=columns,low_memory=False)


def schema(p):
    p=find_table(p)
    if p.suffix=='.parquet':
        import pyarrow.parquet as pq
        return list(pq.read_schema(p).names)
    return list(pd.read_csv(p,nrows=0).columns)


def chunk_id(p):
    m=re.search(r'chunk_(\d+)',Path(p).name)
    if not m:raise ValueError(f'Not a chunk filename: {p}')
    return int(m.group(1))


def chunks(root,prefix):
    out={}
    for ext in ('parquet','csv.gz','csv'):
        for p in sorted(Path(root).glob(f'{prefix}_chunk_*.{ext}')):
            ci=chunk_id(p)
            if ci in out:raise RuntimeError(f'Ambiguous duplicate chunk {ci}: {out[ci]}, {p}')
            out[ci]=p.resolve()
    return dict(sorted(out.items()))


def module_from(p,name):
    spec=importlib.util.spec_from_file_location(name,str(p));mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def source_paths(repo_root=None):
    if repo_root:
        roots=[Path(repo_root).resolve()]
    else:
        roots=list(Path(__file__).resolve().parents)+[Path.cwd().resolve()]
    for root in roots:
        main=root/'s1_empirical_dynamics/main'/SOURCE_NAME
        if main.is_file():
            sem=root/'s1_empirical_dynamics/robustness/semantic_specificity/semantic_specificity_common.py'
            old=root/'s1_empirical_dynamics/robustness/construction_null/run_construction_matched_null.py'
            if not sem.is_file() or not old.is_file():raise FileNotFoundError('Incomplete original EdDynamics repository')
            return dict(repo=str(root),stage1=str(main),semantic=str(sem),original_null=str(old))
    raise FileNotFoundError('Original EdDynamics source repository not found; place this module in the repository or pass --repo-root. Existing macro panels cannot replace source code.')


def code_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def resolve_data(output_root,explicit=None):

    output=Path(output_root).resolve();mfile=output/'stage1/metadata/stage1_empirical_v3_manifest.json'
    m=read_json(mfile)
    pp=m.get('input_preprocess_manifest',{})
    c=[]
    def add(p,why):
        if p:
            p=Path(p).expanduser()
            if not p.is_absolute():p=mfile.parent/p
            p=p.resolve()
            if p not in [a for a,_ in c]:c.append((p,why))
    if explicit:add(explicit,'explicit override')
    add(pp.get('output_root'),'Stage-1 input_preprocess_manifest.output_root')
    add(os.environ.get('EDNET_KT4_DATA_ROOT'),'EDNET_KT4_DATA_ROOT')
    for p in (output,output/'data_297915',output.parent/'data_297915',output.parent/'preprocessed_data',Path('/data/datasets/KT4/data_297915')):add(p,'source layout/default')

    for path in output.glob('*/metadata/preprocess_manifest.json'):add(path.parent.parent,'relocated preprocessing manifest')
    audit=[];valid=[]
    for root,why in c:
        k=root/'kt4';ct=root/'contents'
        b=chunks(k/'bundle_attempts','bundle_attempts');q=chunks(k/'question_attempts','question_attempts');s=chunks(k/'study_episodes','study_episodes')
        content_ok=all(any((ct/(name+ext)).is_file() for ext in ('.parquet','.csv.gz','.csv')) for name in ('questions_clean','bundles_clean','lectures_clean','explanations_clean'))
        ok=bool(b and q and content_ok)
        audit.append(dict(root=str(root),source=why,bundle_chunks=len(b),question_chunks=len(q),study_chunks=len(s),contents_complete=content_ok,valid=ok))
        if ok:valid.append((root,b,q,s))
        if ok and (explicit or why=='Stage-1 input_preprocess_manifest.output_root'):break
    if not valid:
        raise FileNotFoundError('Cannot resolve source-schema preprocessing tables. Checked:\n'+json.dumps(audit,indent=2)+'\nNo macro-panel substitution is allowed.')
    root,b,q,s=valid[0]
    if explicit and root!=Path(explicit).resolve():raise FileNotFoundError('Explicit data-root is not valid; no silent fallback.')
    for name,paths,required in [('bundle',b,B_REQ),('question',q,Q_REQ),('study',s,S_REQ)]:
        for ci,p in paths.items():
            missing=set(required)-set(schema(p))
            if missing:raise RuntimeError(f'{p} lacks {sorted(missing)}')
    if set(q)-set(b):raise RuntimeError('Question chunks without matching bundle chunks require reconciliation with original source.')
    return dict(data_root=str(root),kt4_root=str(root/'kt4'),contents_root=str(root/'contents'),metadata_root=str(root/'metadata'),bundle={str(i):str(p) for i,p in b.items()},question={str(i):str(p) for i,p in q.items()},study={str(i):str(p) for i,p in s.items()},discovery=audit)


def source_module(protocol):
    paths=protocol['sources'];data=protocol['data'];out=protocol['output_root']
    os.environ.update(EDNET_KT4_DATA_ROOT=data['data_root'],EDNET_KT4_PREPROCESSED_ROOT=data['kt4_root'],EDNET_CONTENTS_ROOT=data['contents_root'],EDNET_KT4_METADATA_ROOT=data['metadata_root'],EDNET_KT4_OUTPUT_ROOT=out)
    m=module_from(paths['stage1'],'cn_original_stage1')
    for k,v in CONTRACT.items():
        if abs(float(getattr(m,k))-v)>1e-12:raise RuntimeError(f'Original parameter {k} changed: {getattr(m,k)} != {v}')
    if not np.array_equal(m.GRID_BINS_SIGNED,np.linspace(-1,1,41)):raise RuntimeError('Original grid differs from 40x40.')
    return m


def semantic_module(protocol):return module_from(protocol['sources']['semantic'],'cn_original_semantic')


def frozen_inputs(m):
    users={}
    for s in SPLITS:
        ids=read_table(m.SPLIT_ROOT/f'{s}_users',['user_id']).user_id
        if ids.isna().any() or ids.duplicated().any():raise RuntimeError(f'Invalid frozen split {s}')
        users[s]=set(ids.astype('int64'))
    if any(users[a]&users[b] for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]):raise RuntimeError('Frozen user splits overlap.')
    content=m.ContentsBuilder().build()
    qtab=read_table(m.META_OUT_ROOT/'A_train_question_itemEB_correctness_priors')
    ttab=read_table(m.META_OUT_ROOT/'A_train_tag_correctness_priors')
    if qtab.question_id.astype(str).duplicated().any() or ttab.tag.duplicated().any():raise RuntimeError('Duplicate prior keys')
    qp=dict(zip(qtab.question_id.astype(str),qtab.prior_correct_itemEB_Atrain.astype(float)))
    tp=dict(zip(ttab.tag.astype(int),ttab.prior_correct_Atrain_shrunk.astype(float)))
    prior_array=np.asarray([tp[content.idx_to_tag[i]] for i in range(len(content.idx_to_tag))])
    glob=ttab.global_prior_correct_Atrain.dropna().astype(float).unique()
    if len(glob)!=1:raise RuntimeError('Inconsistent global prior')
    priors=m.TagPriorEstimate(prior_array,ttab,float(glob[0]),qp,qtab)
    if np.any(~np.isfinite(prior_array)) or any(not 0<p<1 for p in qp.values()):raise RuntimeError('Invalid item/tag priors')
    return users,content,priors


def probabilities(q,content,builder):
    return np.asarray([builder._question_prior(str(k),content.question_tag_idx.get(str(k))) for k in q.question_id],float)


def stable_rng(seed,split,mode,replicate,uid,stream=0):
    words=[int(seed),SPLITS.index(split)+1,{'null_a':1,'null_b':2}.get(mode,3),int(replicate)+1,int(uid)&0xffffffff,(int(uid)>>32)&0xffffffff,int(stream)]
    return np.random.default_rng(np.random.SeedSequence(words))


def calibrated_intercepts(p,effects):
    p=np.asarray(p,float);effects=np.asarray(effects,float)
    if np.any(~np.isfinite(p)) or np.any((p<=0)|(p>=1)) or not len(effects):raise ValueError('Invalid calibration probabilities/effects')
    ans=np.empty_like(p)
    for j in range(0,len(p),256):
        target=p[j:j+256];lo=np.full(len(target),-40.);hi=np.full(len(target),40.)
        for _ in range(64):
            mid=(lo+hi)/2;low=expit(mid[:,None]+effects[None,:]).mean(axis=1)<target
            lo=np.where(low,mid,lo);hi=np.where(low,hi,mid)
        ans[j:j+len(target)]=(lo+hi)/2
    return ans


def simulate_questions(template,p,seed,split,mode,rid,uid,calibration=None):
    q=template.copy();y=pd.to_numeric(q.is_correct,errors='raise').to_numpy(dtype=float,copy=True)
    ok=np.isfinite(y);g=stable_rng(seed,split,mode,rid,uid,0);pr=p.copy()
    if mode=='null_b':
        if calibration is None:raise ValueError('Null B needs frozen training calibration')
        effects=calibration['effects'];rg=stable_rng(seed,split,mode,rid,uid,1)
        b=effects[rg.integers(len(effects))]
        idx=np.searchsorted(calibration['p'],p)
        if np.any(idx>=len(calibration['p'])) or not np.array_equal(calibration['p'][idx],p):raise RuntimeError('Uncalibrated prior; cannot use global-alpha fallback.')
        pr=expit(calibration['alpha'][idx]+b)
    y[ok]=g.binomial(1,pr[ok]);q['is_correct']=y
    audit=np.asarray([ok.sum(),y[ok].sum(),pr[ok].sum(),np.sum(pr[ok]*(1-pr[ok]))],float)
    return q,audit


class FieldAccumulator:

    def __init__(self,m,spec):
        self.m=m;self.spec=spec;self.shape=(len(spec.xbins)-1,len(spec.ybins)-1)
        self.a={k:np.zeros(self.shape,float) for k in ['occ','occn','uc','n','w','w2','sx','sy','sxx','syy','sxy']}
        self.rows=self.users=self.state_rows=self.drift_rows=0
    def add(self,frame):
        if frame.empty:return
        x,y,dx,dy,sv,dv=self.m.spec_arrays(frame,self.spec);weights=self.m.user_balanced_weights(frame)
        ix=self.m.digitize_closed_right(x,self.spec.xbins);iy=self.m.digitize_closed_right(y,self.spec.ybins);nx,ny=self.shape
        inside=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny);sv&=inside;dv&=inside
        c=ix*ny+iy;size=nx*ny
        def bc(mask,values=None):return np.bincount(c[mask],weights=None if values is None else values[mask],minlength=size).reshape(self.shape)
        self.a['occ']+=bc(sv,weights);self.a['occn']+=bc(sv)
        uid=frame.user_id.to_numpy('int64');pairs=np.rec.fromarrays([c[sv],uid[sv]],names='cell,user')
        if len(pairs):self.a['uc']+=np.bincount(np.unique(pairs)['cell'],minlength=size).reshape(self.shape)
        for k,v in [('n',None),('w',weights),('w2',weights**2),('sx',weights*dx),('sy',weights*dy),('sxx',weights*dx*dx),('syy',weights*dy*dy),('sxy',weights*dx*dy)]:self.a[k]+=bc(dv,v)
        self.rows+=len(frame);self.users+=frame.user_id.nunique();self.state_rows+=int(sv.sum());self.drift_rows+=int(dv.sum())
    def finish(self):
        a=self.a;den=np.maximum(a['w'],1e-12);u=a['sx']/den;v=a['sy']/den
        return dict(u=u,v=v,occ=a['occ'],occn=a['occn'],user_count=a['uc'],n=a['n'],w=a['w'],state_mask=(a['occn']>=self.m.MIN_STATE_BIN_COUNT)&(a['uc']>=self.m.MIN_CELL_USERS),drift_mask=a['n']>=self.m.MIN_DRIFT_BIN_COUNT,rows=np.asarray(self.rows),users=np.asarray(self.users),valid_state_rows=np.asarray(self.state_rows),valid_drift_rows=np.asarray(self.drift_rows))


def add_fields(m,sem,accs,rows):
    if not rows:return
    fin=m.finalize_panel_chunk(pd.DataFrame(rows))


    primary=m.downcast_frame(fin)
    phi,_=sem.add_nonsemantic_coordinate(fin)
    phi=m.downcast_frame(phi)
    accs['psi'].add(primary);accs['phi'].add(phi)


def new_accumulators(m,sem):return {'psi':FieldAccumulator(m,m.coordinate_specs()[0]),'phi':FieldAccumulator(m,sem.control_spec(m))}
def pack_fields(accs):return {f'{c}_{k}':v for c,a in accs.items() for k,v in a.finish().items()}
def unpack_fields(z,c):return {k[len(c)+1:]:np.asarray(z[k]) for k in z if k.startswith(c+'_')}


def replay_identity(rebuilt,saved):
    if rebuilt.duplicated(KEYS).any() or saved.duplicated(KEYS).any():raise RuntimeError('Duplicate replay key')
    a=rebuilt.sort_values(KEYS).reset_index(drop=True);b=saved.sort_values(KEYS).reset_index(drop=True)
    if len(a)!=len(b) or not np.array_equal(a[KEYS].to_numpy(),b[KEYS].to_numpy()):raise RuntimeError('Observed raw replay keys/rows differ from archived raw panel')
    maximum=0.
    for k in IDENTITY_COLS:
        if k not in b or k not in a:raise RuntimeError(f'Identity column absent: {k}')
        av=pd.to_numeric(a[k]).to_numpy(float);bv=pd.to_numeric(b[k]).to_numpy(float)
        if not np.array_equal(np.isfinite(av),np.isfinite(bv)):raise RuntimeError(f'Identity missingness mismatch: {k}')
        ok=np.isfinite(av)
        if ok.any():
            maximum=max(maximum,float(np.max(np.abs(av[ok]-bv[ok]))))
            if not np.allclose(av[ok],bv[ok],atol=2e-9,rtol=2e-10):raise RuntimeError(f'Observed replay mismatch: {k}, max error {np.max(np.abs(av[ok]-bv[ok]))}')
    return dict(rows=len(a),max_abs_difference=maximum,checked_columns=len(IDENTITY_COLS),passed=True)


def archived_field_base(output,split,coord):
    suffix='_output_only' if split=='B_confirm' else ''
    root=Path(output)/'stage1' if coord=='psi' else Path(output)/'semantic_specificity_control/coordinate/stage1'
    name='MR_PsiA' if coord=='psi' else 'MR_PhiAI'
    return root/'dynamics/coordinate_analysis'/name/f'{split}_publication_field_grid{suffix}'


def audit_field(field,base,tolerance=1e-10):
    t=read_table(base).sort_values(['x_bin','y_bin']);shape=field['u'].shape
    if len(t)!=np.prod(shape):raise RuntimeError(f'Archived grid size mismatch: {base}')
    expected=np.column_stack([np.repeat(np.arange(shape[0]),shape[1]),np.tile(np.arange(shape[1]),shape[0])])
    if not np.array_equal(t[['x_bin','y_bin']].to_numpy(),expected):raise RuntimeError(f'Archived grid keys differ: {base}')
    target={'u':t.drift_M.to_numpy(float).reshape(shape),'v':t.drift_Psi.to_numpy(float).reshape(shape),'drift_mask':t.drift_supported.to_numpy(bool).reshape(shape),'n':t.drift_count.to_numpy(float).reshape(shape),
            'state_mask':t.state_supported.to_numpy(bool).reshape(shape),'occn':t.occupancy_count.to_numpy(float).reshape(shape),'user_count':t.user_count.to_numpy(float).reshape(shape),
            'w':t.drift_user_balanced_weight.to_numpy(float).reshape(shape),'occ':t.occupancy_weighted.to_numpy(float).reshape(shape)}
    for k in ('drift_mask','n','state_mask','occn','user_count'):
        if not np.array_equal(field[k],target[k]):raise RuntimeError(f'Archived {k} mismatch: {base}')
    errors={k:float(np.max(np.abs(field[k][field['drift_mask']]-target[k][field['drift_mask']]))) if field['drift_mask'].any() else 0. for k in ('u','v')}
    p=field['occ']/max(float(field['occ'].sum()),1e-12)
    errors['occupancy']=float(np.max(np.abs(p-t.occupancy_probability.to_numpy(float).reshape(shape))))
    if max(errors.values())>tolerance:raise RuntimeError(f'Archived field numerical mismatch: {base}: {errors}')


    for k in ('w','occ'):
        errors[k]=float(np.max(np.abs(field[k]-target[k])))
        if not np.allclose(field[k],target[k],atol=tolerance,rtol=1e-12):raise RuntimeError(f'Archived {k} mismatch: {base}, max error {errors[k]}')
    return dict(path=str(find_table(base)),max_errors=errors,checked_exact=['drift_mask','n','state_mask','occn','user_count'],raw_weight_tolerance=dict(atol=tolerance,rtol=1e-12),passed=True)


def _require_metadata(value,expected,label):
    for key,wanted in expected.items():
        if value.get(key)!=wanted:raise RuntimeError(f'{label} metadata mismatch: {key}')


def _verified_file(path,expected,label):
    path=Path(path)
    if not path.is_file() or not isinstance(expected,str) or sha256(path)!=expected:raise RuntimeError(f'{label} missing or checksum mismatch: {path}')
    return path


def validate_field_arrays(arrays,p,rows,users,label):

    for coord in p['coordinates']:
        f=unpack_fields(arrays,coord)
        for key in ('u','v','occ','occn','user_count','n','w','state_mask','drift_mask'):
            if key not in f or f[key].shape!=(40,40):raise RuntimeError(f'{label}: invalid {coord}_{key} shape')
            if not np.isfinite(f[key]).all():raise RuntimeError(f'{label}: nonfinite {coord}_{key}')
        for key in ('occ','occn','user_count','n','w'):
            if np.any(f[key]<0):raise RuntimeError(f'{label}: negative {coord}_{key}')
        for key in ('occn','user_count','n'):
            if np.any(f[key]!=np.floor(f[key])):raise RuntimeError(f'{label}: noninteger {coord}_{key}')
        for key,wanted in [('rows',rows),('users',users)]:
            if key not in f or f[key].shape!=() or int(f[key])!=wanted or float(f[key])!=wanted:raise RuntimeError(f'{label}: {coord}_{key} differs from audited cohort')
        for key,count in [('valid_state_rows',f['occn'].sum()),('valid_drift_rows',f['n'].sum())]:
            if key not in f or f[key].shape!=() or float(f[key])!=count or not 0<=count<=rows:raise RuntimeError(f'{label}: inconsistent {coord}_{key}')
        sm=(f['occn']>=CONTRACT['MIN_STATE_BIN_COUNT'])&(f['user_count']>=CONTRACT['MIN_CELL_USERS'])
        dm=f['n']>=CONTRACT['MIN_DRIFT_BIN_COUNT']
        if not np.array_equal(f['state_mask'],sm) or not np.array_equal(f['drift_mask'],dm):raise RuntimeError(f'{label}: stored {coord} support mask differs from original thresholds')


def validate_prepared_split(work,p,split):

    work=Path(work)
    if split not in ('A_val','B_confirm') or (p['smoke'] and split!='A_val'):raise RuntimeError('Invalid prepared cohort')
    done=read_json(work/'cache'/split/'complete.json')
    _require_metadata(done,dict(protocol_id=p['protocol_id'],split=split,smoke=p['smoke']), 'Prepared cache')
    blocks=done.get('blocks',[])
    if not blocks or len({b['path'] for b in blocks})!=len(blocks):raise RuntimeError('Empty or duplicate cache block manifest')
    if any(int(b['rows'])<=0 or int(b['users'])<=0 for b in blocks):raise RuntimeError('Empty cache block')
    if sum(b['rows'] for b in blocks)!=done['rows'] or sum(b['users'] for b in blocks)!=done['users']:raise RuntimeError('Cache block cohort totals differ')
    expected_prefix=(work/'cache'/split).resolve()
    for block in blocks:
        path=(work/block['path']).resolve()
        if expected_prefix not in path.parents:raise RuntimeError('Cache block is outside its cohort directory')
        _verified_file(path,block['sha256'],'Cache block')
    observed=work/'observed'/f'{split}.npz';identity=work/'observed'/f'{split}_identity.json'
    if (work/done['observed_file']).resolve()!=observed.resolve():raise RuntimeError('Observed file belongs to another cohort')
    _verified_file(observed,done['observed_sha256'],'Observed field')
    _verified_file(identity,done.get('identity_sha256'),'Replay identity')
    proof=read_json(identity)
    _require_metadata(proof,dict(protocol_id=p['protocol_id'],split=split,smoke=p['smoke'],passed=True),'Replay identity')
    replay=proof.get('source_replay',[])
    if not replay or len({r['chunk'] for r in replay})!=len(replay) or sum(r['rows'] for r in replay)!=done['rows']:raise RuntimeError('Replay identity does not cover cache rows')
    if any(r.get('passed') is not True or r.get('checked_columns')!=len(IDENTITY_COLS) or not np.isfinite(r.get('max_abs_difference',np.nan)) or not r.get('source_sha256') for r in replay):raise RuntimeError('Unsuccessful source replay identity')
    if not p['smoke']:
        audits=proof.get('publication_field',{})
        for coord in p['coordinates']:
            if audits.get(coord,{}).get('passed') is not True:raise RuntimeError(f'Missing archived {coord} field audit')
    with np.load(observed,allow_pickle=False) as z:validate_field_arrays(z,p,done['rows'],done['users'],'Observed field')
    return done


def validate_calibration(work,p):

    work=Path(work);meta=read_json(work/'calibration_complete.json');f=work/'null_b_calibration.npz'
    _require_metadata(meta,dict(protocol_id=p['protocol_id'],fit_split='A_train',smoke=p['smoke']),'Null B calibration')
    _verified_file(f,meta.get('sha256'),'Null B calibration')
    with np.load(f,allow_pickle=False) as z:
        if not {'p','alpha','effects'}<=set(z.files):raise RuntimeError('Incomplete Null B calibration arrays')
        prob=np.asarray(z['p']);alpha=np.asarray(z['alpha']);effects=np.asarray(z['effects'])
        if prob.ndim!=1 or len(prob)==0 or alpha.shape!=prob.shape or effects.shape!=(256,):raise RuntimeError('Invalid Null B calibration shapes')
        if not all(np.isfinite(x).all() for x in (prob,alpha,effects)) or np.any((prob<=0)|(prob>=1)) or np.any(np.diff(prob)<=0):raise RuntimeError('Invalid Null B calibration values')
        error=max(float(np.max(np.abs(expit(alpha[i:i+256,None]+effects[None,:]).mean(axis=1)-prob[i:i+256]))) for i in range(0,len(prob),256))
        if error>1e-11 or not np.isfinite(meta.get('max_item_marginal_error',np.nan)) or meta['max_item_marginal_error']>1e-11:raise RuntimeError('Null B calibration marginal error exceeds original tolerance')
    if meta.get('effect_points')!=256 or meta.get('n_train_users',0)<(2 if p['smoke'] else 100):raise RuntimeError('Incomplete Null B training calibration')
    return meta


def validate_replicate(work,p,split,mode,rid,done,calibration=None):

    work=Path(work);f=work/'simulations'/split/mode/f'replicate_{rid:04d}.npz';meta=read_json(f.with_suffix('.json'))
    _require_metadata(meta,dict(protocol_id=p['protocol_id'],split=split,mode=mode,replicate=rid,smoke=p['smoke'],seed=p['seed'],rows=done['rows'],users=done['users']), 'Replicate')
    if meta.get('cache_manifest_sha256')!=sha256(work/'cache'/split/'complete.json'):raise RuntimeError('Replicate cache lineage mismatch')
    if mode=='null_b' and (calibration is None or meta.get('null_b_calibration_sha256')!=calibration['sha256']):raise RuntimeError('Replicate Null B calibration lineage mismatch')
    _verified_file(f,meta.get('sha256'),'Replicate')
    with np.load(f,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    validate_field_arrays(arrays,p,done['rows'],done['users'],'Replicate')
    return arrays,meta


def validate_realizations(work,p,split,mode):

    work=Path(work)
    if mode not in p['modes']:raise RuntimeError('Unknown null mode')
    done=validate_prepared_split(work,p,split)
    calibration=validate_calibration(work,p) if mode=='null_b' else None
    folder=work/'simulations'/split/mode
    for extension in ('npz','json'):
        expected={f'replicate_{r:04d}.{extension}' for r in range(p['replicates'])}
        found={f.name for f in folder.glob(f'replicate_*.{extension}')}
        if found!=expected:raise RuntimeError(f'Wrong replicate {extension} set; missing={sorted(expected-found)[:10]}, extra={sorted(found-expected)[:10]}')
    with np.load(work/done['observed_file'],allow_pickle=False) as z:arrays=[{k:z[k] for k in z.files}]
    metas=[]
    for rid in range(p['replicates']):
        a,meta=validate_replicate(work,p,split,mode,rid,done,calibration);arrays.append(a);metas.append(meta)
    return arrays,metas


def load_protocol(work):
    p=read_json(Path(work)/'protocol.json')
    if p.get('protocol_id')!=digest(protocol_payload(p)):raise RuntimeError('Frozen protocol was modified.')
    if not p['smoke'] and p['replicates']!=999:raise RuntimeError('Formal replicate contract must remain 999.')
    if p['code_hashes']!=code_hashes():raise RuntimeError('Supplement code differs from frozen protocol; do not mix versions.')
    for name,h in p['source_hashes'].items():
        if sha256(p['sources'][name])!=h:raise RuntimeError(f'Original source changed: {name}')
    for f,h in p['frozen_files'].items():
        if sha256(f)!=h:raise RuntimeError(f'Frozen upstream artifact changed: {f}')
    return p


def protocol_payload(p):return {k:v for k,v in p.items() if k!='protocol_id'}
def require_confirm(work,p):
    work=Path(work);f=work/'A_val_release.json'
    if not f.exists():raise RuntimeError('B_confirm is blocked until A_val summaries complete under the same protocol.')
    release=read_json(f)
    _require_metadata(release,dict(protocol_id=p['protocol_id'],schema_version=1),'A_val release')
    for key,filename in [('summaries','summary.json'),('manifests','summary_artifacts.json')]:
        expected={str(work/'summary/A_val'/mode/filename) for mode in p['modes']}
        refs=release.get(key,{})
        if set(refs)!=expected:raise RuntimeError(f'Incomplete A_val release {key}')
        for path,h in refs.items():_verified_file(path,h,'A_val released evidence')
