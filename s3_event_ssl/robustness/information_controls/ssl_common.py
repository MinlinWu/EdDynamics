
from __future__ import annotations
import contextlib, dataclasses, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VERSION = 'ssl-information-controls-v1'
RUN_NAME = 'ssl_information_controls'
DEFAULT_ROOT = 'outputs_KT4'
ALL_SEEDS = (42, 2026, 666, 606, 37, 4669)
NEW_SEEDS = (42, 2026, 666)
SPLITS = ('A_train', 'A_val', 'B_confirm')
EPS = 1e-12
class AuditError(RuntimeError): pass

def require(ok, message):
    if not ok: raise AuditError(message)

def jsonable(x):
    if dataclasses.is_dataclass(x): return jsonable(dataclasses.asdict(x))
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, Path): return str(x)
    return x

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def objhash(x):
    return hashlib.sha256(json.dumps(jsonable(x),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def save_json(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(jsonable(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,p)

def write_csv(rows,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(p,index=False)

def log(s): print(time.strftime('[%Y-%m-%d %H:%M:%S] ')+str(s),flush=True)

def file_record(p):
    p=Path(p).resolve();s=p.stat()
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)}

def run_name(smoke=False):
    return RUN_NAME + ('_smoke' if smoke else '')


def fingerprint_sources(paths, bindings):

    paths=sorted({Path(p).resolve() for p in paths})
    records=[]
    for i,p in enumerate(paths):
        require(p.is_file(),f'Resolved source missing: {p}')
        records.append(file_record(p))
        if i%20==0:log(f'Fingerprint current source artifacts {i+1}/{len(paths)}')
    identity={'files':[{k:r[k] for k in ('path','bytes','sha256')} for r in records],
              'resolved_bindings':bindings}
    return records,objhash(identity)


def verify_record(r, full=False):
    p=Path(r['path']);require(p.is_file(),f'Input disappeared: {p}')
    s=p.stat();require(s.st_size==r['bytes'],f'Input size changed: {p}')
    if full or s.st_mtime_ns!=r['mtime_ns']:
        require(sha(p)==r['sha256'],f'Input changed: {p}')

def source_path(name):
    require(name in load_json(ROOT/'source_contract.json')['files'],'Unknown original dependency: '+name)
    return ROOT.parents[1]/'main'/name

def code_digest():
    paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.sh'))+list(ROOT.glob('*.sbatch'))+[ROOT/'protocol.json',ROOT/'source_contract.json']
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}
    files.update({'original/'+name:sha(source_path(name)) for name in load_json(ROOT/'source_contract.json')['files']})
    return objhash(files)

def module(name):
    path=source_path(name)
    expected=load_json(ROOT/'source_contract.json')['files'][name]
    require(sha(path)==expected,f'Original dependency changed: {name}')
    key='_ssl_original_'+name.replace('.','_')
    if key in sys.modules: return sys.modules[key]
    spec=importlib.util.spec_from_file_location(key,path);m=importlib.util.module_from_spec(spec)
    sys.modules[key]=m
    old=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try: spec.loader.exec_module(m)
    finally: sys.dont_write_bytecode=old
    return m

def table_path(base, required=True):
    p=Path(base)
    paths=[p] if p.suffix in ('.parquet','.csv','.gz') else [Path(str(p)+e) for e in ('.parquet','.csv.gz','.csv')]
    for q in paths:
        if q.is_file(): return q
    if required: raise AuditError('Table not found: '+str(p))
    return None

def iter_table(path, columns=None, batch=200000):
    p=table_path(path)
    if p.suffix=='.parquet':
        try: import pyarrow.parquet as pq
        except ImportError as e: raise AuditError('Parquet inputs require pyarrow in the original environment.') from e
        for b in pq.ParquetFile(p).iter_batches(batch_size=batch,columns=columns,use_threads=False): yield b.to_pandas()
    else:
        yield from pd.read_csv(p,usecols=columns,chunksize=batch,float_precision='round_trip')

def read_table(p,columns=None): return pd.concat(list(iter_table(p,columns)),ignore_index=True)

@contextlib.contextmanager
def run_lock(root):
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise AuditError('Another process is writing this run directory.') from e
        f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush()
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

def stage(root,name,contract,inputs,callback):

    dest=Path(root)/name;receipt=dest/'_SUCCESS.json'
    expected={'contract':contract,'inputs':inputs}
    if receipt.is_file():
        r=load_json(receipt);require(r['identity']==objhash(expected),f'Stage identity changed: {name}')
        for rel,h in r['outputs'].items():
            p=dest/rel;require(p.is_file() and sha(p)==h,f'Committed output changed: {p}')
        log('Verified reuse: '+name);return dest
    require(not dest.exists(),f'Uncommitted final directory: {dest}; preserve it and inspect.')
    work=dest.with_name(dest.name+'.partial');work.mkdir(parents=True,exist_ok=True)
    state=work/'_STAGE_IDENTITY.json'
    if state.exists(): require(load_json(state)==jsonable(expected),f'Partial stage contract changed: {name}')
    else: save_json(expected,state)
    log('Start '+name);callback(work)
    outputs={str(p.relative_to(work)):sha(p) for p in sorted(work.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')}
    save_json({'identity':objhash(expected),'contract':contract,'inputs':inputs,'outputs':outputs,
               'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S'),'technical_completion_only':True},work/'_SUCCESS.json')
    dest.parent.mkdir(parents=True,exist_ok=True);os.replace(work,dest);log('Completed '+name);return dest

def verify_committed(path, contract=None):

    path=Path(path);rp=path/'_SUCCESS.json'
    require(rp.is_file(),f'Missing committed stage: {path}')
    r=load_json(rp)
    if contract is not None:require(r['contract']==contract,f'Finalization contract mismatch: {path}')
    for rel,h in r['outputs'].items():
        target=path/rel
        require(target.is_file() and sha(target)==h,f'Committed artifact changed: {target}')
    return sha(rp)

def receipt(path): return sha(Path(path)/'_SUCCESS.json')

def tensor_digest(state):

    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def assert_close(a,b,label,atol=2e-6,rtol=2e-6):
    a=np.asarray(a);b=np.asarray(b)
    require(a.shape==b.shape,f'{label}: shape mismatch')
    require(np.isfinite(a).all() and np.isfinite(b).all(),f'{label}: non-finite')
    err=float(np.max(np.abs(a-b),initial=0))
    require(np.allclose(a,b,atol=atol,rtol=rtol),f'{label}: maximum error {err:g}')
    return err

def torch_load(path):
    import torch

    try: return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError: return torch.load(path,map_location='cpu')

def torch_save(x,path):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    torch.save(x,t);os.replace(t,p)
