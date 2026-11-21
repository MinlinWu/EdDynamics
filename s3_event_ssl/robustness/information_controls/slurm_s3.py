#!/usr/bin/env python3

from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from types import SimpleNamespace

JOB=Path(__file__).resolve().parent
SEEDS=(2026,666,606,37,4669)
SCRIPTS=('train_gpu.py','eval_cpu.py','finalize_six.py','slurm_s3.py','slurm_s3.sbatch')

def need(ok,message):
 if not ok:raise RuntimeError(message)

def now():return datetime.datetime.now().astimezone().isoformat()

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
 path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with temp.open('w',encoding='utf-8') as stream:
  json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 os.replace(temp,path)

def exclusive_json(path,value):
 with Path(path).open('x',encoding='utf-8') as stream:
  json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def canonical_uuid(value):
 if isinstance(value,bytes):
  try:value=value.decode('ascii')
  except UnicodeDecodeError:value=str(uuid.UUID(bytes=value)) if len(value)==16 else ''
 text=str(value).strip()
 if text.lower().startswith('gpu-'):text=text[4:]
 need(re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',text) is not None,
      f'Cannot establish a physical GPU UUID: {value!r}')
 return 'GPU-'+text.lower()

def parse_gpu_ids(text):

 need(bool(text and text.strip()),'No authoritative Slurm GPU list')
 ids=[]
 for raw in text.split(','):
  token=raw.strip()
  if token.isdigit():ids.append(str(int(token)))
  elif re.fullmatch(r'\d+-\d+',token):
   lo,hi=map(int,token.split('-'));need(0<=lo<=hi and hi-lo<256,'Invalid Slurm GPU range')
   ids.extend(str(i) for i in range(lo,hi+1))
  elif token.lower().startswith('gpu-'):ids.append(canonical_uuid(token))
  else:raise RuntimeError('Unsupported authoritative GPU ID; refusing to guess: '+token)
 need(len(ids)==len(set(ids)),'Duplicated Slurm GPU IDs')
 return ids

def inventory():
 result=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id','--format=csv,noheader,nounits'],text=True)
 rows=[]
 for line in result.splitlines():
  parts=[x.strip() for x in line.split(',')]
  need(len(parts)==4 and parts[0].isdigit(),'Unexpected nvidia-smi inventory')
  rows.append({'index':int(parts[0]),'uuid':canonical_uuid(parts[1]),'name':parts[2],'pci_bus_id':parts[3]})
 need(rows and len({r['uuid'] for r in rows})==len(rows),'No unique physical GPU inventory')
 return rows

def gres_devices():

 config=Path('/etc/slurm/gres.conf');minor_uuid={}
 for info in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
  fields=dict(line.split(':',1) for line in info.read_text().splitlines() if ':' in line)
  minor=fields.get('Device Minor','').strip()
  need(minor.isdigit(),'Cannot identify NVIDIA device minor: '+str(info))
  need(minor not in minor_uuid,'Duplicated NVIDIA device minor')
  minor_uuid[minor]=canonical_uuid(fields.get('GPU UUID',''))
 rows=[]
 for raw in config.read_text().splitlines():
  line=raw.split('#',1)[0].strip()
  if not line:continue
  fields=dict(token.split('=',1) for token in line.split() if '=' in token)
  need('NodeName' not in fields,'Node-scoped GRES configuration needs an explicit configuration parser')
  if fields.get('Name')!='gpu':continue
  match=re.fullmatch(r'/dev/nvidia(\d+)',fields.get('File',''))
  need(match is not None,'Only individually listed NVIDIA GRES files are supported')
  minor=str(int(match[1]));need(minor in minor_uuid,'GRES device has no NVIDIA physical UUID')
  rows.append({'gres_id':str(len(rows)),'device_file':fields['File'],'minor':int(minor),'uuid':minor_uuid[minor]})
 need(len(rows)>=5 and len({r['uuid'] for r in rows})==len(rows),'At least five uniquely configured GPU devices are required')
 return {'config_path':str(config),'config_sha256':sha(config),'devices':rows}

def resolve_authority(env,physical,gres):
 bygres={r['gres_id']:r['uuid'] for r in gres['devices']};known={r['uuid'] for r in physical}
 def resolve(value):
  out=[]
  for item in parse_gpu_ids(value):
   resolved=bygres.get(item) if item.isdigit() else item
   need(resolved in known,'Slurm-authorized GPU not unambiguously present in nvidia-smi inventory: '+item)
   out.append(resolved)
  need(len(out)==len(set(out)),'Slurm IDs alias the same physical GPU')
  return out
 field='SLURM_STEP_GPUS' if env.get('SLURM_STEP_GPUS','').strip() else 'SLURM_JOB_GPUS'
 authorized=resolve(env.get(field,''))
 need(len(authorized)==5,f'{field} must authorize exactly five GPUs, not {len(authorized)}')
 if env.get('SLURM_JOB_GPUS','').strip():
  job=resolve(env['SLURM_JOB_GPUS'])
  need(len(job)==5 and set(authorized)<=set(job),'Step GPU allocation disagrees with the five-GPU job allocation')
 return field,authorized

def slurm_context(env):
 job=env.get('SLURM_JOB_ID') or env.get('SLURM_JOBID')
 step=env.get('SLURM_STEP_ID') or env.get('SLURM_STEPID')
 need(job and re.fullmatch(r'\d+',job),'Must run inside an allocated Slurm job')
 need(step and step not in ('batch','extern') and re.fullmatch(r'\d+',step),'Must run in the foreground of an srun compute step')
 nodes=int(env.get('SLURM_JOB_NUM_NODES',env.get('SLURM_NNODES','0')))
 need(nodes==1,'The five paired workers require a single-node allocation')
 tasks=int(env.get('SLURM_STEP_NUM_TASKS',env.get('SLURM_NTASKS','0')))
 need(tasks==1,'Exactly one Slurm controller task is required')
 cpus=int(env.get('SLURM_CPUS_PER_TASK','0'));need(cpus>=40,'At least 40 allocated CPUs are required for five CPU evaluations')
 return {'job_id':job,'step_id':step,'node':env.get('SLURMD_NODENAME',socket.gethostname()),
         'hostname':socket.gethostname(),'node_list':env.get('SLURM_JOB_NODELIST',env.get('SLURM_NODELIST')),
         'cpus_per_task':cpus,'num_nodes':nodes,'step_num_tasks':tasks,
         'SLURM_JOB_GPUS':env.get('SLURM_JOB_GPUS'),'SLURM_STEP_GPUS':env.get('SLURM_STEP_GPUS'),
         'original_CUDA_VISIBLE_DEVICES':env.get('CUDA_VISIBLE_DEVICES')}

def gpu_probe():

 import torch
 need(torch.cuda.is_available(),'CUDA unavailable inside allocated step')
 count=torch.cuda.device_count();need(count==5,f'Torch sees {count} devices, expected the five explicitly authorized GPUs')
 rows=[]
 for i in range(count):
  prop=torch.cuda.get_device_properties(i)
  tensor=torch.ones(8,device=f'cuda:{i}')
  need(float((tensor+1).sum().item())==16.0,'Allocated GPU tensor check failed')
  torch.cuda.synchronize(i)
  rows.append({'logical_device':i,'uuid':canonical_uuid(getattr(prop,'uuid',None)),
               'name':prop.name,'total_memory_bytes':int(prop.total_memory),
               'compute_capability':list(torch.cuda.get_device_capability(i)), 'tensor_check_passed':True})
 need(len({r['uuid'] for r in rows})==5,'Torch maps multiple logical devices to one physical GPU')
 print(json.dumps({'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,
                   'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'devices':rows}),flush=True)

def allocated_devices(python):
 allocation=slurm_context(os.environ);physical=inventory();gres=gres_devices()
 authority,authorized=resolve_authority(os.environ,physical,gres)


 env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(authorized),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1')
 raw=subprocess.check_output([python,'-u','-B',str(Path(__file__).resolve()),'--gpu-probe'],env=env,text=True,timeout=120)
 actual=json.loads(raw);uuids=[r['uuid'] for r in actual['devices']]
 need(len(uuids)==5 and len(set(uuids))==5 and set(uuids)==set(authorized),'Torch-visible physical UUIDs differ from Slurm authorization')
 allocation.update(authority_field=authority,authorized_uuids=authorized,gres_device_mapping=gres,
                   effective_CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],torch_probe=actual,
                   authorization_policy='Slurm global GRES IDs -> gres.conf device files -> NVIDIA device minors and UUIDs; verified by Torch properties and tensor execution')
 return allocation

def load_context(args):
 code=args.code_root.resolve();history=args.history_root.resolve()
 need(code==JOB,'The controller and scientific modules must share the declared code directory')
 sys.path.insert(0,str(code));sys.dont_write_bytecode=True
 import ssl_common as common
 output=history/common.RUN_NAME
 need(output.is_dir(),'CPU preparation must finish before GPU submission')
 record=read(output/'RUN_CONTRACT.json')
 need(record['code']==common.code_digest() and not record['smoke'],'CPU preparation source identity differs')
 return SimpleNamespace(CODE=code,HISTORY=history,OUT=output,PYTHON=sys.executable,
                        OLD_CONTRACT=record['contract'],CPU_HASH=record['code'],SPEC=output/'S3_EXECUTION_CONTRACT.json')

def validate_bundle():
 return {name:sha(JOB/name) for name in SCRIPTS}

def base_args(context,spec):
 return ['--code-root',str(context.CODE),'--history-root',str(context.HISTORY),
         '--output-root',str(context.OUT),'--contract',spec['contract']]

def worker(args):
 context=load_context(args);spec=read(context.SPEC);seed=args.worker
 need(seed in SEEDS,'Unsupported GPU seed')
 arguments=base_args(context,spec)+['--seed',str(seed)]
 for script,extra,threads in [('train_gpu.py',['--device','cuda:0'],2),('eval_cpu.py',[],8)]:
  need(sha(JOB/script)==spec['scripts_sha256'][script],'Execution source changed: '+script)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=spec['gpu_uuids'][str(seed)] if script=='train_gpu.py' else '',
           OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads))
  subprocess.run([context.PYTHON,'-u','-B',str(JOB/script),*arguments,'--threads',str(threads),*extra],
                 env=env,check=True,pass_fds=(int(env['S3_INHERITED_LOCK_FD']),))

def prepare(dispatch,allocation,hashes):
 sys.path.insert(0,str(dispatch.CODE));sys.dont_write_bytecode=True
 import ssl_common as common
 old=read(dispatch.OUT/'RUN_CONTRACT.json')
 need(old['contract']==dispatch.OLD_CONTRACT and old['code']==dispatch.CPU_HASH==common.code_digest(),'Original scientific identity changed')
 from finalize_six import expected_stages,verify_receipt
 expected=expected_stages()[0]
 actual={str(p.parent.relative_to(dispatch.OUT)):sha(p) for p in dispatch.OUT.rglob('_SUCCESS.json')}
 need(set(actual)==set(expected),'CPU preparation requires exactly the 52 specified committed stages')
 baseline={rel:verify_receipt(common,dispatch.OUT,rel,dispatch.OLD_CONTRACT)[0] for rel in expected}
 need(actual==baseline,'CPU preparation receipt snapshot changed')
 baseline_path=dispatch.OUT/'S3_BASELINE_52.json'
 need(not baseline_path.exists(),'Existing CPU receipt snapshot must not be overwritten')
 need(not dispatch.SPEC.exists(),'S3_EXECUTION_CONTRACT already exists; preserve/archive it explicitly before a fresh managed run')
 for seed in SEEDS:
  path=dispatch.OUT/f'S3/seed_{seed}'
  need(not path.exists() or not any(path.iterdir()),f'Uncleared prior GPU/CPU artifacts for seed {seed}; no automatic deletion or migration')
 exclusive_json(baseline_path,baseline)
 baseline_sha=sha(baseline_path)
 devices=allocation['torch_probe']['devices'];mapping={str(seed):devices[i]['uuid'] for i,seed in enumerate(SEEDS)}
 spec={'created_at':now(),'old_contract':dispatch.OLD_CONTRACT,'frozen_code_digest':dispatch.CPU_HASH,
       'sources':old['sources'],'scripts_sha256':hashes,'baseline_receipts_sha256':baseline_sha,
       'seeds':[42,*SEEDS],'cpu_seeds':[42],'gpu_assignment':{str(seed):i for i,seed in enumerate(SEEDS)},
       'gpu_uuids':mapping,'slurm':allocation,
       'slurm_seed_mapping':{str(seed):{'logical_step_device':i,'physical_uuid':mapping[str(seed)]} for i,seed in enumerate(SEEDS)},
       'comparison':'Both conditions within each seed share initialization, backend, precision and training budget. No isolated backend effect estimated.',
       'scope':'S3 has six paired training seeds: one CPU and five GPU. S1 has six historical seeds and S2 has three. Paired learner bootstrap uses 2000 draws conditional on seed42 fitted models.',
       'training':'CUDA FP32, fixed epoch8, two original paired variants, one Slurm-authorized device per new seed; frozen CPU FP32 probe, evaluations and reference.',
       'resource_scope':'One Slurm srun task, one node, five assigned GPU UUIDs, forty CPUs; no detached controller or arbitrary device fallback.',
       'primary':['clean_minus_aug_current_MSE','clean_minus_aug_velocity_D2']}
 spec['contract']=hashlib.sha256(json.dumps(spec,sort_keys=True,ensure_ascii=True).encode()).hexdigest()
 exclusive_json(dispatch.SPEC,spec)
 return spec

def proc_info(pid):
 try:
  text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
  return {'pid':int(pid),'ppid':int(fields[1]),'start':fields[19]}
 except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):return None

def descendants(roots):
 records={}
 for path in Path('/proc').iterdir():
  if path.name.isdigit():
   item=proc_info(int(path.name))
   if item:records[item['pid']]=item
 wanted=set(roots);changed=True
 while changed:
  before=len(wanted);wanted.update(pid for pid,item in records.items() if item['ppid'] in wanted);changed=len(wanted)>before
 return {pid:records[pid] for pid in wanted if pid in records and pid!=os.getpid()}

def cleanup(children):

 records=descendants([p.pid for p in children.values()])
 for sig in (signal.SIGTERM,signal.SIGKILL):
  for pid,item in reversed(list(records.items())):
   current=proc_info(pid)
   if current and current['start']==item['start']:
    try:os.kill(pid,sig)
    except ProcessLookupError:pass
  until=time.monotonic()+(20 if sig==signal.SIGTERM else 5)
  while time.monotonic()<until:
   current={pid:proc_info(pid) for pid in records}
   if not any(current[pid] and current[pid]['start']==item['start'] for pid,item in records.items()):break
   time.sleep(.25)
 for p in children.values():
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:pass

def supervise(args):
 import fcntl
 hashes=validate_bundle();dispatch=load_context(args);allocation=allocated_devices(dispatch.PYTHON)
 logs=dispatch.OUT/'execution';logs.mkdir(exist_ok=True)
 token=f"job{allocation['job_id']}_step{allocation['step_id']}_{time.time_ns()}"
 state_path=logs/f'S3_SLURM_{token}_state.json';exit_path=logs/f'S3_SLURM_{token}_exit.json'
 state={'created_at':now(),'status':'preparing','controller_pid':os.getpid(),'slurm':allocation,
        'workers':{},'state_file':str(state_path),'exit_file':str(exit_path),'detached':False}
 exclusive_json(state_path,state);children={};result_code=1
 def stop(signum,frame):raise RuntimeError(f'Slurm/controller termination signal {signum}')
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
 with (dispatch.OUT/'RUN.lock').open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
  try:
   spec=prepare(dispatch,allocation,hashes);state.update(status='running',contract=spec['contract']);write(state_path,state)
   print(json.dumps({'status':'running','contract':spec['contract'],'slurm':allocation,'state_file':str(state_path)}),flush=True)
   for seed in SEEDS:
    log=logs/f'S3_SLURM_{token}_seed{seed}.log'
    env=dict(os.environ,S3_INHERITED_LOCK_FD=str(lock.fileno()),PYTHONDONTWRITEBYTECODE='1',
             CUDA_VISIBLE_DEVICES=','.join(allocation['authorized_uuids']),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with log.open('x',buffering=1) as stream:
     p=subprocess.Popen([dispatch.PYTHON,'-u','-B',str(JOB/'slurm_s3.py'),'--code-root',str(dispatch.CODE),
                         '--history-root',str(dispatch.HISTORY),'--worker',str(seed)],
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,env=env,
                        pass_fds=(lock.fileno(),))
    children[seed]=p;state['workers'][str(seed)]={'pid':p.pid,'physical_uuid':spec['gpu_uuids'][str(seed)],'status':'running','log':str(log)}
    write(state_path,state)
   pending=set(SEEDS)
   while pending:
    for seed in list(pending):
     rc=children[seed].poll()
     if rc is not None:
      state['workers'][str(seed)].update(status='complete' if rc==0 else 'failed',exit_code=rc,finished_at=now())
      pending.remove(seed);write(state_path,state)
    if pending:time.sleep(5)
   need(all(children[s].returncode==0 for s in SEEDS),'One or more original workers failed; scientific completion is not claimed')
   need(validate_bundle()==hashes,'Execution files changed while workers ran')
   state['status']='finalizing';write(state_path,state)


   fcntl.flock(lock,fcntl.LOCK_UN)
   log=logs/f'S3_SLURM_{token}_finalize.log'
   with log.open('x',buffering=1) as stream:
    p=subprocess.Popen([dispatch.PYTHON,'-u','-B',str(JOB/'finalize_six.py'),*base_args(dispatch,spec)],
                       stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    children['finalizer']=p;state['finalizer']={'pid':p.pid,'log':str(log)};write(state_path,state)
    need(p.wait()==0,'Finalizer failed; see its job-specific log')
   marker=read(dispatch.OUT/'final/COMPLETE.json')
   need(marker['contract']==spec['contract'] and marker['technical_completion'] is True,'Final completion identity is missing or wrong')
   state.update(status='complete',exit_code=0,final_complete_sha256=sha(dispatch.OUT/'final/COMPLETE.json'));result_code=0
  except BaseException:
   state.update(status='failed',exit_code=1,exception=traceback.format_exc())
   for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
   cleanup(children);print(state['exception'],file=sys.stderr,flush=True)
  finally:
   state['finished_at']=now();write(state_path,state);exclusive_json(exit_path,state)
 return result_code

def main(argv=None):
 ap=argparse.ArgumentParser(description='Five concurrent paired S3 fits within one allocated Slurm step.')
 ap.add_argument('--code-root',type=Path,default=JOB)
 ap.add_argument('--history-root',type=Path)
 ap.add_argument('--worker',type=int,choices=SEEDS,help=argparse.SUPPRESS)
 ap.add_argument('--gpu-probe',action='store_true',help=argparse.SUPPRESS)
 ap.add_argument('--check-allocation',action='store_true',help='Inspect the Slurm GPU allocation without starting training')
 args=ap.parse_args(argv);sys.dont_write_bytecode=True
 if args.gpu_probe:gpu_probe();return 0
 if args.check_allocation:
  print(json.dumps(allocated_devices(sys.executable),indent=2),flush=True);return 0
 if args.history_root is None:ap.error('--history-root is required')
 if args.worker is not None:worker(args);return 0
 return supervise(args)

if __name__=='__main__':raise SystemExit(main())
