#!/usr/bin/env python3

from __future__ import annotations
import argparse, datetime, json, os, re, shlex, shutil, subprocess, sys
from pathlib import Path
from cn_common import *
from run_generative_construction_null import choose_work


def build_jobs(args):
    here=Path(__file__).resolve().parent;run=here/'run_generative_construction_null.py';summary=here/'summarize_generative_construction_null.py'
    smoke=args.phase=='smoke';out=Path(args.output_root).resolve();work=choose_work(out,args.work_root,smoke)
    sources=source_paths(args.repo_root)
    py=str(Path(shutil.which(args.python) or args.python).resolve())
    if not Path(py).is_file():raise FileNotFoundError(f'Python executable not found: {py}')
    if not smoke and not args.dry_run:
        sw=choose_work(out,None,True)
        marker=read_json(sw/'SMOKE_OK.json');sp=load_protocol(sw)
        if not marker.get('passed') or marker['code_hashes']!=code_hashes() or marker['source_hashes']!={k:sha256(sources[k]) for k in ('stage1','semantic','original_null')} or sp['seed']!=args.seed:
            raise RuntimeError('Real-data smoke gate is missing or incompatible. Run --phase smoke with the same code/seed first.')
    common=['--work-root',str(work)]
    def worker(cmd,*more):return [py,str(run),cmd,*common,*more]
    def summ(cmd,*more):return [py,str(summary),cmd,*common,*more]
    init=worker('init','--output-root',str(out),'--repo-root',sources['repo'],'--seed',str(args.seed),'--replicates',str(3 if smoke else 999))
    if smoke:init+=['--smoke','--smoke-users',str(args.smoke_users)]
    if args.data_root:init+=['--data-root',str(Path(args.data_root).resolve())]
    jobs=[]
    def add(name,commands,array=None,mem=None):
        jobs.append(dict(name=name,commands=commands,array=array,mem=mem or args.mem));return name
    add('init_calibrate',[init,worker('calibrate')],mem=args.prepare_mem)
    add('prepare_A_val',[worker('prepare','--split','A_val')],mem=args.prepare_mem)
    R=3 if smoke else 999


    for split in (['A_val'] if smoke else ['A_val','B_confirm']):
        if split=='B_confirm':add('prepare_B_confirm',[worker('prepare','--split',split)],mem=args.prepare_mem)
        for mode in ('null_a','null_b'):
            task_count=(R+args.replicates_per_task-1)//args.replicates_per_task
            for first_task in range(0,task_count,args.array_size):
                begin=first_task*args.replicates_per_task
                batch_tasks=min(args.array_size,task_count-first_task)


                cmd=worker('simulate','--split',split,'--mode',mode)
                add(f'{split}_{mode}_{begin:04d}',[cmd],array=dict(start=begin,count=batch_tasks,batch=args.replicates_per_task,total=R))
            if not smoke:add(f'summarize_{split}_{mode}',[summ('summarize','--split',split,'--mode',mode)])
        if split=='A_val' and not smoke:add('release_validation',[summ('release')])
    add('smoke_finish' if smoke else 'final_report',[worker('smoke-finish')] if smoke else [summ('final')])
    return work,jobs


def submit(args):
    if not 1<=args.concurrency<=16:raise ValueError('--concurrency must be 1..16; each task requests one CPU and no GPU.')
    if not 1<=args.array_size<=999:raise ValueError('--array-size must be 1..999')
    if not 1<=args.replicates_per_task<=64:raise ValueError('--replicates-per-task must be 1..64')
    for m in (args.mem,args.prepare_mem):
        if not re.fullmatch(r'[1-9][0-9]*[MG]',m):raise ValueError('Memory requires an explicit unit, e.g. 16G, 64G')
    work,jobs=build_jobs(args);logs=work/'logs'
    if not args.dry_run:
        if shutil.which('sbatch') is None:raise RuntimeError('sbatch not found; run on the Slurm submission host.')
        logs.mkdir(parents=True,exist_ok=True)
    previous=None;receipts=[]
    for i,job in enumerate(jobs):
        lines=['set -euo pipefail','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1','export PYTHONUNBUFFERED=1']
        for cmd in job['commands']:
            line=shlex.join(cmd)
            if job['array'] is not None:
                ar=job['array']
                lines += [f'FIRST=$(( {ar["start"]} + SLURM_ARRAY_TASK_ID * {ar["batch"]} ))',f'LAST=$(( FIRST + {ar["batch"]} ))',f'if (( LAST > {ar["total"]} )); then LAST={ar["total"]}; fi','for (( RID=FIRST; RID<LAST; RID++ )); do',line+' --replicate "$RID"','done']
            else:lines.append(line)
        cmd=['sbatch','--parsable','--job-name=cn_'+job['name'],'--nodes=1','--ntasks=1','--cpus-per-task=1','--mem='+job['mem'],'--time='+args.time,'--output='+str(logs/(job['name']+'_%A_%a.out')),'--error='+str(logs/(job['name']+'_%A_%a.err'))]
        if args.partition:cmd+=['--partition',args.partition]
        if args.account:cmd+=['--account',args.account]
        if previous:cmd+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        if job['array'] is not None:cmd+=['--array=0-'+str(job['array']['count']-1)+'%'+str(args.concurrency)]
        cmd+=['--wrap',shlex.join(['bash','-c','\n'.join(lines)])]
        if args.dry_run:
            print(shlex.join(cmd));previous=f'DRYRUN_JOB_{i+1}'
        else:
            result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if result.returncode:raise RuntimeError(f'sbatch failed for {job["name"]}: {result.stderr}\nAlready submitted jobs: {receipts}')
            jid=result.stdout.strip().split(';')[0]
            if not jid.isdigit():raise RuntimeError(f'Unexpected sbatch response: {result.stdout}')
            previous=jid;receipts.append(dict(name=job['name'],job_id=jid,array=job['array']))
            atomic_json(work/'submission_latest.json',dict(phase=args.phase,work_root=str(work),jobs=receipts))
            print(f'{job["name"]}: job {jid}',flush=True)
    print(f'Work directory: {work}')
    if not args.dry_run:print('Success marker: '+str(work/('SMOKE_OK.json' if args.phase=='smoke' else 'final/COMPLETE.json')))


def main():
    p=argparse.ArgumentParser(description='Submit dependency-ordered CPU construction-control jobs through Slurm.')
    p.add_argument('--phase',choices=['smoke','all'],required=True);p.add_argument('--output-root',default=DEFAULT_OUTPUT);p.add_argument('--work-root');p.add_argument('--repo-root');p.add_argument('--data-root')
    p.add_argument('--python',default=sys.executable);p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke-users',type=int,default=200)
    p.add_argument('--concurrency',type=int,default=8);p.add_argument('--array-size',type=int,default=999);p.add_argument('--replicates-per-task',type=int,default=8)
    p.add_argument('--mem',default='16G');p.add_argument('--prepare-mem',default='64G');p.add_argument('--time',default='24:00:00')
    p.add_argument('--partition',default='');p.add_argument('--account',default='');p.add_argument('--dry-run',action='store_true')
    submit(p.parse_args())
if __name__=='__main__':main()
