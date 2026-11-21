#!/usr/bin/env python3

from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys
from pathlib import Path
from ssl_common import *

def main():
    ap=argparse.ArgumentParser(description='CPU preparation, reference controls, and integrity checks for SSL information controls.')
    ap.add_argument('command',choices=['check','smoke','cpu','all','s1','s2','s3','report','status','selftest'])
    ap.add_argument('--output-root',type=Path,default=Path(DEFAULT_ROOT),help='Existing outputs_KT4 root; historical files are read-only')
    ap.add_argument('--fourterm-root',type=Path,help='Optional actual completed four-term root if not underneath outputs_KT4')
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--loader-workers',type=int,default=0)
    ap.add_argument('--infer-batch',type=int,default=16,help='Contiguous sequence inference batch, not training batch')
    ap.add_argument('--chunk-len',type=int,default=512)
    args=ap.parse_args()
    require(args.threads>0 and args.loader_workers>=0 and args.infer_batch>0 and args.chunk_len>0,'Invalid runtime resources')
    os.environ['OMP_NUM_THREADS']=str(args.threads);os.environ['MKL_NUM_THREADS']=str(args.threads)
    if args.command=='selftest':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
        result=unittest.TextTestRunner(verbosity=2).run(suite);return 0 if result.wasSuccessful() else 1
    if args.command=='status':
        for name in (run_name(True),run_name(False)):
            root=args.output_root/name
            print(json.dumps({'directory':str(root),'committed_stages':[str(p.parent.relative_to(root)) for p in sorted(root.glob('**/_SUCCESS.json'))],
                'complete_marker_present':(root/'final/COMPLETE.json').exists(),
                'warning':'Status listing is not a hash verification or scientific verdict.'},indent=2))
        return 0
    from ssl_pipeline import Context,check_history,s0,run_s1,run_s2,run_s3
    if args.command=='check':
        paths,part,ref,info=check_history(args.output_root,strict=args.fourterm_root is None)
        if args.fourterm_root:
            from ssl_data import discover_reference
            ref=discover_reference(args.fourterm_root);require(ref is not None,'No completed reference under explicit fourterm root')
        print(json.dumps(jsonable({'paths':paths,'K6':part,'C3_reference':ref,'checkpoint_count':len(info),
          'scope':'file/config checks only; smoke performs actual CPU inference/training; full source hashing occurs at S0'}),indent=2));return 0
    smoke=args.command=='smoke'
    root=args.output_root/run_name(smoke)
    with run_lock(root):
        ctx=Context(args.output_root,smoke,args.threads,args.loader_workers,args.infer_batch,args.chunk_len,args.fourterm_root)
        s0(ctx)
        if not smoke:
            gate=args.output_root/run_name(True)/'SMOKE_OK.json'
            require(gate.is_file(),'Real-data smoke has not passed. Run: python run_ssl.py smoke --output-root ...')
            g=load_json(gate);require(g['code']==ctx.code and g['sources']==ctx.signature,'Smoke source/code differs from formal run')
        cmd=args.command
        if cmd in ('smoke','all','cpu','s1'):run_s1(ctx)
        if cmd in ('smoke','all','cpu','s2'):run_s2(ctx)
        if cmd in ('smoke','all','cpu','s3'):
            seeds=ctx.newseeds
            if cmd in ('cpu','s3'):ctx.newseeds=(42,)
            try:run_s3(ctx)
            finally:ctx.newseeds=seeds
        if cmd in ('smoke','all','report'):
            from ssl_reporting import finalize
            p=finalize(ctx)
            if smoke:save_json({'code':ctx.code,'sources':ctx.signature,'complete_receipt':receipt(p),
                               'technical_only':True},root/'SMOKE_OK.json')
        log('Command finished: '+cmd)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (AuditError,FileNotFoundError) as e:
        print('AUDIT STOP: '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
