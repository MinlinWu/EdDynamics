#!/usr/bin/env python3

from pathlib import Path
import argparse,subprocess,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ssl_common import *
from synthetic_fixture import create

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir',type=Path,required=True);a=ap.parse_args()
    require(not a.workdir.exists(),'Use a new empty synthetic work directory')
    create(a.workdir);logs=a.workdir/'acceptance_logs';logs.mkdir()
    args=[sys.executable,str(ROOT/'run_ssl.py')]
    common=['--output-root',str(a.workdir),'--threads','1','--infer-batch','8']
    def run(command,name):
        with (logs/name).open('w') as f:
            p=subprocess.run(args+[command]+common,stdout=f,stderr=subprocess.STDOUT)
        return p.returncode
    blocked=run('all','no_smoke_gate.log');require(blocked==2,'Formal pipeline did not reject absent smoke')
    require(run('smoke','smoke.log')==0,'Synthetic smoke failed; inspect logs')
    require(run('all','formal.log')==0,'Synthetic formal flow failed; inspect logs')
    root=a.workdir/RUN_NAME;final=load_json(root/'final/COMPLETE.json')
    require(final['synthetic'] and final['technical_completion'],'Synthetic result mislabeled')
    require(final['actual_encoder_runs']==6 and final['seed42_bootstrap_replicates']==2000,'Incomplete full-budget control test')
    before={str(p.relative_to(root)):sha(p) for p in root.rglob('_SUCCESS.json')}
    require(run('all','verified_reuse.log')==0,'Committed output reuse failed')
    after={str(p.relative_to(root)):sha(p) for p in root.rglob('_SUCCESS.json')}
    require(before==after,'Reuse rewrote committed stages')

    victim=root/'S3/seed_42/clean/training/fixed_epoch_model.pt';raw=victim.read_bytes()
    try:
        victim.write_bytes(raw+b'changed')
        tamper=run('report','tampered_model_gate.log');require(tamper==2,'Report accepted a changed frozen model')
    finally:victim.write_bytes(raw)
    result={'passed':True,'synthetic_only':True,'not_EdNet':True,'not_Parquet_backend':True,
      'no_smoke_gate_exit':blocked,'tampered_model_gate_exit':tamper,'committed_stage_count':len(before),
      'reuse_does_not_refit_or_rewrite':True,'formal_result':final,'package_code_sha256':code_digest(),
      'run_receipts':before,'logs':str(logs)}
    save_json(result,a.workdir/'INTEGRATION_ACCEPTANCE.json')
    print(json.dumps(jsonable(result),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
