#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft_common import code_digest, load_json, save_json, sha, require
from synthetic_fixture import build_fixture

def main():
    p = argparse.ArgumentParser(description='Synthetic integration tests with 2,000 paired learner resamples.')
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--work-root', required=True, type=Path)
    a = p.parse_args()
    root = a.work_root.resolve()
    require(not root.exists(), 'Use a new empty integration work directory; never overwrite a dataset')
    root.mkdir(parents=True)
    history = root / 'synthetic_history'
    build_fixture(history, a.source.resolve())
    cli = Path(__file__).resolve().parents[1] / 'run_fourterm.py'
    env = os.environ.copy()
    env['OPENBLAS_NUM_THREADS'] = '2'
    commands = []

    def run(name, phase, workers=2, expected=0):
        cmd = [sys.executable, str(cli), phase, '--output-root', str(history), '--threads', '2', '--bootstrap-workers', str(workers), '--chunk-size', '500']
        log = root / (name + '.log')
        with log.open('w') as out:
            r = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env)
        require((r.returncode == 0) == (expected == 0), f'Unexpected exit for {name}; read {log}')
        commands.append({'case': name, 'phase': phase, 'expected_success': expected == 0, 'exit_code': r.returncode, 'passed': True, 'log_sha256': sha(log)})
    run('01_no_smoke_gate', 'all', expected=1)
    run('02_smoke_and_full_2000', 'both')
    final = history / 'four_term_mechanism_revision/final'
    complete = load_json(final / 'COMPLETE.json')
    require(complete['synthetic_fixture'] is True, 'Synthetic results must be explicitly labelled')
    marker = final / '_SUCCESS.json'
    digest = sha(marker)
    run('03_verified_resume_different_workers', 'all', workers=1)
    require(sha(marker) == digest, 'Verified resume changed committed results')
    models = history / 'four_term_mechanism_revision/models/final_models.json'
    before = models.read_bytes()
    models.write_bytes(before + b' ')
    try:
        run('04_changed_model_blocks', 'all', expected=1)
    finally:
        models.write_bytes(before)
    save_json({'scope': 'SYNTHETIC ONLY; NOT EDNET RESULTS', 'code_digest': code_digest(), 'original_source_sha256': sha(a.source), 'unit_tests_run_separately': True, 'integration_cases': commands, 'full_M1_bootstrap_replicates': 2000, 'full_M3_seed42_bootstrap_replicates': 2000, 'frozen_neural_seed_outputs_checked': [42, 2026, 666, 606, 37, 4669], 'parquet_backend_tested': False, 'historical_EdNet_data_tested': False}, root / 'integration_acceptance.json')
    print(root / 'integration_acceptance.json')
if __name__ == '__main__':
    main()
