#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(description='Calibrated conditional predictors and frozen four-term model comparisons.')
    p.add_argument('phase', choices=['check', 'smoke', 'all', 'both', 'status', 'tests'], nargs='?', default='both')
    p.add_argument('--output-root', type=Path, default=Path('/path/to/outputs_KT4'), help='Existing historical output root, read-only except for the new analysis output directories')
    p.add_argument('--run-root', type=Path, help='New result directory; its sibling with _smoke suffix is used for technical validation')
    p.add_argument('--threads', type=int, default=8, help='CPU BLAS/thread limit, not a scientific parameter')
    p.add_argument('--bootstrap-workers', type=int, default=4, help='Shared-memory CPU bootstrap workers; draws independent of worker count')
    p.add_argument('--chunk-size', type=int, default=200000, help='Streaming I/O/prediction block rows')
    p.add_argument('--test-source', type=Path, default=None, help='Only for tests: original Phase-1 .py')
    a = p.parse_args()
    if a.threads < 1 or a.bootstrap_workers < 1 or a.chunk_size < 1:
        p.error('CPU counts and chunk size must be positive')
    if a.bootstrap_workers > a.threads:
        p.error('--bootstrap-workers cannot exceed --threads')
    return a

def main():
    args = parse_args()
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[k] = str(args.threads)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from types import SimpleNamespace
    import traceback
    import numpy as np
    from threadpoolctl import threadpool_limits
    from ft_common import DEFAULT_ROOT, VERSION, MODELS, ALPHAS, NBOOT, SEED, FEATURES, load_json, save_json, objhash, code_digest, require, log, exclusive_lock, validate_sources, sha, verify_stage_receipt
    from ft_data import load_partition, assert_disjoint
    from ft_pipeline import stage_dataset, fit_models, evaluation, bootstrap_main, crossmodel
    from ft_reporting import collect_existing, final_report
    history = args.output_root.expanduser().resolve()
    formal = args.run_root.expanduser().resolve() if args.run_root else history / 'four_term_mechanism_revision'
    smoke = formal.with_name(formal.name + '_smoke')
    if formal == history or formal == history / 'stage1' or history / 'stage1' in formal.parents:
        raise ValueError('Result directory must not replace original inputs')
    if args.phase == 'tests':
        import unittest
        src = args.test_source or history / 'stage2_phase2_freeze/metadata/frozen_phase1_implementation.py'
        require(src.is_file(), f'Test source not found: {src}')
        os.environ['FOURTERM_TEST_SOURCE'] = str(src)
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.phase == 'status':
        for p in (smoke, formal):
            print(str(p))
            for f in sorted(p.rglob('_SUCCESS.json')) if p.exists() else []:
                print('  stage marker (status does not revalidate hashes):', f.relative_to(p))
            f = p / 'final/COMPLETE.json'
            print('  final marker:', f.exists())
        return 0
    require(history.is_dir(), f'History directory not found: {history}')

    def context(is_smoke):
        out = smoke if is_smoke else formal
        out.mkdir(parents=True, exist_ok=True)
        p1, params, traincal, finalcal, prov = validate_sources(history, out / 'runtime')
        signature = objhash({Path(x['path']).relative_to(history).as_posix(): x['sha256'] for x in prov['artifacts']})
        protocol = {'version': VERSION, 'code_digest': code_digest(), 'original_sources_signature': signature, 'source_archive_contract_sha256': sha(Path(__file__).parent / 'source_contract.json'), 'history_root': str(history), 'smoke': is_smoke, 'synthetic_fixture': (history / 'SYNTHETIC_ONLY.json').exists(), 'seed': SEED, 'bootstrap_replicates': 20 if is_smoke else NBOOT, 'smoke_max_users_per_split': 500 if is_smoke else None, 'models': MODELS, 'features': FEATURES, 'alphas': ALPHAS, 'grid': [40, 40, -1, 1], 'minimum_count': 30, 'statistical_protocol': 'README.md', 'conditional_one_step': True, 'new_fits': 'C3 and ARX only', 'run_via_scheduler': False}
        rh = objhash(protocol)
        rc = out / 'run_contract.json'
        if rc.exists():
            require(load_json(rc)['run_hash'] == rh, 'Run contract changed; never merge different scientific/code versions')
        else:
            save_json({'run_hash': rh, 'protocol': protocol}, rc)
        save_json(prov, out / 'source_audit.json')
        ctx = SimpleNamespace(history=history, out=out, p1=p1, params=params, train_cal=traincal, final_cal=finalcal, provenance=prov, run_hash=rh, source_signature=signature, smoke=is_smoke, synthetic_fixture=(history / 'SYNTHETIC_ONLY.json').exists(), smoke_users=500 if is_smoke else None, nboot=20 if is_smoke else NBOOT, chunk_size=args.chunk_size, workers=args.bootstrap_workers, partition=load_partition(history, prov['partition_centers_path']))
        return ctx

    def pipeline(ctx):
        log(('SMOKE' if ctx.smoke else 'FORMAL') + ' run: ' + str(ctx.out))
        if not ctx.smoke:
            marker = smoke / 'SMOKE_OK.json'
            require(marker.is_file(), 'Run smoke first (or use both); no technical gate bypass')
            sm = load_json(marker)
            require(sm['code_digest'] == code_digest() and sm['source_signature'] == ctx.source_signature, 'Smoke does not match current code/frozen sources')
            require(sm['final_receipt_sha256'] == sha(smoke / 'final/_SUCCESS.json'), 'Smoke completion receipt changed')
            for stage in ('prepared/A_train', 'prepared/A_val', 'models', 'prepared/B_confirm', 'confirmation', 'inference', 'cross_model', 'existing_evidence', 'final'):
                verify_stage_receipt(smoke / stage, run_hash=sm['run_hash'])
        train = stage_dataset(ctx, 'A_train')
        val = stage_dataset(ctx, 'A_val')
        assert_disjoint([train, val])
        mp, models = fit_models(ctx, train, val)
        confirm = stage_dataset(ctx, 'B_confirm')
        assert_disjoint([train, val, confirm])
        ep = evaluation(ctx, confirm, mp, models)
        with threadpool_limits(limits=1):
            bp = bootstrap_main(ctx, confirm, ep)
            cp = crossmodel(ctx, confirm, ep, mp)
        lp = collect_existing(ctx)
        fp = final_report(ctx, ep, bp, cp, lp, mp)
        if ctx.smoke:
            save_json({'code_digest': code_digest(), 'source_signature': ctx.source_signature, 'run_hash': ctx.run_hash, 'final_receipt_sha256': sha(fp / '_SUCCESS.json'), 'scope': 'technical test only; no formal inference'}, smoke / 'SMOKE_OK.json')
        log('Finished technical pipeline. See ' + str(fp / 'numeric_report.md'))
    with exclusive_lock(formal):
        current = None
        try:
            with threadpool_limits(limits=args.threads):
                if args.phase == 'check':
                    current = context(True)
                    log('Source/manifest/K6 checks passed; raw panels and predictions will be audited during smoke.')
                else:
                    for mode in [True, False] if args.phase == 'both' else [args.phase == 'smoke']:
                        current = context(mode)
                        pipeline(current)
        except Exception as e:
            path = (current.out if current is not None else formal) / 'failure_latest.json'
            save_json({'error': str(e), 'type': type(e).__name__, 'traceback': traceback.format_exc(), 'scientific_results_not_reinterpreted_to_pass': True}, path)
            log('FAILED: ' + str(e))
            log('Diagnostic: ' + str(path))
            raise
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
