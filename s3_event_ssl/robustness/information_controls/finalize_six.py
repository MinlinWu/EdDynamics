#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ALL_SEEDS = (42, 2026, 666, 606, 37, 4669)
GPU_SEEDS = ALL_SEEDS[1:]
S2_SEEDS = ALL_SEEDS[:3]
SPLITS = ('A_val', 'B_confirm')
VARIANTS = ('aug', 'clean')


def expected_stages():
    old = ['S0']
    for seed in ALL_SEEDS:
        for kind in ('full', 'shuffled'):
            old.extend(f'S1/{kind}/seed_{seed}/{split}' for split in SPLITS)
    for seed in S2_SEEDS:
        old.extend(f'S2/seed_{seed}/{name}' for name in
                   ('sample_A_train', 'sample_A_val', 'head_H', 'head_HZ', *SPLITS))
    old.extend(f'S3/seed_42/{variant}/{name}' for variant in VARIANTS
               for name in ('training', 'probe', *SPLITS))
    old.append('S3/seed_42/inference')
    new = [f'S3/seed_{seed}/{variant}/{name}' for seed in GPU_SEEDS
           for variant in VARIANTS for name in ('training', 'probe', *SPLITS)]
    assert len(old) == 52 and len(new) == 40 and len(set(old + new)) == 92
    return old, new


def verify_receipt(common, root, rel, contract):
    path = root / rel
    record = common.load_json(path / '_SUCCESS.json')
    common.require(record.get('contract') == contract, f'Wrong stage contract: {rel}')
    identity = common.objhash({'contract': contract, 'inputs': record['inputs']})
    common.require(record.get('identity') == identity, f'Invalid stage identity: {rel}')
    common.require(bool(record.get('outputs')), f'Empty stage receipt: {rel}')
    for filename in record['outputs']:
        target = (path / filename).resolve()
        common.require(target.is_relative_to(path.resolve()), f'Unsafe stage output path: {rel}/{filename}')
    digest = common.verify_committed(path, contract)
    return digest, record


def verify_amendment(common, args, root):
    amendment_path = root / 'S3_EXECUTION_CONTRACT.json'
    amendment = common.load_json(amendment_path)
    common.require(amendment.get('contract') == args.contract, 'Amendment contract argument mismatch')
    payload = {k: v for k, v in amendment.items() if k != 'contract'}
    common.require(hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest() == args.contract,
                   'Amendment content hash mismatch')
    common.require(tuple(map(int, amendment['seeds'])) == ALL_SEEDS, 'Wrong six-seed allocation')
    common.require(list(map(int, amendment['cpu_seeds'])) == [42], 'Only completed CPU42 may be imported')
    common.require(set(map(int, amendment['gpu_assignment'])) == set(GPU_SEEDS), 'Wrong GPU seed allocation')
    original = common.load_json(root / 'RUN_CONTRACT.json')
    s0 = common.load_json(root / 'S0/run_contract.json')
    common.require(original['contract'] == amendment['old_contract'] == s0['run_contract'], 'Original CPU contract mismatch')
    common.require(original['code'] == amendment['frozen_code_digest'] == s0['code'] == common.code_digest(),
                   'Frozen original code changed')
    common.require(original['sources'] == amendment['sources'] == s0['sources_signature'], 'Source signature mismatch')
    common.require(not s0.get('smoke') and not s0.get('synthetic'), 'Smoke/synthetic evidence cannot complete formal run')
    common.require(root.resolve() == (args.history_root.resolve() / common.RUN_NAME).resolve(),
                   'Output root is not the original formal history root')
    source_records = common.load_json(root / 'S0/source_records.json')
    bindings = common.load_json(root / 'S0/resolved_source_bindings.json')
    common.require(Path(bindings['history_root']).resolve() == args.history_root.resolve(), 'History binding mismatch')
    signature = common.objhash({'files': [{k: r[k] for k in ('path', 'bytes', 'sha256')}
                                         for r in source_records], 'resolved_bindings': bindings})
    common.require(signature == amendment['sources'], 'Original source inventory identity changed')
    for record in source_records:
        common.verify_record(record, full=True)
    scripts = amendment['scripts_sha256']
    common.require(bool(scripts) and Path(__file__).name in {Path(k).name for k in scripts},
                   'Amendment must bind the actual finalizer source')
    for filename, digest in scripts.items():
        path = Path(filename)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        common.require(path.is_file() and common.sha(path) == digest, f'Amendment script changed: {path}')
    baseline_path = root / 'S3_BASELINE_52.json'
    common.require(common.sha(baseline_path) == amendment['baseline_receipts_sha256'], 'Original receipt snapshot changed')
    baseline = common.load_json(baseline_path)
    common.require(set(baseline) == set(expected_stages()[0]), 'Original 52-stage snapshot has wrong scope')
    for rel, digest in baseline.items():
        common.require(common.receipt(root / rel) == digest, f'Original stage receipt changed: {rel}')
    return amendment, source_records, common.sha(amendment_path)


def verify_pairs(common, root, dependencies, records, amendment):

    backend_rows = []
    for seed in ALL_SEEDS:
        contracts = []
        for variant in VARIANTS:
            base = f'S3/seed_{seed}/{variant}'
            train = common.load_json(root / base / 'training/training_contract.json')
            cfg = train['config']
            common.require(int(cfg['seed']) == seed and int(cfg['epochs']) == 8, f'Wrong S3 seed/epoch: {base}')
            common.require(train['variant'] == variant, f'Wrong condition identity: {base}')
            common.require(cfg['lambda_future'] == 1 and cfg['lambda_state'] == 0 and cfg['lambda_closure'] == 0,
                           f'Wrong S3 objectives: {base}')
            history = common.load_json(root / base / 'training/training_history.json')
            common.require([int(x['epoch']) for x in history] == list(range(1, 9)), f'Incomplete fixed eight epochs: {base}')
            common.require(all(x['selection'] == 'none; fixed epoch' for x in history), f'Changed checkpoint selection: {base}')
            if seed != 42:
                runtime = train['numerical_runtime']
                common.require(train['stage_contract'] == amendment['contract'] and train['smoke'] is False,
                               f'Wrong GPU training identity: {base}')
                common.require(runtime['precision'] == 'GPU float32' and runtime['AMP'] is False and runtime['TF32'] is False,
                               f'Wrong GPU precision: {base}')
                common.require(train['physical_runtime']['CUDA_VISIBLE_DEVICES'] == amendment['gpu_uuids'][str(seed)],
                               f'Wrong allocated GPU: {base}')
            probe_inputs = records[base + '/probe']['inputs']
            common.require(probe_inputs.get('encoder') == dependencies[base + '/training'], f'Probe encoder chain mismatch: {base}')
            for split in SPLITS:
                inputs = records[base + '/' + split]['inputs']
                common.require(inputs.get('encoder') == dependencies[base + '/training'] and
                               inputs.get('probe') == dependencies[base + '/probe'], f'Evaluation chain mismatch: {base}/{split}')
                common.require(inputs.get('split') == split, f'Evaluation split mismatch: {base}/{split}')
            contracts.append(train)
            backend_rows.append({'seed': seed, 'model': variant,
                'training_backend': 'CPU' if seed == 42 else 'GPU',
                'training_stage_contract': records[base + '/training']['contract'],
                'initial_state_sha256': train['initial_state_sha256'],
                'epochs': 8, 'training_contract_source': base + '/training/training_contract.json',
                'training_contract_sha256': common.sha(root / base / 'training/training_contract.json'),
                'training_metadata_json': json.dumps(train, ensure_ascii=False, sort_keys=True)})
        a, b = contracts
        common.require(a['initial_state_sha256'] == b['initial_state_sha256'], f'Pair initial states differ: {seed}')
        common.require(a['config'] == b['config'], f'Pair training configurations differ: {seed}')
        common.require(a['original_data_windows'] == b['original_data_windows'], f'Pair windows differ: {seed}')
        common.require(a.get('features_excluded_from_target_denominator') == [], f'Aug future targets excluded: {seed}')
        common.require(len(b.get('features_excluded_from_target_denominator', [])) == 14, f'Clean target mask not 14 columns: {seed}')
    for seed in S2_SEEDS:
        heads = [common.load_json(root / f'S2/seed_{seed}/head_{mode}/head_contract.json') for mode in ('H', 'HZ')]
        for key in ('initial_state_sha256', 'train_keys_sha256', 'val_keys_sha256', 'settings'):
            common.require(heads[0][key] == heads[1][key], f'Existing S2 pair mismatch: {seed}/{key}')
    return backend_rows


def verify_gpu_completion(common, root, dependencies, records, amendment):
    markers = {}
    for seed in GPU_SEEDS:
        base = f'S3/seed_{seed}'
        training_path = root / base / 'GPU_TRAINING_COMPLETE.json'
        evaluation_path = root / base / 'CPU_EVALUATION_COMPLETE.json'
        tr = common.load_json(training_path)
        ev = common.load_json(evaluation_path)
        common.require(tr['seed'] == ev['seed'] == seed and tr['contract'] == ev['contract'] == amendment['contract'],
                       f'GPU completion identity mismatch: {seed}')
        common.require(tr['smoke'] is False and ev['bootstrap_run'] is False, f'Wrong GPU completion scope: {seed}')
        identity = ev['evaluation_identity']
        common.require(identity['module_sha256'] == amendment['scripts_sha256']['eval_cpu.py'] and
                       identity['new_training_contract'] == amendment['contract'] and
                       identity['original_S0_receipt'] == dependencies['S0'], f'Evaluation implementation mismatch: {seed}')
        common.require(tr['gpu_driver_sha256'] == amendment['scripts_sha256']['train_gpu.py'], f'GPU driver mismatch: {seed}')
        for variant in VARIANTS:
            common.require(tr['training_receipts'][variant] == dependencies[f'{base}/{variant}/training'],
                           f'Training completion receipt mismatch: {seed}/{variant}')
            for name in ('training', 'probe', *SPLITS):
                rel = f'{base}/{variant}/{name}'
                common.require(ev['outputs'][variant][name] == dependencies[rel], f'Evaluation completion receipt mismatch: {rel}')
                if name != 'training':
                    common.require(records[rel]['inputs']['CPU_evaluation_spec'] == common.objhash(identity),
                                   f'Evaluation specification mismatch: {rel}')
        markers[str(training_path.relative_to(root))] = common.sha(training_path)
        markers[str(evaluation_path.relative_to(root))] = common.sha(evaluation_path)
    return markers


def gather_tables(common, pd, root, stage_names):
    frames, intervals, bridges, futures, replay = [], [], [], [], []
    for rel in stage_names:
        path = root / rel
        parts = Path(rel).parts
        seed = next((int(x[5:]) for x in parts if x.startswith('seed_')), None)
        block = 'S1_' + parts[1] if parts[0] == 'S1' else parts[0]
        for name in ('metrics',):
            p = path / 'evaluation' / (name + '.csv')
            if p.is_file():
                df = pd.read_csv(p, float_precision='round_trip')
                df['seed'], df['block'], df['source'] = seed, block, str(p.relative_to(root))
                df['training_backend'] = ('CPU' if seed == 42 else 'GPU') if block == 'S3' else (
                    'CPU_heads_on_historical_frozen_encoder' if block == 'S2' else 'historical_checkpoint_CPU_replay')
                df['stage_contract'] = common.load_json(path / '_SUCCESS.json')['contract']
                frames.append(df)
        candidates = [path / 'inference/paired_intervals.csv']
        if parts[-1] == 'inference':
            candidates.append(path / 'paired_intervals.csv')
        for p in candidates:
            if p.is_file():
                df = pd.read_csv(p, float_precision='round_trip')
                df['block'], df['seed'], df['source'] = parts[0], seed, str(p.relative_to(root))
                intervals.append(df)
        p = path / 'bridge/C3_bridge_metrics.csv'
        if p.is_file() and parts[-1] == 'B_confirm':
            df = pd.read_csv(p, float_precision='round_trip'); df['block'], df['seed'], df['source'] = block, seed, str(p.relative_to(root)); bridges.append(df)
        p = path / 'future_prediction_heldout.csv'
        if p.is_file() and parts[-1] == 'B_confirm' and parts[0] == 'S3':
            df = pd.read_csv(p, float_precision='round_trip'); df['seed'], df['variant'], df['source'] = seed, parts[2], str(p.relative_to(root)); futures.append(df)
        for p in sorted(path.rglob('S0_archive_bridge.json')):
            r = common.load_json(p)
            replay.append({'source': str(p.relative_to(root)), 'status': r['status'],
                           'max_absolute_error': max(r['max_absolute_cpu_minus_archive'])})
    metrics = pd.concat(frames, ignore_index=True)
    iv = pd.concat(intervals, ignore_index=True)
    common.require(len(iv) == 24 and set(iv.seed) == {42}, 'Expected 24 original seed42 interval rows')
    common.require((iv.requested_replicates == 2000).all(), 'Nonformal bootstrap count')
    common.require(not metrics.duplicated(['block', 'seed', 'model', 'split']).any(), 'Duplicate metric identity')
    for split in SPLITS:
        part = metrics[metrics.split == split]
        common.require(len(part) == 42 and len(part[part.block == 'S3']) == 12, 'Missing/extra formal metric rows')
    return metrics, iv, bridges, futures, replay


def paired_differences(pd, metrics):
    rows = []
    s3 = metrics[metrics.block == 'S3']
    numeric = [c for c in s3.select_dtypes(include='number').columns if c != 'seed']
    for (seed, split), frame in s3.groupby(['seed', 'split'], sort=False):
        pair = frame.set_index('model')
        if set(pair.index) != set(VARIANTS):
            raise ValueError(f'Incomplete S3 pair {seed}/{split}')
        for metric in numeric:
            a, b = pair.loc['aug', metric], pair.loc['clean', metric]
            rows.append({'seed': int(seed), 'split': split, 'training_backend': pair.loc['aug', 'training_backend'],
                         'metric': metric, 'aug': a, 'clean': b, 'clean_minus_aug': b - a,
                         'primary_metric': metric in ('current_MSE', 'velocity_D2'),
                         'interpretation': 'paired point difference; metric-specific direction; not device effect'})
    return pd.DataFrame(rows)


BOUNDARY = (
    'S3 uses CPU seed42 and GPU seeds2026/666/606/37/4669. Both conditions use the same device within each seed. '
    'Seed and training device are confounded; the comparisons do not estimate a device effect or establish CPU/GPU '
    'training equivalence. The all-six summaries describe this specified mixed-device set. '
    'S1 retains six historical seeds and S2 retains three. The 2000 paired whole-learner draws per block '
    'condition on the CPU seed42 fitted models and do not quantify all-six training or GPU variability. '
    'Completion does not depend on effect direction or benchmark advantage. Field-local cosine differs from interval '
    'cosine, and the new same-readout trajectory cannot inherit correlations from the original conditional proposal. '
    'Negative values, unavailable metrics, support differences and method sensitivities are retained.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify all 92 stages and summarize the six-seed mixed-device information controls.')
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--history-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--contract', required=True, help='Amended contract SHA256 string')
    args = parser.parse_args(argv)
    if len(args.contract) != 64 or any(c not in '0123456789abcdef' for c in args.contract):
        parser.error('--contract must be a lowercase SHA256 string')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.code_root.resolve()))
    import pandas as pd
    import ssl_common as common
    import ssl_reporting as reporting
    root = args.output_root.resolve()
    with common.run_lock(root):
        amendment, sources, amendment_sha = verify_amendment(common, args, root)
        old, new = expected_stages()
        dependencies, receipts = {}, {}
        for rel in old + new:
            digest, record = verify_receipt(common, root, rel,
                                            amendment['old_contract'] if rel in old else args.contract)
            dependencies[rel], receipts[rel] = digest, record
        backends = verify_pairs(common, root, dependencies, receipts, amendment)
        completion_markers = verify_gpu_completion(common, root, dependencies, receipts, amendment)
        metrics, intervals, bridges, futures, replay = gather_tables(common, pd, root, old + new)
        differences = paired_differences(pd, metrics)
        inputs = {'amendment_sha256': amendment_sha, 'old_contract': amendment['old_contract'],
                  'source_signature': amendment['sources'], 'stage_receipts': dependencies,
                  'completion_markers': completion_markers,
                  'finalizer_sha256': common.sha(__file__)}

        def produce(dst):
            confirmation = metrics[metrics.split == 'B_confirm'].copy()
            common.write_csv(confirmation, dst / 'confirmation_metrics_all.csv')
            common.write_csv(metrics[metrics.split == 'A_val'], dst / 'validation_metrics_all.csv')
            common.write_csv(intervals, dst / 'paired_intervals_all.csv')
            common.write_csv(reporting.interval_interpretations(intervals), dst / 'result_interpretation.csv')
            common.write_csv(pd.concat(bridges, ignore_index=True), dst / 'C3_bridge_all.csv')
            common.write_csv(pd.concat(futures, ignore_index=True), dst / 'common_future_targets_confirmation.csv')
            common.write_csv(replay, dst / 'CPU_vs_archive_precision.csv')
            common.write_csv(backends, dst / 'S3_training_backend_and_contracts.csv')
            common.write_csv(differences, dst / 'S3_paired_differences_all_metrics.csv')
            primary = differences[differences.primary_metric]
            common.write_csv(primary, dst / 'S3_primary_paired_differences.csv')
            summary = []
            for label, df in [('all_six_mixed', differences), ('GPU_five', differences[differences.training_backend == 'GPU']),
                              ('CPU42_only', differences[differences.training_backend == 'CPU'])]:
                for (split, metric), group in df.groupby(['split', 'metric'], sort=False):
                    vals = group.clean_minus_aug
                    finite = vals.map(lambda x: pd.notna(x) and math.isfinite(float(x)))
                    v = vals[finite]
                    complete = len(v) == len(group)
                    summary.append({'group': label, 'split': split, 'metric': metric, 'requested_seeds': len(group),
                                    'finite_seeds': len(v), 'mean_clean_minus_aug': v.mean() if complete else float('nan'),
                                    'minimum': v.min() if complete else float('nan'),
                                    'maximum': v.max() if complete else float('nan'), 'negative_count': int((v < 0).sum()),
                                    'positive_count': int((v > 0).sum()), 'zero_count': int((v == 0).sum()),
                                    'status': 'descriptive_only' if len(v) == len(group) else 'incomplete_NA_retained'})
            common.write_csv(summary, dst / 'S3_seed_spread_descriptive.csv')
            common.write_csv([dict(zip(['analysis', 'comparison', 'evidence', 'interpretation_boundary'], row))
                              for row in reporting.MAP], dst / 'analysis_index.csv')
            ctx = SimpleNamespace(out=root, history=args.history_root, contract=args.contract, smoke=False)
            reporting.make_figures(dst, confirmation, intervals)
            reporting.make_field_figures(ctx, dst)
            table = reporting.metric_sections(confirmation)
            uncertainty = reporting.interval_sections(intervals)
            primary_table = reporting.markdown(primary, ['seed', 'split', 'training_backend', 'metric', 'aug', 'clean', 'clean_minus_aug'])
            text = '# SSL information controls: six-seed S3\n\n' + BOUNDARY + '\n\n'
            text += f'CPU preparation contract: `{amendment["old_contract"]}`. GPU execution contract: `{args.contract}`. '
            text += 'Verified dependent stages:52 original +40 GPU =92, excluding this final reporting stage.\n\n'
            text += '## S3 paired primary point contrasts\n\n' + primary_table + '\n\n' + table + '\n\n' + uncertainty
            text += '\n\n## C3 boundary\n\n' + reporting.C3_BOUNDARY
            text += '\n\n## Existing evidence and remaining scope\n\nOriginal pure-SSL/task-only, construction-null, mechanism and C3/P2 results '
            text += 'remain independently identified evidence. No individual-loss or reconstructed raw-order control is added. '
            text += 'S3 jointly changes alignment inputs and future targets; post-freeze probes use training labels. '
            (dst / 'numeric_report.md').write_text(text, encoding='utf-8')
            scope = reporting.scope_status()
            scope.update(S3_training_design=BOUNDARY)
            common.save_json(scope, dst / 'SCOPE_STATUS.json')
            common.save_json({'amendment': amendment, 'amendment_sha256': amendment_sha,
                              'original_stage_count': 52, 'gpu_stage_count': 40, 'dependency_count': 92,
                              'stage_receipts': dependencies, 'source_records_verified': len(sources),
                              'completion_markers': completion_markers,
                              'excluded_from_metrics': ['uncommitted/partial stages'],
                              'finalizer_sha256': common.sha(__file__)}, dst / 'mixed_contract_manifest.json')

            common.save_json({'technical_completion': True, 'scientific_success_required': False,
                              'smoke': False, 'synthetic': False, 'contract': args.contract,
                              'original_contract': amendment['old_contract'],
                              'completion_scope': '52 original CPU-contract stages +40 amended GPU S3 stages',
                              'original_cpu_full_run_completed': False,
                              'S1_seeds': list(ALL_SEEDS), 'S2_seeds': list(S2_SEEDS), 'S3_seeds': list(ALL_SEEDS),
                              'S3_CPU_seeds': [42], 'S3_GPU_seeds': list(GPU_SEEDS),
                              'actual_completed_S3_encoder_runs': 12,
                              'seed42_bootstrap_replicates_per_block': 2000,
                              'bootstrap_scope': 'original CPU seed42 only', 'dependency_count_excluding_final': 92,
                              'source_records_verified': len(sources), 'outputs_are_real_computations_not_filled_results': True,
                              'device_seed_confounding_disclosed': True}, dst / 'COMPLETE.json')
        result = common.stage(root, 'final', args.contract, inputs, produce)
        common.log('Amended six-seed final report committed: ' + str(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
