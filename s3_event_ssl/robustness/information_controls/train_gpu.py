#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace

SEEDS = (2026, 666, 606, 37, 4669)
VARIANTS = ('aug', 'clean')
VERSION = 's3-five-gpu-fp32-v1'
RNG_SCOPE = ('CPU/CUDA/loader/Python/NumPy RNG and optimizer states are saved; '
             'cuDNN internal recurrent-dropout state is not separately serialized. '
             'A resumed GPU trajectory is not certified bitwise identical to an uninterrupted run.')
PAIR_SCOPE = ('Identical CPU-initialized parameter tensors and matched windows/update budgets; '
              'no claim of identical per-step GPU dropout masks across conditions.')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + message, flush=True)


def inside(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


@contextlib.contextmanager
def seed_lock(seed_root):

    import fcntl
    seed_root.mkdir(parents=True, exist_ok=True)
    with (seed_root / 'GPU_TRAIN.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another GPU trainer owns this seed') from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + '\n')
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_runtime(code_root, device_text, threads):

    need(threads > 0, 'threads must be positive')
    need(re.fullmatch(r'cuda:\d+', device_text), 'Use an explicit logical device, e.g. cuda:0')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root))
    import numpy as np
    import torch
    common = importlib.import_module('ssl_common')
    data = importlib.import_module('ssl_data')
    neural = importlib.import_module('ssl_neural')
    for module in (common, data, neural):
        need(Path(module.__file__).resolve().parent == code_root, 'Wrong frozen module imported')
    need(torch.cuda.is_available(), 'CUDA unavailable; no silent CPU fallback')
    device = torch.device(device_text)
    need(device.index < torch.cuda.device_count(), 'Logical CUDA device does not exist')
    torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    prop = torch.cuda.get_device_properties(device)
    numerical_runtime = {
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'gpu_name': prop.name,
        'compute_capability': list(torch.cuda.get_device_capability(device)),
        'threads': threads, 'loader_workers': 0, 'precision': 'GPU float32',
        'AMP': False, 'TF32': False, 'compile': False,
        'deterministic_algorithms': True, 'cudnn_benchmark': False,
        'cudnn_deterministic': True, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'AdamW_foreach': False, 'AdamW_fused': False,
    }
    physical_runtime = {
        'logical_device': str(device),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu_uuid': str(getattr(prop, 'uuid', 'unavailable')),
        'total_memory_bytes': int(prop.total_memory),
    }
    return SimpleNamespace(torch=torch, np=np, common=common, data=data,
                           neural=neural, vendor=common.module('train_event_ssl.py'),
                           device=device, numerical_runtime=numerical_runtime,
                           physical_runtime=physical_runtime)


def future_objective_gpu(rt, model, x, cat, observed, removed=None, score_keep=None):

    torch, np = rt.torch, rt.np
    if removed is not None:
        x = x.clone()
        removed_device = torch.as_tensor(removed, dtype=torch.bool, device=x.device)
        x[..., removed_device] = 0
    out = model(x, cat)
    keep = np.ones(model.n_num, bool) if removed is None else ~np.asarray(removed, bool)
    keep = torch.as_tensor(keep, dtype=torch.bool, device=x.device)
    total = torch.zeros((), dtype=x.dtype, device=x.device)
    common = torch.zeros((), dtype=x.dtype, device=x.device)
    nh = 0
    commonkeep = keep if score_keep is None else torch.as_tensor(score_keep, dtype=torch.bool, device=x.device)
    for k, pred in out['future'].items():
        target, valid = rt.vendor.shifted_future_target(x, observed, k)
        if not bool((valid > 0).any()):
            continue
        total = total + rt.vendor.masked_mse(pred[..., keep], target[..., keep], valid)
        common = common + rt.vendor.masked_mse(pred[..., commonkeep], target[..., commonkeep], valid)
        nh += 1
    need(nh > 0, 'No real source/target future pair')
    return total / nh, common / nh


def cpu_tree(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, v) for v in value)
    return value


def atomic_torch_save(torch, value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(cpu_tree(torch, value), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_one(rt, train, val, template, seed, variant, removed, dest,
              settings, contract, inputs, smoke=False):

    torch, np, common = rt.torch, rt.np, rt.common
    dest = Path(dest)
    cfg = dict(template['config'])
    shapes = template['model_shapes']
    cfg.update(model_kind='pure_ssl', seed=seed, epochs=settings['epochs'],
               lambda_future=1., lambda_state=0., lambda_closure=0.,
               seq_len=settings['seq_len'], stride=settings['stride'],
               min_seq_len=settings['min_seq_len'], warmup_steps=settings['warmup_steps'],
               batch_size=settings['batch_size'], lr=settings['lr'],
               weight_decay=settings['weight_decay'], future_steps=settings['future_steps'],
               use_compile=False, amp_dtype='none_gpu_fp32', allow_truncated_supervision=False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = rt.neural.make_model(cfg, shapes)
    initial_digest = common.tensor_digest(model.state_dict())
    model = model.to(device=rt.device, dtype=torch.float32)
    need(all(p.dtype == torch.float32 for p in model.parameters()), 'Non-FP32 parameter')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'], foreach=False, fused=False)
    gen = torch.Generator().manual_seed(seed + 101)
    ds = rt.neural.FutureWindows(train, settings['seq_len'], settings['stride'], settings['min_seq_len'])
    vs = rt.neural.FutureWindows(val, settings['seq_len'], settings['stride'], settings['min_seq_len'])
    need(smoke or len(ds) >= settings['batch_size'], 'Fewer than one original-size training batch')
    dl = torch.utils.data.DataLoader(ds, batch_size=settings['batch_size'], shuffle=True,
                                    generator=gen, num_workers=0, drop_last=not smoke,
                                    persistent_workers=False)
    vl = torch.utils.data.DataLoader(vs, batch_size=settings['batch_size'], shuffle=False,
                                    num_workers=0, persistent_workers=False)
    identity = common.objhash({'version': VERSION, 'stage_contract': contract,
                              'inputs': inputs, 'config': cfg, 'variant': variant,
                              'feature_mask': removed, 'train_rows': train.n,
                              'val_rows': val.n, 'smoke': smoke,
                              'numerical_runtime': rt.numerical_runtime})
    start, history = 1, []
    checkpoint_path = dest / 'last_epoch.pt'
    if checkpoint_path.exists():
        ck = common.torch_load(checkpoint_path)
        need(ck.get('identity') == identity, 'GPU resume identity changed; never migrate a CPU partial')
        need(ck.get('initial_state_sha256') == initial_digest, 'Initialisation changed on resume')
        need(ck.get('training_backend') == 'cuda_fp32' and 'cuda_rng' in ck, 'Not a GPU continuation checkpoint')
        model.load_state_dict(ck['model_state_dict'], strict=True)
        opt.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['rng'])
        torch.cuda.set_rng_state(ck['cuda_rng'], device=rt.device)
        gen.set_state(ck['loader_rng'])
        np.random.set_state(ck['numpy_rng'])
        random.setstate(ck['python_rng'])
        history = ck['history']
        need([r['epoch'] for r in history] == list(range(1, ck['epoch'] + 1)), 'Checkpoint/history mismatch')
        need(0 < ck['epoch'] <= settings['epochs'], 'Invalid checkpoint epoch')
        start = ck['epoch'] + 1
        log(f'GPU RESUME seed={seed} variant={variant} next_epoch={start}; CPU/CUDA/loader/optimizer states restored')
    mask = removed if variant == 'clean' else None
    log(f'GPU TRAIN seed={seed} variant={variant} initial={initial_digest} '
        f'train_windows={len(ds)} val_windows={len(vs)} batches={len(dl)} epochs={settings["epochs"]}')
    for epoch in range(start, settings['epochs'] + 1):
        model.train()
        total = common_total = count = 0
        last_log = time.monotonic()
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            x = batch['x'].to(rt.device, dtype=torch.float32)
            cat = batch['cat'].to(rt.device)
            observed = batch['mask'].to(rt.device, dtype=torch.float32)
            loss, co = future_objective_gpu(rt, model, x, cat, observed, mask, ~removed)
            need(bool(torch.isfinite(loss)), 'Nonfinite future training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += float(loss.detach())
            common_total += float(co.detach())
            count += 1
            if count == 1 or time.monotonic() - last_log >= 30:
                log(f'GPU BATCH seed={seed} variant={variant} epoch={epoch} '
                    f'batch={count}/{len(dl)} loss={float(loss.detach()):.9g} device={rt.device} FP32 AMP=off TF32=off')
                last_log = time.monotonic()
        need(count > 0, 'No optimiser updates')
        model.eval()
        validation = validation_common = validation_count = 0
        with torch.no_grad():
            for batch in vl:
                loss, co = future_objective_gpu(
                    rt, model, batch['x'].to(rt.device, dtype=torch.float32),
                    batch['cat'].to(rt.device), batch['mask'].to(rt.device, dtype=torch.float32),
                    mask, ~removed)
                need(bool(torch.isfinite(loss)) and bool(torch.isfinite(co)), 'Nonfinite validation loss')
                validation += float(loss)
                validation_common += float(co)
                validation_count += 1
        need(validation_count > 0, 'No validation batches')
        history.append({'epoch': epoch, 'train_future_loss': total / count,
                        'train_common_target_loss': common_total / count,
                        'val_future_loss': validation / validation_count,
                        'val_common_target_loss': validation_common / validation_count,
                        'batches': count, 'selection': 'none; fixed epoch'})
        ck = {'model_state_dict': model.state_dict(), 'config': cfg, 'model_shapes': shapes,
              'epoch': epoch, 'best_epoch': epoch, 'identity': identity,
              'optimizer': opt.state_dict(), 'rng': torch.get_rng_state(),
              'cuda_rng': torch.cuda.get_rng_state(rt.device), 'loader_rng': gen.get_state(),
              'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
              'history': history, 'variant': variant, 'clean_mask': removed,
              'initial_state_sha256': initial_digest, 'future_only': True,
              'macro_labels_in_loss': False, 'checkpoint_selection': 'fixed epoch',
              'training_backend': 'cuda_fp32', 'training_precision': 'GPU float32; no AMP or TF32',
              'rng_resume_scope': RNG_SCOPE, 'paired_randomness_scope': PAIR_SCOPE,
              'numerical_runtime': rt.numerical_runtime, 'physical_runtime': rt.physical_runtime,
              'original_source_sha256': common.load_json(common.ROOT / 'source_contract.json')['files']['train_event_ssl.py']}
        atomic_torch_save(torch, ck, checkpoint_path)
        common.save_json(history, dest / 'training_history.json')
        log(f'GPU EPOCH DONE seed={seed} variant={variant} epoch={epoch}; checkpoint saved; validation not selected')
    need(len(history) == settings['epochs'], 'Did not reach the fixed epoch endpoint')
    final = common.torch_load(checkpoint_path)
    for key in ('optimizer', 'rng', 'cuda_rng', 'loader_rng', 'numpy_rng', 'python_rng'):
        final.pop(key, None)
    atomic_torch_save(torch, final, dest / 'fixed_epoch_model.pt')
    common.save_json(history, dest / 'training_history.json')
    common.save_json({
        'identity': identity, 'stage_contract': contract, 'config': cfg, 'variant': variant,
        'seed': seed, 'initial_state_sha256': initial_digest,
        'original_data_windows': len(ds), 'validation_data_windows': len(vs),
        'train_sample_filter': 'original finite macrostate eligibility inherited; not label-free sample selection',
        'gradient': 'future loss only; direct macrostate losses absent',
        'selection': 'fixed epoch, never M/Psi validation', 'parameter_width_unchanged': True,
        'features_excluded_from_target_denominator': np.flatnonzero(removed).tolist() if variant == 'clean' else [],
        'common_retained_target_indices': np.flatnonzero(~removed).tolist(),
        'reference': 'GPU FP32 matched aug/clean pair; distinct from retained CPU seed42 and historical best-only runs',
        'smoke': smoke, 'numerical_runtime': rt.numerical_runtime,
        'rng_resume_scope': RNG_SCOPE, 'paired_randomness_scope': PAIR_SCOPE,
        'physical_runtime': rt.physical_runtime, 'inputs': inputs,
        'checkpoint_export': 'CPU-resident tensors; compatible with frozen ssl_neural.load_model for CPU evaluation',
    }, dest / 'training_contract.json')
    del model, opt, ck, final, dl, vl
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Paired fixed-budget future-only encoder training on one allocated GPU.')
    ap.add_argument('--code-root', type=Path, required=True)
    ap.add_argument('--history-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True,
                    help='Existing formal root for formal runs; a separate empty root for technical smoke')
    ap.add_argument('--source-s0', type=Path,
                    help='Default: history-root/ssl_information_controls/S0; always read-only')
    ap.add_argument('--contract', required=True, help='New deployment contract SHA256 supplied by the supervisor')
    ap.add_argument('--seed', type=int, choices=SEEDS, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--smoke', action='store_true', help='32 whole users, one epoch; never manuscript evidence')
    args = ap.parse_args(argv)
    need(re.fullmatch(r'[0-9a-f]{64}', args.contract), '--contract must be a SHA256 hex digest')
    code = args.code_root.resolve()
    history = args.history_root.resolve()
    output = args.output_root.resolve()
    s0 = (args.source_s0 or history / 'ssl_information_controls/S0').resolve()
    need((s0 / '_SUCCESS.json').is_file(), 'Committed original S0 is required')
    need(not inside(output, code), 'Never write inside the frozen code directory')
    if args.smoke:
        need(output != s0.parent and not inside(output, s0.parent), 'Smoke must have an independent output root')
    else:
        need(output == s0.parent, 'Formal output must be the existing S0 parent')
    seed_root = output / 'S3' / f'seed_{args.seed}'
    need(seed_root.name != 'seed_42' and seed_root.parent == output / 'S3', 'Protected seed/path')
    rt = load_runtime(code, args.device, args.threads)
    common = rt.common
    s0_receipt = common.verify_committed(s0)
    old_run = common.load_json(s0 / 'run_contract.json')
    frozen_code_digest = common.code_digest()
    need(old_run['code'] == frozen_code_digest, 'Original frozen source changed')
    need(args.contract != old_run['run_contract'], 'GPU stages require a distinct contract; never label them CPU')
    paths = common.load_json(s0 / 'path_resolution.json')
    bindings = common.load_json(s0 / 'resolved_source_bindings.json')
    need(Path(bindings['history_root']).resolve() == history, 'History root differs from frozen S0')
    entry = paths[str(args.seed)]
    source_records = common.load_json(s0 / 'source_records.json')
    bypath = {str(Path(r['path']).resolve()): r for r in source_records}
    template_path = Path(entry['full']).resolve()
    template_sha = sha(template_path)
    need(template_sha == bypath[str(template_path)]['sha256'], 'Original template changed')
    template = common.torch_load(template_path)
    need(int(template['config']['seed']) == args.seed, 'Template seed mismatch')
    maxusers = 32 if args.smoke else None
    train = rt.data.Prepared(entry['input_root'], 'A_train', maxusers)
    val = rt.data.Prepared(entry['input_root'], 'A_val', maxusers)
    removed, lineage = rt.data.alignment_mask(train.manifest)
    need(int(template['model_shapes']['n_num']) == train.n_num, 'Template/input width mismatch')
    selected_paths = set(map(lambda p: Path(p).resolve(), train.files + val.files))
    selected_paths.update(Path(entry['input_root']).resolve().joinpath('metadata').glob('*.json'))
    selected_paths.add(template_path)
    selected_records = []
    for path in sorted(selected_paths):
        need(str(path) in bypath, f'Input is not bound by S0: {path}')
        selected_records.append(bypath[str(path)])
    def verify_sources(full=False):
        need(common.code_digest() == frozen_code_digest, 'Frozen source changed during training')
        need(common.receipt(s0) == s0_receipt, 'Source S0 receipt changed')
        for record in selected_records:
            common.verify_record(record, full=full)
    log(f'GPU SOURCE CHECK seed={args.seed}; hashing {len(selected_records)} original S0-bound input files')
    verify_sources(full=True)
    settings = dict(common.load_json(code / 'protocol.json')['s3'])
    if args.smoke:
        settings.update(epochs=1, batch_size=16, probe_rows=512)
    driver_sha = sha(Path(__file__).resolve())
    base_inputs = {
        'version': VERSION, 'source_S0': s0_receipt, 'source_S0_contract': old_run['run_contract'],
        'source_signature': old_run['sources_signature'], 'template': template_sha,
        'template_path': str(template_path), 'input_root': str(Path(entry['input_root']).resolve()),
        'seed': args.seed, 'settings': settings, 'mask': removed.tolist(),
        'smoke': args.smoke, 'train_rows': train.n, 'validation_rows': val.n,
        'frozen_code': frozen_code_digest, 'gpu_driver_sha256': driver_sha,
        'numerical_runtime': rt.numerical_runtime,
        'reference_policy': 'new GPU FP32 pair; CPU seed42 and all S0/S1/S2 retained unchanged',
    }
    log(f'GPU JOB seed={args.seed} device={args.device} contract={args.contract} smoke={args.smoke} '
        f'physical={json.dumps(rt.physical_runtime, sort_keys=True)}')
    with seed_lock(seed_root):
        completed = {}
        for variant in VARIANTS:
            verify_sources()
            need(sha(Path(__file__).resolve()) == driver_sha, 'GPU driver changed during training')
            inputs = dict(base_inputs, variant=variant)
            path = common.stage(output, f'S3/seed_{args.seed}/{variant}/training', args.contract, inputs,
                                lambda dst, v=variant, inp=inputs: train_one(
                                    rt, train, val, template, args.seed, v, removed, dst,
                                    settings, args.contract, inp, args.smoke))
            completed[variant] = path
            verify_sources()
        pair = [common.load_json(completed[v] / 'training_contract.json') for v in VARIANTS]
        need(pair[0]['initial_state_sha256'] == pair[1]['initial_state_sha256'], 'Paired initial states differ')
        common.save_json({'seed': args.seed, 'contract': args.contract, 'smoke': args.smoke,
                          'training_only': True, 'evaluation_pending': True,
                          'initial_state_sha256': pair[0]['initial_state_sha256'],
                          'training_receipts': {v: common.receipt(p) for v, p in completed.items()},
                          'source_S0': s0_receipt, 'gpu_driver_sha256': driver_sha},
                         seed_root / 'GPU_TRAINING_COMPLETE.json')
    log(f'GPU PAIR COMPLETE seed={args.seed}; evaluation and uncertainty remain separate')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        log('GPU AUDIT STOP: ' + repr(exc))
        raise
