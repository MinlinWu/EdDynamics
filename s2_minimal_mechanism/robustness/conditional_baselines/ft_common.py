from __future__ import annotations
import ast
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
import numpy as np
import pandas as pd
VERSION = 'fourterm-supplement-v1'
DEFAULT_ROOT = '/path/to/outputs_KT4'
SPLITS = ('A_train', 'A_val', 'B_confirm')
FEATURES = ('M', 'Psi', 'E', 'B', 'G', 'answered_count_proxy', 'response_alignable_interval', 'support_alignable_interval', 'response_neutral_interval', 'support_neutral_interval', 'idle_mass_interval', 'next_gap_days')
AUX = ('V', 'target_V_next', 'target_E_next', 'target_B_next', 'emp_delta_M_response', 'emp_delta_V_response', 'emp_delta_Psi_active', 'emp_delta_Psi_idle', 'phase_columns_available')
PARAMS = ('theta0', 'thetaM', 'thetaPsi', 'thetaMPsi', 'phi0', 'deltaS', 'phiPsi', 'lambdaR', 'lambdaA', 'lambdaI')
MODELS = ('four_term', 'C3', 'ARX', 'persistence', 'zero_signed')
ALPHAS = (0.0, 1e-06, 0.0001, 0.01, 1.0, 100.0)
EPS = 1e-12
BINS = np.linspace(-1.0, 1.0, 41)
NCELL = 1600
MIN_COUNT = 30
SEED = 42
NBOOT = 2000

class AuditError(RuntimeError):
    pass

def require(condition, message):
    if not condition:
        raise AuditError(message)

def clean(obj):
    if dataclasses.is_dataclass(obj):
        return clean(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def objhash(obj):
    return hashlib.sha256(json.dumps(clean(obj), sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)

def log(text):
    print(time.strftime('[%Y-%m-%d %H:%M:%S] ') + str(text), flush=True)

def find_table(base, required=True):
    p = Path(base)
    candidates = [p] if p.suffix in ('.parquet', '.csv', '.gz') else [Path(str(p) + s) for s in ('.parquet', '.csv.gz', '.csv')]
    for q in candidates:
        if q.is_file():
            return q
    if required:
        raise FileNotFoundError('Missing table: ' + str(base))
    return None

def read_table(path, columns=None):
    p = find_table(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(p, columns=columns)
    return pd.read_csv(p, usecols=columns)

def iter_table(path, columns=None, chunk_size=200000, allow_missing=False):
    p = find_table(path)
    if p.suffix == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise AuditError('Parquet input requires pyarrow in the original EdDynamics environment.') from e
        pf = pq.ParquetFile(p)
        chosen = columns
        if columns is not None and allow_missing:
            chosen = [c for c in columns if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=chunk_size, columns=chosen, use_threads=False):
            yield batch.to_pandas()
    else:
        cols = (lambda s: s in columns) if columns is not None and allow_missing else columns
        yield from pd.read_csv(p, usecols=cols, chunksize=chunk_size)

def write_csv(df, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def fingerprint(path):
    p = Path(path).resolve()
    st = p.stat()
    return {'path': str(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha(p)}

def check_fingerprint(record, content=True):
    p = Path(record['path'])
    require(p.is_file(), f'Input disappeared: {p}')
    st = p.stat()
    require(st.st_size == record['size'], f'Input size changed: {p}')
    if content or st.st_mtime_ns != record['mtime_ns']:
        require(sha(p) == record['sha256'], f'Input content changed: {p}')

def code_digest():
    root = Path(__file__).resolve().parent
    paths = sorted(list(root.glob('*.py')) + list(root.glob('*.sh')) + [root / 'source_contract.json', root / 'README.md'])
    return objhash({p.name: sha(p) for p in paths})

def verify_stage_receipt(path, run_hash=None, input_hash=None):
    root = Path(path).resolve()
    receipt = root / '_SUCCESS.json'
    require(receipt.is_file(), f'Missing stage receipt: {receipt}')
    r = load_json(receipt)
    if run_hash is not None:
        require(r.get('run_hash') == run_hash, f'Stage run contract changed: {root}')
    if input_hash is not None:
        require(r.get('input_hash') == input_hash, f'Stage input contract changed: {root}')
    require(isinstance(r.get('outputs'), dict), f'Invalid stage output manifest: {root}')
    for rel, digest in r['outputs'].items():
        require(isinstance(rel, str) and (not Path(rel).is_absolute()), f'Invalid artifact path: {rel}')
        p = (root / rel).resolve()
        require(p.is_relative_to(root), f'Artifact path escapes stage: {rel}')
        require(p.is_file() and sha(p) == digest, f'Stage output changed or incomplete: {p}')
    return r

def atomic_stage(root, name, run_hash, inputs, callback):
    dst = Path(root) / name
    ih = objhash(inputs)
    receipt = dst / '_SUCCESS.json'
    if receipt.exists():
        verify_stage_receipt(dst, run_hash=run_hash, input_hash=ih)
        log('Verified reuse: ' + name)
        return dst
    require(not dst.exists(), f'Uncommitted destination exists: {dst}. Preserve it for inspection; do not merge results.')
    tmp = Path(root) / ('.work_' + name.replace('/', '_') + '_' + uuid.uuid4().hex)
    tmp.mkdir(parents=True)
    try:
        callback(tmp)
        outputs = {str(p.relative_to(tmp)): sha(p) for p in sorted(tmp.rglob('*')) if p.is_file()}
        save_json({'run_hash': run_hash, 'input_hash': ih, 'inputs': inputs, 'outputs': outputs, 'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, tmp / '_SUCCESS.json')
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dst)
        log('Completed: ' + name)
    except Exception:
        log('Failed stage retained for diagnosis: ' + str(tmp))
        raise
    return dst

def receipt_hash(path):
    return sha(Path(path) / '_SUCCESS.json')

def assert_close(a, b, label, atol=2e-09, rtol=2e-08):
    a = np.asarray(a)
    b = np.asarray(b)
    require(a.shape == b.shape, f'{label}: shapes differ {a.shape} vs {b.shape}')
    require(np.isfinite(a).all() and np.isfinite(b).all(), f'{label}: non-finite values')
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(np.allclose(a, b, atol=atol, rtol=rtol), f'{label}: max absolute difference {err:g}')
    return err

def structured_keys(keys):
    a = np.ascontiguousarray(keys, dtype=np.int64)
    return a.view(np.dtype([('user', '<i8'), ('step', '<i8')])).reshape(-1)

def key_positions(sorted_keys, query):
    s = structured_keys(sorted_keys)
    q = structured_keys(query)
    idx = np.searchsorted(s, q)
    valid = idx < len(s)
    safe = np.minimum(idx, max(len(s) - 1, 0))
    if len(s):
        valid &= s[safe] == q
    return (idx, valid)

def import_source(path, cache_root):
    os.environ['NUMBA_CACHE_DIR'] = str(Path(cache_root).resolve() / 'numba')
    Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
    forced = {'EDNET_STAGE1_SIGNED_GRID_N': '41', 'EDNET_STAGE1_MIN_DRIFT_BIN_COUNT': '30', 'EDNET_STAGE1_TAU_RESPONSE_DAYS': '10', 'EDNET_STAGE1_TAU_ACTIVITY_DAYS': '10', 'EDNET_STAGE1_RANDOM_STATE': '42', 'MECH_PHASE1_RANDOM_STATE': '42', 'EDNET_STAGE1_KMEANS_N_INIT': '20', 'EDNET_STAGE1_KMEANS_FIT_MAX_ROWS': '500000'}
    for k, v in forced.items():
        if k in os.environ and os.environ[k] != v:
            raise AuditError(f'Conflicting source environment {k}={os.environ[k]}; expected {v}.')
        os.environ[k] = v
    name = 'fourterm_frozen_' + sha(path)[:16]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = prev
    require(np.array_equal(mod.GRID_BINS_SIGNED, BINS), 'Frozen source grid mismatch')
    require(mod.MIN_DRIFT_BIN_COUNT == 30, 'Frozen source support threshold mismatch')
    return mod

def recover_historical_k6(history, current, old, certificate):
    require(isinstance(certificate, dict), 'Frozen K6 changed and has no registered recovery certificate')
    require(old['metadata_sha256'] == current['metadata_sha256'] == certificate['metadata_sha256'], 'K6 recovery metadata mismatch')
    require(old['centers_sha256'] == certificate['legacy_centers_sha256'] and current['centers_sha256'] == certificate['current_centers_sha256'], 'Unregistered K6 center file change')
    h = Path(history).resolve()
    references = certificate['references']
    artifacts = []
    canonical = None
    require([r['seed'] for r in references] == [42, 2026, 666, 606, 37, 4669], 'Incomplete K6 recovery evidence')
    columns = ['macrostate', 'center_M', 'center_Psi', 'center_M_standardized', 'center_Psi_standardized']
    for r in references:
        mp = h / r['manifest']
        cp = h / r['centers']
        require(mp.is_file() and sha(mp) == r['manifest_sha256'], 'K6 historical manifest changed')
        require(cp.is_file() and sha(cp) == certificate['export_sha256'], 'K6 historical export changed')
        declared = load_json(mp).get('macro_partition', {})
        require(declared.get('verified') is True and declared.get('fit_split') == 'A_train' and (declared.get('centers_sha256') == old['centers_sha256']) and (declared.get('metadata_sha256') == old['metadata_sha256']), 'Historical export does not declare the frozen K6 contract')
        frame = pd.read_csv(cp, float_precision='round_trip')
        require(list(frame.columns) == columns and len(frame) == 6, 'Invalid historical K6 export schema')
        require(np.array_equal(frame.macrostate.to_numpy(), np.arange(6)) and np.isfinite(frame.to_numpy(float)).all(), 'Invalid historical K6 labels or values')
        if canonical is None:
            canonical = frame
            canonical_path = cp
        else:
            require(np.array_equal(canonical.to_numpy(), frame.to_numpy()), 'Historical K6 exports disagree')
        artifacts.extend([mp, cp])
    metadata = load_json(current['metadata_path'])
    mean = np.asarray(metadata['scaler_mean'])
    scale = np.asarray(metadata['scaler_scale'])
    assert_close((canonical[['center_M', 'center_Psi']].to_numpy() - mean) / scale, canonical[['center_M_standardized', 'center_Psi_standardized']].to_numpy(), 'Historical K6 scaler consistency', atol=1e-12, rtol=0)
    actual = read_table(current['centers_path'])
    difference = assert_close(actual[columns].to_numpy(), canonical.to_numpy(), 'Registered K6 recovery numeric bound', atol=1e-12, rtol=0)
    return {'status': 'verified_historical_export_recovery', 'legacy_centers_sha256': old['centers_sha256'], 'current_centers_sha256': current['centers_sha256'], 'metadata_sha256': current['metadata_sha256'], 'export_sha256': certificate['export_sha256'], 'canonical_centers_path': str(canonical_path), 'max_current_vs_historical_abs_difference': difference, 'historical_files_modified': False, 'centers_refitted': False, 'note': 'Historical CSV values are used; current Parquet is preserved. This is not byte identity.', 'evidence_paths': [str(p) for p in artifacts]}

def validate_sources(history, out):
    h = Path(history).resolve()
    meta = h / 'stage2_phase2_freeze/metadata'
    mp = meta / 'phase2_frozen_model_manifest.json'
    manifest = load_json(mp)
    side = meta / 'phase2_frozen_model_manifest.sha256.json'
    require(side.is_file(), 'Phase-2 checksum sidecar is required')
    require(load_json(side).get('manifest_sha256') == sha(mp), 'Frozen manifest checksum mismatch')
    require(manifest.get('primary_macrostate') == ['M', 'Psi'], 'Wrong primary macrostate')
    g = manifest.get('guardrails', {})
    for name in ('B_confirm_read', 'parameter_search_opened', 'mechanism_family_reselected', 'mechanism_parameters_refit', 'kmeans_refit', 'macrostate_k_selected', 'region_redefinition'):
        require(name in g and g[name] is False, f'Frozen guardrail failed or absent: {name}')
    require(g.get('calibration_refit_scope') == 'A_train_plus_A_val_only', 'Wrong final calibration scope')
    fp = manifest['frozen_parameters']
    require(fp.get('family_key') == 'offset_dual_channel', 'Wrong frozen family')
    require(set(fp.get('free_mechanism_parameters', [])) == {'theta0', 'thetaM', 'phi0', 'deltaS'}, 'Wrong four terms')
    p = fp.get('full_parameter_vector') or {**fp.get('mechanism_parameters', {}), **fp.get('structural_zero_values', {}), **fp.get('fixed_nuisance_scales', {})}
    p = {n: float(p[n]) for n in PARAMS}
    require(all(np.isfinite(list(p.values()))), 'Non-finite parameters')
    require(all((abs(p[n]) <= EPS for n in ('thetaPsi', 'thetaMPsi', 'phiPsi'))), 'Structural zero changed')
    if fp.get('parameter_hash'):
        require(objhash(p) == fp['parameter_hash'], 'Parameter hash mismatch')
    cal = manifest['frozen_calibration']
    require(objhash(cal) == manifest['calibration_hash'], 'Calibration hash mismatch')
    cp = meta / 'phase2_pooled_development_calibration.json'
    require(objhash(load_json(cp)) == objhash(cal), 'Pooled calibration file differs from manifest')
    src = meta / 'frozen_phase1_implementation.py'
    expected = manifest['phase1_implementation'].get('frozen_copy_sha256') or manifest['phase1_implementation']['source_sha256']
    require(src.is_file() and sha(src) == expected, 'Relocated frozen source hash mismatch')
    contract = load_json(Path(__file__).parent / 'source_contract.json')
    require(sha(src) == contract['phase1_sha256'], 'Frozen source differs from the registered source version; no silent adapter substitution')
    trainp = h / 'stage2_phase1/metadata/A_train_only_skeleton_calibration.json'
    train = load_json(trainp)
    p1manifest = h / 'stage2_phase1/metadata/phase1_manifest.json'
    pm = load_json(p1manifest)
    require(pm.get('B_confirm_policy') == 'not read or used', 'Phase-1 B exclusion not established')
    lm = pm.get('stage1_load_manifest', {})
    require(lm.get('B_confirm_policy') == 'not read or used in Phase 1', 'Train calibration load scope differs')
    for name, c in [('train', train), ('pooled', cal)]:
        for k in ('eta', 'tau_response_days', 'tau_activity_days', 'residual_mass_per_answer', 'lambda_E', 'response_signed_gain', 'alignment_signed_gain'):
            require(c.get(k) is not None and np.isfinite(c[k]) and (c[k] > 0), f'Invalid {name} calibration: {k}')
        require(0.1 <= c['response_signed_gain'] <= 1 and 0.1 <= c['alignment_signed_gain'] <= 1, 'C3 feasible-gain contract violated')
        require(c['tau_response_days'] == 10 and c['tau_activity_days'] == 10, 'Not the primary 10-day experiment')
    require(abs(train['eta'] - cal['eta']) <= 1e-10, 'Train/pooled eta differ: cannot silently share prepared inputs')
    mod = import_source(src, out)
    k6 = mod.audit_stage1_kmeans_contract(h / 'stage1')
    old = manifest['stage1_fixed_k6_contract']
    require(old['metadata_sha256'] == k6['metadata_sha256'], 'Frozen K6 changed: metadata_sha256')
    recovery = None
    partition_centers = Path(k6['centers_path'])
    if old['centers_sha256'] != k6['centers_sha256']:
        recovery = recover_historical_k6(h, k6, old, contract.get('k6_historical_export_recovery'))
        partition_centers = Path(recovery['canonical_centers_path'])
    paths = [mp, side, src, cp, trainp, p1manifest, Path(k6['metadata_path']), Path(k6['centers_path'])]
    region = h / 'stage1/dynamics/candidate_regions/MR_PsiA'
    paths += [region / 'A_train_primary_convergence_core_mask.npy', region / 'training_convergence_thresholds.json']
    if recovery:
        paths += [Path(p) for p in recovery['evidence_paths']]
    return (mod, p, train, cal, {'artifacts': [fingerprint(x) for x in paths], 'stage1_root': str(h / 'stage1'), 'frozen_source': str(src), 'manifest': manifest, 'k6_audit': k6, 'k6_recovery': recovery, 'partition_centers_path': str(partition_centers), 'train_calibration_scope': 'A_train_only, verified against registered source and Phase-1 load manifest', 'relocation': 'canonical artifacts under supplied outputs root; original absolute strings retained in manifest'})

class Dataset:

    def __init__(self, path):
        self.path = Path(path)
        self.F = np.load(self.path / 'features.npy', mmap_mode='r')
        self.Y = np.load(self.path / 'target.npy', mmap_mode='r')
        self.keys = np.load(self.path / 'keys.npy', mmap_mode='r')
        self.aux = np.load(self.path / 'aux.npy', mmap_mode='r')
        self.users = np.load(self.path / 'users.npy', mmap_mode='r')
        self.ui = np.load(self.path / 'user_index.npy', mmap_mode='r')
        self.w = np.load(self.path / 'weights.npy', mmap_mode='r')
        self.cells = np.load(self.path / 'cells.npy', mmap_mode='r')
        self.meta = load_json(self.path / 'metadata.json')

    def __len__(self):
        return len(self.F)

    def blocks(self, size=200000):
        for a in range(0, len(self), size):
            yield slice(a, min(a + size, len(self)))

@contextlib.contextmanager
def exclusive_lock(root):
    import fcntl
    p = Path(root) / '.pipeline.lock'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise AuditError(f'Another process owns {p}') from e
        yield
