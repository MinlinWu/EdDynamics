from __future__ import annotations
import json
import contextlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from ft_common import *

def grid_cells(X):
    X = np.asarray(X, dtype=float)
    require(X.ndim == 2 and X.shape[1] == 2 and np.isfinite(X).all(), 'Invalid states for grid')
    require(((X >= -1) & (X <= 1)).all(), 'State outside frozen [-1,1] domain')
    adjusted = np.where(X == 1.0, np.nextafter(1.0, -1.0), X)
    ix = np.digitize(adjusted[:, 0], BINS) - 1
    iy = np.digitize(adjusted[:, 1], BINS) - 1
    return (40 * ix + iy).astype(np.int64)

def recover_b_confirm_coordinates(history, source, dst, certificate, chunk_size, selected_smoke_users=None):
    require(isinstance(certificate, dict), 'Missing registered B coordinate recovery certificate')
    require(certificate.get('version') == 'b-confirm-coordinate-recovery-v1' and certificate.get('split') == 'B_confirm', 'Unknown coordinate recovery contract')
    require(certificate.get('maximum_coordinate_absolute_difference') == 1e-12 and certificate.get('expected_full_grid_changes') == {'features': 2, 'target': 2}, 'Coordinate recovery bounds changed')
    expected_rows = certificate.get('expected_full_rows')
    require(isinstance(expected_rows, int) and expected_rows > 0, 'Invalid historical coordinate row count')
    records = certificate.get('corrections', [])
    require(len(records) == 4 and sum((r.get('array') == 'features' for r in records)) == 2 and (sum((r.get('array') == 'target' for r in records)) == 2), 'Coordinate recovery requires exactly four registered values')
    identifiers = []
    for r in records:
        key = r.get('key')
        require(isinstance(key, list) and len(key) == 2 and all((isinstance(x, int) for x in key)) and (r.get('column') == 1), 'Invalid coordinate recovery key or column')
        native = float.fromhex(r['native_hex'])
        historical = float.fromhex(r['historical_hex'])
        require(np.isfinite(native) and -1e-16 < native < 0 and (historical == 0.0) and (abs(native - historical) <= 1e-12), 'Unregistered coordinate recovery value')
        identifiers.append((tuple(key), r['array']))
    require(len(set(identifiers)) == 4, 'Duplicate coordinate recovery entry')
    bindings = certificate.get('bindings', [])
    require(isinstance(bindings, list) and len(bindings) == 7, 'Incomplete coordinate recovery source bindings')
    h = Path(history).resolve()
    artifacts = []
    paths = []
    for record in [certificate['panel'], certificate['archive']] + bindings:
        relative = Path(record['relative_path'])
        path = (h / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(h), 'Coordinate source escapes history')
        require(path.is_file(), 'Coordinate recovery source missing: ' + str(path))
        fp = fingerprint(path)
        require(fp['sha256'] == record['sha256'], 'Coordinate recovery source changed: ' + str(path))
        paths.append(path)
        artifacts.append(fp)
    require(len(set(paths)) == len(paths), 'Duplicate coordinate recovery source binding')
    require(paths[0] == Path(source).resolve(), 'Coordinate recovery panel path mismatch')
    dst = Path(dst)
    K = np.load(dst / 'keys.npy', mmap_mode='r')
    F = np.load(dst / 'features.npy', mmap_mode='r+')
    Y = np.load(dst / 'target.npy', mmap_mode='r+')
    require(K.shape == (len(F), 2) and Y.shape == (len(F), 2) and (F.shape[1] == len(FEATURES)) and (len(K) > 0), 'Invalid coordinate recovery cache shape')
    ordered = lambda k: np.all((k[1:, 0] > k[:-1, 0]) | (k[1:, 0] == k[:-1, 0]) & (k[1:, 1] > k[:-1, 1]))
    require(ordered(K), 'Prepared coordinate keys are not unique and sorted')
    if selected_smoke_users is None:
        require(len(K) == expected_rows, 'Full coordinate recovery row count differs')
    else:
        require(np.isin(K[:, 0], selected_smoke_users).all(), 'Prepared coordinate user outside smoke selection')
    columns = ['user_id', 'bundle_step_index', 'M', 'Psi', 'target_M_next', 'target_Psi_next']
    archive = paths[1]
    chunks = iter_table(archive, columns, chunk_size) if archive.suffix == '.parquet' else pd.read_csv(archive, usecols=columns, chunksize=chunk_size, float_precision='round_trip')
    seen = np.zeros(len(K), bool)
    historical_seen = np.zeros(4, np.int64)
    pending = []
    grid_changes = {'features': 0, 'target': 0}
    max_difference = {'features': 0.0, 'target': 0.0}
    archive_rows = 0
    previous = None
    with contextlib.closing(chunks):
        for frame in chunks:
            if frame.empty:
                continue
            keys = frame[['user_id', 'bundle_step_index']].to_numpy(dtype=np.int64)
            require(ordered(keys) and (previous is None or tuple(keys[0]) > previous), 'Historical coordinate keys are not unique and sorted')
            previous = tuple(keys[-1])
            archive_rows += len(keys)
            idx, ok = key_positions(K, keys)
            if selected_smoke_users is None:
                require(ok.all(), 'Historical coordinate key absent from full prepared data')
            else:
                require(ok[np.isin(keys[:, 0], selected_smoke_users)].all(), 'Historical key missing for selected smoke user')
            require(not seen[idx[ok]].any(), 'Duplicate historical coordinate join')
            seen[idx[ok]] = True
            historical = {'features': frame[['M', 'Psi']].to_numpy(float), 'target': frame[['target_M_next', 'target_Psi_next']].to_numpy(float)}
            selected_positions = np.flatnonzero(ok)
            selected_records = []
            for ri, r in enumerate(records):
                position, found = key_positions(keys, np.asarray([r['key']], dtype=np.int64))
                if not found[0]:
                    continue
                pos = int(position[0])
                historical_seen[ri] += 1
                require(float(historical[r['array']][pos, 1]).hex() == r['historical_hex'], 'Certified historical coordinate value changed')
                if ok[pos]:
                    selected_records.append((ri, int(np.searchsorted(selected_positions, pos)), int(idx[pos])))
            if not ok.any():
                continue
            for name, array in [('features', F), ('target', Y)]:
                current = np.array(array[idx[ok], :2])
                old = historical[name][ok]
                difference = assert_close(current, old, 'Native/archive coordinates exceed recovery bound', atol=1e-12, rtol=0)
                max_difference[name] = max(max_difference[name], difference)
                corrected = current.copy()
                listed = np.zeros(len(current), bool)
                for ri, local, prepared_index in selected_records:
                    r = records[ri]
                    if r['array'] != name:
                        continue
                    require(float(current[local, 1]).hex() == r['native_hex'], 'Certified native coordinate value changed')
                    corrected[local, 1] = float.fromhex(r['historical_hex'])
                    listed[local] = True
                    pending.append((name, prepared_index, 1, float.fromhex(r['historical_hex'])))
                changed = grid_cells(current) != grid_cells(old)
                require(np.array_equal(changed, listed), 'Unlisted or missing coordinate grid change')
                require(np.array_equal(grid_cells(corrected), grid_cells(old)), 'Coordinate recovery did not restore historical grid')
                grid_changes[name] += int(changed.sum())
    require(archive_rows == expected_rows and seen.all(), 'Historical coordinate join is incomplete')
    require(np.array_equal(historical_seen, np.ones(4, dtype=np.int64)), 'Historical certificate entries are incomplete')
    if selected_smoke_users is None:
        require(grid_changes == certificate['expected_full_grid_changes'] and len(pending) == 4, 'Full coordinate recovery change count differs')
    for name, index, column, value in pending:
        (F if name == 'features' else Y)[index, column] = value
    F.flush()
    Y.flush()
    proof = {'status': 'verified_four_value_historical_coordinate_recovery', 'split': 'B_confirm', 'certificate_hash': objhash(certificate), 'archive_rows': archive_rows, 'prepared_rows': len(K), 'scope': 'all original S2 valid confirmation rows' if selected_smoke_users is None else 'selected smoke users', 'historical_certificate_values_verified': 4, 'applied_value_count': len(pending), 'grid_changes': grid_changes, 'max_native_archive_absolute_difference': max_difference, 'historical_files_modified': False, 'global_snapping': False, 'models_refitted': False, 'applied_corrections': [r for r in records if any((name == r['array'] and tuple(K[index]) == tuple(r['key']) for name, index, _, _ in pending))], 'artifacts': artifacts}
    save_json(proof, dst / 'coordinate_recovery_audit.json')
    return proof

def prepare_dataset(p1, history, split, eta, dst, chunk_size, smoke_users=None, coordinate_recovery=None):
    require(coordinate_recovery is None or split == 'B_confirm', 'Coordinate recovery is restricted to B_confirm')
    h = Path(history)
    dst = Path(dst)
    source = find_table(h / 'stage1/dynamics' / ('student_dynamics_panel_core_' + split))
    srcfp = fingerprint(source)
    splitfile = find_table(h / 'stage1/splits' / (split + '_users'))
    sf = read_table(splitfile)
    require('user_id' in sf, 'Split table missing user_id')
    uid_values = pd.to_numeric(sf['user_id'], errors='raise').to_numpy(dtype=np.int64)
    require(len(np.unique(uid_values)) == len(uid_values), 'Duplicate user in split manifest')
    allowed = np.sort(uid_values)
    if smoke_users is not None and len(allowed) > smoke_users:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 91, SPLITS.index(split)]))
        allowed = np.sort(rng.choice(allowed, smoke_users, replace=False))
    pieces = dst / '_pieces'
    pieces.mkdir()
    total = raw_selected = raw_all = 0
    fallback = {'E_pre': 0, 'q_primary': 0, 'B_pre': 0, 'G_pre': 0}
    for i, raw in enumerate(iter_table(source, p1.PHASE1_PANEL_COLUMNS, chunk_size, allow_missing=True)):
        raw_all += len(raw)
        require({'user_id', 'bundle_step_index'}.issubset(raw.columns), 'Panel lacks join keys')
        if smoke_users is not None:
            raw = raw[raw['user_id'].isin(allowed)].copy()
        else:
            rawids = pd.to_numeric(raw['user_id'], errors='coerce').dropna().to_numpy(dtype=np.int64)
            require(np.isin(rawids, allowed).all(), f'Raw {split} includes users not in its split manifest')
        raw_selected += len(raw)
        if raw.empty:
            continue
        num = lambda c: pd.to_numeric(raw[c], errors='coerce').to_numpy(dtype=float) if c in raw else np.full(len(raw), np.nan)
        e = num('response_evidence_mass_pre')
        q = num('bundle_n_questions') * num('answered_fraction_interval')
        b = num('activity_active_mass_pre') + num('activity_idle_mass_pre')
        off = 'activity_off_target_mass_pre' if 'activity_off_target_mass_pre' in raw else 'activity_non_aligned_mass_pre'
        g = num('activity_aligned_mass_pre') - num(off)
        fallback['E_pre'] += int((~np.isfinite(e) | (e < 0)).sum())
        fallback['q_primary'] += int((~np.isfinite(q) | (q <= 0)).sum())
        fallback['B_pre'] += int((~np.isfinite(b) | (b <= 0)).sum())
        fallback['G_pre'] += int((~np.isfinite(g)).sum())
        d = p1.prepare_panel(raw, split, float(eta))
        if d.empty:
            continue
        F = d[list(FEATURES)].to_numpy(dtype=np.float64)
        Y = d[['target_M_next', 'target_Psi_next']].to_numpy(dtype=np.float64)
        K = d[['user_id', 'bundle_step_index']].to_numpy(dtype=np.int64)
        A = d[list(AUX)].to_numpy(dtype=np.float64)
        require(np.isfinite(F).all() and np.isfinite(Y).all(), 'Non-finite prepared predictor/target: no additional row exclusion allowed')
        np.savez(pieces / f'{i:06d}.npz', F=F, Y=Y, K=K, A=A)
        total += len(d)
        if i % 10 == 0:
            log(f'{split}: scanned {raw_all:,} rows, retained {total:,}')
    require(total > 0, f'{split} has no original-S2-valid rows')
    specifications = {'features': ('F', len(FEATURES), np.float64), 'target': ('Y', 2, np.float64), 'keys': ('K', 2, np.int64), 'aux': ('A', len(AUX), np.float64)}
    unsorted = {name: open_memmap(dst / (name + '_unsorted.npy'), mode='w+', dtype=dt, shape=(total, dim)) for name, (_, dim, dt) in specifications.items()}
    start = 0
    for p in sorted(pieces.glob('*.npz')):
        with np.load(p, allow_pickle=False) as a:
            n = len(a['K'])
            for name, (key, _, _) in specifications.items():
                unsorted[name][start:start + n] = a[key]
            start += n
    keys = unsorted['keys']
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    for name, (_, dim, dt) in specifications.items():
        target = open_memmap(dst / (name + '.npy'), mode='w+', dtype=dt, shape=(total, dim))
        for a in range(0, total, chunk_size):
            target[a:a + chunk_size] = unsorted[name][order[a:a + chunk_size]]
        target.flush()
        del target
    unsorted.clear()
    del keys
    for p in dst.glob('*_unsorted.npy'):
        p.unlink()
    shutil.rmtree(pieces)
    recovery = None
    if coordinate_recovery is not None:
        recovery = recover_b_confirm_coordinates(h, source, dst, coordinate_recovery, chunk_size, allowed if smoke_users is not None else None)
    K = np.load(dst / 'keys.npy', mmap_mode='r')
    F = np.load(dst / 'features.npy', mmap_mode='r')
    require(not ((K[1:, 0] == K[:-1, 0]) & (K[1:, 1] == K[:-1, 1])).any(), 'Duplicate valid (user_id,bundle_step_index)')
    users, ui, counts = np.unique(K[:, 0], return_inverse=True, return_counts=True)
    np.save(dst / 'users.npy', users)
    np.save(dst / 'user_index.npy', ui.astype(np.int64))
    np.save(dst / 'weights.npy', (1.0 / counts[ui]).astype(np.float64))
    cells = open_memmap(dst / 'cells.npy', mode='w+', dtype=np.int64, shape=(total,))
    for a in range(0, total, chunk_size):
        cells[a:a + chunk_size] = grid_cells(F[a:a + chunk_size, :2])
    cells.flush()
    del cells
    metadata = {'split': split, 'raw_total': raw_all, 'raw_selected': raw_selected, 'valid_rows': total, 'users': len(users), 'dropped_by_original_prepare_panel': raw_selected - total, 'source': srcfp, 'split_source': fingerprint(splitfile), 'eta': eta, 'selected_smoke_users': allowed.tolist() if smoke_users else None, 'fallback_counts_before_original_validity_filter': fallback, 'keys_sha256': sha(dst / 'keys.npy'), 'features': FEATURES, 'aux': AUX, 'filter': 'frozen prepare_panel only; no model-dependent row selection', 'weight': '1 / number of original-S2-valid rows per learner'}
    if recovery is not None:
        metadata['coordinate_recovery'] = recovery
    save_json(metadata, dst / 'metadata.json')

def load_partition(history, centers_path=None):
    p = Path(history) / 'stage1/dynamics/fixed_k6_mesostates'
    m = load_json(p / 'fixed_k6_model_metadata.json')
    c = (pd.read_csv(centers_path, float_precision='round_trip') if centers_path is not None and str(centers_path).endswith('.csv') else read_table(centers_path or p / 'fixed_k6_centers')).sort_values('macrostate')
    require(np.array_equal(c['macrostate'].to_numpy(), np.arange(6)), 'Fixed K6 label order differs')
    mean = np.asarray(m['scaler_mean'], float)
    scale = np.asarray(m['scaler_scale'], float)
    cent = c[['center_M_standardized', 'center_Psi_standardized']].to_numpy(dtype=float)
    raw = c[['center_M', 'center_Psi']].to_numpy(dtype=float)
    assert_close((raw - mean) / scale, cent, 'Fixed K6 standardized centers', atol=1e-08)
    return {'mean': mean, 'scale': scale, 'centers': cent, 'raw_centers': raw}

def assign_states(X, part):
    X = np.asarray(X, dtype=float)
    require(np.isfinite(X).all(), 'Cannot assign non-finite state')
    z = (X - part['mean']) / part['scale']
    return np.argmin(((z[:, None, :] - part['centers'][None, :, :]) ** 2).sum(axis=2), axis=1).astype(np.int64)

def assert_disjoint(datasets):
    users = [np.asarray(d.users) for d in datasets]
    for i in range(len(users)):
        for j in range(i):
            require(not np.intersect1d(users[i], users[j]).size, 'User overlap across splits')

def match_archive(dataset, pred_path, history, chunk_size=200000, smoke=False):
    ds = dataset
    pred = np.load(pred_path, mmap_mode='r')
    path = find_table(Path(history) / 'stage2_phase3_confirm/tables/phase3_B_confirm_full_predictions')
    columns = ['user_id', 'bundle_step_index', 'M', 'Psi', 'target_M_next', 'target_Psi_next', 'pred_next_M', 'pred_next_Psi']
    seen = np.zeros(len(ds), dtype=bool)
    n_archive = 0
    maxerr = np.zeros(3)
    for t in iter_table(path, columns, chunk_size):
        n_archive += len(t)
        k = t[['user_id', 'bundle_step_index']].to_numpy(dtype=np.int64)
        idx, ok = key_positions(ds.keys, k)
        if smoke:
            selected = np.isin(k[:, 0], ds.users)
            require(ok[selected].all(), 'Archive has unexpected rows for smoke users')
            t = t.loc[ok]
            idx = idx[ok]
        else:
            require(ok.all(), 'Archived predictions contain unknown valid keys')
        if not len(idx):
            continue
        require(len(np.unique(idx)) == len(idx) and (not seen[idx].any()), 'Duplicate archived prediction key')
        seen[idx] = True
        maxerr[0] = max(maxerr[0], assert_close(ds.F[idx, :2], t[['M', 'Psi']].to_numpy(), 'M0 observed origins'))
        maxerr[1] = max(maxerr[1], assert_close(ds.Y[idx], t[['target_M_next', 'target_Psi_next']].to_numpy(), 'M0 targets'))
        maxerr[2] = max(maxerr[2], assert_close(pred[idx, 0, :], t[['pred_next_M', 'pred_next_Psi']].to_numpy(), 'M0 frozen predictions'))
    require(seen.all(), 'Archive is missing prepared confirmation predictions')
    if not smoke:
        require(n_archive == len(ds), 'Full archive row count mismatch')
    return {'status': 'passed', 'rows_compared': int(seen.sum()), 'archive_rows': n_archive, 'max_abs_origin_target_prediction_error': maxerr.tolist(), 'source': fingerprint(path), 'scope': 'selected smoke users' if smoke else 'all original S2 valid confirmation rows'}
