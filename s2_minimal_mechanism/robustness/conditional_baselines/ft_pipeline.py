from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import gc
import json
import shutil
import time
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy.ndimage import distance_transform_edt
from ft_common import *
from ft_data import *
from ft_models import *
from ft_statistics import *

def stage_dataset(ctx, split):
    recovery = None
    freeze_input = None
    if split == 'B_confirm':
        modelroot = ctx.out / 'models'
        verify_stage_receipt(modelroot, run_hash=ctx.run_hash)
        frozen = load_json(modelroot / 'FIT_FROZEN.json')
        models = load_json(modelroot / 'final_models.json')
        require(frozen.get('ready_for_B_evaluation') is True and frozen.get('protocol_code_hash') == ctx.run_hash and (frozen.get('model_payload_sha256') == sha(modelroot / 'final_models.json')) and (frozen.get('model_hash') == objhash(models)), 'B preparation requires a verified frozen fit')
        freeze_input = {'receipt_sha256': receipt_hash(modelroot), 'fit_frozen_sha256': sha(modelroot / 'FIT_FROZEN.json')}
        if not getattr(ctx, 'synthetic_fixture', False):
            recovery = load_json(Path(__file__).with_name('source_contract.json')).get('b_confirm_coordinate_recovery')
            require(isinstance(recovery, dict), 'B preparation requires a registered coordinate recovery certificate')
    source = find_table(ctx.history / 'stage1/dynamics' / ('student_dynamics_panel_core_' + split))
    splitfile = find_table(ctx.history / 'stage1/splits' / (split + '_users'))
    sourcefp = fingerprint(source)
    splitfp = fingerprint(splitfile)
    eta = (ctx.final_cal if split == 'B_confirm' else ctx.train_cal)['eta']
    inputs = {'panel': sourcefp, 'split_table': splitfp, 'eta': eta, 'original_preparer_sha256': ctx.provenance['artifacts'][2]['sha256']}
    if split == 'B_confirm':
        inputs.update(frozen_fit=freeze_input, coordinate_recovery=recovery, synthetic_fixture=getattr(ctx, 'synthetic_fixture', False))
    path = atomic_stage(ctx.out, 'prepared/' + split, ctx.run_hash, inputs, lambda dst: prepare_dataset(ctx.p1, ctx.history, split, eta, dst, ctx.chunk_size, ctx.smoke_users, coordinate_recovery=recovery))
    d = Dataset(path)
    require(d.meta['source']['sha256'] == sourcefp['sha256'], 'Cache/source fingerprint mismatch')
    return d

def c3_fit(datasets, params, cal, block_size):
    count = np.zeros(NCELL)
    W = np.zeros(NCELL)
    errors = np.zeros((NCELL, 2))
    bases = np.zeros((NCELL, 2, 3))
    max_affine = max_psi = 0.0
    probe = np.array([0.271, -0.413, 0.719])
    for ds in datasets:
        for s in ds.blocks(block_size):
            F = np.asarray(ds.F[s])
            y = np.asarray(ds.Y[s])
            w = np.asarray(ds.w[s])
            c = np.asarray(ds.cells[s])
            base, A = c3_basis(F, params, cal)
            max_affine = max(max_affine, assert_close(base + np.einsum('nij,j->ni', A, probe), constant_predict(F, probe, params, cal), 'C3 exact affine representation', atol=1e-11, rtol=1e-10))
            count += np.bincount(c, minlength=NCELL)
            W += np.bincount(c, weights=w, minlength=NCELL)
            for axis in range(2):
                errors[:, axis] += np.bincount(c, weights=w * (base[:, axis] - y[:, axis]), minlength=NCELL)
                for term in range(3):
                    bases[:, axis, term] += np.bincount(c, weights=w * A[:, axis, term], minlength=NCELL)
    S = count >= MIN_COUNT
    require(S.any(), 'C3 not estimable: no fitted-origin cell has >=30 transitions; threshold is not changed')
    norm = np.sqrt(W[S] / W[S].sum())
    A = (bases[S] / W[S, None, None] * norm[:, None, None]).reshape(-1, 3)
    b = (-errors[S] / W[S, None] * norm[:, None]).reshape(-1)
    coeff, audit = bounded_min_norm(A, b)
    embedded = np.array([0.0, np.tanh(params['phi0']), np.tanh(params['phi0'] + params['deltaS'])])
    fit_res = (A @ coeff - b).reshape(-1, 2)
    embed_res = (A @ embedded - b).reshape(-1, 2)
    audit['optimized_Psi_field_SSE'] = float((fit_res[:, 1] ** 2).sum())
    audit['embedded_original_Psi_field_SSE'] = float((embed_res[:, 1] ** 2).sum())
    require(audit['optimized_Psi_field_SSE'] <= audit['embedded_original_Psi_field_SSE'] + 1e-11, 'C3 optimizer worse than its feasible original-Psi embedding on fitted data')
    audit.update(max_affine_error=max_affine, supported_cells=int(S.sum()), supported_mass=float(W[S].sum()), coefficients=coeff.tolist(), bounds=[-1, 1], normalization='sum occupied user mass on fitted support', objective='population-field weighted squared error', calibration=cal)
    return {'coefficients': coeff.tolist(), 'fit_audit': audit}

def eval_callable(ctx, dataset, names, predictors, split):
    e = Evaluator(names, ctx.partition)
    for s in dataset.blocks(ctx.chunk_size):
        F = np.asarray(dataset.F[s])
        K = np.asarray(dataset.keys[s])
        P = np.stack([fn(F, K) for fn in predictors], axis=1)
        e.add(F, np.asarray(dataset.Y[s]), P, np.asarray(dataset.w[s]))
    return e

def fit_models(ctx, train, val):
    assert_disjoint([train, val])
    inputs = {'train': receipt_hash(train.path), 'validation': receipt_hash(val.path), 'train_calibration': ctx.train_cal, 'final_calibration': ctx.final_cal, 'alphas': ALPHAS}

    def run(dst):
        log('M1: fit development C3 using A_train only')
        c3dev = c3_fit([train], ctx.params, ctx.train_cal, ctx.chunk_size)
        log('M1: fit development ARX sufficient moments')
        arx0 = ridge_fit([train], 0.0, ctx.chunk_size)
        candidates = [ridge_change_alpha(arx0, a) for a in ALPHAS]
        ev = eval_callable(ctx, val, [str(a) for a in ALPHAS], [lambda F, K, m=m: ridge_predict(F, m) for m in candidates], 'A_val_development')
        scores = [ev.primary(ctx.p1, i) for i in range(len(ALPHAS))]
        scoretab = pd.DataFrame([{'alpha': a, **scores[i]} for i, a in enumerate(ALPHAS)])
        require(np.isfinite(scoretab.objective_primary_score).all(), 'Non-finite alpha selection score')
        minimum = float(scoretab.objective_primary_score.min())
        choices = [i for i, x in enumerate(scores) if x['objective_primary_score'] <= minimum + 1e-12]
        best = max(choices, key=lambda i: ALPHAS[i])
        alpha = ALPHAS[best]
        scoretab['selected'] = np.arange(len(ALPHAS)) == best
        write_csv(scoretab, dst / 'ARX_all_validation_candidates.csv')
        save_json({'C3': c3dev, 'ARX_candidates': candidates, 'selected_alpha': alpha, 'data_scope': 'A_train fitting; A_val selection only; no B_confirm read', 'selection_score': 'five original primary components only; no fabricated phase or coverage predictions'}, dst / 'development_models.json')
        dev = eval_callable(ctx, val, ['C3', 'ARX'], [lambda F, K: constant_predict(F, c3dev['coefficients'], ctx.params, ctx.train_cal), lambda F, K: ridge_predict(F, candidates[best])], 'A_val_development')
        mt, tr, gr = dev.tables(ctx.p1, 'A_val_development')
        write_csv(mt, dst / 'development_metrics.csv')
        write_csv(tr, dst / 'development_transitions.csv')
        log('M1: final pooled development refit, without changing the original four-term model')
        c3final = c3_fit([train, val], ctx.params, ctx.final_cal, ctx.chunk_size)
        arxfinal = ridge_fit([train, val], alpha, ctx.chunk_size)
        result = {'C3': c3final, 'ARX': arxfinal, 'frozen_original_parameters': ctx.params, 'frozen_original_calibration': ctx.final_cal, 'alpha_selected_without_B': alpha, 'fit_scope': ['A_train', 'A_val'], 'B_confirm_used_for_fitting': False, 'analysis_label': 'post hoc supplementary; B is fit-held-out, not a new blinded cohort', 'development_inputs': inputs}
        save_json(result, dst / 'final_models.json')
        save_json({'model_payload_sha256': sha(dst / 'final_models.json'), 'model_hash': objhash(result), 'protocol_code_hash': ctx.run_hash, 'ready_for_B_evaluation': True}, dst / 'FIT_FROZEN.json')
    path = atomic_stage(ctx.out, 'models', ctx.run_hash, inputs, run)
    m = load_json(path / 'final_models.json')
    frozen = load_json(path / 'FIT_FROZEN.json')
    require(frozen['model_payload_sha256'] == sha(path / 'final_models.json'), 'New model freeze hash mismatch')
    return (path, m)

def write_prediction_export(ds, P, path, blocksize):
    cols = ['user_id', 'bundle_step_index', 'M', 'Psi', 'target_M_next', 'target_Psi_next']
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pa = None
    target = Path(path) / ('row_predictions.parquet' if pa else 'row_predictions.csv.gz')

    def frame(s):
        x = np.column_stack([ds.keys[s], ds.F[s, :2], ds.Y[s], np.asarray(P[s]).reshape(-1, len(MODELS) * 2)])
        names = cols + [f'{name}_next_{axis}' for name in MODELS for axis in ('M', 'Psi')]
        t = pd.DataFrame(x, columns=names)
        t['user_id'] = np.asarray(ds.keys[s, 0])
        t['bundle_step_index'] = np.asarray(ds.keys[s, 1])
        return t
    if pa:
        writer = None
        try:
            for s in ds.blocks(blocksize):
                table = pa.Table.from_pandas(frame(s), preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(target, table.schema, compression='snappy')
                writer.write_table(table)
        finally:
            if writer:
                writer.close()
    else:
        import gzip
        with gzip.open(target, 'wt', encoding='utf-8', newline='') as f:
            for i, s in enumerate(ds.blocks(blocksize)):
                frame(s).to_csv(f, index=False, header=i == 0)
    return target

def m0_aggregate_audit(ctx, ds, evaluator):
    root = ctx.history / 'stage2_phase3_confirm'
    checks = []
    reference = read_table(root / 'tables/phase3_B_confirm_structural_alignment_metrics')
    require(len(reference) == 1, 'Expected one formal Phase-3 metric record')
    metrics = evaluator.primary(ctx.p1, 0)
    for k in ('objective_primary_score', 'one_step_mse_main_norm', 'one_step_rmse_M', 'one_step_rmse_Psi', 'occupancy_js_MR_PsiA', 'drift_vector_corr_MR_PsiA', 'drift_local_rmse_loss_MR_PsiA', 'drift_direction_loss_MR_PsiA', 'drift_magnitude_loss_MR_PsiA'):
        require(k in reference, f'Original metric missing: {k}')
        err = assert_close([metrics[k]], [float(reference[k].iloc[0])], f'M0 original metric {k}', atol=2e-08, rtol=2e-07)
        checks.append({'item': k, 'current': metrics[k], 'archived': float(reference[k].iloc[0]), 'abs_error': err})
    g = read_table(root / 'tables/phase3_B_confirm_field_grid').sort_values(['M_center', 'Psi_center'])
    require(len(g) == NCELL, 'Original field table not 40x40')
    emp, model = evaluator.fields()
    for name, values in [('empirical_drift_M', emp[:, 0]), ('empirical_drift_Psi', emp[:, 1]), ('mechanism_drift_M', model[0, :, 0]), ('mechanism_drift_Psi', model[0, :, 1])]:
        err = assert_close(values, g[name].to_numpy(), f'M0 {name}')
        checks.append({'item': name, 'abs_error': err})
    require(np.array_equal(g['common_supported'].to_numpy(dtype=bool), evaluator.count >= 30), 'Original field support mismatch')
    tr = root / 'figures_publication_minimal_mechanism/tables'
    for name, j in [('empirical', 0), ('mechanism', 1)]:
        orig = read_table(tr / f'B_confirm_{name}_mesostate_transition_counts').to_numpy(dtype=float)
        err = assert_close(evaluator.transitions[j], orig, f'M0 raw interval transition counts {name}', atol=0, rtol=0)
        checks.append({'item': f'{name}_transition_counts', 'abs_error': err})
    return checks

def phase3_provenance(ctx):
    meta = ctx.history / 'stage2_phase3_confirm/metadata'
    path = meta / 'phase3_confirmation_manifest.json'
    m = load_json(path)
    side = meta / 'phase3_confirmation_manifest.sha256.json'
    require(load_json(side).get('manifest_sha256') == sha(path), 'Phase-3 manifest checksum mismatch')
    require(m.get('confirm_split') == 'B_confirm' and m.get('confirmation_status') == 'completed_output_only', 'Wrong original confirmation status')
    require(m.get('frozen_parameter_hash') == objhash(ctx.params), 'Phase-3 parameter identity mismatch')
    require(m.get('frozen_calibration_hash') == objhash(ctx.final_cal), 'Phase-3 calibration identity mismatch')
    for key in ('parameter_search_opened', 'calibration_reestimated', 'mechanism_family_reselected', 'mechanism_parameters_refit', 'kmeans_refit', 'macrostate_k_selected', 'region_redefinition', 'B_confirm_used_for_update'):
        require(m.get('guardrails', {}).get(key) is False, f'Phase-3 no-update guardrail failed: {key}')
    require(m.get('frozen_manifest_checksum', {}).get('manifest_sha256') == sha(ctx.history / 'stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json'), 'Phase-3 points to a different Phase-2 freeze')
    t = ctx.history / 'stage2_phase3_confirm'
    required = [path, side, meta / 'phase3_no_update_audit.json', find_table(t / 'tables/phase3_B_confirm_structural_alignment_metrics'), find_table(t / 'tables/phase3_B_confirm_field_grid')]
    for name in ('empirical', 'mechanism'):
        required.append(find_table(t / f'figures_publication_minimal_mechanism/tables/B_confirm_{name}_mesostate_transition_counts'))
    return [fingerprint(p) for p in required]

def evaluation(ctx, ds, modelpath, models):
    inputs = {'dataset': receipt_hash(ds.path), 'models': receipt_hash(modelpath)}
    archive = find_table(ctx.history / 'stage2_phase3_confirm/tables/phase3_B_confirm_full_predictions')
    inputs['saved_predictions'] = fingerprint(archive)
    inputs['original_confirmation_provenance'] = phase3_provenance(ctx)

    def run(dst):
        n = len(ds)
        P = open_memmap(dst / 'predictions.npy', mode='w+', dtype=np.float64, shape=(n, len(MODELS), 2))
        C = open_memmap(dst / 'components.npy', mode='w+', dtype=np.float64, shape=(n, len(COMPONENTS)))
        phase = open_memmap(dst / 'original_stage_contributions.npy', mode='w+', dtype=np.float64, shape=(n, 3))
        e = Evaluator(MODELS, ctx.partition)
        clip = np.zeros(2, dtype=np.int64)
        rawmin = np.full(2, np.inf)
        rawmax = np.full(2, -np.inf)
        maxpsi = 0.0
        affine = 0.0
        addback = 0.0
        embedded = [0, np.tanh(ctx.params['phi0']), np.tanh(ctx.params['phi0'] + ctx.params['deltaS'])]
        for s in ds.blocks(ctx.chunk_size):
            F = np.asarray(ds.F[s])
            K = np.asarray(ds.keys[s])
            four, ph = four_predict(ctx.p1, F, K, ctx.params, ctx.final_cal)
            c3 = constant_predict(F, models['C3']['coefficients'], ctx.params, ctx.final_cal)
            arx, raw = ridge_predict(F, models['ARX'], True)
            values = np.stack([four, c3, arx, F[:, :2], constant_predict(F, [0, 0, 0], ctx.params, ctx.final_cal)], axis=1)
            P[s] = values
            phase[s] = ph
            C[s] = decompose(F, four, ctx.params, ctx.final_cal)
            maxpsi = max(maxpsi, assert_close(four[:, 1], constant_predict(F, embedded, ctx.params, ctx.final_cal)[:, 1], 'C3 contains original Psi function', atol=1e-11, rtol=1e-10))
            e.add(F, np.asarray(ds.Y[s]), values, np.asarray(ds.w[s]))
            clip += (np.abs(raw) > 1).sum(axis=0)
            rawmin = np.minimum(rawmin, raw.min(axis=0))
            rawmax = np.maximum(rawmax, raw.max(axis=0))
        P.flush()
        C.flush()
        phase.flush()
        audit = match_archive(ds, dst / 'predictions.npy', ctx.history, ctx.chunk_size, smoke=ctx.smoke)
        save_json(audit, dst / 'M0_row_identity.json')
        if not ctx.smoke:
            write_csv(pd.DataFrame(m0_aggregate_audit(ctx, ds, e)), dst / 'M0_aggregate_identity.csv')
        else:
            save_json({'status': 'not_applicable_to_subsample', 'full_M0_aggregate_audit_required_in_formal_run': True}, dst / 'M0_aggregate_smoke_note.json')
        mt, tr, gr = e.tables(ctx.p1, 'B_confirm')
        write_csv(mt, dst / 'model_metrics.csv')
        write_csv(tr, dst / 'statewise_transitions.csv')
        write_csv(gr, dst / 'field_grid.csv.gz')
        main = mt.set_index('model')
        contrasts = []
        for baseline in ('C3', 'ARX'):
            dd = main.loc[baseline, ['D2_M', 'D2_Psi']].to_numpy(float) - main.loc['four_term', ['D2_M', 'D2_Psi']].to_numpy(float)
            contrasts.append({'baseline': baseline, 'Delta': float(dd.sum()), 'Delta_M': dd[0], 'Delta_Psi': dd[1], 'relative_skill': 1 - main.loc['four_term', 'D2'] / main.loc[baseline, 'D2'] if main.loc[baseline, 'D2'] > EPS else np.nan})
        write_csv(pd.DataFrame(contrasts), dst / 'primary_point_contrasts.csv')
        save_json({'all_predictions_finite': True, 'same_rows_all_models': n, 'models': MODELS, 'C3_Psi_embedding_max_error': maxpsi, 'ARX_clipped_fraction': (clip / n).tolist(), 'ARX_preclip_min': rawmin, 'ARX_preclip_max': rawmax, 'frozen_model_hash_unchanged': sha(modelpath / 'final_models.json') == load_json(modelpath / 'FIT_FROZEN.json')['model_payload_sha256']}, dst / 'prediction_audit.json')
        write_prediction_export(ds, P, dst, ctx.chunk_size)
        corep = ctx.history / 'stage1/dynamics/candidate_regions/MR_PsiA'
        core = np.load(corep / 'A_train_primary_convergence_core_mask.npy').astype(bool)
        require(core.shape == (40, 40) and core.any(), 'Invalid original training core')
        radius = float(load_json(corep / 'training_convergence_thresholds.json')['shell_radius'])
        distance = distance_transform_edt(~core, sampling=(0.05, 0.05))
        support = (e.count >= 30).reshape(40, 40)
        regions = {'all': np.ones(NCELL, bool), 'supported': support.ravel(), 'core': (core & support).ravel(), 'shell': (support & ~core & (distance > 0) & (distance <= radius)).ravel()}
        rows = []
        gridcomp = np.zeros((NCELL, len(COMPONENTS)))
        for region, mask in regions.items():
            num = np.zeros(len(FEATURES) + len(COMPONENTS))
            num_i = num.copy()
            mass = nr = 0
            for s in ds.blocks(ctx.chunk_size):
                ok = mask[ds.cells[s]]
                if not ok.any():
                    continue
                vals = np.column_stack([ds.F[s], C[s]])[ok]
                ww = np.asarray(ds.w[s])[ok]
                num += (ww[:, None] * vals).sum(axis=0)
                num_i += vals.sum(axis=0)
                mass += ww.sum()
                nr += ok.sum()
            for j, name in enumerate(FEATURES + COMPONENTS):
                rows.append({'region': region, 'variable': name, 'rows': nr, 'learner_mass': mass, 'user_weighted_mean': num[j] / mass if mass > 0 else np.nan, 'interval_mean': num_i[j] / nr if nr > 0 else np.nan, 'interpretation': 'conditional descriptive composition, not a causal contribution percentage'})
        for s in ds.blocks(ctx.chunk_size):
            for j in range(len(COMPONENTS)):
                gridcomp[:, j] += np.bincount(ds.cells[s], weights=np.asarray(ds.w[s]) * C[s, j], minlength=NCELL)
        gridcomp = np.divide(gridcomp, e.mass[:, None], out=np.full_like(gridcomp, np.nan), where=e.mass[:, None] > 0)
        cg = pd.DataFrame(gridcomp, columns=COMPONENTS)
        cg.insert(0, 'cell', np.arange(NCELL))
        cg['mass'] = e.mass
        cg['supported'] = e.count >= 30
        write_csv(cg, dst / 'M2_field_components.csv')
        write_csv(pd.DataFrame(rows), dst / 'M2_core_shell_composition.csv')
        save_json({'component_names': COMPONENTS, 'stage_names': ['pred_delta_M_response', 'pred_delta_Psi_active', 'pred_delta_Psi_idle'], 'stage_contributions_distinct_from_algebraic_components': True, 'addback_checked_each_chunk': True, 'idle_in_signed_numerator': False, 'squared_error_improvement_is_not_additive_over_mechanism_terms': True}, dst / 'M2_decomposition_contract.json')
    path = atomic_stage(ctx.out, 'confirmation', ctx.run_hash, inputs, run)
    return path

def bootstrap_main(ctx, ds, evalpath):
    inputs = {'evaluation': receipt_hash(evalpath), 'replicates': ctx.nboot, 'seed': SEED}

    def run(dst):
        stats = LearnerStatistics(ds, np.load(evalpath / 'predictions.npy', mmap_mode='r'), MODELS, ctx.partition, ctx.chunk_size)
        point = stats.summarize(stats.point_vector, -1)[0]
        old = pd.read_csv(evalpath / 'model_metrics.csv').set_index('model')
        for name in MODELS:
            assert_close([point['D2_' + name]], [old.loc[name, 'D2']], f'Sparse/row-level D2 {name}', atol=1e-12, rtol=1e-09)
        run_bootstrap(stats, dst, ctx.nboot, ctx.workers)
    return atomic_stage(ctx.out, 'inference', ctx.run_hash, inputs, run)

def subset_dataset(ds, idx):
    idx = np.asarray(idx, dtype=np.int64)
    require(np.all(np.diff(idx) > 0), 'M3 subset must be unique and ordered')
    obj = SimpleNamespace(F=np.asarray(ds.F[idx]), Y=np.asarray(ds.Y[idx]), keys=np.asarray(ds.keys[idx]), cells=np.asarray(ds.cells[idx]))
    obj.users, obj.ui, counts = np.unique(obj.keys[:, 0], return_inverse=True, return_counts=True)
    obj.w = 1.0 / counts[obj.ui]

    class CommonDataset:

        def __init__(self, data):
            self.__dict__.update(data.__dict__)

        def __len__(self):
            return len(self.F)

        def blocks(self, size=200000):
            for a in range(0, len(self), size):
                yield slice(a, min(a + size, len(self)))
    return CommonDataset(obj)

def read_neural_join(ctx, ds, path):
    columns = ['user_id', 'bundle_step_index', 'M', 'Psi', 'target_M_next', 'target_Psi_next', 'pred_next_M', 'pred_next_Psi']
    indices = []
    preds = []
    seen = set()
    total = 0
    max_origin = max_target = 0.0
    for t in iter_table(path, columns, ctx.chunk_size):
        k = t[['user_id', 'bundle_step_index']].to_numpy(dtype=np.int64)
        total += len(k)
        q = structured_keys(k)
        require(len(np.unique(q)) == len(q), 'Duplicate neural keys within a chunk')
        keytuples = set(map(tuple, k.tolist()))
        require(not seen & keytuples, 'Duplicate neural keys across chunks')
        seen.update(keytuples)
        idx, ok = key_positions(ds.keys, k)
        if not ok.any():
            continue
        t = t.loc[ok]
        idx = idx[ok]
        max_origin = max(max_origin, assert_close(ds.F[idx, :2], t[['M', 'Psi']].to_numpy(), 'M3 common origins', atol=2e-07, rtol=2e-07))
        max_target = max(max_target, assert_close(ds.Y[idx], t[['target_M_next', 'target_Psi_next']].to_numpy(), 'M3 common targets', atol=2e-07, rtol=2e-07))
        p = t[['pred_next_M', 'pred_next_Psi']].to_numpy(float)
        require(np.isfinite(p).all() and ((p >= -1 - 1e-07) & (p <= 1 + 1e-07)).all(), 'Invalid frozen neural prediction')
        indices.append(idx)
        preds.append(p)
    require(bool(indices), 'No M3 overlapping rows')
    idx = np.concatenate(indices)
    pred = np.concatenate(preds)
    order = np.argsort(idx)
    idx = idx[order]
    pred = pred[order]
    require(not (idx[1:] == idx[:-1]).any(), 'M3 join not one-to-one')
    return (idx, pred, {'source_rows': total, 'joined_rows': len(idx), 'M1_rows': len(ds), 'M1_row_fraction': len(idx) / len(ds), 'max_origin_error': max_origin, 'max_target_error': max_target, 'source': fingerprint(path), 'same_information_budget_claimed': False})

def neural_paths(history):
    h = Path(history)
    result = {42: find_table(h / 'stage4_event_ssl/evaluation_predictive_state/predictions/stage4_event_ssl_predictions_B_confirm')}
    for seed in (2026, 666, 606, 37, 4669):
        p = find_table(h / f'random_seed_experiments/seed_{seed}/stage4_event_ssl/evaluation_predictive_state/predictions/stage4_event_ssl_predictions_B_confirm', required=False)
        result[seed] = p
    return result

def neural_provenance(ctx, seed, path):
    eroot = path.parent.parent
    mp = eroot / 'metadata/stage4_event_ssl_evaluation_manifest.json'
    m = load_json(mp)
    require(m.get('model_kind') == 'predictive_state' and m.get('primary_coordinates') == ['M', 'Psi'], 'Wrong frozen neural model kind')
    g = m.get('guardrails', {})
    for k in ('B_confirm_used_for_update', 'kmeans_refit', 'macrostate_k_selected'):
        require(g.get(k) is False, f'Neural provenance guardrail failed: {k}')
    require(g.get('macrostate_k') == 6, 'Neural partition is not frozen K=6')
    require({'empirical_anchor', 'learned_plane'}.issubset(m.get('evaluation_views', {})), 'Neural evaluation gauges undocumented')
    record = m.get('macro_partition', {})
    require(record.get('verified') is True and record.get('fit_split') == 'A_train', 'Neural partition provenance unverified')
    for k in ('metadata_sha256', 'centers_sha256'):
        expected = ctx.provenance['k6_audit'][k]
        recovery = ctx.provenance.get('k6_recovery')
        if k == 'centers_sha256' and recovery:
            expected = recovery['legacy_centers_sha256']
        require(record.get(k) == expected, f'Neural/source K6 mismatch: {k}')
    declared = Path(str(m.get('splits', {}).get('B_confirm', {}).get('prediction_path', '')))
    require(declared.name == path.name, 'Neural prediction filename differs from evaluation manifest')
    modelroot = eroot.parent / 'models/predictive_state'
    checkpoint = modelroot / 'best_model.pt'
    training = modelroot / 'training_manifest.json'
    require(checkpoint.is_file() and training.is_file(), 'Frozen checkpoint/training provenance is missing')
    require(sha(checkpoint) == m.get('checkpoint_sha256'), 'Frozen neural checkpoint checksum changed')
    cfg = load_json(training).get('config', {})
    require(cfg.get('seed') == seed and cfg.get('model_kind') == 'predictive_state', 'Neural seed/model metadata mismatch')
    return {'seed': seed, 'artifacts': [fingerprint(p) for p in (mp, checkpoint, training)], 'checkpoint_deserialized': False, 'evaluation_gauges': m['evaluation_views']}

def crossmodel(ctx, ds, evalpath, modelpath):
    paths = neural_paths(ctx.history)
    if not ctx.smoke:
        require(all((p is not None for p in paths.values())), 'A declared frozen neural seed file is missing; no silent six-seed truncation')
    inputs = {'M1': receipt_hash(evalpath), 'frozen_new_models': receipt_hash(modelpath), 'neural': {s: fingerprint(p) if p else None for s, p in paths.items()}, 'neural_provenance': {s: neural_provenance(ctx, s, p) if p else None for s, p in paths.items()}, 'replicates': ctx.nboot}

    def run(dst):
        base = np.load(evalpath / 'predictions.npy', mmap_mode='r')
        rows = []
        audits = []
        for seed, path in paths.items():
            if path is None:
                continue
            log(f'M3: frozen neural seed {seed}')
            idx, npred, audit = read_neural_join(ctx, ds, path)
            common = subset_dataset(ds, idx)
            p = np.stack([base[idx, 0], base[idx, 1], npred], axis=1)
            names = ('four_term', 'C3', 'SSL')
            ev = Evaluator(names, ctx.partition)
            for s in common.blocks(ctx.chunk_size):
                ev.add(common.F[s], common.Y[s], p[s], common.w[s])
            emp, fs = ev.fields()
            S = ev.count >= 30
            row = {'seed': seed, 'rows': len(common), 'users': len(common.users), **pair_diagnostics(emp, fs[0], fs[1], fs[2], ev.mass, S)}
            rows.append(row)
            audits.append({'seed': seed, **audit})
            if seed == 42:
                sd = dst / 'seed42_uncertainty'
                sd.mkdir()
                stats = LearnerStatistics(common, p, names, ctx.partition, ctx.chunk_size)
                run_bootstrap(stats, sd, ctx.nboot, ctx.workers, include_pairs=True)
                mt, tr, grid = ev.tables(ctx.p1, 'B_confirm_M3_common_seed42')
                write_csv(mt, dst / 'seed42_same_sample_metrics.csv')
                write_csv(grid, dst / 'seed42_common_field_grid.csv.gz')
                np.save(dst / 'seed42_M1_row_indices.npy', idx)
            del common, p
            gc.collect()
        write_csv(pd.DataFrame(rows), dst / 'C3_referenced_cross_model_by_seed.csv')
        save_json(audits, dst / 'join_audits.json')
        save_json({'no_model_retraining': True, 'C3_coefficients_refitted_on_join': False, 'seed_variation_separate_from_user_bootstrap': True, 'common_subtraction_not_mechanistic_identity': True}, dst / 'M3_scope.json')
    return atomic_stage(ctx.out, 'cross_model', ctx.run_hash, inputs, run)
