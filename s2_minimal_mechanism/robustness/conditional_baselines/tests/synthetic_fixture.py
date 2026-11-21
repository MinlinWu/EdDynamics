from pathlib import Path
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft_common import save_json, sha, objhash, PARAMS
from ft_data import load_partition, assign_states

def load_p1(source, name='fixture_original_p1'):
    spec = importlib.util.spec_from_file_location(name, str(source))
    p1 = importlib.util.module_from_spec(spec)
    sys.modules[name] = p1
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(p1)
    finally:
        sys.dont_write_bytecode = prev
    p1.CONFIG_USE_NUMBA = False
    return p1

def build_fixture(root, source, users=48, steps=64):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p1 = load_p1(source, 'fixture_original_' + sha(source)[:8])
    params = {'theta0': -0.17, 'thetaM': 0.64, 'thetaPsi': 0.0, 'thetaMPsi': 0.0, 'phi0': -1.35, 'deltaS': 6.0, 'phiPsi': 0.0, 'lambdaR': 0.46, 'lambdaA': 1.1, 'lambdaI': 0.85}
    cal = {'eta': 20.0, 'tau_response_days': 10.0, 'tau_activity_days': 10.0, 'residual_mass_per_answer': 0.3865, 'lambda_E': 15.0, 'response_signed_gain': 1.0, 'alignment_signed_gain': 0.9537, 'sigma_U0': 0.02, 'sigma_Psi0': 0.01}
    kdir = root / 'stage1/dynamics/fixed_k6_mesostates'
    kdir.mkdir(parents=True)
    rawcent = np.array([[-0.8, -0.4], [-0.45, 0.4], [-0.1, -0.5], [0.2, 0.3], [0.5, -0.5], [0.85, 0.2]])
    meta = {'coordinate': 'MR_PsiA', 'macrostate_k': 6, 'macrostate_k_rule': 'fixed a priori', 'features': ['M_response_prebalanced_pre', 'activity_alignment_order_Psi_pre'], 'fit_split': 'A_train', 'user_balanced_sampling': True, 'user_balanced_kmeans_fit': True, 'kmeans_n_init': 20, 'fit_max_rows': 500000, 'random_state': 42, 'scaler_mean': [0.0, 0.0], 'scaler_scale': [1.0, 1.0], 'raw_to_ordered_label': {str(k): k for k in range(6)}}
    save_json(meta, kdir / 'fixed_k6_model_metadata.json')
    pd.DataFrame({'macrostate': range(6), 'center_M': rawcent[:, 0], 'center_Psi': rawcent[:, 1], 'center_M_standardized': rawcent[:, 0], 'center_Psi_standardized': rawcent[:, 1]}).to_csv(kdir / 'fixed_k6_centers.csv', index=False)
    reg = root / 'stage1/dynamics/candidate_regions/MR_PsiA'
    reg.mkdir(parents=True)
    core = np.zeros((40, 40), bool)
    core[18:23, 8:14] = True
    np.save(reg / 'A_train_primary_convergence_core_mask.npy', core)
    save_json({'shell_radius': 0.25}, reg / 'training_convergence_thresholds.json')
    saved = {}
    for si, split in enumerate(('A_train', 'A_val', 'B_confirm')):
        rng = np.random.default_rng(100 + si)
        n = users * steps
        uid = np.repeat(np.arange((si + 1) * 1000, (si + 1) * 1000 + users), steps)
        st = np.tile(np.arange(steps), users)
        mc = np.array([-0.775, -0.325, 0.125, 0.575])
        pc = np.array([-0.725, -0.375, 0.075, 0.525])
        M = mc[(st + uid) % 4] + rng.uniform(-0.007, 0.007, n)
        P = pc[(st // 4 + uid) % 4] + rng.uniform(-0.007, 0.007, n)
        E = rng.uniform(2, 40, n)
        B = rng.uniform(2, 25, n)
        G = P * B
        q = rng.integers(1, 5, n).astype(float)
        AR = rng.uniform(0.1, 1, n)
        AS = rng.uniform(0, 0.6, n)
        AR0 = rng.uniform(0, 0.2, n)
        AS0 = rng.uniform(0, 0.2, n)
        I = rng.uniform(0, 0.9, n)
        gap = rng.uniform(0, 3, n)
        d = pd.DataFrame({'user_id': uid, 'bundle_step_index': st, 'M_response_prebalanced_pre': M, 'activity_alignment_order_Psi_pre': P, 'response_evidence_mass_pre': E, 'response_evidence_maturity_V_pre': 1 - np.exp(-E / 20), 'next_gap_days': gap, 'has_next_submitted_bundle': (st < steps - 1).astype(int), 'bundle_n_questions': 4, 'answered_fraction_interval': q / 4, 'total_response_count_diagnostic': q, 'response_active_mass_interval': AR + AR0, 'support_active_total_interval': AS + AS0, 'response_aligned_mass_interval': AR * 0.4, 'response_off_target_mass_interval': AR * 0.6, 'response_neutral_mass_interval': AR0, 'support_aligned_mass_interval': AS * 0.8, 'support_off_target_mass_interval': AS * 0.2, 'support_neutral_mass_interval': AS0, 'idle_mass_interval': I, 'activity_active_mass_pre': 0.8 * B, 'activity_idle_mass_pre': 0.2 * B, 'activity_aligned_mass_pre': np.maximum(G, 0) + 0.05 * B, 'activity_off_target_mass_pre': np.maximum(-G, 0) + 0.05 * B, 'next_M_response_prebalanced': M, 'next_activity_alignment_order_Psi': P, 'next_response_evidence_mass': E, 'next_response_evidence_maturity_V': 1 - np.exp(-E / 20), 'next_activity_active_mass': 0.8 * B, 'next_activity_idle_mass': 0.2 * B, 'M_response_prebalanced_resp': M, 'response_evidence_maturity_V_resp': 1 - np.exp(-E / 20), 'activity_alignment_order_Psi_support': P, 'activity_alignment_order_Psi_post': P})
        ready = p1.prepare_panel(d, split, 20.0)
        cache = p1.make_metric_cache(ready, [], split)
        sim = p1.simulate_arrays(cache, params, p1.Calibration(**cal))
        active = st < steps - 1
        d.loc[active, 'next_M_response_prebalanced'] = np.clip(sim.pred_next_M + rng.normal(0, 0.015, len(ready)), -1, 1)
        d.loc[active, 'next_activity_alignment_order_Psi'] = np.clip(sim.pred_next_Psi + rng.normal(0, 0.012, len(ready)), -1, 1)
        d.loc[active, 'next_response_evidence_mass'] = sim.pred_next_E
        d.loc[active, 'next_response_evidence_maturity_V'] = sim.pred_next_V
        d.loc[active, 'next_activity_active_mass'] = 0.8 * sim.pred_next_B
        d.loc[active, 'next_activity_idle_mass'] = 0.2 * sim.pred_next_B
        path = root / 'stage1/dynamics' / f'student_dynamics_panel_core_{split}.csv.gz'
        d.to_csv(path, index=False)
        sp = root / 'stage1/splits'
        sp.mkdir(exist_ok=True)
        pd.DataFrame({'user_id': np.unique(uid)}).to_csv(sp / f'{split}_users.csv', index=False)
        ready = p1.prepare_panel(d, split, 20.0)
        cache = p1.make_metric_cache(ready, [], split)
        sim = p1.simulate_arrays(cache, params, p1.Calibration(**cal))
        saved[split] = (ready, cache, sim)
    p1meta = root / 'stage2_phase1/metadata'
    p1meta.mkdir(parents=True)
    save_json(cal, p1meta / 'A_train_only_skeleton_calibration.json')
    save_json({'B_confirm_policy': 'not read or used', 'stage1_load_manifest': {'B_confirm_policy': 'not read or used in Phase 1'}}, p1meta / 'phase1_manifest.json')
    save_json({'fixture_only': True, 'deltaS_saturation_plateau': 'synthetic metadata not paper evidence'}, p1meta / 'phase1_parameter_search_audit.json')
    fmeta = root / 'stage2_phase2_freeze/metadata'
    fmeta.mkdir(parents=True)
    shutil.copyfile(source, fmeta / 'frozen_phase1_implementation.py')
    kcontract = p1.audit_stage1_kmeans_contract(root / 'stage1')
    flags = {k: False for k in ('B_confirm_read', 'parameter_search_opened', 'mechanism_family_reselected', 'mechanism_parameters_refit', 'kmeans_refit', 'macrostate_k_selected', 'region_redefinition')}
    flags['calibration_refit_scope'] = 'A_train_plus_A_val_only'
    mf = {'primary_macrostate': ['M', 'Psi'], 'guardrails': flags, 'frozen_parameters': {'family_key': 'offset_dual_channel', 'free_mechanism_parameters': ['theta0', 'thetaM', 'phi0', 'deltaS'], 'full_parameter_vector': params, 'parameter_hash': objhash(params)}, 'frozen_calibration': cal, 'calibration_hash': objhash(cal), 'stage1_fixed_k6_contract': kcontract, 'phase1_implementation': {'frozen_copy_path': '/old/server/root/frozen_phase1_implementation.py', 'frozen_copy_sha256': sha(source)}}
    save_json(cal, fmeta / 'phase2_pooled_development_calibration.json')
    save_json(mf, fmeta / 'phase2_frozen_model_manifest.json')
    save_json({'manifest_sha256': sha(fmeta / 'phase2_frozen_model_manifest.json')}, fmeta / 'phase2_frozen_model_manifest.sha256.json')
    t = root / 'stage2_phase3_confirm/tables'
    t.mkdir(parents=True)
    ready, cache, sim = saved['B_confirm']
    pred = p1.prediction_frame_from_cache(cache, sim, 'B_confirm', 0, 42)
    pred.to_csv(t / 'phase3_B_confirm_full_predictions.csv.gz', index=False)
    m = p1.structure_metrics_fast_no_regions(cache, sim, 'B_confirm')
    m['objective_primary_score'] = p1.primary_objective_score(m)
    pd.DataFrame([m]).to_csv(t / 'phase3_B_confirm_structural_alignment_metrics.csv', index=False)
    p1.field_grid_table(cache, sim, 'B_confirm').to_csv(t / 'phase3_B_confirm_field_grid.csv', index=False)
    part = load_partition(root)
    origin = assign_states(ready[['M', 'Psi']].to_numpy(), part)
    tr = root / 'stage2_phase3_confirm/figures_publication_minimal_mechanism/tables'
    tr.mkdir(parents=True)
    for name, values in [('empirical', ready[['target_M_next', 'target_Psi_next']].to_numpy()), ('mechanism', np.column_stack([sim.pred_next_M, sim.pred_next_Psi]))]:
        dest = assign_states(values, part)
        C = np.bincount(origin * 6 + dest, minlength=36).reshape(6, 6)
        pd.DataFrame(C).to_csv(tr / f'B_confirm_{name}_mesostate_transition_counts.csv', index=False)
    pm = root / 'stage2_phase3_confirm/metadata'
    pm.mkdir(parents=True)
    pg = {k: False for k in ('parameter_search_opened', 'calibration_reestimated', 'mechanism_family_reselected', 'mechanism_parameters_refit', 'kmeans_refit', 'macrostate_k_selected', 'region_redefinition', 'B_confirm_used_for_update')}
    pmanifest = {'confirm_split': 'B_confirm', 'confirmation_status': 'completed_output_only', 'frozen_parameter_hash': objhash(params), 'frozen_calibration_hash': objhash(cal), 'guardrails': pg, 'frozen_manifest_checksum': {'manifest_sha256': sha(fmeta / 'phase2_frozen_model_manifest.json')}}
    save_json(pmanifest, pm / 'phase3_confirmation_manifest.json')
    save_json({'manifest_sha256': sha(pm / 'phase3_confirmation_manifest.json')}, pm / 'phase3_confirmation_manifest.sha256.json')
    save_json(pg, pm / 'phase3_no_update_audit.json')
    for seed in (42, 2026, 666, 606, 37, 4669):
        nrng = np.random.default_rng(seed)
        nt = pred[['user_id', 'bundle_step_index', 'M', 'Psi', 'target_M_next', 'target_Psi_next']].copy()
        nt['pred_M'] = nt.M
        nt['pred_Psi'] = nt.Psi
        nt['pred_next_M'] = np.clip(sim.pred_next_M + nrng.normal(0, 0.02, len(nt)), -1, 1)
        nt['pred_next_Psi'] = np.clip(sim.pred_next_Psi + nrng.normal(0, 0.02, len(nt)), -1, 1)
        nr = root / ('stage4_event_ssl' if seed == 42 else f'random_seed_experiments/seed_{seed}/stage4_event_ssl') / 'evaluation_predictive_state/predictions'
        nr.mkdir(parents=True)
        nt.to_csv(nr / 'stage4_event_ssl_predictions_B_confirm.csv.gz', index=False)
        eroot = nr.parent
        modelroot = eroot.parent / 'models/predictive_state'
        modelroot.mkdir(parents=True)
        (modelroot / 'best_model.pt').write_bytes(f'SYNTHETIC TEST CHECKPOINT IDENTITY ONLY seed={seed}'.encode())
        save_json({'config': {'seed': seed, 'model_kind': 'predictive_state'}}, modelroot / 'training_manifest.json')
        em = {'model_kind': 'predictive_state', 'primary_coordinates': ['M', 'Psi'], 'guardrails': {'B_confirm_used_for_update': False, 'kmeans_refit': False, 'macrostate_k_selected': False, 'macrostate_k': 6}, 'evaluation_views': {'empirical_anchor': 'test anchor', 'learned_plane': 'test learned'}, 'macro_partition': {'verified': True, 'fit_split': 'A_train', 'macrostate_k': 6, 'metadata_sha256': kcontract['metadata_sha256'], 'centers_sha256': kcontract['centers_sha256']}, 'checkpoint_sha256': sha(modelroot / 'best_model.pt'), 'splits': {'B_confirm': {'prediction_path': str(nr / 'stage4_event_ssl_predictions_B_confirm.csv.gz')}}}
        save_json(em, eroot / 'metadata/stage4_event_ssl_evaluation_manifest.json')
    legacy = root / 'frozen_headline_learner_cluster_uncertainty/tables'
    legacy.mkdir(parents=True)
    rows = []
    for support in ('fixed_formal_support', 'support_reselected'):
        for metric in ('drift_vector_corr', 'drift_speed_corr', 'weighted_local_cosine'):
            rows.append({'comparison': 'mechanism_vs_empirical', 'support_contract': support, 'metric': metric, 'point_estimate': 0.9, 'requested_replicates': 2000, 'finite_replicates': 2000, 'bootstrap_mean': 0.89, 'bootstrap_sd': 0.01, 'bootstrap_bias': -0.01, 'se_centered_ci_low': 0.88, 'se_centered_ci_high': 0.92, 'percentile_low': 0.87, 'percentile_high': 0.91, 'basic_ci_low': 0.89, 'basic_ci_high': 0.93, 'fixture_only': True})
    pd.DataFrame(rows).to_csv(legacy / 'field_cluster_bootstrap_intervals.csv', index=False)
    save_json({'WARNING': 'SYNTHETIC TEST FIXTURE, NOT EDNET DATA OR MANUSCRIPT RESULTS'}, root / 'SYNTHETIC_ONLY.json')
    return root
