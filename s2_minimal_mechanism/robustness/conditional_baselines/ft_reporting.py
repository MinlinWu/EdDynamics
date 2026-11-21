from __future__ import annotations
from pathlib import Path
import json
import math
import shutil
import numpy as np
import pandas as pd
from scipy.stats import norm
from ft_common import *
from ft_statistics import basic_interval
LEGACY_BASES = ('stage2_phase1_unified_minimality/tables/model_family_results', 'stage2_phase1_unified_minimality/tables/selected_model_parameter_deletions', 'stage2_phase1_unified_minimality/tables/equivalence_margin_sensitivity', 'stage2_phase1_score_contract_robustness/tables/score_contract_robustness_summary', 'stage2_phase1_score_contract_robustness/formal_audit/tables/frozen_fit_pareto_mean_components', 'stage2_phase1/metadata/phase1_parameter_search_audit.json', 'stage2_phase3_confirm/supplementary_minimal_mechanism/tables/supplementary_parameter_boundary_audit', 'stage2_phase3_confirm/supplementary_minimal_mechanism/tables/supplementary_table4_statewise_kinetics', 'six_seed_null_common_target_audit/tables/common_target_conditioned_seed_summary', 'null_referenced_downstream_recovery/tables/B_confirm_null_referenced_recovery_metrics', 'null_referenced_downstream_recovery/tables/B_confirm_null_referenced_geometry_metrics', 'frozen_headline_learner_cluster_uncertainty/numeric_report/tables/bootstrap_bias_diagnostics', 'frozen_headline_learner_cluster_uncertainty/numeric_report/tables/support_reselection_summary')

def legacy_interval(values, point, transform, level=0.95):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v) or not np.isfinite(point):
        return {'finite_replicates': len(v)}
    a = 1 - level
    z = norm.ppf(1 - a / 2)
    if transform == 'fisher_z':
        tv = np.arctanh(np.clip(v, -1 + 1e-12, 1 - 1e-12))
        tp = float(np.arctanh(np.clip(point, -1 + 1e-12, 1 - 1e-12)))
        q = np.quantile(tv, [a / 2, 1 - a / 2])
        sd = np.std(tv, ddof=1) if len(v) > 1 else 0.0
        basic = np.tanh([2 * tp - q[1], 2 * tp - q[0]])
        per = np.tanh(q)
        se = np.tanh([tp - z * sd, tp + z * sd])
    else:
        q = np.quantile(v, [a / 2, 1 - a / 2])
        sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
        basic = np.clip([2 * point - q[1], 2 * point - q[0]], -1, 1)
        per = np.clip(q, -1, 1)
        se = np.clip([point - z * sd, point + z * sd], -1, 1)
    return {'finite_replicates': len(v), 'bootstrap_mean': v.mean(), 'bootstrap_sd': np.std(v, ddof=1) if len(v) > 1 else 0.0, 'bootstrap_bias': v.mean() - point, 'basic_ci_low': basic[0], 'basic_ci_high': basic[1], 'percentile_low': per[0], 'percentile_high': per[1], 'se_centered_ci_low': se[0], 'se_centered_ci_high': se[1]}

def collect_existing(ctx):
    h = ctx.history
    tabroot = h / 'frozen_headline_learner_cluster_uncertainty/tables'
    ip = find_table(tabroot / 'field_cluster_bootstrap_intervals', False)
    rp = find_table(tabroot / 'field_cluster_bootstrap_replicates', False)
    pp = find_table(tabroot / 'formal_point_estimates', False)
    require(ip is not None or (rp is not None and pp is not None), 'Both stored legacy interval table and reconstructible replicate/point tables are missing. Restore the declared original artifacts; no fabricated Table 11 audit.')
    paths = []
    for base in LEGACY_BASES:
        p = h / base if base.endswith('.json') else find_table(h / base, False)
        paths.append((base, p if p is not None and p.is_file() else None))
    inputs = {'intervals': fingerprint(ip) if ip else None, 'fallback_replicates': fingerprint(rp) if ip is None and rp else None, 'fallback_points': fingerprint(pp) if ip is None and pp else None, 'legacy': {base: fingerprint(p) if p else None for base, p in paths}}

    def run(dst):
        inventory = []
        for i, (base, p) in enumerate(paths):
            entry = {'artifact': base, 'exists': p is not None, 'action': 'reuse; no re-training/reselection'}
            if p:
                entry.update(fingerprint(p))
                dest = dst / 'legacy' / f'{i:02d}_{p.name}'
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(p, dest)
                entry['copy'] = str(dest.relative_to(dst))
            inventory.append(entry)
        write_csv(pd.DataFrame(inventory), dst / 'legacy_artifact_inventory.csv')
        if ip:
            data = read_table(ip)
            source = 'stored_interval_summaries'
        else:
            reps = read_table(rp)
            point = read_table(pp)
            rows = []
            for (comparison, support), g in reps.groupby(['comparison', 'support_contract'], sort=False):
                for metric in ('drift_vector_corr', 'drift_speed_corr', 'weighted_local_cosine'):
                    r = point[(point['domain'] == 'field') & (point['comparison'] == comparison) & (point['metric'] == metric)]
                    require(len(r) == 1, f'Missing/duplicate legacy formal point {comparison}/{metric}')
                    pc = 'value' if 'value' in r else 'point_estimate'
                    pv = float(r[pc].iloc[0])
                    tr = 'fisher_z' if metric != 'weighted_local_cosine' else 'bounded_raw'
                    rows.append({'comparison': comparison, 'support_contract': support, 'metric': metric, 'point_estimate': pv, 'requested_replicates': g.replicate.nunique(), **legacy_interval(g[metric], pv, tr)})
            data = pd.DataFrame(rows)
            source = 'reaggregated_existing_replicates_no_new_sampling'
        required = {'comparison', 'support_contract', 'metric', 'point_estimate', 'se_centered_ci_low', 'se_centered_ci_high', 'percentile_low', 'percentile_high', 'basic_ci_low', 'basic_ci_high', 'bootstrap_bias', 'bootstrap_sd'}
        require(required.issubset(data.columns), 'Legacy intervals lack the source-defined methods; no selective substitution')
        write_csv(data, dst / 'Table11_same_method_cross_comparison.csv')
        notes = []
        for _, r in data.iterrows():
            sd = float(r['bootstrap_sd'])
            bias = float(r['bootstrap_bias'])
            notes.append({'comparison': r['comparison'], 'support_contract': r['support_contract'], 'metric': r['metric'], 'bias_over_sd': bias / sd if sd > EPS else np.nan, 'point_inside_percentile': bool(r['percentile_low'] <= r['point_estimate'] <= r['percentile_high']), 'point_inside_basic': bool(r['basic_ci_low'] <= r['point_estimate'] <= r['basic_ci_high'])})
        write_csv(pd.DataFrame(notes), dst / 'Table11_bias_diagnostics_extracted.csv')
        save_json({'source': source, 'new_sampling_performed': False, 'higher_or_lower_interval_not_selected_for_convenience': True, 'existing_null_skill_label': 'old innovation-reassignment reference only'}, dst / 'reuse_audit.json')
    return atomic_stage(ctx.out, 'existing_evidence', ctx.run_hash, inputs, run)

def fmt(x):
    if x is None:
        return 'NA'
    try:
        return f'{float(x):.6g}' if np.isfinite(float(x)) else 'NA'
    except (TypeError, ValueError):
        return str(x)

def mdtable(frame, columns):
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for _, r in frame.iterrows():
        lines.append('| ' + ' | '.join((fmt(r[c]) for c in columns)) + ' |')
    return '\n'.join(lines)

def legacy_reuse_statement(inventory):
    categories = (('term-deletion refits', 'stage2_phase1_unified_minimality/tables/selected_model_parameter_deletions'), ('equal-weight selection', 'stage2_phase1_score_contract_robustness/tables/score_contract_robustness_summary'), ('Pareto checks', 'stage2_phase1_score_contract_robustness/formal_audit/tables/frozen_fit_pareto_mean_components'))
    present = set(inventory.loc[inventory['exists'].astype(str).str.lower() == 'true', 'artifact'])
    available = [label for label, path in categories if path in present]
    missing = [label for label, path in categories if path not in present]
    text = 'Historical evidence reused: ' + ', '.join(available) + '. These are not new experiments. ' if available else 'No historical term-deletion, equal-weight-selection or Pareto artifact was available for reuse. '
    if missing:
        text += 'Not available in this run and not claimed as reused: ' + ', '.join(missing) + '. '
    return text

def crossmodel_report_view(cross, audits):
    result = cross.copy()
    records = {}
    for audit in audits:
        seed = int(audit['seed'])
        require(seed not in records, 'Duplicate neural join audit seed')
        records[seed] = audit
    for _, row in result.iterrows():
        seed = int(row['seed'])
        require(seed in records, f'Missing neural join audit for seed {seed}')
        require(int(row['rows']) == int(records[seed]['joined_rows']), f'M3 report/join row count mismatch for seed {seed}')
    result['M1_row_fraction'] = [records[int(seed)]['M1_row_fraction'] for seed in result['seed']]
    for name in ('improvement_four_over_C3', 'improvement_SSL_over_C3'):
        if name not in result:
            result[name] = np.nan
    return result

def figures(ctx, dst, metrics, intervals, grid, components):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = dst / 'figures'
    figdir.mkdir()
    prefix = 'SYNTHETIC FIXTURE — NOT EDNET RESULTS\n' if getattr(ctx, 'synthetic_fixture', False) else 'SMOKE SUBSAMPLE — NOT PAPER RESULTS\n' if ctx.smoke else ''

    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'svg', 'pdf'):
            fig.savefig(figdir / f'{name}.{ext}', dpi=320, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(metrics['model'], metrics['D'])
    ax.set_ylabel('Population-field error (state units / bundle step)')
    ax.set_title(prefix + 'Matched-information conditional one-step comparison')
    ax.tick_params(axis='x', rotation=20)
    save(fig, 'M1_total_field_error')
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, metrics['D2_M'] * 1000, 0.36, label='Response evidence M')
    ax.bar(x + 0.18, metrics['D2_Psi'] * 1000, 0.36, label='Activity alignment Ψ')
    ax.set_xticks(x, metrics.model, rotation=20)
    ax.set_ylabel('Squared field error × 1,000')
    ax.legend()
    ax.set_title(prefix + 'Axis-specific errors, not causal contributions')
    save(fig, 'M1_axis_squared_errors')
    pick = intervals[(intervals.support_contract == 'fixed') & intervals.metric.isin(['Delta_C3', 'Delta_ARX']) & np.isclose(intervals.level, 0.975)]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, (_, r) in enumerate(pick.iterrows()):
        pt = float(r.point) * 1000
        lo = float(r.basic_low) * 1000
        hi = float(r.basic_high) * 1000
        if np.isfinite(pt):
            ax.plot(i, pt, 'o', label='Point estimate' if i == 0 else None)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.vlines(i, lo, hi)
            ax.plot([i - 0.05, i + 0.05], [lo, lo])
            ax.plot([i - 0.05, i + 0.05], [hi, hi])
    ax.axhline(0, linestyle='--')
    ax.set_xticks(range(len(pick)), pick.metric)
    ax.set_ylabel('(Baseline D² − four-term D²) × 1,000')
    ax.set_title(prefix + 'Predeclared 97.5% individual basic intervals\nPositive values favour the frozen four-term model')
    save(fig, 'M1_paired_field_error_contrasts')
    for name in ('four_term', 'C3', 'ARX'):
        q = grid[grid.model == name].sort_values('cell')
        ok = q.supported.astype(bool).to_numpy()
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        ax.quiver(q.M_center.to_numpy()[ok], q.Psi_center.to_numpy()[ok], q.model_M.to_numpy()[ok], q.model_Psi.to_numpy()[ok], angles='xy', scale_units='xy', scale=1)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel='Response evidence M', ylabel='Activity alignment Ψ', aspect='equal')
        ax.set_title(prefix + f'{name}: conditional field\nArrows: state change per bundle step')
        save(fig, 'field_' + name)
    for axis, cols in [('M', ['M_dilution', 'M_signed_drive', 'M_numeric_branch_residual']), ('Psi', ['Psi_dilution', 'Psi_response_drive', 'Psi_support_drive', 'Psi_G_consistency', 'Psi_numeric_branch_residual'])]:
        fig, ax = plt.subplots(figsize=(7.7, 4.7))
        width = 0.8 / len(cols)
        for j, col in enumerate(cols):
            vals = []
            for reg in ('core', 'shell'):
                r = components[(components.region == reg) & (components.variable == col)]
                vals.append(float(r.user_weighted_mean.iloc[0]) if len(r) else np.nan)
            ax.bar(np.arange(2) + (j - (len(cols) - 1) / 2) * width, np.array(vals) * 100, width, label=col)
        ax.set_xticks([0, 1], ['Frozen core', 'Frozen shell'])
        ax.axhline(0, linewidth=0.8)
        ax.set_ylabel('Mean component change × 100 (state units / step)')
        ax.set_title(prefix + f'{axis}: additive update components, not error shares')
        ax.legend(fontsize=7, loc='best')
        save(fig, 'M2_components_' + axis)

def final_report(ctx, evalpath, bootpath, crosspath, legacypath, modelpath):
    inputs = {k: receipt_hash(p) for k, p in [('evaluation', evalpath), ('inference', bootpath), ('crossmodel', crosspath), ('legacy', legacypath), ('models', modelpath)]}

    def run(dst):
        metrics = pd.read_csv(evalpath / 'model_metrics.csv')
        ci = pd.read_csv(bootpath / 'paired_intervals.csv')
        cross = pd.read_csv(crosspath / 'C3_referenced_cross_model_by_seed.csv')
        crossview = crossmodel_report_view(cross, load_json(crosspath / 'join_audits.json'))
        crosscols = ['seed', 'rows', 'users', 'M1_row_fraction', 'raw_vector_r', 'C3_correction_vector_r', 'direct_empirical_error_vector_r', 'partial_r_four_SSL_given_empirical_descriptive', 'improvement_four_over_C3', 'improvement_SSL_over_C3']
        crossnote = "M1_row_fraction is the fraction of M1 rows retained by this seed's join, not spatial support coverage. Each seed is evaluated on its own common sample with recomputed learner weights. Positive improvement means lower squared field error than C3 on that same sample; correction correlation alone is not evidence of improvement. "
        legacy_note = legacy_reuse_statement(pd.read_csv(legacypath / 'legacy_artifact_inventory.csv'))
        grid = pd.read_csv(evalpath / 'field_grid.csv.gz')
        components = pd.read_csv(evalpath / 'M2_core_shell_composition.csv')
        figures(ctx, dst, metrics, ci, grid, components)
        p = ci[(ci.support_contract == 'fixed') & ci.metric.isin(['Delta_C3', 'Delta_ARX']) & np.isclose(ci.level, 0.975)]
        outcome = []
        for _, r in p.iterrows():
            if r.status != 'complete':
                status = 'NOT_ESTIMABLE'
            elif bool(r.method_sign_conflict):
                status = 'INTERVAL_METHOD_SENSITIVE'
            elif r.basic_low > 0:
                status = 'FOUR_TERM_LOWER_FIELD_ERROR'
            elif r.basic_high < 0:
                status = 'BASELINE_LOWER_FIELD_ERROR'
            else:
                status = 'NO_CLEAR_ADVANTAGE_NOT_EQUIVALENCE'
            outcome.append({'comparison': r.metric, 'classification': status, 'point': r.point, 'basic_97_5_low': r.basic_low, 'basic_97_5_high': r.basic_high})
        decisions = pd.DataFrame(outcome)
        write_csv(decisions, dst / 'result_interpretation.csv')
        write_csv(metrics, dst / 'main_comparison_table.csv')
        write_csv(ci, dst / 'paired_uncertainty_table.csv')
        title = '# Four-term mechanism supplementary results\n\n'
        note = '**SYNTHETIC FIXTURE ONLY — not EdNet scientific results.**\n\n' if getattr(ctx, 'synthetic_fixture', False) else '**SMOKE TEST ONLY — no formal scientific inference.**\n\n' if ctx.smoke else ''
        text = title + note + 'The original four-term model is frozen. New fits use development users only. All predictions condition on realized interval opportunities.\n\n'
        text += '## Matched-information comparison\n\n' + mdtable(metrics, ['model', 'D', 'D2_M', 'D2_Psi', 'one_step_rmse_M', 'one_step_rmse_Psi', 'mean_row_TV']) + '\n\n'
        text += '## Predeclared paired contrasts\n\n' + mdtable(decisions, list(decisions.columns)) + '\n\n'
        text += 'Positive Delta means lower four-term error. A crossing interval is not evidence of equivalence. Interval-method conflicts are reported rather than resolved by selecting a favourable method.\n\n'
        text += '## Scope of inference\n\nThese are post hoc supplementary analyses on fit-held-out confirmation users, not a new blind replication. Neither C3 nor ARX is an autonomous trajectory generator. The old innovation-reassignment null and the new generative construction-null study have different roles. This package does not retrain Event-SSL, test policy interventions, or settle informative censoring.\n\n'
        text += '## Original model-family evidence\n\n' + legacy_note + '\n\n'
        text += '## Frozen neural comparison\n\n' + crossnote + '\n\n' + mdtable(crossview, crosscols) + '\n'
        (dst / 'numeric_report.md').write_text(text, encoding='utf-8')
        save_json({'technical_completion': True, 'scientific_success_required': False, 'smoke_only': ctx.smoke, 'synthetic_fixture': getattr(ctx, 'synthetic_fixture', False), 'analysis_scope': 'conditional four-term comparison; no autonomous or causal-mechanism claim', 'rows': int(metrics.rows.iloc[0]), 'model_comparisons': decisions.to_dict('records'), 'simultaneous_advantage_over_both_declared_baselines': bool(len(decisions) == 2 and (decisions.classification == 'FOUR_TERM_LOWER_FIELD_ERROR').all()), 'run_hash': ctx.run_hash, 'new_model_freeze': receipt_hash(modelpath)}, dst / 'COMPLETE.json')
    return atomic_stage(ctx.out, 'final', ctx.run_hash, inputs, run)
