from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
PRIMARY_MODELS = ('four_term', 'C3', 'ARX', 'persistence', 'zero_signed')
FIVE_COMPONENTS = {'one_step_mse_main_norm': 0.1, 'occupancy_js_MR_PsiA': 0.2, 'drift_local_rmse_loss_MR_PsiA': 0.3, 'drift_direction_loss_MR_PsiA': 0.2, 'drift_magnitude_loss_MR_PsiA': 0.2}

def _within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path.resolve()), str(root.resolve())]) == str(root.resolve())
    except ValueError:
        return False

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8')
    temp.replace(path)

def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)

def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Required frozen input is missing: {path}')
    return pd.read_csv(path, float_precision='round_trip')

def _bool(values) -> np.ndarray:
    return pd.Series(values).astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()

def _pearson(a, b) -> float:
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if a.shape != b.shape or a.size < 3 or (not (np.isfinite(a).all() and np.isfinite(b).all())):
        return np.nan
    a, b = (a - a.mean(), b - b.mean())
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / den, -1, 1)) if den > 1e-14 else np.nan

def _transition_descriptions(frame: pd.DataFrame):
    summaries, state_rows = ([], [])
    group_columns = [c for c in ('split', 'model') if c in frame.columns]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        if group.duplicated(['origin', 'destination']).any():
            raise ValueError(f'Duplicate transition entries for {identity}')
        p = group.pivot(index='origin', columns='destination', values='model_probability').reindex(index=range(6), columns=range(6)).to_numpy(float)
        e = group.pivot(index='origin', columns='destination', values='empirical_probability').reindex(index=range(6), columns=range(6)).to_numpy(float)
        if not (np.isfinite(p).all() and np.isfinite(e).all()):
            raise ValueError(f'Incomplete fixed-K=6 transition matrix for {identity}')
        if not (np.allclose(p.sum(axis=1), 1, atol=1e-09) and np.allclose(e.sum(axis=1), 1, atol=1e-09)):
            raise ValueError(f'Non-normalized transition matrix for {identity}')
        mp, ep = (np.diag(p), np.diag(e))
        mtop, etop = (np.argmax(p, axis=1), np.argmax(e, axis=1))
        tie_m = [np.flatnonzero(np.isclose(p[i], p[i].max(), atol=1e-14, rtol=0)).tolist() for i in range(6)]
        tie_e = [np.flatnonzero(np.isclose(e[i], e[i].max(), atol=1e-14, rtol=0)).tolist() for i in range(6)]
        mean_tv = float(np.mean(0.5 * np.abs(p - e).sum(axis=1)))
        summaries.append({**identity, 'K': 6, 'mean_row_TV': mean_tv, 'self_transition_Pearson_descriptive': _pearson(mp, ep), 'self_transition_Spearman_descriptive': _pearson(pd.Series(mp).rank(method='average'), pd.Series(ep).rank(method='average')), 'self_transition_RMSE': float(np.sqrt(np.mean((mp - ep) ** 2))), 'self_transition_MAE': float(np.mean(np.abs(mp - ep))), 'dominant_edge_overlap_original_argmax': float(np.mean(mtop == etop)), 'diagonal_dominance_match_original_argmax': float(np.mean((mtop == np.arange(6)) == (etop == np.arange(6)))), 'model_diagonal_dominant_states_original_argmax': int(np.sum(mtop == np.arange(6))), 'empirical_diagonal_dominant_states_original_argmax': int(np.sum(etop == np.arange(6))), 'top_set_exact_match_fraction': float(np.mean([a == b for a, b in zip(tie_m, tie_e)])), 'model_rows_with_top_ties': int(sum((len(x) > 1 for x in tie_m))), 'empirical_rows_with_top_ties': int(sum((len(x) > 1 for x in tie_e))), 'role': 'descriptive fixed-six-state summary; not a new primary endpoint'})
        for i in range(6):
            state_rows.append({**identity, 'state': i, 'empirical_Pii': ep[i], 'model_Pii': mp[i], 'Pii_difference_model_minus_empirical': mp[i] - ep[i], 'row_TV': float(0.5 * np.abs(p[i] - e[i]).sum()), 'empirical_argmax_destination': int(etop[i]), 'model_argmax_destination': int(mtop[i]), 'argmax_match': bool(mtop[i] == etop[i]), 'empirical_top_tie_set': json.dumps(tie_e[i]), 'model_top_tie_set': json.dumps(tie_m[i]), 'top_sets_exactly_match': tie_m[i] == tie_e[i], 'tie_rule': 'original np.argmax selects first maximum; tie sets use absolute tolerance 1e-14'})
    return (pd.DataFrame(summaries), pd.DataFrame(state_rows))

def _annotate_metrics(frame, source, ssl=False):
    frame = frame.copy()
    frame['view'] = 'empirical_current_state_anchor'
    frame['seed'] = 42 if ssl else np.nan
    frame['information_condition'] = ['frozen Event-SSL history/current-interval features; different information budget' if str(model) == 'SSL' else 'same declared 12-input budget; restricted baseline' if str(model) in ('persistence', 'zero_signed') else 'same declared 12-input budget' for model in frame['model']]
    frame['metric_source'] = str(source)
    frame['D2_weight_contract'] = 'S2 eligible-interval user-balanced origin-cell mass; normalized on supported cells'
    return frame

def _merge_extra(metrics, extra, source):
    warnings = []
    if 'model' not in extra:
        return (metrics, ['Supplementary metrics have no model column; preserved separately without joining.'])
    extra = extra.copy()
    extra['model'] = extra['model'].replace({'Event-SSL': 'SSL', 'event_ssl': 'SSL', 'ssl': 'SSL'})
    if 'seed' not in extra:
        extra['seed'] = np.nan
        if extra['model'].eq('SSL').any():
            warnings.append('Supplementary SSL rows lack a seed; preserved separately and not merged into seed42.')
    output = metrics.copy()
    output['seed'] = pd.to_numeric(output['seed'], errors='coerce')
    extra['seed'] = pd.to_numeric(extra['seed'], errors='coerce')
    keys = extra[['model', 'seed']].copy()
    keys['seed'] = keys.seed.fillna(-1)
    if keys.duplicated().any():
        warnings.append('Supplementary metrics repeat a model/seed key; preserved separately without joining.')
        return (output, warnings)
    for _, row in extra.iterrows():
        if row['model'] == 'SSL' and pd.isna(row['seed']):
            continue
        match = output['model'].eq(row['model'])
        match &= output['seed'].eq(row['seed']) if pd.notna(row['seed']) else output['seed'].isna()
        if match.sum() > 1:
            raise ValueError(f"Ambiguous P1 supplementary merge for {row['model']}/{row['seed']}")
        if match.any():
            index = output.index[match][0]
            for name, value in row.items():
                if name in ('model', 'seed', 'split') or pd.isna(value):
                    continue
                if name in output.columns and pd.notna(output.at[index, name]):
                    old = output.at[index, name]
                    if isinstance(value, (int, float, np.integer, np.floating)) and isinstance(old, (int, float, np.integer, np.floating)):
                        if not np.isclose(float(value), float(old), atol=1e-10, rtol=1e-07):
                            warnings.append(f"Supplementary {row['model']}/{row['seed']} {name} differs from frozen metric; frozen value retained, extra value preserved separately.")
                    elif str(value) != str(old):
                        warnings.append(f"Supplementary {row['model']}/{row['seed']} {name} text differs; frozen value retained.")
                else:
                    output.loc[index, name] = value
            output.loc[index, 'supplementary_metric_source'] = str(source)
        else:
            new = row.to_dict()
            new.update({'view': 'empirical_current_state_anchor', 'metric_source': str(source), 'supplementary_metric_source': str(source), 'information_condition': 'frozen Event-SSL history/current-interval features; different information budget' if row['model'] == 'SSL' else 'supplementary frozen-prediction metrics', 'D2_weight_contract': 'S2 eligible-interval user-balanced origin-cell mass; normalized on supported cells'})
            output = pd.concat([output, pd.DataFrame([new])], ignore_index=True, sort=False)
    return (output, warnings)

def _grid_models(frame, names):
    fields = {}
    reference = None
    for name in names:
        block = frame[frame.model.eq(name)].sort_values('cell')
        if len(block) != 1600 or not np.array_equal(block.cell.to_numpy(), np.arange(1600)):
            raise ValueError(f'Expected complete 40x40 archived field grid for {name}')
        index = block[['cell', 'M_center', 'Psi_center']].to_numpy()
        if reference is not None and (not np.allclose(index, reference)):
            raise ValueError('Archived models do not share identical grid coordinates')
        reference = index
        fields[name] = {'u': block.model_M.to_numpy(float), 'v': block.model_Psi.to_numpy(float), 'supported': _bool(block.supported), 'mass': block.mass.to_numpy(float)}
        if 'empirical' not in fields:
            fields['empirical'] = {'u': block.emp_M.to_numpy(float), 'v': block.emp_Psi.to_numpy(float), 'supported': _bool(block.supported), 'mass': block.mass.to_numpy(float)}
        elif not (np.allclose(fields['empirical']['u'], block.emp_M.to_numpy(float)) and np.allclose(fields['empirical']['v'], block.emp_Psi.to_numpy(float))):
            raise ValueError('Archived empirical fields differ across selected models')
    return fields

def _four_panel(fields, names, titles, path, residual=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    common = np.logical_and.reduce([fields[name]['supported'] for name in names])
    vectors = []
    for name in names:
        vector = np.column_stack([fields[name]['u'], fields[name]['v']])
        if residual:
            vector -= np.column_stack([fields['empirical']['u'], fields['empirical']['v']])
        if not np.isfinite(vector[common]).all():
            raise ValueError('Non-finite vector on common supported plot cells')
        vectors.append(vector)
    speeds = [np.linalg.norm(vector, axis=1) for vector in vectors]
    vmax = max((float(speed[common].max()) for speed in speeds)) if common.any() else 1.0
    vmax = max(vmax, 1e-12)
    norm = Normalize(vmin=0, vmax=vmax)
    centers = -1 + (np.arange(40) + 0.5) * 0.05
    xx, yy = np.meshgrid(centers, centers, indexing='ij')
    mask2d = common.reshape(40, 40)
    arrow = mask2d & (np.indices((40, 40))[0] % 2 == 0) & (np.indices((40, 40))[1] % 2 == 0)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, vector, speed, title in zip(axes.ravel(), vectors, speeds, titles):
        color = np.ma.array(speed.reshape(40, 40), mask=~mask2d)
        mesh = ax.pcolormesh(np.linspace(-1, 1, 41), np.linspace(-1, 1, 41), color.T, cmap='viridis', norm=norm, shading='flat', rasterized=True)
        u, v = (vector[:, 0].reshape(40, 40), vector[:, 1].reshape(40, 40))
        ax.quiver(xx[arrow], yy[arrow], u[arrow], v[arrow], color='white', alpha=0.92, angles='xy', scale_units='xy', scale=vmax / 0.12, width=0.003)
        ax.set(title=title, xlabel='Response order M', ylabel='Exposure alignment Ψ', xlim=(-1, 1), ylim=(-1, 1))
        ax.set_aspect('equal')
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_facecolor('#efefef')
        ax.tick_params(labelsize=8)
    label = 'Residual magnitude ||model - empirical||' if residual else 'One-step field speed ||b||'
    fig.colorbar(mesh, ax=axes, label=label, shrink=0.83, pad=0.025)
    subtitle = 'Empirical anchors; identical axes, color scale and arrow scale; gray = unsupported'
    fig.suptitle(('Archived population-field residuals' if residual else 'Archived population fields') + '\n' + subtitle, fontsize=12)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)
    return {'png': path.name, 'svg': path.with_suffix('.svg').name, 'supported_cells': int(common.sum()), 'common_color_min': 0.0, 'common_color_max': vmax, 'quiver_scale_xy': vmax / 0.12, 'arrow_sampling': 'every second grid center in each dimension; common support only', 'residual': residual}

def build_p1(oldroot, out, supplementary_metrics_path=None):
    old = Path(oldroot).expanduser().resolve()
    if not (old / 'confirmation' / 'model_metrics.csv').is_file() and (old / 'formal_outputs').is_dir():
        old = (old / 'formal_outputs').resolve()
    destination = (Path(out).expanduser().resolve() / 'P1').resolve()
    if _within(destination, old) or _within(old, destination):
        raise ValueError('P1 output must be separate from the frozen historical output tree')
    destination.mkdir(parents=True, exist_ok=True)
    lineage, warnings = ([], [])

    def copy_input(relative, target, required=True):
        source = old / relative
        if not source.is_file():
            if required:
                raise FileNotFoundError(source)
            warnings.append(f'Optional historical file absent: {relative}')
            return None
        dest = destination / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == dest.resolve():
            raise ValueError('Refusing to overwrite historical input')
        shutil.copy2(source, dest)
        lineage.append({'source': str(source), 'destination': str(dest.relative_to(destination)), 'sha256': _sha(source), 'bytes': source.stat().st_size, 'operation': 'byte-for-byte copy'})
        return dest
    copies = [('confirmation/model_metrics.csv', 'model_metrics_original.csv', True), ('confirmation/primary_point_contrasts.csv', 'baseline_contrasts_original.csv', True), ('confirmation/statewise_transitions.csv', 'statewise_transitions_original.csv', True), ('confirmation/prediction_audit.json', 'contracts/prediction_audit.json', True), ('confirmation/field_grid.csv.gz', 'grids/confirmation_field_grid.csv.gz', True), ('confirmation/M2_field_components.csv', 'M2/field_components.csv', True), ('confirmation/M2_core_shell_composition.csv', 'M2/core_shell_composition.csv', True), ('confirmation/M2_decomposition_contract.json', 'M2/decomposition_contract.json', True), ('inference/paired_intervals.csv', 'uncertainty/mechanism_paired_intervals_original.csv', True), ('inference/point_estimands.csv', 'uncertainty/mechanism_point_estimands_original.csv', True), ('inference/bootstrap_contract.json', 'uncertainty/mechanism_bootstrap_contract.json', True), ('inference/inference_audit.json', 'uncertainty/mechanism_inference_audit.json', True), ('cross_model/seed42_same_sample_metrics.csv', 'SSL_anchor/seed42_same_sample_metrics_original.csv', True), ('cross_model/seed42_common_field_grid.csv.gz', 'grids/seed42_common_field_grid.csv.gz', True), ('cross_model/C3_referenced_cross_model_by_seed.csv', 'cross_model/six_seed_C3_common_target_original.csv', True), ('cross_model/seed42_uncertainty/paired_intervals.csv', 'uncertainty/cross_model_seed42_paired_intervals_original.csv', True), ('cross_model/seed42_uncertainty/bootstrap_contract.json', 'uncertainty/cross_model_seed42_bootstrap_contract.json', True), ('cross_model/seed42_uncertainty/inference_audit.json', 'uncertainty/cross_model_seed42_inference_audit.json', True), ('cross_model/join_audits.json', 'contracts/SSL_join_audits.json', True), ('cross_model/M3_scope.json', 'contracts/M3_scope.json', True), ('existing_evidence/Table11_same_method_cross_comparison.csv', 'historical_CI/Table11_same_method_cross_comparison.csv', True), ('existing_evidence/Table11_bias_diagnostics_extracted.csv', 'historical_CI/Table11_bias_diagnostics_extracted.csv', True), ('models/final_models.json', 'contracts/final_models.json', True), ('models/FIT_FROZEN.json', 'contracts/FIT_FROZEN.json', True), ('models/ARX_all_validation_candidates.csv', 'development/ARX_all_validation_candidates.csv', True), ('models/development_metrics.csv', 'development/model_metrics.csv', True), ('models/development_transitions.csv', 'development/statewise_transitions.csv', True)]
    for relative, target, required in copies:
        copy_input(relative, target, required)
    primary = _read(old / 'confirmation/model_metrics.csv')
    if set(primary.model) != set(PRIMARY_MODELS) or primary.model.duplicated().any():
        raise ValueError('Frozen comparison does not contain exactly the five declared models')
    ssl_all = _read(old / 'cross_model/seed42_same_sample_metrics.csv')
    ssl = ssl_all[ssl_all.model.eq('SSL')].copy()
    if len(ssl) != 1:
        raise ValueError('Expected one frozen seed42 SSL-anchor metric row')
    metrics = pd.concat([_annotate_metrics(primary, 'confirmation/model_metrics.csv'), _annotate_metrics(ssl, 'cross_model/seed42_same_sample_metrics.csv', ssl=True)], ignore_index=True)
    if supplementary_metrics_path is not None:
        supplementary = Path(supplementary_metrics_path).expanduser().resolve()
        if not supplementary.is_file():
            raise FileNotFoundError(f'Requested supplementary metrics are absent: {supplementary}')
        extra = _read(supplementary)
        shutil.copy2(supplementary, destination / 'supplementary_metrics_original.csv')
        lineage.append({'source': str(supplementary), 'destination': 'supplementary_metrics_original.csv', 'sha256': _sha(supplementary), 'bytes': supplementary.stat().st_size, 'operation': 'byte-for-byte copy'})
        metrics, notes = _merge_extra(metrics, extra, supplementary)
        warnings.extend(notes)
    else:
        warnings.append('No supplementary metrics supplied: next-state Pearson correlations and additional SSL per-seed metric rows are unavailable.')
    _write_csv(metrics, destination / 'comparison_metrics.csv')
    score_columns = ['split', 'model', 'seed', 'information_condition', *FIVE_COMPONENTS, 'objective_primary_score', 'metric_source']
    score = metrics[[c for c in score_columns if c in metrics]].copy()
    _write_csv(score, destination / 'five_component_metrics.csv')
    _write_csv(pd.DataFrame([{'component': name, 'weight': value, 'role': 'original family-score component; supplementary comparison, no new model selection'} for name, value in FIVE_COMPONENTS.items()]), destination / 'five_component_definitions.csv')
    transition_frame = _read(old / 'confirmation/statewise_transitions.csv')
    transition_summary, state_summary = _transition_descriptions(transition_frame)
    _write_csv(transition_summary, destination / 'transition_descriptive_summary.csv')
    _write_csv(state_summary, destination / 'statewise_persistence_and_ties.csv')
    for _, row in transition_summary.iterrows():
        frozen = primary[primary.model.eq(row.model)].iloc[0]
        if not np.isclose(row.mean_row_TV, frozen.mean_row_TV, atol=1e-12, rtol=1e-10):
            raise ValueError(f'Transition TV reconstruction mismatch for {row.model}')
        point = row.self_transition_Pearson_descriptive
        expected = frozen.self_transition_r_descriptive
        if not (pd.isna(point) and pd.isna(expected) or np.isclose(point, expected, atol=1e-12, rtol=1e-10)):
            raise ValueError(f'Self-transition Pearson reconstruction mismatch for {row.model}')
    history = _read(old / 'cross_model/C3_referenced_cross_model_by_seed.csv')
    history_columns = [c for c in history if c in ('seed', 'rows', 'users') or c.startswith(('raw_', 'C3_correction_', 'direct_empirical_error_', 'partial_r_', 'common_target_', 'same_subtraction_'))]
    _write_csv(history[history_columns], destination / 'cross_model/common_target_and_pair_diagnostics.csv')
    cross_notes = 'All raw_* columns in this archived cross-model table compare four_term with SSL-anchor, not SSL with empirical.\nC3_correction_* compares (four-C3) with (SSL-C3); it does not compare either with (empirical-C3).\ndirect_empirical_error_* compares (four-empirical) with (SSL-empirical); OLS partial correlation is a third diagnostic.\nCommon subtraction leaves cross-model error vectors and their RMSE unchanged. Positive correction correlation alone is not mechanistic identity.\nThese archived point/interval files are separate from the three-way excess bootstrap reported here. Missing/not_estimable intervals remain missing.\n'
    _write_text(cross_notes, destination / 'cross_model/READ_THIS_FIRST.txt')
    plot_contracts = []
    if os.environ.get('P1_PLOT_ARCHIVED_GRIDS', '0') == '1':
        main_grid = _read(old / 'confirmation/field_grid.csv.gz')
        fields = _grid_models(main_grid, ('four_term', 'C3', 'ARX'))
        cross_fields = _grid_models(_read(old / 'cross_model/seed42_common_field_grid.csv.gz'), ('four_term', 'C3', 'SSL'))
        for key in ('u', 'v'):
            if not np.allclose(fields['empirical'][key], cross_fields['empirical'][key], atol=1e-12, rtol=1e-10):
                raise ValueError('SSL archived field is not on the same empirical sample/grid')
        fields['SSL'] = cross_fields['SSL']
        figure_dir = destination / 'figures'
        figure_dir.mkdir(exist_ok=True)
        plot_contracts = [_four_panel(fields, ('empirical', 'four_term', 'C3', 'ARX'), ('Empirical field', 'Frozen four-term mechanism', 'C3 accounting predictor', 'ARX conditional mean'), figure_dir / 'population_fields_four_panels.png'), _four_panel(fields, ('four_term', 'C3', 'ARX', 'SSL'), ('Four-term minus empirical', 'C3 minus empirical', 'ARX minus empirical', 'SSL anchor minus empirical\n(different input information)'), figure_dir / 'field_residuals_four_panels.png', residual=True)]
        _write_text(json.dumps(plot_contracts, indent=2), figure_dir / 'plot_contract.json')
    display = metrics[metrics.model.ne('SSL') | metrics.seed.eq(42)].copy()

    def compact(columns):
        cols = [c for c in ['model', 'seed', *columns] if c in display]
        return display[cols].to_string(index=False, na_rep='NA', float_format=lambda value: f'{value:.6g}')
    core_values = primary.set_index('model')
    four, c3 = (core_values.loc['four_term'], core_values.loc['C3'])
    summary = ['Frozen-output comparison in the original metric framework', '', 'This report rearranges and derives descriptions from frozen outputs. No model was fitted.', f'Frozen source: {old}', f'Output: {destination}', '', 'Direction, speed and cell-equal raw field error (common empirical anchors)', compact(['vector_r', 'speed_r', 'local_cosine', 'amplitude_slope', 'local_cosine_mass_fraction', 'unweighted_rmse']), '', 'One-step, next-state occupancy and fixed-K=6 transition recovery', compact(['one_step_rmse_M', 'one_step_rmse_Psi', 'one_step_corr_M', 'one_step_corr_Psi', 'occupancy_js_MR_PsiA', 'mean_row_TV', 'self_transition_r_descriptive']), '', 'Additional eligible-origin-mass-weighted error (distinct from original SI Table 9 panel-occupancy weights)', compact(['D2', 'D2_M', 'D2_Psi', 'D']), '', 'Supplementary findings', f'- Four-term total field D2 is {four.D2:.9g}, versus C3 {c3.D2:.9g}; this is not uniform superiority across endpoints.', f'- The original study found lower transition TV and lower Psi field error for C3. On the supplied inputs, transition TV is {c3.mean_row_TV:.6g} for C3 versus {four.mean_row_TV:.6g} for the four-term mechanism, and Psi field error is {c3.D2_Psi:.6g} versus {four.D2_Psi:.6g}, respectively.', '- Persistence has zero drift, so field correlation/cosine is undefined even when its next-state occupancy JS is low.', '- SSL is presented separately because its histories/current-interval features differ from the 12-input mechanism comparison.', '- Six-state Pearson/Spearman, dominant edges and tie summaries are descriptive; no new primary endpoint or selection rule was introduced.', '- Original paired intervals are copied unchanged with requested/finite replicate counts and not_estimable statuses.', '- Raw field RMSE is cell-equal; D2 uses supported eligible-origin mass; interval RMSE and user-equal RMSE are different estimands.', '- The five-component score is the original mechanism-family score, not the Event-SSL representation composite.', '- Historical Table 11 fixed/reselected columns change both support handling and interval construction; compare matched methods and retain bias diagnostics.', '- The three-way excess comparison remains an empirical-anchor analysis, not a learned-plane or same-readout trajectory experiment.', '', 'Supplementary-metric and optional-input notes', *(warnings if warnings else ['No source/merge warnings.']), '', 'Files', 'comparison_metrics.csv: full original-framework columns, information/view/source labels, optional new per-seed metrics.', 'five_component_metrics.csv: original five components and score; transition_descriptive_summary.csv and statewise_persistence_and_ties.csv: exact fixed-state descriptions.', 'uncertainty/: unchanged existing paired bootstrap results; historical_CI/: original Table 11 same-method and bias tables.', 'M2/: unchanged accounting decomposition and core/shell composition; cross_model/: original common-target and pair diagnostics, with interpretation notes.', 'Figures are supplied by the execution driver. Optional standalone archived figures require P1_PLOT_ARCHIVED_GRIDS=1.']
    _write_text('\n'.join(summary) + '\n', destination / 'OVERVIEW.txt')
    _write_csv(pd.DataFrame(lineage), destination / 'source_lineage.csv')
    manifest = {'status': 'complete', 'operation': 'read-only historical-output reporting', 'oldroot': str(old), 'out': str(destination), 'no_model_fitting': True, 'no_historical_output_writes': True, 'comparison_rows': int(len(metrics)), 'supplementary_metrics_path': str(supplementary_metrics_path) if supplementary_metrics_path is not None else None, 'transition_rule': 'original argmax plus explicit tie-set descriptions; Spearman average ranks; fixed K=6; descriptive only', 'warnings': warnings, 'plots': plot_contracts, 'input_files': len(lineage)}
    _write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', destination / 'P1_COMPLETE.json')
    return manifest
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize frozen predictions in the original metric framework.')
    parser.add_argument('--oldroot', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--supplementary-metrics-path')
    args = parser.parse_args()
    print(json.dumps(build_p1(args.oldroot, args.out, args.supplementary_metrics_path), indent=2, ensure_ascii=False))
