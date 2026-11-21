from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from ft_common import *
from ft_data import grid_cells, assign_states

def corr(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if len(a) < 3 or a.shape != b.shape or (not (np.isfinite(a).all() and np.isfinite(b).all())):
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / den, -1, 1)) if den > EPS else np.nan

def normalized_transition(C):
    C = np.asarray(C, float)
    den = C.sum(axis=1)
    return np.divide(C, den[:, None], out=np.full_like(C, np.nan), where=den[:, None] > 0)

def field_compare(a, b, W, mask):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return {k: np.nan for k in ('vector_r', 'speed_r', 'local_cosine', 'amplitude_slope', 'local_cosine_mass_fraction', 'unweighted_rmse')}
    a = np.asarray(a)[mask]
    b = np.asarray(b)[mask]
    w = np.asarray(W)[mask]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {k: np.nan for k in ('vector_r', 'speed_r', 'local_cosine', 'amplitude_slope', 'local_cosine_mass_fraction', 'unweighted_rmse')}
    sa = np.linalg.norm(a, axis=1)
    sb = np.linalg.norm(b, axis=1)
    ok = sa * sb > EPS
    local = float(np.sum(w[ok] * np.sum(a[ok] * b[ok], axis=1) / (sa[ok] * sb[ok])) / w[ok].sum()) if w[ok].sum() > EPS else np.nan
    denom = np.sum(w * np.sum(a * a, axis=1))
    return {'vector_r': corr(a, b), 'speed_r': corr(sa, sb), 'local_cosine': local, 'amplitude_slope': float(np.sum(w * np.sum(a * b, axis=1)) / denom) if denom > EPS else np.nan, 'local_cosine_mass_fraction': float(w[ok].sum() / w.sum()) if w.sum() > EPS else np.nan, 'unweighted_rmse': float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))}

class Evaluator:

    def __init__(self, names, partition):
        self.names = tuple(names)
        self.partition = partition
        k = len(names)
        self.N = 0
        self.Umass = 0.0
        self.count = np.zeros(NCELL)
        self.mass = np.zeros(NCELL)
        self.emp_sum = np.zeros((NCELL, 2))
        self.pred_sum = np.zeros((k, NCELL, 2))
        self.occ_emp = np.zeros(NCELL)
        self.occ_pred = np.zeros((k, NCELL))
        self.err = np.zeros((k, 2))
        self.errw = np.zeros((k, 2))
        self.ymean = np.zeros(2)
        self.yM2 = np.zeros(2)
        self.transitions = np.zeros((k + 1, 6, 6))
        self.next_cross = np.zeros((k, 2))
        self.next_sum = np.zeros((k, 2))
        self.next_sq = np.zeros((k, 2))

    def add(self, F, Y, P, w):
        F = np.asarray(F)
        Y = np.asarray(Y)
        P = np.asarray(P)
        w = np.asarray(w)
        require(P.shape == (len(F), len(self.names), 2), 'Prediction array shape mismatch')
        require(np.isfinite(P).all(), 'Non-finite predictions; no model-specific row removal')
        cells = grid_cells(F[:, :2])
        n = len(F)
        nb = self.N + n
        ym = Y.mean(axis=0)
        dy = ym - self.ymean
        self.yM2 += np.sum((Y - ym) ** 2, axis=0) + dy ** 2 * (self.N * n / max(nb, 1))
        self.ymean += dy * n / max(nb, 1)
        self.N = nb
        self.Umass += float(w.sum())
        self.count += np.bincount(cells, minlength=NCELL)
        self.mass += np.bincount(cells, weights=w, minlength=NCELL)
        for a in range(2):
            self.emp_sum[:, a] += np.bincount(cells, weights=w * (Y[:, a] - F[:, a]), minlength=NCELL)
        self.occ_emp += np.bincount(grid_cells(Y), weights=w, minlength=NCELL)
        origin = assign_states(F[:, :2], self.partition)
        target = assign_states(Y, self.partition)
        self.transitions[0] += np.bincount(origin * 6 + target, minlength=36).reshape(6, 6)
        for j in range(len(self.names)):
            e = P[:, j] - Y
            self.err[j] += (e ** 2).sum(axis=0)
            self.errw[j] += (w[:, None] * e ** 2).sum(axis=0)
            self.next_sum[j] += P[:, j].sum(axis=0)
            self.next_sq[j] += (P[:, j] ** 2).sum(axis=0)
            self.next_cross[j] += (P[:, j] * Y).sum(axis=0)
            for a in range(2):
                self.pred_sum[j, :, a] += np.bincount(cells, weights=w * (P[:, j, a] - F[:, a]), minlength=NCELL)
            self.occ_pred[j] += np.bincount(grid_cells(P[:, j]), weights=w, minlength=NCELL)
            dest = assign_states(P[:, j], self.partition)
            self.transitions[j + 1] += np.bincount(origin * 6 + dest, minlength=36).reshape(6, 6)

    def fields(self):
        den = np.maximum(self.mass, EPS)
        return (self.emp_sum / den[:, None], self.pred_sum / den[None, :, None])

    def primary(self, p1, j):
        empirical, model = self.fields()
        mask = self.count >= MIN_COUNT
        ef = {'u': empirical[:, 0].reshape(40, 40), 'v': empirical[:, 1].reshape(40, 40), 'mask': mask.reshape(40, 40), 'weight': self.mass.reshape(40, 40)}
        pf = {'u': model[j, :, 0].reshape(40, 40), 'v': model[j, :, 1].reshape(40, 40), 'mask': mask.reshape(40, 40), 'weight': self.mass.reshape(40, 40)}
        scale = np.maximum(np.sqrt(np.maximum(self.yM2 / self.N, 0)), 0.05)
        r = p1.vector_corr(ef['u'], ef['v'], pf['u'], pf['v'], ef['mask'])
        occ = self.occ_emp / max(self.occ_emp.sum(), EPS)
        op = self.occ_pred[j] / max(self.occ_pred[j].sum(), EPS)
        m = {'one_step_mse_main_norm': min(float(np.mean(self.err[j] / self.N / scale ** 2)), 1.0), 'occupancy_js_MR_PsiA': p1.js_divergence(occ + EPS, op + EPS), 'drift_local_rmse_loss_MR_PsiA': p1.drift_local_rmse_loss(ef, pf, ef['mask']), 'drift_direction_loss_MR_PsiA': 1.0 if not np.isfinite(r) else 0.5 * (1 - r), 'drift_magnitude_loss_MR_PsiA': p1.drift_magnitude_loss(ef, pf, ef['mask'])}
        m['objective_primary_score'] = p1.primary_objective_score(m)
        m['one_step_rmse_M'] = float(np.sqrt(self.err[j, 0] / self.N))
        m['one_step_rmse_Psi'] = float(np.sqrt(self.err[j, 1] / self.N))
        m['drift_vector_corr_MR_PsiA'] = r
        return m

    def tables(self, p1, split):
        emp, models = self.fields()
        S = self.count >= MIN_COUNT
        mass = self.mass[S].sum()
        rows = []
        trans = []
        grid = []
        Pep = normalized_transition(self.transitions[0])
        for j, name in enumerate(self.names):
            d2 = np.sum(self.mass[S, None] * (models[j, S] - emp[S]) ** 2, axis=0) / mass if mass > EPS else np.full(2, np.nan)
            fm = field_compare(emp, models[j], self.mass, S)
            row = {'split': split, 'model': name, 'rows': self.N, 'learner_mass': self.Umass, 'supported_cells': int(S.sum()), 'supported_mass_fraction': float(mass / max(self.mass.sum(), EPS)), 'D2': float(d2.sum()), 'D': float(np.sqrt(d2.sum())), 'D2_M': d2[0], 'D2_Psi': d2[1], **fm, **self.primary(p1, j), 'user_equal_rmse_M': float(np.sqrt(self.errw[j, 0] / self.Umass)), 'user_equal_rmse_Psi': float(np.sqrt(self.errw[j, 1] / self.Umass))}
            P = normalized_transition(self.transitions[j + 1])
            tv = 0.5 * np.sum(np.abs(P - Pep), axis=1)
            row['mean_row_TV'] = float(np.mean(tv))
            row['self_transition_r_descriptive'] = corr(np.diag(Pep), np.diag(P))
            rows.append(row)
            for i in range(6):
                top = np.flatnonzero(np.isclose(P[i], np.nanmax(P[i]), atol=1e-14, rtol=0)).tolist() if np.isfinite(P[i]).any() else []
                etop = np.flatnonzero(np.isclose(Pep[i], np.nanmax(Pep[i]), atol=1e-14, rtol=0)).tolist() if np.isfinite(Pep[i]).any() else []
                for dest in range(6):
                    trans.append({'split': split, 'model': name, 'origin': i, 'destination': dest, 'empirical_count': self.transitions[0, i, dest], 'model_count': self.transitions[j + 1, i, dest], 'empirical_probability': Pep[i, dest], 'model_probability': P[i, dest], 'difference': P[i, dest] - Pep[i, dest], 'row_TV': tv[i], 'model_top_destinations': str(top), 'empirical_top_destinations': str(etop), 'top_destination_sets_match': top == etop})
            for c in range(NCELL):
                grid.append({'split': split, 'model': name, 'cell': c, 'M_center': -1 + (c // 40 + 0.5) * 0.05, 'Psi_center': -1 + (c % 40 + 0.5) * 0.05, 'count': self.count[c], 'mass': self.mass[c], 'supported': bool(S[c]), 'emp_M': emp[c, 0], 'emp_Psi': emp[c, 1], 'model_M': models[j, c, 0], 'model_Psi': models[j, c, 1]})
        return (pd.DataFrame(rows), pd.DataFrame(trans), pd.DataFrame(grid))

def basic_interval(values, point, level=0.95, strict=True):
    v = np.asarray(values, float)
    nfinite = int(np.isfinite(v).sum())
    result = {'point': point, 'requested_replicates': len(v), 'finite_replicates': nfinite, 'level': level}
    if not np.isfinite(point) or not nfinite or (strict and nfinite != len(v)):
        return {**result, 'status': 'not_estimable', 'basic_low': np.nan, 'basic_high': np.nan, 'percentile_low': np.nan, 'percentile_high': np.nan, 'se_low': np.nan, 'se_high': np.nan}
    v = v[np.isfinite(v)]
    alpha = 1 - level
    q = np.quantile(v, [alpha / 2, 1 - alpha / 2])
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    se = norm.ppf(1 - alpha / 2) * sd
    bias = float(v.mean() - point)
    result.update(status='complete' if nfinite == len(values) else 'diagnostic_incomplete', basic_low=2 * point - q[1], basic_high=2 * point - q[0], percentile_low=q[0], percentile_high=q[1], se_low=point - se, se_high=point + se, bootstrap_mean=float(v.mean()), bootstrap_sd=sd, bootstrap_bias=bias, bias_over_sd=bias / sd if sd > EPS else np.nan)
    directional = lambda low, high: 1 if low > 0 else -1 if high < 0 else 0
    result['method_sign_conflict'] = len({directional(result[a], result[b]) for a, b in [('basic_low', 'basic_high'), ('percentile_low', 'percentile_high'), ('se_low', 'se_high')]}) > 1
    return result

def pair_diagnostics(emp, four, c3, ssl, W, S):
    raw = field_compare(four, ssl, W, S)
    correction = field_compare(four - c3, ssl - c3, W, S)
    direct = field_compare(four - emp, ssl - emp, W, S)
    a = four[S].ravel()
    b = ssl[S].ravel()
    t = emp[S].ravel()
    if len(t) >= 3 and np.isfinite(t).all() and (np.std(t) > EPS):
        T = np.column_stack([np.ones(len(t)), t])
        ra = a - T @ np.linalg.lstsq(T, a, rcond=None)[0]
        rb = b - T @ np.linalg.lstsq(T, b, rcond=None)[0]
        partial = corr(ra, rb)
    else:
        partial = np.nan
    out = {'raw_' + k: v for k, v in raw.items()}
    out.update({'C3_correction_' + k: v for k, v in correction.items()})
    out.update({'direct_empirical_error_' + k: v for k, v in direct.items()})
    out['partial_r_four_SSL_given_empirical_descriptive'] = partial
    out['common_target_product_benchmark_descriptive'] = corr(four[S], emp[S]) * corr(ssl[S], emp[S])
    out['same_subtraction_identity_error'] = float(np.max(np.abs((four - c3 - (ssl - c3) - (four - ssl))[S]))) if S.any() else np.nan
    if S.any() and W[S].sum() > EPS:
        for name, field in [('four_term', four), ('C3', c3), ('SSL', ssl)]:
            err = np.sum(W[S, None] * (field[S] - emp[S]) ** 2, axis=0) / W[S].sum()
            out['D2_' + name] = float(err.sum())
            out['D2_M_' + name] = err[0]
            out['D2_Psi_' + name] = err[1]
        out['improvement_four_over_C3'] = out['D2_C3'] - out['D2_four_term']
        out['improvement_SSL_over_C3'] = out['D2_C3'] - out['D2_SSL']
    return out

class LearnerStatistics:

    def __init__(self, dataset, predictions, names, partition, block_size=200000):
        self.names = tuple(names)
        self.users = np.asarray(dataset.users)
        self.U = len(self.users)
        self.K = len(names)
        ui = np.asarray(dataset.ui)
        cells = np.asarray(dataset.cells)
        w = np.asarray(dataset.w)
        flat = ui.astype(np.int64) * NCELL + cells
        unique, inverse = np.unique(flat, return_inverse=True)
        del flat
        ru = unique // NCELL
        co = unique % NCELL
        n = len(unique)

        def make(vals):
            grouped = np.bincount(inverse, weights=vals, minlength=n)
            return sparse.coo_matrix((grouped, (ru, co)), shape=(self.U, NCELL)).tocsr()
        mats = [make(np.ones(len(dataset))), make(w)]
        origin = np.asarray(dataset.F[:, :2])
        Y = np.asarray(dataset.Y)
        for a in range(2):
            mats.append(make(w * (Y[:, a] - origin[:, a])))
        for j in range(self.K):
            for a in range(2):
                mats.append(make(w * (np.asarray(predictions[:, j, a]) - Y[:, a])))
        os = np.empty(len(dataset), np.int64)
        ts = np.empty((len(dataset), self.K + 1), np.int64)
        for s in dataset.blocks(block_size):
            os[s] = assign_states(origin[s], partition)
            ts[s, 0] = assign_states(Y[s], partition)
            for j in range(self.K):
                ts[s, j + 1] = assign_states(predictions[s, j], partition)
        for j in range(self.K + 1):
            mats.append(sparse.coo_matrix((np.ones(len(dataset)), (ui, os * 6 + ts[:, j])), shape=(self.U, 36)).tocsr())
        self.matrix = sparse.hstack(mats, format='csr')
        self.matrix.eliminate_zeros()
        self.point_vector = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.fixed = self.point_vector[:NCELL] >= MIN_COUNT
        self.fixed_mass = float(self.point_vector[NCELL:2 * NCELL][self.fixed].sum())

    def parse(self, v):
        v = np.asarray(v).ravel()
        count = v[:NCELL]
        W = v[NCELL:2 * NCELL]
        emp = np.column_stack([v[2 * NCELL:3 * NCELL], v[3 * NCELL:4 * NCELL]])
        emp = np.divide(emp, W[:, None], out=np.full_like(emp, np.nan), where=W[:, None] > 0)
        residual = []
        off = 4 * NCELL
        for j in range(self.K):
            a = np.column_stack([v[off:off + NCELL], v[off + NCELL:off + 2 * NCELL]])
            off += 2 * NCELL
            residual.append(a)
        T = v[off:].reshape(self.K + 1, 6, 6)
        return (count, W, emp, np.asarray(residual), T)

    def summarize(self, v, replicate, include_pairs=False):
        count, W, emp, res, T = self.parse(v)
        records = []
        for contract, S in [('fixed', self.fixed), ('reselected', count >= MIN_COUNT)]:
            positive = S & (W > 0)
            mass = W[positive].sum()
            nempty = int(np.sum(S & (W <= 0)))
            d = {}
            for j, name in enumerate(self.names):
                axes = np.sum(res[j, positive] ** 2 / W[positive, None], axis=0) / mass if mass > EPS else np.full(2, np.nan)
                d[name] = axes
            row = {'replicate': replicate, 'support_contract': contract, 'empty_support_cells': nempty, 'supported_cells': int(S.sum()), 'positive_mass_cells': int(positive.sum()), 'support_mass': mass, 'retained_mass_fraction': mass / W.sum() if W.sum() > 0 else np.nan, 'support_jaccard': float((S & self.fixed).sum() / max((S | self.fixed).sum(), 1))}
            for name, axes in d.items():
                row['D2_' + name] = float(axes.sum())
                row['D2_M_' + name] = axes[0]
                row['D2_Psi_' + name] = axes[1]
            if 'four_term' in d:
                for name in ('C3', 'ARX'):
                    if name in d:
                        delta = d[name] - d['four_term']
                        row['Delta_' + name] = float(delta.sum())
                        row['Delta_M_' + name] = delta[0]
                        row['Delta_Psi_' + name] = delta[1]
            Pep = normalized_transition(T[0])
            for j, name in enumerate(self.names):
                P = normalized_transition(T[j + 1])
                tv = 0.5 * np.sum(np.abs(P - Pep), axis=1)
                row['mean_row_TV_' + name] = float(np.mean(tv))
                for st in range(6):
                    row[f'row_TV_{name}_S{st}'] = tv[st]
                    row[f'delta_Pii_{name}_S{st}'] = P[st, st] - Pep[st, st]
            if include_pairs:
                if S.any() and nempty == 0:
                    fs = emp[None, :, :] + np.divide(res, W[None, :, None], out=np.full_like(res, np.nan), where=W[None, :, None] > 0)
                    i = {n: j for j, n in enumerate(self.names)}
                    row.update(pair_diagnostics(emp, fs[i['four_term']], fs[i['C3']], fs[i['SSL']], W, S))
                else:
                    row['pair_correlations_status'] = 'undefined_empty_or_no_support'
            records.append(row)
        return records

    def replicate(self, index, include_pairs=False):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 112358, int(index)]))
        multiplicity = np.bincount(rng.integers(0, self.U, size=self.U), minlength=self.U).astype(float)
        v = np.asarray(self.matrix.T @ multiplicity).ravel()
        return self.summarize(v, index, include_pairs)

def run_bootstrap(stats, out, replicates, workers, include_pairs=False):
    out = Path(out)
    points = pd.DataFrame(stats.summarize(stats.point_vector, -1, include_pairs))
    write_csv(points, out / 'point_estimands.csv')
    np.save(out / 'bootstrap_users.npy', stats.users)
    save_json({'seed': SEED, 'replicates': replicates, 'generator': 'numpy.default_rng(SeedSequence([42,112358,replicate]))', 'sample': 'U integer draws with replacement, then whole-learner multiplicities', 'models_refitted': False, 'names': stats.names}, out / 'bootstrap_contract.json')
    records = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            it = ex.map(lambda r: stats.replicate(r, include_pairs), range(replicates))
            for i, rows in enumerate(it):
                records.extend(rows)
                if (i + 1) % 100 == 0:
                    log(f'Bootstrap {i + 1}/{replicates}')
    else:
        for i in range(replicates):
            records.extend(stats.replicate(i, include_pairs))
            if (i + 1) % 100 == 0:
                log(f'Bootstrap {i + 1}/{replicates}')
    b = pd.DataFrame(records)
    require(set(b['replicate']) == set(range(replicates)), 'Missing bootstrap indices')
    require(not b.duplicated(['replicate', 'support_contract']).any(), 'Duplicate bootstrap replicate')
    write_csv(b, out / 'paired_learner_bootstrap_replicates.csv.gz')
    summaries = []
    excluded = {'replicate', 'empty_support_cells', 'supported_cells', 'positive_mass_cells', 'support_mass', 'retained_mass_fraction', 'support_jaccard'}
    for contract, g in b.groupby('support_contract', sort=False):
        point = points[points.support_contract == contract].iloc[0]
        for name in g.select_dtypes(include='number').columns:
            if name in excluded:
                continue
            if name not in point or not np.isscalar(point[name]):
                continue
            for level in [0.95, 0.975] if name in ('Delta_C3', 'Delta_ARX') else [0.95]:
                summaries.append({'support_contract': contract, 'metric': name, 'primary_family': name in ('Delta_C3', 'Delta_ARX') and contract == 'fixed', 'interval_scale': 'raw; no Fisher transform and no [-1,1] clipping', **basic_interval(g[name].to_numpy(), float(point[name]), level, strict=True)})
    summary = pd.DataFrame(summaries)
    write_csv(summary, out / 'paired_intervals.csv')
    save_json({'requested_replicates': replicates, 'observed_rows': len(b), 'all_primary_replicates_finite': {k: bool(np.isfinite(b[k]).all()) for k in ('Delta_C3', 'Delta_ARX') if k in b}, 'invalid_statistics_are_retained': True, 'technical_completion_not_significance': True}, out / 'inference_audit.json')
    return summary
