from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
import argparse, datetime, gc, hashlib, json, shutil, time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
EXPECTED_RUN = 'a7fd95330a1f8764dc40cf5b0c6f86928f220d2c2665bc83530bbbd28d344072'
EXPECTED_CODE = 'aa978eeb556fe4181887f10b3f871990d05b28feba0f32c96144d139eb982837'
SEEDS = (42, 2026, 666, 606, 37, 4669)
NBOOT = 2000
EPS = 1e-12
PAIRS = (('E_to_M', 'E', 'M'), ('E_to_SSL', 'E', 'SSL'), ('M_to_SSL', 'M', 'SSL'))
DIMS = (('full', slice(None)), ('M', slice(0, 1)), ('Psi', slice(1, 2)))

def require(x, msg):
    if not bool(x):
        raise RuntimeError(msg)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x

def save(x, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(clean(x), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')

def write(df, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def log(msg):
    print(datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + msg, flush=True)

def close(a, b, label, atol=2e-10, rtol=2e-09):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    require(a.shape == b.shape, label + ' shape')
    require(np.array_equal(np.isfinite(a), np.isfinite(b)), label + ' finite mask')
    require(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + ' values')
    q = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[q] - b[q]))) if q.any() else 0.0

def metric_contract():
    return {'analysis': 'Post hoc C3-reference triplet extension, after prior results were observed', 'pairs': PAIRS, 'dimensions': ['full', 'M', 'Psi'], 'bootstrap_seed_models': [42], 'six_seed_point_estimates': SEEDS, 'replicates': NBOOT, 'models_refitted': False, 'vector_r': 'unweighted Pearson of row-major flattened supported components under the archived vector metric', 'speed_r': 'unweighted Pearson of Euclidean norms; per-axis absolute-value correlation', 'local_cosine': 'full 2D only; occupancy-weighted on norm-product > 1e-12; coverage recorded', 'scalar_sign_agreement': 'per-axis weighted signed cosine (+1/-1), not a 2D angle', 'amplitude_slope': 'sum W*a*b / sum W*a*a; reference is first member of named pair; denominator > EPS', 'energies': 'normalized by positive supported cell mass; raw reference denominator also reported', 'skill': 'E_to_M/E_to_SSL only: 1 - absolute_D2 / mean empirical-excess squared norm; NA if <=EPS', 'absolute_D2': 'all positive-mass support; empty cells have zero measure, never an imputed vector', 'vector_existence': 'full fixed support required; any empty support cell makes vector diagnostics NA', 'intervals': 'strict 2000/2000 finite; 95% basic plus percentile and point-centred SE diagnostics; no multiplicity-adjusted new claims', 'primary_comparisons': 'archived Delta_C3 and Delta_ARX primary comparisons remain unchanged; no new primary endpoints', 'scope': 'common empirical-anchor conditional fields; not learned-plane, not mechanistic identity'}

def triplet_metrics(emp, fields, W, S, field_compare):
    require(fields.shape[0] == 3, 'Expected four, C3, SSL fields')
    C = fields[1]
    values = {'E': emp - C, 'M': fields[0] - C, 'SSL': fields[2] - C}
    pos = S & (W > 0)
    mass = float(W[pos].sum())
    complete = bool(S.any() and np.all(W[S] > 0))
    out = {}
    for pair, an, bn in PAIRS:
        for dim, sl in DIMS:
            a = values[an][:, sl]
            b = values[bn][:, sl]
            prefix = f'P2__{pair}__{dim}__'
            er = float(np.sum(W[pos, None] * a[pos] ** 2)) if mass > EPS else np.nan
            et = float(np.sum(W[pos, None] * b[pos] ** 2)) if mass > EPS else np.nan
            cp = float(np.sum(W[pos, None] * a[pos] * b[pos])) if mass > EPS else np.nan
            err = float(np.sum(W[pos, None] * (b[pos] - a[pos]) ** 2) / mass) if mass > EPS else np.nan
            out[prefix + 'absolute_D2'] = err
            out[prefix + 'reference_energy'] = er / mass if mass > EPS else np.nan
            out[prefix + 'target_energy'] = et / mass if mass > EPS else np.nan
            out[prefix + 'cross_moment'] = cp / mass if mass > EPS else np.nan
            out[prefix + 'reference_denominator_raw'] = er
            if an == 'E':
                out[prefix + 'reference_relative_skill'] = 1 - err / (er / mass) if mass > EPS and er / mass > EPS else np.nan
            if complete:
                m = field_compare(a, b, W, S)
                av = a[S]
                bv = b[S]
                normok = np.linalg.norm(av, axis=1) * np.linalg.norm(bv, axis=1) > EPS
                m['local_cosine_valid_cells'] = int(normok.sum())
            else:
                m = {k: np.nan for k in ('vector_r', 'speed_r', 'local_cosine', 'amplitude_slope', 'local_cosine_mass_fraction', 'unweighted_rmse')}
                m['local_cosine_valid_cells'] = np.nan
            for key, val in m.items():
                name = 'scalar_sign_agreement' if key == 'local_cosine' and dim != 'full' else key
                out[prefix + name] = val
    return out

def independent_tests(field_compare):
    W = np.array([1.0, 2.0, 3.0, 4.0])
    S = np.ones(4, bool)
    e = np.array([[1.0, -2.0], [2.0, 1.0], [-3.0, 2.0], [4.0, -1.0]])
    c = np.array([[0.1, 0.2], [-0.3, 0.4], [0.2, -0.2], [0.1, -0.3]])
    f = np.stack([c + 2 * e, c, c + 3 * e])
    r = triplet_metrics(c + e, f, W, S, field_compare)
    close([r['P2__E_to_M__full__amplitude_slope'], r['P2__E_to_SSL__full__amplitude_slope'], r['P2__M_to_SSL__full__amplitude_slope']], [2, 3, 1.5], 'Known slopes')
    close([r['P2__E_to_M__full__vector_r'], r['P2__E_to_M__full__local_cosine']], [1, 1], 'Known directions')
    W0 = W.copy()
    W0[0] = 0
    t = triplet_metrics(c + e, f, W0, S, field_compare)
    require(np.isnan(t['P2__E_to_M__full__vector_r']) and np.isfinite(t['P2__E_to_M__full__absolute_D2']), 'Empty-support gate')
    z = triplet_metrics(c, np.stack([c, c, c]), W, S, field_compare)
    require(np.isnan(z['P2__E_to_M__full__reference_relative_skill']), 'Zero excess skill undefined')
    close(r['P2__E_to_M__full__absolute_D2'], np.sum(W[:, None] * (f[0] - (c + e)) ** 2) / W.sum(), 'Error identity')
    close(r['P2__M_to_SSL__full__absolute_D2'], np.sum(W[:, None] * (f[2] - f[0]) ** 2) / W.sum(), 'Pair gap identity')
    for pair, _, _ in PAIRS:
        close(r[f'P2__{pair}__full__absolute_D2'], sum((r[f'P2__{pair}__{d}__absolute_D2'] for d in ('M', 'Psi'))), 'Axis additivity')
    return {'known_slopes': True, 'empty_support': True, 'zero_reference': True, 'error_identities': True, 'axis_additivity': True}

def source_inventory(old, package, history, c, source_run):
    expected = {}
    for receipt in sorted(old.rglob('_SUCCESS.json')):
        r = read(receipt)
        require(r['run_hash'] == source_run, 'Prior run receipt identity')
        expected[receipt.resolve()] = sha(receipt)
        for rel, digest in r['outputs'].items():
            p = (receipt.parent / rel).resolve()
            require(p.is_relative_to(old), 'Receipt path escape')
            if p in expected:
                require(expected[p] == digest, 'Conflicting receipt SHA')
            expected[p] = digest

    def collect(o):
        if isinstance(o, dict):
            if {'path', 'sha256', 'size'}.issubset(o):
                p = Path(o['path']).resolve()
                if p.is_file():
                    if p in expected:
                        require(expected[p] == o['sha256'], 'Conflicting history SHA')
                    expected[p] = o['sha256']
            for v in o.values():
                collect(v)
        elif isinstance(o, list):
            for v in o:
                collect(v)
    collect(read(old / 'source_audit.json'))
    for receipt in old.rglob('_SUCCESS.json'):
        collect(read(receipt)['inputs'])
    for p in list(package.glob('*.py')) + list(package.glob('*.sh')) + [package / 'source_contract.json', package / 'README.md']:
        expected[p.resolve()] = sha(p)
    for p in (old / 'run_contract.json', old / 'source_audit.json', old / 'final/COMPLETE.json'):
        expected[p.resolve()] = sha(p)
    records = []
    for i, (p, digest) in enumerate(sorted(expected.items())):
        require(sha(p) == digest, 'Pre-run historical hash mismatch: ' + str(p))
        records.append({'path': str(p), 'sha256': digest, 'size': p.stat().st_size})
        if (i + 1) % 50 == 0:
            log(f'Pre-run source hashes {i + 1}/{len(expected)}')
    return records

def extras(ds, P, names, seed=None):
    n = 0
    sx = np.zeros((len(names), 2))
    sy = np.zeros(2)
    xx = np.zeros_like(sx)
    yy = np.zeros(2)
    xy = np.zeros_like(sx)
    for sl in ds.blocks():
        Y = np.asarray(ds.Y[sl])
        a = np.asarray(P[sl])
        n += len(Y)
        sx += a.sum(axis=0)
        sy += Y.sum(axis=0)
        xx += (a * a).sum(axis=0)
        yy += (Y * Y).sum(axis=0)
        xy += (a * Y[:, None, :]).sum(axis=0)
    den = np.sqrt(np.maximum(xx - sx * sx / n, 0) * np.maximum(yy - sy * sy / n, 0))
    r = np.divide(xy - sx * sy / n, den, out=np.full_like(xy, np.nan), where=den > EPS)
    return [{'model': name, 'seed': seed, 'rows': n, 'one_step_corr_M': r[i, 0], 'one_step_corr_Psi': r[i, 1], 'view': 'common empirical anchors; interval Pearson; no new inferential test'} for i, name in enumerate(names)]

def legacy_compare(new, old, label):
    new = new.sort_values(['replicate', 'support_contract']).reset_index(drop=True)
    old = old.sort_values(['replicate', 'support_contract']).reset_index(drop=True)
    require(new[['replicate', 'support_contract']].equals(old[['replicate', 'support_contract']]), label + ' keys')
    result = {}
    for k in old.select_dtypes(include='number').columns:
        require(k in new, label + ' missing ' + k)
        result[k] = close(new[k], old[k], label + ' ' + k)
    return {'numeric_columns': len(result), 'max_absolute_error': max(result.values()), 'by_column': result, 'finite_masks_identical': True}

def make_figures(out, grid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = grid[grid.model == 'four_term'].sort_values('cell')
    S = g.supported.to_numpy(bool)
    pos = g[['M_center', 'Psi_center']].to_numpy()
    emp = g[['emp_M', 'emp_Psi']].to_numpy()
    fields = {name: grid[grid.model == name].sort_values('cell')[['model_M', 'model_Psi']].to_numpy() for name in ('four_term', 'C3', 'SSL')}
    panels = [('Empirical', emp), ('C3', fields['C3']), ('Four-term', fields['four_term']), ('Event-SSL (empirical anchor)', fields['SSL'])]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
    for ax, (name, f) in zip(axs, panels):
        ax.quiver(pos[S, 0], pos[S, 1], f[S, 0], f[S, 1], angles='xy', scale_units='xy', scale=1, width=0.003)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal', title=name, xlabel='Response evidence M')
    axs[0].set_ylabel('Exposure alignment Psi')
    fig.suptitle('Frozen conditional fields: identical support and arrow scale; seed 42')
    fig.tight_layout()
    fig.savefig(out / 'raw_fields_seed42.png', dpi=220)
    fig.savefig(out / 'raw_fields_seed42.svg')
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.7), sharex=True, sharey=True)
    panels = [('Empirical - C3', emp - fields['C3']), ('Four-term - C3', fields['four_term'] - fields['C3']), ('Event-SSL - C3', fields['SSL'] - fields['C3'])]
    for ax, (name, f) in zip(axs, panels):
        ax.quiver(pos[S, 0], pos[S, 1], f[S, 0], f[S, 1], angles='xy', scale_units='xy', scale=1, width=0.003)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal', title=name, xlabel='Response evidence M')
    axs[0].set_ylabel('Exposure alignment Psi')
    fig.suptitle('C3-reference fields: identical support and arrow scale; no per-panel rescaling')
    fig.tight_layout()
    fig.savefig(out / 'C3_excess_fields_seed42.png', dpi=220)
    fig.savefig(out / 'C3_excess_fields_seed42.svg')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[3] / 's2_minimal_mechanism/robustness/conditional_baselines')
    ap.add_argument('--history', type=Path, required=True)
    ap.add_argument('--fourterm-root', type=Path, help='Completed conditional-baseline output root')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--self-test-only', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='Full-row replay smoke on 21 original draws including an empty fixed-support draw; not formal evidence')
    args = ap.parse_args()
    package = args.package.resolve()
    history = args.history.resolve()
    old = args.fourterm_root.resolve() if args.fourterm_root else history / 'four_term_mechanism_revision'
    out = args.output.resolve()
    require(not out.exists(), 'Output must be new: ' + str(out))
    require(not out.is_relative_to(old) and (not package.is_relative_to(out)), 'Unsafe output path')
    sys.path.insert(0, str(package))
    import ft_common as c
    from ft_data import load_partition
    from ft_pipeline import read_neural_join, neural_paths, neural_provenance, subset_dataset
    from ft_statistics import LearnerStatistics, Evaluator, field_compare, pair_diagnostics, basic_interval
    compat = read(Path(__file__).with_name('compatibility_contract.json'))
    require(c.code_digest() == compat['public_fourterm_code_digest'], 'Conditional-baseline implementation does not match the public compatibility contract')
    tests = independent_tests(field_compare)
    if args.self_test_only:
        print(json.dumps(tests))
        return
    source_contract = read(old / 'run_contract.json')
    source_run = source_contract['run_hash']
    source_code = source_contract['protocol']['code_digest']
    require(c.objhash(source_contract['protocol']) == source_run, 'Invalid source run contract digest')
    recorded = source_run == EXPECTED_RUN and source_code == EXPECTED_CODE
    current = source_code == c.code_digest() and source_contract['protocol'].get('bootstrap_replicates') == NBOOT and (source_contract['protocol'].get('seed') == 42) and (not source_contract['protocol'].get('smoke'))
    require(recorded or current, 'Source outputs do not match the recorded or public execution contract')
    out.mkdir(parents=True)
    start = time.perf_counter()
    source = read(old / 'source_audit.json')
    execution_sources = {str(p.resolve()): sha(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}
    require(any((Path(p).name == 'p1_reports.py' for p in execution_sources)), 'P1 reporting helper missing')
    olddraw_source = old / 'cross_model/seed42_uncertainty/paired_learner_bootstrap_replicates.csv.gz'
    olddraw_all = pd.read_csv(olddraw_source, float_precision='round_trip')
    indices = list(range(NBOOT))
    seeds = SEEDS
    if args.smoke:
        first_empty = int(olddraw_all.loc[(olddraw_all.support_contract == 'fixed') & (olddraw_all.empty_support_cells > 0), 'replicate'].iloc[0])
        indices = sorted(set(range(20)) | {first_empty})
        seeds = (42,)
    contract = {**metric_contract(), 'source_run': source_run, 'source_code': source_code, 'execution_code_sha256': sha(__file__), 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'execution_sources': execution_sources, 'smoke_only': args.smoke, 'actual_replicate_indices': indices, 'actual_point_seeds': seeds, 'replicates': len(indices), 'formal_target_replicates': NBOOT, 'intervals': 'SMOKE ONLY: strict finite repeats over selected old draw indices; not paper evidence' if args.smoke else metric_contract()['intervals'], 'numpy_version': np.__version__, 'pandas_version': pd.__version__, 'new_directory': str(out), 'rng': 'default_rng(SeedSequence([42,112358,replicate])) with sorted original users; verified against every saved M3 draw'}
    save(contract, out / 'PROTOCOL.json')
    save(tests, out / 'mathematical_tests.json')
    before = source_inventory(old, package, history, c, source_run)
    save(before, out / 'history_hashes_before.json')
    partition = load_partition(history, source['partition_centers_path'])
    p1 = c.import_source(source['frozen_source'], out / 'runtime_cache')
    ctx = SimpleNamespace(chunk_size=200000, history=history, provenance=source)
    ds = c.Dataset(old / 'prepared/B_confirm')
    base = np.load(old / 'confirmation/predictions.npy', mmap_mode='r')
    require(base.shape == (len(ds), 5, 2) and len(ds) == 3233208 and (len(ds.users) == 56195), 'Wrong confirmation data')
    extra = extras(ds, base, c.MODELS)
    paths = neural_paths(history)
    allpoints = []
    oldpoints = []
    joins = []
    grid42 = None
    prior = pd.read_csv(old / 'cross_model/C3_referenced_cross_model_by_seed.csv', float_precision='round_trip')
    for seed in seeds:
        log(f'Joining frozen seed {seed}')
        provenance = neural_provenance(ctx, seed, paths[seed])
        idx, npred, audit = read_neural_join(ctx, ds, paths[seed])
        require(len(idx) == len(ds) and np.array_equal(idx, np.arange(len(ds))), 'Common sample must match full original M3')
        common = subset_dataset(ds, idx)
        P = np.stack([base[idx, 0], base[idx, 1], npred], axis=1)
        ev = Evaluator(('four_term', 'C3', 'SSL'), partition)
        for sl in common.blocks():
            ev.add(common.F[sl], common.Y[sl], P[sl], common.w[sl])
        emp, fs = ev.fields()
        S = ev.count >= 30
        raw = pair_diagnostics(emp, fs[0], fs[1], fs[2], ev.mass, S)
        expected = prior[prior.seed == seed].iloc[0]
        for k, v in raw.items():
            if isinstance(v, (int, float, np.number)) and k in expected:
                close(v, expected[k], f'Seed{seed} {k}')
        oldpoints.append({'seed': seed, **raw})
        allpoints.append({'seed': seed, 'rows': len(common), 'users': len(common.users), 'supported_cells': int(S.sum()), **triplet_metrics(emp, fs, ev.mass, S, field_compare)})
        joins.append({'seed': seed, **audit, 'neural_provenance': provenance})
        full_metrics, transitions, _ = ev.tables(p1, f'B_confirm_common_seed{seed}')
        write(full_metrics, out / f'original_metrics/seed_{seed}.csv')
        write(transitions, out / f'original_metrics/seed_{seed}_transitions.csv')
        erow = extras(common, P[:, 2:3, :], ['SSL'], seed)[0]
        extra.append({**full_metrics[full_metrics.model == 'SSL'].iloc[0].to_dict(), **erow})
        rows = []
        for j, name in enumerate(('four_term', 'C3', 'SSL')):
            for g in range(1600):
                rows.append({'model': name, 'cell': g, 'M_center': -1 + (g // 40 + 0.5) * 0.05, 'Psi_center': -1 + (g % 40 + 0.5) * 0.05, 'count': ev.count[g], 'mass': ev.mass[g], 'supported': bool(S[g]), 'emp_M': emp[g, 0], 'emp_Psi': emp[g, 1], 'model_M': fs[j, g, 0], 'model_Psi': fs[j, g, 1]})
        grid = pd.DataFrame(rows)
        write(grid, out / f'grids/seed_{seed}.csv.gz')
        if seed == 42:
            grid42 = grid

            class TripletStats(LearnerStatistics):

                def summarize(self, v, replicate, include_pairs=True):
                    records = super().summarize(v, replicate, True)
                    count, W, e, res, _ = self.parse(v)
                    f = e[None, :, :] + np.divide(res, W[None, :, None], out=np.full_like(res, np.nan), where=W[None, :, None] > 0)
                    for row in records:
                        mask = self.fixed if row['support_contract'] == 'fixed' else count >= 30
                        row.update(triplet_metrics(e, f, W, mask, field_compare))
                        pos = mask & (W > 0)
                        mass = W[pos].sum()
                        expected_gap = np.sum((res[2, pos] - res[0, pos]) ** 2 / W[pos, None]) / mass
                        close(row['P2__M_to_SSL__full__absolute_D2'], expected_gap, 'Every-draw model-gap identity', atol=1e-11, rtol=1e-08)
                    return records
            log('Building original learner sufficient statistics')
            stats = TripletStats(common, P, ('four_term', 'C3', 'SSL'), partition)
            oldusers = np.load(old / 'cross_model/seed42_uncertainty/bootstrap_users.npy')
            require(np.array_equal(stats.users, oldusers), 'Bootstrap user order differs')
            np.save(out / 'bootstrap_users.npy', stats.users)
            points = pd.DataFrame(stats.summarize(stats.point_vector, -1))
            oldpt = pd.read_csv(old / 'cross_model/seed42_uncertainty/point_estimands.csv', float_precision='round_trip')
            pointaudit = legacy_compare(points, oldpt, 'M3 original point')
            write(points, out / 'point_estimands_seed42.csv')
            sparse.save_npz(out / 'learner_sufficient_statistics_seed42.npz', stats.matrix, compressed=True)
            np.save(out / 'point_sufficient_vector.npy', stats.point_vector)
            olddraw = olddraw_all[olddraw_all.replicate.isin(indices)].copy()
            records = []
            log(f'Starting exact replay of {len(indices)} paired learner draws; smoke={args.smoke}')
            with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=args.workers) as pool:
                for i, r in enumerate(pool.map(lambda j: stats.replicate(j, True), indices)):
                    records.extend(r)
                    if (i + 1) % 100 == 0:
                        log(f'Paired bootstrap {i + 1}/{len(indices)}')
            draws = pd.DataFrame(records)
            drawaudit = legacy_compare(draws, olddraw, 'All selected original M3 draws')
            write(draws, out / 'paired_bootstrap_replicates.csv.gz')
            summaries = []
            for support, g in draws.groupby('support_contract', sort=False):
                p = points[points.support_contract == support].iloc[0]
                for k in [x for x in draws.columns if x.startswith('P2__')]:
                    _, pair, dimension, metric = k.split('__')
                    summaries.append({'support_contract': support, 'pair': pair, 'dimension': dimension, 'metric': metric, 'column': k, 'analysis_role': 'post hoc descriptive/sensitivity; not new multiplicity-controlled primary', **basic_interval(g[k].to_numpy(), float(p[k]), 0.95, strict=True)})
            write(pd.DataFrame(summaries), out / 'paired_intervals.csv')
            identities = {}
            for pair, model in [('E_to_M', 'four_term'), ('E_to_SSL', 'SSL')]:
                for dim, suffix in [('full', ''), ('M', 'M_'), ('Psi', 'Psi_')]:
                    identities[f'{pair}_{dim}'] = close(draws[f'P2__{pair}__{dim}__absolute_D2'], draws[f'D2_{suffix}{model}'], 'Every-draw empirical-error identity', atol=1e-11, rtol=1e-08)
            for pair, _, _ in PAIRS:
                close(draws[f'P2__{pair}__full__absolute_D2'], draws[f'P2__{pair}__M__absolute_D2'] + draws[f'P2__{pair}__Psi__absolute_D2'], 'Every-draw axis sum')
            save({'point': pointaudit, 'all_replayed_draws': drawaudit, 'replicate_count': len(indices), 'smoke_only': args.smoke, 'error_identity_max_differences': identities, 'user_order_exact': True, 'seed42_old_and_new_draws_same': True, 'all_draws_retained': True}, out / 'replay_audit.json')
            del stats, records, draws, olddraw
        del common, P, ev, npred
        gc.collect()
    write(pd.DataFrame(allpoints), out / 'three_way_point_metrics_by_seed.csv')
    write(pd.DataFrame(oldpoints), out / 'original_pair_metrics_reproduced.csv')
    write(pd.DataFrame(extra), out / 'P1_extra_metrics.csv')
    save(joins, out / 'join_audits.json')
    long = []
    for row in allpoints:
        for k, v in row.items():
            if k.startswith('P2__'):
                _, pair, dimension, metric = k.split('__')
                long.append({'seed': row['seed'], 'pair': pair, 'dimension': dimension, 'metric': metric, 'value': v})
    write(pd.DataFrame(long), out / 'three_way_point_metrics_long.csv')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from p1_reports import build_p1
    build_p1(old, out, out / 'P1_extra_metrics.csv')
    figdir = out / 'figures'
    figdir.mkdir()
    make_figures(figdir, grid42)
    log('Verifying historical files remain unchanged')
    changed = []
    for r in before:
        if sha(r['path']) != r['sha256']:
            changed.append(r['path'])
    require(not changed, 'Historical file modified: ' + repr(changed))
    require(all((sha(p) == digest for p, digest in execution_sources.items())), 'New execution source changed during run')
    save({'files_checked': len(before), 'bytes_checked': sum((r['size'] for r in before)), 'unchanged': True, 'changed': changed}, out / 'history_unchanged.json')
    outputs = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    save({'technical_completion': True, 'scientific_positive_result_required': False, 'models_refitted': False, 'run_source': source_run, 'code_sha256': sha(__file__), 'execution_sources': execution_sources, 'bootstrap_replicates': len(indices), 'bootstrap_seed': 42, 'smoke_only': args.smoke, 'point_seeds': seeds, 'elapsed_seconds': time.perf_counter() - start, 'outputs': outputs}, out / 'COMPLETE.json')
    log('COMPLETE: new analysis only; original outputs unchanged')
if __name__ == '__main__':
    main()
