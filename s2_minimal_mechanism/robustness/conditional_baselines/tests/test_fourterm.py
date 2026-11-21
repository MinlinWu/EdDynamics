from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft_common import *
from ft_models import *
from ft_data import grid_cells, assign_states
from ft_statistics import *
from synthetic_fixture import load_p1, build_fixture
SOURCE = os.environ.get('FOURTERM_TEST_SOURCE')

class MemoryDataset:

    def __init__(self, F, Y, keys):
        self.F = np.array(F)
        self.Y = np.array(Y)
        self.keys = np.array(keys, dtype=np.int64)
        self.users, self.ui, c = np.unique(self.keys[:, 0], return_inverse=True, return_counts=True)
        self.w = 1.0 / c[self.ui]
        self.cells = grid_cells(self.F[:, :2])

    def __len__(self):
        return len(self.F)

    def blocks(self, size=200000):
        for i in range(0, len(self), size):
            yield slice(i, min(i + size, len(self)))

class MathematicsTests(unittest.TestCase):

    def setUp(self):
        self.p = {'theta0': -0.17, 'thetaM': 0.64, 'thetaPsi': 0.0, 'thetaMPsi': 0.0, 'phi0': -1.35, 'deltaS': 6.0, 'phiPsi': 0.0, 'lambdaR': 0.46, 'lambdaA': 1.1, 'lambdaI': 0.85}
        self.c = {'eta': 20.0, 'tau_response_days': 10.0, 'tau_activity_days': 10.0, 'residual_mass_per_answer': 0.3865, 'lambda_E': 10.0, 'response_signed_gain': 0.9, 'alignment_signed_gain': 0.9537}
        rng = np.random.default_rng(2)
        n = 256
        self.F = rng.uniform(0, 3, (n, len(FEATURES)))
        self.F[:, :2] = rng.uniform(-1, 1, (n, 2))
        self.F[:, 2:4] = rng.uniform(0, 20, (n, 2))
        self.F[:, 4] = self.F[:, 1] * self.F[:, 3]
        self.K = np.column_stack([np.repeat(np.arange(16), 16), np.tile(np.arange(16), 16)])
        self.part = {'mean': np.zeros(2), 'scale': np.ones(2), 'centers': np.array([[-0.8, -0.8], [-0.5, 0.5], [0, -0.5], [0.2, 0.2], [0.6, -0.6], [0.8, 0.8]])}

    def test_01_closed_boundary_grid(self):
        X = np.array([[-1, -1], [1, 1], [0, 0], [1, -1]])
        np.testing.assert_array_equal(grid_cells(X), [0, 1599, 820, 1560])

    def test_02_grid_rejects_out_of_range(self):
        with self.assertRaises(AuditError):
            grid_cells(np.array([[1.01, 0]]))

    def test_03_affine_constants_interior(self):
        b, A = c3_basis(self.F, self.p, self.c)
        c = np.array([0.25, -0.8, 0.99])
        np.testing.assert_allclose(b + np.einsum('nij,j->ni', A, c), constant_predict(self.F, c, self.p, self.c), atol=1e-14)

    def test_04_affine_constants_at_bounds(self):
        b, A = c3_basis(self.F, self.p, self.c)
        for c in ([1, -1, 1], [-1, 1, -1]):
            np.testing.assert_allclose(b + np.einsum('nij,j->ni', A, c), constant_predict(self.F, c, self.p, self.c), atol=1e-14)

    def test_05_long_gap_EPS_branch(self):
        F = self.F[:1].copy()
        F[:, 11] = 100000.0
        np.testing.assert_array_equal(constant_predict(F, [1, 1, 1], self.p, self.c), np.zeros((1, 2)))

    def test_06_target_not_a_predictor(self):
        with self.assertRaises(AuditError):
            constant_predict(np.column_stack([self.F, np.zeros(len(self.F))]), [0, 0, 0], self.p, self.c)

    def test_07_nonfinite_rejected(self):
        F = self.F.copy()
        F[3, 2] = np.nan
        with self.assertRaises(AuditError):
            constant_predict(F, [0, 0, 0], self.p, self.c)

    def test_08_bounded_solver_full_rank(self):
        A = np.eye(3)
        b = np.array([2, -0.2, -3])
        c, a = bounded_min_norm(A, b)
        np.testing.assert_allclose(c, [1, -0.2, -1])
        self.assertEqual(a['rank'], 3)

    def test_09_bounded_solver_min_norm(self):
        c, a = bounded_min_norm(np.array([[1, 1, 0.0]]), np.array([0.8]))
        np.testing.assert_allclose(c, [0.4, 0.4, 0], atol=1e-10)

    def test_10_ridge_average_loss_units(self):
        F = self.F.copy()
        Y = F[:, :2] + np.column_stack([0.3 * F[:, 5], -0.2 * F[:, 6]])
        d = MemoryDataset(F, Y, self.K)
        m = ridge_fit([d], 0.1, 31)
        z = (F[:, m['active']] - np.array(m['mean'])[m['active']]) / np.array(m['scale'])[m['active']]
        yy = Y - F[:, :2]
        expected = np.linalg.solve(z.T @ z / len(z) + 0.1 * np.eye(z.shape[1]), z.T @ (yy - yy.mean(axis=0)) / len(z))
        np.testing.assert_allclose(m['coef'], expected, atol=1e-10)

    def test_11_ridge_zero_variance_and_collinearity(self):
        F = self.F.copy()
        F[:, 4] = 1
        F[:, 8] = F[:, 7]
        d = MemoryDataset(F, F[:, :2], self.K)
        m = ridge_fit([d], 0, 33)
        self.assertFalse(m['active'][4])
        np.testing.assert_allclose(ridge_predict(F, m), F[:, :2], atol=1e-12)

    def test_12_ridge_chunk_invariance(self):
        d = MemoryDataset(self.F, self.F[:, :2] * 0.9, self.K)
        a = ridge_fit([d], 0.01, 17)
        b = ridge_fit([d], 0.01, 256)
        np.testing.assert_allclose(ridge_predict(self.F, a), ridge_predict(self.F, b), atol=1e-10)

    def test_13_decomposition_addback_edges(self):
        F = self.F.copy()
        F[0, 2] = 0
        F[1, 11] = 100000.0
        F[2, 4] += 0.03
        P = constant_predict(F, [0, -0.8, 0.9], self.p, self.c)
        c = decompose(F, P, self.p, self.c)
        np.testing.assert_allclose(c[:, :3].sum(axis=1), P[:, 0] - F[:, 0], atol=1e-12)
        np.testing.assert_allclose(c[:, 3:].sum(axis=1), P[:, 1] - F[:, 1], atol=1e-12)

    def test_14_idle_dilutes_negative_Psi(self):
        F = np.zeros((2, len(FEATURES)))
        F[:, 1] = -0.5
        F[:, 2] = 5
        F[:, 3] = 2
        F[:, 4] = -1
        F[1, 10] = 1
        P = constant_predict(F, [0, 0, 0], self.p, self.c)
        self.assertGreater(P[1, 1], P[0, 1])
        self.assertLess(P[1, 1], 0)

    def test_15_basic_raw_scale_not_bounded(self):
        a = basic_interval(np.array([2.0, 3.0, 4.0]), 3.0, 0.95)
        self.assertGreater(a['basic_high'], 1)

    def test_16_invalid_replicates_not_filtered(self):
        a = basic_interval([1.0, np.nan, 2.0], 1.5)
        self.assertEqual(a['status'], 'not_estimable')
        self.assertEqual(a['finite_replicates'], 2)

    def test_17_correlation_zero_variance_is_NA(self):
        self.assertTrue(np.isnan(corr(np.ones(5), np.arange(5))))

    def test_18_structured_join(self):
        k = np.array([[1, 2], [1, 5], [3, 0]])
        idx, ok = key_positions(k, np.array([[1, 5], [2, 0], [3, 0]]))
        np.testing.assert_array_equal(ok, [True, False, True])
        np.testing.assert_array_equal(idx[ok], [1, 2])

    def test_19_sparse_multiplicity(self):
        F = np.tile(self.F[:2], (40, 1))
        F[:, :2] = [0.125, -0.375]
        keys = np.column_stack([np.repeat([10, 20], 40), np.tile(np.arange(40), 2)])
        Y = F[:, :2] + np.array([0.1, -0.05])
        D = MemoryDataset(F, Y, keys)
        P = np.stack([Y + 0.01, Y + 0.03], axis=1)
        st = LearnerStatistics(D, P, ['four_term', 'C3'], self.part)
        mult = np.array([2.0, 0.0])
        v = st.matrix.T @ mult
        ct, W, emp, res, T = st.parse(v)
        self.assertAlmostEqual(ct.sum(), 80)
        self.assertAlmostEqual(W.sum(), 2.0)
        row = st.summarize(v, 0)[0]
        self.assertAlmostEqual(row['D2_four_term'], 0.0002, places=10)
        self.assertAlmostEqual(T[0].sum(), 80)

    def test_20_pair_subtraction_identity(self):
        x = np.arange(8.0).reshape(4, 2)
        a = x * 0.9
        b = x * 0.95
        c = x * 0.3
        r = pair_diagnostics(x, a, c, b, np.ones(4), np.ones(4, bool))
        self.assertLess(r['same_subtraction_identity_error'], 1e-12)

    def test_21_artifact_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = atomic_stage(td, 'x', 'hash', {}, lambda d: (d / 'a.txt').write_text('a'))
            (p / 'a.txt').write_text('b')
            with self.assertRaises(AuditError):
                atomic_stage(td, 'x', 'hash', {}, lambda d: None)

    def test_22_bootstrap_rng_worker_invariance(self):
        F = np.tile(self.F[:2], (40, 1))
        F[:, :2] = [0.125, -0.375]
        K = np.column_stack([np.repeat([1, 2], 40), np.tile(np.arange(40), 2)])
        d = MemoryDataset(F, F[:, :2] + 0.01, K)
        P = np.stack([d.Y + 0.01, d.Y + 0.03], axis=1)
        st = LearnerStatistics(d, P, ['four_term', 'C3'], self.part)
        a = [st.replicate(i) for i in range(8)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(3) as ex:
            b = list(ex.map(st.replicate, range(8)))
        self.assertEqual(objhash(a), objhash(b))

    def test_27_negative_method_conflict_is_not_hidden(self):
        r = basic_interval(np.linspace(-0.01, 0.05, 200), -0.1, 0.95)
        self.assertLess(r['basic_high'], 0)
        self.assertGreater(r['percentile_high'], 0)
        self.assertTrue(r['method_sign_conflict'])

    def test_28_failed_stage_never_has_success_receipt(self):
        with tempfile.TemporaryDirectory() as td:

            def fail(d):
                (d / 'partial.txt').write_text('not complete')
                raise AuditError('injected failure')
            with self.assertRaises(AuditError):
                atomic_stage(td, 'fail', 'hash', {}, fail)
            self.assertFalse((Path(td) / 'fail/_SUCCESS.json').exists())

    def test_29_empty_fixed_cell_has_zero_measure_not_fake_drift(self):
        F = np.tile(self.F[:1], (80, 1))
        F[:40, :2] = [0.125, -0.375]
        F[40:, :2] = [0.575, 0.525]
        K = np.column_stack([np.repeat([1, 2], 40), np.tile(np.arange(40), 2)])
        d = MemoryDataset(F, F[:, :2] + 0.01, K)
        P = np.stack([d.Y + 0.01, d.Y + 0.03], axis=1)
        st = LearnerStatistics(d, P, ['four_term', 'C3'], self.part)
        v = st.matrix.T @ np.array([2.0, 0.0])
        count, W, emp, res, T = st.parse(v)
        row = st.summarize(v, 0)[0]
        self.assertEqual(row['empty_support_cells'], 1)
        self.assertTrue(np.isnan(emp[st.fixed & (W == 0)]).all())
        self.assertAlmostEqual(row['D2_four_term'], 0.0002, places=12)

@unittest.skipUnless(SOURCE, 'Set FOURTERM_TEST_SOURCE to the unchanged original Phase-1 source')
class OriginalSourceTests(unittest.TestCase):
    setUp = MathematicsTests.setUp

    @classmethod
    def setUpClass(cls):
        cls.p1 = load_p1(Path(SOURCE), 'unit_original_p1')

    def test_23_constant_scalar_equivalence(self):
        F = self.F.copy()
        F[0, 2] = 0
        F[1, 11] = 100000.0
        F[2, 3] = 1e-15
        F[2, 4] = 0
        c = np.array([0.2, -0.7, 0.8])
        p = c3_coefficients_to_four(c, self.p)
        a, _ = four_predict(self.p1, F, self.K, p, self.c)
        np.testing.assert_allclose(a, constant_predict(F, c, self.p, self.c), atol=1e-13)

    def test_24_Psi_class_contains_frozen_four(self):
        a, _ = four_predict(self.p1, self.F, self.K, self.p, self.c)
        c = [0, np.tanh(self.p['phi0']), np.tanh(self.p['phi0'] + self.p['deltaS'])]
        np.testing.assert_allclose(a[:, 1], constant_predict(self.F, c, self.p, self.c)[:, 1], atol=1e-13)

    def test_25_grid_exact_original(self):
        vals = np.vstack([self.F[:, :2], [[-1, -1], [1, 1], [0, 0]]])
        ref = self.p1.digitize_closed_right(vals[:, 0], BINS) * 40 + self.p1.digitize_closed_right(vals[:, 1], BINS)
        np.testing.assert_array_equal(ref, grid_cells(vals))

    def test_26_primary_five_terms_exact(self):
        F = np.tile(self.F, (4, 1))
        F[:, :2] = np.tile(np.array([[-0.775, -0.725], [-0.325, -0.375], [0.125, 0.075], [0.575, 0.525]]), (len(F) // 4, 1))
        keys = np.column_stack([np.repeat(np.arange(32), 32), np.tile(np.arange(32), 32)])
        P, ph = four_predict(self.p1, F, keys, self.p, self.c)
        Y = np.clip(P + 0.01, -1, 1)
        D = MemoryDataset(F, Y, keys)
        ev = Evaluator(['four'], self.part)
        for s in D.blocks(55):
            ev.add(F[s], Y[s], P[s, None, :], D.w[s])
        ef = self.p1.field_stats_from_arrays_weighted(F[:, 0], F[:, 1], Y[:, 0] - F[:, 0], Y[:, 1] - F[:, 1], D.w)
        pf = self.p1.field_stats_from_arrays_weighted(F[:, 0], F[:, 1], P[:, 0] - F[:, 0], P[:, 1] - F[:, 1], D.w)
        r = self.p1.vector_corr(ef['u'], ef['v'], pf['u'], pf['v'], ef['mask'])
        ref = {'one_step_mse_main_norm': min(float(np.mean((P - Y) ** 2 / np.maximum(Y.std(axis=0), 0.05) ** 2)), 1.0), 'occupancy_js_MR_PsiA': self.p1.js_divergence(self.p1.occupancy_grid_weighted(Y[:, 0], Y[:, 1], D.w) + EPS, self.p1.occupancy_grid_weighted(P[:, 0], P[:, 1], D.w) + EPS), 'drift_local_rmse_loss_MR_PsiA': self.p1.drift_local_rmse_loss(ef, pf, ef['mask']), 'drift_direction_loss_MR_PsiA': 0.5 * (1 - r), 'drift_magnitude_loss_MR_PsiA': self.p1.drift_magnitude_loss(ef, pf, ef['mask'])}
        got = ev.primary(self.p1, 0)
        for k, v in ref.items():
            self.assertAlmostEqual(got[k], v, places=10, msg=k)

@unittest.skipUnless(SOURCE, 'Source needed for failure-path integration checks')
class ProvenanceAndFailureTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from types import SimpleNamespace
        from ft_data import prepare_dataset, load_partition
        cls.temp = tempfile.TemporaryDirectory()
        cls.h = Path(cls.temp.name) / 'history'
        build_fixture(cls.h, Path(SOURCE), users=16, steps=48)
        p1, p, tr, fc, prov = validate_sources(cls.h, Path(cls.temp.name) / 'runtime')
        cls.ctx = SimpleNamespace(history=cls.h, p1=p1, params=p, train_cal=tr, final_cal=fc, provenance=prov, smoke=False, chunk_size=37, partition=load_partition(cls.h))
        cls.cache = Path(cls.temp.name) / 'prepared'
        cls.cache.mkdir()
        prepare_dataset(p1, cls.h, 'B_confirm', fc['eta'], cls.cache, 71)
        cls.ds = Dataset(cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @contextlib.contextmanager
    def changed(self, path, mutator):
        old = path.read_bytes()
        try:
            mutator(path)
            yield
        finally:
            path.write_bytes(old)

    def test_30_changed_manifest_checksum_blocks(self):
        p = self.h / 'stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json'
        with self.changed(p, lambda f: f.write_text(f.read_text() + ' ')):
            with self.assertRaises(AuditError):
                validate_sources(self.h, Path(self.temp.name) / 'runtime')

    def test_31_changed_frozen_source_blocks(self):
        p = self.h / 'stage2_phase2_freeze/metadata/frozen_phase1_implementation.py'
        with self.changed(p, lambda f: f.write_text(f.read_text() + '\n# injected change\n')):
            with self.assertRaises(AuditError):
                validate_sources(self.h, Path(self.temp.name) / 'runtime')

    def test_32_phase3_update_guardrail_blocks(self):
        from ft_pipeline import phase3_provenance
        m = self.h / 'stage2_phase3_confirm/metadata/phase3_confirmation_manifest.json'
        s = m.with_name('phase3_confirmation_manifest.sha256.json')

        def mutate(p):
            data = load_json(p)
            data['guardrails']['B_confirm_used_for_update'] = True
            save_json(data, p)
        with self.changed(m, mutate), self.changed(s, lambda p: save_json({'manifest_sha256': sha(m)}, p)):
            with self.assertRaises(AuditError):
                phase3_provenance(self.ctx)

    def test_33_neural_checkpoint_tamper_blocks(self):
        from ft_pipeline import neural_paths, neural_provenance
        p = self.h / 'stage4_event_ssl/models/predictive_state/best_model.pt'
        with self.changed(p, lambda f: f.write_bytes(b'wrong checkpoint')):
            with self.assertRaises(AuditError):
                neural_provenance(self.ctx, 42, neural_paths(self.h)[42])

    def test_34_neural_wrong_seed_blocks(self):
        from ft_pipeline import neural_paths, neural_provenance
        p = self.h / 'stage4_event_ssl/models/predictive_state/training_manifest.json'

        def mutate(f):
            d = load_json(f)
            d['config']['seed'] = 777
            save_json(d, f)
        with self.changed(p, mutate):
            with self.assertRaises(AuditError):
                neural_provenance(self.ctx, 42, neural_paths(self.h)[42])

    def test_35_neural_mismatched_target_blocks(self):
        from ft_pipeline import neural_paths, read_neural_join
        p = neural_paths(self.h)[42]

        def mutate(f):
            d = read_table(f)
            d.loc[0, 'target_M_next'] += 0.1
            d.to_csv(f, index=False)
        with self.changed(p, mutate):
            with self.assertRaises(AuditError):
                read_neural_join(self.ctx, self.ds, p)

    def test_36_neural_duplicate_keys_across_blocks_blocks(self):
        from ft_pipeline import neural_paths, read_neural_join
        p = neural_paths(self.h)[42]

        def mutate(f):
            d = read_table(f)
            pd.concat([d, d.iloc[:1]], ignore_index=True).to_csv(f, index=False)
        with self.changed(p, mutate):
            with self.assertRaises(AuditError):
                read_neural_join(self.ctx, self.ds, p)

    def test_37_raw_wrong_split_user_blocks(self):
        from ft_data import prepare_dataset
        p = self.h / 'stage1/splits/B_confirm_users.csv'

        def mutate(f):
            d = read_table(f)
            d.iloc[1:].to_csv(f, index=False)
        dst = Path(self.temp.name) / 'bad_split'
        dst.mkdir()
        with self.changed(p, mutate):
            with self.assertRaises(AuditError):
                prepare_dataset(self.ctx.p1, self.h, 'B_confirm', 20.0, dst, 71)
if __name__ == '__main__':
    unittest.main()
