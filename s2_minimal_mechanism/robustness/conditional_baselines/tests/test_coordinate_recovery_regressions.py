from __future__ import annotations
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import ft_common as common
from ft_common import AuditError, AUX, FEATURES, load_json, objhash, save_json, sha
from ft_data import grid_cells, prepare_dataset, recover_b_confirm_coordinates

class CoordinateRecoveryRegressionTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fourterm_coordinate_recovery_')
        self.addCleanup(self.temp.cleanup)
        self.h = Path(self.temp.name) / 'history'
        self.h.mkdir()
        self.dst = Path(self.temp.name) / 'prepared'
        self.dst.mkdir()
        self.panel = self.h / 'stage1/dynamics/student_dynamics_panel_core_B_confirm.csv'
        self.panel.parent.mkdir(parents=True)
        self.panel.write_text('fixture source bytes\n', encoding='utf-8')
        self.archive = self.h / 'stage2_phase3_confirm/tables/phase3_B_confirm_full_predictions.csv'
        self.archive.parent.mkdir(parents=True)
        self.K = np.array([[10, 1], [10, 2], [11, 1], [11, 2], [12, 1], [12, 2]], np.int64)
        self.F = np.ones((6, len(FEATURES)))
        self.F[:, 0] = [-0.55, -0.55, -0.25, -0.25, 0.2, 0.2]
        self.F[:, 1] = [-0.1, -1.8e-17, -0.1, -1.9e-17, 0.123, 0.124]
        self.Y = self.F[:, :2].copy()
        self.Y[:, 1] = [-1.8e-17, 0.12, -1.9e-17, 0.11, 0.121, 0.122]
        self.certificate = {'version': 'b-confirm-coordinate-recovery-v1', 'split': 'B_confirm', 'expected_full_rows': 6, 'maximum_coordinate_absolute_difference': 1e-12, 'expected_full_grid_changes': {'features': 2, 'target': 2}, 'corrections': []}
        oldF = self.F[:, :2].copy()
        oldY = self.Y.copy()
        for name, index in [('target', 0), ('features', 1), ('target', 2), ('features', 3)]:
            array = self.F if name == 'features' else self.Y
            self.certificate['corrections'].append({'key': self.K[index].tolist(), 'array': name, 'column': 1, 'native_hex': float(array[index, 1]).hex(), 'historical_hex': float(0.0).hex()})
            (oldF if name == 'features' else oldY)[index, 1] = 0.0
        self.old = pd.DataFrame({'user_id': self.K[:, 0], 'bundle_step_index': self.K[:, 1], 'M': oldF[:, 0], 'Psi': oldF[:, 1], 'target_M_next': oldY[:, 0], 'target_Psi_next': oldY[:, 1]})
        self.old.to_csv(self.archive, index=False)
        self.certificate['panel'] = self.binding(self.panel)
        self.certificate['archive'] = self.binding(self.archive)
        self.certificate['bindings'] = []
        for index in range(7):
            path = self.h / f'frozen_binding_{index}.json'
            save_json({'binding': index}, path)
            self.certificate['bindings'].append(self.binding(path))
        self.indices = np.arange(6)
        self.write_cache()

    def binding(self, path):
        return {'relative_path': path.relative_to(self.h).as_posix(), 'sha256': sha(path)}

    def write_cache(self, indices=None):
        self.indices = np.arange(6) if indices is None else np.asarray(indices)
        for name, array in [('keys', self.K), ('features', self.F), ('target', self.Y)]:
            np.save(self.dst / (name + '.npy'), array[self.indices])

    def recover(self, smoke=None):
        return recover_b_confirm_coordinates(self.h, self.panel, self.dst, self.certificate, 2, smoke)

    def history_hashes(self):
        return {str(p.relative_to(self.h)): sha(p) for p in self.h.rglob('*') if p.is_file()}

    def rewrite_archive(self, frame):
        frame.to_csv(self.archive, index=False)
        self.certificate['archive'] = self.binding(self.archive)

    def test_01_exact_four_values_and_history_unchanged(self):
        before = self.history_hashes()
        proof = self.recover()
        gotF = np.load(self.dst / 'features.npy')
        gotY = np.load(self.dst / 'target.npy')
        expectedF = self.F.copy()
        expectedY = self.Y.copy()
        expectedF[[1, 3], 1] = 0.0
        expectedY[[0, 2], 1] = 0.0
        np.testing.assert_array_equal(gotF, expectedF)
        np.testing.assert_array_equal(gotY, expectedY)
        self.assertEqual(np.count_nonzero(gotF != self.F) + np.count_nonzero(gotY != self.Y), 4)
        self.assertEqual(proof['applied_value_count'], 4)
        self.assertEqual(proof['historical_certificate_values_verified'], 4)
        self.assertEqual(proof['grid_changes'], {'features': 2, 'target': 2})
        self.assertFalse(proof['global_snapping'])
        self.assertFalse(proof['models_refitted'])
        self.assertEqual(before, self.history_hashes())

    def test_02_smoke_intersection_only_and_full_historical_evidence(self):
        self.write_cache([0, 1])
        proof = self.recover([10])
        self.assertEqual(proof['applied_value_count'], 2)
        self.assertEqual(proof['archive_rows'], 6)
        self.assertEqual(proof['historical_certificate_values_verified'], 4)
        np.testing.assert_array_equal(np.load(self.dst / 'features.npy')[:, 1], [-0.1, 0.0])

    def test_03_smoke_with_no_affected_users(self):
        self.write_cache([4, 5])
        before = [sha(self.dst / (n + '.npy')) for n in ['features', 'target']]
        proof = self.recover([12])
        self.assertEqual(proof['applied_value_count'], 0)
        self.assertEqual(proof['historical_certificate_values_verified'], 4)
        self.assertEqual(before, [sha(self.dst / (n + '.npy')) for n in ['features', 'target']])

    def test_04_unknown_source_or_archive_or_binding_hash_rejected(self):
        for part in ['panel', 'archive'] + list(range(7)):
            with self.subTest(part=part):
                record = self.certificate[part] if isinstance(part, str) else self.certificate['bindings'][part]
                saved = record['sha256']
                record['sha256'] = 'a' * 64
                try:
                    with self.assertRaisesRegex(AuditError, 'source changed'):
                        self.recover()
                finally:
                    record['sha256'] = saved

    def test_05_changed_source_bytes_rejected(self):
        self.panel.write_text('changed\n', encoding='utf-8')
        with self.assertRaisesRegex(AuditError, 'source changed'):
            self.recover()

    def test_06_wrong_native_value_rejected_without_partial_write(self):
        bad = self.F.copy()
        bad[3, 1] = -1.91e-17
        np.save(self.dst / 'features.npy', bad)
        before = sha(self.dst / 'features.npy')
        with self.assertRaisesRegex(AuditError, 'native coordinate value changed'):
            self.recover()
        self.assertEqual(before, sha(self.dst / 'features.npy'))
        np.testing.assert_array_equal(np.load(self.dst / 'target.npy'), self.Y)

    def test_07_wrong_historical_value_outside_smoke_rejected(self):
        self.write_cache([4, 5])
        old = self.old.copy()
        old.loc[3, 'Psi'] = 1e-17
        self.rewrite_archive(old)
        with self.assertRaisesRegex(AuditError, 'historical coordinate value changed'):
            self.recover([12])

    def test_08_unknown_grid_change_rejected_even_within_tolerance(self):
        bad = self.F.copy()
        bad[4, 1] = np.nextafter(0.0, -1.0)
        np.save(self.dst / 'features.npy', bad)
        old = self.old.copy()
        old.loc[4, 'Psi'] = 0.0
        self.rewrite_archive(old)
        with self.assertRaisesRegex(AuditError, 'Unlisted or missing coordinate grid change'):
            self.recover()

    def test_09_unlisted_within_cell_roundoff_preserved(self):
        bad = self.F.copy()
        bad[4, 1] += 5e-16
        np.save(self.dst / 'features.npy', bad)
        self.recover()
        self.assertEqual(float(np.load(self.dst / 'features.npy')[4, 1]).hex(), float(bad[4, 1]).hex())

    def test_10_coordinates_outside_bound_rejected(self):
        bad = self.Y.copy()
        bad[5, 0] += 1e-09
        np.save(self.dst / 'target.npy', bad)
        with self.assertRaisesRegex(AuditError, 'exceed recovery bound'):
            self.recover()

    def test_11_duplicate_archive_key_rejected(self):
        old = self.old.copy()
        old.loc[5, ['user_id', 'bundle_step_index']] = old.loc[4, ['user_id', 'bundle_step_index']].to_numpy()
        self.rewrite_archive(old)
        with self.assertRaisesRegex(AuditError, 'keys are not unique and sorted'):
            self.recover()

    def test_12_missing_archive_key_rejected(self):
        self.rewrite_archive(self.old.iloc[:-1])
        with self.assertRaisesRegex(AuditError, 'join is incomplete'):
            self.recover()

    def test_13_missing_prepared_key_full_and_smoke_rejected(self):
        self.write_cache([0, 1, 2, 3, 4])
        with self.assertRaisesRegex(AuditError, 'row count differs'):
            self.recover()
        self.write_cache([0])
        with self.assertRaisesRegex(AuditError, 'key missing for selected smoke user'):
            self.recover([10])

    def test_14_bad_certificate_key_column_count_and_scope_rejected(self):
        original = copy.deepcopy(self.certificate)
        cases = [lambda c: c['corrections'][0].update(key=[999, 1]), lambda c: c['corrections'][0].update(column=0), lambda c: c['corrections'].pop(), lambda c: c.update(split='A_train'), lambda c: c.update(maximum_coordinate_absolute_difference=1e-08), lambda c: c['bindings'].pop(), lambda c: c['corrections'].__setitem__(0, copy.deepcopy(c['corrections'][1]))]
        for mutate in cases:
            with self.subTest(case=str(mutate)):
                self.certificate = copy.deepcopy(original)
                mutate(self.certificate)
                with self.assertRaises(AuditError):
                    self.recover()
        self.certificate = original

    def test_15_recovery_rejected_for_training_or_validation(self):
        for split in ['A_train', 'A_val']:
            with self.subTest(split=split), self.assertRaisesRegex(AuditError, 'restricted to B_confirm'):
                prepare_dataset(None, self.h, split, 20.0, self.dst, 2, coordinate_recovery=self.certificate)

    def test_16_artifact_path_escape_rejected(self):
        self.certificate['bindings'][0]['relative_path'] = '../outside.json'
        with self.assertRaisesRegex(AuditError, 'escapes history'):
            self.recover()

    def test_17_duplicate_prepared_key_rejected(self):
        keys = self.K.copy()
        keys[5] = keys[4]
        np.save(self.dst / 'keys.npy', keys)
        with self.assertRaisesRegex(AuditError, 'Prepared coordinate keys'):
            self.recover()

    def stage_function(self):
        tree = ast.parse((PACKAGE / 'ft_pipeline.py').read_text(encoding='utf-8'))
        node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'stage_dataset'))
        namespace = {k: v for k, v in vars(common).items() if not k.startswith('__')}
        namespace.update(__file__=str(PACKAGE / 'ft_pipeline.py'), prepare_dataset=Mock())
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'real_stage_dataset_gate', 'exec'), namespace)
        return namespace

    def context(self):
        return SimpleNamespace(history=self.h, out=Path(self.temp.name) / 'out', run_hash='fixture-run', synthetic_fixture=False, final_cal={'eta': 20.0}, train_cal={'eta': 20.0}, p1=None, provenance={'artifacts': [{}, {}, {'sha256': 'preparer'}]}, chunk_size=2, smoke_users=None)

    def freeze(self, ctx):
        root = ctx.out / 'models'
        root.mkdir(parents=True)
        model = {'fixture_model': True}
        save_json(model, root / 'final_models.json')
        save_json({'ready_for_B_evaluation': True, 'protocol_code_hash': ctx.run_hash, 'model_payload_sha256': sha(root / 'final_models.json'), 'model_hash': objhash(model)}, root / 'FIT_FROZEN.json')
        save_json({'run_hash': ctx.run_hash, 'outputs': {n: sha(root / n) for n in ['final_models.json', 'FIT_FROZEN.json']}}, root / '_SUCCESS.json')

    def test_18_B_targets_not_loaded_without_frozen_models(self):
        ns = self.stage_function()
        ctx = self.context()
        with self.assertRaises(AuditError):
            ns['stage_dataset'](ctx, 'B_confirm')
        ns['prepare_dataset'].assert_not_called()

    def test_19_changed_frozen_models_block_before_B(self):
        ns = self.stage_function()
        ctx = self.context()
        self.freeze(ctx)
        path = ctx.out / 'models/final_models.json'
        path.write_text(path.read_text() + ' ', encoding='utf-8')
        with self.assertRaises(AuditError):
            ns['stage_dataset'](ctx, 'B_confirm')
        ns['prepare_dataset'].assert_not_called()

    def ready_stage(self, synthetic=False, certificate=True):
        ns = self.stage_function()
        ctx = self.context()
        ctx.synthetic_fixture = synthetic
        self.freeze(ctx)
        split = self.h / 'stage1/splits/B_confirm_users.csv'
        split.parent.mkdir(parents=True)
        pd.DataFrame({'user_id': [10, 11, 12]}).to_csv(split, index=False)
        original_load = ns['load_json']
        ns['load_json'] = lambda path: ({'b_confirm_coordinate_recovery': self.certificate} if certificate else {}) if Path(path).name == 'source_contract.json' else original_load(path)
        destination = ctx.out / 'prepared/B_confirm'
        ns['atomic_stage'] = Mock(side_effect=lambda root, name, rh, inputs, callback: (callback(destination), destination)[1])
        ns['Dataset'] = Mock(return_value=SimpleNamespace(meta={'source': {'sha256': sha(self.panel)}}))
        return (ns, ctx)

    def test_20_verified_fit_passes_certificate_only_to_B_callback(self):
        ns, ctx = self.ready_stage()
        ns['stage_dataset'](ctx, 'B_confirm')
        ns['prepare_dataset'].assert_called_once()
        self.assertEqual(ns['prepare_dataset'].call_args.kwargs['coordinate_recovery'], self.certificate)
        inputs = ns['atomic_stage'].call_args.args[3]
        self.assertEqual(inputs['coordinate_recovery'], self.certificate)
        self.assertEqual(inputs['frozen_fit']['fit_frozen_sha256'], sha(ctx.out / 'models/FIT_FROZEN.json'))

    def test_21_unmarked_real_flow_cannot_skip_missing_certificate(self):
        ns, ctx = self.ready_stage(certificate=False)
        with self.assertRaisesRegex(AuditError, 'requires a registered coordinate recovery certificate'):
            ns['stage_dataset'](ctx, 'B_confirm')
        ns['prepare_dataset'].assert_not_called()

    def test_22_explicit_synthetic_flow_does_not_apply_real_certificate(self):
        ns, ctx = self.ready_stage(synthetic=True)
        ns['stage_dataset'](ctx, 'B_confirm')
        self.assertIsNone(ns['prepare_dataset'].call_args.kwargs['coordinate_recovery'])
        self.assertTrue(ns['atomic_stage'].call_args.args[3]['synthetic_fixture'])

    def test_23_false_freeze_ready_flag_rejected_even_with_matching_receipt(self):
        ns, ctx = self.ready_stage()
        path = ctx.out / 'models/FIT_FROZEN.json'
        frozen = load_json(path)
        frozen['ready_for_B_evaluation'] = False
        save_json(frozen, path)
        receipt = ctx.out / 'models/_SUCCESS.json'
        r = load_json(receipt)
        r['outputs']['FIT_FROZEN.json'] = sha(path)
        save_json(r, receipt)
        with self.assertRaisesRegex(AuditError, 'requires a verified frozen fit'):
            ns['stage_dataset'](ctx, 'B_confirm')
        ns['prepare_dataset'].assert_not_called()

    def test_24_prepare_integration_recovers_before_grid_assignment(self):
        raw = pd.DataFrame(self.F, columns=FEATURES)
        raw['user_id'] = self.K[:, 0]
        raw['bundle_step_index'] = self.K[:, 1]
        raw['target_M_next'] = self.Y[:, 0]
        raw['target_Psi_next'] = self.Y[:, 1]
        for column in AUX:
            raw[column] = 0.0
        raw.to_csv(self.panel, index=False)
        parsed = pd.read_csv(self.panel)
        for r in self.certificate['corrections']:
            row = int(np.flatnonzero((self.K == r['key']).all(axis=1))[0])
            column = 'Psi' if r['array'] == 'features' else 'target_Psi_next'
            r['native_hex'] = float(parsed.loc[row, column]).hex()
        self.certificate['panel'] = self.binding(self.panel)
        split = self.h / 'stage1/splits/B_confirm_users.csv'
        split.parent.mkdir(parents=True)
        pd.DataFrame({'user_id': [10, 11, 12]}).to_csv(split, index=False)
        p1 = SimpleNamespace(PHASE1_PANEL_COLUMNS=list(raw.columns), prepare_panel=lambda frame, split, eta: frame.copy())
        destination = Path(self.temp.name) / 'integrated'
        destination.mkdir()
        prepare_dataset(p1, self.h, 'B_confirm', 20.0, destination, 2, coordinate_recovery=self.certificate)
        F = np.load(destination / 'features.npy')
        Y = np.load(destination / 'target.npy')
        self.assertEqual(F[1, 1], 0.0)
        self.assertEqual(F[3, 1], 0.0)
        self.assertEqual(Y[0, 1], 0.0)
        self.assertEqual(Y[2, 1], 0.0)
        np.testing.assert_array_equal(np.load(destination / 'cells.npy'), grid_cells(F[:, :2]))
        np.testing.assert_array_equal(np.load(destination / 'cells.npy'), grid_cells(self.old[['M', 'Psi']].to_numpy()))
        self.assertEqual(load_json(destination / 'metadata.json')['coordinate_recovery']['applied_value_count'], 4)
if __name__ == '__main__':
    unittest.main(verbosity=2)
