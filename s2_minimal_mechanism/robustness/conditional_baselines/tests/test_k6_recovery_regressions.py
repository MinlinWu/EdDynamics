from __future__ import annotations
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft_common import AuditError, load_json, recover_historical_k6, save_json, sha
from ft_data import load_partition

class K6RecoveryRegressionTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fourterm_k6_recovery_test_')
        self.addCleanup(self.temp.cleanup)
        self.history = Path(self.temp.name)
        folder = self.history / 'stage1/dynamics/fixed_k6_mesostates'
        folder.mkdir(parents=True)
        self.metadata_path = folder / 'fixed_k6_model_metadata.json'
        self.metadata = {'scaler_mean': [0.1, -0.2], 'scaler_scale': [1.3, 0.4]}
        save_json(self.metadata, self.metadata_path)
        raw = np.array([[-0.8, -0.7], [-0.6, 0.4], [-0.2, -0.1], [0.1, -0.6], [0.5, 0.2], [0.9, 0.8]])
        standardized = (raw - self.metadata['scaler_mean']) / self.metadata['scaler_scale']
        self.frame = pd.DataFrame({'macrostate': np.arange(6), 'center_M': raw[:, 0], 'center_Psi': raw[:, 1], 'center_M_standardized': standardized[:, 0], 'center_Psi_standardized': standardized[:, 1]})
        current_frame = self.frame.copy()
        current_frame.loc[0, 'center_M_standardized'] += 2.5e-13
        current_path = folder / 'fixed_k6_centers.csv'
        current_frame.to_csv(current_path, index=False)
        self.old = {'metadata_sha256': sha(self.metadata_path), 'centers_sha256': 'f' * 64}
        self.current = {'metadata_sha256': sha(self.metadata_path), 'centers_sha256': sha(current_path), 'metadata_path': str(self.metadata_path), 'centers_path': str(current_path)}
        self.certificate = {'metadata_sha256': self.old['metadata_sha256'], 'legacy_centers_sha256': self.old['centers_sha256'], 'current_centers_sha256': self.current['centers_sha256'], 'references': []}
        content = self.frame.to_csv(index=False).encode('utf-8')
        for seed in [42, 2026, 666, 606, 37, 4669]:
            root = self.history / f'export_seed_{seed}'
            root.mkdir()
            centers = root / 'centers.csv'
            centers.write_bytes(content)
            manifest = root / 'evaluation_manifest.json'
            save_json({'macro_partition': {'verified': True, 'fit_split': 'A_train', 'metadata_sha256': self.old['metadata_sha256'], 'centers_sha256': self.old['centers_sha256']}}, manifest)
            self.certificate['references'].append({'seed': seed, 'manifest': str(manifest.relative_to(self.history)), 'manifest_sha256': sha(manifest), 'centers': str(centers.relative_to(self.history))})
        self.certificate['export_sha256'] = sha(self.history / self.certificate['references'][0]['centers'])

    def recover(self):
        return recover_historical_k6(self.history, self.current, self.old, self.certificate)

    def hashes(self):
        return {str(f.relative_to(self.history)): sha(f) for f in self.history.rglob('*') if f.is_file()}

    def rewrite_exports_and_certificate(self, frame):
        data = frame.to_csv(index=False).encode('utf-8')
        for reference in self.certificate['references']:
            (self.history / reference['centers']).write_bytes(data)
        self.certificate['export_sha256'] = sha(self.history / self.certificate['references'][0]['centers'])

    def test_01_certified_recovery_preserves_history_and_distinct_hashes(self):
        before = self.hashes()
        proof = self.recover()
        self.assertEqual(proof['status'], 'verified_historical_export_recovery')
        self.assertFalse(proof['historical_files_modified'])
        self.assertFalse(proof['centers_refitted'])
        self.assertEqual(proof['legacy_centers_sha256'], self.old['centers_sha256'])
        self.assertEqual(proof['current_centers_sha256'], self.current['centers_sha256'])
        self.assertEqual(proof['export_sha256'], self.certificate['export_sha256'])
        self.assertEqual(len(proof['evidence_paths']), 12)
        self.assertGreater(proof['max_current_vs_historical_abs_difference'], 0)
        self.assertLessEqual(proof['max_current_vs_historical_abs_difference'], 1e-12)
        self.assertEqual(before, self.hashes())

    def test_02_partition_uses_verified_historical_csv_with_round_trip(self):
        proof = self.recover()
        part = load_partition(self.history, proof['canonical_centers_path'])
        np.testing.assert_array_equal(part['raw_centers'], self.frame[['center_M', 'center_Psi']].to_numpy())
        np.testing.assert_array_equal(part['centers'], self.frame[['center_M_standardized', 'center_Psi_standardized']].to_numpy())
        self.assertEqual(Path(proof['canonical_centers_path']), self.history / self.certificate['references'][0]['centers'])

    def test_03_unregistered_current_hash_rejected(self):
        self.current['centers_sha256'] = 'a' * 64
        with self.assertRaisesRegex(AuditError, 'Unregistered K6'):
            self.recover()

    def test_04_unregistered_legacy_hash_rejected(self):
        self.old['centers_sha256'] = 'b' * 64
        with self.assertRaisesRegex(AuditError, 'Unregistered K6'):
            self.recover()

    def test_05_metadata_hash_mismatch_rejected(self):
        for where in ('old', 'current', 'certificate'):
            with self.subTest(where=where):
                obj = getattr(self, where)
                saved = obj['metadata_sha256']
                obj['metadata_sha256'] = 'c' * 64
                try:
                    with self.assertRaisesRegex(AuditError, 'metadata mismatch'):
                        self.recover()
                finally:
                    obj['metadata_sha256'] = saved

    def test_06_missing_certificate_rejected(self):
        with self.assertRaisesRegex(AuditError, 'no registered recovery certificate'):
            recover_historical_k6(self.history, self.current, self.old, None)

    def test_07_changed_export_bytes_rejected(self):
        path = self.history / self.certificate['references'][3]['centers']
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(AuditError, 'historical export changed'):
            self.recover()

    def test_08_changed_manifest_bytes_rejected(self):
        path = self.history / self.certificate['references'][2]['manifest']
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(AuditError, 'historical manifest changed'):
            self.recover()

    def test_09_incomplete_or_misordered_seed_evidence_rejected(self):
        original = copy.deepcopy(self.certificate['references'])
        for refs in (original[:-1], original[::-1], [original[0]] * 6):
            with self.subTest(seeds=[r['seed'] for r in refs]):
                self.certificate['references'] = refs
                with self.assertRaisesRegex(AuditError, 'Incomplete K6 recovery evidence'):
                    self.recover()
        self.certificate['references'] = original

    def test_10_missing_seed_file_rejected(self):
        (self.history / self.certificate['references'][-1]['centers']).unlink()
        with self.assertRaisesRegex(AuditError, 'historical export changed'):
            self.recover()

    def test_11_hash_consistent_wrong_manifest_contract_rejected(self):
        reference = self.certificate['references'][1]
        path = self.history / reference['manifest']
        data = load_json(path)
        data['macro_partition']['centers_sha256'] = 'd' * 64
        save_json(data, path)
        reference['manifest_sha256'] = sha(path)
        with self.assertRaisesRegex(AuditError, 'does not declare the frozen K6'):
            self.recover()

    def test_12_hash_consistent_invalid_labels_rejected(self):
        bad = self.frame.copy()
        bad.loc[5, 'macrostate'] = 4
        self.rewrite_exports_and_certificate(bad)
        with self.assertRaisesRegex(AuditError, 'Invalid historical K6 labels'):
            self.recover()

    def test_13_hash_consistent_wrong_scaler_rejected(self):
        self.metadata['scaler_scale'][0] *= 2
        save_json(self.metadata, self.metadata_path)
        digest = sha(self.metadata_path)
        for obj in (self.old, self.current, self.certificate):
            obj['metadata_sha256'] = digest
        for reference in self.certificate['references']:
            path = self.history / reference['manifest']
            data = load_json(path)
            data['macro_partition']['metadata_sha256'] = digest
            save_json(data, path)
            reference['manifest_sha256'] = sha(path)
        with self.assertRaisesRegex(AuditError, 'Historical K6 scaler consistency'):
            self.recover()

    def test_14_even_certified_large_numeric_drift_rejected(self):
        path = Path(self.current['centers_path'])
        data = pd.read_csv(path, float_precision='round_trip')
        data.loc[0, 'center_M_standardized'] += 1e-06
        data.to_csv(path, index=False)
        self.current['centers_sha256'] = self.certificate['current_centers_sha256'] = sha(path)
        with self.assertRaisesRegex(AuditError, 'Registered K6 recovery numeric bound'):
            self.recover()
if __name__ == '__main__':
    unittest.main(verbosity=2)
