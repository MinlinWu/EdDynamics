from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ft_common as c

class StageReceiptRegressionTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ft_receipt_test_')
        self.root = Path(self.temp.name)
        self.run_hash = 'synthetic-run'
        self.inputs = {'fixture': 'synthetic-only'}

        def write(dst):
            (dst / 'nested').mkdir()
            (dst / 'nested' / 'result.txt').write_text('verified output', encoding='utf-8')
        self.stage = c.atomic_stage(self.root, 'final', self.run_hash, self.inputs, write)
        self.receipt = self.stage / '_SUCCESS.json'
        self.result = self.stage / 'nested' / 'result.txt'

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_receipt_returns_record(self):
        record = c.verify_stage_receipt(self.stage, self.run_hash, c.objhash(self.inputs))
        self.assertEqual(record, c.load_json(self.receipt))

    def test_missing_receipt_blocks(self):
        self.receipt.unlink()
        with self.assertRaises(c.AuditError):
            c.verify_stage_receipt(self.stage)

    def test_deleted_output_blocks(self):
        self.result.unlink()
        with self.assertRaises(c.AuditError):
            c.verify_stage_receipt(self.stage)

    def test_changed_output_blocks_even_when_receipt_unchanged(self):
        old_receipt_hash = c.sha(self.receipt)
        self.result.write_text('damaged output', encoding='utf-8')
        self.assertEqual(c.sha(self.receipt), old_receipt_hash)
        with self.assertRaises(c.AuditError):
            c.verify_stage_receipt(self.stage)

    def test_wrong_run_hash_blocks(self):
        with self.assertRaises(c.AuditError):
            c.verify_stage_receipt(self.stage, run_hash='another-run')

    def test_wrong_input_hash_blocks(self):
        with self.assertRaises(c.AuditError):
            c.verify_stage_receipt(self.stage, input_hash='another-input')

    def test_parent_path_output_blocks(self):
        outside = self.root / 'outside.txt'
        outside.write_text('outside the committed stage', encoding='utf-8')
        record = c.load_json(self.receipt)
        record['outputs'] = {'../outside.txt': c.sha(outside)}
        c.save_json(record, self.receipt)
        with self.assertRaises(c.AuditError):
            c.verify_stage_receipt(self.stage)

    def test_absolute_output_path_blocks(self):
        record = c.load_json(self.receipt)
        record['outputs'] = {str(self.result.resolve()): c.sha(self.result)}
        c.save_json(record, self.receipt)
        with self.assertRaises(c.AuditError):
            c.verify_stage_receipt(self.stage)
if __name__ == '__main__':
    unittest.main()
