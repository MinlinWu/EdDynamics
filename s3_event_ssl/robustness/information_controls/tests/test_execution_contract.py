import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import finalize_six
import slurm_s3
import ssl_common


class ExecutionContractTests(unittest.TestCase):
    def test_stage_set_covers_six_paired_seeds_without_duplicates(self):
        cpu,gpu=finalize_six.expected_stages()
        self.assertEqual((len(cpu),len(gpu),len(set(cpu+gpu))),(52,40,92))
        self.assertIn('S3/seed_42/inference',cpu)
        self.assertFalse(any('/inference' in name for name in gpu))
        for seed in finalize_six.GPU_SEEDS:
            self.assertEqual(sum(name.startswith(f'S3/seed_{seed}/') for name in gpu),8)

    def test_paired_differences_preserve_negative_values_and_missing_metrics(self):
        frame=pd.DataFrame([
            dict(block='S3',seed=2026,split='B_confirm',model='aug',training_backend='GPU',current_MSE=3.,velocity_D2=np.nan),
            dict(block='S3',seed=2026,split='B_confirm',model='clean',training_backend='GPU',current_MSE=1.,velocity_D2=2.)])
        result=finalize_six.paired_differences(pd,frame).set_index('metric')
        self.assertEqual(result.loc['current_MSE','clean_minus_aug'],-2.)
        self.assertTrue(pd.isna(result.loc['velocity_D2','clean_minus_aug']))
        self.assertTrue(result.primary_metric.all())

    def test_incomplete_pair_is_rejected(self):
        frame=pd.DataFrame([dict(block='S3',seed=666,split='B_confirm',model='aug',training_backend='GPU',current_MSE=1.)])
        with self.assertRaisesRegex(ValueError,'Incomplete S3 pair'):
            finalize_six.paired_differences(pd,frame)

    def test_original_dependencies_are_imported_from_main_and_hash_bound(self):
        contract=json.loads((ssl_common.ROOT/'source_contract.json').read_text())
        for name,digest in contract['files'].items():
            path=ssl_common.source_path(name)
            self.assertEqual(path.parent.name,'main')
            self.assertEqual(ssl_common.sha(path),digest)

    def test_gpu_identifiers_reject_duplicates_and_ambiguous_tokens(self):
        self.assertEqual(slurm_s3.parse_gpu_ids('0-2,4,6'),['0','1','2','4','6'])
        for value in ('0,0','0-2,2','five','0-300'):
            with self.assertRaises(RuntimeError):slurm_s3.parse_gpu_ids(value)


if __name__=='__main__':unittest.main(verbosity=2)
