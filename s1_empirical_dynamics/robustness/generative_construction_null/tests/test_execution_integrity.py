from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cn_common as c
import run_generative_construction_null as run


class ExecutionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cn_integrity_')
        self.work=Path(self.temp.name)
        self.p=dict(protocol_id='integrity-fixture',smoke=True,replicates=3,seed=42,
                    coordinates=['psi','phi'],modes=['null_a','null_b'],code_hashes={},source_hashes={})
        self.fields={}
        for coord in self.p['coordinates']:
            f={k:np.zeros((40,40)) for k in ('u','v','occ','occn','user_count','n','w')}
            f['occ'][20,20]=59/60;f['occn'][20,20]=59;f['user_count'][20,20]=1
            f['n'][20,20]=58;f['w'][20,20]=58/60
            f.update(state_mask=(f['occn']>=50)&(f['user_count']>=5),drift_mask=f['n']>=30,
                     rows=np.asarray(60),users=np.asarray(1),valid_state_rows=np.asarray(59),valid_drift_rows=np.asarray(58))
            self.fields.update({coord+'_'+k:v for k,v in f.items()})
        block=self.work/'cache/A_val/chunk_0000_block_0000.pkl.gz'
        c.atomic_pickle(block,[('synthetic-cache-marker',)])
        observed=self.work/'observed/A_val.npz';c.atomic_npz(observed,**self.fields)
        identity=self.work/'observed/A_val_identity.json'
        c.atomic_json(identity,dict(protocol_id=self.p['protocol_id'],split='A_val',smoke=True,passed=True,
                      source_replay=[dict(chunk=0,rows=60,checked_columns=len(c.IDENTITY_COLS),max_abs_difference=0.,source_sha256='synthetic-source',passed=True)],
                      publication_field={'scope':'smoke subset'}))
        self.done=dict(protocol_id=self.p['protocol_id'],split='A_val',smoke=True,rows=60,users=1,
                       blocks=[dict(path=str(block.relative_to(self.work)),sha256=c.sha256(block),rows=60,users=1)],
                       observed_file=str(observed.relative_to(self.work)),observed_sha256=c.sha256(observed),identity_sha256=c.sha256(identity))
        c.atomic_json(self.work/'cache/A_val/complete.json',self.done)
        calibration=self.work/'null_b_calibration.npz'
        c.atomic_npz(calibration,p=np.array([.5]),alpha=np.array([0.]),effects=np.zeros(256))
        self.cal=dict(protocol_id=self.p['protocol_id'],fit_split='A_train',smoke=True,sha256=c.sha256(calibration),
                      effect_points=256,n_train_users=32,max_item_marginal_error=0.)
        c.atomic_json(self.work/'calibration_complete.json',self.cal)
        for mode in self.p['modes']:
            for rid in range(self.p['replicates']):
                path=self.work/'simulations/A_val'/mode/f'replicate_{rid:04d}.npz'
                c.atomic_npz(path,**self.fields)
                c.atomic_json(path.with_suffix('.json'),dict(protocol_id=self.p['protocol_id'],split='A_val',mode=mode,
                              replicate=rid,seed=42,smoke=True,rows=60,users=1,sha256=c.sha256(path),
                              cache_manifest_sha256=c.sha256(self.work/'cache/A_val/complete.json'),
                              null_b_calibration_sha256=self.cal['sha256'] if mode=='null_b' else None))

    def tearDown(self):self.temp.cleanup()

    def edit_json(self,relative,update):
        path=self.work/relative;v=c.read_json(path);v.update(update);c.atomic_json(path,v)

    def test_valid_zero_fields_and_unsupported_state_are_not_rejected(self):
        for mode in self.p['modes']:
            arrays,meta=c.validate_realizations(self.work,self.p,'A_val',mode)
            self.assertEqual(len(arrays),4);self.assertEqual(len(meta),3)
            for a in arrays:np.testing.assert_array_equal(a['psi_u'],self.fields['psi_u'])
        with mock.patch.object(run,'load_protocol',return_value=self.p):run.smoke_finish(self.work)
        marker=c.read_json(self.work/'SMOKE_OK.json')
        self.assertTrue(marker['passed']);self.assertEqual(len(marker['evidence']),17)

    def test_missing_simulation_npz_refuses_smoke_and_clears_stale_marker(self):
        c.atomic_json(self.work/'SMOKE_OK.json',{'passed':True})
        (self.work/'simulations/A_val/null_a/replicate_0000.npz').unlink()
        with mock.patch.object(run,'load_protocol',return_value=self.p):
            with self.assertRaises(RuntimeError):run.smoke_finish(self.work)
        self.assertFalse((self.work/'SMOKE_OK.json').exists())

    def test_corrupt_simulation_bytes_are_rejected(self):
        with (self.work/'simulations/A_val/null_a/replicate_0000.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaises(RuntimeError):c.validate_realizations(self.work,self.p,'A_val','null_a')

    def test_wrong_replicate_metadata_is_rejected(self):
        self.edit_json('simulations/A_val/null_a/replicate_0000.json',{'split':'B_confirm'})
        with self.assertRaises(RuntimeError):c.validate_realizations(self.work,self.p,'A_val','null_a')

    def test_extra_replicate_json_is_rejected(self):
        c.atomic_json(self.work/'simulations/A_val/null_a/replicate_9999.json',{'protocol_id':self.p['protocol_id']})
        with self.assertRaises(RuntimeError):c.validate_realizations(self.work,self.p,'A_val','null_a')

    def test_corrupt_cache_is_rejected(self):
        with (self.work/self.done['blocks'][0]['path']).open('ab') as f:f.write(b'changed')
        with self.assertRaises(RuntimeError):c.validate_realizations(self.work,self.p,'A_val','null_a')

    def test_changed_identity_is_rejected(self):
        self.edit_json('observed/A_val_identity.json',{'passed':False})
        with self.assertRaises(RuntimeError):c.validate_prepared_split(self.work,self.p,'A_val')

    def test_explicit_failed_identity_with_updated_hash_is_rejected(self):
        self.edit_json('observed/A_val_identity.json',{'passed':False})
        self.edit_json('cache/A_val/complete.json',{'identity_sha256':c.sha256(self.work/'observed/A_val_identity.json')})
        with self.assertRaises(RuntimeError):c.validate_prepared_split(self.work,self.p,'A_val')

    def test_observed_hash_and_cache_totals_are_checked(self):
        self.edit_json('cache/A_val/complete.json',{'rows':61})
        with self.assertRaises(RuntimeError):c.validate_prepared_split(self.work,self.p,'A_val')

    def test_null_b_calibration_and_replicate_lineage_are_checked(self):
        self.edit_json('simulations/A_val/null_b/replicate_0000.json',{'null_b_calibration_sha256':'wrong'})
        with self.assertRaises(RuntimeError):c.validate_realizations(self.work,self.p,'A_val','null_b')
        self.edit_json('calibration_complete.json',{'fit_split':'A_val'})
        with self.assertRaises(RuntimeError):c.validate_calibration(self.work,self.p)

    def test_changed_cache_manifest_cannot_relabel_completed_realization(self):
        self.edit_json('cache/A_val/complete.json',{'diagnostic_note':'different completion record'})
        with self.assertRaises(RuntimeError):c.validate_realizations(self.work,self.p,'A_val','null_a')

    def test_confirmation_requires_complete_release_references(self):
        c.atomic_json(self.work/'A_val_release.json',dict(protocol_id=self.p['protocol_id'],schema_version=1,summaries={},manifests={}))
        with self.assertRaises(RuntimeError):c.require_confirm(self.work,self.p)

    def test_float64_response_copy_preserves_rng_and_missingness(self):
        template=pd.DataFrame({'question_id':['q1','q2','q3'],'is_correct':[1.,np.nan,0.]})
        prob=np.array([.2,.5,.8]);before=template.copy(deep=True)
        expected=c.stable_rng(42,'A_val','null_a',1,7,0).binomial(1,prob[[0,2]])
        simulated,_=c.simulate_questions(template,prob,42,'A_val','null_a',1,7)
        np.testing.assert_array_equal(simulated.is_correct.to_numpy()[[0,2]],expected)
        self.assertTrue(pd.isna(simulated.is_correct.iloc[1]));pd.testing.assert_frame_equal(template,before)

    def archived_table(self):
        f=c.unpack_fields(self.fields,'psi')
        return pd.DataFrame(dict(x_bin=np.repeat(np.arange(40),40),y_bin=np.tile(np.arange(40),40),
                    drift_M=f['u'].ravel(),drift_Psi=f['v'].ravel(),drift_supported=f['drift_mask'].ravel(),
                    drift_count=f['n'].ravel(),state_supported=f['state_mask'].ravel(),occupancy_count=f['occn'].ravel(),
                    user_count=f['user_count'].ravel(),drift_user_balanced_weight=f['w'].ravel(),occupancy_weighted=f['occ'].ravel(),
                    occupancy_probability=(f['occ']/f['occ'].sum()).ravel()))

    def test_archived_all_sufficient_quantities_match(self):
        path=self.work/'grid.csv';self.archived_table().to_csv(path,index=False)
        result=c.audit_field(c.unpack_fields(self.fields,'psi'),path)
        self.assertTrue(result['passed']);self.assertIn('user_count',result['checked_exact'])

    def test_archived_support_counts_and_raw_weights_are_checked(self):
        for column in ('state_supported','occupancy_count','user_count','drift_user_balanced_weight','occupancy_weighted'):
            table=self.archived_table()
            if column=='state_supported':table.loc[0,column]=True
            else:table.loc[0,column]+=1
            path=self.work/'grid.csv';table.to_csv(path,index=False)
            with self.subTest(column=column):
                with self.assertRaises(RuntimeError):c.audit_field(c.unpack_fields(self.fields,'psi'),path)


if __name__=='__main__':unittest.main(verbosity=2)
