import contextlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import summarize_generative_construction_null as s


def write_summary_fixture(work,p,split,mode,estimable=False):

    out=work/'summary'/split/mode;out.mkdir(parents=True,exist_ok=True)
    coordinates={};original=np.ones((40,40),bool);common=original.copy() if estimable else np.zeros_like(original)
    core=np.zeros_like(original);core[19:21,19:21]=True
    coverage=s.scope_coverage(np.ones((40,40)),np.ones((40,40)),original,original,common,common,core,.35)
    for coord in ('psi','phi'):
        tests=[]
        for metric in s.METRICS:
            row=dict(metric=metric,direction='less' if metric=='flow_core_to_shell_speed_ratio' else 'greater',
                     observed=.1 if estimable else None,null_mean=.1 if estimable else None,
                     null_q025=.05 if estimable else None,null_q975=.15 if estimable else None,
                     observed_minus_null_mean=0. if estimable else None,observed_original_domain=.1,
                     p_mc=1. if estimable else None,status='estimable' if estimable else 'not_estimable_no_replica_deleted',
                     interpretation=s.analysis_role(mode,coord,metric),scope='common-supported domain only; not the full original observed domain',
                     field_retained_weight_fraction=1. if estimable else 0.)
            if metric!='full_field_distance':row['q_bh_three_basin_metrics']=1. if estimable else None
            tests.append(row)
        coordinates[coord]=dict(coordinate=coord,common_cells=int(common.sum()),retained_observed_occupancy=1. if estimable else 0.,
                                normalized_departure=0. if estimable else None,scope_coverage=coverage,tests=tests)
        z=np.zeros((40,40));s.atomic_npz(out/f'{coord}_field_comparison.npz',observed_u=z,observed_v=z,null_mean_u=z,null_mean_v=z,
                                       excess_u=z,excess_v=z,common_support=common,interior_common=np.zeros_like(common),observed_occupancy=np.ones_like(z),simulation_support_frequency=common.astype(float))
        pd.DataFrame(tests).to_csv(out/f'{coord}_tests.csv',index=False)
        pd.DataFrame(coverage).to_csv(out/f'{coord}_scope_coverage.csv',index=False)
        for suffix in ('realization_statistics.csv','field_grid.csv','support_sensitivity.csv'):
            (out/f'{coord}_{suffix}').write_text('fixture\n',encoding='utf-8')
    (out/'numeric_report.md').write_text('Synthetic integrity fixture, not a scientific result.\n',encoding='utf-8')
    (out/'response_calibration.csv').write_text('fixture\n',encoding='utf-8')
    summary=dict(schema_version=1,artifacts_manifest='summary_artifacts.json',protocol_id=p['protocol_id'],split=split,mode=mode,
                 replicates=999,seed=p['seed'],technical_complete=True,coordinates=coordinates)
    s.atomic_json(out/'summary.json',summary)
    s.atomic_json(out/'summary_artifacts.json',s.artifact_manifest(out,p,split,mode))
    return out


class SummaryExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.work=Path(self.temp.name)
        self.p=dict(protocol_id='synthetic-integrity-fixture',smoke=False,replicates=999,seed=42,modes=['null_a','null_b'],coordinates=['psi','phi'],output_root=str(self.work/'historical'))
    def tearDown(self):self.temp.cleanup()
    def fixture(self,estimable=False):
        for split in ('A_val','B_confirm'):
            for mode in self.p['modes']:write_summary_fixture(self.work,self.p,split,mode,estimable)
    def reseal(self,out,split='A_val',mode='null_a'):
        s.atomic_json(out/'summary_artifacts.json',s.artifact_manifest(out,self.p,split,mode))
    def assembly_mocks(self):
        stack=contextlib.ExitStack()
        stack.enter_context(mock.patch.object(s,'load_protocol',return_value=self.p))
        stack.enter_context(mock.patch.object(s,'lock',side_effect=lambda *a:contextlib.nullcontext()))
        stack.enter_context(mock.patch.object(s,'read_realizations',return_value=([],[])))
        return stack

    def test_statistical_regression_values_unchanged(self):
        u=np.array([0.,1.,2.])[:,None,None]*np.ones((3,2,2));v=np.zeros_like(u);occ=np.ones_like(u)
        np.testing.assert_array_equal(s.symmetric_distances(u,v,occ,np.ones((2,2),bool)),[1.5,0.,1.5])
        self.assertEqual(s.conservative_rank_p([1.5,0.,1.5]),2/3)
        np.testing.assert_allclose(s.bh_three([.01,.04,None])[:2],[.03,.06])
        self.assertIsNone(s.bh_three([.01,.04,None])[2])

    def test_metric_specific_coverage_does_not_use_global_mass(self):
        original=np.ones((40,40),bool);common=original.copy();common[19,19]=False
        core=np.zeros_like(original);core[19:21,19:21]=True
        rows=s.scope_coverage(np.ones((40,40)),np.full((40,40),9.),original,original,common,common,core,.35)
        r={x['domain']:x for x in rows}
        self.assertEqual(r['field']['observed_domain_cells'],1600)
        self.assertEqual(r['field']['common_domain_cells'],1599)
        self.assertEqual(r['interior_stencil']['observed_domain_cells'],1444)
        self.assertEqual(r['interior_stencil']['common_domain_cells'],1439)
        self.assertEqual(r['core']['retained_weight_fraction'],.75)
        self.assertEqual(r['core']['weight_kind'],'sqrt(observed drift weight)')

    def test_empty_supported_domain_is_valid_completed_evidence(self):
        out=write_summary_fixture(self.work,self.p,'A_val','null_a')
        summary,_=s.validate_summary_evidence(self.work,self.p,'A_val','null_a',False)
        self.assertEqual(summary['coordinates']['psi']['common_cells'],0)
        self.assertTrue(all(t['p_mc'] is None for t in summary['coordinates']['psi']['tests']))
        self.assertTrue((out/'summary_artifacts.json').exists())

    def test_changed_comparison_hash_rejected(self):
        out=write_summary_fixture(self.work,self.p,'A_val','null_a')
        with (out/'psi_field_comparison.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(RuntimeError,'artifact missing/changed'):
            s.validate_summary_evidence(self.work,self.p,'A_val','null_a',False)

    def test_resealed_wrong_identity_or_incomplete_summary_rejected(self):
        for key,value in [('split','B_confirm'),('mode','null_b'),('replicates',998),('technical_complete',False),('schema_version',999)]:
            with self.subTest(key=key):
                out=write_summary_fixture(self.work,self.p,'A_val','null_a')
                data=s.read_json(out/'summary.json');data[key]=value;s.atomic_json(out/'summary.json',data);self.reseal(out)
                with self.assertRaises(RuntimeError):s.validate_summary_evidence(self.work,self.p,'A_val','null_a',False)

    def test_resealed_wrong_comparison_schema_rejected(self):
        out=write_summary_fixture(self.work,self.p,'A_val','null_a')
        s.atomic_npz(out/'psi_field_comparison.npz',common_support=np.zeros((2,2),bool));self.reseal(out)
        with self.assertRaisesRegex(RuntimeError,'Comparison artifact schema'):
            s.validate_summary_evidence(self.work,self.p,'A_val','null_a',False)

    def test_missing_manifest_and_path_escape_rejected(self):
        out=write_summary_fixture(self.work,self.p,'A_val','null_a')
        manifest=s.read_json(out/'summary_artifacts.json');manifest['artifacts']['../outside']='none'
        s.atomic_json(out/'summary_artifacts.json',manifest)
        with self.assertRaises(RuntimeError):s.validate_summary_evidence(self.work,self.p,'A_val','null_a',False)
        (out/'summary_artifacts.json').unlink()
        with self.assertRaises(FileNotFoundError):s.validate_summary_evidence(self.work,self.p,'A_val','null_a',False)

    def test_release_and_final_allow_all_non_estimable(self):
        self.fixture()
        with self.assembly_mocks():
            s.release_validation(self.work)
            release=s.read_json(self.work/'A_val_release.json');self.assertEqual(len(release['manifests']),2)
            s.final_report(self.work)
        complete=s.read_json(self.work/'final/COMPLETE.json')
        self.assertTrue(complete['technical_complete']);self.assertEqual(len(complete['evidence_hashes']),16)
        table=pd.read_csv(self.work/'final/construction_null_statistics.csv')
        self.assertEqual(len(table),32);self.assertTrue(table.p_mc.isna().all())
        text=(self.work/'final/construction_null_report.md').read_text(encoding='utf-8')
        self.assertIn('not the full original observed domain',text)
        self.assertIn('Retained observed weight',text)

    def test_all_p_one_is_not_a_failure(self):
        self.fixture(estimable=True)
        with self.assembly_mocks():s.release_validation(self.work);s.final_report(self.work)
        table=pd.read_csv(self.work/'final/construction_null_statistics.csv')
        self.assertTrue((table.p_mc==1).all())

    def test_failed_final_removes_old_complete(self):
        self.fixture()
        with self.assembly_mocks():s.release_validation(self.work);s.final_report(self.work)
        out=self.work/'summary/B_confirm/null_b';data=s.read_json(out/'summary.json');data['technical_complete']=False
        s.atomic_json(out/'summary.json',data);self.reseal(out,'B_confirm','null_b')
        with self.assembly_mocks(),self.assertRaises(RuntimeError):s.final_report(self.work)
        self.assertFalse((self.work/'final/COMPLETE.json').exists())
        self.assertFalse((self.work/'final/final_artifacts.json').exists())

    def test_failed_output_write_never_retains_complete(self):
        self.fixture()
        with self.assembly_mocks():s.release_validation(self.work);s.final_report(self.work)
        with self.assembly_mocks(),mock.patch.object(pd.DataFrame,'to_csv',side_effect=OSError('injected output failure')):
            with self.assertRaises(OSError):s.final_report(self.work)
        self.assertFalse((self.work/'final/COMPLETE.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
