
import ast
import copy
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ssl_reporting as report

TREE=ast.parse((ROOT/'ssl_statistics.py').read_text(encoding='utf-8'))
NODES={n.name:n for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
ENV={'np':np,'EPS':1e-12,'norm':SimpleNamespace(ppf=NormalDist().inv_cdf)}
exec(compile(ast.Module(body=[NODES['basic_interval'],NODES['interval_metadata']],type_ignores=[]),
             str(ROOT/'ssl_statistics.py'),'exec'),ENV)
interval=ENV['basic_interval'];metadata=ENV['interval_metadata']

NUMERIC_AST={
    'cells':'b116dd826a4eebd82e3b68e06d1601fba886d5cae968d4c22661158ec3d6d5e9',
    'corr':'87372a4d3efb5ceaad88fa002a19bd6954da6c4a40aa3e2324f5319540423616',
    'field':'46b205b4dc9c0b7b96abbf661c99d5aacd5b565b5bd0bec70932032f25da9f9d',
    'field_metrics':'e7837954d3b1951816cdeb47c14ad56d94504c163b2006c67ab2583ecb5b4f06',
    'transition':'560fb5724d17e816aef1982bbdc295588c3ed25f12e3cec51204e779ea363602',
    'evaluate_predictions':'3b565c7287385117794900e90b7e7e2b1757c03e2ba61e43732cdf360b62ef5e',
    'UserSufficient':'9339c93e98c58b1dc10da7a85d47f17a6c24676cc95ccb827e00a74264ec4dae',
    'decomposition':'057c81b873a19d7c88f45cb92b1483821f728954ef6b520e9489ec824427de72'}


def numerical_ast_dump(node):

    node=copy.deepcopy(node)
    for part in ast.walk(node):
        if isinstance(part,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and 'type_params' not in part._fields:
            part._fields=(*part._fields,'type_params')
            part.type_params=[]
    return ast.dump(node,include_attributes=False)

def row(support='fixed',level=.975,values=(14.,14.5,15.,15.5,16.),point=10.):
    return {'block':'S1','contrast':'same_readout_minus_proposal_velocity_D2',
        'model_a':'same_readout','model_b':'proposal','metric':'velocity_D2',
        'support_contract':support,'primary_level':level==.975,
        **metadata(support,level,2),**interval(values,point,level)}

def metric_frame():
    base={'block':'S1_full','seed':42,'model':'same_readout'}
    for view in ('endpoint','velocity'):
        for metric in report.FIELD_DISPLAY_METRICS:
            base[view+'_'+metric]='complete' if metric=='status' else -.25 if metric=='local_cosine' else .1
    for name in ['current_MSE','current_rmse_M','current_rmse_Psi','current_r_M','current_r_Psi',
                 'next_MSE','one_step_rmse_M','one_step_rmse_Psi','next_r_M','next_r_Psi',
                 'user_equal_next_rmse_M','user_equal_next_rmse_Psi','interval_velocity_MSE',
                 'anchor_transition_mean_row_TV','anchor_self_transition_r_descriptive',
                 'readout_transition_mean_row_TV','readout_self_transition_r_descriptive']:
        base[name]=.2
    empty={'block':'S3','seed':2026,'model':'clean','endpoint_status':'not_estimable',
           'velocity_status':'not_estimable','endpoint_D2':np.nan,'velocity_D2':np.nan,
           'endpoint_cells':0,'velocity_cells':0}
    return pd.DataFrame([base,empty])


class ReportingContractTests(unittest.TestCase):
    def test_numerical_estimators_ast_unchanged(self):
        for name,expected in NUMERIC_AST.items():
            with self.subTest(function=name):
                actual=hashlib.sha256(numerical_ast_dump(NODES[name]).encode()).hexdigest()
                self.assertEqual(actual,expected)

    def test_bootstrap_draw_loop_ast_unchanged(self):
        loop=next(n for n in NODES['paired_bootstrap'].body if isinstance(n,ast.For)
                  and isinstance(n.target,ast.Name) and n.target.id=='first')
        self.assertEqual(hashlib.sha256(numerical_ast_dump(loop).encode()).hexdigest(),
                         '89fadbca4a617c69839acc6c245ab95db6bd79aa28cc64a41d7c23729e09dddf')

    def test_ast_identity_normalizes_only_optional_version_field(self):
        for name in NUMERIC_AST:
            older=copy.deepcopy(NODES[name])
            for part in ast.walk(older):
                if isinstance(part,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and 'type_params' in part._fields:
                    self.assertEqual(part.type_params,[])
                    part._fields=tuple(k for k in part._fields if k!='type_params')
                    del part.type_params
            self.assertEqual(numerical_ast_dump(older),numerical_ast_dump(NODES[name]))
        changed=copy.deepcopy(NODES['cells'])
        constant=next(n for n in ast.walk(changed) if isinstance(n,ast.Constant) and n.value==40)
        constant.value=41
        self.assertNotEqual(numerical_ast_dump(changed),numerical_ast_dump(NODES['cells']))

    def test_all_original_interval_numbers_unchanged(self):
        expected={.95:[-.8400000000000001,.38,-.18,1.04,-.8858443495986643,1.0858443495986643],
                  .975:[-.8700000000000003,.39,-.19,1.0700000000000003,-1.0274055194962761,1.2274055194962763]}
        keys=['basic_low','basic_high','percentile_low','percentile_high','se_low','se_high']
        for level,limits in expected.items():
            out=interval([-.2,0,.3,.5,1.1],.1,level)
            np.testing.assert_allclose([out[k] for k in keys],limits,atol=1e-15,rtol=1e-15)
            self.assertEqual(out['bootstrap_bias'],.24000000000000002)
            self.assertEqual(out['bootstrap_sd'],.5029910535983717)
            self.assertEqual(out['bias_over_sd'],.47714566349252646)
            self.assertFalse(out['method_sign_conflict'])
            self.assertEqual(out['bootstrap_mean'],.34)

    def test_large_same_sign_shift_is_only_nominal_direction(self):
        original=pd.DataFrame([row()]);before=original.copy(deep=True)
        out=report.interval_interpretations(original).iloc[0]
        self.assertEqual(out.interpretation,'nominal_basic_range_above_zero')
        self.assertGreater(out.bias_over_sd,6)
        self.assertFalse(out.method_sign_conflict)
        self.assertEqual(out.coverage_status,'not_verified')
        self.assertTrue(out.availability_status_is_not_coverage)
        for name in report.INTERVAL_DIAGNOSTICS:self.assertIn(name,out.index)
        pd.testing.assert_frame_equal(original,before)
        self.assertNotIn('model_a_larger_error',str(out.to_dict()))

    def test_sign_conflict_remains_visible_without_method_selection(self):
        out=report.interval_interpretations(pd.DataFrame([row(values=[.9,1.,1.1],point=.1)])).iloc[0]
        self.assertTrue(out.method_sign_conflict)
        self.assertEqual(out.interpretation,'nominal_basic_range_below_zero')
        self.assertGreater(out.percentile_low,0)

    def test_incomplete_draws_remain_na(self):
        original=row(values=[1.,np.nan,2.],point=1.5)
        self.assertEqual(original['status'],'not_estimable')
        self.assertEqual(original['finite_replicates'],2)
        out=report.interval_interpretations(pd.DataFrame([original])).iloc[0]
        self.assertEqual(out.interpretation,'not_estimable')
        for name in ['bootstrap_mean','bootstrap_bias','bootstrap_sd','bias_over_sd',
                     'basic_low','basic_high','percentile_low','percentile_high','se_low','se_high']:
            self.assertTrue(np.isnan(out[name]),name)
        text=report.interval_sections(pd.DataFrame([original]))
        self.assertIn('NA',text);self.assertIn('not_estimable',text)

    def test_only_fixed_975_is_primary_analysis(self):
        for support in ['fixed','reselected']:
            for level in [.95,.975]:
                m=metadata(support,level,2)
                self.assertEqual(m['primary_analysis'],support=='fixed' and level==.975)
                self.assertEqual(m['family_size'],2)
                self.assertEqual(m['primary_support'],'fixed')
                self.assertEqual(m['coverage_status'],'not_verified')
                self.assertIn('within this block',m['family_scope'])
        rows=pd.DataFrame([row(s,l) for s in ['fixed','reselected'] for l in [.95,.975]])
        out=report.interval_interpretations(rows)
        self.assertEqual(out.primary_analysis.sum(),1)
        self.assertEqual(out.primary_level.sum(),2)

    def test_endpoint_velocity_full_display_keeps_negative_and_empty(self):
        frame=metric_frame();before=frame.copy(deep=True);text=report.metric_sections(frame)
        for view in ['endpoint','velocity']:
            for metric in report.FIELD_DISPLAY_METRICS:self.assertIn(view+'_'+metric,text)
        for name in ['current_r_M','next_r_Psi','interval_velocity_MSE',
                     'anchor_transition_mean_row_TV','readout_self_transition_r_descriptive']:
            self.assertIn(name,text)
        self.assertIn('-0.25',text);self.assertIn('not_estimable',text);self.assertIn('NA',text)
        self.assertIn('2026',text);self.assertIn('clean',text)
        pd.testing.assert_frame_equal(frame,before)

    def test_proposal_displacement_is_not_called_actual_readout(self):
        frame=pd.DataFrame([{'block':b,'seed':42,'model':m} for b,m in [
            ('S1_full','proposal'),('S1_full','same_readout'),('S2','H'),('S2','HZ'),('S3','aug'),('S3','clean')]])
        text=report.metric_sections(frame)
        self.assertIn('Model displacement N-C versus Y-X',text)
        self.assertNotIn('### velocity: Actual readout change',text)
        for model in ['proposal','H','HZ']:
            self.assertIn('| '+model+' | conditional proposal displacement P-S |',text)
        for model in ['same_readout','aug','clean']:
            self.assertIn('| '+model+' | actual same-readout change T-S |',text)

    def test_c3_quantity_and_scope_boundaries(self):
        self.assertIn('endpoint N-X versus Y-X',report.C3_BOUNDARY)
        self.assertIn('T-X, not the actual trajectory T-S',report.C3_BOUNDARY)
        self.assertIn('not',report.C3_BOUNDARY)
        status=report.scope_status()
        self.assertIn('pre-existing',status['original_pure_SSL_and_task_only'])
        self.assertIn('not a prerequisite',status['S1_role'])
        self.assertIn('not a core-success requirement',status['S3_role'])

    def test_report_materialization_handles_no_supported_smoke_cells(self):
        with tempfile.TemporaryDirectory(prefix='report_contract_',dir=ROOT.parent) as temp:
            self.assertEqual(Path(temp).resolve().parent,ROOT.parent.resolve())
            root=Path(temp);out=root/'run';ev=out/'S1/full/seed_42/B_confirm/evaluation';ev.mkdir(parents=True)
            empty=metric_frame().iloc[[1]].drop(columns=['block','seed'])
            empty.to_csv(ev/'metrics.csv',index=False)
            inf=ev.parent/'inference';inf.mkdir()
            rows=pd.DataFrame([row(s,l,values=[1.,np.nan,2.]) for s in ['fixed','reselected'] for l in [.95,.975]])
            rows.drop(columns=['block']).to_csv(inf/'paired_intervals.csv',index=False)
            ctx=SimpleNamespace(out=out,history=root,contract='test-report-only',smoke=True,newseeds=(42,),nboot=20)
            def stage(base,name,contract,deps,produce):
                dst=Path(base)/name;dst.mkdir();produce(dst);return dst
            with patch.dict(sys.modules,{'ssl_pipeline':SimpleNamespace(assert_complete=lambda ctx:{})}), \
                 patch.object(report,'stage',stage),patch.object(report,'make_figures'),patch.object(report,'make_field_figures'):
                final=report.finalize(ctx)
            text=(final/'numeric_report.md').read_text(encoding='utf-8')
            self.assertFalse((final/'response_to_reviewers_SSL.md').exists())
            self.assertIn('SMOKE',text);self.assertIn('velocity_speed_r',text);self.assertIn('NA',text)
            self.assertIn('one training epoch',text)
            self.assertIn('20 paired learner draws',text)
            self.assertIn('Population coverage has not been verified',text)
            self.assertIn('T-X, not the actual trajectory T-S',text)
            self.assertNotIn('model_a_larger_error',(final/'result_interpretation.csv').read_text(encoding='utf-8'))
            parsed=pd.read_csv(final/'result_interpretation.csv')
            self.assertEqual(len(parsed),4)
            self.assertTrue((parsed.interpretation=='not_estimable').all())
            self.assertTrue(parsed.basic_low.isna().all())
            self.assertTrue(json.loads((final/'COMPLETE.json').read_text(encoding='utf-8'))['smoke'])


if __name__=='__main__':unittest.main(verbosity=2)
