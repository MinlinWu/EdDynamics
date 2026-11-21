from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ft_common import AuditError
from ft_reporting import legacy_reuse_statement, crossmodel_report_view

class ReportingRegressionTests(unittest.TestCase):

    def test_unavailable_legacy_is_not_claimed_as_reused(self):
        table = pd.DataFrame({'artifact': ['stage2_phase1_unified_minimality/tables/selected_model_parameter_deletions', 'stage2_phase1_score_contract_robustness/tables/score_contract_robustness_summary', 'stage2_phase1_score_contract_robustness/formal_audit/tables/frozen_fit_pareto_mean_components'], 'exists': [True, False, False]})
        statement = legacy_reuse_statement(table)
        self.assertIn('Historical evidence reused: term-deletion refits.', statement)
        self.assertIn('Not available in this run and not claimed as reused: equal-weight selection, Pareto checks.', statement)
        table['exists'] = False
        self.assertNotIn('Historical evidence reused:', legacy_reuse_statement(table))

    def test_crossmodel_display_keeps_estimates_and_maps_row_coverage(self):
        data = pd.DataFrame({'seed': [666, 42], 'rows': [60, 50], 'improvement_four_over_C3': [-0.002, 0.001], 'improvement_SSL_over_C3': [-0.003, -0.004]})
        before = data.copy(deep=True)
        view = crossmodel_report_view(data, [{'seed': 42, 'joined_rows': 50, 'M1_row_fraction': 0.5}, {'seed': 666, 'joined_rows': 60, 'M1_row_fraction': 0.6}])
        pd.testing.assert_frame_equal(data, before)
        pd.testing.assert_frame_equal(view[data.columns], data)
        self.assertEqual(view.M1_row_fraction.tolist(), [0.6, 0.5])

    def test_crossmodel_display_rejects_inconsistent_audit(self):
        data = pd.DataFrame({'seed': [42], 'rows': [50]})
        with self.assertRaises(AuditError):
            crossmodel_report_view(data, [{'seed': 42, 'joined_rows': 49, 'M1_row_fraction': 0.5}])
        with self.assertRaises(AuditError):
            crossmodel_report_view(data, [])

    def test_undefined_improvement_is_displayed_as_missing(self):
        data = pd.DataFrame({'seed': [42], 'rows': [50]})
        view = crossmodel_report_view(data, [{'seed': 42, 'joined_rows': 50, 'M1_row_fraction': 0.5}])
        self.assertTrue(view[['improvement_four_over_C3', 'improvement_SSL_over_C3']].isna().all().all())
if __name__ == '__main__':
    unittest.main(verbosity=2)
