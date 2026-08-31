# Numeric reference outputs

Commands: [reproduction order](../README.md#dependency-order). Code paths are relative to the repository root.

## Core experiments

| Public report | Source report | Extraction script |
| --- | --- | --- |
| [01_empirical_dynamics_and_kinetics_numeric_results.md](01_empirical_dynamics_and_kinetics_numeric_results.md) | `publication_results_fill_map.md` | `s1_empirical_dynamics/scripts/extract_empirical_effective_dynamics_publication_statistics.py` |
| [02_construction_matched_null_numeric_results.md](02_construction_matched_null_numeric_results.md) | `construction_matched_null_numeric_report.md` | `s1_empirical_dynamics/scripts/extract_construction_matched_null_numeric_report.py` |
| [03_minimal_mechanism_numeric_results.md](03_minimal_mechanism_numeric_results.md) | `minimal_mechanism_manuscript_numeric_report.md` | `s2_minimal_mechanism/scripts/extract_minimal_mechanism_publication_statistics.py` |
| [04_event_ssl_and_controls_numeric_results.md](04_event_ssl_and_controls_numeric_results.md) | `stage4_all_experiments_comparison_report.md` | `s3_event_ssl/scripts/extract_event_ssl_stage4_publication_statistics.py` |
| [05_event_ssl_macro_sufficiency_and_representation_geometry_numeric_results.md](05_event_ssl_macro_sufficiency_and_representation_geometry_numeric_results.md) | `stage5_joint_macro_geometry_report.md` | `s3_event_ssl/scripts/extract_event_ssl_stage5_publication_statistics.py` |
| [06_mechanism_event_ssl_cross_model_numeric_results.md](06_mechanism_event_ssl_cross_model_numeric_results.md) | `mechanism_event_ssl_macro_closure_comparison.md` | `s4_cross_analysis/scripts/extract_mechanism_event_ssl_publication_statistics.py` |

## Supplementary experiments

Reports below are generated locally.

| Source report | Generation code |
| --- | --- |
| `six_seed_null_common_target_audit_report.md` | `s4_cross_analysis/robustness/run_six_seed_null_common_target_audit.sh`<br>`s4_cross_analysis/main/run_six_seed_null_common_target_audit.py`<br>`s4_cross_analysis/scripts/extract_six_seed_null_common_target_audit_report.py` |
| `mechanism_score_contract_numeric_report.md` | `s2_minimal_mechanism/robustness/run_mechanism_score_contract_robustness.sh`<br>`s2_minimal_mechanism/robustness/audit_mechanism_score_contract_pareto.py`<br>`s2_minimal_mechanism/robustness/run_minimal_mechanism_score_contract_rerun.py`<br>`s2_minimal_mechanism/scripts/extract_mechanism_score_contract_numeric_report.py` |
| `frozen_headline_learner_cluster_uncertainty_report.md` | `s4_cross_analysis/robustness/run_frozen_headline_learner_cluster_uncertainty.sh`<br>`s4_cross_analysis/main/run_frozen_headline_learner_cluster_uncertainty.py`<br>`s4_cross_analysis/scripts/extract_frozen_headline_learner_cluster_uncertainty_report.py` |
| `objective_control_hidden_geometry_numeric_report.md` | `s3_event_ssl/robustness/objective_control/run_objective_control_hidden_geometry.sh`<br>`s3_event_ssl/robustness/objective_control/run_objective_control_hidden_geometry.py`<br>`s3_event_ssl/robustness/objective_control/extract_objective_control_hidden_geometry_report.py` |
| `state_only_closure_numeric_report.md` | `s3_event_ssl/robustness/state_only/run_state_only_closure_audit.sh`<br>`s3_event_ssl/robustness/state_only/run_state_only_closure_audit.py`<br>`s3_event_ssl/robustness/state_only/extract_state_only_closure_audit_report.py` |
| `empirical_excess_reliability_soft_core_report.md` | `s1_empirical_dynamics/robustness/excess_reliability/run_empirical_excess_reliability_soft_core.sh`<br>`s1_empirical_dynamics/robustness/excess_reliability/run_empirical_excess_reliability_soft_core.py`<br>`s1_empirical_dynamics/robustness/excess_reliability/extract_empirical_excess_reliability_soft_core_report.py` |
| `kinetic_robustness_numerical_report.md` | `s1_empirical_dynamics/robustness/kinetic/run_kinetic_robustness.sh`<br>`s1_empirical_dynamics/robustness/kinetic/run_partition_cluster_kinetic_robustness.py`<br>`s1_empirical_dynamics/robustness/kinetic/run_recursive_construction_inertia_null.py`<br>`s1_empirical_dynamics/robustness/kinetic/summarize_kinetic_robustness.py`<br>`s1_empirical_dynamics/robustness/kinetic/extract_kinetic_robustness_statistics.py` |
| `supplementary_robustness_numerical_report.md` | `s1_empirical_dynamics/robustness/aggregate/run_supplementary_robustness.sh`<br>`s1_empirical_dynamics/robustness/aggregate/run_empirical_user_robustness.py`<br>`s1_empirical_dynamics/robustness/aggregate/run_model_user_robustness.py`<br>`s1_empirical_dynamics/robustness/aggregate/run_representation_robustness.py`<br>`s1_empirical_dynamics/robustness/aggregate/summarize_supplementary_robustness.py`<br>`s1_empirical_dynamics/robustness/aggregate/extract_supplementary_robustness_statistics.py` |
| `supplementary_robustness_report.md` | `s1_empirical_dynamics/robustness/aggregate/run_supplementary_robustness.sh`<br>`s1_empirical_dynamics/robustness/aggregate/summarize_supplementary_robustness.py`<br>`s1_empirical_dynamics/robustness/aggregate/extract_supplementary_robustness_statistics.py` |
| `empirical_coordinate_sensitivity_report.md` | `s1_empirical_dynamics/robustness/sensitivity/run_stage1_empirical_sensitivity.sh`<br>`s1_empirical_dynamics/robustness/sensitivity/run_stage1_empirical_sensitivity.py`<br>`s1_empirical_dynamics/robustness/sensitivity/summarize_stage1_empirical_sensitivity.py`<br>`s1_empirical_dynamics/robustness/sensitivity/extract_empirical_coordinate_sensitivity_statistics.py` |
| `null_referenced_downstream_recovery_supplementary_report.md` | `s4_cross_analysis/robustness/run_null_referenced_downstream_recovery.sh`<br>`s4_cross_analysis/main/evaluate_null_referenced_downstream_recovery.py`<br>`s4_cross_analysis/scripts/extract_null_referenced_downstream_recovery_supplementary_statistics.py` |
| `event_ssl_supplementary_numerical_report.md` | `s5_supplementary/scripts/event_ssl/extract_event_ssl_supplementary_statistics.py`<br>`s5_supplementary/scripts/event_ssl/publication_event_ssl_supplementary_comparison.py` |
| `minimal_mechanism_supplementary_report.md` | `s5_supplementary/scripts/minimal_mechanism/extract_minimal_mechanism_supplementary_statistics.py` |
| `empirical_effective_dynamics_supplementary_report.md` | `s5_supplementary/scripts/empirical/extract_empirical_effective_dynamics_supplementary_statistics.py` |
| `event_ssl_random_seed_additional_information.md` | `s3_event_ssl/robustness/random_seed_workflows/seed_<seed>/run_all.sh`<br>`s3_event_ssl/scripts/extract_event_ssl_random_seed_statistics.py` |
| `semantic_specificity_control_report.md` | `s1_empirical_dynamics/robustness/semantic_specificity/run_semantic_specificity_control.sh`<br>`s1_empirical_dynamics/robustness/semantic_specificity/freeze_semantic_specificity_protocol.py`<br>`s1_empirical_dynamics/robustness/semantic_specificity/run_nonsemantic_coordinate_control.py`<br>`s1_empirical_dynamics/robustness/semantic_specificity/run_nonsemantic_construction_null.py`<br>`s1_empirical_dynamics/robustness/semantic_specificity/summarize_semantic_specificity_control.py` |
