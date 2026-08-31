---
source_report: "publication_results_fill_map.md"
source_extraction_script: "s1_empirical_dynamics/scripts/extract_empirical_effective_dynamics_publication_statistics.py"
---

# Empirical dynamics and mesostate kinetics numerical results

| Manuscript quantity | Output key | Value |
| --- | --- | --- |
| Training-defined compact flow-localization M interval (50% evidence) | `training_primary_convergence_flow_localization_M_shortest_interval_50_low / training_primary_convergence_flow_localization_M_shortest_interval_50_high` | -0.12499999999999994, 0.275 |
| Training-defined compact flow-localization Psi interval (50% evidence) | `training_primary_convergence_flow_localization_Psi_shortest_interval_50_low / training_primary_convergence_flow_localization_Psi_shortest_interval_50_high` | -0.575, -0.525 |
| Training-defined flow-weighted convergence centre | `training_primary_convergence_convergence_center_M / training_primary_convergence_convergence_center_Psi` | -0.019744368853791446, -0.5820368461646965 |
| Training local-affine fixed-point estimate | `training_primary_convergence_local_fixed_point_M / training_primary_convergence_local_fixed_point_Psi` | -0.042538681338687406, -0.7628689146630685 |
| Validation supported drift cells | `validation_global_drift_supported_cells` | 940 |
| Validation interior-only negative-divergence occupancy fraction | `validation_global_weighted_negative_divergence_fraction_interior_only` | 0.7762411953842572 |
| Validation interior-only weighted mean divergence | `validation_global_weighted_mean_local_divergence_interior_only` | -0.2152712779777765 |
| Validation occupancy mass in frozen convergence core | `validation_frozen_primary_convergence_occupancy_mass_fraction` | 0.18509147620595368 |
| Validation flow-weighted inward shell fraction | `validation_frozen_primary_convergence_flow_weighted_shell_fraction_inward` | 0.6693353086453588 |
| Validation flow-weighted core-to-shell drift-speed ratio | `validation_frozen_primary_convergence_flow_core_to_shell_speed_ratio` | 0.5225984094550938 |
| Training-validation convergence-mask Jaccard | `train_validation_convergence_mask_jaccard` | 0.43776824034334766 |
| Training-validation convergence-centre distance | `train_validation_convergence_convergence_center_distance` | 0.17214716951200584 |
| Validation first-shell flow-weighted inward fraction | `validation_convergence_radial_first_shell_flow_weighted_fraction_inward` | 0.6387594472835875 |
| Validation first-shell flow-weighted inward cosine | `validation_convergence_radial_first_shell_flow_weighted_mean_inward_cosine` | 0.23108874380824435 |
| Validation Psi 50% shortest occupancy interval | `validation_occupancy_Psi_shortest_interval_50_low / validation_occupancy_Psi_shortest_interval_50_high` | -0.875, -0.525 |
| Validation M outermost-bin occupancy mass | `validation_occupancy_M_outermost_one_bin_mass_fraction` | 0.2790799290874363 |
| Macrostate count | `macrostate_k` | 6 |
| Validation transitions | `validation_transition_count` | 3328409 |
| Diagonal-dominant states | `validation_diagonal_dominant_states` | 6 |
| Mean self-transition probability | `validation_mean_diagonal_probability` | 0.7496129751536813 |
| Self-transition range | `validation_diagonal_probability_range` | 0.362--0.987 |
| Validation residence runs | `validation_residence_runs_total` | 256009 |
| Completed exits / right-censored runs | `validation_residence_completed_exits_total / validation_residence_right_censored_total` | 198364, 57645 |
| Overall residence right-censoring fraction | `validation_residence_right_censoring_fraction` | 0.22516786519223933 |
| State-specific RMST horizon range | `validation_residence_rmst_tau_range` | 35--4302 |
| Censor-aware restricted-mean residence-ratio range | `validation_restricted_mean_residence_lift_range` | 1.127--6.473 |
| Reference residence length | `residence_reference_length` | 10 |
| At-risk range at the reference length | `validation_reference_at_risk_range` | 318--14961 |
| Kaplan--Meier tail-ratio range at the reference length | `validation_tail_ratio_at_reference_range` | 0.463--170.821 |
| States with descriptive tail excess at the reference length | `validation_tail_excess_states_descriptive_at_reference` | 3 |
| States with BH-significant tail excess at the reference length | `validation_tail_excess_states_significant_bh` | 3 |
| States with descriptive KM tail intervals | `validation_km_tail_excess_states_descriptive` | 6 |
| Reliable KM tail horizon maximum | `validation_km_tail_max_reliable_length` | 4302 |
| Occupancy JS divergence | `train_validation_occupancy_js_divergence` | 0.0005238103314522346 |
| Common supported drift cells | `train_validation_common_supported_drift_cells` | 939 |
| Mean local drift cosine | `train_validation_mean_local_drift_cosine` | 0.9656753869703304 |
| Drift component RMSE | `train_validation_drift_component_rmse` | 0.04513859173177252 |
| Drift-speed correlation | `train_validation_drift_speed_pearson` | 0.9295011460697471 |
| Residence RMST-ratio mean absolute log difference | `train_validation_residence_rmst_mean_abs_log_ratio` | 0.05708827532218013 |
| Residence fixed-reference tail-ratio mean absolute log difference | `train_validation_residence_tail_ratio_mean_abs_log_ratio` | 0.10221534023083789 |
| Residence right-censoring-fraction mean absolute difference | `train_validation_residence_right_censoring_fraction_mean_abs_difference` | 0.0020658689643619395 |
| Transition mean row TV | `train_validation_transition_mean_row_total_variation` | 0.0062051756171805595 |
| Transition maximum row TV | `train_validation_transition_max_row_total_variation` | 0.016572165008165322 |
