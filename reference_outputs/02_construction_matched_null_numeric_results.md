---
source_report: "construction_matched_null_numeric_report.md"
source_extraction_script: "s1_empirical_dynamics/scripts/extract_construction_matched_null_numeric_report.py"
---

# Construction-matched null numerical results

## A_val: analysis contract

| split | analysis_role | rows_in_analysis_panel | users_in_analysis_panel | valid_drift_rows | replicates | base_seed | minimum_monte_carlo_p | primary_coordinates | shell_radius | frozen_core_sha256 | frozen_thresholds_sha256 | formal_stage1_script_sha256 | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | formal primary test | 3,447,590 | 59,492 | 3,328,409 | 100 | 42 | 0.009901 | M;Psi | 0.350000 | 751d5cd8a4cf6eb9c38c87f0850691e4b0fc165908d336daff5c1d1807a4bf79 | 5db23ca83b0e9790e73172ddb8795e35f71ee2647814b3d7e23c6bb804321250 | 5f5e3e5c52949a9f8217d803c05aea8d34f296860f9b7b32f620859c7810c5ea | 279.4163 |

## A_val: prespecified formal tests

| metric_label | test_direction | observed | pure_ratio_contraction | null_mean | null_2p5 | null_97p5 | monte_carlo_p | BH_q_across_three_basin_metrics | descriptive_standardized_separation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Occupancy-weighted full-field distance from the matched-null mean | greater | 0.090968 | 0.178778 | 0.009336 | 0.008908 | 0.009905 | 0.009901 | NA | 324.6823 |
| Interior occupancy fraction with negative local divergence | greater | 0.776241 | 0.808573 | 0.737841 | 0.718350 | 0.759556 | 0.009901 | 0.029703 | 3.365850 |
| Flow-weighted inward fraction in the frozen shell | greater | 0.669335 | 0.567136 | 0.662474 | 0.654120 | 0.669531 | 0.039604 | 0.059406 | 1.693909 |
| Flow-weighted core-to-shell drift-speed ratio | less | 0.522598 | 0.652353 | 0.492139 | 0.477744 | 0.505044 | 1 | 1 | -4.586620 |

### A_val: formal test metrics

| split | metric | test_role | direction | formal_value |
| --- | --- | --- | --- | --- |
| A_val | occupancy_weighted_full_field_distance_from_null_mean | primary full-field test | greater | 0.009901 |
| A_val | negative_divergence_occupancy_fraction | prespecified basin metric | greater | 0.029703 |
| A_val | flow_weighted_shell_fraction_inward | prespecified basin metric | greater | 0.059406 |
| A_val | flow_core_to_shell_speed_ratio | prespecified basin metric | less | 1 |

### A_val: observed, matched-null, pure-ratio and excess values

| metric | test_direction | observed | pure_ratio_contraction | null_mean | null_sd | null_2p5 | null_50 | null_97p5 | monte_carlo_p | BH_q_across_three_basin_metrics | excess_field_value_descriptive | observed_minus_null_mean | directional_difference | descriptive_standardized_separation | tail_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupancy_weighted_full_field_distance_from_null_mean | greater | 0.090968 | 0.178778 | 0.009336 | 0.000251 | 0.008908 | 0.009306 | 0.009905 | 0.009901 | NA | 0.090968 | 0.081632 | 0.081632 | 324.6823 | 1 |
| negative_divergence_occupancy_fraction | greater | 0.776241 | 0.808573 | 0.737841 | 0.011409 | 0.718350 | 0.736970 | 0.759556 | 0.009901 | 0.029703 | 0.699861 | 0.038400 | 0.038400 | 3.365850 | 1 |
| flow_weighted_shell_fraction_inward | greater | 0.669335 | 0.567136 | 0.662474 | 0.004050 | 0.654120 | 0.662444 | 0.669531 | 0.039604 | 0.059406 | 0.777743 | 0.006861 | 0.006861 | 1.693909 | 0.970000 |
| flow_core_to_shell_speed_ratio | less | 0.522598 | 0.652353 | 0.492139 | 0.006641 | 0.477744 | 0.492253 | 0.505044 | 1 | 1 | 0.736233 | 0.030460 | -0.030460 | -4.586620 | 0 |

### A_val: null-replicate distributions

| split | metric | metric_label | finite_replicates | minimum | q2p5 | q25 | median | q75 | q97p5 | maximum | mean | sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | negative_divergence_occupancy_fraction | Interior occupancy fraction with negative local divergence | 100 | 0.709426 | 0.718350 | 0.730054 | 0.736970 | 0.746611 | 0.759556 | 0.763393 | 0.737841 | 0.011409 |
| A_val | weighted_mean_divergence | Interior occupancy-weighted mean local divergence | 100 | -0.174292 | -0.171065 | -0.165153 | -0.161772 | -0.158713 | -0.152617 | -0.151338 | -0.161697 | 0.004856 |
| A_val | flow_weighted_shell_fraction_inward | Flow-weighted inward fraction in the frozen shell | 100 | 0.650230 | 0.654120 | 0.660398 | 0.662444 | 0.665293 | 0.669531 | 0.670914 | 0.662474 | 0.004050 |
| A_val | flow_weighted_shell_inward_cosine | Flow-weighted shell inward cosine | 100 | 0.341820 | 0.344217 | 0.349887 | 0.352000 | 0.354617 | 0.358066 | 0.360210 | 0.351938 | 0.003809 |
| A_val | flow_core_to_shell_speed_ratio | Flow-weighted core-to-shell drift-speed ratio | 100 | 0.473004 | 0.477744 | 0.488257 | 0.492253 | 0.496386 | 0.505044 | 0.506627 | 0.492139 | 0.006641 |
| A_val | occupancy_core_to_shell_speed_ratio | Occupancy-weighted core-to-shell drift-speed ratio | 100 | 0.474149 | 0.477699 | 0.486081 | 0.490339 | 0.494284 | 0.500065 | 0.503410 | 0.490262 | 0.005883 |
| A_val | occupancy_weighted_full_field_distance_from_null_mean | Occupancy-weighted full-field distance from the matched-null mean | 100 | 0.008795 | 0.008908 | 0.009174 | 0.009306 | 0.009469 | 0.009905 | 0.010007 | 0.009336 | 0.000251 |

### A_val: matching summary

| split | analysis_rows | randomizable_rows_before_singleton_exemption | randomized_rows | zero_innovation_rows | within_user_matched_rows | across_user_matched_rows | global_opportunity_rows | global_last_resort_rows | unmatched_singleton_self_exempt_rows | within_user_fraction_of_randomizable | across_user_fraction_of_randomizable | global_opportunity_fraction_of_randomizable | weak_fallback_fraction_of_randomizable | randomized_fraction_of_analysis_rows | zero_innovation_fraction_of_analysis_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | 3,328,409 | 3,328,409 | 3,328,409 | 0 | 3,312,375 | 16,022 | 9 | 3 | 0 | 0.995183 | 0.004814 | 2.7040e-06 | 9.0133e-07 | 1 | 0 |

### A_val: hierarchical matching coverage

| level | matching_keys | rows_assigned | fraction_of_randomizable_rows | rows_remaining_after_level |
| --- | --- | --- | --- | --- |
| within_user_fine | user_id;response_present;support_present;idle_present;gap_bin | 3,305,784 | 0.993202 | 22,625 |
| within_user_coarse | user_id;response_present;exposure_present | 6,591 | 0.001980 | 16,034 |
| across_user_fine | part;response_present;support_present;idle_present;gap_bin;a_m_bin;a_psi_bin;support_share_bin;idle_share_bin;sequence_length_bin | 15,345 | 0.004610 | 689 |
| across_user_coarse | part;response_present;support_present;idle_present;gap_bin | 677 | 0.000203 | 12 |
| global_opportunity | response_present;support_present;idle_present;exposure_present | 9 | 2.7040e-06 | 3 |
| global_last_resort | constant | 3 | 9.0133e-07 | 0 |
| deterministic_zero_innovation | not randomized because both denominator increments are zero | 0 | NA | 0 |

### A_val: preserved opportunity composition

| analysis_rows | randomizable_rows | zero_innovation_rows | response_increment_present_fraction | exposure_increment_present_fraction | support_present_fraction | idle_present_fraction | mean_response_active_mass | mean_support_active_mass | mean_idle_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3,328,409 | 3,328,409 | 0 | 0.959571 | 1 | 0.874112 | 0.999978 | 0.920582 | 0.162685 | 0.023384 |

### A_val: A_train matching cutpoints

| split | matching_variable | cutpoints | fit_split | fit_rows_sampled | fit_users | rows_scanned | eligible_transition_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | log_a_m | 0.227358;0.335070;0.449304 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| A_val | log_a_psi | 0.702339;0.725295;0.779396 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| A_val | support_share | 0.039612;0.131518 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| A_val | idle_share | 1.2834e-06;3.5832e-05 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| A_val | sequence_length | 4;5;7;8;13;27;95 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |

### A_val: field-pair diagnostics

| split | comparison | common_supported_cells | occupancy_mass_on_common_support | field_vector_correlation | field_speed_correlation | occupancy_weighted_local_cosine | occupancy_weighted_rms_vector_distance | occupancy_weighted_mean_speed_first | occupancy_weighted_mean_speed_second | occupancy_weighted_fraction_positive_local_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | observed_vs_matched_null_mean | 940 | 0.996276 | 0.845964 | 0.826007 | 0.707215 | 0.090968 | 0.164213 | 0.155225 | 0.890985 |
| A_val | observed_vs_pure_ratio_contraction | 940 | 0.996276 | 0.734760 | 0.645359 | 0.139118 | 0.174564 | 0.164213 | 0.174584 | 0.607843 |
| A_val | matched_null_mean_vs_pure_ratio_contraction | 940 | 0.996276 | 0.682653 | 0.749909 | 0.103380 | 0.178778 | 0.155225 | 0.174584 | 0.583837 |

### A_val: cellwise excess-field diagnostics

| split | grid_shape | grid_cells | state_supported_cells | drift_supported_cells | frozen_core_cells | supported_core_cells | supported_occupancy_mass | core_occupancy_fraction_within_supported_field | occupancy_weighted_observed_speed | occupancy_weighted_matched_null_mean_speed | occupancy_weighted_pure_ratio_speed | occupancy_weighted_excess_speed | occupancy_weighted_null_vector_sd_magnitude | occupancy_fraction_with_nonzero_null_sd | occupancy_weighted_median_excess_to_null_sd | occupancy_fraction_excess_to_null_sd_gt_1_within_nonzero_sd | occupancy_fraction_excess_to_null_sd_gt_2_within_nonzero_sd | occupancy_fraction_observed_distance_above_cellwise_null_q97p5_descriptive | occupancy_fraction_excess_aligned_with_observed | occupancy_weighted_excess_speed_in_core | occupancy_weighted_excess_speed_outside_core |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | 40x40 | 1,600 | 842 | 940 | 164 | 164 | 0.996276 | 0.185783 | 0.164213 | 0.155225 | 0.174584 | 0.067896 | 0.006268 | 1 | 9.130077 | 0.986582 | 0.943728 | 0.942181 | 0.576177 | 0.033004 | 0.075857 |

### A_val: reconstruction, archived-field and permutation audit

| split | audit_group | metric | value |
| --- | --- | --- | --- |
| A_val | reconstruction | rows | 3,447,590 |
| A_val | reconstruction | formal_drift_rows | 3,328,409 |
| A_val | reconstruction | response_positive_increment_rows | 3,193,845 |
| A_val | reconstruction | exposure_positive_increment_rows | 3,328,409 |
| A_val | reconstruction | max_abs_next_M_reconstruction_error | 1.2212e-15 |
| A_val | reconstruction | max_abs_next_Psi_reconstruction_error | 2.7645e-14 |
| A_val | reconstruction | max_abs_delta_M_reconstruction_error | 1.1102e-15 |
| A_val | reconstruction | max_abs_delta_Psi_reconstruction_error | 2.7579e-14 |
| A_val | reconstruction | max_response_Z_bound_excess_before_clipping | 1.2991e-11 |
| A_val | reconstruction | max_exposure_Z_bound_excess_before_clipping | 0 |
| A_val | reconstruction | innovation_source | same-row pre-to-response and pre-to-post phase accounting |
| A_val | reconstruction | max_abs_next_response_mass_decay_error | 1.3642e-12 |
| A_val | reconstruction | max_rel_next_response_mass_decay_error | 8.2011e-13 |
| A_val | reconstruction | max_abs_next_exposure_denominator_decay_error | 2.2737e-12 |
| A_val | reconstruction | max_rel_next_exposure_denominator_decay_error | 9.2967e-13 |
| A_val | reconstruction | max_abs_next_exposure_numerator_decay_error | 1.3642e-12 |
| A_val | reconstruction | max_rel_next_exposure_numerator_decay_error | 3.8075e-08 |
| A_val | reconstruction | formal_field_estimator_reproduced_to_1e-12 | true |
| A_val | archived_stage1_field | skipped | false |
| A_val | archived_stage1_field | saved_field_path | /data/datasets/KT4/outputs_KT4/stage1/dynamics/coordinate_analysis/MR_PsiA/A_val_publication_field_grid.parquet |
| A_val | archived_stage1_field | max_abs_drift_M_difference | 0 |
| A_val | archived_stage1_field | max_abs_drift_Psi_difference | 0 |
| A_val | archived_stage1_field | max_abs_occupancy_probability_difference | 0 |
| A_val | archived_stage1_field | drift_mask_exact_match | true |
| A_val | first_permutation | randomized_rows | 3,328,409 |
| A_val | first_permutation | fixed_points_among_randomized_rows | 0 |
| A_val | first_permutation | mean_Z_M_before | 0.266270 |
| A_val | first_permutation | mean_Z_M_after | 0.266270 |
| A_val | first_permutation | mean_Z_Psi_before | -0.694644 |
| A_val | first_permutation | mean_Z_Psi_after | -0.694644 |
| A_val | first_permutation | mean_product_Z_before | -0.203805 |
| A_val | first_permutation | mean_product_Z_after | -0.203805 |
| A_val | first_permutation | joint_pair_moved_together | true |
| A_val | first_permutation | overall_mapping_bijective_by_disjoint_group_permutations | true |

## B_confirm: analysis contract

| split | analysis_role | rows_in_analysis_panel | users_in_analysis_panel | valid_drift_rows | replicates | base_seed | minimum_monte_carlo_p | primary_coordinates | shell_radius | frozen_core_sha256 | frozen_thresholds_sha256 | formal_stage1_script_sha256 | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | frozen output-only replication | 3,352,259 | 59,496 | 3,233,208 | 100 | 42 | 0.009901 | M;Psi | 0.350000 | 751d5cd8a4cf6eb9c38c87f0850691e4b0fc165908d336daff5c1d1807a4bf79 | 5db23ca83b0e9790e73172ddb8795e35f71ee2647814b3d7e23c6bb804321250 | 5f5e3e5c52949a9f8217d803c05aea8d34f296860f9b7b32f620859c7810c5ea | 274.2362 |

## B_confirm: prespecified formal tests

| metric_label | test_direction | observed | pure_ratio_contraction | null_mean | null_2p5 | null_97p5 | monte_carlo_p | BH_q_across_three_basin_metrics | descriptive_standardized_separation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Occupancy-weighted full-field distance from the matched-null mean | greater | 0.090738 | 0.178476 | 0.009161 | 0.008775 | 0.009561 | 0.009901 | NA | 358.6195 |
| Interior occupancy fraction with negative local divergence | greater | 0.810259 | 0.799854 | 0.738205 | 0.715721 | 0.756671 | 0.009901 | 0.014851 | 6.468299 |
| Flow-weighted inward fraction in the frozen shell | greater | 0.676868 | 0.564699 | 0.664248 | 0.656695 | 0.671418 | 0.009901 | 0.014851 | 3.412460 |
| Flow-weighted core-to-shell drift-speed ratio | less | 0.504716 | 0.649190 | 0.491946 | 0.480728 | 0.504851 | 0.970297 | 0.970297 | -2.001862 |

### B_confirm: formal test metrics

| split | metric | test_role | direction | formal_value |
| --- | --- | --- | --- | --- |
| B_confirm | occupancy_weighted_full_field_distance_from_null_mean | primary full-field test | greater | 0.009901 |
| B_confirm | negative_divergence_occupancy_fraction | prespecified basin metric | greater | 0.014851 |
| B_confirm | flow_weighted_shell_fraction_inward | prespecified basin metric | greater | 0.014851 |
| B_confirm | flow_core_to_shell_speed_ratio | prespecified basin metric | less | 0.970297 |

### B_confirm: observed, matched-null, pure-ratio and excess values

| metric | test_direction | observed | pure_ratio_contraction | null_mean | null_sd | null_2p5 | null_50 | null_97p5 | monte_carlo_p | BH_q_across_three_basin_metrics | excess_field_value_descriptive | observed_minus_null_mean | directional_difference | descriptive_standardized_separation | tail_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupancy_weighted_full_field_distance_from_null_mean | greater | 0.090738 | 0.178476 | 0.009161 | 0.000227 | 0.008775 | 0.009150 | 0.009561 | 0.009901 | NA | 0.090738 | 0.081578 | 0.081578 | 358.6195 | 1 |
| negative_divergence_occupancy_fraction | greater | 0.810259 | 0.799854 | 0.738205 | 0.011140 | 0.715721 | 0.738206 | 0.756671 | 0.009901 | 0.014851 | 0.695037 | 0.072054 | 0.072054 | 6.468299 | 1 |
| flow_weighted_shell_fraction_inward | greater | 0.676868 | 0.564699 | 0.664248 | 0.003698 | 0.656695 | 0.664698 | 0.671418 | 0.009901 | 0.014851 | 0.775664 | 0.012620 | 0.012620 | 3.412460 | 1 |
| flow_core_to_shell_speed_ratio | less | 0.504716 | 0.649190 | 0.491946 | 0.006379 | 0.480728 | 0.492129 | 0.504851 | 0.970297 | 0.970297 | 0.731598 | 0.012770 | -0.012770 | -2.001862 | 0.030000 |

### B_confirm: null-replicate distributions

| split | metric | metric_label | finite_replicates | minimum | q2p5 | q25 | median | q75 | q97p5 | maximum | mean | sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | negative_divergence_occupancy_fraction | Interior occupancy fraction with negative local divergence | 100 | 0.710269 | 0.715721 | 0.731273 | 0.738206 | 0.745978 | 0.756671 | 0.768066 | 0.738205 | 0.011140 |
| B_confirm | weighted_mean_divergence | Interior occupancy-weighted mean local divergence | 100 | -0.181372 | -0.177930 | -0.174384 | -0.171112 | -0.167752 | -0.163267 | -0.160256 | -0.170934 | 0.004341 |
| B_confirm | flow_weighted_shell_fraction_inward | Flow-weighted inward fraction in the frozen shell | 100 | 0.653847 | 0.656695 | 0.662102 | 0.664698 | 0.666606 | 0.671418 | 0.673864 | 0.664248 | 0.003698 |
| B_confirm | flow_weighted_shell_inward_cosine | Flow-weighted shell inward cosine | 100 | 0.339126 | 0.341727 | 0.346949 | 0.350255 | 0.353444 | 0.357707 | 0.358836 | 0.350161 | 0.004476 |
| B_confirm | flow_core_to_shell_speed_ratio | Flow-weighted core-to-shell drift-speed ratio | 100 | 0.478618 | 0.480728 | 0.486818 | 0.492129 | 0.495979 | 0.504851 | 0.511221 | 0.491946 | 0.006379 |
| B_confirm | occupancy_core_to_shell_speed_ratio | Occupancy-weighted core-to-shell drift-speed ratio | 100 | 0.478949 | 0.481599 | 0.488762 | 0.491920 | 0.496105 | 0.502162 | 0.509446 | 0.492144 | 0.005467 |
| B_confirm | occupancy_weighted_full_field_distance_from_null_mean | Occupancy-weighted full-field distance from the matched-null mean | 100 | 0.008488 | 0.008775 | 0.009036 | 0.009150 | 0.009312 | 0.009561 | 0.009708 | 0.009161 | 0.000227 |

### B_confirm: matching summary

| split | analysis_rows | randomizable_rows_before_singleton_exemption | randomized_rows | zero_innovation_rows | within_user_matched_rows | across_user_matched_rows | global_opportunity_rows | global_last_resort_rows | unmatched_singleton_self_exempt_rows | within_user_fraction_of_randomizable | across_user_fraction_of_randomizable | global_opportunity_fraction_of_randomizable | weak_fallback_fraction_of_randomizable | randomized_fraction_of_analysis_rows | zero_innovation_fraction_of_analysis_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | 3,233,208 | 3,233,208 | 3,233,208 | 0 | 3,217,047 | 16,149 | 12 | 0 | 0 | 0.995002 | 0.004995 | 3.7115e-06 | 0 | 1 | 0 |

### B_confirm: hierarchical matching coverage

| level | matching_keys | rows_assigned | fraction_of_randomizable_rows | rows_remaining_after_level |
| --- | --- | --- | --- | --- |
| within_user_fine | user_id;response_present;support_present;idle_present;gap_bin | 3,210,493 | 0.992974 | 22,715 |
| within_user_coarse | user_id;response_present;exposure_present | 6,554 | 0.002027 | 16,161 |
| across_user_fine | part;response_present;support_present;idle_present;gap_bin;a_m_bin;a_psi_bin;support_share_bin;idle_share_bin;sequence_length_bin | 15,538 | 0.004806 | 623 |
| across_user_coarse | part;response_present;support_present;idle_present;gap_bin | 611 | 0.000189 | 12 |
| global_opportunity | response_present;support_present;idle_present;exposure_present | 12 | 3.7115e-06 | 0 |
| deterministic_zero_innovation | not randomized because both denominator increments are zero | 0 | NA | 0 |

### B_confirm: preserved opportunity composition

| analysis_rows | randomizable_rows | zero_innovation_rows | response_increment_present_fraction | exposure_increment_present_fraction | support_present_fraction | idle_present_fraction | mean_response_active_mass | mean_support_active_mass | mean_idle_mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3,233,208 | 3,233,208 | 0 | 0.956358 | 1 | 0.869599 | 0.999976 | 0.917137 | 0.164969 | 0.024157 |

### B_confirm: A_train matching cutpoints

| split | matching_variable | cutpoints | fit_split | fit_rows_sampled | fit_users | rows_scanned | eligible_transition_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | log_a_m | 0.227358;0.335070;0.449304 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| B_confirm | log_a_psi | 0.702339;0.725295;0.779396 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| B_confirm | support_share | 0.039612;0.131518 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| B_confirm | idle_share | 1.2834e-06;3.5832e-05 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |
| B_confirm | sequence_length | 4;5;7;8;13;27;95 | A_train | 500,000 | 168,549 | 9,688,212 | 9,331,219 |

### B_confirm: field-pair diagnostics

| split | comparison | common_supported_cells | occupancy_mass_on_common_support | field_vector_correlation | field_speed_correlation | occupancy_weighted_local_cosine | occupancy_weighted_rms_vector_distance | occupancy_weighted_mean_speed_first | occupancy_weighted_mean_speed_second | occupancy_weighted_fraction_positive_local_cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | observed_vs_matched_null_mean | 954 | 0.996143 | 0.850315 | 0.819078 | 0.708216 | 0.090738 | 0.165128 | 0.155950 | 0.882120 |
| B_confirm | observed_vs_pure_ratio_contraction | 954 | 0.996143 | 0.725345 | 0.628210 | 0.143794 | 0.174375 | 0.165128 | 0.174877 | 0.621532 |
| B_confirm | matched_null_mean_vs_pure_ratio_contraction | 954 | 0.996143 | 0.680792 | 0.749535 | 0.107300 | 0.178476 | 0.155950 | 0.174877 | 0.588283 |

### B_confirm: cellwise excess-field diagnostics

| split | grid_shape | grid_cells | state_supported_cells | drift_supported_cells | frozen_core_cells | supported_core_cells | supported_occupancy_mass | core_occupancy_fraction_within_supported_field | occupancy_weighted_observed_speed | occupancy_weighted_matched_null_mean_speed | occupancy_weighted_pure_ratio_speed | occupancy_weighted_excess_speed | occupancy_weighted_null_vector_sd_magnitude | occupancy_fraction_with_nonzero_null_sd | occupancy_weighted_median_excess_to_null_sd | occupancy_fraction_excess_to_null_sd_gt_1_within_nonzero_sd | occupancy_fraction_excess_to_null_sd_gt_2_within_nonzero_sd | occupancy_fraction_observed_distance_above_cellwise_null_q97p5_descriptive | occupancy_fraction_excess_aligned_with_observed | occupancy_weighted_excess_speed_in_core | occupancy_weighted_excess_speed_outside_core |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | 40x40 | 1,600 | 850 | 954 | 164 | 164 | 0.996143 | 0.184148 | 0.165128 | 0.155950 | 0.174877 | 0.067811 | 0.006254 | 1 | 9.209151 | 0.987138 | 0.941924 | 0.942912 | 0.569502 | 0.033121 | 0.075641 |

### B_confirm: reconstruction, archived-field and permutation audit

| split | audit_group | metric | value |
| --- | --- | --- | --- |
| B_confirm | reconstruction | rows | 3,352,259 |
| B_confirm | reconstruction | formal_drift_rows | 3,233,208 |
| B_confirm | reconstruction | response_positive_increment_rows | 3,092,104 |
| B_confirm | reconstruction | exposure_positive_increment_rows | 3,233,208 |
| B_confirm | reconstruction | max_abs_next_M_reconstruction_error | 9.9920e-16 |
| B_confirm | reconstruction | max_abs_next_Psi_reconstruction_error | 2.1427e-14 |
| B_confirm | reconstruction | max_abs_delta_M_reconstruction_error | 8.8818e-16 |
| B_confirm | reconstruction | max_abs_delta_Psi_reconstruction_error | 2.1316e-14 |
| B_confirm | reconstruction | max_response_Z_bound_excess_before_clipping | 5.6453e-12 |
| B_confirm | reconstruction | max_exposure_Z_bound_excess_before_clipping | 0 |
| B_confirm | reconstruction | innovation_source | same-row pre-to-response and pre-to-post phase accounting |
| B_confirm | reconstruction | max_abs_next_response_mass_decay_error | 1.3642e-12 |
| B_confirm | reconstruction | max_rel_next_response_mass_decay_error | 9.0002e-13 |
| B_confirm | reconstruction | max_abs_next_exposure_denominator_decay_error | 1.8190e-12 |
| B_confirm | reconstruction | max_rel_next_exposure_denominator_decay_error | 8.4541e-13 |
| B_confirm | reconstruction | max_abs_next_exposure_numerator_decay_error | 1.3642e-12 |
| B_confirm | reconstruction | max_rel_next_exposure_numerator_decay_error | 1.9619e-08 |
| B_confirm | reconstruction | formal_field_estimator_reproduced_to_1e-12 | true |
| B_confirm | archived_stage1_field | skipped | false |
| B_confirm | archived_stage1_field | saved_field_path | /data/datasets/KT4/outputs_KT4/stage1/dynamics/coordinate_analysis/MR_PsiA/B_confirm_publication_field_grid_output_only.parquet |
| B_confirm | archived_stage1_field | max_abs_drift_M_difference | 0 |
| B_confirm | archived_stage1_field | max_abs_drift_Psi_difference | 0 |
| B_confirm | archived_stage1_field | max_abs_occupancy_probability_difference | 0 |
| B_confirm | archived_stage1_field | drift_mask_exact_match | true |
| B_confirm | first_permutation | randomized_rows | 3,233,208 |
| B_confirm | first_permutation | fixed_points_among_randomized_rows | 0 |
| B_confirm | first_permutation | mean_Z_M_before | 0.256758 |
| B_confirm | first_permutation | mean_Z_M_after | 0.256758 |
| B_confirm | first_permutation | mean_Z_Psi_before | -0.681750 |
| B_confirm | first_permutation | mean_Z_Psi_after | -0.681750 |
| B_confirm | first_permutation | mean_product_Z_before | -0.192918 |
| B_confirm | first_permutation | mean_product_Z_after | -0.192918 |
| B_confirm | first_permutation | joint_pair_moved_together | true |
| B_confirm | first_permutation | overall_mapping_bijective_by_disjoint_group_permutations | true |

## Validation–confirmation numerical comparison

### Cross-split field agreement

| comparison | common_supported_cells | occupancy_mass_on_common_support | field_vector_correlation | field_speed_correlation | occupancy_weighted_local_cosine | occupancy_weighted_rms_vector_distance | occupancy_weighted_mean_speed_first | occupancy_weighted_mean_speed_second | occupancy_weighted_fraction_positive_local_cosine | first_split | second_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| observed_fields | 917 | 0.995228 | 0.927234 | 0.908507 | 0.970685 | 0.014341 | 0.164518 | 0.164723 | 0.995196 | A_val | B_confirm |
| matched_null_mean_fields | 917 | 0.995228 | 0.959780 | 0.951669 | 0.995731 | 0.007985 | 0.155635 | 0.155626 | 1 | A_val | B_confirm |
| excess_fields | 917 | 0.995228 | 0.716408 | 0.736176 | 0.960443 | 0.015239 | 0.067995 | 0.067555 | 0.988589 | A_val | B_confirm |
| pure_ratio_fields | 917 | 0.995228 | 0.991628 | 0.979127 | 0.999954 | 0.004970 | 0.174887 | 0.174669 | 1 | A_val | B_confirm |

### Cross-split formal-metric comparison

| metric | metric_label | A_val_observed | B_confirm_observed | absolute_observed_difference | A_val_null_mean | B_confirm_null_mean | absolute_null_mean_difference | A_val_directional_difference | B_confirm_directional_difference | A_val_monte_carlo_p | B_confirm_monte_carlo_p | A_val_BH_q | B_confirm_BH_q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupancy_weighted_full_field_distance_from_null_mean | Occupancy-weighted full-field distance from the matched-null mean | 0.090968 | 0.090738 | 0.000230 | 0.009336 | 0.009161 | 0.000175 | 0.081632 | 0.081578 | 0.009901 | 0.009901 | NA | NA |
| negative_divergence_occupancy_fraction | Interior occupancy fraction with negative local divergence | 0.776241 | 0.810259 | 0.034018 | 0.737841 | 0.738205 | 0.000364 | 0.038400 | 0.072054 | 0.009901 | 0.009901 | 0.029703 | 0.014851 |
| flow_weighted_shell_fraction_inward | Flow-weighted inward fraction in the frozen shell | 0.669335 | 0.676868 | 0.007532 | 0.662474 | 0.664248 | 0.001774 | 0.006861 | 0.012620 | 0.039604 | 0.009901 | 0.059406 | 0.014851 |
| flow_core_to_shell_speed_ratio | Flow-weighted core-to-shell drift-speed ratio | 0.522598 | 0.504716 | 0.017882 | 0.492139 | 0.491946 | 0.000192 | -0.030460 | -0.012770 | 1 | 0.970297 | 1 | 0.970297 |

## Metric ledger

| category | split | metric | value | formatted_value | source | estimator | weighting | support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis contract | A_val | analysis panel rows | 3,447,590 | 3,447,590 | A_val construction-null outputs | frozen output audit | not applicable | formal A_val inference |
| analysis contract | A_val | analysis panel users | 59,492 | 59,492 | A_val construction-null outputs | frozen output audit | not applicable | formal A_val inference |
| analysis contract | A_val | valid drift rows | 3,328,409 | 3,328,409 | A_val construction-null outputs | frozen output audit | not applicable | formal A_val inference |
| analysis contract | A_val | null replicates | 100 | 100 | A_val construction-null outputs | frozen output audit | not applicable | formal A_val inference |
| analysis contract | A_val | minimum attainable Monte Carlo p | 0.009901 | 0.009901 | A_val construction-null outputs | frozen output audit | not applicable | formal A_val inference |
| analysis contract | A_val | frozen shell radius | 0.350000 | 0.350000 | A_val construction-null outputs | frozen output audit | not applicable | formal A_val inference |
| formal construction-null test | A_val | Occupancy-weighted full-field distance from the matched-null mean: observed | 0.090968 | 0.090968 | A_val construction-null outputs | occupancy-weighted root mean squared vector distance between the observed field and the mean of the matched permutation fields | user-balanced field / prespecified frozen geometry | 940 supported grid cells; formal A_val inference |
| formal construction-null test | A_val | Occupancy-weighted full-field distance from the matched-null mean: matched-null mean | 0.009336 | 0.009336 | A_val construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | A_val | Occupancy-weighted full-field distance from the matched-null mean: null 2.5% | 0.008908 | 0.008908 | A_val construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Occupancy-weighted full-field distance from the matched-null mean: null 97.5% | 0.009905 | 0.009905 | A_val construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Occupancy-weighted full-field distance from the matched-null mean: Monte Carlo p | 0.009901 | 0.009901 | A_val construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| effect description | A_val | Occupancy-weighted full-field distance from the matched-null mean: pure ratio-contraction baseline | 0.178778 | 0.178778 | A_val construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | formal A_val inference |
| effect description | A_val | Occupancy-weighted full-field distance from the matched-null mean: descriptive standardized separation | 324.6823 | 324.6823 | A_val construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | formal A_val inference |
| effect description | A_val | Occupancy-weighted full-field distance from the matched-null mean: excess-field diagnostic | 0.090968 | 0.090968 | A_val construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | formal A_val inference |
| formal construction-null test | A_val | Interior occupancy fraction with negative local divergence: observed | 0.776241 | 0.776241 | A_val construction-null outputs | user-balanced occupancy fraction over complete supported five-cell stencils with negative divergence | user-balanced field / prespecified frozen geometry | 940 supported grid cells; formal A_val inference |
| formal construction-null test | A_val | Interior occupancy fraction with negative local divergence: matched-null mean | 0.737841 | 0.737841 | A_val construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | A_val | Interior occupancy fraction with negative local divergence: null 2.5% | 0.718350 | 0.718350 | A_val construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Interior occupancy fraction with negative local divergence: null 97.5% | 0.759556 | 0.759556 | A_val construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Interior occupancy fraction with negative local divergence: Monte Carlo p | 0.009901 | 0.009901 | A_val construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Interior occupancy fraction with negative local divergence: BH q across three basin metrics | 0.029703 | 0.029703 | A_val construction-null outputs | Benjamini–Hochberg adjustment across the three prespecified basin metrics | three one-sided Monte Carlo tests | formal A_val inference |
| effect description | A_val | Interior occupancy fraction with negative local divergence: pure ratio-contraction baseline | 0.808573 | 0.808573 | A_val construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | formal A_val inference |
| effect description | A_val | Interior occupancy fraction with negative local divergence: descriptive standardized separation | 3.365850 | 3.365850 | A_val construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | formal A_val inference |
| effect description | A_val | Interior occupancy fraction with negative local divergence: excess-field diagnostic | 0.699861 | 0.699861 | A_val construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | formal A_val inference |
| formal construction-null test | A_val | Flow-weighted inward fraction in the frozen shell: observed | 0.669335 | 0.669335 | A_val construction-null outputs | flow-magnitude-weighted fraction of frozen-shell cells directed toward the A_train core | user-balanced field / prespecified frozen geometry | 940 supported grid cells; formal A_val inference |
| formal construction-null test | A_val | Flow-weighted inward fraction in the frozen shell: matched-null mean | 0.662474 | 0.662474 | A_val construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted inward fraction in the frozen shell: null 2.5% | 0.654120 | 0.654120 | A_val construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted inward fraction in the frozen shell: null 97.5% | 0.669531 | 0.669531 | A_val construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted inward fraction in the frozen shell: Monte Carlo p | 0.039604 | 0.039604 | A_val construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted inward fraction in the frozen shell: BH q across three basin metrics | 0.059406 | 0.059406 | A_val construction-null outputs | Benjamini–Hochberg adjustment across the three prespecified basin metrics | three one-sided Monte Carlo tests | formal A_val inference |
| effect description | A_val | Flow-weighted inward fraction in the frozen shell: pure ratio-contraction baseline | 0.567136 | 0.567136 | A_val construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | formal A_val inference |
| effect description | A_val | Flow-weighted inward fraction in the frozen shell: descriptive standardized separation | 1.693909 | 1.693909 | A_val construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | formal A_val inference |
| effect description | A_val | Flow-weighted inward fraction in the frozen shell: excess-field diagnostic | 0.777743 | 0.777743 | A_val construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | formal A_val inference |
| formal construction-null test | A_val | Flow-weighted core-to-shell drift-speed ratio: observed | 0.522598 | 0.522598 | A_val construction-null outputs | flow-weighted mean drift speed in the frozen core divided by that in its frozen shell | user-balanced field / prespecified frozen geometry | 940 supported grid cells; formal A_val inference |
| formal construction-null test | A_val | Flow-weighted core-to-shell drift-speed ratio: matched-null mean | 0.492139 | 0.492139 | A_val construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted core-to-shell drift-speed ratio: null 2.5% | 0.477744 | 0.477744 | A_val construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted core-to-shell drift-speed ratio: null 97.5% | 0.505044 | 0.505044 | A_val construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted core-to-shell drift-speed ratio: Monte Carlo p | 1 | 1 | A_val construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| formal construction-null test | A_val | Flow-weighted core-to-shell drift-speed ratio: BH q across three basin metrics | 1 | 1 | A_val construction-null outputs | Benjamini–Hochberg adjustment across the three prespecified basin metrics | three one-sided Monte Carlo tests | formal A_val inference |
| effect description | A_val | Flow-weighted core-to-shell drift-speed ratio: pure ratio-contraction baseline | 0.652353 | 0.652353 | A_val construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | formal A_val inference |
| effect description | A_val | Flow-weighted core-to-shell drift-speed ratio: descriptive standardized separation | -4.586620 | -4.586620 | A_val construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | formal A_val inference |
| effect description | A_val | Flow-weighted core-to-shell drift-speed ratio: excess-field diagnostic | 0.736233 | 0.736233 | A_val construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | formal A_val inference |
| matching and opportunity audit | A_val | randomizable fraction of analysis rows | 1 | 1 | A_val construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | formal A_val inference |
| matching and opportunity audit | A_val | within-user matched fraction of randomizable rows | 0.995183 | 0.995183 | A_val construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | formal A_val inference |
| matching and opportunity audit | A_val | across-user matched fraction of randomizable rows | 0.004814 | 0.004814 | A_val construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | formal A_val inference |
| matching and opportunity audit | A_val | weak fallback fraction of randomizable rows | 9.0133e-07 | 9.0133e-07 | A_val construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | formal A_val inference |
| matching and opportunity audit | A_val | support-present fraction | 0.874112 | 0.874112 | A_val construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | formal A_val inference |
| matching and opportunity audit | A_val | idle-present fraction | 0.999978 | 0.999978 | A_val construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | formal A_val inference |
| cellwise excess diagnostics | A_val | supported field cells | 940 | 940 | A_val construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | formal A_val inference |
| cellwise excess diagnostics | A_val | supported occupancy mass | 0.996276 | 0.996276 | A_val construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | formal A_val inference |
| cellwise excess diagnostics | A_val | occupancy-weighted excess speed | 0.067896 | 0.067896 | A_val construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | formal A_val inference |
| cellwise excess diagnostics | A_val | occupancy fraction above cellwise null-radius 97.5% | 0.942181 | 0.942181 | A_val construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | formal A_val inference |
| numerical identity audit | A_val | maximum next-M reconstruction error | 1.2212e-15 | 1.2212e-15 | A_val construction-null outputs | direct numerical audit | not applicable | formal A_val inference |
| numerical identity audit | A_val | maximum next-Psi reconstruction error | 2.7645e-14 | 2.7645e-14 | A_val construction-null outputs | direct numerical audit | not applicable | formal A_val inference |
| numerical identity audit | A_val | maximum archived-field M-drift difference | 0 | 0 | A_val construction-null outputs | direct numerical audit | not applicable | formal A_val inference |
| numerical identity audit | A_val | maximum archived-field Psi-drift difference | 0 | 0 | A_val construction-null outputs | direct numerical audit | not applicable | formal A_val inference |
| numerical identity audit | A_val | maximum archived occupancy difference | 0 | 0 | A_val construction-null outputs | direct numerical audit | not applicable | formal A_val inference |
| numerical identity audit | A_val | fixed points among randomized rows | 0 | 0 | A_val construction-null outputs | direct numerical audit | not applicable | formal A_val inference |
| analysis contract | B_confirm | analysis panel rows | 3,352,259 | 3,352,259 | B_confirm construction-null outputs | frozen output audit | not applicable | output-only confirmation replication |
| analysis contract | B_confirm | analysis panel users | 59,496 | 59,496 | B_confirm construction-null outputs | frozen output audit | not applicable | output-only confirmation replication |
| analysis contract | B_confirm | valid drift rows | 3,233,208 | 3,233,208 | B_confirm construction-null outputs | frozen output audit | not applicable | output-only confirmation replication |
| analysis contract | B_confirm | null replicates | 100 | 100 | B_confirm construction-null outputs | frozen output audit | not applicable | output-only confirmation replication |
| analysis contract | B_confirm | minimum attainable Monte Carlo p | 0.009901 | 0.009901 | B_confirm construction-null outputs | frozen output audit | not applicable | output-only confirmation replication |
| analysis contract | B_confirm | frozen shell radius | 0.350000 | 0.350000 | B_confirm construction-null outputs | frozen output audit | not applicable | output-only confirmation replication |
| formal construction-null test | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: observed | 0.090738 | 0.090738 | B_confirm construction-null outputs | occupancy-weighted root mean squared vector distance between the observed field and the mean of the matched permutation fields | user-balanced field / prespecified frozen geometry | 954 supported grid cells; output-only confirmation replication |
| formal construction-null test | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: matched-null mean | 0.009161 | 0.009161 | B_confirm construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: null 2.5% | 0.008775 | 0.008775 | B_confirm construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: null 97.5% | 0.009561 | 0.009561 | B_confirm construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: Monte Carlo p | 0.009901 | 0.009901 | B_confirm construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| effect description | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: pure ratio-contraction baseline | 0.178476 | 0.178476 | B_confirm construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | output-only confirmation replication |
| effect description | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: descriptive standardized separation | 358.6195 | 358.6195 | B_confirm construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | output-only confirmation replication |
| effect description | B_confirm | Occupancy-weighted full-field distance from the matched-null mean: excess-field diagnostic | 0.090738 | 0.090738 | B_confirm construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | output-only confirmation replication |
| formal construction-null test | B_confirm | Interior occupancy fraction with negative local divergence: observed | 0.810259 | 0.810259 | B_confirm construction-null outputs | user-balanced occupancy fraction over complete supported five-cell stencils with negative divergence | user-balanced field / prespecified frozen geometry | 954 supported grid cells; output-only confirmation replication |
| formal construction-null test | B_confirm | Interior occupancy fraction with negative local divergence: matched-null mean | 0.738205 | 0.738205 | B_confirm construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | B_confirm | Interior occupancy fraction with negative local divergence: null 2.5% | 0.715721 | 0.715721 | B_confirm construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Interior occupancy fraction with negative local divergence: null 97.5% | 0.756671 | 0.756671 | B_confirm construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Interior occupancy fraction with negative local divergence: Monte Carlo p | 0.009901 | 0.009901 | B_confirm construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Interior occupancy fraction with negative local divergence: BH q across three basin metrics | 0.014851 | 0.014851 | B_confirm construction-null outputs | Benjamini–Hochberg adjustment across the three prespecified basin metrics | three one-sided Monte Carlo tests | output-only confirmation replication |
| effect description | B_confirm | Interior occupancy fraction with negative local divergence: pure ratio-contraction baseline | 0.799854 | 0.799854 | B_confirm construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | output-only confirmation replication |
| effect description | B_confirm | Interior occupancy fraction with negative local divergence: descriptive standardized separation | 6.468299 | 6.468299 | B_confirm construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | output-only confirmation replication |
| effect description | B_confirm | Interior occupancy fraction with negative local divergence: excess-field diagnostic | 0.695037 | 0.695037 | B_confirm construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | output-only confirmation replication |
| formal construction-null test | B_confirm | Flow-weighted inward fraction in the frozen shell: observed | 0.676868 | 0.676868 | B_confirm construction-null outputs | flow-magnitude-weighted fraction of frozen-shell cells directed toward the A_train core | user-balanced field / prespecified frozen geometry | 954 supported grid cells; output-only confirmation replication |
| formal construction-null test | B_confirm | Flow-weighted inward fraction in the frozen shell: matched-null mean | 0.664248 | 0.664248 | B_confirm construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted inward fraction in the frozen shell: null 2.5% | 0.656695 | 0.656695 | B_confirm construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted inward fraction in the frozen shell: null 97.5% | 0.671418 | 0.671418 | B_confirm construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted inward fraction in the frozen shell: Monte Carlo p | 0.009901 | 0.009901 | B_confirm construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted inward fraction in the frozen shell: BH q across three basin metrics | 0.014851 | 0.014851 | B_confirm construction-null outputs | Benjamini–Hochberg adjustment across the three prespecified basin metrics | three one-sided Monte Carlo tests | output-only confirmation replication |
| effect description | B_confirm | Flow-weighted inward fraction in the frozen shell: pure ratio-contraction baseline | 0.564699 | 0.564699 | B_confirm construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | output-only confirmation replication |
| effect description | B_confirm | Flow-weighted inward fraction in the frozen shell: descriptive standardized separation | 3.412460 | 3.412460 | B_confirm construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | output-only confirmation replication |
| effect description | B_confirm | Flow-weighted inward fraction in the frozen shell: excess-field diagnostic | 0.775664 | 0.775664 | B_confirm construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | output-only confirmation replication |
| formal construction-null test | B_confirm | Flow-weighted core-to-shell drift-speed ratio: observed | 0.504716 | 0.504716 | B_confirm construction-null outputs | flow-weighted mean drift speed in the frozen core divided by that in its frozen shell | user-balanced field / prespecified frozen geometry | 954 supported grid cells; output-only confirmation replication |
| formal construction-null test | B_confirm | Flow-weighted core-to-shell drift-speed ratio: matched-null mean | 0.491946 | 0.491946 | B_confirm construction-null outputs | mean over matched joint signed-innovation permutations | same anchors, denominator increments and user-balanced weights as observed | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted core-to-shell drift-speed ratio: null 2.5% | 0.480728 | 0.480728 | B_confirm construction-null outputs | empirical 2.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted core-to-shell drift-speed ratio: null 97.5% | 0.504851 | 0.504851 | B_confirm construction-null outputs | empirical 97.5th percentile of the null ensemble | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted core-to-shell drift-speed ratio: Monte Carlo p | 0.970297 | 0.970297 | B_confirm construction-null outputs | one-sided exact Monte Carlo rank with +1 correction | same matched-null contract | 100 null replicates |
| formal construction-null test | B_confirm | Flow-weighted core-to-shell drift-speed ratio: BH q across three basin metrics | 0.970297 | 0.970297 | B_confirm construction-null outputs | Benjamini–Hochberg adjustment across the three prespecified basin metrics | three one-sided Monte Carlo tests | output-only confirmation replication |
| effect description | B_confirm | Flow-weighted core-to-shell drift-speed ratio: pure ratio-contraction baseline | 0.649190 | 0.649190 | B_confirm construction-null outputs | denominator-growth-only zero-signed-innovation baseline | same observed anchors and denominator increments | output-only confirmation replication |
| effect description | B_confirm | Flow-weighted core-to-shell drift-speed ratio: descriptive standardized separation | -2.001862 | -2.001862 | B_confirm construction-null outputs | supportive-direction observed-minus-null-mean divided by null SD | matched-null distribution | output-only confirmation replication |
| effect description | B_confirm | Flow-weighted core-to-shell drift-speed ratio: excess-field diagnostic | 0.731598 | 0.731598 | B_confirm construction-null outputs | field geometry recomputed on observed drift minus matched-null mean drift | user-balanced field with frozen core and support | output-only confirmation replication |
| matching and opportunity audit | B_confirm | randomizable fraction of analysis rows | 1 | 1 | B_confirm construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | output-only confirmation replication |
| matching and opportunity audit | B_confirm | within-user matched fraction of randomizable rows | 0.995002 | 0.995002 | B_confirm construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | output-only confirmation replication |
| matching and opportunity audit | B_confirm | across-user matched fraction of randomizable rows | 0.004995 | 0.004995 | B_confirm construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | output-only confirmation replication |
| matching and opportunity audit | B_confirm | weak fallback fraction of randomizable rows | 0 | 0 | B_confirm construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | output-only confirmation replication |
| matching and opportunity audit | B_confirm | support-present fraction | 0.869599 | 0.869599 | B_confirm construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | output-only confirmation replication |
| matching and opportunity audit | B_confirm | idle-present fraction | 0.999976 | 0.999976 | B_confirm construction-null outputs | hierarchical disjoint permutation coverage or opportunity composition | row count or row fraction | output-only confirmation replication |
| cellwise excess diagnostics | B_confirm | supported field cells | 954 | 954 | B_confirm construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | output-only confirmation replication |
| cellwise excess diagnostics | B_confirm | supported occupancy mass | 0.996143 | 0.996143 | B_confirm construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | output-only confirmation replication |
| cellwise excess diagnostics | B_confirm | occupancy-weighted excess speed | 0.067811 | 0.067811 | B_confirm construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | output-only confirmation replication |
| cellwise excess diagnostics | B_confirm | occupancy fraction above cellwise null-radius 97.5% | 0.942912 | 0.942912 | B_confirm construction-null outputs | supported-cell descriptive field diagnostic | user-balanced occupancy | output-only confirmation replication |
| numerical identity audit | B_confirm | maximum next-M reconstruction error | 9.9920e-16 | 9.9920e-16 | B_confirm construction-null outputs | direct numerical audit | not applicable | output-only confirmation replication |
| numerical identity audit | B_confirm | maximum next-Psi reconstruction error | 2.1427e-14 | 2.1427e-14 | B_confirm construction-null outputs | direct numerical audit | not applicable | output-only confirmation replication |
| numerical identity audit | B_confirm | maximum archived-field M-drift difference | 0 | 0 | B_confirm construction-null outputs | direct numerical audit | not applicable | output-only confirmation replication |
| numerical identity audit | B_confirm | maximum archived-field Psi-drift difference | 0 | 0 | B_confirm construction-null outputs | direct numerical audit | not applicable | output-only confirmation replication |
| numerical identity audit | B_confirm | maximum archived occupancy difference | 0 | 0 | B_confirm construction-null outputs | direct numerical audit | not applicable | output-only confirmation replication |
| numerical identity audit | B_confirm | fixed points among randomized rows | 0 | 0 | B_confirm construction-null outputs | direct numerical audit | not applicable | output-only confirmation replication |
