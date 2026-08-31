---
source_report: "mechanism_event_ssl_macro_closure_comparison.md"
source_extraction_script: "s4_cross_analysis/scripts/extract_mechanism_event_ssl_publication_statistics.py"
---

# Minimal mechanism and Event-SSL cross-model numerical results

## Join and partition audit

| mechanism_rows | event_ssl_rows | joined_rows | joined_users | join_fraction_of_mechanism | join_fraction_of_event_ssl | minimum_required_join_fraction | anchor_tolerance | max_abs_difference_M_between_sources | mean_abs_difference_M_between_sources | max_abs_difference_Psi_between_sources | mean_abs_difference_Psi_between_sources | max_abs_difference_target_M_next_between_sources | mean_abs_difference_target_M_next_between_sources | max_abs_difference_target_Psi_next_between_sources | mean_abs_difference_target_Psi_next_between_sources | analysis_rows_after_subsample | analysis_users_after_subsample | max_rows_argument | joined_rows_missing_stage1_macrostate | stage1_assignment_match_fraction | current_macrostate_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3233208 | 3233208 | 3233208 | 56195 | 1 | 1 | 0.999999 | 1e-06 | 2.98023e-08 | 4.71594e-09 | 2.98023e-08 | 1.31676e-08 | 2.98023e-08 | 4.83714e-09 | 2.98023e-08 | 1.33733e-08 | 3233208 | 56195 | 0 | 0 | 1 | frozen Stage-1 fixed K=6 assignment table |

| source | coordinate | macrostate_k | macrostate_k_rule | fit_split | features | user_balanced_sampling | user_balanced_kmeans_fit | fit_max_rows | kmeans_n_init | random_state | metadata_sha256 | centers_sha256 | kmeans_refit | macrostate_k_selected | confirmation_data_used_for_partition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen Stage-1 fixed K=6 partition | MR_PsiA | 6 | fixed a priori | A_train | ['M_response_prebalanced_pre', 'activity_alignment_order_Psi_pre'] | True | True | 500000 | 20 | 42 | 13de2191cd5d543b867cfebb7f65c956e4f88a58ba8eeb24caaea79e0cbabc7a | 54b237424538211653f8a798f3eb62a4fb6f55b1526a7ce581541c7f5438bb85 | False | False | False |

## Summary metrics

| signal | value |
| --- | --- |
| next-state M correlation | 0.923789 |
| next-state Psi correlation | 0.977914 |
| displacement-vector correlation | 0.719639 |
| mean interval displacement cosine | 0.120075 |
| mechanism-EventSSL occupancy JS | 0.209556 |
| mechanism-EventSSL field vector correlation | 0.866107 |
| mechanism-EventSSL field speed correlation | 0.855204 |
| mechanism-EventSSL weighted local drift cosine | 0.785083 |
| mechanism-EventSSL transition mean row-TV | 0.149714 |
| mechanism-EventSSL transition max row-TV | 0.288033 |
| mechanism-EventSSL self-transition correlation | 0.965971 |
| mechanistic surrogate composite | 0.859904 |

## Metric ledger

| category | metric | value | estimator | weighting | support |
| --- | --- | --- | --- | --- | --- |
| analysis contract | confirmation split | B_confirm | frozen output-only comparison | not applicable | 3233208 joined intervals; 56195 users |
| analysis contract | join fraction of mechanism rows | 1.0 | one-to-one key join | not applicable | all formal prediction rows |
| analysis contract | join fraction of Event-SSL rows | 1.0 | one-to-one key join | not applicable | all formal prediction rows |
| analysis contract | fixed mesostate K | 6 | frozen Stage-1 partition | A_train user-balanced KMeans fit | K=6; no refit |
| cross-model agreement | next-state M correlation | 0.9237886526901545 | interval-level Pearson correlation | interval | common confirmation rows and frozen K=6 partition |
| cross-model agreement | next-state Psi correlation | 0.9779144756731221 | interval-level Pearson correlation | interval | common confirmation rows and frozen K=6 partition |
| cross-model agreement | displacement-vector correlation | 0.719638790325695 | Pearson correlation of flattened two-coordinate displacements | interval | common confirmation rows and frozen K=6 partition |
| cross-model agreement | mean displacement cosine | 0.12007526960989084 | mean interval-level vector cosine | interval | common confirmation rows and frozen K=6 partition |
| cross-model agreement | next-state occupancy JS | 0.20955587867517234 | Jensen-Shannon divergence | user-balanced occupancy | common confirmation rows and frozen K=6 partition |
| cross-model agreement | population drift vector correlation | 0.866106645150386 | Pearson correlation of flattened supported drift components | user-balanced field | common confirmation rows and frozen K=6 partition |
| cross-model agreement | population drift speed correlation | 0.8552041194718198 | Pearson correlation of cellwise drift speeds | user-balanced field | common confirmation rows and frozen K=6 partition |
| cross-model agreement | occupancy-weighted local drift cosine | 0.785082874714253 | local vector cosine averaged with empirical-anchor field weights | user-balanced field | common confirmation rows and frozen K=6 partition |
| cross-model agreement | common supported drift cells | 954.0 | common support count | minimum 30 intervals per cell | common confirmation rows and frozen K=6 partition |
| cross-model agreement | transition mean row-wise TV | 0.14971388883774264 | mean row-wise total variation | interval transition counts | common confirmation rows and frozen K=6 partition |
| cross-model agreement | transition max row-wise TV | 0.2880332200830502 | maximum row-wise total variation | interval transition counts | common confirmation rows and frozen K=6 partition |
| cross-model agreement | self-transition correlation | 0.9659708838297931 | Pearson correlation across six statewise self-transition probabilities | six fixed states | common confirmation rows and frozen K=6 partition |
| cross-model agreement | mechanistic surrogate composite | 0.8599040693588111 | descriptive mean of eight normalized agreement signals | mixed descriptive scales | not a training or selection target |
| residual-field diagnostics | residual_vector_corr | 0.1957375722777599 | comparison after subtracting the same empirical field | user-balanced field | common supported cells |
| residual-field diagnostics | residual_speed_corr | 0.36821225745265873 | comparison after subtracting the same empirical field | user-balanced field | common supported cells |
| residual-field diagnostics | occupancy_weighted_residual_cosine | -0.22148020786192119 | comparison after subtracting the same empirical field | user-balanced field | common supported cells |
| residual-field diagnostics | residual_field_rmse_between_models | 0.1097786412600068 | comparison after subtracting the same empirical field | user-balanced field | common supported cells |

## Single-model metrics

| model | metric | value | contract |
| --- | --- | --- | --- |
| minimal_mechanism | one_step_rmse_M | 0.11268589906152346 | formal Phase-3 confirmation contract |
| minimal_mechanism | one_step_rmse_Psi | 0.03589755077805571 | formal Phase-3 confirmation contract |
| minimal_mechanism | occupancy_js_MR_PsiA | 0.16525664778274118 | formal Phase-3 confirmation contract |
| minimal_mechanism | drift_vector_corr_MR_PsiA | 0.9457023936005611 | formal Phase-3 confirmation contract |
| minimal_mechanism | drift_local_rmse_loss_MR_PsiA | 0.3231465185920794 | formal Phase-3 confirmation contract |
| minimal_mechanism | drift_direction_loss_MR_PsiA | 0.027148803199719462 | formal Phase-3 confirmation contract |
| minimal_mechanism | drift_magnitude_loss_MR_PsiA | 0.1663908454161036 | formal Phase-3 confirmation contract |
| minimal_mechanism | objective_primary_score | 0.17648453714278337 | formal Phase-3 confirmation contract |
| predictive_state_event_ssl | coordinate_corr_M | 0.883671374679744 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | coordinate_corr_Psi | 0.9904926938031048 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | coordinate_rmse_M | 0.1621923297643661 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | coordinate_rmse_Psi | 0.0296872947365045 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | one_step_rmse_M | 0.1099505946040153 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | one_step_rmse_Psi | 0.035347256809473 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | next_state_occupancy_js | 0.2005163284559438 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | anchor_drift_vector_corr | 0.8802278296481509 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | anchor_occupancy_weighted_local_drift_cosine | 0.8192059015796935 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | learned_plane_drift_vector_corr | 0.6877476377668944 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | learned_plane_occupancy_weighted_local_drift_cosine | 0.8629410393217833 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | learned_plane_transition_mean_row_tv | 0.0941665085170334 | formal Stage-4 evaluation contract |
| predictive_state_event_ssl | learned_plane_self_transition_corr | 0.8462329343587597 | formal Stage-4 evaluation contract |
| minimal_mechanism | family | offset_dual_channel | frozen Phase-2/Phase-3 manifest |
| predictive_state_event_ssl | model_kind | predictive_state | frozen Stage-4 evaluation manifest |

## Residual-field diagnostics

| comparison | common_drift_cells | residual_vector_corr | residual_speed_corr | occupancy_weighted_residual_cosine | residual_field_rmse_between_models | mechanism_residual_magnitude_mean | event_ssl_residual_magnitude_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mechanism_residual_vs_event_ssl_residual | 954 | 0.195738 | 0.368212 | -0.22148 | 0.109779 | 0.0295794 | 0.0689574 |

## Statewise transition diagnostics

| macrostate | empirical_self_transition | mechanism_self_transition | event_ssl_self_transition | mechanism_minus_empirical_self_transition | event_ssl_minus_empirical_self_transition | event_ssl_minus_mechanism_self_transition | mechanism_vs_empirical_row_tv | event_ssl_vs_empirical_row_tv | event_ssl_vs_mechanism_row_tv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.404973 | 0.362023 | 0.316723 | -0.0429493 | -0.0882494 | -0.0453001 | 0.277549 | 0.320841 | 0.288033 |
| 1 | 0.888067 | 0.959172 | 0.80699 | 0.0711052 | -0.0810764 | -0.152182 | 0.0711052 | 0.0841412 | 0.152235 |
| 2 | 0.910944 | 0.966525 | 0.883287 | 0.0555808 | -0.0276565 | -0.0832373 | 0.0555808 | 0.0283516 | 0.0832373 |
| 3 | 0.986391 | 0.994741 | 0.977253 | 0.0083506 | -0.00913806 | -0.0174887 | 0.0083506 | 0.00930113 | 0.0175065 |
| 4 | 0.763408 | 0.81033 | 0.663878 | 0.0469225 | -0.0995296 | -0.146452 | 0.0469225 | 0.103007 | 0.146584 |
| 5 | 0.626727 | 0.55076 | 0.340245 | -0.0759669 | -0.286482 | -0.210515 | 0.153231 | 0.361831 | 0.210688 |

## Interval-level metrics

| comparison | metric | value |
| --- | --- | --- |
| mechanism_vs_empirical | next_M_corr | 0.937983 |
| mechanism_vs_empirical | next_Psi_corr | 0.982223 |
| mechanism_vs_empirical | next_M_rmse | 0.112686 |
| mechanism_vs_empirical | next_Psi_rmse | 0.0358976 |
| mechanism_vs_empirical | next_M_mae | 0.0450376 |
| mechanism_vs_empirical | next_Psi_mae | 0.00979788 |
| mechanism_vs_empirical | displacement_vector_corr | 0.60615 |
| mechanism_vs_empirical | mean_displacement_cosine | -0.0278839 |
| event_ssl_vs_empirical | next_M_corr | 0.941843 |
| event_ssl_vs_empirical | next_Psi_corr | 0.982729 |
| event_ssl_vs_empirical | next_M_rmse | 0.109951 |
| event_ssl_vs_empirical | next_Psi_rmse | 0.0353473 |
| event_ssl_vs_empirical | next_M_mae | 0.052222 |
| event_ssl_vs_empirical | next_Psi_mae | 0.0164238 |
| event_ssl_vs_empirical | displacement_vector_corr | 0.755095 |
| event_ssl_vs_empirical | mean_displacement_cosine | 0.30455 |
| mechanism_vs_event_ssl | next_M_corr | 0.923789 |
| mechanism_vs_event_ssl | next_Psi_corr | 0.977914 |
| mechanism_vs_event_ssl | next_M_rmse | 0.117138 |
| mechanism_vs_event_ssl | next_Psi_rmse | 0.0395445 |
| mechanism_vs_event_ssl | next_M_mae | 0.0596543 |
| mechanism_vs_event_ssl | next_Psi_mae | 0.0199134 |
| mechanism_vs_event_ssl | displacement_vector_corr | 0.719639 |
| mechanism_vs_event_ssl | mean_displacement_cosine | 0.120075 |

## Weighted interval-level metrics

| comparison | metric | value |
| --- | --- | --- |
| mechanism_vs_empirical | interval_weighted_rmse_M | 0.112686 |
| mechanism_vs_empirical | interval_weighted_mae_M | 0.0450376 |
| mechanism_vs_empirical | interval_weighted_bias_M | -0.00232667 |
| mechanism_vs_empirical | user_balanced_rmse_M | 0.299756 |
| mechanism_vs_empirical | user_balanced_mae_M | 0.228006 |
| mechanism_vs_empirical | user_balanced_bias_M | 0.00556367 |
| mechanism_vs_empirical | interval_weighted_rmse_Psi | 0.0358976 |
| mechanism_vs_empirical | interval_weighted_mae_Psi | 0.00979788 |
| mechanism_vs_empirical | interval_weighted_bias_Psi | 0.00223381 |
| mechanism_vs_empirical | user_balanced_rmse_Psi | 0.0939573 |
| mechanism_vs_empirical | user_balanced_mae_Psi | 0.0374692 |
| mechanism_vs_empirical | user_balanced_bias_Psi | 0.00340377 |
| mechanism_vs_empirical | median_displacement_cosine | -0.0105973 |
| mechanism_vs_empirical | displacement_cosine_q25 | -0.768391 |
| mechanism_vs_empirical | displacement_cosine_q75 | 0.682087 |
| mechanism_vs_empirical | fraction_positive_displacement_cosine | 0.486049 |
| mechanism_vs_empirical | mean_displacement_norm_first | 0.0296874 |
| mechanism_vs_empirical | mean_displacement_norm_second | 0.0501669 |
| event_ssl_vs_empirical | interval_weighted_rmse_M | 0.109951 |
| event_ssl_vs_empirical | interval_weighted_mae_M | 0.052222 |
| event_ssl_vs_empirical | interval_weighted_bias_M | 3.15083e-05 |
| event_ssl_vs_empirical | user_balanced_rmse_M | 0.299016 |
| event_ssl_vs_empirical | user_balanced_mae_M | 0.196576 |
| event_ssl_vs_empirical | user_balanced_bias_M | 0.0401608 |
| event_ssl_vs_empirical | interval_weighted_rmse_Psi | 0.0353473 |
| event_ssl_vs_empirical | interval_weighted_mae_Psi | 0.0164238 |
| event_ssl_vs_empirical | interval_weighted_bias_Psi | -0.00152246 |
| event_ssl_vs_empirical | user_balanced_rmse_Psi | 0.0935259 |
| event_ssl_vs_empirical | user_balanced_mae_Psi | 0.0555918 |
| event_ssl_vs_empirical | user_balanced_bias_Psi | 0.0111388 |
| event_ssl_vs_empirical | median_displacement_cosine | 0.745276 |
| event_ssl_vs_empirical | displacement_cosine_q25 | -0.504382 |
| event_ssl_vs_empirical | displacement_cosine_q75 | 0.979067 |
| event_ssl_vs_empirical | fraction_positive_displacement_cosine | 0.655019 |
| event_ssl_vs_empirical | mean_displacement_norm_first | 0.0806957 |
| event_ssl_vs_empirical | mean_displacement_norm_second | 0.0501669 |
| mechanism_vs_event_ssl | interval_weighted_rmse_M | 0.117138 |
| mechanism_vs_event_ssl | interval_weighted_mae_M | 0.0596543 |
| mechanism_vs_event_ssl | interval_weighted_bias_M | -0.00235818 |
| mechanism_vs_event_ssl | user_balanced_rmse_M | 0.305472 |
| mechanism_vs_event_ssl | user_balanced_mae_M | 0.214225 |
| mechanism_vs_event_ssl | user_balanced_bias_M | -0.0345971 |
| mechanism_vs_event_ssl | interval_weighted_rmse_Psi | 0.0395445 |
| mechanism_vs_event_ssl | interval_weighted_mae_Psi | 0.0199134 |
| mechanism_vs_event_ssl | interval_weighted_bias_Psi | 0.00375628 |
| mechanism_vs_event_ssl | user_balanced_rmse_Psi | 0.0927325 |
| mechanism_vs_event_ssl | user_balanced_mae_Psi | 0.057076 |
| mechanism_vs_event_ssl | user_balanced_bias_Psi | -0.00773504 |
| mechanism_vs_event_ssl | median_displacement_cosine | 0.226951 |
| mechanism_vs_event_ssl | displacement_cosine_q25 | -0.570216 |
| mechanism_vs_event_ssl | displacement_cosine_q75 | 0.820817 |
| mechanism_vs_event_ssl | fraction_positive_displacement_cosine | 0.573369 |
| mechanism_vs_event_ssl | mean_displacement_norm_first | 0.0296874 |
| mechanism_vs_event_ssl | mean_displacement_norm_second | 0.0806957 |

## Landscape

| comparison | metric | value |
| --- | --- | --- |
| mechanism_vs_empirical | next_occupancy_js | 0.165257 |
| mechanism_vs_empirical | next_occupancy_overlap | 0.570483 |
| mechanism_vs_empirical | next_occupancy_tv | 0.429517 |
| mechanism_vs_empirical | next_occupancy_corr | 0.208279 |
| event_ssl_vs_empirical | next_occupancy_js | 0.200528 |
| event_ssl_vs_empirical | next_occupancy_overlap | 0.560513 |
| event_ssl_vs_empirical | next_occupancy_tv | 0.439487 |
| event_ssl_vs_empirical | next_occupancy_corr | 0.137902 |
| mechanism_vs_event_ssl | next_occupancy_js | 0.209556 |
| mechanism_vs_event_ssl | next_occupancy_overlap | 0.551759 |
| mechanism_vs_event_ssl | next_occupancy_tv | 0.448241 |
| mechanism_vs_event_ssl | next_occupancy_corr | 0.227897 |

## Field

| comparison | metric | value |
| --- | --- | --- |
| mechanism_vs_empirical | common_drift_cells | 954 |
| mechanism_vs_empirical | drift_vector_corr | 0.945703 |
| mechanism_vs_empirical | drift_speed_corr | 0.942282 |
| mechanism_vs_empirical | occupancy_weighted_local_drift_cosine | 0.954183 |
| mechanism_vs_empirical | field_rmse | 0.0470557 |
| event_ssl_anchor_vs_empirical | common_drift_cells | 954 |
| event_ssl_anchor_vs_empirical | drift_vector_corr | 0.880231 |
| event_ssl_anchor_vs_empirical | drift_speed_corr | 0.864944 |
| event_ssl_anchor_vs_empirical | occupancy_weighted_local_drift_cosine | 0.819201 |
| event_ssl_anchor_vs_empirical | field_rmse | 0.108529 |
| event_ssl_learned_vs_empirical | common_drift_cells | 842 |
| event_ssl_learned_vs_empirical | drift_vector_corr | 0.687744 |
| event_ssl_learned_vs_empirical | drift_speed_corr | 0.492581 |
| event_ssl_learned_vs_empirical | occupancy_weighted_local_drift_cosine | 0.862954 |
| event_ssl_learned_vs_empirical | field_rmse | 0.0757142 |
| mechanism_vs_event_ssl_anchor | common_drift_cells | 954 |
| mechanism_vs_event_ssl_anchor | drift_vector_corr | 0.866107 |
| mechanism_vs_event_ssl_anchor | drift_speed_corr | 0.855204 |
| mechanism_vs_event_ssl_anchor | occupancy_weighted_local_drift_cosine | 0.785083 |
| mechanism_vs_event_ssl_anchor | field_rmse | 0.109779 |

## Transition

| comparison | metric | value |
| --- | --- | --- |
| mechanism_vs_empirical | mean_row_tv | 0.102123 |
| mechanism_vs_empirical | max_row_tv | 0.277549 |
| mechanism_vs_empirical | self_transition_corr | 0.986957 |
| mechanism_vs_empirical | self_transition_rmse | 0.0548213 |
| mechanism_vs_empirical | self_transition_mae | 0.0501459 |
| mechanism_vs_empirical | top_edge_overlap | 0.833333 |
| mechanism_vs_empirical | diagonal_match_fraction | 0.833333 |
| event_ssl_vs_empirical | mean_row_tv | 0.151246 |
| event_ssl_vs_empirical | max_row_tv | 0.361831 |
| event_ssl_vs_empirical | self_transition_corr | 0.952938 |
| event_ssl_vs_empirical | self_transition_rmse | 0.133659 |
| event_ssl_vs_empirical | self_transition_mae | 0.0986887 |
| event_ssl_vs_empirical | top_edge_overlap | 0.666667 |
| event_ssl_vs_empirical | diagonal_match_fraction | 0.666667 |
| event_ssl_vs_mechanism | mean_row_tv | 0.149714 |
| event_ssl_vs_mechanism | max_row_tv | 0.288033 |
| event_ssl_vs_mechanism | self_transition_corr | 0.965971 |
| event_ssl_vs_mechanism | self_transition_rmse | 0.127939 |
| event_ssl_vs_mechanism | self_transition_mae | 0.109196 |
| event_ssl_vs_mechanism | top_edge_overlap | 0.833333 |
| event_ssl_vs_mechanism | diagonal_match_fraction | 0.833333 |

## Decomposition scores

| domain | score | raw_value |
| --- | --- | --- |
| row_next_M_corr | 0.923789 | 0.923789 |
| row_next_Psi_corr | 0.977914 | 0.977914 |
| row_displacement_vector_corr | 0.719639 | 0.719639 |
| landscape_similarity_1minusJS | 0.790444 | 0.790444 |
| field_vector_corr | 0.866107 | 0.866107 |
| field_weighted_cosine | 0.785083 | 0.785083 |
| transition_similarity_1minusTV | 0.850286 | 0.850286 |
| transition_self_corr | 0.965971 | 0.965971 |
| mechanistic_surrogate_composite | 0.859904 | 0.859904 |

## Transition Reconstruction Audit

| joined_empirical_transition_count | stage1_empirical_transition_count | transition_count_max_abs_difference | transition_matrix_max_abs_difference | exact_full_stage1_reconstruction |
| --- | --- | --- | --- | --- |
| 3.23321e+06 | 3.23321e+06 | 0 | 0 | True |
