---
source_report: "stage4_all_experiments_comparison_report.md"
source_extraction_script: "s3_event_ssl/scripts/extract_event_ssl_stage4_publication_statistics.py"
---

# Event-SSL and control-model numerical results

## Metric ledger

| category | model_label | split | metric | value | source |
| --- | --- | --- | --- | --- | --- |
| main model confirmation | predictive_state_event_ssl | B_confirm | Confirmation intervals | nan | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Confirmation users | nan | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Coordinate correlation M | 0.883671 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Coordinate correlation Psi | 0.990493 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Coordinate RMSE M | 0.162192 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Coordinate RMSE Psi | 0.0296873 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | One-step RMSE M | 0.109951 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | One-step RMSE Psi | 0.0353473 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Next-state occupancy JS | 0.200516 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Empirical-anchor drift correlation | 0.880228 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Learned-plane drift correlation | 0.687748 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Learned-plane local drift cosine | 0.862941 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Learned-plane transition mean row TV | 0.0941665 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Learned-plane self-transition correlation | 0.846233 | predictive-state B_confirm structural metrics |
| main model confirmation | predictive_state_event_ssl | B_confirm | Learned-plane inward-flow fraction | 0.305878 | predictive-state B_confirm structural metrics |
| temporal control | time_shuffle_control | B_confirm | Time-shuffle learned-plane drift correlation | -0.419145 | time-shuffle B_confirm metrics |
| temporal control | time_shuffle_control | B_confirm | Time-shuffle learned-plane local cosine | 0.144615 | time-shuffle B_confirm metrics |
| semantic control | tag_support_randomized | B_confirm | Tag/support-randomized inward-flow fraction | 0.101163 | tag/support B_confirm metrics |
| semantic control | tag_support_randomized | B_confirm | Relative inward-flow change | -0.669269 | derived from main and tag/support B_confirm metrics |
| objective controls | pure_event_ssl_probe | B_confirm | coordinate_corr_M | 0.798753 | pure_event_ssl_probe B_confirm metrics |
| objective controls | pure_event_ssl_probe | B_confirm | coordinate_corr_Psi | 0.914775 | pure_event_ssl_probe B_confirm metrics |
| objective controls | pure_event_ssl_probe | B_confirm | learned_plane_drift_vector_corr | 0.520447 | pure_event_ssl_probe B_confirm metrics |
| objective controls | pure_event_ssl_probe | B_confirm | learned_plane_transition_mean_row_tv | 0.175608 | pure_event_ssl_probe B_confirm metrics |
| objective controls | pure_event_ssl_probe | B_confirm | learned_plane_self_transition_corr | 0.330612 | pure_event_ssl_probe B_confirm metrics |
| objective controls | task_only | B_confirm | coordinate_corr_M | 0.816093 | task_only B_confirm metrics |
| objective controls | task_only | B_confirm | coordinate_corr_Psi | 0.870265 | task_only B_confirm metrics |
| objective controls | task_only | B_confirm | learned_plane_drift_vector_corr | 0.562233 | task_only B_confirm metrics |
| objective controls | task_only | B_confirm | learned_plane_transition_mean_row_tv | 0.141673 | task_only B_confirm metrics |
| objective controls | task_only | B_confirm | learned_plane_self_transition_corr | 0.544 | task_only B_confirm metrics |
| task-only performance | task_only | B_confirm | task_auc_binary_rows | 0.651066 | task-only B_confirm metrics |
| task-only performance | task_only | B_confirm | task_bce | 0.622634 | task-only B_confirm metrics |
| validation-confirmation stability | predictive_state_event_ssl | A_val_to_B_confirm | absolute gap coordinate_corr_M | 0.000290579 | Stage-4 stability table |
| validation-confirmation stability | predictive_state_event_ssl | A_val_to_B_confirm | absolute gap coordinate_corr_Psi | 0.000923222 | Stage-4 stability table |
| validation-confirmation stability | predictive_state_event_ssl | A_val_to_B_confirm | absolute gap next_state_occupancy_js | 0.000633962 | Stage-4 stability table |
| validation-confirmation stability | predictive_state_event_ssl | A_val_to_B_confirm | absolute gap learned_plane_drift_vector_corr | 0.0285756 | Stage-4 stability table |
| validation-confirmation stability | predictive_state_event_ssl | A_val_to_B_confirm | absolute gap learned_plane_transition_mean_row_tv | 0.00302852 | Stage-4 stability table |

## Key numerical summary: A_val

| model_label | coordinate_corr_M | coordinate_corr_Psi | coordinate_rmse_M | coordinate_rmse_Psi | one_step_rmse_M | one_step_rmse_Psi | current_state_occupancy_js | next_state_occupancy_js | anchor_drift_vector_corr | anchor_occupancy_weighted_local_drift_cosine | learned_plane_drift_vector_corr | learned_plane_occupancy_weighted_local_drift_cosine | anchor_inward_fraction_to_reference | learned_plane_inward_fraction_to_reference | anchor_negative_divergence_weighted_fraction | learned_plane_negative_divergence_weighted_fraction | learned_plane_transition_mean_row_tv | learned_plane_self_transition_corr | learned_plane_diagonal_dominance_match_fraction | learned_plane_top_transition_edge_overlap | task_auc_binary_rows | task_bce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictive_state_event_ssl | 0.883381 | 0.989569 | 0.159713 | 0.0305708 | 0.108253 | 0.0362712 | 0.292299 | 0.199882 | 0.877716 | 0.819147 | 0.716323 | 0.875541 | 0.51241 | 0.314202 | 0.848568 | 0.638544 | 0.097195 | 0.896012 | 1 | 1 | nan | nan |
| pure_event_ssl_probe | 0.8038 | 0.913984 | 0.202783 | 0.0820613 | 0.159456 | 0.0841134 | 0.291177 | 0.230158 | 0.820103 | 0.720887 | 0.546843 | 0.836476 | 0.478134 | 0.385342 | 0.908924 | 0.975684 | 0.184228 | 0.326778 | 1 | 1 | nan | nan |
| task_only | 0.814755 | 0.870369 | 0.196896 | 0.0995652 | 0.149716 | 0.0983488 | 0.269403 | 0.180714 | 0.877194 | 0.728963 | 0.545465 | 0.850141 | 0.545231 | 0.217213 | 0.914103 | 0.675535 | 0.133681 | 0.703075 | 1 | 1 | 0.652738 | 0.620029 |
| time_shuffle_control | 0.794357 | 0.944921 | 0.207119 | 0.0667432 | 0.174967 | 0.0732243 | 0.267119 | 0.201156 | 0.813612 | 0.68337 | -0.439093 | 0.128823 | 0.591263 | 0.1835 | 0.812909 | 0.354391 | 0.196572 | 0.39449 | 1 | 1 | nan | nan |
| tag_support_randomized | 0.881102 | 0.985516 | 0.161452 | 0.03564 | 0.111439 | 0.0413229 | 0.291298 | 0.187936 | 0.882164 | 0.723672 | 0.665256 | 0.838599 | 0.59719 | 0.0889111 | 0.846957 | 0.76241 | 0.084731 | 0.903558 | 1 | 1 | nan | nan |

### Domain scores: A_val

| model_label | anchor_drift | convergence | coordinate | landscape | learned_plane_drift | macrostructure_composite | task | transition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictive_state_event_ssl | 0.864451 | 0.578431 | 0.926392 | 0.64485 | 0.762209 | 0.810507 | nan | 0.949704 |
| pure_event_ssl_probe | 0.760183 | 0.687021 | 0.868338 | 0.59951 | 0.711014 | 0.756879 | nan | 0.785638 |
| tag_support_randomized | 0.840769 | 0.573867 | 0.923299 | 0.656349 | 0.719454 | 0.801718 | nan | 0.954707 |
| task_only | 0.817575 | 0.58802 | 0.857517 | 0.660469 | 0.652765 | 0.763876 | 0.618253 | 0.892348 |
| time_shuffle_control | 0.759294 | 0.485516 | 0.876282 | 0.658191 | 0.350053 | 0.687808 | nan | 0.79948 |

## Key numerical summary: B_confirm

| model_label | coordinate_corr_M | coordinate_corr_Psi | coordinate_rmse_M | coordinate_rmse_Psi | one_step_rmse_M | one_step_rmse_Psi | current_state_occupancy_js | next_state_occupancy_js | anchor_drift_vector_corr | anchor_occupancy_weighted_local_drift_cosine | learned_plane_drift_vector_corr | learned_plane_occupancy_weighted_local_drift_cosine | anchor_inward_fraction_to_reference | learned_plane_inward_fraction_to_reference | anchor_negative_divergence_weighted_fraction | learned_plane_negative_divergence_weighted_fraction | learned_plane_transition_mean_row_tv | learned_plane_self_transition_corr | learned_plane_diagonal_dominance_match_fraction | learned_plane_top_transition_edge_overlap | task_auc_binary_rows | task_bce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictive_state_event_ssl | 0.883671 | 0.990493 | 0.162192 | 0.0296873 | 0.109951 | 0.0353473 | 0.292807 | 0.200516 | 0.880228 | 0.819206 | 0.687748 | 0.862941 | 0.503397 | 0.305878 | 0.842866 | 0.650024 | 0.0941665 | 0.846233 | 1 | 1 | nan | nan |
| pure_event_ssl_probe | 0.798753 | 0.914775 | 0.208069 | 0.08317 | 0.164704 | 0.0852275 | 0.291588 | 0.230471 | 0.824053 | 0.732283 | 0.520447 | 0.799114 | 0.472395 | 0.396183 | 0.891475 | 0.971553 | 0.175608 | 0.330612 | 1 | 1 | nan | nan |
| task_only | 0.816093 | 0.870265 | 0.199469 | 0.101399 | 0.151617 | 0.100196 | 0.271017 | 0.182285 | 0.882822 | 0.740405 | 0.562233 | 0.870876 | 0.541378 | 0.222275 | 0.916215 | 0.672902 | 0.141673 | 0.544 | 1 | 1 | 0.651066 | 0.622634 |
| time_shuffle_control | 0.795198 | 0.944407 | 0.210071 | 0.0681102 | 0.177418 | 0.0745771 | 0.268448 | 0.202171 | 0.804936 | 0.680097 | -0.419145 | 0.144615 | 0.578401 | 0.182425 | 0.813012 | 0.378529 | 0.187396 | 0.202306 | 1 | 1 | nan | nan |
| tag_support_randomized | 0.880768 | 0.986369 | 0.164175 | 0.0350799 | 0.113476 | 0.0407693 | 0.292208 | 0.189085 | 0.887614 | 0.720953 | 0.6645 | 0.835572 | 0.59385 | 0.101163 | 0.848808 | 0.785826 | 0.082915 | 0.864764 | 1 | 1 | nan | nan |

### Domain scores: B_confirm

| model_label | anchor_drift | convergence | coordinate | landscape | learned_plane_drift | macrostructure_composite | task | transition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predictive_state_event_ssl | 0.86662 | 0.575541 | 0.926444 | 0.644456 | 0.743217 | 0.80602 | nan | 0.938017 |
| pure_event_ssl_probe | 0.758869 | 0.682901 | 0.866128 | 0.599098 | 0.676378 | 0.750935 | nan | 0.788751 |
| tag_support_randomized | 0.843954 | 0.582412 | 0.923056 | 0.655159 | 0.724728 | 0.802625 | nan | 0.945462 |
| task_only | 0.818519 | 0.588193 | 0.856999 | 0.658692 | 0.664415 | 0.759376 | 0.616042 | 0.850582 |
| time_shuffle_control | 0.750307 | 0.488092 | 0.875559 | 0.65671 | 0.355223 | 0.680714 | nan | 0.753728 |

## Validation-confirmation stability

| model_label | metric | direction | validation_value | confirmation_value | confirmation_minus_validation | absolute_gap | degradation_amount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predictive_state_event_ssl | coordinate_corr_M | higher | 0.883381 | 0.883671 | 0.000290579 | 0.000290579 | -0.000290579 |
| predictive_state_event_ssl | coordinate_corr_Psi | higher | 0.989569 | 0.990493 | 0.000923222 | 0.000923222 | -0.000923222 |
| predictive_state_event_ssl | coordinate_rmse_M | lower | 0.159713 | 0.162192 | 0.00247966 | 0.00247966 | 0.00247966 |
| predictive_state_event_ssl | coordinate_rmse_Psi | lower | 0.0305708 | 0.0296873 | -0.000883466 | 0.000883466 | -0.000883466 |
| predictive_state_event_ssl | one_step_rmse_M | lower | 0.108253 | 0.109951 | 0.00169802 | 0.00169802 | 0.00169802 |
| predictive_state_event_ssl | one_step_rmse_Psi | lower | 0.0362712 | 0.0353473 | -0.000923976 | 0.000923976 | -0.000923976 |
| predictive_state_event_ssl | next_state_occupancy_js | lower | 0.199882 | 0.200516 | 0.000633962 | 0.000633962 | 0.000633962 |
| predictive_state_event_ssl | anchor_drift_vector_corr | higher | 0.877716 | 0.880228 | 0.00251154 | 0.00251154 | -0.00251154 |
| predictive_state_event_ssl | learned_plane_drift_vector_corr | higher | 0.716323 | 0.687748 | -0.0285756 | 0.0285756 | 0.0285756 |
| predictive_state_event_ssl | learned_plane_occupancy_weighted_local_drift_cosine | higher | 0.875541 | 0.862941 | -0.0126003 | 0.0126003 | 0.0126003 |
| predictive_state_event_ssl | learned_plane_transition_mean_row_tv | lower | 0.097195 | 0.0941665 | -0.00302852 | 0.00302852 | -0.00302852 |
| predictive_state_event_ssl | learned_plane_self_transition_corr | higher | 0.896012 | 0.846233 | -0.0497793 | 0.0497793 | 0.0497793 |
| predictive_state_event_ssl | learned_plane_inward_fraction_to_reference | higher | 0.314202 | 0.305878 | -0.00832348 | 0.00832348 | 0.00832348 |
| pure_event_ssl_probe | coordinate_corr_M | higher | 0.8038 | 0.798753 | -0.00504723 | 0.00504723 | 0.00504723 |
| pure_event_ssl_probe | coordinate_corr_Psi | higher | 0.913984 | 0.914775 | 0.000791891 | 0.000791891 | -0.000791891 |
| pure_event_ssl_probe | coordinate_rmse_M | lower | 0.202783 | 0.208069 | 0.0052858 | 0.0052858 | 0.0052858 |
| pure_event_ssl_probe | coordinate_rmse_Psi | lower | 0.0820613 | 0.08317 | 0.00110862 | 0.00110862 | 0.00110862 |
| pure_event_ssl_probe | one_step_rmse_M | lower | 0.159456 | 0.164704 | 0.00524853 | 0.00524853 | 0.00524853 |
| pure_event_ssl_probe | one_step_rmse_Psi | lower | 0.0841134 | 0.0852275 | 0.00111414 | 0.00111414 | 0.00111414 |
| pure_event_ssl_probe | next_state_occupancy_js | lower | 0.230158 | 0.230471 | 0.000312463 | 0.000312463 | 0.000312463 |
| pure_event_ssl_probe | anchor_drift_vector_corr | higher | 0.820103 | 0.824053 | 0.00395059 | 0.00395059 | -0.00395059 |
| pure_event_ssl_probe | learned_plane_drift_vector_corr | higher | 0.546843 | 0.520447 | -0.0263961 | 0.0263961 | 0.0263961 |
| pure_event_ssl_probe | learned_plane_occupancy_weighted_local_drift_cosine | higher | 0.836476 | 0.799114 | -0.0373619 | 0.0373619 | 0.0373619 |
| pure_event_ssl_probe | learned_plane_transition_mean_row_tv | lower | 0.184228 | 0.175608 | -0.00861949 | 0.00861949 | -0.00861949 |
| pure_event_ssl_probe | learned_plane_self_transition_corr | higher | 0.326778 | 0.330612 | 0.00383395 | 0.00383395 | -0.00383395 |
| pure_event_ssl_probe | learned_plane_inward_fraction_to_reference | higher | 0.385342 | 0.396183 | 0.0108409 | 0.0108409 | -0.0108409 |
| task_only | coordinate_corr_M | higher | 0.814755 | 0.816093 | 0.00133795 | 0.00133795 | -0.00133795 |
| task_only | coordinate_corr_Psi | higher | 0.870369 | 0.870265 | -0.00010322 | 0.00010322 | 0.00010322 |
| task_only | coordinate_rmse_M | lower | 0.196896 | 0.199469 | 0.002573 | 0.002573 | 0.002573 |
| task_only | coordinate_rmse_Psi | lower | 0.0995652 | 0.101399 | 0.00183342 | 0.00183342 | 0.00183342 |
| task_only | one_step_rmse_M | lower | 0.149716 | 0.151617 | 0.00190082 | 0.00190082 | 0.00190082 |
| task_only | one_step_rmse_Psi | lower | 0.0983488 | 0.100196 | 0.00184735 | 0.00184735 | 0.00184735 |
| task_only | next_state_occupancy_js | lower | 0.180714 | 0.182285 | 0.00157138 | 0.00157138 | 0.00157138 |
| task_only | anchor_drift_vector_corr | higher | 0.877194 | 0.882822 | 0.00562812 | 0.00562812 | -0.00562812 |
| task_only | learned_plane_drift_vector_corr | higher | 0.545465 | 0.562233 | 0.0167683 | 0.0167683 | -0.0167683 |
| task_only | learned_plane_occupancy_weighted_local_drift_cosine | higher | 0.850141 | 0.870876 | 0.0207348 | 0.0207348 | -0.0207348 |
| task_only | learned_plane_transition_mean_row_tv | lower | 0.133681 | 0.141673 | 0.00799204 | 0.00799204 | 0.00799204 |
| task_only | learned_plane_self_transition_corr | higher | 0.703075 | 0.544 | -0.159075 | 0.159075 | 0.159075 |
| task_only | learned_plane_inward_fraction_to_reference | higher | 0.217213 | 0.222275 | 0.00506268 | 0.00506268 | -0.00506268 |
| time_shuffle_control | coordinate_corr_M | higher | 0.794357 | 0.795198 | 0.000841022 | 0.000841022 | -0.000841022 |
| time_shuffle_control | coordinate_corr_Psi | higher | 0.944921 | 0.944407 | -0.000514162 | 0.000514162 | 0.000514162 |
| time_shuffle_control | coordinate_rmse_M | lower | 0.207119 | 0.210071 | 0.00295141 | 0.00295141 | 0.00295141 |
| time_shuffle_control | coordinate_rmse_Psi | lower | 0.0667432 | 0.0681102 | 0.001367 | 0.001367 | 0.001367 |
| time_shuffle_control | one_step_rmse_M | lower | 0.174967 | 0.177418 | 0.00245069 | 0.00245069 | 0.00245069 |
| time_shuffle_control | one_step_rmse_Psi | lower | 0.0732243 | 0.0745771 | 0.00135284 | 0.00135284 | 0.00135284 |
| time_shuffle_control | next_state_occupancy_js | lower | 0.201156 | 0.202171 | 0.00101443 | 0.00101443 | 0.00101443 |
| time_shuffle_control | anchor_drift_vector_corr | higher | 0.813612 | 0.804936 | -0.00867645 | 0.00867645 | 0.00867645 |
| time_shuffle_control | learned_plane_drift_vector_corr | higher | -0.439093 | -0.419145 | 0.0199481 | 0.0199481 | -0.0199481 |
| time_shuffle_control | learned_plane_occupancy_weighted_local_drift_cosine | higher | 0.128823 | 0.144615 | 0.0157918 | 0.0157918 | -0.0157918 |
| time_shuffle_control | learned_plane_transition_mean_row_tv | lower | 0.196572 | 0.187396 | -0.00917623 | 0.00917623 | -0.00917623 |
| time_shuffle_control | learned_plane_self_transition_corr | higher | 0.39449 | 0.202306 | -0.192184 | 0.192184 | 0.192184 |
| time_shuffle_control | learned_plane_inward_fraction_to_reference | higher | 0.1835 | 0.182425 | -0.00107465 | 0.00107465 | 0.00107465 |
| tag_support_randomized | coordinate_corr_M | higher | 0.881102 | 0.880768 | -0.000334396 | 0.000334396 | 0.000334396 |
| tag_support_randomized | coordinate_corr_Psi | higher | 0.985516 | 0.986369 | 0.000852427 | 0.000852427 | -0.000852427 |
| tag_support_randomized | coordinate_rmse_M | lower | 0.161452 | 0.164175 | 0.0027227 | 0.0027227 | 0.0027227 |
| tag_support_randomized | coordinate_rmse_Psi | lower | 0.03564 | 0.0350799 | -0.000560164 | 0.000560164 | -0.000560164 |
| tag_support_randomized | one_step_rmse_M | lower | 0.111439 | 0.113476 | 0.00203685 | 0.00203685 | 0.00203685 |
| tag_support_randomized | one_step_rmse_Psi | lower | 0.0413229 | 0.0407693 | -0.000553533 | 0.000553533 | -0.000553533 |
| tag_support_randomized | next_state_occupancy_js | lower | 0.187936 | 0.189085 | 0.0011495 | 0.0011495 | 0.0011495 |
| tag_support_randomized | anchor_drift_vector_corr | higher | 0.882164 | 0.887614 | 0.00545004 | 0.00545004 | -0.00545004 |
| tag_support_randomized | learned_plane_drift_vector_corr | higher | 0.665256 | 0.6645 | -0.000756336 | 0.000756336 | 0.000756336 |
| tag_support_randomized | learned_plane_occupancy_weighted_local_drift_cosine | higher | 0.838599 | 0.835572 | -0.00302714 | 0.00302714 | 0.00302714 |
| tag_support_randomized | learned_plane_transition_mean_row_tv | lower | 0.084731 | 0.082915 | -0.00181599 | 0.00181599 | -0.00181599 |
| tag_support_randomized | learned_plane_self_transition_corr | higher | 0.903558 | 0.864764 | -0.0387942 | 0.0387942 | 0.0387942 |
| tag_support_randomized | learned_plane_inward_fraction_to_reference | higher | 0.0889111 | 0.101163 | 0.0122522 | 0.0122522 | -0.0122522 |

## Numerical contrasts

| split | contrast | model_a | model_b | metric | model_a_value | model_b_value | direction | model_b_relative_to_model_a | difference_b_minus_a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | coordinate_corr_M | 0.883381 | 0.8038 | higher | 0.909913 | -0.0795809 |
| A_val | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | coordinate_corr_Psi | 0.989569 | 0.913984 | higher | 0.923617 | -0.075586 |
| A_val | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | learned_plane_drift_vector_corr | 0.716323 | 0.546843 | higher | 0.763403 | -0.16948 |
| A_val | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | learned_plane_self_transition_corr | 0.896012 | 0.326778 | higher | 0.364703 | -0.569234 |
| A_val | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | next_state_occupancy_js | 0.199882 | 0.230158 | lower | 0.868456 | 0.0302759 |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | coordinate_corr_M | 0.883381 | 0.814755 | higher | 0.922314 | -0.0686262 |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | coordinate_corr_Psi | 0.989569 | 0.870369 | higher | 0.879543 | -0.119201 |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | learned_plane_drift_vector_corr | 0.716323 | 0.545465 | higher | 0.761479 | -0.170858 |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | learned_plane_transition_mean_row_tv | 0.097195 | 0.133681 | lower | 0.727065 | 0.0364864 |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | learned_plane_self_transition_corr | 0.896012 | 0.703075 | higher | 0.784672 | -0.192937 |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | task_auc_binary_rows | nan | 0.652738 | higher | nan | nan |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | task_auc_thresholded_all_rows | nan | nan | higher | nan | nan |
| A_val | main_vs_task_only | predictive_state_event_ssl | task_only | task_bce | nan | 0.620029 | lower | nan | nan |
| A_val | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | coordinate_corr_M | 0.883381 | 0.794357 | higher | 0.899224 | -0.0890239 |
| A_val | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | coordinate_corr_Psi | 0.989569 | 0.944921 | higher | 0.954881 | -0.0446483 |
| A_val | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | anchor_drift_vector_corr | 0.877716 | 0.813612 | higher | 0.926965 | -0.0641039 |
| A_val | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_drift_vector_corr | 0.716323 | -0.439093 | higher | -0.612982 | -1.15542 |
| A_val | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_transition_mean_row_tv | 0.097195 | 0.196572 | lower | 0.494451 | 0.0993767 |
| A_val | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_self_transition_corr | 0.896012 | 0.39449 | higher | 0.440273 | -0.501522 |
| A_val | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_inward_fraction_to_reference | 0.314202 | 0.1835 | higher | 0.58402 | -0.130702 |
| A_val | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_corr_M | 0.883381 | 0.881102 | higher | 0.997421 | -0.00227834 |
| A_val | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_corr_Psi | 0.989569 | 0.985516 | higher | 0.995904 | -0.00405323 |
| A_val | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_rmse_M | 0.159713 | 0.161452 | lower | 0.989225 | 0.00173962 |
| A_val | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_rmse_Psi | 0.0305708 | 0.03564 | lower | 0.857765 | 0.00506927 |
| A_val | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | learned_plane_drift_vector_corr | 0.716323 | 0.665256 | higher | 0.928709 | -0.0510674 |
| A_val | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | learned_plane_inward_fraction_to_reference | 0.314202 | 0.0889111 | higher | 0.282975 | -0.22529 |
| A_val | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | learned_plane_transition_mean_row_tv | 0.097195 | 0.084731 | lower | 1.1471 | -0.012464 |
| A_val | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | coordinate_corr_M | 0.8038 | 0.814755 | higher | 1.01363 | 0.0109546 |
| A_val | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | coordinate_corr_Psi | 0.913984 | 0.870369 | higher | 0.95228 | -0.0436149 |
| A_val | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | learned_plane_drift_vector_corr | 0.546843 | 0.545465 | higher | 0.99748 | -0.00137827 |
| A_val | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | learned_plane_self_transition_corr | 0.326778 | 0.703075 | higher | 2.15154 | 0.376297 |
| A_val | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | coordinate_corr_M | 0.794357 | 0.881102 | higher | 1.1092 | 0.0867456 |
| A_val | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | coordinate_corr_Psi | 0.944921 | 0.985516 | higher | 1.04296 | 0.0405951 |
| A_val | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | learned_plane_drift_vector_corr | -0.439093 | 0.665256 | higher | 1.51507 | 1.10435 |
| A_val | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | learned_plane_transition_mean_row_tv | 0.196572 | 0.084731 | lower | 2.31995 | -0.111841 |
| A_val | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | learned_plane_inward_fraction_to_reference | 0.1835 | 0.0889111 | higher | 0.48453 | -0.0945887 |
| A_val | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.877716 | 0.716323 | higher | 0.816122 | -0.161393 |
| A_val | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.819147 | 0.875541 | higher | 1.06885 | 0.0563943 |
| A_val | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.153145 | 0.097195 | lower | 1.57565 | -0.0559501 |
| A_val | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.956792 | 0.896012 | higher | 0.936476 | -0.0607796 |
| A_val | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.820103 | 0.546843 | higher | 0.666799 | -0.273259 |
| A_val | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.720887 | 0.836476 | higher | 1.16034 | 0.115589 |
| A_val | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.305981 | 0.184228 | lower | 1.66088 | -0.121753 |
| A_val | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.807467 | 0.326778 | higher | 0.404695 | -0.480689 |
| A_val | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.877194 | 0.545465 | higher | 0.621829 | -0.331729 |
| A_val | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.728963 | 0.850141 | higher | 1.16623 | 0.121179 |
| A_val | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.327809 | 0.133681 | lower | 2.45216 | -0.194127 |
| A_val | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.957717 | 0.703075 | higher | 0.734116 | -0.254641 |
| A_val | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.813612 | -0.439093 | higher | -0.539684 | -1.25271 |
| A_val | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.68337 | 0.128823 | higher | 0.188511 | -0.554547 |
| A_val | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.244512 | 0.196572 | lower | 1.24388 | -0.0479405 |
| A_val | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.948537 | 0.39449 | higher | 0.415894 | -0.554046 |
| A_val | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.882164 | 0.665256 | higher | 0.754118 | -0.216908 |
| A_val | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.723672 | 0.838599 | higher | 1.15881 | 0.114927 |
| A_val | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.168407 | 0.084731 | lower | 1.98754 | -0.0836757 |
| A_val | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.945564 | 0.903558 | higher | 0.955576 | -0.0420057 |
| A_val | M_vs_Psi_coordinate_asymmetry | predictive_state_event_ssl | predictive_state_event_ssl | coordinate_corr_M - coordinate_corr_Psi | 0.883381 | 0.989569 | diagnostic | 1.12021 | 0.106189 |
| A_val | M_vs_Psi_coordinate_asymmetry | pure_event_ssl_probe | pure_event_ssl_probe | coordinate_corr_M - coordinate_corr_Psi | 0.8038 | 0.913984 | diagnostic | 1.13708 | 0.110184 |
| A_val | M_vs_Psi_coordinate_asymmetry | task_only | task_only | coordinate_corr_M - coordinate_corr_Psi | 0.814755 | 0.870369 | diagnostic | 1.06826 | 0.055614 |
| A_val | M_vs_Psi_coordinate_asymmetry | time_shuffle_control | time_shuffle_control | coordinate_corr_M - coordinate_corr_Psi | 0.794357 | 0.944921 | diagnostic | 1.18954 | 0.150564 |
| A_val | M_vs_Psi_coordinate_asymmetry | tag_support_randomized | tag_support_randomized | coordinate_corr_M - coordinate_corr_Psi | 0.881102 | 0.985516 | diagnostic | 1.1185 | 0.104414 |
| B_confirm | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | coordinate_corr_M | 0.883671 | 0.798753 | higher | 0.903902 | -0.0849187 |
| B_confirm | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | coordinate_corr_Psi | 0.990493 | 0.914775 | higher | 0.923556 | -0.0757173 |
| B_confirm | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | learned_plane_drift_vector_corr | 0.687748 | 0.520447 | higher | 0.756742 | -0.1673 |
| B_confirm | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | learned_plane_self_transition_corr | 0.846233 | 0.330612 | higher | 0.390687 | -0.515621 |
| B_confirm | main_vs_pure_ssl | predictive_state_event_ssl | pure_event_ssl_probe | next_state_occupancy_js | 0.200516 | 0.230471 | lower | 0.87003 | 0.0299544 |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | coordinate_corr_M | 0.883671 | 0.816093 | higher | 0.923525 | -0.0675789 |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | coordinate_corr_Psi | 0.990493 | 0.870265 | higher | 0.878619 | -0.120227 |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | learned_plane_drift_vector_corr | 0.687748 | 0.562233 | higher | 0.8175 | -0.125514 |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | learned_plane_transition_mean_row_tv | 0.0941665 | 0.141673 | lower | 0.664673 | 0.047507 |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | learned_plane_self_transition_corr | 0.846233 | 0.544 | higher | 0.642849 | -0.302233 |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | task_auc_binary_rows | nan | 0.651066 | higher | nan | nan |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | task_auc_thresholded_all_rows | nan | nan | higher | nan | nan |
| B_confirm | main_vs_task_only | predictive_state_event_ssl | task_only | task_bce | nan | 0.622634 | lower | nan | nan |
| B_confirm | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | coordinate_corr_M | 0.883671 | 0.795198 | higher | 0.89988 | -0.0884735 |
| B_confirm | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | coordinate_corr_Psi | 0.990493 | 0.944407 | higher | 0.953472 | -0.0460857 |
| B_confirm | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | anchor_drift_vector_corr | 0.880228 | 0.804936 | higher | 0.914463 | -0.0752919 |
| B_confirm | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_drift_vector_corr | 0.687748 | -0.419145 | higher | -0.609446 | -1.10689 |
| B_confirm | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_transition_mean_row_tv | 0.0941665 | 0.187396 | lower | 0.502501 | 0.093229 |
| B_confirm | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_self_transition_corr | 0.846233 | 0.202306 | higher | 0.239067 | -0.643927 |
| B_confirm | main_vs_time_shuffle | predictive_state_event_ssl | time_shuffle_control | learned_plane_inward_fraction_to_reference | 0.305878 | 0.182425 | higher | 0.596398 | -0.123453 |
| B_confirm | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_corr_M | 0.883671 | 0.880768 | higher | 0.996714 | -0.00290331 |
| B_confirm | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_corr_Psi | 0.990493 | 0.986369 | higher | 0.995836 | -0.00412403 |
| B_confirm | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_rmse_M | 0.162192 | 0.164175 | lower | 0.987924 | 0.00198266 |
| B_confirm | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | coordinate_rmse_Psi | 0.0296873 | 0.0350799 | lower | 0.846277 | 0.00539257 |
| B_confirm | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | learned_plane_drift_vector_corr | 0.687748 | 0.6645 | higher | 0.966197 | -0.0232481 |
| B_confirm | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | learned_plane_inward_fraction_to_reference | 0.305878 | 0.101163 | higher | 0.330731 | -0.204715 |
| B_confirm | main_vs_tag_support_randomized | predictive_state_event_ssl | tag_support_randomized | learned_plane_transition_mean_row_tv | 0.0941665 | 0.082915 | lower | 1.1357 | -0.0112515 |
| B_confirm | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | coordinate_corr_M | 0.798753 | 0.816093 | higher | 1.02171 | 0.0173398 |
| B_confirm | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | coordinate_corr_Psi | 0.914775 | 0.870265 | higher | 0.951343 | -0.04451 |
| B_confirm | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | learned_plane_drift_vector_corr | 0.520447 | 0.562233 | higher | 1.08029 | 0.0417862 |
| B_confirm | pure_ssl_vs_task_only | pure_event_ssl_probe | task_only | learned_plane_self_transition_corr | 0.330612 | 0.544 | higher | 1.64543 | 0.213388 |
| B_confirm | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | coordinate_corr_M | 0.795198 | 0.880768 | higher | 1.10761 | 0.0855701 |
| B_confirm | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | coordinate_corr_Psi | 0.944407 | 0.986369 | higher | 1.04443 | 0.0419617 |
| B_confirm | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | learned_plane_drift_vector_corr | -0.419145 | 0.6645 | higher | 1.58537 | 1.08364 |
| B_confirm | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | learned_plane_transition_mean_row_tv | 0.187396 | 0.082915 | lower | 2.26009 | -0.104481 |
| B_confirm | time_shuffle_vs_tag_support_randomized | time_shuffle_control | tag_support_randomized | learned_plane_inward_fraction_to_reference | 0.182425 | 0.101163 | higher | 0.554547 | -0.0812618 |
| B_confirm | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.880228 | 0.687748 | higher | 0.781329 | -0.19248 |
| B_confirm | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.819206 | 0.862941 | higher | 1.05339 | 0.0437351 |
| B_confirm | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.151246 | 0.0941665 | lower | 1.60615 | -0.0570791 |
| B_confirm | anchor_vs_learned_plane_within_model | predictive_state_event_ssl: empirical-anchor | predictive_state_event_ssl: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.952938 | 0.846233 | higher | 0.888025 | -0.106705 |
| B_confirm | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.824053 | 0.520447 | higher | 0.63157 | -0.303606 |
| B_confirm | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.732283 | 0.799114 | higher | 1.09126 | 0.0668318 |
| B_confirm | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.312108 | 0.175608 | lower | 1.7773 | -0.1365 |
| B_confirm | anchor_vs_learned_plane_within_model | pure_event_ssl_probe: empirical-anchor | pure_event_ssl_probe: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.800246 | 0.330612 | higher | 0.413138 | -0.469634 |
| B_confirm | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.882822 | 0.562233 | higher | 0.636859 | -0.320589 |
| B_confirm | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.740405 | 0.870876 | higher | 1.17622 | 0.130471 |
| B_confirm | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.338497 | 0.141673 | lower | 2.38928 | -0.196824 |
| B_confirm | anchor_vs_learned_plane_within_model | task_only: empirical-anchor | task_only: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.957722 | 0.544 | higher | 0.568015 | -0.413722 |
| B_confirm | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.804936 | -0.419145 | higher | -0.520719 | -1.22408 |
| B_confirm | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.680097 | 0.144615 | higher | 0.212638 | -0.535483 |
| B_confirm | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.254997 | 0.187396 | lower | 1.36074 | -0.0676019 |
| B_confirm | anchor_vs_learned_plane_within_model | time_shuffle_control: empirical-anchor | time_shuffle_control: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.963133 | 0.202306 | higher | 0.21005 | -0.760827 |
| B_confirm | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_drift_vector_corr vs learned_plane_drift_vector_corr | 0.887614 | 0.6645 | higher | 0.748636 | -0.223114 |
| B_confirm | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_occupancy_weighted_local_drift_cosine vs learned_plane_occupancy_weighted_local_drift_cosine | 0.720953 | 0.835572 | higher | 1.15898 | 0.114619 |
| B_confirm | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_transition_mean_row_tv vs learned_plane_transition_mean_row_tv | 0.16603 | 0.082915 | lower | 2.00241 | -0.0831146 |
| B_confirm | anchor_vs_learned_plane_within_model | tag_support_randomized: empirical-anchor | tag_support_randomized: learned-plane | anchor_self_transition_corr vs learned_plane_self_transition_corr | 0.941798 | 0.864764 | higher | 0.918205 | -0.0770339 |
| B_confirm | M_vs_Psi_coordinate_asymmetry | predictive_state_event_ssl | predictive_state_event_ssl | coordinate_corr_M - coordinate_corr_Psi | 0.883671 | 0.990493 | diagnostic | 1.12088 | 0.106821 |
| B_confirm | M_vs_Psi_coordinate_asymmetry | pure_event_ssl_probe | pure_event_ssl_probe | coordinate_corr_M - coordinate_corr_Psi | 0.798753 | 0.914775 | diagnostic | 1.14525 | 0.116023 |
| B_confirm | M_vs_Psi_coordinate_asymmetry | task_only | task_only | coordinate_corr_M - coordinate_corr_Psi | 0.816093 | 0.870265 | diagnostic | 1.06638 | 0.0541729 |
| B_confirm | M_vs_Psi_coordinate_asymmetry | time_shuffle_control | time_shuffle_control | coordinate_corr_M - coordinate_corr_Psi | 0.795198 | 0.944407 | diagnostic | 1.18764 | 0.149209 |
| B_confirm | M_vs_Psi_coordinate_asymmetry | tag_support_randomized | tag_support_randomized | coordinate_corr_M - coordinate_corr_Psi | 0.880768 | 0.986369 | diagnostic | 1.1199 | 0.105601 |
| A_val_to_B_confirm | development_to_confirmation_stability | predictive_state_event_ssl: A_val | predictive_state_event_ssl: B_confirm | coordinate_corr_M | 0.883381 | 0.883671 | higher | 1.00033 | 0.000290579 |
| A_val_to_B_confirm | development_to_confirmation_stability | predictive_state_event_ssl: A_val | predictive_state_event_ssl: B_confirm | coordinate_corr_Psi | 0.989569 | 0.990493 | higher | 1.00093 | 0.000923222 |
| A_val_to_B_confirm | development_to_confirmation_stability | predictive_state_event_ssl: A_val | predictive_state_event_ssl: B_confirm | next_state_occupancy_js | 0.199882 | 0.200516 | lower | 0.996838 | 0.000633962 |
| A_val_to_B_confirm | development_to_confirmation_stability | predictive_state_event_ssl: A_val | predictive_state_event_ssl: B_confirm | learned_plane_drift_vector_corr | 0.716323 | 0.687748 | higher | 0.960108 | -0.0285756 |
| A_val_to_B_confirm | development_to_confirmation_stability | predictive_state_event_ssl: A_val | predictive_state_event_ssl: B_confirm | learned_plane_transition_mean_row_tv | 0.097195 | 0.0941665 | lower | 1.03216 | -0.00302852 |
| A_val_to_B_confirm | development_to_confirmation_stability | pure_event_ssl_probe: A_val | pure_event_ssl_probe: B_confirm | coordinate_corr_M | 0.8038 | 0.798753 | higher | 0.993721 | -0.00504723 |
| A_val_to_B_confirm | development_to_confirmation_stability | pure_event_ssl_probe: A_val | pure_event_ssl_probe: B_confirm | coordinate_corr_Psi | 0.913984 | 0.914775 | higher | 1.00087 | 0.000791891 |
| A_val_to_B_confirm | development_to_confirmation_stability | pure_event_ssl_probe: A_val | pure_event_ssl_probe: B_confirm | next_state_occupancy_js | 0.230158 | 0.230471 | lower | 0.998644 | 0.000312463 |
| A_val_to_B_confirm | development_to_confirmation_stability | pure_event_ssl_probe: A_val | pure_event_ssl_probe: B_confirm | learned_plane_drift_vector_corr | 0.546843 | 0.520447 | higher | 0.95173 | -0.0263961 |
| A_val_to_B_confirm | development_to_confirmation_stability | pure_event_ssl_probe: A_val | pure_event_ssl_probe: B_confirm | learned_plane_transition_mean_row_tv | 0.184228 | 0.175608 | lower | 1.04908 | -0.00861949 |
| A_val_to_B_confirm | development_to_confirmation_stability | task_only: A_val | task_only: B_confirm | coordinate_corr_M | 0.814755 | 0.816093 | higher | 1.00164 | 0.00133795 |
| A_val_to_B_confirm | development_to_confirmation_stability | task_only: A_val | task_only: B_confirm | coordinate_corr_Psi | 0.870369 | 0.870265 | higher | 0.999881 | -0.00010322 |
| A_val_to_B_confirm | development_to_confirmation_stability | task_only: A_val | task_only: B_confirm | next_state_occupancy_js | 0.180714 | 0.182285 | lower | 0.99138 | 0.00157138 |
| A_val_to_B_confirm | development_to_confirmation_stability | task_only: A_val | task_only: B_confirm | learned_plane_drift_vector_corr | 0.545465 | 0.562233 | higher | 1.03074 | 0.0167683 |
| A_val_to_B_confirm | development_to_confirmation_stability | task_only: A_val | task_only: B_confirm | learned_plane_transition_mean_row_tv | 0.133681 | 0.141673 | lower | 0.943588 | 0.00799204 |
| A_val_to_B_confirm | development_to_confirmation_stability | time_shuffle_control: A_val | time_shuffle_control: B_confirm | coordinate_corr_M | 0.794357 | 0.795198 | higher | 1.00106 | 0.000841022 |
| A_val_to_B_confirm | development_to_confirmation_stability | time_shuffle_control: A_val | time_shuffle_control: B_confirm | coordinate_corr_Psi | 0.944921 | 0.944407 | higher | 0.999456 | -0.000514162 |
| A_val_to_B_confirm | development_to_confirmation_stability | time_shuffle_control: A_val | time_shuffle_control: B_confirm | next_state_occupancy_js | 0.201156 | 0.202171 | lower | 0.994982 | 0.00101443 |
| A_val_to_B_confirm | development_to_confirmation_stability | time_shuffle_control: A_val | time_shuffle_control: B_confirm | learned_plane_drift_vector_corr | -0.439093 | -0.419145 | higher | -0.95457 | 0.0199481 |
| A_val_to_B_confirm | development_to_confirmation_stability | time_shuffle_control: A_val | time_shuffle_control: B_confirm | learned_plane_transition_mean_row_tv | 0.196572 | 0.187396 | lower | 1.04897 | -0.00917623 |
| A_val_to_B_confirm | development_to_confirmation_stability | tag_support_randomized: A_val | tag_support_randomized: B_confirm | coordinate_corr_M | 0.881102 | 0.880768 | higher | 0.99962 | -0.000334396 |
| A_val_to_B_confirm | development_to_confirmation_stability | tag_support_randomized: A_val | tag_support_randomized: B_confirm | coordinate_corr_Psi | 0.985516 | 0.986369 | higher | 1.00086 | 0.000852427 |
| A_val_to_B_confirm | development_to_confirmation_stability | tag_support_randomized: A_val | tag_support_randomized: B_confirm | next_state_occupancy_js | 0.187936 | 0.189085 | lower | 0.993921 | 0.0011495 |
| A_val_to_B_confirm | development_to_confirmation_stability | tag_support_randomized: A_val | tag_support_randomized: B_confirm | learned_plane_drift_vector_corr | 0.665256 | 0.6645 | higher | 0.998863 | -0.000756336 |
| A_val_to_B_confirm | development_to_confirmation_stability | tag_support_randomized: A_val | tag_support_randomized: B_confirm | learned_plane_transition_mean_row_tv | 0.084731 | 0.082915 | lower | 1.0219 | -0.00181599 |

## Supplementary numerical ledger

| category | model_label | split | metric | value | source |
| --- | --- | --- | --- | --- | --- |
| control contrasts | time_shuffle_control | B_confirm | main_vs_time_shuffle::coordinate_corr_M | -0.0884735 | Stage-4 contrast table |
| control contrasts | time_shuffle_control | B_confirm | main_vs_time_shuffle::coordinate_corr_Psi | -0.0460857 | Stage-4 contrast table |
| control contrasts | time_shuffle_control | B_confirm | main_vs_time_shuffle::anchor_drift_vector_corr | -0.0752919 | Stage-4 contrast table |
| control contrasts | time_shuffle_control | B_confirm | main_vs_time_shuffle::learned_plane_drift_vector_corr | -1.10689 | Stage-4 contrast table |
| control contrasts | time_shuffle_control | B_confirm | main_vs_time_shuffle::learned_plane_transition_mean_row_tv | 0.093229 | Stage-4 contrast table |
| control contrasts | time_shuffle_control | B_confirm | main_vs_time_shuffle::learned_plane_self_transition_corr | -0.643927 | Stage-4 contrast table |
| control contrasts | time_shuffle_control | B_confirm | main_vs_time_shuffle::learned_plane_inward_fraction_to_reference | -0.123453 | Stage-4 contrast table |
| control contrasts | tag_support_randomized | B_confirm | main_vs_tag_support_randomized::coordinate_corr_M | -0.00290331 | Stage-4 contrast table |
| control contrasts | tag_support_randomized | B_confirm | main_vs_tag_support_randomized::coordinate_corr_Psi | -0.00412403 | Stage-4 contrast table |
| control contrasts | tag_support_randomized | B_confirm | main_vs_tag_support_randomized::coordinate_rmse_M | 0.00198266 | Stage-4 contrast table |
| control contrasts | tag_support_randomized | B_confirm | main_vs_tag_support_randomized::coordinate_rmse_Psi | 0.00539257 | Stage-4 contrast table |
| control contrasts | tag_support_randomized | B_confirm | main_vs_tag_support_randomized::learned_plane_drift_vector_corr | -0.0232481 | Stage-4 contrast table |
| control contrasts | tag_support_randomized | B_confirm | main_vs_tag_support_randomized::learned_plane_inward_fraction_to_reference | -0.204715 | Stage-4 contrast table |
| control contrasts | tag_support_randomized | B_confirm | main_vs_tag_support_randomized::learned_plane_transition_mean_row_tv | -0.0112515 | Stage-4 contrast table |
