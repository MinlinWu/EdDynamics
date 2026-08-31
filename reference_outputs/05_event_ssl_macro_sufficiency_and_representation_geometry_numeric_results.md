---
source_report: "stage5_joint_macro_geometry_report.md"
source_extraction_script: "s3_event_ssl/scripts/extract_event_ssl_stage5_publication_statistics.py"
---

# Event-SSL macro-sufficiency and representation-geometry numerical results

## Metric ledger

| category | split | metric | value | source |
| --- | --- | --- | --- | --- |
| macro-sufficiency | B_confirm | Macro-only composite retention | 0.908364 | Stage-5 macro retention table |
| macro-sufficiency | B_confirm | Macro-only coordinate retention | 0.995318 | Stage-5 macro retention table |
| macro-sufficiency | B_confirm | Macro-only closure retention | 0.79875 | Stage-5 macro retention table |
| macro-sufficiency | B_confirm | Macro-only transition retention | 1.00254 | Stage-5 macro retention table |
| macro-sufficiency | B_confirm | Macro-only drift retention | 0.810319 | Stage-5 macro retention table |
| residual hidden | B_confirm | Residual-hidden composite retention | 0.59732 | Stage-5 macro retention table |
| residual hidden | B_confirm | Residual-hidden drift retention | 0.802109 | Stage-5 macro retention table |
| residual hidden | B_confirm | Residual-hidden coordinate retention | 0.558612 | Stage-5 macro retention table |
| residual hidden | B_confirm | Residual-hidden transition retention | 0.517981 | Stage-5 macro retention table |
| task retention | B_confirm | Macro-only task retention | 0.916934 | Stage-5 macro retention table |
| representation geometry | B_confirm | Linear-hidden coordinate retention | 1.0047 | Stage-5 geometry retention table |
| representation geometry | B_confirm | Linear-hidden composite retention | 1.02492 | Stage-5 geometry retention table |
| representation geometry | B_confirm | Linear-hidden drift retention | 1.02889 | Stage-5 geometry retention table |
| representation geometry | B_confirm | Linear-hidden transition retention | 0.985396 | Stage-5 geometry retention table |
| representation geometry | B_confirm | First canonical correlation | 0.99126 | Stage-5 representation-geometry metrics |
| representation geometry | B_confirm | Second canonical correlation | 0.899766 | Stage-5 representation-geometry metrics |
| representation geometry | B_confirm | TwoNN intrinsic-dimension estimate | 7.47141 | Stage-5 representation-geometry metrics |
| representation geometry | B_confirm | Participation ratio | 11.2286 | Stage-5 representation-geometry metrics |
| representation geometry | B_confirm | Effective rank | 18.8367 | Stage-5 representation-geometry metrics |
| representation geometry | B_confirm | Geometry sample rows | 250000 | Stage-5 representation-geometry metrics |
| principal components | B_confirm | Maximum absolute PC correlation with M | 0.779113 | Stage-5 PC correlation table |
| principal components | B_confirm | Maximum absolute PC correlation with Psi | 0.635294 | Stage-5 PC correlation table |
| nonlinear probe | B_confirm | nonlinear_gain_corr_M | 0.00692185 | Stage-5 nonlinear-gain table |
| nonlinear probe | B_confirm | nonlinear_gain_corr_Psi | -0.00106691 | Stage-5 nonlinear-gain table |
| nonlinear probe | B_confirm | nonlinear_gain_rmse_reduction_M | 0.00507829 | Stage-5 nonlinear-gain table |
| nonlinear probe | B_confirm | nonlinear_gain_rmse_reduction_Psi | -0.00163426 | Stage-5 nonlinear-gain table |
| validation-confirmation stability | A_val_to_B_confirm | absolute gap representation_geometry/linear_hidden/coordinate_corr_M | 0.000171445 | Stage-5 stability table |
| validation-confirmation stability | A_val_to_B_confirm | absolute gap representation_geometry/linear_hidden/coordinate_corr_Psi | 6.18182e-05 | Stage-5 stability table |

## A_val: numerical contrasts

| split | contrast_id | primary_value | secondary_value | contrast_value |
| --- | --- | --- | --- | --- |
| A_val | macro_bottleneck_retains_structure | 0.90339 | 0.797132 | 0.269837 |
| A_val | residual_hidden_loses_macrostructure | 0.603912 | 0.806111 | 0.668496 |
| A_val | macro_bottleneck_task_retention | 0.918482 | 0.5915 | 0.00267453 |
| A_val | linear_hidden_accesses_macrostate | 1.00496 | 1.02356 | 0.409174 |
| A_val | canonical_macro_alignment | 0.99132 | 0.900557 | nan |
| A_val | nonlinear_probe_limited_gain | 0.00675459 | -0.0011754 | nan |
| A_val | leading_pc_macro_alignment | 0.779246 | 0.656267 | nan |

### A_val: macro-sufficiency retention

| split | domain | full_hidden_score | macro_only_score | residual_hidden_score | macro_retention_vs_full | residual_retention_vs_full | residual_retention_vs_macro | macro_minus_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | coordinate_score | 0.97306 | 0.968238 | 0.562909 | 0.995044 | 0.578494 | 0.581375 | 0.405329 |
| A_val | closure_score | 0.767699 | 0.612907 | 0.38721 | 0.79837 | 0.504377 | 0.631759 | 0.225697 |
| A_val | drift_score | 0.91648 | 0.730556 | 0.738785 | 0.797132 | 0.806111 | 1.01126 | -0.00822866 |
| A_val | transition_score | 0.946862 | 0.944208 | 0.487658 | 0.997197 | 0.515025 | 0.516473 | 0.45655 |
| A_val | task_score | 0.643997 | 0.5915 | 0.588825 | 0.918482 | 0.914329 | 0.995478 | 0.00267453 |
| A_val | macro_label_score | 0.368396 | 0.512615 | 0.285526 | 1.39148 | 0.775052 | 0.556998 | 0.22709 |
| A_val | macrostructure_composite_descriptive | 0.901025 | 0.813977 | 0.54414 | 0.90339 | 0.603912 | 0.668496 | 0.269837 |

### A_val: representation-geometry retention

| split | domain | model_readout_score | linear_hidden_score | residual_hidden_score | nonlinear_hidden_score | linear_retention_vs_model | residual_retention_vs_model | residual_retention_vs_linear | linear_minus_residual | nonlinear_retention_vs_linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_val | coordinate_score | 0.968238 | 0.973036 | 0.563862 | 0.974431 | 1.00496 | 0.58236 | 0.579488 | 0.409174 | 1.00143 |
| A_val | closure_score | 0.693052 | 0.767458 | 0.387228 | nan | 1.10736 | 0.558728 | 0.504559 | 0.38023 | nan |
| A_val | drift_score | 0.897966 | 0.917703 | 0.757218 | nan | 1.02198 | 0.843259 | 0.825123 | 0.160485 | nan |
| A_val | transition_score | 0.962703 | 0.946723 | 0.488444 | nan | 0.983401 | 0.507368 | 0.515932 | 0.458278 | nan |
| A_val | macrostructure_composite_descriptive | 0.88049 | 0.90123 | 0.549188 | 0.974431 | 1.02356 | 0.62373 | 0.609376 | 0.352042 | 1.08122 |

## B_confirm: numerical contrasts

| split | contrast_id | primary_value | secondary_value | contrast_value |
| --- | --- | --- | --- | --- |
| B_confirm | macro_bottleneck_retains_structure | 0.908364 | 0.810319 | 0.279608 |
| B_confirm | residual_hidden_loses_macrostructure | 0.59732 | 0.802109 | 0.657578 |
| B_confirm | macro_bottleneck_task_retention | 0.916934 | 0.590805 | 0.000955888 |
| B_confirm | linear_hidden_accesses_macrostate | 1.0047 | 1.02492 | 0.429803 |
| B_confirm | canonical_macro_alignment | 0.99126 | 0.899766 | nan |
| B_confirm | nonlinear_probe_limited_gain | 0.00692185 | -0.00106691 | nan |
| B_confirm | leading_pc_macro_alignment | 0.779113 | 0.635294 | nan |

### B_confirm: macro-sufficiency retention

| split | domain | full_hidden_score | macro_only_score | residual_hidden_score | macro_retention_vs_full | residual_retention_vs_full | residual_retention_vs_macro | macro_minus_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | coordinate_score | 0.973097 | 0.968541 | 0.543584 | 0.995318 | 0.558612 | 0.56124 | 0.424957 |
| B_confirm | closure_score | 0.764337 | 0.610515 | 0.381739 | 0.79875 | 0.499438 | 0.625274 | 0.228776 |
| B_confirm | drift_score | 0.914792 | 0.741273 | 0.733762 | 0.810319 | 0.802109 | 0.989868 | 0.00751074 |
| B_confirm | transition_score | 0.94351 | 0.945908 | 0.48872 | 1.00254 | 0.517981 | 0.516668 | 0.457188 |
| B_confirm | task_score | 0.644326 | 0.590805 | 0.589849 | 0.916934 | 0.915451 | 0.998382 | 0.000955888 |
| B_confirm | macro_label_score | 0.366352 | 0.510926 | 0.284159 | 1.39463 | 0.775645 | 0.556164 | 0.226768 |
| B_confirm | macrostructure_composite_descriptive | 0.898934 | 0.816559 | 0.536951 | 0.908364 | 0.59732 | 0.657578 | 0.279608 |

### B_confirm: representation-geometry retention

| split | domain | model_readout_score | linear_hidden_score | residual_hidden_score | nonlinear_hidden_score | linear_retention_vs_model | residual_retention_vs_model | residual_retention_vs_linear | linear_minus_residual | nonlinear_retention_vs_linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_confirm | coordinate_score | 0.968541 | 0.973094 | 0.543292 | 0.974558 | 1.0047 | 0.560938 | 0.558313 | 0.429803 | 1.0015 |
| B_confirm | closure_score | 0.693162 | 0.764327 | 0.381751 | nan | 1.10267 | 0.550738 | 0.49946 | 0.382576 | nan |
| B_confirm | drift_score | 0.887672 | 0.913321 | 0.740653 | nan | 1.02889 | 0.834376 | 0.810944 | 0.172669 | nan |
| B_confirm | transition_score | 0.957237 | 0.943258 | 0.490819 | nan | 0.985396 | 0.512746 | 0.520345 | 0.452438 | nan |
| B_confirm | macrostructure_composite_descriptive | 0.876653 | 0.8985 | 0.539129 | 0.974558 | 1.02492 | 0.614985 | 0.600032 | 0.359371 | 1.08465 |

## Nonlinear probe gain

| split | nonlinear_gain_corr_M | nonlinear_gain_rmse_reduction_M | nonlinear_gain_corr_Psi | nonlinear_gain_rmse_reduction_Psi |
| --- | --- | --- | --- | --- |
| A_val | 0.00675459 | 0.0048731 | -0.0011754 | -0.00175976 |
| B_confirm | 0.00692185 | 0.00507829 | -0.00106691 | -0.00163426 |

## PC macro correlations

| split | component | corr_M | corr_Psi | abs_corr_M | abs_corr_Psi | explained_variance_ratio_train |
| --- | --- | --- | --- | --- | --- | --- |
| A_val | PC1 | -0.260614 | 0.656267 | 0.260614 | 0.656267 | 0.187451 |
| A_val | PC2 | 0.779246 | -0.07556 | 0.779246 | 0.07556 | 0.142263 |
| A_val | PC3 | -0.136701 | -0.435926 | 0.136701 | 0.435926 | 0.0976445 |
| A_val | PC4 | 0.16392 | 0.334513 | 0.16392 | 0.334513 | 0.0648607 |
| A_val | PC5 | -0.0151124 | 0.258398 | 0.0151124 | 0.258398 | 0.057609 |
| A_val | PC6 | -0.0232198 | 0.0509236 | 0.0232198 | 0.0509236 | 0.0516659 |
| A_val | PC7 | 0.0158063 | 0.0399834 | 0.0158063 | 0.0399834 | 0.0405791 |
| A_val | PC8 | 0.0914423 | 0.218981 | 0.0914423 | 0.218981 | 0.0372457 |
| A_val | PC9 | -0.013444 | -0.201729 | 0.013444 | 0.201729 | 0.0348592 |
| A_val | PC10 | -0.0260972 | -0.048939 | 0.0260972 | 0.048939 | 0.0278204 |
| A_val | PC11 | -0.0335759 | 0.0752825 | 0.0335759 | 0.0752825 | 0.0260316 |
| A_val | PC12 | 0.00764867 | -0.062603 | 0.00764867 | 0.062603 | 0.0219516 |
| A_val | PC13 | -0.0347888 | 0.0584261 | 0.0347888 | 0.0584261 | 0.0183589 |
| A_val | PC14 | 0.0776874 | 0.0109059 | 0.0776874 | 0.0109059 | 0.0157866 |
| A_val | PC15 | 0.00724481 | -0.0458313 | 0.00724481 | 0.0458313 | 0.0144837 |
| A_val | PC16 | 0.11145 | -0.014808 | 0.11145 | 0.014808 | 0.0121843 |
| B_confirm | PC1 | -0.250705 | 0.635294 | 0.250705 | 0.635294 | 0.187451 |
| B_confirm | PC2 | 0.779113 | -0.0823974 | 0.779113 | 0.0823974 | 0.142263 |
| B_confirm | PC3 | -0.140206 | -0.469775 | 0.140206 | 0.469775 | 0.0976445 |
| B_confirm | PC4 | 0.162935 | 0.345771 | 0.162935 | 0.345771 | 0.0648607 |
| B_confirm | PC5 | -0.012772 | 0.258301 | 0.012772 | 0.258301 | 0.057609 |
| B_confirm | PC6 | 0.0139791 | 0.0155395 | 0.0139791 | 0.0155395 | 0.0516659 |
| B_confirm | PC7 | 0.00557234 | 0.0721791 | 0.00557234 | 0.0721791 | 0.0405791 |
| B_confirm | PC8 | 0.0573965 | 0.256567 | 0.0573965 | 0.256567 | 0.0372457 |
| B_confirm | PC9 | -0.0728473 | -0.153866 | 0.0728473 | 0.153866 | 0.0348592 |
| B_confirm | PC10 | -0.0114694 | -0.06075 | 0.0114694 | 0.06075 | 0.0278204 |
| B_confirm | PC11 | -0.0163568 | 0.0207668 | 0.0163568 | 0.0207668 | 0.0260316 |
| B_confirm | PC12 | 0.00201426 | -0.0817695 | 0.00201426 | 0.0817695 | 0.0219516 |
| B_confirm | PC13 | -0.0331991 | 0.0598564 | 0.0331991 | 0.0598564 | 0.0183589 |
| B_confirm | PC14 | 0.0699077 | 0.00959978 | 0.0699077 | 0.00959978 | 0.0157866 |
| B_confirm | PC15 | 6.50984e-05 | -0.0353433 | 6.50984e-05 | 0.0353433 | 0.0144837 |
| B_confirm | PC16 | 0.119994 | -0.0546612 | 0.119994 | 0.0546612 | 0.0121843 |

## Validation-confirmation stability

| experiment | representation | metric | validation_value | confirmation_value | confirmation_minus_validation | absolute_gap |
| --- | --- | --- | --- | --- | --- | --- |
| macro_sufficiency | full_hidden | coordinate_corr_M | 0.901081 | 0.901219 | 0.000137735 | 0.000137735 |
| macro_sufficiency | full_hidden | coordinate_corr_Psi | 0.991157 | 0.99117 | 1.26397e-05 | 1.26397e-05 |
| macro_sufficiency | full_hidden | coordinate_rmse_M | 0.147257 | 0.149583 | 0.00232631 | 0.00232631 |
| macro_sufficiency | full_hidden | coordinate_rmse_Psi | 0.0268245 | 0.0273051 | 0.000480616 | 0.000480616 |
| macro_sufficiency | full_hidden | one_step_rmse_M | 0.069047 | 0.0703224 | 0.0012754 | 0.0012754 |
| macro_sufficiency | full_hidden | one_step_rmse_Psi | 0.0263434 | 0.0269173 | 0.000573851 | 0.000573851 |
| macro_sufficiency | full_hidden | next_state_occupancy_js | 0.112112 | 0.112968 | 0.00085571 | 0.00085571 |
| macro_sufficiency | full_hidden | learned_plane_drift_vector_corr | 0.731239 | 0.717232 | -0.0140062 | 0.0140062 |
| macro_sufficiency | full_hidden | learned_plane_occupancy_weighted_local_drift_cosine | 0.934682 | 0.941935 | 0.0072537 | 0.0072537 |
| macro_sufficiency | full_hidden | learned_plane_transition_mean_row_tv | 0.109382 | 0.106896 | -0.00248622 | 0.00248622 |
| macro_sufficiency | full_hidden | learned_plane_self_transition_corr | 0.793662 | 0.761873 | -0.0317888 | 0.0317888 |
| macro_sufficiency | full_hidden | task_auc | 0.643997 | 0.644326 | 0.000329181 | 0.000329181 |
| macro_sufficiency | full_hidden | task_bce | 0.637499 | 0.640262 | 0.00276263 | 0.00276263 |
| macro_sufficiency | full_hidden | representation_nmi_with_empirical_macrostate | 0.187826 | 0.185231 | -0.0025956 | 0.0025956 |
| macro_sufficiency | full_hidden | representation_ari_with_empirical_macrostate | 0.0979299 | 0.094945 | -0.00298481 | 0.00298481 |
| macro_sufficiency | full_hidden | cca_corr_1 | nan | nan | nan | nan |
| macro_sufficiency | full_hidden | cca_corr_2 | nan | nan | nan | nan |
| macro_sufficiency | full_hidden | twonn_dimension | nan | nan | nan | nan |
| macro_sufficiency | full_hidden | participation_ratio_train | nan | nan | nan | nan |
| macro_sufficiency | full_hidden | effective_rank_train | nan | nan | nan | nan |
| macro_sufficiency | macro_only | coordinate_corr_M | 0.883381 | 0.883671 | 0.000290579 | 0.000290579 |
| macro_sufficiency | macro_only | coordinate_corr_Psi | 0.989569 | 0.990493 | 0.000923222 | 0.000923222 |
| macro_sufficiency | macro_only | coordinate_rmse_M | 0.159713 | 0.162192 | 0.00247966 | 0.00247966 |
| macro_sufficiency | macro_only | coordinate_rmse_Psi | 0.0305708 | 0.0296873 | -0.000883466 | 0.000883466 |
| macro_sufficiency | macro_only | one_step_rmse_M | 0.15017 | 0.152381 | 0.00221136 | 0.00221136 |
| macro_sufficiency | macro_only | one_step_rmse_Psi | 0.0565838 | 0.056906 | 0.000322234 | 0.000322234 |
| macro_sufficiency | macro_only | next_state_occupancy_js | 0.24858 | 0.249812 | 0.00123209 | 0.00123209 |
| macro_sufficiency | macro_only | learned_plane_drift_vector_corr | 0.523119 | 0.564541 | 0.0414224 | 0.0414224 |
| macro_sufficiency | macro_only | learned_plane_occupancy_weighted_local_drift_cosine | 0.399105 | 0.400552 | 0.00144645 | 0.00144645 |
| macro_sufficiency | macro_only | learned_plane_transition_mean_row_tv | 0.150227 | 0.137776 | -0.0124517 | 0.0124517 |
| macro_sufficiency | macro_only | learned_plane_self_transition_corr | 0.854116 | 0.842816 | -0.0113003 | 0.0113003 |
| macro_sufficiency | macro_only | task_auc | 0.5915 | 0.590805 | -0.000695219 | 0.000695219 |
| macro_sufficiency | macro_only | task_bce | 0.640665 | 0.643421 | 0.0027569 | 0.0027569 |
| macro_sufficiency | macro_only | representation_nmi_with_empirical_macrostate | 0.402951 | 0.40295 | -3.62968e-07 | 3.62968e-07 |
| macro_sufficiency | macro_only | representation_ari_with_empirical_macrostate | 0.24456 | 0.237805 | -0.00675554 | 0.00675554 |
| macro_sufficiency | macro_only | cca_corr_1 | nan | nan | nan | nan |
| macro_sufficiency | macro_only | cca_corr_2 | nan | nan | nan | nan |
| macro_sufficiency | macro_only | twonn_dimension | nan | nan | nan | nan |
| macro_sufficiency | macro_only | participation_ratio_train | nan | nan | nan | nan |
| macro_sufficiency | macro_only | effective_rank_train | nan | nan | nan | nan |
| macro_sufficiency | residual_hidden | coordinate_corr_M | 0.123832 | 0.136651 | 0.0128191 | 0.0128191 |
| macro_sufficiency | residual_hidden | coordinate_corr_Psi | 0.127803 | 0.0376849 | -0.0901183 | 0.0901183 |
| macro_sufficiency | residual_hidden | coordinate_rmse_M | 0.337179 | 0.342256 | 0.00507686 | 0.00507686 |
| macro_sufficiency | residual_hidden | coordinate_rmse_Psi | 0.200873 | 0.205947 | 0.00507462 | 0.00507462 |
| macro_sufficiency | residual_hidden | one_step_rmse_M | 0.310682 | 0.316292 | 0.0056099 | 0.0056099 |
| macro_sufficiency | residual_hidden | one_step_rmse_Psi | 0.184213 | 0.189527 | 0.00531419 | 0.00531419 |
| macro_sufficiency | residual_hidden | next_state_occupancy_js | 0.494174 | 0.492922 | -0.00125136 | 0.00125136 |
| macro_sufficiency | residual_hidden | learned_plane_drift_vector_corr | 0.645573 | 0.651587 | 0.0060142 | 0.0060142 |
| macro_sufficiency | residual_hidden | learned_plane_occupancy_weighted_local_drift_cosine | 0.309565 | 0.283463 | -0.0261029 | 0.0261029 |
| macro_sufficiency | residual_hidden | learned_plane_transition_mean_row_tv | 0.513799 | 0.512551 | -0.00124765 | 0.00124765 |
| macro_sufficiency | residual_hidden | learned_plane_self_transition_corr | 0.595525 | 0.601532 | 0.00600692 | 0.00600692 |
| macro_sufficiency | residual_hidden | task_auc | 0.588825 | 0.589849 | 0.00102342 | 0.00102342 |
| macro_sufficiency | residual_hidden | task_bce | 0.655141 | 0.659411 | 0.00426999 | 0.00426999 |
| macro_sufficiency | residual_hidden | representation_nmi_with_empirical_macrostate | 0.0501491 | 0.0481777 | -0.00197136 | 0.00197136 |
| macro_sufficiency | residual_hidden | representation_ari_with_empirical_macrostate | 0.0418041 | 0.0402796 | -0.0015244 | 0.0015244 |
| macro_sufficiency | residual_hidden | cca_corr_1 | nan | nan | nan | nan |
| macro_sufficiency | residual_hidden | cca_corr_2 | nan | nan | nan | nan |
| macro_sufficiency | residual_hidden | twonn_dimension | nan | nan | nan | nan |
| macro_sufficiency | residual_hidden | participation_ratio_train | nan | nan | nan | nan |
| macro_sufficiency | residual_hidden | effective_rank_train | nan | nan | nan | nan |
| representation_geometry | model_readout | coordinate_corr_M | 0.883381 | 0.883671 | 0.000290579 | 0.000290579 |
| representation_geometry | model_readout | coordinate_corr_Psi | 0.989569 | 0.990493 | 0.000923222 | 0.000923222 |
| representation_geometry | model_readout | coordinate_rmse_M | 0.159713 | 0.162192 | 0.00247966 | 0.00247966 |
| representation_geometry | model_readout | coordinate_rmse_Psi | 0.0305708 | 0.0296873 | -0.000883466 | 0.000883466 |
| representation_geometry | model_readout | one_step_rmse_M | 0.108253 | 0.109951 | 0.00169802 | 0.00169802 |
| representation_geometry | model_readout | one_step_rmse_Psi | 0.0362712 | 0.0353473 | -0.000923976 | 0.000923976 |
| representation_geometry | model_readout | next_state_occupancy_js | 0.199882 | 0.200516 | 0.000633962 | 0.000633962 |
| representation_geometry | model_readout | learned_plane_drift_vector_corr | 0.716323 | 0.687748 | -0.0285756 | 0.0285756 |
| representation_geometry | model_readout | learned_plane_occupancy_weighted_local_drift_cosine | 0.875541 | 0.862941 | -0.0126003 | 0.0126003 |
| representation_geometry | model_readout | learned_plane_transition_mean_row_tv | 0.097195 | 0.0941665 | -0.00302852 | 0.00302852 |
| representation_geometry | model_readout | learned_plane_self_transition_corr | 0.896012 | 0.846233 | -0.0497793 | 0.0497793 |
| representation_geometry | model_readout | task_auc | nan | nan | nan | nan |
| representation_geometry | model_readout | task_bce | nan | nan | nan | nan |
| representation_geometry | model_readout | representation_nmi_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | model_readout | representation_ari_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | model_readout | cca_corr_1 | 0.99132 | 0.99126 | -6.02284e-05 | 6.02284e-05 |
| representation_geometry | model_readout | cca_corr_2 | 0.900557 | 0.899766 | -0.000790427 | 0.000790427 |
| representation_geometry | model_readout | twonn_dimension | 7.76637 | 7.47141 | -0.294953 | 0.294953 |
| representation_geometry | model_readout | participation_ratio_train | 11.2286 | 11.2286 | 0 | 0 |
| representation_geometry | model_readout | effective_rank_train | 18.8367 | 18.8367 | 0 | 0 |
| representation_geometry | linear_hidden | coordinate_corr_M | 0.901029 | 0.901201 | 0.000171445 | 0.000171445 |
| representation_geometry | linear_hidden | coordinate_corr_Psi | 0.991115 | 0.991177 | 6.18182e-05 | 6.18182e-05 |
| representation_geometry | linear_hidden | coordinate_rmse_M | 0.147294 | 0.149597 | 0.00230277 | 0.00230277 |
| representation_geometry | linear_hidden | coordinate_rmse_Psi | 0.026888 | 0.0272922 | 0.000404168 | 0.000404168 |
| representation_geometry | linear_hidden | one_step_rmse_M | 0.0691081 | 0.0703328 | 0.0012247 | 0.0012247 |
| representation_geometry | linear_hidden | one_step_rmse_Psi | 0.0264037 | 0.026915 | 0.000511328 | 0.000511328 |
| representation_geometry | linear_hidden | next_state_occupancy_js | 0.111695 | 0.11273 | 0.00103537 | 0.00103537 |
| representation_geometry | linear_hidden | learned_plane_drift_vector_corr | 0.734881 | 0.717816 | -0.0170647 | 0.0170647 |
| representation_geometry | linear_hidden | learned_plane_occupancy_weighted_local_drift_cosine | 0.93593 | 0.935469 | -0.000460511 | 0.000460511 |
| representation_geometry | linear_hidden | learned_plane_transition_mean_row_tv | 0.109415 | 0.106901 | -0.00251361 | 0.00251361 |
| representation_geometry | linear_hidden | learned_plane_self_transition_corr | 0.79261 | 0.759866 | -0.0327443 | 0.0327443 |
| representation_geometry | linear_hidden | task_auc | nan | nan | nan | nan |
| representation_geometry | linear_hidden | task_bce | nan | nan | nan | nan |
| representation_geometry | linear_hidden | representation_nmi_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | linear_hidden | representation_ari_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | linear_hidden | cca_corr_1 | 0.99132 | 0.99126 | -6.02284e-05 | 6.02284e-05 |
| representation_geometry | linear_hidden | cca_corr_2 | 0.900557 | 0.899766 | -0.000790427 | 0.000790427 |
| representation_geometry | linear_hidden | twonn_dimension | 7.76637 | 7.47141 | -0.294953 | 0.294953 |
| representation_geometry | linear_hidden | participation_ratio_train | 11.2286 | 11.2286 | 0 | 0 |
| representation_geometry | linear_hidden | effective_rank_train | 18.8367 | 18.8367 | 0 | 0 |
| representation_geometry | residual_hidden | coordinate_corr_M | 0.123382 | 0.13597 | 0.0125888 | 0.0125888 |
| representation_geometry | residual_hidden | coordinate_corr_Psi | 0.132068 | 0.037196 | -0.094872 | 0.094872 |
| representation_geometry | residual_hidden | coordinate_rmse_M | 0.337211 | 0.342311 | 0.00509992 | 0.00509992 |
| representation_geometry | residual_hidden | coordinate_rmse_Psi | 0.200807 | 0.205947 | 0.00514041 | 0.00514041 |
| representation_geometry | residual_hidden | one_step_rmse_M | 0.3107 | 0.316324 | 0.00562373 | 0.00562373 |
| representation_geometry | residual_hidden | one_step_rmse_Psi | 0.184177 | 0.189492 | 0.00531453 | 0.00531453 |
| representation_geometry | residual_hidden | next_state_occupancy_js | 0.493037 | 0.491617 | -0.00142055 | 0.00142055 |
| representation_geometry | residual_hidden | learned_plane_drift_vector_corr | 0.659332 | 0.6522 | -0.00713245 | 0.00713245 |
| representation_geometry | residual_hidden | learned_plane_occupancy_weighted_local_drift_cosine | 0.369538 | 0.310411 | -0.0591275 | 0.0591275 |
| representation_geometry | residual_hidden | learned_plane_transition_mean_row_tv | 0.51191 | 0.508997 | -0.00291312 | 0.00291312 |
| representation_geometry | residual_hidden | learned_plane_self_transition_corr | 0.59804 | 0.611216 | 0.0131757 | 0.0131757 |
| representation_geometry | residual_hidden | task_auc | nan | nan | nan | nan |
| representation_geometry | residual_hidden | task_bce | nan | nan | nan | nan |
| representation_geometry | residual_hidden | representation_nmi_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | residual_hidden | representation_ari_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | residual_hidden | cca_corr_1 | 0.99132 | 0.99126 | -6.02284e-05 | 6.02284e-05 |
| representation_geometry | residual_hidden | cca_corr_2 | 0.900557 | 0.899766 | -0.000790427 | 0.000790427 |
| representation_geometry | residual_hidden | twonn_dimension | 7.76637 | 7.47141 | -0.294953 | 0.294953 |
| representation_geometry | residual_hidden | participation_ratio_train | 11.2286 | 11.2286 | 0 | 0 |
| representation_geometry | residual_hidden | effective_rank_train | 18.8367 | 18.8367 | 0 | 0 |
| representation_geometry | nonlinear_hidden | coordinate_corr_M | 0.907784 | 0.908123 | 0.000338701 | 0.000338701 |
| representation_geometry | nonlinear_hidden | coordinate_corr_Psi | 0.98994 | 0.99011 | 0.000170304 | 0.000170304 |
| representation_geometry | nonlinear_hidden | coordinate_rmse_M | 0.142421 | 0.144518 | 0.00209758 | 0.00209758 |
| representation_geometry | nonlinear_hidden | coordinate_rmse_Psi | 0.0286478 | 0.0289264 | 0.000278661 | 0.000278661 |
| representation_geometry | nonlinear_hidden | one_step_rmse_M | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | one_step_rmse_Psi | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | next_state_occupancy_js | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | learned_plane_drift_vector_corr | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | learned_plane_occupancy_weighted_local_drift_cosine | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | learned_plane_transition_mean_row_tv | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | learned_plane_self_transition_corr | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | task_auc | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | task_bce | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | representation_nmi_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | representation_ari_with_empirical_macrostate | nan | nan | nan | nan |
| representation_geometry | nonlinear_hidden | cca_corr_1 | 0.99132 | 0.99126 | -6.02284e-05 | 6.02284e-05 |
| representation_geometry | nonlinear_hidden | cca_corr_2 | 0.900557 | 0.899766 | -0.000790427 | 0.000790427 |
| representation_geometry | nonlinear_hidden | twonn_dimension | 7.76637 | 7.47141 | -0.294953 | 0.294953 |
| representation_geometry | nonlinear_hidden | participation_ratio_train | 11.2286 | 11.2286 | 0 | 0 |
| representation_geometry | nonlinear_hidden | effective_rank_train | 18.8367 | 18.8367 | 0 | 0 |
