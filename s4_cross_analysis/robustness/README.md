# Cross-analysis robustness

Set shared paths:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
S1_MAIN=${REPO_ROOT}/s1_empirical_dynamics/main
CONSTRUCTION=${REPO_ROOT}/s1_empirical_dynamics/robustness/construction_null
S2_MAIN=${REPO_ROOT}/s2_minimal_mechanism/main
S3_MAIN=${REPO_ROOT}/s3_event_ssl/main
CROSS_MAIN=${REPO_ROOT}/s4_cross_analysis/main
CROSS_SCRIPTS=${REPO_ROOT}/s4_cross_analysis/scripts
CROSS_ROBUSTNESS=${REPO_ROOT}/s4_cross_analysis/robustness
```

## Null-referenced downstream recovery

```bash
OUTPUTS_ROOT=${OUTPUT_ROOT} \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
CONSTRUCTION_NULL_ROOT=${OUTPUT_ROOT}/stage1_construction_matched_null \
FROZEN_MECHANISM_MANIFEST=${OUTPUT_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json \
RESULT_ROOT=${OUTPUT_ROOT}/null_referenced_downstream_recovery \
ANALYSIS_SCRIPT=${CROSS_MAIN}/evaluate_null_referenced_downstream_recovery.py \
STAGE1_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
CONSTRUCTION_NULL_SCRIPT=${CONSTRUCTION}/run_construction_matched_null.py \
MECHANISM_CONFIRM_SCRIPT=${S2_MAIN}/confirm_offset_dual_channel_phase3.py \
MECHANISM_PHASE1_SCRIPT=${S2_MAIN}/tune_offset_dual_channel_phase1.py \
EVENT_SSL_SEED42_ROOT=${OUTPUT_ROOT}/stage4_event_ssl/evaluation_predictive_state \
bash ${CROSS_ROBUSTNESS}/run_null_referenced_downstream_recovery.sh
```

## Frozen-headline learner-cluster uncertainty

```bash
OUTPUTS_ROOT=${OUTPUT_ROOT} \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
PHASE3_ROOT=${OUTPUT_ROOT}/stage2_phase3_confirm \
FROZEN_MANIFEST=${OUTPUT_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json \
EVENT_SSL_ROOT=${OUTPUT_ROOT}/stage4_event_ssl/evaluation_predictive_state \
RESULT_ROOT=${OUTPUT_ROOT}/frozen_headline_learner_cluster_uncertainty \
MECHANISM_EXPORT_ROOT=${OUTPUT_ROOT}/frozen_headline_learner_cluster_uncertainty_mechanism_export \
ANALYSIS_SCRIPT=${CROSS_MAIN}/run_frozen_headline_learner_cluster_uncertainty.py \
EXTRACT_SCRIPT=${CROSS_SCRIPTS}/extract_frozen_headline_learner_cluster_uncertainty_report.py \
STAGE1_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
PHASE3_SCRIPT=${S2_MAIN}/confirm_offset_dual_channel_phase3.py \
PHASE1_SCRIPT=${S2_MAIN}/tune_offset_dual_channel_phase1.py \
EVENT_EVALUATE_SCRIPT=${S3_MAIN}/evaluate_event_ssl_structure.py \
bash ${CROSS_ROBUSTNESS}/run_frozen_headline_learner_cluster_uncertainty.sh
```

## Six-seed null common-target audit

Run after all six predictive evaluation roots and the frozen-headline workflow exist:

```bash
OUTPUTS_ROOT=${OUTPUT_ROOT} \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
CONSTRUCTION_NULL_ROOT=${OUTPUT_ROOT}/stage1_construction_matched_null \
FROZEN_MECHANISM_MANIFEST=${OUTPUT_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json \
NULL_RECOVERY_ROOT=${OUTPUT_ROOT}/six_seed_null_referenced_recovery \
RESULT_ROOT=${OUTPUT_ROOT}/six_seed_null_common_target_audit \
HEADLINE_BOOTSTRAP_ROOT=${OUTPUT_ROOT}/frozen_headline_learner_cluster_uncertainty \
REQUIRE_HEADLINE_BOOTSTRAP=1 \
NULL_EVALUATOR=${CROSS_MAIN}/evaluate_null_referenced_downstream_recovery.py \
AUDIT_SCRIPT=${CROSS_MAIN}/run_six_seed_null_common_target_audit.py \
REPORT_SCRIPT=${CROSS_SCRIPTS}/extract_six_seed_null_common_target_audit_report.py \
STAGE1_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
CONSTRUCTION_NULL_SCRIPT=${CONSTRUCTION}/run_construction_matched_null.py \
MECHANISM_CONFIRM_SCRIPT=${S2_MAIN}/confirm_offset_dual_channel_phase3.py \
MECHANISM_PHASE1_SCRIPT=${S2_MAIN}/tune_offset_dual_channel_phase1.py \
EVENT_SSL_SEED42_ROOT=${OUTPUT_ROOT}/stage4_event_ssl/evaluation_predictive_state \
EVENT_SSL_SEED2026_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_2026/stage4_event_ssl/evaluation_predictive_state \
EVENT_SSL_SEED666_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_666/stage4_event_ssl/evaluation_predictive_state \
EVENT_SSL_SEED606_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_606/stage4_event_ssl/evaluation_predictive_state \
EVENT_SSL_SEED37_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_37/stage4_event_ssl/evaluation_predictive_state \
EVENT_SSL_SEED4669_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_4669/stage4_event_ssl/evaluation_predictive_state \
bash ${CROSS_ROBUSTNESS}/run_six_seed_null_common_target_audit.sh
```

## Common-reference comparisons

Complete the [conditional-baseline workflow](../../s2_minimal_mechanism/robustness/conditional_baselines/README.md) and all six original neural prediction exports first. Use the conditional-baseline Python environment. `FOURTERM_ROOT` supplies the frozen predictors, confirmation predictions matched by learner and interval, and the 2,000 seed-42 paired learner draws. Choose a new destination for each run.

```bash
COMMON_REFERENCE=${CROSS_ROBUSTNESS}/common_reference
FOURTERM_ROOT=/path/to/conditional_baseline_results
COMMON_REFERENCE_ROOT=/path/to/common_reference_results
COMMON_REFERENCE_SMOKE=/path/to/common_reference_smoke

python ${COMMON_REFERENCE}/run_excess.py \
  --history ${OUTPUT_ROOT} --fourterm-root ${FOURTERM_ROOT} \
  --output ${COMMON_REFERENCE_SMOKE} --smoke --workers 8
python ${COMMON_REFERENCE}/run_excess.py \
  --history ${OUTPUT_ROOT} --fourterm-root ${FOURTERM_ROOT} \
  --output ${COMMON_REFERENCE_ROOT} --workers 8
```

The module locates the conditional-baseline code in this repository. The smoke run checks selected saved draws. The full run uses all 2,000 paired learner draws and computes point estimates at six training seeds. See the [module guide](common_reference/README.md) for the output files, support rules and figure commands. Pass the completed `COMMON_REFERENCE_ROOT` to the [statistical reporting workflow](../../s5_supplementary/scripts/statistical_closure/README.md).
