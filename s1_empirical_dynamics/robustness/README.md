# Empirical robustness

Set shared paths:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
PREPROCESSED_ROOT=/path/to/preprocessed_data
S1_MAIN=${REPO_ROOT}/s1_empirical_dynamics/main
S1_ROBUSTNESS=${REPO_ROOT}/s1_empirical_dynamics/robustness
S2_MAIN=${REPO_ROOT}/s2_minimal_mechanism/main
```

## Construction-matched null

Run after `${OUTPUT_ROOT}/stage1` exists:

```bash
python ${S1_ROBUSTNESS}/construction_null/run_construction_matched_null.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --stage1-script ${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
  --output-root ${OUTPUT_ROOT}/stage1_construction_matched_null \
  --analysis-split A_val --replicates 100 --seed 42

python ${S1_ROBUSTNESS}/construction_null/run_construction_matched_null.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --stage1-script ${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
  --output-root ${OUTPUT_ROOT}/stage1_construction_matched_null_confirm \
  --analysis-split B_confirm --confirmation-output-only --replicates 100 --seed 42
```

## Empirical sensitivity

```bash
SOURCE_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
EDNET_KT4_DATA_ROOT=${PREPROCESSED_ROOT} \
BASELINE_OUTPUT_ROOT=${OUTPUT_ROOT} \
SENSITIVITY_ROOT=${OUTPUT_ROOT}/stage1_sensitivity \
MAX_PARALLEL=5 \
bash ${S1_ROBUSTNESS}/sensitivity/run_stage1_empirical_sensitivity.sh

python ${S1_ROBUSTNESS}/sensitivity/extract_empirical_coordinate_sensitivity_statistics.py \
  --baseline-root ${OUTPUT_ROOT} \
  --sensitivity-root ${OUTPUT_ROOT}/stage1_sensitivity \
  --output-dir ${OUTPUT_ROOT}/stage1_sensitivity/summary
```

## Aggregate robustness

Run after the sensitivity report, frozen mechanism manifest, predictive-state evaluation, time-shuffle evaluation, tag/support evaluation and macro-sufficiency evaluation exist:

```bash
OUTPUT_BASE=${OUTPUT_ROOT} \
ROBUSTNESS_ROOT=${OUTPUT_ROOT}/supplementary_robustness \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
PHASE3_ROOT=${OUTPUT_ROOT}/stage2_phase3_confirm \
FROZEN_MANIFEST=${OUTPUT_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json \
MAIN_ROOT=${OUTPUT_ROOT}/stage4_event_ssl/evaluation_predictive_state \
TIME_SHUFFLE_ROOT=${OUTPUT_ROOT}/stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs \
TAG_SUPPORT_ROOT=${OUTPUT_ROOT}/stage4_event_ssl_tag_support_randomized_control/evaluation \
MACRO_ROOT=${OUTPUT_ROOT}/stage5_macro_sufficiency/evaluation \
COORDINATE_SUMMARY_ROOT=${OUTPUT_ROOT}/stage1_sensitivity/summary \
STAGE1_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
PHASE3_SCRIPT=${S2_MAIN}/confirm_offset_dual_channel_phase3.py \
SUMMARY_SCRIPT=${S1_ROBUSTNESS}/aggregate/summarize_supplementary_robustness.py \
REPORT_EXTRACTOR=${S1_ROBUSTNESS}/aggregate/extract_supplementary_robustness_statistics.py \
MAX_PARALLEL=8 \
bash ${S1_ROBUSTNESS}/aggregate/run_supplementary_robustness.sh
```

## Kinetic robustness

```bash
OUTPUT_BASE=${OUTPUT_ROOT} \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
CONSTRUCTION_NULL_ROOT=${OUTPUT_ROOT}/stage1_construction_matched_null \
RESULT_ROOT=${OUTPUT_ROOT}/stage1_kinetic_robustness \
STAGE1_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
CONSTRUCTION_NULL_SCRIPT=${S1_ROBUSTNESS}/construction_null/run_construction_matched_null.py \
PARTITION_K_VALUES=4,5,6,7,8 \
BOOTSTRAP_REPLICATES=1000 \
NULL_REPLICATES=100 \
MAX_PARALLEL=8 \
bash ${S1_ROBUSTNESS}/kinetic/run_kinetic_robustness.sh

python ${S1_ROBUSTNESS}/kinetic/extract_kinetic_robustness_statistics.py \
  --partition-root ${OUTPUT_ROOT}/stage1_kinetic_robustness/partition_cluster \
  --recursive-null-root ${OUTPUT_ROOT}/stage1_kinetic_robustness/recursive_null \
  --output-dir ${OUTPUT_ROOT}/stage1_kinetic_robustness/summary
```

## Excess-reliability analysis

```bash
OUTPUTS_ROOT=${OUTPUT_ROOT} \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
CONSTRUCTION_NULL_ROOT=${OUTPUT_ROOT}/stage1_construction_matched_null \
CONSTRUCTION_NULL_CONFIRM_ROOT=${OUTPUT_ROOT}/stage1_construction_matched_null_confirm \
ANALYSIS_ROOT=${OUTPUT_ROOT}/empirical_excess_reliability_soft_core \
STAGE1_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
CONSTRUCTION_NULL_SCRIPT=${S1_ROBUSTNESS}/construction_null/run_construction_matched_null.py \
bash ${S1_ROBUSTNESS}/excess_reliability/run_empirical_excess_reliability_soft_core.sh
```

## Semantic-specificity control

```bash
OUTPUT_BASE=${OUTPUT_ROOT} \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
FORMAL_NULL_ROOT=${OUTPUT_ROOT}/stage1_construction_matched_null \
FORMAL_CONFIRM_NULL_ROOT=${OUTPUT_ROOT}/stage1_construction_matched_null_confirm \
RESULT_ROOT=${OUTPUT_ROOT}/semantic_specificity_control \
STAGE1_SCRIPT=${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
CONSTRUCTION_NULL_SCRIPT=${S1_ROBUSTNESS}/construction_null/run_construction_matched_null.py \
bash ${S1_ROBUSTNESS}/semantic_specificity/run_semantic_specificity_control.sh
```
