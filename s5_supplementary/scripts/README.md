# Supplementary result scripts

Set paths:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
S1_MAIN=${REPO_ROOT}/s1_empirical_dynamics/main
SCRIPT_ROOT=${REPO_ROOT}/s5_supplementary/scripts
```

## Empirical extraction

```bash
python ${SCRIPT_ROOT}/empirical/extract_empirical_effective_dynamics_supplementary_statistics.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --stage1-script ${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py \
  --null-validation-root ${OUTPUT_ROOT}/stage1_construction_matched_null \
  --null-confirmation-root ${OUTPUT_ROOT}/stage1_construction_matched_null_confirm \
  --output-root ${OUTPUT_ROOT}/stage1_empirical_supplementary \
  --minimum-null-replicates 100 \
  --sensitivity-splits A_val B_confirm
```

## Minimal-mechanism extraction

```bash
python ${SCRIPT_ROOT}/minimal_mechanism/extract_minimal_mechanism_supplementary_statistics.py \
  --output-base ${OUTPUT_ROOT} \
  --output-root ${OUTPUT_ROOT}/stage2_phase3_confirm/supplementary_minimal_mechanism \
  --confirm-split B_confirm \
  --strict
```

## Event-SSL extraction

```bash
EVENT_SUPP_ROOT=${OUTPUT_ROOT}/stage4_event_ssl/supplementary_event_ssl
python ${SCRIPT_ROOT}/event_ssl/extract_event_ssl_supplementary_statistics.py \
  --output-base ${OUTPUT_ROOT} \
  --output-root ${EVENT_SUPP_ROOT} \
  --strict

python ${SCRIPT_ROOT}/event_ssl/publication_event_ssl_supplementary_comparison.py \
  --comparison-root ${OUTPUT_ROOT}/cross_stage_mechanism_event_ssl_comparison \
  --output-root ${EVENT_SUPP_ROOT} \
  --split B_confirm \
  --max-density-rows 0
```
