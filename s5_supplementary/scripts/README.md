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

## Statistical reporting from preserved outputs

Use the saved field and transition uncertainty summaries, residence-time bootstrap draws, and objective-control and geometry evaluations for all six seeds. Set `FOURTERM_ROOT` and `COMMON_REFERENCE_ROOT` to the completed [conditional-baseline](../../s2_minimal_mechanism/robustness/conditional_baselines/README.md) and [common-reference](../../s4_cross_analysis/robustness/common_reference/README.md) result directories. Choose a new report destination.

```bash
STATISTICAL_REPORTING=${SCRIPT_ROOT}/statistical_closure
FOURTERM_ROOT=/path/to/conditional_baseline_results
COMMON_REFERENCE_ROOT=/path/to/common_reference_results
STATISTICAL_REPORT_ROOT=/path/to/statistical_report

python -m pip install -r ${STATISTICAL_REPORTING}/requirements.txt
python ${STATISTICAL_REPORTING}/build_statistical_reports.py \
  --output-root ${OUTPUT_ROOT} \
  --mechanism-root ${FOURTERM_ROOT} \
  --excess-root ${COMMON_REFERENCE_ROOT} \
  --destination ${STATISTICAL_REPORT_ROOT}
```

The default command collects the original field and transition intervals (Supplementary Table 11), residence-time estimates (Table 4) and matched-seed objective-control and geometry results (Tables 7–8). It also collects the paired contrasts from the conditional-baseline and common-reference analyses (Note 8 and Table 13). The [output guide](statistical_closure/README.md) lists the files, estimands and source hashes for each section.
