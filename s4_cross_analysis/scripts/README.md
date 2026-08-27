# Cross-analysis result scripts

After null-referenced downstream recovery:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
SCRIPT_ROOT=${REPO_ROOT}/s4_cross_analysis/scripts

python ${SCRIPT_ROOT}/extract_null_referenced_downstream_recovery_supplementary_statistics.py \
  --analysis-root ${OUTPUT_ROOT}/null_referenced_downstream_recovery
```

Run the comparison extractor before Figure 7:

```bash
python ${SCRIPT_ROOT}/extract_mechanism_event_ssl_publication_statistics.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --phase2-root ${OUTPUT_ROOT}/stage2_phase2_freeze \
  --phase3-root ${OUTPUT_ROOT}/stage2_phase3_confirm \
  --event-ssl-eval-root ${OUTPUT_ROOT}/stage4_event_ssl/evaluation_predictive_state \
  --output-root ${OUTPUT_ROOT}/cross_stage_mechanism_event_ssl_comparison \
  --split B_confirm --max-rows 0

python ${SCRIPT_ROOT}/publication_event_ssl_figure7.py \
  --macro-root ${OUTPUT_ROOT}/stage5_macro_sufficiency/evaluation \
  --geometry-root ${OUTPUT_ROOT}/stage5_representation_geometry/evaluation \
  --cross-root ${OUTPUT_ROOT}/cross_stage_mechanism_event_ssl_comparison \
  --output-root ${OUTPUT_ROOT}/stage4_event_ssl/figures_publication_event_ssl \
  --split B_confirm
```

The frozen-headline and six-seed wrappers invoke their respective report extractors directly; do not run those extractors a second time.
