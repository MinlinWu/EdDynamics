# Minimal-mechanism result scripts

Run the figure producer before the primary extractor:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
SCRIPT_ROOT=${REPO_ROOT}/s2_minimal_mechanism/scripts

python ${SCRIPT_ROOT}/publication_minimal_mechanism_figures.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --phase2-root ${OUTPUT_ROOT}/stage2_phase2_freeze \
  --phase3-root ${OUTPUT_ROOT}/stage2_phase3_confirm \
  --minimality-root ${OUTPUT_ROOT}/stage2_phase1_unified_minimality

python ${SCRIPT_ROOT}/extract_minimal_mechanism_publication_statistics.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --phase1-root ${OUTPUT_ROOT}/stage2_phase1 \
  --minimality-root ${OUTPUT_ROOT}/stage2_phase1_unified_minimality \
  --phase2-root ${OUTPUT_ROOT}/stage2_phase2_freeze \
  --phase3-root ${OUTPUT_ROOT}/stage2_phase3_confirm
```

After score-contract robustness completes:

```bash
python ${SCRIPT_ROOT}/extract_mechanism_score_contract_numeric_report.py \
  --score-root ${OUTPUT_ROOT}/stage2_phase1_score_contract_robustness
```
