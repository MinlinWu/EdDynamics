# Empirical result scripts

Run primary extraction before the figure:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
SCRIPT_ROOT=${REPO_ROOT}/s1_empirical_dynamics/scripts

python ${SCRIPT_ROOT}/extract_empirical_effective_dynamics_publication_statistics.py \
  --stage1-root ${OUTPUT_ROOT}/stage1
python ${SCRIPT_ROOT}/publication_empirical_effective_dynamics.py \
  --stage1-root ${OUTPUT_ROOT}/stage1
```

After both construction-matched null runs:

```bash
python ${SCRIPT_ROOT}/extract_construction_matched_null_numeric_report.py \
  --validation-root ${OUTPUT_ROOT}/stage1_construction_matched_null \
  --confirmation-root ${OUTPUT_ROOT}/stage1_construction_matched_null_confirm \
  --output-root ${OUTPUT_ROOT}/stage1_construction_matched_null/numerical_report \
  --minimum-replicates 100 \
  --maximum-weak-fallback-fraction 0.01 \
  --alpha 0.05
```
