# Minimal-mechanism robustness

Run after `${OUTPUT_ROOT}/stage1` and the formal family-ablation output exist:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
FORMAL_PHASE1_SCRIPT=${REPO_ROOT}/s2_minimal_mechanism/main/run_minimal_mechanism_family_ablation.py \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
FORMAL_PHASE1_ROOT=${OUTPUT_ROOT}/stage2_phase1_unified_minimality \
SCORE_ROOT=${OUTPUT_ROOT}/stage2_phase1_score_contract_robustness \
NUMBA_THREADS=32 \
bash ${REPO_ROOT}/s2_minimal_mechanism/robustness/run_mechanism_score_contract_robustness.sh
```

The corresponding numeric report is the final command in `../scripts/README.md`.
