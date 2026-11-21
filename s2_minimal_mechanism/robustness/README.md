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

The final command in the [mechanism result scripts](../scripts/README.md) generates the numerical report.

## Conditional-predictor comparisons

Use the original EdDynamics environment. The comparison requires the empirical and frozen-mechanism outputs, all six original neural prediction exports, and the [learner-cluster uncertainty analysis](../../s4_cross_analysis/robustness/README.md). Exact replay of the confirmation analysis also requires nine external files listed under `b_confirm_coordinate_recovery` in the [source contract](conditional_baselines/source_contract.json). The contract fixes the SHA-256 of each file, including the two manifests for the frozen and confirmed models. Regenerating those manifests can change their hashes because they include execution times and paths.

Set `FOURTERM_ROOT` to a new directory and reuse this exact path in the SSL information-control, common-reference and statistical-reporting commands:

```bash
CONDITIONAL_BASELINES=${REPO_ROOT}/s2_minimal_mechanism/robustness/conditional_baselines
FOURTERM_ROOT=/path/to/conditional_baseline_results

python -m pip install -r ${CONDITIONAL_BASELINES}/requirements.txt
python ${CONDITIONAL_BASELINES}/run_fourterm.py tests --output-root ${OUTPUT_ROOT}
python ${CONDITIONAL_BASELINES}/run_fourterm.py check \
  --output-root ${OUTPUT_ROOT} --run-root ${FOURTERM_ROOT}
python ${CONDITIONAL_BASELINES}/run_fourterm.py smoke \
  --output-root ${OUTPUT_ROOT} --run-root ${FOURTERM_ROOT} \
  --threads 8 --bootstrap-workers 4
python ${CONDITIONAL_BASELINES}/run_fourterm.py all \
  --output-root ${OUTPUT_ROOT} --run-root ${FOURTERM_ROOT} \
  --threads 8 --bootstrap-workers 4
```

Smoke results use the sibling directory `${FOURTERM_ROOT}_smoke`; formal execution requires its matching receipt. The comparison retains 2,000 paired whole-learner draws at seed 42 and all six frozen neural runs. Model definitions, statistical contracts, integrity checks and output tables are described in the [module README](conditional_baselines/README.md).

The selected signed activity drives have no explicit feedback from exposure alignment (Ψ), and the support drive is near saturation. The frozen parameters are in `models/final_models.json`; `confirmation/prediction_audit.json` records the row-wise check that the original Ψ update is contained in the three-constant accounting class. A response order (`M`) near zero indicates signed response evidence close to the item-specific baseline. `confirmation/M2_core_shell_composition.csv`, `confirmation/M2_field_components.csv` and `confirmation/M2_decomposition_contract.json` report frozen-region composition and the additive update components, including dilution and numerical residuals. These are conditional descriptions, not causal contribution percentages.
