#!/usr/bin/env bash
set -euo pipefail

# Post hoc, development-only mechanism score-contract robustness.
# The component/Pareto audit and the equal-primary-component re-optimisation run
# sequentially by default because each materialises the full A_train/A_val
# mechanism panels. Phase 2 and Phase 3 are never invoked.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

FORMAL_PHASE1_SCRIPT="${FORMAL_PHASE1_SCRIPT:-${SCRIPT_DIR}/run_minimal_mechanism_family_ablation.py}"
if [[ ! -f "${FORMAL_PHASE1_SCRIPT}" && -f "${SCRIPT_DIR}/run_minimal_mechanism_family_ablation.py" ]]; then
  FORMAL_PHASE1_SCRIPT="${SCRIPT_DIR}/run_minimal_mechanism_family_ablation.py"
fi
STAGE1_ROOT="${STAGE1_ROOT:-/data/datasets/KT4/outputs_KT4/stage1}"
FORMAL_PHASE1_ROOT="${FORMAL_PHASE1_ROOT:-/data/datasets/KT4/outputs_KT4/stage2_phase1_unified_minimality}"
SCORE_ROOT="${SCORE_ROOT:-/data/datasets/KT4/outputs_KT4/stage2_phase1_score_contract_robustness}"
FORMAL_AUDIT_ROOT="${FORMAL_AUDIT_ROOT:-${SCORE_ROOT}/formal_audit}"
EQUAL_PRIMARY_RERUN_ROOT="${EQUAL_PRIMARY_RERUN_ROOT:-${SCORE_ROOT}/equal_primary_rerun}"
NUMBA_THREADS="${NUMBA_THREADS:-32}"
RESUME_EQUAL_RERUN="${RESUME_EQUAL_RERUN:-0}"

AUDIT_SCRIPT="${SCRIPT_DIR}/audit_mechanism_score_contract_pareto.py"
RERUN_SCRIPT="${SCRIPT_DIR}/run_minimal_mechanism_score_contract_rerun.py"
LOG_ROOT="${SCORE_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

# Pin all reviewed formal Phase-1 defaults. The audit fails if an inherited
# environment variable changes a source-level search or scoring contract.
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_THREADS}"

export MECH_MINIMALITY_RANDOM_STATE=42
export MECH_MINIMALITY_USE_NUMBA=1
export MECH_MINIMALITY_GRID_PROFILE=publication
export MECH_MINIMALITY_SCREENING_TRAIN_USERS=20000
export MECH_MINIMALITY_SCREENING_MAX_CANDIDATES=96
export MECH_MINIMALITY_FULL_TRAIN_TOP_K=16
export MECH_MINIMALITY_VAL_SHORTLIST_K=8
export MECH_MINIMALITY_LOCAL_REFINE_MAX_EVALS=48
export MECH_MINIMALITY_DELETION_EXHAUSTIVE_MAX_COMBINATIONS=5000
export MECH_MINIMALITY_DELETION_FULL_TRAIN_TOP_K=32
export MECH_MINIMALITY_DELETION_VAL_SHORTLIST_K=16
export MECH_MINIMALITY_DELETION_LOCAL_REFINE_MAX_EVALS=96
export MECH_MINIMALITY_DELETION_REFINE_STARTS=5
export MECH_MINIMALITY_BOOTSTRAP_REPS=300
export MECH_MINIMALITY_EQ_MARGIN=0.02
# This supplementary experiment fixes the formal PE margin; it does not
# repeat the existing 0.010--0.030 margin-sensitivity analysis.
export MECH_MINIMALITY_MARGIN_SENSITIVITY=0.020
export MECH_MINIMALITY_BOOTSTRAP_ENGINE=optimized
export MECH_MINIMALITY_VERIFY_BOOTSTRAP=1
export MECH_MINIMALITY_VERIFY_BOOTSTRAP_REPS=2
export MECH_MINIMALITY_DECISION_BOOTSTRAP_SEED_OFFSET=777

export MECH_PHASE1_DISTRIBUTION_LOSS_MAX_ROWS=200000
export MECH_PHASE1_SIGNED_GAIN_QUANTILE=0.75
export MECH_PHASE1_SANITY_PENALTY_WEIGHT=0.25
export MECH_PHASE1_IDENTITY_REG_WEIGHT=0.05
export MECH_MINIMALITY_LAMBDAR=0.46
export MECH_MINIMALITY_LAMBDAA=1.10
export MECH_MINIMALITY_LAMBDAI=0.85
export MECH_MINIMALITY_DELTA_S_SATURATION_TOL=0.002
export MECH_MINIMALITY_DELTA_S_PLATEAU_MAX_NEXT_PSI=0.0001
export MECH_MINIMALITY_DELTA_S_PLATEAU_MAX_SCORE_DIFF=0.00001
export MECH_MINIMALITY_DELTA_S_PLATEAU_MAX_OBJECTIVE_DIFF=0.00001

export EDNET_STAGE1_TAU_RESPONSE_DAYS=10.0
export EDNET_STAGE1_TAU_ACTIVITY_DAYS=10.0
export EDNET_STAGE1_EVIDENCE_MATURITY_SCALE=20.0
export EDNET_STAGE1_SIGNED_GRID_N=41
export EDNET_STAGE1_MIN_DRIFT_BIN_COUNT=30
export EDNET_STAGE1_KMEANS_N_INIT=20
export EDNET_STAGE1_KMEANS_FIT_MAX_ROWS=500000
export EDNET_STAGE1_RANDOM_STATE=42

for path in "${FORMAL_PHASE1_SCRIPT}" "${AUDIT_SCRIPT}" "${RERUN_SCRIPT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required script not found: ${path}" >&2
    exit 1
  fi
done
for path in \
  "${FORMAL_PHASE1_ROOT}/metadata/minimality_experiment_manifest.json" \
  "${FORMAL_PHASE1_ROOT}/metadata/phase1_minimal_mechanism_handoff.json" \
  "${FORMAL_PHASE1_ROOT}/tables/model_family_results.csv" \
  "${FORMAL_PHASE1_ROOT}/tables/model_family_bootstrap_scores.csv.gz"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required formal Phase-1 output not found: ${path}" >&2
    exit 1
  fi
done

printf '%s\n' \
  "Formal Phase-1 script: ${FORMAL_PHASE1_SCRIPT}" \
  "Stage-1 root: ${STAGE1_ROOT}" \
  "Formal Phase-1 outputs: ${FORMAL_PHASE1_ROOT}" \
  "Robustness root: ${SCORE_ROOT}" \
  "B_confirm policy: not read or used"

"${PYTHON_BIN}" -m py_compile "${AUDIT_SCRIPT}" "${RERUN_SCRIPT}"
"${PYTHON_BIN}" "${AUDIT_SCRIPT}" self-test
"${PYTHON_BIN}" "${RERUN_SCRIPT}" \
  --formal-script "${FORMAL_PHASE1_SCRIPT}" \
  --objective-contract equal_primary \
  --self-test \
  --no-numba

# Stage 1: exact reconstruction of the archived formal bootstrap score,
# frozen-fit component reweighting, and six-objective complexity-Pareto audit.
"${PYTHON_BIN}" "${AUDIT_SCRIPT}" formal-audit \
  --formal-script "${FORMAL_PHASE1_SCRIPT}" \
  --formal-root "${FORMAL_PHASE1_ROOT}" \
  --stage1-root "${STAGE1_ROOT}" \
  --output-root "${FORMAL_AUDIT_ROOT}" \
  --reconstruction-tolerance 1e-10 \
  --pareto-tolerance 1e-12 \
  --overwrite \
  2>&1 | tee "${LOG_ROOT}/01_formal_component_pareto_audit.log"

# Stage 2: one complete equal-primary-component Phase-1 family re-optimisation. The
# publication grid, search budgets, bootstrap bank, scalar-deletion fixed
# point and PE margin remain formal. The output is sensitivity-only.
RERUN_MODE=(--overwrite)
if [[ "${RESUME_EQUAL_RERUN}" == "1" ]]; then
  RERUN_MODE=(--resume)
fi

"${PYTHON_BIN}" "${RERUN_SCRIPT}" \
  --formal-script "${FORMAL_PHASE1_SCRIPT}" \
  --stage1-root "${STAGE1_ROOT}" \
  --output-root "${EQUAL_PRIMARY_RERUN_ROOT}" \
  --objective-contract equal_primary \
  --grid-profile publication \
  --random-state 42 \
  --screening-train-users 20000 \
  --screening-max-candidates 96 \
  --full-train-top-k 16 \
  --val-shortlist-k 8 \
  --local-refine-max-evals 48 \
  --deletion-exhaustive-max-combinations 5000 \
  --deletion-full-train-top-k 32 \
  --deletion-val-shortlist-k 16 \
  --deletion-local-refine-max-evals 96 \
  --deletion-refine-starts 5 \
  --bootstrap-reps 300 \
  --equivalence-margin 0.02 \
  --bootstrap-engine optimized \
  --verify-optimized-bootstrap \
  --verify-bootstrap-reps 2 \
  --numba-threads "${NUMBA_THREADS}" \
  "${RERUN_MODE[@]}" \
  2>&1 | tee "${LOG_ROOT}/02_equal_primary_full_reoptimization.log"

# Stage 3: verify that only the primary component weights changed, validate
# all isolation guardrails, and combine the frozen-fit and full-rerun reports.
"${PYTHON_BIN}" "${AUDIT_SCRIPT}" finalize \
  --score-root "${SCORE_ROOT}" \
  --formal-audit-root "${FORMAL_AUDIT_ROOT}" \
  --equal-rerun-root "${EQUAL_PRIMARY_RERUN_ROOT}" \
  2>&1 | tee "${LOG_ROOT}/03_finalize_score_contract_robustness.log"

printf '\nCompleted. Primary outputs:\n'
printf '  %s\n' \
  "${SCORE_ROOT}/tables/score_contract_robustness_summary.csv" \
  "${SCORE_ROOT}/metadata/score_contract_robustness_summary.json" \
  "${SCORE_ROOT}/metadata/score_contract_robustness_manifest.json" \
  "${SCORE_ROOT}/metadata/equal_primary_configuration_audit.json"
