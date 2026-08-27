# Event-SSL robustness

Primary controls are documented in `../README.md`; this file covers additional-seed and audit workflows.

Set shared paths:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
EVENT_MAIN=${REPO_ROOT}/s3_event_ssl/main
EVENT_ROBUSTNESS=${REPO_ROOT}/s3_event_ssl/robustness
CONTROLS=${EVENT_MAIN}/controls
STRUCTURAL_ANALYSES=${EVENT_MAIN}/structural_analyses
```

## Additional seeds

Complete `random_seed_workflows/README.md` before the following two workflows.

## State-only closure audit

The `stage5_` prefix in the referenced training-script filename denotes its actual execution order.

```bash
STATE_ONLY=${EVENT_ROBUSTNESS}/state_only
OUTPUTS_ROOT=${OUTPUT_ROOT} \
STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
RESULT_ROOT=${OUTPUT_ROOT}/state_only_closure_audit \
STAGE5_SEED42_ROOT=${OUTPUT_ROOT}/stage5_macro_sufficiency \
STAGE5_SEED2026_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_2026/stage5_macro_sufficiency \
STAGE5_SEED666_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_666/stage5_macro_sufficiency \
STAGE5_SEED606_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_606/stage5_macro_sufficiency \
STAGE5_SEED37_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_37/stage5_macro_sufficiency \
STAGE5_SEED4669_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_4669/stage5_macro_sufficiency \
ANALYSIS_SCRIPT=${STATE_ONLY}/run_state_only_closure_audit.py \
REPORT_SCRIPT=${STATE_ONLY}/extract_state_only_closure_audit_report.py \
TRAIN_SCRIPT=${EVENT_MAIN}/train_event_ssl.py \
EVALUATE_SCRIPT=${EVENT_MAIN}/evaluate_event_ssl_structure.py \
STAGE5_TRAIN_SCRIPT=${STRUCTURAL_ANALYSES}/stage5_macro_sufficiency_train.py \
bash ${STATE_ONLY}/run_state_only_closure_audit.sh
```

## Objective-control hidden geometry

```bash
OBJECTIVE=${EVENT_ROBUSTNESS}/objective_control
OUTPUTS_ROOT=${OUTPUT_ROOT} \
RESULT_ROOT=${OUTPUT_ROOT}/stage5_objective_control_hidden_geometry \
ANALYSIS_SCRIPT=${OBJECTIVE}/run_objective_control_hidden_geometry.py \
REPORT_SCRIPT=${OBJECTIVE}/extract_objective_control_hidden_geometry_report.py \
TRAIN_SCRIPT=${EVENT_MAIN}/train_event_ssl.py \
EVALUATE_SCRIPT=${EVENT_MAIN}/evaluate_event_ssl_structure.py \
TASK_SCRIPT=${CONTROLS}/control_task_only.py \
SEEDS=42,2026,666,606,37,4669 \
SEED42_ROOT=${OUTPUT_ROOT} \
SEED2026_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_2026 \
SEED666_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_666 \
SEED606_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_606 \
SEED37_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_37 \
SEED4669_ROOT=${OUTPUT_ROOT}/random_seed_experiments/seed_4669 \
bash ${OBJECTIVE}/run_objective_control_hidden_geometry.sh
```
