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

## SSL information controls

Use the prepared memory maps, frozen six-state partition, full and record-shuffled checkpoints and their evaluation records for each of the six seeds. Set `FOURTERM_ROOT` to the completed [conditional-baseline results](../../s2_minimal_mechanism/robustness/conditional_baselines/README.md), which supply predictions for each matched interval.

Use one Python environment for CPU and GPU stages. Configure `slurm_s3.sbatch` before the smoke test because the code hash includes its resource directives. The supplied script requests one node, one task, five GPUs, 40 CPUs, 320 GB of memory and eight hours in the `gpu` partition. Its controller reads individually listed NVIDIA device files from `/etc/slurm/gres.conf`. See the [cluster requirements](information_controls/README.md#run-the-six-seed-design) when adapting the script.

```bash
SSL_INFORMATION=${EVENT_ROBUSTNESS}/information_controls
FOURTERM_ROOT=/path/to/conditional_baseline_results
export PYTHON_BIN=/path/to/environment/bin/python
export SSL_CODE_ROOT=${SSL_INFORMATION}

${PYTHON_BIN} -m pip install -r ${SSL_INFORMATION}/requirements.txt
${PYTHON_BIN} ${SSL_INFORMATION}/run_ssl.py selftest
${PYTHON_BIN} ${SSL_INFORMATION}/run_ssl.py check \
  --output-root ${OUTPUT_ROOT} --fourterm-root ${FOURTERM_ROOT}
${PYTHON_BIN} ${SSL_INFORMATION}/run_ssl.py smoke \
  --output-root ${OUTPUT_ROOT} --fourterm-root ${FOURTERM_ROOT}
${PYTHON_BIN} ${SSL_INFORMATION}/run_ssl.py cpu \
  --output-root ${OUTPUT_ROOT} --fourterm-root ${FOURTERM_ROOT}

(
  cd ${SSL_INFORMATION}
  sbatch slurm_s3.sbatch --history-root ${OUTPUT_ROOT}
)
```

Run the Slurm submission only after the CPU command succeeds. The CPU stages prepare the source contract, replay full and record-shuffled checkpoints at six seeds, fit three pairs of information heads, and complete the future-only seed-42 pair with 2,000 paired learner draws. The Slurm job then fits seeds 2026, 666, 606, 37 and 4669 on five separate GPUs, evaluates their paired conditions on CPU, and verifies all 92 stages before writing `${OUTPUT_ROOT}/ssl_information_controls/final/COMPLETE.json`.

The six-seed workflow uses `cpu` followed by Slurm. The GPU controller reads the C3 reference recorded during CPU preparation and calls `finalize_six.py` to assemble the six-seed report. The `all` and `report` commands belong to the separate three-seed CPU design. The standalone `s3` command runs the seed-42 CPU information-removal block. The [analysis guide](information_controls/README.md) maps each comparison in Supplementary Table 14 to its measurements, uncertainty and output files.
