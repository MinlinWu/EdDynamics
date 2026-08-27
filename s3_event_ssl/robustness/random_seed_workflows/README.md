# Additional Event-SSL seeds

Run after the complete primary seed-42 Event-SSL workflow:

The `stage5_` prefix in the bundled snapshot filenames denotes their actual downstream position after Event-SSL training and evaluation.

```bash
REPO_ROOT=/path/to/EdDynamics
SEED_WORKFLOWS=${REPO_ROOT}/s3_event_ssl/robustness/random_seed_workflows
export STAGE1_ROOT=/path/to/outputs/stage1
export OUTPUT_BASE=/path/to/outputs/random_seed_experiments
cd ${SEED_WORKFLOWS}

for SEED in 2026 666 606 37 4669; do
  (
    cd "seed_${SEED}"
    TRAIN_GPU_LIST=0 EVAL_GPU_LIST=0 POST_GPU_LIST=0 bash run_all.sh
  )
done
```

To run or resume one seed, enter its directory and run:

```bash
REPO_ROOT=/path/to/EdDynamics
SEED_WORKFLOWS=${REPO_ROOT}/s3_event_ssl/robustness/random_seed_workflows
export STAGE1_ROOT=/path/to/outputs/stage1
export OUTPUT_BASE=/path/to/outputs/random_seed_experiments
SEED=2026
cd ${SEED_WORKFLOWS}/seed_${SEED}
TRAIN_GPU_LIST=0 EVAL_GPU_LIST=0 POST_GPU_LIST=0 bash run_all.sh
```

For manual execution inside one seed directory:

```bash
REPO_ROOT=/path/to/EdDynamics
SEED_WORKFLOWS=${REPO_ROOT}/s3_event_ssl/robustness/random_seed_workflows
export STAGE1_ROOT=/path/to/outputs/stage1
export OUTPUT_BASE=/path/to/outputs/random_seed_experiments
SEED=2026
cd ${SEED_WORKFLOWS}/seed_${SEED}
bash prepare.sh
TRAIN_GPU_LIST=0 bash launch_train_pair.sh
EVAL_GPU_LIST=0 bash launch_evaluation_pair.sh
POST_GPU_LIST=0 bash launch_post_stage.sh
```

Do not add another seed-42 bundle run. The primary seed-42 outputs are combined with these five directories during result extraction.
