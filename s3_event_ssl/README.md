# Event-SSL

Run after `${OUTPUT_ROOT}/stage1` has been generated. Set paths:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
EVENT_MAIN=${REPO_ROOT}/s3_event_ssl/main
CONTROLS=${EVENT_MAIN}/controls
STRUCTURAL_ANALYSES=${EVENT_MAIN}/structural_analyses
STAGE1_ROOT=${OUTPUT_ROOT}/stage1
MAIN_ROOT=${OUTPUT_ROOT}/stage4_event_ssl
INPUT_ROOT=${MAIN_ROOT}/prepared_inputs
```

Prepare inputs, then train and evaluate both primary models:

```bash
python ${EVENT_MAIN}/prepare_event_ssl_inputs.py \
  --stage1-root ${STAGE1_ROOT} \
  --output-root ${MAIN_ROOT} \
  --hash-buckets 32768 --seed 42

python ${EVENT_MAIN}/train_event_ssl.py \
  --input-root ${INPUT_ROOT} \
  --output-root ${MAIN_ROOT}/models/predictive_state \
  --model-kind predictive_state \
  --epochs 8 --batch-size 192 --seq-len 256 --stride 128

python ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --input-root ${INPUT_ROOT} \
  --checkpoint ${MAIN_ROOT}/models/predictive_state/best_model.pt \
  --output-root ${MAIN_ROOT}/evaluation_predictive_state \
  --stage1-root ${STAGE1_ROOT}

python ${EVENT_MAIN}/train_event_ssl.py \
  --input-root ${INPUT_ROOT} \
  --output-root ${MAIN_ROOT}/models/pure_ssl \
  --model-kind pure_ssl \
  --epochs 8 --batch-size 192 --seq-len 256 --stride 128

python ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --input-root ${INPUT_ROOT} \
  --checkpoint ${MAIN_ROOT}/models/pure_ssl/best_model.pt \
  --output-root ${MAIN_ROOT}/evaluation_pure_ssl_probe
```

Run the three primary controls:

```bash
# Task-only control
python ${CONTROLS}/control_task_only.py train \
  --input-root ${INPUT_ROOT} \
  --output-root ${MAIN_ROOT}/controls/task_only/model \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --epochs 8 --batch-size 192 --seq-len 256 --stride 128 --min-seq-len 3 \
  --warmup-steps 8 --hidden-dim 320 --input-dim 224 --num-layers 2 --dropout 0.10 \
  --num-workers 8 --categorical-emb-dim 16 --future-steps 1,2,4 --delta-scale 0.50 \
  --seed 42 --amp-dtype bf16

python ${CONTROLS}/control_task_only.py evaluate \
  --input-root ${INPUT_ROOT} \
  --checkpoint ${MAIN_ROOT}/controls/task_only/model/best_model.pt \
  --output-root ${MAIN_ROOT}/controls/task_only/evaluation \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --evaluate-script ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --stage1-root ${STAGE1_ROOT} --splits A_val B_confirm --chunk-len 512 \
  --probe-max-rows 300000 --seed 42

# Time-shuffle control
TIME_ROOT=${OUTPUT_ROOT}/stage4_event_ssl_time_shuffle_control
python ${CONTROLS}/control_time_shuffle.py prepare \
  --prepare-script ${EVENT_MAIN}/prepare_event_ssl_inputs.py \
  --stage1-root ${STAGE1_ROOT} --output-root ${TIME_ROOT} --hash-buckets 32768 --seed 42
python ${CONTROLS}/control_time_shuffle.py train \
  --input-root ${TIME_ROOT}/prepared_inputs --output-root ${TIME_ROOT}/model \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --epochs 8 --batch-size 192 --seq-len 256 --stride 128 --min-seq-len 3 \
  --warmup-steps 8 --hidden-dim 320 --input-dim 224 --num-layers 2 --dropout 0.10 \
  --num-workers 8 --future-steps 1,2,4 --lambda-future 1.0 --lambda-state 0.5 \
  --lambda-closure 0.5 --delta-scale 0.50 --categorical-emb-dim 16 --seed 42 --amp-dtype bf16
python ${CONTROLS}/control_time_shuffle.py evaluate \
  --input-root ${INPUT_ROOT} --train-input-root ${TIME_ROOT}/prepared_inputs \
  --checkpoint ${TIME_ROOT}/model/best_model.pt \
  --output-root ${TIME_ROOT}/evaluation_on_ordered_inputs \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --evaluate-script ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --stage1-root ${STAGE1_ROOT} --splits A_val B_confirm --chunk-len 512 --seed 42

# Tag/support-randomisation control
TAG_ROOT=${OUTPUT_ROOT}/stage4_event_ssl_tag_support_randomized_control
python ${CONTROLS}/control_tag_support_randomization.py prepare \
  --prepare-script ${EVENT_MAIN}/prepare_event_ssl_inputs.py \
  --stage1-root ${STAGE1_ROOT} --output-root ${TAG_ROOT} --hash-buckets 32768
python ${CONTROLS}/control_tag_support_randomization.py train \
  --input-root ${TAG_ROOT}/prepared_inputs --output-root ${TAG_ROOT}/model \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --epochs 8 --batch-size 192 --seq-len 256 --stride 128 --min-seq-len 3 \
  --warmup-steps 8 --hidden-dim 320 --input-dim 224 --num-layers 2 --dropout 0.10 \
  --num-workers 8 --future-steps 1,2,4 --lambda-future 1.0 --lambda-state 0.5 \
  --lambda-closure 0.5 --delta-scale 0.50 --categorical-emb-dim 16 --seed 42 --amp-dtype bf16
python ${CONTROLS}/control_tag_support_randomization.py evaluate \
  --input-root ${TAG_ROOT}/prepared_inputs \
  --checkpoint ${TAG_ROOT}/model/best_model.pt \
  --output-root ${TAG_ROOT}/evaluation \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --evaluate-script ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --stage1-root ${STAGE1_ROOT} --splits A_val B_confirm --chunk-len 512 --seed 42
```

The `stage5_` prefix in the following filenames denotes their actual downstream position after the main Event-SSL training and evaluation workflow.

Run macro sufficiency and representation geometry after the predictive-state evaluation:

```bash
CHECKPOINT=${MAIN_ROOT}/models/predictive_state/best_model.pt
MACRO_ROOT=${OUTPUT_ROOT}/stage5_macro_sufficiency
python ${STRUCTURAL_ANALYSES}/stage5_macro_sufficiency_train.py \
  --input-root ${INPUT_ROOT} --checkpoint ${CHECKPOINT} --output-root ${MACRO_ROOT} \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --evaluate-script ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --train-split A_train --sample-max-rows 600000 --chunk-len 512 --ridge-alpha 1.0 --seed 42
python ${STRUCTURAL_ANALYSES}/stage5_macro_sufficiency_evaluate.py \
  --input-root ${INPUT_ROOT} --checkpoint ${CHECKPOINT} \
  --artifacts ${MACRO_ROOT}/metadata/stage5_macro_sufficiency_artifacts.pkl \
  --output-root ${MACRO_ROOT}/evaluation \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --evaluate-script ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --stage1-root ${STAGE1_ROOT} --splits A_val B_confirm --chunk-len 512 --seed 42

GEOMETRY_ROOT=${OUTPUT_ROOT}/stage5_representation_geometry
python ${STRUCTURAL_ANALYSES}/stage5_representation_geometry_train.py \
  --input-root ${INPUT_ROOT} --checkpoint ${CHECKPOINT} --output-root ${GEOMETRY_ROOT} \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --evaluate-script ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --train-split A_train --sample-max-rows 300000 --chunk-len 512 \
  --pca-components 64 --ridge-alpha 1.0 --seed 42
python ${STRUCTURAL_ANALYSES}/stage5_representation_geometry_evaluate.py \
  --input-root ${INPUT_ROOT} --checkpoint ${CHECKPOINT} \
  --artifacts ${GEOMETRY_ROOT}/metadata/stage5_representation_geometry_artifacts.pkl \
  --output-root ${GEOMETRY_ROOT}/evaluation \
  --train-script ${EVENT_MAIN}/train_event_ssl.py \
  --evaluate-script ${EVENT_MAIN}/evaluate_event_ssl_structure.py \
  --stage1-root ${STAGE1_ROOT} --splits A_val B_confirm --chunk-len 512 \
  --sample-max-rows 250000 --seed 42
```

Next run `robustness/README.md`, followed by `scripts/README.md`. Run the additional seeds before the state-only and objective-control workflows.
