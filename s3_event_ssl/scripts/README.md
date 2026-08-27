# Event-SSL result scripts

Run after the corresponding model and control outputs exist:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
SCRIPT_ROOT=${REPO_ROOT}/s3_event_ssl/scripts
MAIN_ROOT=${OUTPUT_ROOT}/stage4_event_ssl
FIGURE_ROOT=${MAIN_ROOT}/figures_publication_event_ssl

python ${SCRIPT_ROOT}/publication_event_ssl_figure5.py \
  --eval-root ${MAIN_ROOT}/evaluation_predictive_state \
  --output-root ${FIGURE_ROOT} \
  --split B_confirm --validation-split A_val

python ${SCRIPT_ROOT}/extract_event_ssl_stage4_publication_statistics.py \
  --main-root ${MAIN_ROOT}/evaluation_predictive_state \
  --pure-root ${MAIN_ROOT}/evaluation_pure_ssl_probe \
  --task-root ${MAIN_ROOT}/controls/task_only/evaluation \
  --time-shuffle-root ${OUTPUT_ROOT}/stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs \
  --tag-support-root ${OUTPUT_ROOT}/stage4_event_ssl_tag_support_randomized_control/evaluation \
  --output-root ${MAIN_ROOT}/all_experiment_comparison \
  --splits A_val B_confirm

python ${SCRIPT_ROOT}/publication_event_ssl_figure6.py \
  --comparison-root ${MAIN_ROOT}/all_experiment_comparison \
  --output-root ${FIGURE_ROOT} \
  --main-root ${MAIN_ROOT}/evaluation_predictive_state \
  --pure-root ${MAIN_ROOT}/evaluation_pure_ssl_probe \
  --task-root ${MAIN_ROOT}/controls/task_only/evaluation \
  --time-shuffle-root ${OUTPUT_ROOT}/stage4_event_ssl_time_shuffle_control/evaluation_on_ordered_inputs \
  --tag-support-root ${OUTPUT_ROOT}/stage4_event_ssl_tag_support_randomized_control/evaluation \
  --split B_confirm --validation-split A_val

python ${SCRIPT_ROOT}/extract_event_ssl_stage5_publication_statistics.py \
  --macro-root ${OUTPUT_ROOT}/stage5_macro_sufficiency/evaluation \
  --macro-train-root ${OUTPUT_ROOT}/stage5_macro_sufficiency \
  --geometry-root ${OUTPUT_ROOT}/stage5_representation_geometry/evaluation \
  --geometry-train-root ${OUTPUT_ROOT}/stage5_representation_geometry \
  --output-root ${OUTPUT_ROOT}/stage5_joint_macro_geometry_analysis \
  --splits A_val B_confirm
```

`publication_event_ssl_figure6.py` must follow `extract_event_ssl_stage4_publication_statistics.py`. After all five additional seed runs finish:

```bash
python ${SCRIPT_ROOT}/extract_event_ssl_random_seed_statistics.py \
  --seed-root ${OUTPUT_ROOT}/random_seed_experiments \
  --main-output-root ${OUTPUT_ROOT} \
  --output-root ${OUTPUT_ROOT}/random_seed_experiments/additional_information_random_seed_summary \
  --reference-seed 42 \
  --seeds 42,2026,666,606,37,4669 \
  --splits A_val B_confirm --strict
```
