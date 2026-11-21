# Empirical dynamics

Use the EdNet-KT4 learner files and the EdNet Contents metadata from the [EdNet data release](https://github.com/riiid/ednet). Set `PREPROCESSED_ROOT` to a fresh, dedicated directory. The preprocessing step clears its `contents/`, `kt4/` and `metadata/` subdirectories by default.

Set the repository, input and result paths:

```bash
REPO_ROOT=/path/to/EdDynamics
KT4_INPUT=/path/to/KT4
CONTENTS_INPUT=/path/to/contents
PREPROCESSED_ROOT=/path/to/preprocessed_data
OUTPUT_ROOT=/path/to/outputs
S1_MAIN=${REPO_ROOT}/s1_empirical_dynamics/main

EDNET_KT4_INPUT=${KT4_INPUT} \
EDNET_CONTENTS_INPUT=${CONTENTS_INPUT} \
EDNET_OUTPUT_ROOT=${PREPROCESSED_ROOT} \
python ${S1_MAIN}/preprocess_ednet_kt4_full_297915.py

EDNET_KT4_DATA_ROOT=${PREPROCESSED_ROOT} \
EDNET_KT4_OUTPUT_ROOT=${OUTPUT_ROOT} \
python ${S1_MAIN}/build_effective_dynamics_kt4_stage1_empirical.py
```

Next:

1. Run construction-matched null and empirical sensitivity from `robustness/README.md`.
2. Complete the minimal-mechanism and Event-SSL primary workflows.
3. Return to `robustness/README.md` for the remaining robustness workflows.
4. Run `scripts/README.md`.

The additional [generative construction controls](robustness/generative_construction_null/README.md) jointly rebuild the semantic coordinates from simulated responses under the recorded opportunity schedule. Run them after the semantic-specificity workflow has produced the activity-idle coordinate and its training-defined regions. The entry commands and prerequisite paths are in [Empirical robustness](robustness/README.md#generative-construction-controls).
