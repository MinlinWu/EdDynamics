# Cross-analysis workflows

Prerequisites:

1. Primary empirical, mechanism and Event-SSL outputs.
2. Both construction-matched null runs.
3. All five additional Event-SSL seeds before the six-seed audit.

Run the workflows in `robustness/README.md`, then run `scripts/README.md`.

Supplementary Note 8 and Table 13 compare empirical, mechanistic and neural fields after subtracting the field of the calibrated three-constant accounting predictor (C3). Use the completed [conditional-baseline results](../s2_minimal_mechanism/robustness/conditional_baselines/README.md) and frozen predictions from all six original neural seeds. The [launch commands](robustness/README.md#common-reference-comparisons) and [output guide](robustness/common_reference/README.md) cover both the common-reference comparison and the original field, state and transition summaries.
