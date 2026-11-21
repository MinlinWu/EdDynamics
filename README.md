# EdDynamics result reproduction

This repository contains the analyses for *Macroscopic dynamics link mechanistic and neural models in adaptive learning*. The [figure and table index](reproducibility/Reproduction_index.md) maps each result to its analysis and the [numerical source records](reproducibility/Numerical_source_records.md) document corrections from preserved outputs. An earlier version of the study appeared as [*Reproducible macroscopic dynamics in a closed-loop human-AI learning system*](https://doi.org/10.48550/arXiv.2608.30946).

In EdNet-KT4 learner histories, response order (M) measures response evidence relative to item-specific expectations, and exposure alignment (Ψ) measures how activity aligns with unresolved demand. The empirical flow in this plane provides a common reference for the four-term mechanism and predictive event-level self-supervised learning (Event-SSL). The models are fitted independently and compared on the same observed intervals.

The cohort labels used in commands are `A_train` (development training), `A_val` (development validation) and `B_confirm` (independent confirmation). Replace every `/path/to/...` placeholder before running a command.

## Layout

```text
s1_empirical_dynamics/
├── main/
├── scripts/
└── robustness/
s2_minimal_mechanism/
├── main/
├── scripts/
└── robustness/
s3_event_ssl/
├── main/
│   ├── controls/
│   └── structural_analyses/
├── scripts/
└── robustness/
s4_cross_analysis/
├── main/
├── scripts/
└── robustness/
s5_supplementary/
└── scripts/
reference_outputs/
reproducibility/
```

See [reference outputs](reference_outputs/README.md) for core numeric reports and supplementary report-generation scripts.

The `s1`–`s5` directories group related analyses. Names beginning with `stage` identify stages in the original computation. Within the neural information controls, S0–S3 denote four analysis blocks, beginning with source verification, as described in the [design table](s3_event_ssl/robustness/information_controls/README.md#design).

`main/` contains primary experiments and downstream analyses, `robustness/` contains additional robustness workflows and audits, and `scripts/` contains result extraction and figure commands.

## Dependency order

1. Generate the empirical outputs in `s1_empirical_dynamics/main/`.
2. Run the original innovation-reassignment control on the validation and confirmation cohorts, then run the empirical sensitivity analysis in the [empirical robustness guide](s1_empirical_dynamics/robustness/README.md).
3. Run the complete ordered mechanism sequence in `s2_minimal_mechanism/main/`.
4. Run primary Event-SSL, its controls, macro sufficiency and representation geometry using `s3_event_ssl/README.md`.
5. Run the five additional Event-SSL seeds, then the state-only and objective-control workflows.
6. Complete the original empirical and mechanism robustness workflows. The supplementary response-generating controls can run once the activity–idle comparator is available.
7. Complete the original [cross-model comparisons](s4_cross_analysis/README.md) and learner-cluster uncertainty analysis. Generate the [original mechanism parameter and statewise kinetics tables](s5_supplementary/scripts/README.md#minimal-mechanism-extraction) before starting the conditional predictors. Then follow the [supplementary workflow order](SUPPLEMENTARY_ANALYSES.md#execution-order) for the conditional predictors, common-reference comparisons and neural information controls.
8. Follow each component’s `scripts/README.md` once its inputs are available. Generate the remaining supplementary figures and finish with the [statistical reports](s5_supplementary/scripts/README.md#statistical-reporting-from-preserved-outputs).

Resume a wrapper with the same result root when it provides completion markers. Do not combine outputs generated from different code copies in one result root.

## Supplementary analyses

The additional response-generating controls, conditional predictor comparisons, common-reference field comparisons, SSL information controls and statistical reporting workflows are indexed in [Supplementary analyses](SUPPLEMENTARY_ANALYSES.md). Each module documents its inputs, commands and output tables. The [fixed-source guide](reproducibility/Reproduction_index.md#fixed-sources) identifies preserved inputs needed for exact replay. The analyses extend the macroscopic-recovery and cross-model correspondence framework without replacing the original models or comparisons.
