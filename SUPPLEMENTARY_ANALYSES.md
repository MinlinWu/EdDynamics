# Supplementary analyses

These workflows extend the empirical, mechanism and neural-model analyses with response-generating controls, conditional predictor comparisons and information restrictions. The empirical population flow provides the common reference for testing recovery by each model and for comparing their spatial fields. A further analysis subtracts the field predicted by the same calibrated accumulation-and-normalisation model from the empirical, mechanistic and neural fields to compare the remaining structure.

## Workflows

| Workflow | Code and instructions | Numerical outputs |
|---|---|---|
| Response-generating construction controls | [Generative construction null](s1_empirical_dynamics/robustness/generative_construction_null/README.md) | Field departure and convergence diagnostics for two generators, two held-out cohorts and two coordinate definitions; 999 simulations per generator and cohort |
| Frozen mechanism and conditional predictors | [Conditional baselines](s2_minimal_mechanism/robustness/conditional_baselines/README.md) | Original model-family evidence, complete state/field/transition comparisons, update components, and 2,000 paired learner draws |
| Empirical, mechanism and neural fields under a common reference | [Common-reference comparisons](s4_cross_analysis/robustness/common_reference/README.md) | Six-seed point estimates, exact replay of 2,000 seed-42 learner draws, original metric assembly and field figures |
| Neural readouts and information restrictions | [SSL information controls](s3_event_ssl/robustness/information_controls/README.md) | Source checks, same-readout changes, matched prediction heads, six paired information-restriction fits, and 2,000 paired learner draws conditional on seed 42 |
| Complete uncertainty and original numerical summaries | [Statistical reporting](s5_supplementary/scripts/statistical_closure/README.md) | Finite counts, bias and interval methods; all residence quantiles; original six-seed objective-control and representation summaries |

## Execution order

1. Complete the original workflows and preserve their manifests, prepared inputs, model checkpoints, predictions and numerical summaries. Exact replay of the four-term confirmation analysis also requires the [nine preserved inputs](reproducibility/Reproduction_index.md#fixed-sources) identified in its source contract.
2. Run the response-generating controls and conditional-baseline workflow using their separate new result directories. These two analyses can run independently.
3. Run common-reference comparisons after the conditional-baseline workflow and the original six neural-model runs are complete. Run SSL controls with the same declared frozen mechanism reference. Their models are not fitted to one another.
4. Generate statistical reports from the preserved original and supplementary numerical outputs. Reporting does not train models or generate additional bootstrap draws.

Every module provides technical tests or integrity checks. Synthetic and reduced-data outputs are distinct from full-data results. Choose a fresh output directory when changing source versions; completed outputs remain bound to their original computational source and inputs. The original reports and new reports may have distinct scientific estimands and must not be substituted for one another solely because their metric names are similar.

## Statistical interpretation

The construction controls use Monte Carlo realizations. The paired field comparisons resample whole learners. The original residence summaries use saved learner-multiplier draws, and the original matched objective-control interval summarizes variability across training seeds. These sources of uncertainty are different.

The scripts retain all signed estimates, missing values, support-dependent results and interval-method differences. Technical completion does not require a positive finding. High field correlation does not imply amplitude calibration, autonomous forecasting or identification of an internal mechanism. Simple models are evaluated for their sufficiency within the specified model family, rather than universal superiority across all metrics or model classes.

The [source manifest](supplementary_source_manifest.json) lists the repository path, SHA-256 digest and role of each file in its scope. For files linked to an earlier run, `archived_source_binding` describes the corresponding archived source or supporting record. Paths beginning with `archives/` identify external execution records. The historical and public digests identify their respective file versions.
