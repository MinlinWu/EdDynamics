# Generative construction controls

This workflow generates responses under the recorded opportunity schedule. It reconstructs response order (M) from signed response evidence relative to item baselines, and exposure alignment (Ψ) from the alignment of activity with unresolved demand. The [innovation-reassignment analysis](../construction_null/) jointly reassigns the observed response and exposure innovations. The controls here generate responses under two probability models.

The original empirical state builder, training item probabilities, cohort membership, support thresholds and training-defined convergence regions are reused. Null A uses item-specific response probabilities. Null B adds a stable learner effect calibrated on training data using 256 equiprobable support points. Missing responses are not imputed. The alternative activity-idle coordinate uses its own training support and convergence region.

## Inputs and environment

Use Linux with Python 3.10 or later and the original EdDynamics dependencies. Additional Python requirements are listed in `requirements.txt`. The direct CPU wrapper also uses `/usr/bin/time`; the Slurm wrapper requires `sbatch`.

Required inputs are the original preprocessed KT4 bundle, question and study tables, content metadata, and completed empirical outputs including the activity-idle comparator. The preprocessing location is read from the empirical manifest or supplied explicitly to the worker or Slurm wrapper with `--data-root`. Repository source paths are resolved from this module's ancestors or from `--repo-root`.

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs_KT4
CODE_ROOT=${REPO_ROOT}/s1_empirical_dynamics/robustness/generative_construction_null
python -m pip install -r ${CODE_ROOT}/requirements.txt
python -m unittest discover -s ${CODE_ROOT}/tests -p 'test_*.py'
```

The tests generate synthetic fixtures and verify the original source interface, joint reconstruction, summary integrity and retention of unsupported results. They do not estimate scientific effects from the study data.

## Direct CPU execution

Set new result directories when using a new source version. The first command performs a real-data smoke test; the second runs the full experiment after the smoke checks pass.

```bash
python ${CODE_ROOT}/execution_tools/run_direct_cpu.py \
  --code-root ${CODE_ROOT} --repo-root ${REPO_ROOT} \
  --output-root ${OUTPUT_ROOT} \
  --work-root ${OUTPUT_ROOT}/generative_construction_null \
  --smoke-root ${OUTPUT_ROOT}/generative_construction_null_smoke \
  --phase smoke --workers 8 --seed 42 --smoke-users 200

python ${CODE_ROOT}/execution_tools/run_direct_cpu.py \
  --code-root ${CODE_ROOT} --repo-root ${REPO_ROOT} \
  --output-root ${OUTPUT_ROOT} \
  --work-root ${OUTPUT_ROOT}/generative_construction_null \
  --smoke-root ${OUTPUT_ROOT}/generative_construction_null_smoke \
  --phase formal --workers 8 --seed 42
```

The wrapper delegates all scientific operations to the same worker and summary modules used by the Slurm workflow. Each subprocess uses one numerical thread. The worker count changes concurrency, not replicate IDs or random seeds.

## Slurm execution

The workflow uses CPU jobs and chained dependencies. Adapt memory, time and partition settings to the cluster. A dry run prints the exact submission commands without submitting jobs.

```bash
python ${CODE_ROOT}/submit_construction_null.py \
  --phase smoke --output-root ${OUTPUT_ROOT} --repo-root ${REPO_ROOT} \
  --seed 42 --smoke-users 200 --concurrency 8 --dry-run

python ${CODE_ROOT}/submit_construction_null.py \
  --phase smoke --output-root ${OUTPUT_ROOT} --repo-root ${REPO_ROOT} \
  --seed 42 --smoke-users 200 --concurrency 8
```

After `${OUTPUT_ROOT}/generative_construction_null_smoke/SMOKE_OK.json` has been written successfully:

```bash
python ${CODE_ROOT}/submit_construction_null.py \
  --phase all --output-root ${OUTPUT_ROOT} --repo-root ${REPO_ROOT} \
  --seed 42 --concurrency 8 --replicates-per-task 8
```

Direct execution and Slurm are alternative launchers. Do not start both against the same result directory.

## Statistical contract

- Each generating reference is evaluated in validation and confirmation, with 999 simulations per reference and cohort. These are Monte Carlo simulations, not learner-bootstrap draws.
- Seed 42 and deterministic replicate identifiers define the random-number streams. The summaries include every realization.
- The common inferential domain intersects training support with all observed and simulated drift supports. Field departure uses symmetric leave-one-realization-out comparison and each realization's learner-normalized occupancy weights on that common domain.
- Negative-divergence occupancy, inward flow and core slowing retain the original definitions and frozen regions. Conservative Monte Carlo ranking handles ties. Benjamini-Hochberg adjustment applies to these three convergence diagnostics within each reference, cohort and coordinate; it does not pool all endpoints.
- Null A on the original response/alignment coordinates is primary. Null B assesses stable heterogeneity; the activity-idle coordinate is a secondary sensitivity comparison. Validation is summarized before confirmation is released.
- These post hoc tests evaluate field departure and three convergence diagnostics conditional on the recorded opportunity schedule and the response probabilities estimated from training data. Policy counterfactuals and autonomous state dependence fall outside this design.
- Insufficient common support yields an explicit non-estimable result. Negative, null and non-estimable outcomes remain in the tables and do not prevent technically complete output.

## Outputs

The result root contains the frozen protocol, source snapshots, preparation/calibration records, simulation files and `summary/<cohort>/<reference>/`. Each summary includes coordinate-specific tests, support coverage, field comparisons, figures and `numeric_report.md`.

The final directory contains:

| File | Content |
|---|---|
| `construction_null_statistics.csv` | All cohort, reference, coordinate and diagnostic results, including signs, Monte Carlo p-values and adjusted values. |
| `construction_null_report.md` | Scientific definitions, interpretation, complete results and metric-specific support coverage. |
| `legacy_innovation_reassignment_tests.csv` | Results from the original innovation-reassignment analysis, when its output is available. |
| `excess_field_replication_descriptive.csv` | Descriptive cross-cohort comparisons on common overlap. |
| `final_artifacts.json`, `COMPLETE.json` | File integrity and completion records; completion does not require statistical significance. |

The workflow checks source and output hashes before it reuses a completed run.
