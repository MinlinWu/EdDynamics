# Statistical reporting from preserved outputs

This workflow reads saved bootstrap draws, fitted-model evaluations and numerical summaries to reproduce parts of Supplementary Tables 4, 7, 8, 11 and 13. It writes new summaries to a separate destination. The models and resampling draws come from the corresponding upstream analyses.

## Usage

Run from this directory. Point `FOURTERM_ROOT` and `COMMON_REFERENCE_ROOT` at the completed [conditional-baseline](../../../s2_minimal_mechanism/robustness/conditional_baselines/README.md) and [common-reference](../../../s4_cross_analysis/robustness/common_reference/README.md) outputs. The destination must be new.

```bash
OUTPUT_ROOT=/path/to/outputs_KT4
FOURTERM_ROOT=/path/to/conditional_baseline_results
COMMON_REFERENCE_ROOT=/path/to/common_reference_results
STATISTICAL_REPORT_ROOT=/path/to/statistical_report

python -m pip install -r requirements.txt
python build_statistical_reports.py \
  --output-root "$OUTPUT_ROOT" \
  --mechanism-root "$FOURTERM_ROOT" \
  --excess-root "$COMMON_REFERENCE_ROOT" \
  --destination "$STATISTICAL_REPORT_ROOT"
```

If `--mechanism-root` is omitted, the program looks under `four_term_mechanism_revision` within the original output root. Supply `--excess-root` to include the three-way C3 intervals and six-seed common-target results.

The default runs all three reporting sections. To select one, add its `--sections` value.

| Value | Inputs and manuscript results |
|---|---|
| `b1b2` | Field and transition bootstrap diagnostics (Supplementary Table 11), weighting definitions, and paired conditional-baseline and common-reference estimates (Note 8 and Table 13). |
| `residence` | Saved learner-multiplier draws for residence times (Supplementary Table 4). |
| `pure-ssl` | Evaluations of the predictive-state and future-prediction-only Event-SSL models (Supplementary Table 7), plus representation geometry for the three objective conditions (Table 8). |

The default seed order is 42, 2026, 666, 606, 37 and 4669. `B_confirm` is the held-out confirmation cohort used for the reported matched-seed summaries.

## Field and transition uncertainty

The field report joins the preserved same-method comparison and bias tables using comparison, support definition and metric. It recalculates mean-minus-point bias and bias divided by sample standard deviation, and verifies agreement with the stored values. The original reported interval and the standard-error, basic and percentile intervals remain separate. Transition counts are derived from the preserved finite fraction and the archived requested replicate count. The transition basic interval is reconstructed as twice the point minus the opposite percentile endpoint, without clipping.

The two primary paired field-discrepancy contrasts retain their 97.5% basic intervals. Raw and common-target intervals are copied in full, including missing values and every positive, negative and mixed result. The reporting code does not change their resampling streams, endpoints or missing-value policy.

The original field analysis allowed finite-value reporting at a finite fraction of at least 0.99. The later accounting-reference analysis used a different random stream and required every requested draw to be finite. These contracts remain distinct. A finite interval is not a guarantee of nominal coverage, and the size of bootstrap bias must be considered when interpreting precision. Standard-error intervals may use the original Fisher transformation, whereas the stored basic and percentile intervals retain their original scale.

`estimand_definitions.json` distinguishes full-panel learner weights from weights computed on eligible prediction intervals. Point estimates and confidence intervals must belong to the same estimand. This report does not claim to numerically attribute the difference between those estimands to individual processing steps.

## Residence summaries

The residence report reads every preserved learner-multiplier bootstrap replicate and its corresponding original summary. It groups by macrostate and statistic, checks unique replicate identifiers and complete agreement of group identities, and recomputes the 2.5th and 97.5th percentiles using linear interpolation. The original data contain 18 groups of 1,000 draws. Requested and finite counts, point estimates, means, medians, sample standard deviations, bias, basic intervals and the original stored percentile limits are retained. The recomputed percentiles must agree with the preserved numerical summary to within 1e-12. These recomputed values provide a direct check of the stored numerical summaries.

## Original objective controls

The original six-seed comparison reads the confirmation-cohort proposal-field correlations for the predictive-state and pure self-supervised models, pairs values by seed and computes full minus pure. The summary reports the mean, sample standard deviation, range, sign counts and a two-sided 95% Student-t interval with five degrees of freedom for the default six seeds. This interval describes variability across matched training seeds; it is not a learner-bootstrap interval. These original proposal fields are distinct from the supplementary trajectories obtained using the same readout before and after an event.

The geometry section retains each seed's original results for all three objective conditions. It checks that the two coordinate-associated principal components are distinct, drawn from the first 16 components and equal to the components selected in the training cohort. It does not reselect components on held-out data. All continuous numeric geometry metrics are summarized with finite counts, means, sample standard deviations and ranges, including canonical correlations and raw linear-readout recovery. Original per-seed geometry values and component identities are preserved in a separate table.

## Output and provenance

The principal outputs are:

- `Table17_field_diagnostics_complete.csv` contains the field diagnostics corresponding to Supplementary Table 11. `Table17` is the saved output filename. `original_transition_diagnostics.csv` contains the matching transition diagnostics.
- `retained_primary_Delta_97p5.csv` contains the two fixed-support 97.5% basic intervals for the calibrated accounting and ridge predictor discrepancies relative to the four-term mechanism (Supplementary Table 13).
- `baseline_paired_intervals_complete.csv`, `raw_and_common_target_intervals_complete.csv`, and, when supplied, `C3_three_way_intervals_complete.csv`: full preserved interval families.
- `B2_estimand_ledger.csv` specifies which learner intervals contribute to each field comparison, their weights, cell support and resampling rules. `B2` is the file label for these estimand definitions.
- `residence_intervals_recalculated.csv` contains all 18 macrostate–statistic combinations, including the 12 corrected percentile intervals for tail excess and restricted mean survival time lift in Supplementary Table 4.
- `pure_ssl_paired_seed_values.csv` and `pure_ssl_paired_seed_summary.csv` contain the matched predictive-state minus future-prediction-only field correlations and the interval across six seeds (Supplementary Table 7). The eight corrected six-run retention summaries in that table use the [original scoring functions and verification records](../../../reproducibility/Numerical_source_records.md).
- `geometry_metrics_by_seed.csv` and `geometry_seed_summary.csv` contain the original representation measures and their across-seed summaries (Supplementary Table 8).
- `manifest.json`: input hashes, output hashes, numerical checks and completion status.

This reporting workflow reads preserved experimental outputs and writes separate aggregate summaries.
