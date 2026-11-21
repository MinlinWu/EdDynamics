# Common-reference population-field comparisons

Supplementary Note 8 and Table 13 compare three population fields after subtracting the same reference field. That reference is the calibrated three-constant accounting predictor (`C3`). The three fields come from observations, the frozen four-term mechanism and the event-based self-supervised learning model (Event-SSL). The workflow also assembles the original state, field and transition measures from frozen predictions.

## Execution

Complete the [six original Event-SSL runs](../../../s3_event_ssl/robustness/random_seed_workflows/README.md) and the [conditional-baseline comparison](../../../s2_minimal_mechanism/robustness/conditional_baselines/README.md). Use the conditional-baseline Python environment. Run these commands from the repository root with the same `OUTPUT_ROOT` and `FOURTERM_ROOT` as the upstream analysis.

```bash
CODE=s4_cross_analysis/robustness/common_reference
OUTPUT_ROOT=/path/to/outputs_KT4
FOURTERM_ROOT=/path/to/conditional_baseline_results
COMMON_REFERENCE_ROOT=/path/to/common_reference_results
COMMON_REFERENCE_SMOKE=/path/to/common_reference_smoke

python "$CODE/run_excess.py" --history "$OUTPUT_ROOT" --fourterm-root "$FOURTERM_ROOT" \
  --output "$COMMON_REFERENCE_SMOKE" --smoke --workers 8
python "$CODE/run_excess.py" --history "$OUTPUT_ROOT" --fourterm-root "$FOURTERM_ROOT" \
  --output "$COMMON_REFERENCE_ROOT" --workers 8
```

Choose new smoke and formal destinations. By default `--package` points to the conditional-baseline code in this repository. The compatibility check accepts the recorded mechanism run or a formal run of the registered public implementation and verifies committed input and output hashes.

For the algebraic self-test, add `--self-test-only` to either command with an unused output path. The parser still requires `--history` and `--output`. This mode checks the registered code version and returns without reading experimental outputs. The smoke run checks 21 saved draws from the seed-42 model, including one with incomplete fixed support.

## Metrics and uncertainty

The three comparisons are empirical-minus-C3 versus mechanism-minus-C3, empirical-minus-C3 versus neural-minus-C3, and mechanism-minus-C3 versus neural-minus-C3. They retain vector correlation, speed correlation, local direction, amplitude slope, absolute squared distance, reference energy and component-specific summaries on the same supported empirical-origin grid. Subtracting a common field leaves the absolute model prediction error and the mechanism-neural difference unchanged; the implementation checks these identities for every resampling draw.

Six seeds (42, 2026, 666, 606, 37, 4669) provide point estimates. For seed 42, 2,000 paired whole-learner draws exactly replay the mechanism comparison's learner order and `SeedSequence([42,112358,replicate])`. Fixed and reselected support are separate. These additional comparisons are descriptive/sensitivity analyses with nominal 95% intervals, not another multiplicity-controlled primary endpoint. An interval is withheld if any required draw is nonfinite. Basic, percentile and standard-error intervals, finite counts and bootstrap displacement remain available rather than selecting a preferred method.

The empirical-origin conditional fields are distinct from a neural model's own-coordinate field and from applying one readout before and after a recurrent update. The opportunity information supplied to the mechanistic and neural models also differs. Correspondence does not imply statistical independence, common internal implementation, complete calibration or autonomous forecasting.

## Reports

- `three_way_point_metrics_by_seed.csv` and `three_way_point_metrics_long.csv` give the six-seed point estimates by field pair, for both axes jointly and for response order (M) and exposure alignment (Ψ) separately.
- `paired_bootstrap_replicates.csv.gz` and `paired_intervals.csv`: complete draw-level and interval-level summaries, including unavailable values.
- `original_metrics/` and `P1/`: the original five-component mechanism score and separate field, state and transition descriptors. The score is not the neural representation composite.
- `grids/`: all field values and support masks used by the figures.
- `replay_audit.json`: numerical identity with the previous paired comparisons and algebraic checks.
- `COMPLETE.json`: technical completion independent of positive or negative findings.

To regenerate the metric report and figures separately, read the saved metrics and grid from the formal run. Write the new reports and figures in separate destinations.

```bash
python "$CODE/p1_reports.py" --oldroot "$FOURTERM_ROOT" \
  --out /path/to/new_metric_report \
  --supplementary-metrics-path "$COMMON_REFERENCE_ROOT/P1_extra_metrics.csv"
python "$CODE/presentation_figures.py" \
  --grid "$COMMON_REFERENCE_ROOT/grids/seed_42.csv.gz" \
  --output /path/to/new_field_figures
```

The report writer creates `P1/` under `--out`. Its `OVERVIEW.txt` includes fixed explanatory text describing the ordering observed in the original study. For another set of saved inputs, read the estimates in `P1/comparison_metrics.csv` to establish the ordering. The figure command reads the seed-42 grid and preserves the common axes, support mask and arrow scale for the raw and reference-subtracted fields.

The output names retain analysis identifiers used in the saved files. `M1` denotes the conditional-predictor comparison, `M2` the additive-update components and `M3` the earlier mechanism–neural comparison. `P1` contains the original metric summaries. `P2` labels the three-way C3 comparison. In a field-pair name, `E` means empirical, `M` mechanistic and `SSL` neural. In a coordinate name, `M` means response order (signed response evidence relative to item baselines), and `Psi` denotes exposure alignment (Ψ). Some saved figures label the response-order axis “Response evidence M” and the exposure-alignment axis “Exposure alignment Psi”.
