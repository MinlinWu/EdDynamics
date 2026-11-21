# SSL information controls

Supplementary Note 9 and Table 14 ask what an event-based self-supervised learning model (Event-SSL) reveals about response order (M) and exposure alignment (Ψ). The analyses measure whether the hidden state encodes these coordinates, how much the current event contributes to next-state prediction, and how recovery changes when alignment-derived information is removed from both the inputs and the prediction targets.

## Design

| Block | Comparison | Seeds and fitting | Uncertainty |
| --- | --- | --- | --- |
| S0 | Source identities, coordinate lineage, learner separation, frozen partition, and existing reference outputs | Original prepared inputs and checkpoints; no refitting | Integrity checks |
| S1 | The same frozen state readout before and after each event versus the original conditional proposal; full and record-shuffled models | Full and record-shuffled checkpoints at each of six seeds (42, 2026, 666, 606, 37, 4669), replayed on CPU | 2,000 paired whole-learner draws conditional on seed42 |
| S2 | Matched heads receiving hidden state only (H) or hidden state plus the current event (HZ) | Three paired fits: 42, 2026, 666; same initialization, samples, optimizer budget and validation criterion | 2,000 paired whole-learner draws conditional on seed42 |
| S3 | Future-only encoder with the original inputs (aug) versus joint removal of alignment-derived inputs and their future targets (clean) | Six paired fits: seed42 on CPU, 2026/666/606/37/4669 on five GPUs; eight fixed epochs; one post-freeze ridge readout used before and after each event | 2,000 paired whole-learner draws conditional on CPU seed42; six-seed spread is separately descriptive |

The six-seed design reported in Table 14 uses `cpu` followed by the Slurm submission below. Five paired fits were added after the CPU seed-42 result was available. The `all` command runs the separate three-seed CPU design and requires a separate output tree. The learner bootstrap conditions on the fitted seed-42 models and uses its own random-number seed, `20261117`.

The S2 heads use 300,000 training rows and at most 250,000 validation rows, eight epochs, batch size 192, learning rate 0.0002, weight decay 0.0001, gradient clipping 1, and Huber beta 0.05. S3 retains sequence length 256, stride 128, minimum sequence length 3, warm-up 8, batch size 192, learning rate 0.0002, weight decay 0.0001, and future offsets 1, 2 and 4. The readout uses up to 300,000 training rows, sampling seed42, and ridge alpha 1. Complete numerical settings are in `protocol.json`.

All GPU fits use float32 with AMP, TF32 and compilation disabled, deterministic algorithms enabled, and no loader subprocesses. Both conditions in a seed share CPU-initialized parameter tensors and window/update budgets. Identical per-step GPU dropout masks are not assumed. CPU, CUDA, loader, Python and NumPy random states and optimizer states are saved; cuDNN recurrent-dropout state is not separately serialized, so resumed GPU fitting is not claimed to be bitwise identical to uninterrupted fitting. Seed and training device are confounded: this design does not estimate a device effect or establish CPU/GPU training equivalence. Evaluation and probes use CPU float32, eight intra-operation threads and one inter-operation thread.

## Inputs and dependencies

Use the original EdDynamics environment with the packages in `requirements.txt`. The workflow imports three [original Event-SSL modules](../../main/) and verifies them against `source_contract.json`. `OUTPUT_ROOT` supplies the prepared memory maps, original full and record-shuffled checkpoints and evaluation records for each of six seeds, and the frozen six-state partition (`K6`). `FOURTERM_ROOT` supplies the saved row-level predictions of the four-term mechanism and the calibrated three-constant accounting predictor (`C3`) from the [conditional-baseline workflow](../../../s2_minimal_mechanism/robustness/conditional_baselines/README.md). The `check` command prints the paths it resolves. It first scans `OUTPUT_ROOT` for completed C3 models, even when `--fourterm-root` is supplied. Distinct models inside `OUTPUT_ROOT` make the selection ambiguous. Keep alternative versions outside that directory.

Each stage completion record binds code, input paths and file hashes to its outputs. Moving an input changes its recorded identity. Changing only its modification time triggers a hash check. Reuse a stage with the same code, paths and file contents. Formal results are written under `${OUTPUT_ROOT}/ssl_information_controls`.

## Run the six-seed design

Run from this directory with the same `FOURTERM_ROOT` throughout. Configure the Slurm resource directives described below before the smoke test.

```bash
OUTPUT_ROOT=/path/to/outputs_KT4
FOURTERM_ROOT=/path/to/conditional_baseline_results
export PYTHON_BIN=/path/to/environment/bin/python

"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" run_ssl.py selftest
"$PYTHON_BIN" run_ssl.py check --output-root "$OUTPUT_ROOT" --fourterm-root "$FOURTERM_ROOT"
"$PYTHON_BIN" run_ssl.py smoke --output-root "$OUTPUT_ROOT" --fourterm-root "$FOURTERM_ROOT"
"$PYTHON_BIN" run_ssl.py cpu --output-root "$OUTPUT_ROOT" --fourterm-root "$FOURTERM_ROOT"
sbatch slurm_s3.sbatch --history-root "$OUTPUT_ROOT"
```

Submit the Slurm job after `cpu` succeeds. CPU preparation completes 52 stages covering source verification, six-seed checkpoint replay, three pairs of prediction heads and the seed-42 information-removal comparison with its learner bootstrap. The Slurm job reads the C3 reference recorded during CPU preparation, trains the remaining five pairs concurrently, evaluates them on CPU and verifies the 40 additional stages. After verification of all 92 stages, it writes `final/COMPLETE.json`. The `both` mode of `run_direct.sh` combines `smoke` and `cpu` and accepts the same root arguments.

The supplied Slurm script requests one node, one task, five GPUs, 40 CPUs and 320 GB memory. Set the partition, memory and time request for the target cluster before running smoke or CPU preparation, because the script itself is part of the code digest. Submit from this directory or set `SSL_CODE_ROOT` to its absolute path. The controller maps the scheduler's GPU allocation through individually listed NVIDIA device files in `/etc/slurm/gres.conf`, checks UUIDs against the driver and verifies them with actual Torch tensors. Node-scoped or compact GRES configurations are rejected rather than guessed; adapt this infrastructure mapping explicitly for another cluster before freezing a run. No automatic deletion, output migration, device substitution or detached dispatcher is performed.

For the separate three-seed CPU alternative, use `all` after smoke and `report` to reverify its final report. For the six-seed design, `finalize_six.py` is the reporting entry and is called by the Slurm controller; its required contract is recorded in `S3_EXECUTION_CONTRACT.json`. `run_ssl.py report` is specific to the three-seed CPU alternative. Failed stages retain identity-bound partial outputs. Rerunning a trainer or evaluator with the same execution contract may resume its own stage, while the Slurm controller deliberately refuses to replace an existing execution contract.

## Quantities and interpretation

The paper and the output tables use `M` for response order (signed response evidence relative to item baselines) and Ψ (`Psi` in output tables) for exposure alignment, the alignment of activity with unresolved demand. Let X and Y be the observed current and next macrostates, S the readout before the event, T the same readout after the event, and P the conditional next-state proposal. N denotes the candidate next-state estimate in the stated comparison. Endpoint comparisons use N-X versus Y-X. Actual readout change uses T-S, while conditional proposal displacement uses P-S. These quantities and their amplitude, direction and support must not be interchanged. Frozen K6 transitions are interval-count estimates; six-state correlations are descriptive. Interval errors give equal weight to rows, whereas field estimates give each learner equal total valid-row mass. These estimands differ.

Fields use the fixed 40-by-40 grid and minimum cell count 30. The nominal 97.5% basic ranges on fixed support retain the two specified primary contrasts per block. Reselected support, nominal 95% ranges, percentile and standard-error ranges are reported as sensitivities. Adjustment is within a block's two contrasts, not a simultaneous guarantee across all blocks or support choices. Ranges condition on fixed fitted models, heads, probes and preprocessing. Finite draws, a small bootstrap displacement and matching signs across methods do not establish population coverage. Nonfinite draws remain visible; missing values are not filled or selectively discarded. Crossing zero is not evidence of equivalence, and one-sided ranges alone do not establish a population direction.

The shuffled control reorders complete interval records and recomputes indices and successor labels. Upstream state construction, alignment values and time gaps are not all regenerated, so it is a record-shuffled-to-ordered transfer comparison rather than an isolated causal effect of temporal order. S3 jointly removes the declared alignment-derived input lineage and excludes the same columns from the future-target average. Both macro losses are off during encoder training. Labels used for the ridge map after freezing remain supervised measurement. The control does not separately identify the effect of individual losses or each removed feature.

The C3 comparison joins the selected four-term prediction export to the confirmation intervals on `user_id` and `bundle_step_index`. It checks the observed start and next coordinates on the matched rows and records their fraction in `C3_bridge_status.json`. The descriptive comparison uses endpoint displacement N−X against Y−X and subtracts the same C3 field from both fields for the C3-relative measures. `S0/reference_identity.json` identifies the prediction export actually used. `S0/existing_evidence_inventory.json` separately lists recognized files found under `OUTPUT_ROOT`. It can omit results held elsewhere, and a file listed there contributes to a calculation only when the corresponding analysis reads it.

In saved reports, `F00-aug` and `F00-clean` label the future-only fits with the original and restricted information, respectively. `MSE` is the mean squared error of a state estimate. `D2` is the squared discrepancy between the two population fields on the specified support. In references to earlier analyses, `CN` denotes construction controls, `ARX` the ridge conditional-mean predictor and `P2` the [three-way common-reference field comparison](../../../s4_cross_analysis/robustness/common_reference/README.md).

## Output files

Read `final/numeric_report.md` alongside the following files in `final/` for Table 14.

| File | What it reports |
|---|---|
| `analysis_index.csv`, `validation_metrics_all.csv`, `confirmation_metrics_all.csv` | Comparison definitions and point estimates for each held-out cohort. |
| `paired_intervals_all.csv` | All 24 interval rows for the six designated contrasts conditional on the seed-42 models, with both support choices and confidence levels. |
| `S3_primary_paired_differences.csv`, `S3_seed_spread_descriptive.csv` | Removal-minus-original differences by seed and descriptive summaries across the six seeds and by training device. |
| `C3_bridge_all.csv`, `common_future_targets_confirmation.csv`, `CPU_vs_archive_precision.csv` | Common-reference fields, scores on shared future targets and differences between CPU replay and archived predictions. |

`S3_paired_differences_all_metrics.csv` retains the remaining signed contrasts. `S3_training_backend_and_contracts.csv` records each training device. The directory also contains figures and file-verification records. In the three-seed CPU workflow, the legacy field `outputs_are_real_computations_not_filled_results` in `final/COMPLETE.json` is a status flag. The `_SUCCESS.json` records contain the source or input identities and output hashes used to verify each stage.

`python tests/run_integration.py --workdir /path/to/new_synthetic_check` runs the synthetic CPU pipeline in a new directory, verifies stage reuse and checks that the workflow rejects modified outputs. The synthetic run and the reduced-data smoke test are labelled separately from the study results.
