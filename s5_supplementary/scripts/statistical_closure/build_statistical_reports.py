import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 2026, 666, 606, 37, 4669)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Report:
    def __init__(self, destination):
        self.destination = destination
        self.sources = {}
        self.outputs = []
        self.checks = []
        self.complete = False

    def read(self, path):
        path = path.resolve(strict=True)
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        self.sources[str(path)] = {"sha256": sha256(path), "rows": len(frame)}
        return frame

    def write(self, name, frame):
        path = self.destination / name
        frame.to_csv(path, index=False, float_format="%.17g")
        self.outputs.append({"file": name, "rows": len(frame), "sha256": sha256(path)})

    def check(self, name, passed, **details):
        self.checks.append({"name": name, "passed": bool(passed), **details})
        if not passed:
            raise ValueError(f"Verification failed: {name}: {details}")

    def finish(self):
        payload = {"sources": self.sources, "outputs": self.outputs, "checks": self.checks,
                   "new_training_runs": 0, "new_bootstrap_runs": 0, "complete": self.complete}
        (self.destination / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def first_existing(paths):
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("No input found among: " + ", ".join(map(str, paths)))


def difference(values, stored):
    actual = np.asarray(values, dtype=float)
    expected = np.asarray(stored, dtype=float)
    if actual.shape != expected.shape or not np.array_equal(np.isfinite(actual), np.isfinite(expected)):
        return float("inf")
    finite = np.isfinite(actual)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0


def b1b2(report, mechanism_root, excess_root):
    evidence = mechanism_root / "existing_evidence"
    same_method = report.read(evidence / "Table11_same_method_cross_comparison.csv")
    bias = report.read(evidence / "Table11_bias_diagnostics_extracted.csv")
    keys = ["comparison", "support_contract", "metric"]
    report.check("unique_field_rows", not same_method.duplicated(keys).any())
    report.check("unique_bias_rows", not bias.duplicated(keys).any())
    diagnostics = same_method.merge(bias[keys + ["bias_over_sd"]], on=keys, validate="one_to_one", how="left", indicator=True)
    report.check("matching_field_and_bias_rows", len(same_method) == len(bias) and diagnostics["_merge"].eq("both").all())
    computed_bias = diagnostics.bootstrap_mean - diagnostics.point_estimate
    computed_ratio = computed_bias / diagnostics.bootstrap_sd.replace(0, np.nan)
    report.check("field_bias_recalculation", difference(computed_bias, diagnostics.bootstrap_bias) < 1e-12)
    report.check("field_bias_over_sd_recalculation", difference(computed_ratio, diagnostics.bias_over_sd) < 1e-10)
    diagnostics["bootstrap_bias"] = computed_bias
    diagnostics["bias_over_sd"] = computed_ratio
    fields = keys + ["point_estimate", "requested_replicates", "finite_replicates", "bootstrap_mean",
                     "bootstrap_sd", "bootstrap_bias", "bias_over_sd", "reported_interval_method",
                     "reported_ci_low", "reported_ci_high", "se_centered_ci_low", "se_centered_ci_high",
                     "basic_ci_low", "basic_ci_high", "percentile_low", "percentile_high"]
    diagnostics = diagnostics[fields]
    report.write("Table17_field_diagnostics_complete.csv", diagnostics)
    report.write("headline_vector_diagnostics.csv", diagnostics[diagnostics.metric.eq("drift_vector_corr")])
    original_bias = report.read(evidence / "legacy" / "11_bootstrap_bias_diagnostics.parquet")
    support = report.read(evidence / "legacy" / "12_support_reselection_summary.parquet")
    report.write("original_bootstrap_bias_diagnostics_complete.csv", original_bias)
    report.write("original_support_reselection_summary_complete.csv", support)
    requested = support.replicates.dropna().unique()
    report.check("single_original_replicate_count", len(requested) == 1)
    requested = int(requested[0])
    transition = original_bias[original_bias.domain.eq("transition")].copy()
    transition["requested_replicates"] = requested
    transition["finite_replicates"] = (transition.finite_fraction * requested).round().astype(int)
    transition["bias_over_sd"] = transition.bootstrap_bias / transition.bootstrap_sd.replace(0, np.nan)
    transition["basic_ci_low"] = 2 * transition.point_estimate - transition.percentile_high
    transition["basic_ci_high"] = 2 * transition.point_estimate - transition.percentile_low
    report.write("original_transition_diagnostics.csv", transition)
    paired = report.read(mechanism_root / "inference" / "paired_intervals.csv")
    report.write("baseline_paired_intervals_complete.csv", paired)
    primary = paired.primary_family.astype(str).str.lower().eq("true")
    report.write("retained_primary_Delta_97p5.csv", paired[paired.support_contract.eq("fixed") & primary & np.isclose(paired.level, .975)])
    cross = report.read(mechanism_root / "cross_model" / "seed42_uncertainty" / "paired_intervals.csv")
    report.write("raw_and_common_target_intervals_complete.csv", cross)
    if excess_root is not None:
        report.write("C3_three_way_intervals_complete.csv", report.read(excess_root / "paired_intervals.csv"))
        report.write("six_seed_eligible_interval_common_target.csv", report.read(excess_root / "P1" / "cross_model" / "six_seed_C3_common_target_original.csv"))
    ledger_path = Path(__file__).with_name("estimand_definitions.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    report.sources[str(ledger_path.resolve())] = {"sha256": sha256(ledger_path), "rows": len(ledger)}
    report.write("B2_estimand_ledger.csv", pd.DataFrame(ledger))


def residence(report, output_root):
    root = output_root / "supplementary_robustness" / "empirical" / "tables"
    draws = report.read(root / "empirical_residence_user_multiplier_bootstrap_replicates.parquet")
    summary = report.read(root / "empirical_residence_user_multiplier_bootstrap_summary.parquet")
    keys = ["macrostate", "metric"]
    report.check("unique_residence_summary", not summary.duplicated(keys).any())
    report.check("unique_residence_replicates", not draws.duplicated(keys + ["replicate"]).any())
    summaries = summary.set_index(keys)
    grouped = list(draws.groupby(keys, sort=True))
    report.check("residence_groups_match_summary", set(key for key, _ in grouped) == set(summaries.index))
    rows = []
    for key, group in grouped:
        values = group.value.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        report.check(f"finite_residence_group_{key}", len(finite) > 1)
        stored = summaries.loc[key]
        point = float(stored.formal_point_estimate)
        low, high = np.quantile(finite, [.025, .975], method="linear")
        mean = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1))
        row = {"macrostate": key[0], "metric": key[1], "requested_replicates": len(values),
               "finite_replicates": len(finite), "formal_point_estimate": point, "bootstrap_mean": mean,
               "bootstrap_median": float(np.median(finite)), "bootstrap_sd": sd, "bootstrap_bias": mean - point,
               "bias_over_sd": (mean - point) / sd if sd else np.nan,
               "percentile_low": low, "percentile_high": high, "basic_low": 2 * point - high,
               "basic_high": 2 * point - low, "stored_ci_2p5": stored.ci_2p5, "stored_ci_97p5": stored.ci_97p5,
               "quantile_max_absolute_difference": max(abs(low - stored.ci_2p5), abs(high - stored.ci_97p5)),
               "minimum": float(finite.min()), "maximum": float(finite.max())}
        report.check(f"residence_count_matches_{key}", len(finite) == stored.replicates_finite)
        report.check(f"residence_quantiles_match_{key}", row["quantile_max_absolute_difference"] < 1e-12)
        rows.append(row)
    report.write("residence_intervals_recalculated.csv", pd.DataFrame(rows))


def split_row(frame, split, path):
    if "split" in frame:
        frame = frame[frame.split.eq(split)]
    if len(frame) != 1:
        raise ValueError(f"Expected one {split} row: {path}; found {len(frame)}")
    return frame.iloc[0]


def pure_ssl(report, output_root, seeds, split):
    from scipy.stats import t

    metric = "learned_plane_drift_vector_corr"
    seed_rows = []
    geometry = []
    for seed in seeds:
        base = output_root if seed == 42 else output_root / "random_seed_experiments" / f"seed_{seed}"
        values = {}
        for model, relative in [("full", "evaluation_predictive_state"), ("pure", "evaluation_pure_ssl_probe")]:
            folder = base / "stage4_event_ssl" / relative / "tables"
            stems = [f"stage4_event_ssl_structural_metrics_{split}", "stage4_event_ssl_structural_metrics_all_splits"]
            candidates = [folder / (stem + extension) for stem in stems for extension in [".parquet", ".csv.gz", ".csv"]]
            path = first_existing(candidates)
            row = split_row(report.read(path), split, path)
            values[model] = float(row[metric])
            values[f"{model}_source"] = str(path)
        report.check(f"finite_paired_seed_{seed}", all(np.isfinite(values[model]) for model in ["full", "pure"]))
        seed_rows.append({"seed": seed, "split": split, "metric": metric, **values,
                          "full_minus_pure": values["full"] - values["pure"]})
        path = output_root / "stage5_objective_control_hidden_geometry" / "seeds" / f"seed{seed}" / "tables" / "geometry_metrics.parquet"
        frame = report.read(path)
        selected = frame[frame.split.eq(split)].copy()
        report.check(f"unique_geometry_models_{seed}", not selected.model.duplicated().any())
        report.check(f"required_geometry_models_{seed}", {"pure_ssl", "predictive_state", "task_only"}.issubset(set(selected.model)))
        report.check(f"geometry_seed_identity_{seed}", selected.seed.eq(seed).all())
        training = frame[frame.split.eq("A_train")].set_index("model")
        for _, row in selected.iterrows():
            assigned = [int(row.assigned_pc_M), int(row.assigned_pc_Psi)]
            fitted = [int(training.loc[row.model].assigned_pc_M), int(training.loc[row.model].assigned_pc_Psi)]
            report.check(f"training_selected_distinct_components_{seed}_{row.model}", assigned == fitted and assigned[0] != assigned[1] and all(1 <= value <= 16 for value in assigned))
        selected["source"] = str(path)
        geometry.append(selected)
    paired = pd.DataFrame(seed_rows)
    values = paired.full_minus_pure.to_numpy(dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half_width = float(t.ppf(.975, n - 1) * sd / np.sqrt(n))
    report.write("pure_ssl_paired_seed_values.csv", paired)
    report.write("pure_ssl_paired_seed_summary.csv", pd.DataFrame([{
        "split": split, "metric": metric, "contrast": "full_minus_pure", "seeds": n,
        "mean": mean, "sample_sd": sd, "minimum": values.min(), "maximum": values.max(),
        "positive_seeds": int((values > 0).sum()), "zero_seeds": int((values == 0).sum()),
        "negative_seeds": int((values < 0).sum()), "confidence_level": .95,
        "interval_method": "paired_seed_student_t", "degrees_of_freedom": n - 1,
        "ci_low": mean - half_width, "ci_high": mean + half_width}]))
    frame = pd.concat(geometry, ignore_index=True)
    report.write("geometry_metrics_by_seed.csv", frame)
    metrics = [column for column in frame.select_dtypes(include=[np.number]).columns
               if column not in ["seed", "sample_rows", "assigned_pc_M", "assigned_pc_Psi",
                                 "cca_original_component_1", "cca_original_component_2",
                                 "descriptive_split_max_pc_M", "descriptive_split_max_pc_Psi"]]
    rows = []
    for model, group in frame.groupby("model", sort=True):
        report.check(f"all_geometry_seeds_{model}", set(group.seed) == set(seeds))
        for column in metrics:
            values = group[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append({"split": split, "model": model, "metric": column, "requested_seeds": len(values),
                         "finite_seeds": len(finite), "mean": float(finite.mean()) if len(finite) else np.nan,
                         "sample_sd": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                         "minimum": float(finite.min()) if len(finite) else np.nan,
                         "maximum": float(finite.max()) if len(finite) else np.nan})
    report.write("geometry_seed_summary.csv", pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser(description="Recalculate statistical reports from preserved experimental outputs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mechanism-root", type=Path)
    parser.add_argument("--excess-root", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", choices=["b1b2", "residence", "pure-ssl"], default=["b1b2", "residence", "pure-ssl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--split", choices=["A_train", "A_val", "B_confirm"], default="B_confirm")
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error("At least two distinct seeds are required.")
    if not args.output_root.is_dir():
        parser.error("The original output root must exist.")
    args.destination.mkdir(parents=True, exist_ok=False)
    report = Report(args.destination)
    try:
        if "b1b2" in args.sections:
            b1b2(report, args.mechanism_root or args.output_root / "four_term_mechanism_revision", args.excess_root)
        if "residence" in args.sections:
            residence(report, args.output_root)
        if "pure-ssl" in args.sections:
            pure_ssl(report, args.output_root, args.seeds, args.split)
        report.complete = True
    finally:
        report.finish()


if __name__ == "__main__":
    main()
