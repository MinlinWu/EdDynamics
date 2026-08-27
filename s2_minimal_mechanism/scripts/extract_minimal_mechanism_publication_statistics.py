#!/usr/bin/env python3
"""Extract publication statistics for the frozen EdNet-KT4 minimal mechanism."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
DEFAULT_ROOT = Path(os.environ.get("EDNET_KT4_OUTPUT_ROOT", "/data/datasets/KT4/outputs_KT4"))
DEFAULT_STAGE1_ROOT = DEFAULT_ROOT / "stage1"
DEFAULT_PHASE1_ROOT = DEFAULT_ROOT / "stage2_phase1"
DEFAULT_MINIMALITY_ROOT = DEFAULT_ROOT / "stage2_phase1_unified_minimality"
DEFAULT_PHASE2_ROOT = DEFAULT_ROOT / "stage2_phase2_freeze"
DEFAULT_PHASE3_ROOT = DEFAULT_ROOT / "stage2_phase3_confirm"
DEFAULT_CONFIRM_SPLIT = "B_confirm"

MAIN_REQUIRED = "main_text_required"
MAIN_RECOMMENDED = "main_text_recommended"
SUPPLEMENT_REQUIRED = "supplement_required"
AUDIT_ONLY = "audit_only"
OPTIONAL_STRONG = "optional_strong_claim_support"

STABILITY_PANEL_METRICS = [
    ("one_step_mse_main_norm", "Closure loss"),
    ("occupancy_js_MR_PsiA", "Landscape JS"),
    ("drift_local_rmse_loss_MR_PsiA", "Flow residual"),
    ("drift_direction_loss_MR_PsiA", "Flow direction"),
    ("drift_magnitude_loss_MR_PsiA", "Flow speed"),
]


@dataclass
class SourceRecord:
    name: str
    path: Optional[Path]
    status: str
    rows: Optional[int] = None
    cols: Optional[int] = None
    note: str = ""



def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def path_candidates(base_or_path: Path) -> List[Path]:
    p = Path(base_or_path)
    if p.suffix:
        return [p]
    return [p.with_suffix(".parquet"), p.with_suffix(".csv.gz"), p.with_suffix(".csv"), p.with_suffix(".tsv")]


def first_existing(base_or_path: Path) -> Optional[Path]:
    for p in path_candidates(base_or_path):
        if p.exists():
            return p
    return None


def read_table_safe(base_or_path: Path, name: str, sources: List[SourceRecord]) -> pd.DataFrame:
    found = first_existing(base_or_path)
    if found is None:
        sources.append(SourceRecord(name=name, path=None, status="missing", note=str(base_or_path)))
        return pd.DataFrame()
    try:
        if found.suffix == ".parquet":
            df = pd.read_parquet(found)
        elif found.suffixes[-2:] == [".csv", ".gz"] or found.suffix == ".gz":
            df = pd.read_csv(found, low_memory=False)
        elif found.suffix == ".tsv":
            df = pd.read_csv(found, sep="\t", low_memory=False)
        elif found.suffix == ".csv":
            df = pd.read_csv(found, low_memory=False)
        else:
            df = pd.read_csv(found, low_memory=False)
        status = "ok_empty" if df.empty else "ok"
        sources.append(SourceRecord(name=name, path=found, status=status, rows=int(len(df)), cols=int(len(df.columns))))
        return df
    except pd.errors.EmptyDataError:
        sources.append(SourceRecord(name=name, path=found, status="empty_file", rows=0, cols=0))
        return pd.DataFrame()
    except Exception as exc:
        sources.append(SourceRecord(name=name, path=found, status="read_error", note=f"{type(exc).__name__}: {exc}"))
        return pd.DataFrame()


def read_json_safe(path: Path, name: str, sources: List[SourceRecord]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        sources.append(SourceRecord(name=name, path=None, status="missing", note=str(path)))
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        status = "ok_empty" if not obj else "ok"
        sources.append(SourceRecord(name=name, path=p, status=status, rows=None, cols=None))
        return obj if isinstance(obj, dict) else {"_json_value": obj}
    except Exception as exc:
        sources.append(SourceRecord(name=name, path=p, status="read_error", note=f"{type(exc).__name__}: {exc}"))
        return {}


def is_finite_number(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def coerce_float(x: Any, default: float = np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def coerce_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "y", "t"}:
            return True
        if s in {"0", "false", "no", "n", "f"}:
            return False
    return None


def fmt_value(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return "true" if bool(x) else "false"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)) or is_finite_number(x):
        y = coerce_float(x)
        if not np.isfinite(y):
            return "NA"
        ay = abs(y)
        if ay == 0:
            return "0"
        if ay < 1e-3 or ay >= 1e4:
            return f"{y:.{digits}e}"
        return f"{y:.{digits}g}"
    return str(x)


def nested_get(obj: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def first_present(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def filter_true(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            vals = df[col].map(coerce_bool)
            mask = mask | vals.fillna(False).astype(bool)
    return df.loc[mask].copy()


def find_family_row(df: pd.DataFrame, family_key: str) -> pd.DataFrame:
    if df is None or df.empty or "family_key" not in df.columns or not family_key:
        return pd.DataFrame()
    return df[df["family_key"].astype(str).eq(str(family_key))].copy()


def clean_cell(x: Any, max_len: int = 72) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float) and not np.isfinite(x):
        return "NA"
    s = fmt_value(x) if not isinstance(x, str) else x
    s = s.replace("\n", " ").replace("|", "\\|")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def markdown_table(df: pd.DataFrame, columns: Optional[Sequence[str]] = None, max_rows: Optional[int] = None) -> str:
    if df is None or df.empty:
        return "_No rows available._"
    d = df.copy()
    if columns is not None:
        keep = [c for c in columns if c in d.columns]
        if not keep:
            return "_Requested columns unavailable._"
        d = d[keep]
    if max_rows is not None and len(d) > max_rows:
        d = d.head(max_rows).copy()
        trunc = True
    else:
        trunc = False
    headers = list(d.columns)
    rows = []
    for _, r in d.iterrows():
        rows.append([clean_cell(r.get(c)) for c in headers])
    out = []
    out.append("| " + " | ".join(clean_cell(c, 60) for c in headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    if trunc:
        out.append(f"\n_Table truncated to first {max_rows} rows._")
    return "\n".join(out)


def add_metric(
    ledger: List[Dict[str, Any]],
    category: str,
    metric: str,
    value: Any,
    source: str,
    priority: str,
    interpretation: str,
    manuscript_use: str,
    unit: str = "",
) -> None:
    ledger.append({
        "priority": priority,
        "category": category,
        "metric": metric,
        "value": value,
        "unit": unit,
        "source": source,
        "interpretation": interpretation,
        "manuscript_use": manuscript_use,
    })


def add_from_mapping(
    ledger: List[Dict[str, Any]],
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    category: str,
    label: str,
    source: str,
    priority: str,
    interpretation: str,
    manuscript_use: str,
    unit: str = "",
) -> None:
    val = first_present(mapping, keys, default=None)
    add_metric(ledger, category, label, val, source, priority, interpretation, manuscript_use, unit)


def user_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    if df is None or df.empty or "user_id" not in df.columns:
        return np.ones(0 if df is None else len(df), dtype=float)
    counts = df.groupby("user_id")["user_id"].transform("count").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    aa = a[ok] - float(np.mean(a[ok]))
    bb = b[ok] - float(np.mean(b[ok]))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= EPS:
        return np.nan
    return float(np.dot(aa, bb) / den)


def read_matrix_safe(base: Path, name: str, sources: List[SourceRecord]) -> np.ndarray:
    df = read_table_safe(base, name, sources)
    if df.empty:
        return np.zeros((0, 0), dtype=float)
    # Drop accidental index columns from CSV round-trips.
    cols = [c for c in df.columns if not str(c).lower().startswith("unnamed")]
    arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return arr


def transition_statewise_metrics(P_emp: np.ndarray, P_mech: np.ndarray) -> pd.DataFrame:
    if P_emp.size == 0 or P_mech.size == 0 or P_emp.shape != P_mech.shape:
        return pd.DataFrame()
    k = P_emp.shape[0]
    rows: List[Dict[str, Any]] = []
    for s in range(k):
        pe = float(P_emp[s, s])
        pm = float(P_mech[s, s])
        row_tv = 0.5 * float(np.nansum(np.abs(P_mech[s, :] - P_emp[s, :])))
        emp_dom = bool(pe >= np.nanmax(P_emp[s, :]) - 1e-12)
        mech_dom = bool(pm >= np.nanmax(P_mech[s, :]) - 1e-12)
        emp_geo_mean = 1.0 / max(1.0 - min(pe, 1.0 - 1e-9), EPS)
        mech_geo_mean = 1.0 / max(1.0 - min(pm, 1.0 - 1e-9), EPS)
        rows.append({
            "macrostate": f"S{s}",
            "emp_self_transition": pe,
            "mech_self_transition": pm,
            "self_transition_difference": pm - pe,
            "row_total_variation": row_tv,
            "emp_diagonal_dominant": emp_dom,
            "mech_diagonal_dominant": mech_dom,
            "emp_geometric_mean_residence": emp_geo_mean,
            "mech_geometric_mean_residence": mech_geo_mean,
            "geometric_mean_residence_ratio_mech_over_emp": mech_geo_mean / emp_geo_mean if emp_geo_mean > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def source_audit_table(sources: List[SourceRecord]) -> pd.DataFrame:
    rows = []
    for s in sources:
        rows.append({
            "name": s.name,
            "status": s.status,
            "rows": s.rows,
            "columns": s.cols,
            "path_or_note": str(s.path) if s.path is not None else s.note,
            "note": s.note if s.path is not None else "",
        })
    return pd.DataFrame(rows)


def ledger_df(ledger: List[Dict[str, Any]]) -> pd.DataFrame:
    d = pd.DataFrame(ledger)
    if d.empty:
        return d
    priority_order = {MAIN_REQUIRED: 0, MAIN_RECOMMENDED: 1, OPTIONAL_STRONG: 2, SUPPLEMENT_REQUIRED: 3, AUDIT_ONLY: 4}
    d["_priority_order"] = d["priority"].map(priority_order).fillna(99)
    d = d.sort_values(["_priority_order", "category", "metric"], kind="mergesort").drop(columns=["_priority_order"])
    d["formatted_value"] = d["value"].map(fmt_value)
    return d


def report_decision_text() -> str:
    return (
        "**Figure-use decision.** Transition recovery is suitable for the main Figure 3 as a residual matrix, "
        "because it is compact, directly implied by the frozen one-step mechanism, and mirrors the Figure 1 "
        "empirical transition diagnostic without claiming that transition was used for tuning. Residence recovery "
        "should primarily be reported numerically in the main text and shown fully in Supplementary/Additional "
        "information. The reason is not that residence is weak; rather, the mechanism residence curve is a "
        "state-matched geometric reference implied by the mechanism one-step transition matrix, not a free-running "
        "mechanism residence distribution. If a residence visual remains in the main figure, the caption must state "
        "this boundary explicitly and call it a post-selection kinetic check."
    )



def validate_formal_contracts(
    handoff: Mapping[str, Any],
    phase1_manifest: Mapping[str, Any],
    phase2_manifest: Mapping[str, Any],
    phase3_manifest: Mapping[str, Any],
    phase3_audit: Mapping[str, Any],
    kmeans_metadata: Mapping[str, Any],
) -> None:
    required_handoff = (
        "ready_for_phase2_freeze",
        "scalar_parameter_minimality_confirmed",
        "search_adequacy_confirmed",
        "baseline_not_practically_equivalent_to_best",
        "final_model_selected_by_parsimony_rule",
    )
    failed = [name for name in required_handoff if coerce_bool(handoff.get(name)) is not True]
    if failed:
        raise RuntimeError(f"Minimality handoff failed required gates: {failed}")
    if str(handoff.get("final_family_key", "")) != "offset_dual_channel":
        raise RuntimeError("Minimality handoff did not select offset_dual_channel.")
    if tuple(phase1_manifest.get("primary_macrostate", [])) != ("M", "Psi"):
        raise RuntimeError("Phase-1 primary macrostate is not ['M', 'Psi'].")
    if tuple(phase2_manifest.get("primary_macrostate", [])) != ("M", "Psi"):
        raise RuntimeError("Phase-2 primary macrostate is not ['M', 'Psi'].")
    if tuple(phase3_manifest.get("primary_macrostate", [])) != ("M", "Psi"):
        raise RuntimeError("Phase-3 primary macrostate is not ['M', 'Psi'].")
    required_false = (
        "parameter_search_opened",
        "calibration_reestimated",
        "mechanism_family_reselected",
        "mechanism_parameters_refit",
        "kmeans_refit",
        "kmeans_k_selection",
        "macrostate_k_selected",
        "region_redefinition",
        "B_confirm_used_for_model_update",
    )
    failed_audit = [name for name in required_false if bool(phase3_audit.get(name, False))]
    if failed_audit:
        raise RuntimeError(f"Phase-3 no-update audit failed: {failed_audit}")
    checks = {
        "coordinate": kmeans_metadata.get("coordinate") == "MR_PsiA",
        "macrostate_k": int(kmeans_metadata.get("macrostate_k", -1)) == 6,
        "macrostate_k_rule": kmeans_metadata.get("macrostate_k_rule") == "fixed a priori",
        "fit_split": kmeans_metadata.get("fit_split") == "A_train",
        "features": list(kmeans_metadata.get("features", [])) == [
            "M_response_prebalanced_pre",
            "activity_alignment_order_Psi_pre",
        ],
        "user_balanced_sampling": kmeans_metadata.get("user_balanced_sampling") is True,
        "user_balanced_kmeans_fit": kmeans_metadata.get("user_balanced_kmeans_fit") is True,
        "kmeans_n_init": int(kmeans_metadata.get("kmeans_n_init", -1)) == 20,
        "fit_max_rows": int(kmeans_metadata.get("fit_max_rows", -1)) == 500000,
        "random_state": int(kmeans_metadata.get("random_state", -1)) == 42,
    }
    failed_kmeans = [name for name, passed in checks.items() if not passed]
    if failed_kmeans:
        raise RuntimeError(f"Stage-1 fixed K=6 contract failed: {failed_kmeans}")


def compute_prediction_extra_metrics(predictions: pd.DataFrame) -> Dict[str, Any]:
    required = [
        "user_id",
        "M",
        "Psi",
        "target_M_next",
        "target_Psi_next",
        "pred_next_M",
        "pred_next_Psi",
    ]
    if predictions is None or predictions.empty or any(name not in predictions.columns for name in required):
        return {"available": False}
    table = predictions[required].copy()
    for name in required:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    valid = table["user_id"].notna()
    valid &= np.isfinite(table[["M", "Psi", "target_M_next", "target_Psi_next", "pred_next_M", "pred_next_Psi"]]).all(axis=1)
    table = table.loc[valid].copy()
    if table.empty:
        return {"available": False}
    weights = user_balanced_weights(table)
    weights = weights / max(float(weights.sum()), EPS)
    output: Dict[str, Any] = {
        "available": True,
        "prediction_rows_valid": int(len(table)),
        "prediction_users_valid": int(table["user_id"].nunique()),
    }
    for label, target, prediction in (
        ("M", "target_M_next", "pred_next_M"),
        ("Psi", "target_Psi_next", "pred_next_Psi"),
    ):
        error = table[prediction].to_numpy(dtype=float) - table[target].to_numpy(dtype=float)
        absolute = np.abs(error)
        output[f"interval_weighted_rmse_{label}"] = float(np.sqrt(np.mean(error ** 2)))
        output[f"interval_weighted_mae_{label}"] = float(np.mean(absolute))
        output[f"user_balanced_rmse_{label}"] = float(np.sqrt(np.sum(weights * error ** 2)))
        output[f"user_balanced_mae_{label}"] = float(np.sum(weights * absolute))
        output[f"user_balanced_bias_{label}"] = float(np.sum(weights * error))
        output[f"one_step_abs_error_q50_{label}"] = float(np.quantile(absolute, 0.50))
        output[f"one_step_abs_error_q90_{label}"] = float(np.quantile(absolute, 0.90))
        output[f"one_step_abs_error_q95_{label}"] = float(np.quantile(absolute, 0.95))
        output[f"next_state_corr_{label}"] = pearson_safe(
            table[target].to_numpy(dtype=float),
            table[prediction].to_numpy(dtype=float),
        )
    return output


def make_metric_ledger(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[SourceRecord], Dict[str, pd.DataFrame]]:
    sources: List[SourceRecord] = []
    tables: Dict[str, pd.DataFrame] = {}
    ledger: List[Dict[str, Any]] = []
    figure_root = args.figure_root if args.figure_root is not None else args.phase3_root / "figures_publication_minimal_mechanism"
    figure_table_root = figure_root / "tables"

    minimality_manifest = read_json_safe(args.minimality_root / "metadata" / "minimality_experiment_manifest.json", "minimality_experiment_manifest", sources)
    handoff = read_json_safe(args.minimality_root / "metadata" / "phase1_minimal_mechanism_handoff.json", "phase1_minimal_mechanism_handoff", sources)
    phase1_selected = read_json_safe(args.phase1_root / "metadata" / "phase1_selected_parameters.json", "phase1_selected_parameters", sources)
    phase1_manifest = read_json_safe(args.phase1_root / "metadata" / "phase1_manifest.json", "phase1_manifest", sources)
    phase2_manifest = read_json_safe(args.phase2_root / "metadata" / "phase2_frozen_model_manifest.json", "phase2_frozen_model_manifest", sources)
    phase2_calibration = read_json_safe(args.phase2_root / "metadata" / "phase2_pooled_development_calibration.json", "phase2_pooled_development_calibration", sources)
    phase3_manifest = read_json_safe(args.phase3_root / "metadata" / "phase3_confirmation_manifest.json", "phase3_confirmation_manifest", sources)
    phase3_audit = read_json_safe(args.phase3_root / "metadata" / "phase3_no_update_audit.json", "phase3_no_update_audit", sources)
    figure_manifest = read_json_safe(figure_root / "publication_minimal_mechanism_figure_manifest.json", "publication_minimal_mechanism_figure_manifest", sources)
    kmeans_metadata = read_json_safe(args.stage1_root / "dynamics" / "fixed_k6_mesostates" / "fixed_k6_model_metadata.json", "stage1_fixed_k6_model_metadata", sources)

    validate_formal_contracts(
        handoff,
        phase1_manifest,
        phase2_manifest,
        phase3_manifest,
        phase3_audit,
        kmeans_metadata,
    )

    table_sources = [
        ("manuscript_results_summary", args.minimality_root / "tables" / "manuscript_results_summary"),
        ("model_family_results", args.minimality_root / "tables" / "model_family_results"),
        ("nested_mechanism_contrasts", args.minimality_root / "tables" / "nested_mechanism_contrasts"),
        ("selected_model_parameter_deletions", args.minimality_root / "tables" / "selected_model_parameter_deletions"),
        ("global_scalar_deletion_audit", args.minimality_root / "tables" / "global_scalar_deletion_audit"),
        ("parameter_grid_boundaries", args.minimality_root / "tables" / "parameter_grid_boundaries"),
        ("equivalence_margin_sensitivity", args.minimality_root / "tables" / "equivalence_margin_sensitivity"),
        ("next_required_tests", args.minimality_root / "tables" / "next_required_tests"),
        ("phase1_structural_alignment_metrics", args.phase1_root / "tables" / "phase1_structural_alignment_metrics"),
        ("phase2_development_structural_alignment_metrics", args.phase2_root / "tables" / "phase2_development_structural_alignment_metrics"),
        ("phase3_structural_alignment_metrics", args.phase3_root / "tables" / f"phase3_{args.confirm_split}_structural_alignment_metrics"),
        ("phase3_development_vs_confirmation_metric_stability", args.phase3_root / "tables" / "phase3_development_vs_confirmation_metric_stability"),
        ("phase3_prediction_audit_sample", args.phase3_root / "tables" / f"phase3_{args.confirm_split}_prediction_audit_sample"),
        ("phase3_full_predictions", args.phase3_root / "tables" / f"phase3_{args.confirm_split}_full_predictions"),
        ("figure2_field_recovery_metrics", figure_table_root / "figure2_field_recovery_metrics"),
        ("figure3_kinetic_recovery_metrics", figure_table_root / "figure3_kinetic_recovery_metrics"),
        ("phase3_formal_confirmation_metrics", figure_table_root / "phase3_formal_confirmation_metrics"),
        ("empirical_kmeans_partition_fit_table", figure_table_root / "empirical_kmeans_partition_fit_table"),
        ("empirical_kmeans_partition_centers", figure_table_root / "empirical_kmeans_partition_centers"),
    ]
    for name, base in table_sources:
        tables[name] = read_table_safe(base, name, sources)

    empirical_matrix = read_matrix_safe(
        figure_table_root / "B_confirm_empirical_mesostate_transition_matrix",
        "B_confirm_empirical_mesostate_transition_matrix",
        sources,
    )
    mechanism_matrix = read_matrix_safe(
        figure_table_root / "B_confirm_mechanism_mesostate_transition_matrix",
        "B_confirm_mechanism_mesostate_transition_matrix",
        sources,
    )
    statewise_transition = transition_statewise_metrics(empirical_matrix, mechanism_matrix)
    tables["statewise_transition_metrics"] = statewise_transition
    extra = compute_prediction_extra_metrics(tables.get("phase3_full_predictions", pd.DataFrame()))

    source_names = {
        "minimality": "minimality handoff/manifest and model-family tables",
        "phase1": "Phase-1 selected-parameter output",
        "phase2": "Phase-2 frozen manifest/development metrics",
        "phase3": "Phase-3 B_confirm confirmation metrics",
        "figure2": "publication Figure 2 metric table",
        "figure3": "publication Figure 3 metric table",
        "extra": "computed from the formal Phase-3 full prediction table",
    }

    final_family = str(handoff.get("final_family_key", ""))
    final_label = str(handoff.get("final_family_label", ""))
    free_params = list(handoff.get("final_free_mechanism_parameters", []))
    add_metric(ledger, "bounded minimality", "final family", final_family, source_names["minimality"], MAIN_REQUIRED, "Selected family after the bounded ablation protocol.", "Report in the first minimal-mechanism Results sentence.")
    add_metric(ledger, "bounded minimality", "final family label", final_label, source_names["minimality"], MAIN_RECOMMENDED, "Publication label for the selected family.", "Use in Results and the Figure 2 caption.")
    add_metric(ledger, "bounded minimality", "free substantive mechanism parameter count", len(free_params), source_names["minimality"], MAIN_REQUIRED, "Number of active mechanism parameters; fixed scales are excluded.", "Report with the minimality result.")
    add_metric(ledger, "bounded minimality", "free substantive mechanism parameters", "; ".join(map(str, free_params)), source_names["minimality"], MAIN_REQUIRED, "Active terms retained by the family-bounded rule.", "Report with the mechanism definition.")
    for key, label in (
        ("ready_for_phase2_freeze", "ready for Phase-2 freeze"),
        ("scalar_parameter_minimality_confirmed", "scalar parameter minimality confirmed"),
        ("search_adequacy_confirmed", "search adequacy confirmed"),
        ("baseline_not_practically_equivalent_to_best", "baseline not practically equivalent to best"),
    ):
        add_metric(ledger, "bounded minimality", label, handoff.get(key), source_names["minimality"], MAIN_REQUIRED, "Formal minimality gate.", "Report in Methods or Additional information.")
    add_metric(ledger, "bounded minimality", "practical-equivalence margin", minimality_manifest.get("practical_equivalence_margin"), source_names["minimality"], MAIN_REQUIRED, "Pre-specified family-equivalence margin.", "Report with the paired-bootstrap protocol.")
    add_metric(ledger, "bounded minimality", "bootstrap repetitions", minimality_manifest.get("bootstrap_reps"), source_names["minimality"], MAIN_REQUIRED, "Paired user-bootstrap repetitions.", "Report with uncertainty.")

    manuscript = tables.get("manuscript_results_summary", pd.DataFrame())
    selected_rows = filter_true(manuscript, ["Final scalar-minimal family", "Parsimonious family selected"])
    if selected_rows.empty:
        selected_rows = find_family_row(manuscript, final_family)
    if not selected_rows.empty:
        row = selected_rows.iloc[0].to_dict()
        for column, label, priority, interpretation in (
            ("Bootstrap mean primary score", "selected-family bootstrap mean primary score", MAIN_REQUIRED, "Selected-family primary-structure score."),
            ("Bootstrap 95% CI lower", "selected-family bootstrap 95% CI lower", MAIN_REQUIRED, "Lower paired-bootstrap interval endpoint."),
            ("Bootstrap 95% CI upper", "selected-family bootstrap 95% CI upper", MAIN_REQUIRED, "Upper paired-bootstrap interval endpoint."),
            ("Landscape divergence", "selected-family validation landscape divergence", MAIN_RECOMMENDED, "Validation landscape component."),
            ("Local drift discrepancy", "selected-family validation local drift discrepancy", MAIN_RECOMMENDED, "Validation local-flow component."),
            ("Drift-direction discrepancy", "selected-family validation drift-direction discrepancy", MAIN_RECOMMENDED, "Validation direction component."),
            ("Drift-speed discrepancy", "selected-family validation drift-speed discrepancy", MAIN_RECOMMENDED, "Validation speed component."),
            ("Within one standard error of best", "selected family within one-SE of best", MAIN_REQUIRED, "One-standard-error eligibility."),
            ("Practically equivalent to best", "selected family practically equivalent to best", MAIN_REQUIRED, "Practical-equivalence eligibility."),
        ):
            add_metric(ledger, "bounded minimality", label, row.get(column), "manuscript_results_summary", priority, interpretation, "Use in Results or the Figure 2 caption.")

    family_results = tables.get("model_family_results", pd.DataFrame())
    if not family_results.empty and "Bootstrap mean primary score" in family_results.columns:
        best = family_results.sort_values("Bootstrap mean primary score", kind="mergesort").iloc[0].to_dict()
        add_metric(ledger, "bounded minimality", "best family by bootstrap mean", best.get("family_key"), "model_family_results", MAIN_RECOMMENDED, "Best-scoring reference before parsimony selection.", "Report when contrasting the selected and reference families.")
        add_metric(ledger, "bounded minimality", "best-family bootstrap mean primary score", best.get("Bootstrap mean primary score"), "model_family_results", MAIN_RECOMMENDED, "Best reference-family score.", "Report if space permits.")
        if not selected_rows.empty:
            selected_mean = pd.to_numeric(selected_rows.iloc[0].get("Bootstrap mean primary score", np.nan), errors="coerce")
            best_mean = pd.to_numeric(best.get("Bootstrap mean primary score", np.nan), errors="coerce")
            if np.isfinite(selected_mean) and np.isfinite(best_mean):
                add_metric(ledger, "bounded minimality", "selected-minus-best bootstrap mean primary score", float(selected_mean - best_mean), "manuscript_results_summary/model_family_results", MAIN_RECOMMENDED, "Score gap between the parsimonious selected family and the best-scoring reference family.", "Report when emphasizing bounded parsimony.")

    frozen = dict(phase2_manifest.get("frozen_parameters", {}))
    vector = dict(frozen.get("full_parameter_vector", {}))
    active = dict(frozen.get("mechanism_parameters", {}))
    zeros = dict(frozen.get("structural_zero_values", {}))
    nuisance = dict(frozen.get("fixed_nuisance_scales", {}))
    for name in ("theta0", "thetaM", "phi0", "deltaS"):
        add_metric(ledger, "frozen mechanism parameters", name, active.get(name, vector.get(name)), source_names["phase2"], MAIN_REQUIRED, "Frozen active mechanism parameter.", "Report in Results or Methods.")
    add_metric(ledger, "frozen mechanism parameters", "structural-zero parameters", "; ".join(f"{name}={fmt_value(value)}" for name, value in zeros.items()), source_names["phase2"], MAIN_REQUIRED, "Ablated terms fixed at zero.", "Report with the selected family.")
    for name in ("lambdaR", "lambdaA", "lambdaI"):
        add_metric(ledger, "frozen mechanism parameters", name, nuisance.get(name, vector.get(name)), source_names["phase2"], MAIN_RECOMMENDED, "Fixed accounting scale; not counted as a mechanism parameter.", "Report in Methods or Supplement.")
    calibration = dict(phase2_manifest.get("frozen_calibration", {})) or phase2_calibration
    for name in ("eta", "tau_response_days", "tau_activity_days", "residual_mass_per_answer", "lambda_E", "response_signed_gain", "alignment_signed_gain", "sigma_U0", "sigma_Psi0"):
        priority = MAIN_RECOMMENDED if name in {"eta", "tau_response_days", "tau_activity_days"} else SUPPLEMENT_REQUIRED
        add_metric(ledger, "frozen calibration/accounting", name, calibration.get(name), source_names["phase2"], priority, "Calibration frozen after pooled-development estimation.", "Report the complete list in Supplement.")

    for prefix, manifest in (("Phase-2", phase2_manifest), ("Phase-3", phase3_manifest)):
        guardrails = dict(manifest.get("guardrails", {}))
        for name in ("B_confirm_read", "B_confirm_used_for_update", "parameter_search_opened", "calibration_reestimated", "mechanism_family_reselected", "mechanism_parameters_refit", "kmeans_refit", "kmeans_k_selection", "macrostate_k_selected", "region_redefinition", "confirmation_mode", "calibration_refit_scope"):
            if name not in guardrails:
                continue
            priority = MAIN_REQUIRED if name in {"B_confirm_read", "B_confirm_used_for_update", "parameter_search_opened", "calibration_reestimated", "mechanism_parameters_refit", "confirmation_mode", "calibration_refit_scope"} else SUPPLEMENT_REQUIRED
            add_metric(ledger, "freeze/confirmation guardrails", f"{prefix} {name}", guardrails.get(name), f"{prefix} manifest", priority, "Freeze or confirmation guardrail.", "Report briefly; retain the full audit in Additional information.")
    for name in ("frozen_parameter_hash_before_confirmation", "frozen_parameter_hash_after_confirmation", "frozen_calibration_hash_before_confirmation", "frozen_calibration_hash_after_confirmation"):
        add_metric(ledger, "freeze/confirmation guardrails", name, phase3_audit.get(name), "phase3_no_update_audit", AUDIT_ONLY, "No-update hash audit.", "Additional information only.")

    phase2_metrics = tables.get("phase2_development_structural_alignment_metrics", pd.DataFrame())
    if not phase2_metrics.empty:
        for label in ("A_train", "A_val", "A_train_plus_A_val"):
            subset = phase2_metrics[phase2_metrics["label"].astype(str) == label] if "label" in phase2_metrics.columns else pd.DataFrame()
            if subset.empty:
                continue
            row = subset.iloc[0].to_dict()
            for name in ("objective_primary_score", "objective_loss", "one_step_mse_main_norm", "one_step_rmse_M", "one_step_rmse_Psi", "occupancy_js_MR_PsiA", "drift_vector_corr_MR_PsiA", "drift_local_rmse_loss_MR_PsiA", "drift_direction_loss_MR_PsiA", "drift_magnitude_loss_MR_PsiA", "phase_loss_max_qdist", "coverage_loss_max_qdist", "n_rows", "n_users"):
                if name in row:
                    add_metric(ledger, f"development structural metrics/{label}", name, row.get(name), "phase2_development_structural_alignment_metrics", MAIN_RECOMMENDED if label == "A_train_plus_A_val" else SUPPLEMENT_REQUIRED, "Frozen-development structural metric.", "Use the pooled-development row as the main reference.")

    phase3_metrics: Dict[str, Any] = {}
    for table_name in ("phase3_structural_alignment_metrics", "phase3_formal_confirmation_metrics", "figure2_field_recovery_metrics"):
        table = tables.get(table_name, pd.DataFrame())
        if not table.empty:
            for name, value in table.iloc[0].to_dict().items():
                phase3_metrics.setdefault(name, value)
    for keys, label, priority, interpretation in (
        (("n_rows",), "B_confirm valid one-step rows", MAIN_REQUIRED, "Confirmation rows."),
        (("n_users",), "B_confirm users", MAIN_REQUIRED, "Confirmation users."),
        (("one_step_rmse_M",), "one-step RMSE: response-order coordinate", MAIN_REQUIRED, "Interval-weighted response-coordinate RMSE."),
        (("one_step_rmse_Psi",), "one-step RMSE: exposure-alignment coordinate", MAIN_REQUIRED, "Interval-weighted exposure-coordinate RMSE."),
        (("one_step_rmse_V_diagnostic_only",), "one-step RMSE: maturity diagnostic", MAIN_RECOMMENDED, "Accounting diagnostic, not a phase-plane target."),
        (("occupancy_js_MR_PsiA", "next_state_occupancy_js"), "next-state occupancy Jensen-Shannon divergence", MAIN_REQUIRED, "Next-state landscape recovery."),
        (("drift_vector_corr_MR_PsiA", "drift_vector_corr"), "drift vector correlation", MAIN_REQUIRED, "Global drift-field vector agreement."),
        (("drift_local_rmse_loss_MR_PsiA", "drift_local_rmse"), "local drift RMSE/loss", MAIN_REQUIRED, "Local drift discrepancy."),
        (("drift_direction_loss_MR_PsiA",), "drift direction loss", MAIN_REQUIRED, "Direction component of the primary objective."),
        (("drift_magnitude_loss_MR_PsiA",), "drift magnitude loss", MAIN_REQUIRED, "Speed component of the primary objective."),
        (("drift_speed_corr",), "drift speed correlation", MAIN_REQUIRED, "Cellwise drift-speed agreement."),
        (("common_drift_cells",), "common supported drift cells", MAIN_REQUIRED, "Cells used in the direct field comparison."),
        (("objective_primary_score",), "B_confirm objective primary score", MAIN_REQUIRED, "Confirmation primary score."),
        (("objective_loss",), "B_confirm objective loss", MAIN_RECOMMENDED, "Confirmation full objective."),
    ):
        add_from_mapping(ledger, phase3_metrics, keys, "B_confirm field recovery", label, source_names["phase3"] + "/" + source_names["figure2"], priority, interpretation, "Report in the field-recovery Results paragraph.")

    for name, label, priority, interpretation in (
        ("interval_weighted_mae_M", "interval-weighted MAE: response-order coordinate", SUPPLEMENT_REQUIRED, "MAE under the same interval weighting as the reported RMSE."),
        ("interval_weighted_mae_Psi", "interval-weighted MAE: exposure-alignment coordinate", SUPPLEMENT_REQUIRED, "MAE under the same interval weighting as the reported RMSE."),
        ("user_balanced_rmse_M", "user-balanced RMSE: response-order coordinate", SUPPLEMENT_REQUIRED, "User-balanced RMSE."),
        ("user_balanced_rmse_Psi", "user-balanced RMSE: exposure-alignment coordinate", SUPPLEMENT_REQUIRED, "User-balanced RMSE."),
        ("user_balanced_mae_M", "user-balanced MAE: response-order coordinate", MAIN_RECOMMENDED, "User-balanced absolute one-step error."),
        ("user_balanced_mae_Psi", "user-balanced MAE: exposure-alignment coordinate", MAIN_RECOMMENDED, "User-balanced absolute one-step error."),
        ("next_state_corr_M", "next-state correlation: response-order coordinate", OPTIONAL_STRONG, "Direct next-state correlation."),
        ("next_state_corr_Psi", "next-state correlation: exposure-alignment coordinate", OPTIONAL_STRONG, "Direct next-state correlation."),
    ):
        if name in extra:
            add_metric(ledger, "B_confirm full-prediction derived checks", label, extra.get(name), source_names["extra"], priority, interpretation, "Use in Results only when the weighting convention is named.")

    stability_table = tables.get("phase3_development_vs_confirmation_metric_stability", pd.DataFrame())
    if not stability_table.empty and {"metric", "development_value", "confirmation_value"}.issubset(stability_table.columns):
        label_map = {m: lab for m, lab in STABILITY_PANEL_METRICS}
        for metric_name, label in STABILITY_PANEL_METRICS:
            subset = stability_table[stability_table["metric"].astype(str) == metric_name]
            if subset.empty:
                continue
            row = subset.iloc[0]
            dev_value = pd.to_numeric(row.get("development_value", np.nan), errors="coerce")
            con_value = pd.to_numeric(row.get("confirmation_value", np.nan), errors="coerce")
            if np.isfinite(dev_value):
                add_metric(ledger, "development-confirmation stability", f"validation {label}", float(dev_value), "phase3_development_vs_confirmation_metric_stability", MAIN_RECOMMENDED, "Validation-set reference value for the confirmation-stability panel.", "Report with the paired confirmation value when discussing robustness.")
            if np.isfinite(con_value):
                add_metric(ledger, "development-confirmation stability", f"confirmation {label}", float(con_value), "phase3_development_vs_confirmation_metric_stability", MAIN_RECOMMENDED, "Confirmation-set value for the confirmation-stability panel.", "Report with the validation value when discussing robustness.")
            if np.isfinite(dev_value) and np.isfinite(con_value):
                add_metric(ledger, "development-confirmation stability", f"absolute gap {label}", float(abs(con_value - dev_value)), "phase3_development_vs_confirmation_metric_stability", SUPPLEMENT_REQUIRED, "Absolute validation-to-confirmation gap.", "Use when a robustness statement needs an explicit gap.")

    figure3_table = tables.get("figure3_kinetic_recovery_metrics", pd.DataFrame())
    figure3_metrics = figure3_table.iloc[0].to_dict() if not figure3_table.empty else {}
    for keys, label, priority, interpretation in (
        (("macrostate_k",), "coarse macrostates K", MAIN_REQUIRED, "Fixed Stage-1 mesostate count."),
        (("transition_count",), "B_confirm mesostate transition count", MAIN_RECOMMENDED, "Frozen empirical transitions."),
        (("transition_mean_row_tv",), "transition mean row-wise total variation", MAIN_RECOMMENDED, "Mean transition-matrix discrepancy."),
        (("transition_max_row_tv",), "transition max row-wise total variation", MAIN_RECOMMENDED, "Worst transition-row discrepancy."),
        (("self_transition_rmse",), "self-transition RMSE", MAIN_RECOMMENDED, "Persistence-probability RMSE."),
        (("self_transition_mae",), "self-transition MAE", MAIN_RECOMMENDED, "Persistence-probability MAE."),
        (("self_transition_correlation",), "self-transition correlation", MAIN_RECOMMENDED, "Statewise persistence ordering."),
        (("diagonal_dominant_states_empirical",), "empirical diagonal-dominant states", MAIN_RECOMMENDED, "Empirical diagonal-dominant rows."),
        (("diagonal_dominant_states_mechanism",), "mechanism diagonal-dominant states", MAIN_RECOMMENDED, "Mechanism diagonal-dominant rows."),
        (("diagonal_dominance_recall",), "diagonal-dominance recall", MAIN_RECOMMENDED, "Recovery of empirical diagonal dominance."),
        (("top_transition_edge_overlap",), "top transition edge overlap", SUPPLEMENT_REQUIRED, "Overlap among strongest transition edges."),
        (("residence_reference_mean_abs_log_difference",), "residence-reference mean absolute log difference", MAIN_RECOMMENDED, "Difference between empirical and mechanism transition-implied geometric references."),
        (("residence_reference_concordance",), "residence-reference concordance", MAIN_RECOMMENDED, "Exponentiated residence-reference discrepancy."),
        (("residence_source",), "residence source", MAIN_REQUIRED, "Source of the fixed-K residence support."),
        (("empirical_transition_reconstruction_max_abs_difference",), "empirical transition reconstruction maximum difference", AUDIT_ONLY, "Check that Phase-3 target states reproduce the frozen Stage-1 transition matrix."),
    ):
        add_from_mapping(ledger, figure3_metrics, keys, "post-selection kinetic recovery", label, source_names["figure3"], priority, interpretation, "Report only as a post-selection kinetic check.")

    centers = tables.get("empirical_kmeans_partition_centers", pd.DataFrame())
    fit_table = tables.get("empirical_kmeans_partition_fit_table", pd.DataFrame())
    add_metric(ledger, "mesostate partition audit", "fixed macrostate K", int(kmeans_metadata.get("macrostate_k", 0)), "Stage-1 fixed-K metadata", MAIN_REQUIRED, "K fixed before transition and residence analysis.", "Report in Methods or the Figure 3 caption.")
    add_metric(ledger, "mesostate partition audit", "KMeans centers available", int(len(centers)), "empirical_kmeans_partition_centers", SUPPLEMENT_REQUIRED, "Frozen Stage-1 centers copied by the publication script.", "Additional information.")
    add_metric(ledger, "mesostate partition audit", "KMeans refit in publication code", False, "publication figure manifest", MAIN_REQUIRED, "The publication layer reuses the frozen Stage-1 partition.", "Mention in the Figure 3 caption or Methods.")
    add_metric(ledger, "mesostate partition audit", "K selected in publication code", False, "publication figure manifest", MAIN_REQUIRED, "K remains fixed at six.", "Mention in Methods.")
    if not fit_table.empty:
        add_metric(ledger, "mesostate partition audit", "fixed-K fit rows", first_present(fit_table.iloc[0].to_dict(), ["fit_rows"]), "Stage-1 fixed-K fit table", SUPPLEMENT_REQUIRED, "Rows used by the frozen A_train partition fit.", "Additional information.")

    bundle = {
        "minimality_manifest": minimality_manifest,
        "handoff": handoff,
        "phase1_selected": phase1_selected,
        "phase1_manifest": phase1_manifest,
        "phase2_manifest": phase2_manifest,
        "phase2_calibration": phase2_calibration,
        "phase3_manifest": phase3_manifest,
        "phase3_audit": phase3_audit,
        "figure_manifest": figure_manifest,
        "kmeans_metadata": kmeans_metadata,
        "figure_root": figure_root,
        "figure_table_root": figure_table_root,
        "P_emp": empirical_matrix,
        "P_mech": mechanism_matrix,
        "extra_prediction_metrics": extra,
    }
    return ledger, bundle, sources, tables



def build_markdown(args: argparse.Namespace, ledger: List[Dict[str, Any]], bundle: Dict[str, Any], sources: List[SourceRecord], tables: Dict[str, pd.DataFrame]) -> str:
    dledger = ledger_df(ledger)
    source_df = source_audit_table(sources)
    out: List[str] = []
    out.append("# Minimal mechanism numerical report for manuscript")
    out.append("")
    out.append(f"Generated at: `{now_iso()}`")
    out.append("")
    out.append("## Scope and figure-use decision")
    out.append("")
    out.append(report_decision_text())
    out.append("")
    out.append("This report treats the minimal mechanism as a frozen conditional one-step effective closure. Transition and residence diagnostics are post-selection kinetic checks, not Phase-1 selection targets.")
    out.append("")

    out.append("## Input audit")
    out.append("")
    out.append(markdown_table(source_df, max_rows=None))
    out.append("")

    out.append("## Main-text required numbers")
    out.append("")
    req = dledger[dledger["priority"].eq(MAIN_REQUIRED)].copy() if not dledger.empty else pd.DataFrame()
    out.append(markdown_table(req, columns=["category", "metric", "formatted_value", "source", "interpretation", "manuscript_use"], max_rows=None))
    out.append("")

    out.append("## Main-text recommended numbers")
    out.append("")
    rec = dledger[dledger["priority"].eq(MAIN_RECOMMENDED)].copy() if not dledger.empty else pd.DataFrame()
    out.append(markdown_table(rec, columns=["category", "metric", "formatted_value", "source", "interpretation", "manuscript_use"], max_rows=None))
    out.append("")

    out.append("## Optional strong-claim support numbers")
    out.append("")
    opt = dledger[dledger["priority"].eq(OPTIONAL_STRONG)].copy() if not dledger.empty else pd.DataFrame()
    out.append(markdown_table(opt, columns=["category", "metric", "formatted_value", "source", "interpretation", "manuscript_use"], max_rows=None))
    out.append("")

    out.append("## Supplementary / Additional-information required numbers")
    out.append("")
    sup = dledger[dledger["priority"].eq(SUPPLEMENT_REQUIRED)].copy() if not dledger.empty else pd.DataFrame()
    out.append(markdown_table(sup, columns=["category", "metric", "formatted_value", "source", "interpretation", "manuscript_use"], max_rows=None))
    out.append("")

    out.append("## Minimality family summary table")
    out.append("")
    manuscript = tables.get("manuscript_results_summary", pd.DataFrame())
    out.append(markdown_table(manuscript, columns=[
        "family_key", "Model family", "Free mechanism parameters", "Free parameter names",
        "Bootstrap mean primary score", "Bootstrap 95% CI lower", "Bootstrap 95% CI upper",
        "Landscape divergence", "Local drift discrepancy", "Drift-direction discrepancy", "Drift-speed discrepancy",
        "Within one standard error of best", "Practically equivalent to best", "Parsimonious family selected", "Final scalar-minimal family",
    ], max_rows=None))
    out.append("")

    out.append("## Scalar deletion diagnostics")
    out.append("")
    deletion = tables.get("global_scalar_deletion_audit", pd.DataFrame())
    out.append(markdown_table(deletion, max_rows=50))
    out.append("")

    out.append("## Nested mechanism contrasts")
    out.append("")
    contrasts = tables.get("nested_mechanism_contrasts", pd.DataFrame())
    out.append(markdown_table(contrasts, max_rows=80))
    out.append("")

    out.append("## Boundary and margin sensitivity checks")
    out.append("")
    out.append("### Parameter-grid boundary checks")
    out.append(markdown_table(tables.get("parameter_grid_boundaries", pd.DataFrame()), max_rows=80))
    out.append("")
    out.append("### Practical-equivalence margin sensitivity")
    out.append(markdown_table(tables.get("equivalence_margin_sensitivity", pd.DataFrame()), max_rows=80))
    out.append("")
    out.append("### Next required tests")
    out.append(markdown_table(tables.get("next_required_tests", pd.DataFrame()), max_rows=80))
    out.append("")

    out.append("## Phase-2 development structural metrics")
    out.append("")
    phase2_metrics = tables.get("phase2_development_structural_alignment_metrics", pd.DataFrame())
    out.append(markdown_table(phase2_metrics, columns=[
        "label", "n_rows", "n_users", "objective_primary_score", "objective_loss", "one_step_mse_main_norm",
        "one_step_rmse_M", "one_step_rmse_Psi", "occupancy_js_MR_PsiA", "drift_vector_corr_MR_PsiA",
        "drift_local_rmse_loss_MR_PsiA", "drift_direction_loss_MR_PsiA", "drift_magnitude_loss_MR_PsiA",
        "phase_loss_max_qdist", "coverage_loss_max_qdist",
    ], max_rows=None))
    out.append("")

    out.append("## Phase-3 B_confirm structural confirmation metrics")
    out.append("")
    p3 = tables.get("phase3_structural_alignment_metrics", pd.DataFrame())
    out.append(markdown_table(p3, max_rows=10))
    out.append("")

    out.append("## Development-to-confirmation metric stability")
    out.append("")
    stability = tables.get("phase3_development_vs_confirmation_metric_stability", pd.DataFrame())
    out.append(markdown_table(stability, columns=[
        "reference_label", "confirm_label", "metric", "metric_direction", "development_value", "confirmation_value",
        "confirmation_minus_development", "relative_delta_vs_development", "diagnostic_degradation_amount",
    ], max_rows=120))
    out.append("")

    out.append("## Figure 2 field-recovery metrics")
    out.append("")
    out.append(markdown_table(tables.get("figure2_field_recovery_metrics", pd.DataFrame()), max_rows=10))
    out.append("")

    out.append("## Figure 3 post-selection kinetic-recovery metrics")
    out.append("")
    out.append(markdown_table(tables.get("figure3_kinetic_recovery_metrics", pd.DataFrame()), max_rows=10))
    out.append("")
    out.append("**Residence interpretation boundary:** the mechanism residence reference is geometric and transition-implied. It should not be described as a free-running mechanism residence distribution unless a separate free-running simulation is added.")
    out.append("")

    out.append("## Statewise transition and transition-implied residence metrics")
    out.append("")
    out.append(markdown_table(tables.get("statewise_transition_metrics", pd.DataFrame()), max_rows=None))
    out.append("")

    out.append("## Mesostate partition audit")
    out.append("")
    out.append("### KMeans fit table")
    out.append(markdown_table(tables.get("empirical_kmeans_partition_fit_table", pd.DataFrame()), max_rows=None))
    out.append("")
    out.append("### KMeans centers")
    out.append(markdown_table(tables.get("empirical_kmeans_partition_centers", pd.DataFrame()), max_rows=None))
    out.append("")

    P_emp = bundle.get("P_emp", np.zeros((0, 0)))
    P_mech = bundle.get("P_mech", np.zeros((0, 0)))
    if isinstance(P_emp, np.ndarray) and P_emp.size:
        out.append("## B_confirm empirical mesostate transition matrix")
        out.append("")
        out.append(markdown_table(pd.DataFrame(P_emp, columns=[f"S{j}" for j in range(P_emp.shape[1])]).assign(current_state=[f"S{i}" for i in range(P_emp.shape[0])])[["current_state"] + [f"S{j}" for j in range(P_emp.shape[1])]], max_rows=None))
        out.append("")
    if isinstance(P_mech, np.ndarray) and P_mech.size:
        out.append("## B_confirm mechanism-implied mesostate transition matrix")
        out.append("")
        out.append(markdown_table(pd.DataFrame(P_mech, columns=[f"S{j}" for j in range(P_mech.shape[1])]).assign(current_state=[f"S{i}" for i in range(P_mech.shape[0])])[["current_state"] + [f"S{j}" for j in range(P_mech.shape[1])]], max_rows=None))
        out.append("")
    if isinstance(P_emp, np.ndarray) and isinstance(P_mech, np.ndarray) and P_emp.size and P_emp.shape == P_mech.shape:
        out.append("## Transition residual matrix: mechanism minus empirical")
        out.append("")
        D = P_mech - P_emp
        out.append(markdown_table(pd.DataFrame(D, columns=[f"S{j}" for j in range(D.shape[1])]).assign(current_state=[f"S{i}" for i in range(D.shape[0])])[["current_state"] + [f"S{j}" for j in range(D.shape[1])]], max_rows=None))
        out.append("")

    out.append("## Figure manifest guardrails")
    out.append("")
    fig_manifest = bundle.get("figure_manifest", {})
    guardrails = nested_get(fig_manifest, ["guardrails"], {}) or {}
    guard_df = pd.DataFrame([{"guardrail": k, "value": v} for k, v in guardrails.items()])
    out.append(markdown_table(guard_df, max_rows=None))
    out.append("")

    out.append("## Audit-only values")
    out.append("")
    audit = dledger[dledger["priority"].eq(AUDIT_ONLY)].copy() if not dledger.empty else pd.DataFrame()
    out.append(markdown_table(audit, columns=["category", "metric", "formatted_value", "source", "interpretation"], max_rows=None))
    out.append("")

    out.append("## Suggested compact Results wording from the available metrics")
    out.append("")
    out.append("Use the following structure, replacing bracketed items with the report values above:")
    out.append("")
    out.append(
        "> Within the pre-specified mechanism-family hierarchy, the offset dual-channel mechanism was selected under the one-standard-error and practical-equivalence rules, retaining four substantive parameters ($\\theta_0$, $\\theta_M$, $\\phi_0$ and $\\Delta_S$) while fixing $\\theta_\\Psi$, $\\theta_{M\\Psi}$ and $\\phi_\\Psi$ to zero. After pooled-development calibration and freeze, B_confirm output-only confirmation on [rows] rows from [users] users recovered the next-state landscape (JS = [value]) and primary drift geometry (drift vector correlation = [value], local drift error = [value]). Transition and residence diagnostics, which were not used for Phase-1 selection, provided post-selection kinetic checks: the mechanism transition matrix had mean row-wise TV = [value], self-transition RMSE = [value], and transition-implied residence-reference concordance = [value]."
    )
    out.append("")
    out.append("Do not write that the mechanism freely simulates residence-time tails unless an additional free-running residence simulation is added.")
    out.append("")
    return "\n".join(out)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract publication statistics for the frozen EdNet-KT4 minimal mechanism."
    )
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--minimality-root", type=Path, default=DEFAULT_MINIMALITY_ROOT)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--figure-root", type=Path, default=None)
    parser.add_argument("--confirm-split", type=str, default=DEFAULT_CONFIRM_SPLIT)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.confirm_split != "B_confirm":
        raise RuntimeError("Formal publication statistics require confirm_split='B_confirm'.")
    if args.output_md is None:
        args.output_md = args.phase3_root / "tables" / "minimal_mechanism_manuscript_numeric_report.md"
    ledger, bundle, sources, tables = make_metric_ledger(args)
    markdown = build_markdown(args, ledger, bundle, sources, tables)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    ledger_path = args.output_md.with_name("minimal_mechanism_publication_metric_ledger.csv")
    ledger_df(ledger).to_csv(ledger_path, index=False)
    source_path = args.output_md.with_name("minimal_mechanism_publication_source_audit.csv")
    source_audit_table(sources).to_csv(source_path, index=False)
    print(f"[minimal-mechanism-report] wrote {args.output_md}")


if __name__ == "__main__":
    main()
