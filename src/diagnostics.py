from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PredictionPath = str | Path | None
LoadWellFn = Callable[[Path, str, str], object]


def _path_exists(path: PredictionPath) -> bool:
    return path is not None and Path(path).exists()


def _rmse_np(error: np.ndarray) -> float:
    valid = np.isfinite(error)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean(error[valid] ** 2)))


def _corr_np(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    x = a[valid].astype(float, copy=False)
    y = b[valid].astype(float, copy=False)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _parse_row_index(ids: pd.Series) -> np.ndarray:
    return pd.to_numeric(ids.astype(str).str.rsplit("_", n=1).str[-1], errors="coerce").to_numpy()


def _read_prediction(path: PredictionPath, name: str) -> pd.DataFrame:
    if not _path_exists(path):
        raise FileNotFoundError(f"Missing prediction path for {name}: {path}")
    frame = pd.read_csv(path)
    if "id" not in frame.columns or "tvt" not in frame.columns:
        raise ValueError(f"{name} prediction must contain columns id,tvt")
    keep = ["id", "tvt"]
    for col in ("TVT", "well_id", "row_index", "MD", "GR", "TVT_input"):
        if col in frame.columns:
            keep.append(col)
    frame = frame[keep].copy()
    return frame.rename(columns={"tvt": f"{name}_tvt"})


def load_aligned_predictions(
    baseline_path: PredictionPath,
    experiment_path: PredictionPath | None = None,
    truth_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load baseline/experiment predictions and align them to truth rows by id."""
    if _path_exists(baseline_path):
        baseline = _read_prediction(baseline_path, "baseline")
    elif _path_exists(experiment_path):
        baseline = _read_prediction(experiment_path, "baseline")
    else:
        raise FileNotFoundError("At least one prediction path must exist")

    has_experiment = _path_exists(experiment_path)
    experiment = _read_prediction(experiment_path, "experiment") if has_experiment else None

    aligned = baseline[["id", "baseline_tvt"]].copy()
    if experiment is not None:
        aligned = aligned.merge(experiment[["id", "experiment_tvt"]], on="id", how="inner")
    else:
        aligned["experiment_tvt"] = aligned["baseline_tvt"]

    truth_source = truth_df.copy() if truth_df is not None else baseline.copy()
    if "TVT" not in truth_source.columns:
        for candidate in (baseline, experiment):
            if candidate is not None and "TVT" in candidate.columns:
                truth_source = candidate.copy()
                break

    truth_cols = ["id"]
    for col in ("TVT", "well_id", "row_index", "MD", "GR", "TVT_input"):
        if col in truth_source.columns:
            truth_cols.append(col)
    if len(truth_cols) > 1:
        aligned = aligned.merge(truth_source[truth_cols].drop_duplicates("id"), on="id", how="left")

    if "row_index" not in aligned.columns:
        aligned["row_index"] = _parse_row_index(aligned["id"])
    if "well_id" not in aligned.columns:
        aligned["well_id"] = aligned["id"].astype(str).str.rsplit("_", n=1).str[0]
    if "TVT" not in aligned.columns:
        aligned["TVT"] = np.nan

    aligned["baseline_error"] = aligned["baseline_tvt"].to_numpy(dtype=float) - aligned["TVT"].to_numpy(dtype=float)
    aligned["experiment_error"] = aligned["experiment_tvt"].to_numpy(dtype=float) - aligned["TVT"].to_numpy(dtype=float)
    aligned["abs_error_experiment"] = np.abs(aligned["experiment_error"].to_numpy(dtype=float))
    return aligned, has_experiment


def compute_rmse_by_well(aligned: pd.DataFrame) -> pd.DataFrame:
    wells = aligned["well_id"].astype(str).to_numpy()
    base_err = aligned["baseline_error"].to_numpy(dtype=float)
    exp_err = aligned["experiment_error"].to_numpy(dtype=float)
    unique = np.unique(wells)
    rows = []
    for well_id in unique:
        mask = wells == well_id
        baseline_rmse = _rmse_np(base_err[mask])
        experiment_rmse = _rmse_np(exp_err[mask])
        rows.append(
            {
                "well_id": well_id,
                "baseline_rmse": baseline_rmse,
                "experiment_rmse": experiment_rmse,
                "delta_rmse": experiment_rmse - baseline_rmse,
                "count": int(np.isfinite(exp_err[mask]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_rmse").reset_index(drop=True)


def compute_rmse_by_distance_from_ps(aligned: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    wells = aligned["well_id"].astype(str).to_numpy()
    row_index = aligned["row_index"].to_numpy(dtype=float)
    base_err = aligned["baseline_error"].to_numpy(dtype=float)
    exp_err = aligned["experiment_error"].to_numpy(dtype=float)
    bin_ids_all = []
    base_all = []
    exp_all = []

    for well_id in np.unique(wells):
        idx = np.where(wells == well_id)[0]
        order = idx[np.argsort(row_index[idx], kind="mergesort")]
        n = len(order)
        if n == 0:
            continue
        frac = (np.arange(n, dtype=float) + 0.5) / float(n)
        bin_ids = np.minimum((frac * bins).astype(int), bins - 1)
        bin_ids_all.append(bin_ids)
        base_all.append(base_err[order])
        exp_all.append(exp_err[order])

    if not bin_ids_all:
        return pd.DataFrame(columns=["bin", "distance_start", "distance_end", "baseline_rmse", "experiment_rmse"])

    bin_ids = np.concatenate(bin_ids_all)
    base = np.concatenate(base_all)
    exp = np.concatenate(exp_all)
    rows = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        rows.append(
            {
                "bin": int(bin_id),
                "distance_start": bin_id / bins,
                "distance_end": (bin_id + 1) / bins,
                "baseline_rmse": _rmse_np(base[mask]),
                "experiment_rmse": _rmse_np(exp[mask]),
            }
        )
    return pd.DataFrame(rows)


def compute_rmse_by_confidence_decile(aligned: pd.DataFrame, diagnostics: pd.DataFrame | None) -> pd.DataFrame:
    if diagnostics is None or "confidence" not in diagnostics.columns:
        return pd.DataFrame(columns=["decile", "confidence_min", "confidence_max", "experiment_rmse", "count"])
    frame = aligned[["id", "experiment_error"]].merge(diagnostics[["id", "confidence"]], on="id", how="inner")
    conf = frame["confidence"].to_numpy(dtype=float)
    err = frame["experiment_error"].to_numpy(dtype=float)
    valid = np.isfinite(conf) & np.isfinite(err)
    if valid.sum() == 0:
        return pd.DataFrame(columns=["decile", "confidence_min", "confidence_max", "experiment_rmse", "count"])
    conf = conf[valid]
    err = err[valid]
    edges = np.nanpercentile(conf, np.linspace(0, 100, 11))
    edges = np.unique(edges)
    if len(edges) <= 1:
        deciles = np.zeros(len(conf), dtype=int)
        max_decile = 1
    else:
        deciles = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, len(edges) - 2)
        max_decile = len(edges) - 1
    rows = []
    for decile in range(max_decile):
        mask = deciles == decile
        if not np.any(mask):
            continue
        rows.append(
            {
                "decile": int(decile),
                "confidence_min": float(np.nanmin(conf[mask])),
                "confidence_max": float(np.nanmax(conf[mask])),
                "experiment_rmse": _rmse_np(err[mask]),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def interpolate_typewell_gr_np(typewell_tvt: np.ndarray, typewell_gr: np.ndarray, query_tvt: np.ndarray) -> np.ndarray:
    tw_tvt = np.asarray(typewell_tvt, dtype=float)
    tw_gr = np.asarray(typewell_gr, dtype=float)
    query = np.asarray(query_tvt, dtype=float)
    out = np.full(len(query), np.nan, dtype=float)
    valid_tw = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    valid_query = np.isfinite(query)
    if valid_tw.sum() < 2 or valid_query.sum() == 0:
        return out
    tw_tvt = tw_tvt[valid_tw]
    tw_gr = tw_gr[valid_tw]
    order = np.argsort(tw_tvt, kind="mergesort")
    out[valid_query] = np.interp(query[valid_query], tw_tvt[order], tw_gr[order], left=np.nan, right=np.nan)
    return out


def compute_gr_alignment_arrays(
    horizontal_gr: np.ndarray,
    true_tvt: np.ndarray,
    baseline_tvt: np.ndarray,
    experiment_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> dict[str, np.ndarray | float]:
    true_gr = interpolate_typewell_gr_np(typewell_tvt, typewell_gr, true_tvt)
    baseline_gr = interpolate_typewell_gr_np(typewell_tvt, typewell_gr, baseline_tvt)
    experiment_gr = interpolate_typewell_gr_np(typewell_tvt, typewell_gr, experiment_tvt)
    horizontal = np.asarray(horizontal_gr, dtype=float)
    return {
        "true_typewell_gr": true_gr,
        "baseline_typewell_gr": baseline_gr,
        "experiment_typewell_gr": experiment_gr,
        "baseline_corr": _corr_np(horizontal, baseline_gr),
        "experiment_corr": _corr_np(horizontal, experiment_gr),
        "true_corr": _corr_np(horizontal, true_gr),
    }


def select_worst_wells_by_rmse_delta(well_summary: pd.DataFrame, top_k: int = 3) -> list[str]:
    if well_summary.empty:
        return []
    worst_baseline = well_summary.sort_values("baseline_rmse", ascending=False).head(top_k)["well_id"].tolist()
    most_changed = well_summary.assign(abs_delta=lambda x: x["delta_rmse"].abs()).sort_values(
        "abs_delta", ascending=False
    ).head(top_k)["well_id"].tolist()
    selected: list[str] = []
    for well_id in worst_baseline + most_changed:
        if well_id not in selected:
            selected.append(well_id)
    return selected


def _merge_well_context(aligned: pd.DataFrame, well_id: str, data_dir: Path, load_well_fn: LoadWellFn | None) -> pd.DataFrame:
    group = aligned.loc[aligned["well_id"].astype(str) == str(well_id)].copy()
    if group.empty or load_well_fn is None:
        return group.sort_values("row_index")
    well = load_well_fn(data_dir, "train", str(well_id))
    horizontal = well.horizontal.copy()
    keep = [col for col in ("row_index", "MD", "GR", "TVT_input") if col in horizontal.columns]
    if "row_index" in keep:
        group = group.merge(horizontal[keep], on="row_index", how="left", suffixes=("", "_well"))
        for col in ("MD", "GR", "TVT_input"):
            well_col = f"{col}_well"
            if well_col in group.columns:
                group[col] = group[col].combine_first(group[well_col]) if col in group.columns else group[well_col]
                group = group.drop(columns=[well_col])
    return group.sort_values("MD" if "MD" in group.columns else "row_index")


def _save_rmse_delta_plot(well_summary: pd.DataFrame, output_dir: Path) -> None:
    if well_summary.empty:
        return
    plot_df = well_summary.sort_values("delta_rmse")
    height = max(4.0, min(18.0, 0.22 * len(plot_df)))
    fig, ax = plt.subplots(figsize=(10, height))
    colors = np.where(plot_df["delta_rmse"].to_numpy(dtype=float) <= 0.0, "#2f7d32", "#b23b3b")
    ax.barh(plot_df["well_id"], plot_df["delta_rmse"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Per-well RMSE Delta (Experiment - Baseline)")
    ax.set_xlabel("Delta RMSE (negative is better)")
    ax.set_ylabel("well_id")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_delta_by_well.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_drift_plot(group: pd.DataFrame, well_id: str, output_dir: Path) -> None:
    x_col = "MD" if "MD" in group.columns and group["MD"].notna().any() else "row_index"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(group[x_col], group["baseline_error"], label="Baseline error", color="#777777", linewidth=1.2)
    ax.plot(group[x_col], group["experiment_error"], label="Experiment error", color="#b23b3b", linewidth=1.2)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Cumulative Error Drift - Well {well_id}")
    ax.set_xlabel("Measured Depth (MD)" if x_col == "MD" else "row_index")
    ax.set_ylabel("Pred - True TVT")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"cumulative_error_before_after_{well_id}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_gr_alignment_plot(
    group: pd.DataFrame,
    well_id: str,
    data_dir: Path,
    load_well_fn: LoadWellFn | None,
    output_dir: Path,
) -> tuple[float, float]:
    if load_well_fn is None or "GR" not in group.columns:
        return float("nan"), float("nan")
    well = load_well_fn(data_dir, "train", str(well_id))
    if not {"TVT", "GR"}.issubset(well.typewell.columns):
        return float("nan"), float("nan")
    tw = well.typewell[["TVT", "GR"]].dropna(subset=["TVT"]).sort_values("TVT")
    if len(tw) < 2:
        return float("nan"), float("nan")
    arrays = compute_gr_alignment_arrays(
        group["GR"].to_numpy(dtype=float),
        group["TVT"].to_numpy(dtype=float),
        group["baseline_tvt"].to_numpy(dtype=float),
        group["experiment_tvt"].to_numpy(dtype=float),
        tw["TVT"].to_numpy(dtype=float),
        tw["GR"].to_numpy(dtype=float),
    )
    x_col = "MD" if "MD" in group.columns and group["MD"].notna().any() else "row_index"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(group[x_col], group["GR"], label="Horizontal GR", color="black", alpha=0.65, linewidth=1.0)
    ax.plot(group[x_col], arrays["true_typewell_gr"], label="Typewell GR at True TVT", color="#2f7d32", linestyle="--")
    ax.plot(group[x_col], arrays["baseline_typewell_gr"], label="Typewell GR at Baseline TVT", color="#777777", linestyle=":")
    ax.plot(group[x_col], arrays["experiment_typewell_gr"], label="Typewell GR at Experiment TVT", color="#b23b3b", linestyle="-.")
    ax.set_title(f"GR Alignment Before/After - Well {well_id}")
    ax.set_xlabel("Measured Depth (MD)" if x_col == "MD" else "row_index")
    ax.set_ylabel("Gamma Ray (GR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"gr_alignment_before_after_{well_id}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return float(arrays["baseline_corr"]), float(arrays["experiment_corr"])


def _save_correction_confidence_plot(group: pd.DataFrame, diagnostics: pd.DataFrame | None, well_id: str, output_dir: Path) -> None:
    if diagnostics is None:
        return
    available = [col for col in ("confidence", "smooth_offset", "raw_offset", "gr_cost_before", "gr_cost_after") if col in diagnostics.columns]
    if not available:
        return
    diag_cols = ["id"] + available
    merged = group.merge(diagnostics[diag_cols], on="id", how="left")
    x_col = "MD" if "MD" in merged.columns and merged["MD"].notna().any() else "row_index"
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    offset = (
        merged["smooth_offset"].to_numpy(dtype=float)
        if "smooth_offset" in merged.columns
        else merged["experiment_tvt"].to_numpy(dtype=float) - merged["baseline_tvt"].to_numpy(dtype=float)
    )
    axes[0].plot(merged[x_col], offset, color="#1f77b4", label="Correction offset")
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("TVT offset")
    axes[0].legend()
    if "confidence" in merged.columns:
        axes[1].plot(merged[x_col], merged["confidence"], color="#2f7d32", label="Confidence")
        axes[1].set_ylabel("Confidence")
        axes[1].legend()
    if "gr_cost_before" in merged.columns:
        axes[2].plot(merged[x_col], merged["gr_cost_before"], color="#777777", label="GR cost before")
    if "gr_cost_after" in merged.columns:
        axes[2].plot(merged[x_col], merged["gr_cost_after"], color="#b23b3b", label="GR cost after")
    axes[2].set_ylabel("GR cost")
    axes[2].set_xlabel("Measured Depth (MD)" if x_col == "MD" else "row_index")
    axes[2].legend()
    fig.suptitle(f"Correction / Confidence Diagnostics - Well {well_id}")
    fig.tight_layout()
    fig.savefig(output_dir / f"correction_confidence_{well_id}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_distance_plot(distance_summary: pd.DataFrame, output_dir: Path) -> None:
    if distance_summary.empty:
        return
    centers = (distance_summary["distance_start"].to_numpy() + distance_summary["distance_end"].to_numpy()) * 0.5
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(centers, distance_summary["baseline_rmse"], marker="o", label="Baseline RMSE", color="#777777")
    ax.plot(centers, distance_summary["experiment_rmse"], marker="o", label="Experiment RMSE", color="#b23b3b")
    ax.set_title("RMSE by Distance From Prediction Start")
    ax.set_xlabel("Normalized distance after PS")
    ax.set_ylabel("RMSE")
    ax.set_xticks(np.linspace(0.05, 0.95, 10))
    ax.set_xticklabels([f"{i * 10}-{(i + 1) * 10}%" for i in range(10)], rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_by_distance_from_ps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_confidence_decile_plot(confidence_summary: pd.DataFrame, output_dir: Path) -> None:
    if confidence_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(confidence_summary["decile"].astype(str), confidence_summary["experiment_rmse"], color="#1f77b4")
    ax.set_title("Experiment RMSE by Confidence Decile")
    ax.set_xlabel("Confidence decile")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_by_confidence_decile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_error_vs_score(aligned: pd.DataFrame, diagnostics: pd.DataFrame | None, score_col: str, output_path: Path) -> None:
    if diagnostics is None or score_col not in diagnostics.columns:
        return
    frame = aligned[["id", "abs_error_experiment"]].merge(diagnostics[["id", score_col]], on="id", how="inner")
    x = frame[score_col].to_numpy(dtype=float)
    y = frame["abs_error_experiment"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x[valid], y[valid], s=5, alpha=0.25)
    ax.set_title(f"Experiment Absolute Error vs {score_col}")
    ax.set_xlabel(score_col)
    ax.set_ylabel("Absolute TVT error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compute_drift_summary(aligned: pd.DataFrame) -> tuple[bool | None, str | None]:
    rows = []
    for well_id in np.unique(aligned["well_id"].astype(str).to_numpy()):
        group = aligned.loc[aligned["well_id"].astype(str) == well_id].sort_values("row_index")
        n = len(group)
        if n == 0:
            continue
        tail = group.tail(max(1, int(math.ceil(0.2 * n))))
        base_drift = float(np.nanmean(np.abs(tail["baseline_error"].to_numpy(dtype=float))))
        exp_drift = float(np.nanmean(np.abs(tail["experiment_error"].to_numpy(dtype=float))))
        rows.append((well_id, base_drift, exp_drift))
    if not rows:
        return None, None
    base_mean = float(np.nanmean([row[1] for row in rows]))
    exp_mean = float(np.nanmean([row[2] for row in rows]))
    largest = max(rows, key=lambda row: -np.inf if not np.isfinite(row[2]) else row[2])[0]
    return exp_mean < base_mean, str(largest)


def _load_diagnostics(path: PredictionPath) -> pd.DataFrame | None:
    if not _path_exists(path):
        return None
    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise ValueError("Diagnostics CSV must contain id")
    return frame


def build_diagnostic_summary(
    aligned: pd.DataFrame,
    well_summary: pd.DataFrame,
    has_experiment: bool,
    diagnostics: pd.DataFrame | None,
    gr_corr_before: list[float],
    gr_corr_after: list[float],
) -> dict:
    baseline_rmse = _rmse_np(aligned["baseline_error"].to_numpy(dtype=float))
    experiment_rmse = _rmse_np(aligned["experiment_error"].to_numpy(dtype=float))
    improved = well_summary.loc[well_summary["delta_rmse"] < 0, ["well_id", "delta_rmse"]]
    regressed = well_summary.loc[well_summary["delta_rmse"] > 0, ["well_id", "delta_rmse"]]
    drift_reduced, largest_drift = _compute_drift_summary(aligned)
    confidence_corr = float("nan")
    if diagnostics is not None and "confidence" in diagnostics.columns:
        conf_frame = aligned[["id", "abs_error_experiment"]].merge(diagnostics[["id", "confidence"]], on="id", how="inner")
        confidence_corr = _corr_np(
            conf_frame["confidence"].to_numpy(dtype=float),
            conf_frame["abs_error_experiment"].to_numpy(dtype=float),
        )
    avg_gr_before = float(np.nanmean(gr_corr_before)) if len(gr_corr_before) else float("nan")
    avg_gr_after = float(np.nanmean(gr_corr_after)) if len(gr_corr_after) else float("nan")
    delta_rmse = experiment_rmse - baseline_rmse
    if not has_experiment:
        recommendation = "Baseline-only diagnostics; add an experiment prediction for comparison."
    elif np.isfinite(delta_rmse) and delta_rmse < 0 and drift_reduced is not False:
        recommendation = "Candidate for Kaggle submission."
    else:
        recommendation = "Needs more validation before Kaggle submission."
    return {
        "overall_rmse_baseline": baseline_rmse,
        "overall_rmse_experiment": experiment_rmse,
        "delta_rmse": delta_rmse,
        "has_experiment": bool(has_experiment),
        "number_wells_improved": int(len(improved)),
        "number_wells_worse": int(len(regressed)),
        "best_improved_wells": improved.sort_values("delta_rmse").head(10).to_dict(orient="records"),
        "worst_regressed_wells": regressed.sort_values("delta_rmse", ascending=False).head(10).to_dict(orient="records"),
        "average_drift_reduced": drift_reduced,
        "largest_remaining_drift": largest_drift,
        "average_gr_correlation_before": avg_gr_before,
        "average_gr_correlation_after": avg_gr_after,
        "confidence_error_correlation": confidence_corr,
        "recommendation": recommendation,
    }


def _format_float(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_experiment_report(summary: dict, report_path: str | Path) -> None:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    improved = summary.get("best_improved_wells", [])
    regressed = summary.get("worst_regressed_wells", [])
    drift = summary.get("average_drift_reduced")
    drift_text = "NA" if drift is None else ("YES" if drift else "NO")
    lines = [
        "Overall RMSE",
        "-------------",
        f"Baseline : {_format_float(summary.get('overall_rmse_baseline', float('nan')))}",
        f"Experiment : {_format_float(summary.get('overall_rmse_experiment', float('nan')))}",
        f"Delta RMSE = {_format_float(summary.get('delta_rmse', float('nan')))}",
        "",
        "Well Summary",
        "------------",
        "Improved:",
    ]
    lines.extend([f"{row['well_id']}  {_format_float(float(row['delta_rmse']), 3)}" for row in improved] or ["NA"])
    lines.extend(["", "Regressed:"])
    lines.extend([f"{row['well_id']}  +{_format_float(float(row['delta_rmse']), 3).lstrip('+')}" for row in regressed] or ["NA"])
    lines.extend(
        [
            "",
            "Error Drift",
            "-----------",
            f"Average drift reduced: {drift_text}",
            "",
            "Largest remaining drift:",
            str(summary.get("largest_remaining_drift") or "NA"),
            "",
            "GR Alignment",
            "------------",
            "Average correlation:",
            f"{_format_float(summary.get('average_gr_correlation_before', float('nan')), 2)} -> {_format_float(summary.get('average_gr_correlation_after', float('nan')), 2)}",
            "",
            "Confidence",
            "----------",
            "Correlation(confidence,error)",
            _format_float(summary.get("confidence_error_correlation", float("nan")), 2),
            "",
            "Recommendation",
            "--------------",
            str(summary.get("recommendation", "Needs review.")),
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostic_pack(
    baseline_path: PredictionPath,
    experiment_path: PredictionPath | None = None,
    truth_df: pd.DataFrame | None = None,
    diagnostics_path: PredictionPath | None = None,
    data_dir: str | Path = "data",
    output_dir: str | Path = "model/error_plots",
    report_path: str | Path = "model/experiment_report.md",
    top_k_wells: int = 3,
    selected_wells: list[str] | None = None,
    load_well_fn: LoadWellFn | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    aligned, has_experiment = load_aligned_predictions(baseline_path, experiment_path, truth_df=truth_df)
    diagnostics = _load_diagnostics(diagnostics_path)

    well_summary = compute_rmse_by_well(aligned)
    distance_summary = compute_rmse_by_distance_from_ps(aligned)
    confidence_summary = compute_rmse_by_confidence_decile(aligned, diagnostics)

    _save_rmse_delta_plot(well_summary, output_dir)
    _save_distance_plot(distance_summary, output_dir)
    _save_confidence_decile_plot(confidence_summary, output_dir)
    _save_error_vs_score(aligned, diagnostics, "texture_score", output_dir / "error_vs_gr_texture.png")
    _save_error_vs_score(aligned, diagnostics, "uniqueness_score", output_dir / "error_vs_uniqueness.png")

    plot_wells = selected_wells or select_worst_wells_by_rmse_delta(well_summary, top_k=top_k_wells)
    gr_corr_before: list[float] = []
    gr_corr_after: list[float] = []
    for well_id in plot_wells:
        group = _merge_well_context(aligned, str(well_id), data_dir, load_well_fn)
        if group.empty:
            continue
        _save_drift_plot(group, str(well_id), output_dir)
        before, after = _save_gr_alignment_plot(group, str(well_id), data_dir, load_well_fn, output_dir)
        if np.isfinite(before):
            gr_corr_before.append(before)
        if np.isfinite(after):
            gr_corr_after.append(after)
        _save_correction_confidence_plot(group, diagnostics, str(well_id), output_dir)

    summary = build_diagnostic_summary(aligned, well_summary, has_experiment, diagnostics, gr_corr_before, gr_corr_after)
    summary["plot_wells"] = plot_wells
    summary["output_dir"] = str(output_dir)
    summary["report_path"] = str(report_path)
    summary_path = output_dir / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    write_experiment_report(summary, report_path)
    return summary
