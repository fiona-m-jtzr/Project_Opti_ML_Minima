#!/usr/bin/env python3
"""Aggregate W&B minimum-analysis artifacts and plot summary diagnostics.

This variant plots both RAW and NORMALIZED Hessian metrics plus both
scale-invariant and element-wise adaptive sharpness curves produced by the
analysis script:
- raw top Hessian eigenvalue, taken as raw_top_eigenvalues[0];
- normalized top Hessian eigenvalue, taken as normalized_top_eigenvalue;
- raw Hessian trace mean, taken as raw_trace_mean;
- normalized Hessian trace, taken as normalized_trace.

If normalized Hessian values are missing but raw values and weight_norm are
available, the script reconstructs the normalized values as raw_value * ||w||^2.

The W&B loading path intentionally mirrors the original script:
- use the default W&B entity;
- scan runs in PROJECT;
- keep only the newest version of each analysis artifact collection;
- load the first JSON file found in each downloaded artifact.

The plotting/averaging path is configurable from the CLI. Adaptive sharpness
is read from elementwise_adaptive_sharpness_by_radius, whose curve radius key is
rho in the analyzer output. The --model-family option filters W&B analysis
artifacts by collection name before any artifact JSON is downloaded.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb


PROJECT = "OptiML_Minima"

MODEL_FAMILY_ARTIFACT_MARKERS = {
    "resnet20": "model-final_model_resnet20",
    "vit": "model-final_model_vit",
}

SUMMARY_METRICS = (
    "train_loss",
    "test_loss",
    "train_accuracy",
    "test_accuracy",
    "train_test_loss_gap",
    "train_test_accuracy_gap",
    "gradient_norm_full_train_dataset",
)

# These are the Hessian metrics averaged across runs before plotting.
# The raw top eigenvalue is derived from hessian_metrics["raw_top_eigenvalues"][0]
# when hessian_metrics["raw_top_eigenvalue"] is not already present.
# Normalized values are read directly when present and otherwise reconstructed
# from raw_value * weight_norm**2 when possible.
HESSIAN_METRICS = (
    "negative_curvature_ratio",
    "raw_top_eigenvalue",
    "normalized_top_eigenvalue",
    "raw_trace_mean",
    "normalized_trace",
)

PLOT_METRICS = (
    ("train_loss", "Train Loss", "loss"),
    ("test_loss", "Test Loss", "loss"),
    ("train_test_loss_gap", "Train-Test Loss Gap", "test loss - train loss"),
    ("train_accuracy", "Train Accuracy", "accuracy"),
    ("test_accuracy", "Test Accuracy", "accuracy"),
    ("train_test_accuracy_gap", "Train-Test Accuracy Gap", "train accuracy - test accuracy"),
    ("gradient_norm_full_train_dataset", "Gradient Norm", "gradient norm"),
    (("hessian_metrics", "raw_top_eigenvalue"), "Raw Hessian Max Eigenvalue", "λ_max(H)"),
    (("hessian_metrics", "normalized_top_eigenvalue"), "Normalized Hessian Max Eigenvalue", "λ_max(H) · ||w||²"),
    (("hessian_metrics", "raw_trace_mean"), "Raw Hessian Trace", "tr(H)"),
    (("hessian_metrics", "normalized_trace"), "Normalized Hessian Trace", "tr(H) · ||w||²"),
)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def average_numeric(values: Iterable[Any]) -> float | None:
    """Return the finite numeric mean, ignoring None/non-numeric values."""
    clean = [
        float(value)
        for value in values
        if isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return sum(clean) / len(clean) if clean else None


def finite_real(value: Any) -> float | None:
    """Return value as a finite float, otherwise None."""
    if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def get_nested(dct: dict[str, Any], *keys: str) -> Any:
    """Read a nested dictionary value, returning None if any level is missing."""
    value: Any = dct
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def hessian_metric_value(result: dict[str, Any], metric: str) -> Any:
    """Read Hessian metrics with compatibility for older analysis JSON files."""
    hessian_metrics = result.get("hessian_metrics")
    if not isinstance(hessian_metrics, dict):
        return None

    def raw_top_eigenvalue() -> float | None:
        explicit_value = finite_real(hessian_metrics.get("raw_top_eigenvalue"))
        if explicit_value is not None:
            return explicit_value

        eigenvalues = hessian_metrics.get("raw_top_eigenvalues")
        if isinstance(eigenvalues, list) and eigenvalues:
            return finite_real(eigenvalues[0])
        return None

    if metric == "raw_top_eigenvalue":
        return raw_top_eigenvalue()

    if metric == "normalized_top_eigenvalue":
        explicit_value = finite_real(hessian_metrics.get("normalized_top_eigenvalue"))
        if explicit_value is not None:
            return explicit_value

        raw_value = raw_top_eigenvalue()
        weight_norm = finite_real(hessian_metrics.get("weight_norm"))
        if raw_value is not None and weight_norm is not None:
            return raw_value * (weight_norm ** 2)
        return None

    if metric == "normalized_trace":
        explicit_value = finite_real(hessian_metrics.get("normalized_trace"))
        if explicit_value is not None:
            return explicit_value

        raw_trace = finite_real(hessian_metrics.get("raw_trace_mean"))
        weight_norm = finite_real(hessian_metrics.get("weight_norm"))
        if raw_trace is not None and weight_norm is not None:
            return raw_trace * (weight_norm ** 2)
        return None

    return hessian_metrics.get(metric)


def result_name(result: dict[str, Any]) -> str:
    """Best available name for a raw or aggregated result."""
    return (
        result.get("source_run_name")
        or result.get("_artifact_name")
        or result.get("_wandb_run_name")
        or ""
    )


def format_value(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def common_value(values: Iterable[Any]) -> Any:
    """Return the common non-None value, otherwise None."""
    unique = {value for value in values if value is not None}
    return next(iter(unique)) if len(unique) == 1 else None


# -----------------------------------------------------------------------------
# W&B loading -- kept equivalent to the original loading capability
# -----------------------------------------------------------------------------


def load_json_from_artifact(artifact: Any) -> dict[str, Any]:
    artifact_dir = Path(artifact.download())
    json_files = list(artifact_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON found in {artifact.name}")

    with open(json_files[0], "r") as f:
        return json.load(f)


def artifact_matches_model_family(artifact_name: str, model_family: str) -> bool:
    """Return whether an artifact collection/name belongs to the selected model family."""
    marker = MODEL_FAMILY_ARTIFACT_MARKERS[model_family]
    return marker.lower() in artifact_name.lower()


def collect_analysis_results(
    project: str = PROJECT,
    *,
    model_family: str = "resnet20",
) -> list[dict[str, Any]]:
    api = wandb.Api()
    entity = api.default_entity
    runs = api.runs(f"{entity}/{project}")

    latest_by_collection = {}
    skipped_by_model_family = 0

    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type != "analysis":
                continue

            # Example collection names:
            # model-resnet20-sgd_lr0.1_seed42-minimum-analysis:v3 -> model-resnet20-sgd_lr0.1_seed42-minimum-analysis
            # model-vit-adamw_lr0.001_seed42-minimum-analysis:v2 -> model-vit-adamw_lr0.001_seed42-minimum-analysis
            collection = artifact.name.split(":")[0]

            if not artifact_matches_model_family(collection, model_family):
                skipped_by_model_family += 1
                continue

            if (
                collection not in latest_by_collection
                or artifact.created_at > latest_by_collection[collection].created_at
            ):
                latest_by_collection[collection] = artifact

    marker = MODEL_FAMILY_ARTIFACT_MARKERS[model_family]
    print(
        f"Model-family filter: keeping analysis artifacts whose collection name contains "
        f"{marker!r}; skipped {skipped_by_model_family} non-matching analysis artifact version(s)."
    )

    results: list[dict[str, Any]] = []

    for artifact in latest_by_collection.values():
        try:
            data = load_json_from_artifact(artifact)
            data["_artifact_name"] = artifact.name
            data["_model_family"] = model_family
            results.append(data)
            print(f"Loaded latest {artifact.name}")
        except Exception as exc:  # keep collecting even if one artifact is bad
            print(f"Skipping {artifact.name}: {exc}")

    return results


# -----------------------------------------------------------------------------
# Run-name parsing and grouping
# -----------------------------------------------------------------------------


def parse_run_name(name: str) -> dict[str, Any]:
    """Parse model/optimizer/config metadata from artifact or run names."""
    name = (
        name
        .replace("-minimum-analysis", "")
        .replace("_minimum_analysis", "")
    )
    name_lower = name.lower()

    info: dict[str, Any] = {
        "model_family": None,
        "optimizer": "unknown",
        "lr": None,
        "bs": None,
        "seed": None,
        "momentum": None,
        "nesterov": False,
        "schedule": None,
    }

    # Handles:
    # model-FINAL_MODEL_resnet20_sgd_mom0.9_...
    # model-final_model_resnet20_sgd_mom0.9_...
    # model-resnet20-sgd_lr0.1_seed42
    # model-vit-adamw_lr0.001_seed42
    family_match = re.search(
        r"model[-_](?:final_model[-_])?(resnet20|vit)(?:[_-]|$)",
        name_lower,
    )

    if family_match:
        info["model_family"] = family_match.group(1)

        after_family = name_lower[family_match.end():].lstrip("_-")
        opt_match = re.match(r"([a-zA-Z][a-zA-Z0-9]*)(?=[_-]|$)", after_family)

        if opt_match and not opt_match.group(1).startswith(("lr", "bs", "seed", "mom", "wd")):
            info["optimizer"] = opt_match.group(1).lower()
    else:
        opt_match = re.search(
            r"model[-_](?:final_model[-_])?(?:resnet20|vit)[_-]([a-zA-Z][a-zA-Z0-9]*)",
            name_lower,
        )
        if opt_match:
            info["optimizer"] = opt_match.group(1).lower()

    patterns = {
        "lr": r"_lr([0-9.eE+-]+)",
        "bs": r"_bs([0-9]+)",
        "seed": r"_seed([0-9]+)",
        "momentum": r"_mom([0-9.eE+-]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if not match:
            continue

        raw_value = match.group(1)
        info[key] = int(raw_value) if key in {"bs", "seed"} else float(raw_value)

    info["nesterov"] = "_nesterov" in name_lower

    schedule_match = re.search(r"_bs[0-9]+_([a-zA-Z0-9]+)_seed", name_lower)
    if schedule_match:
        info["schedule"] = schedule_match.group(1)

    return info


def run_meta(result: dict[str, Any]) -> dict[str, Any]:
    """Return plotting metadata for raw or aggregated results."""
    return result.get("_plot_meta") or parse_run_name(result_name(result))


def resolve_group_by(scope: str, group_by: str) -> str:
    if group_by != "auto":
        return group_by
    return "batch_size" if scope == "sgd" else "optimizer"


def group_identity(result: dict[str, Any], scope: str, group_by: str) -> tuple[tuple[Any, ...], str, str]:
    """Return (group key, bar/line label, color label)."""
    meta = parse_run_name(result_name(result))
    optimizer = meta["optimizer"]
    bs = meta["bs"]
    lr = meta["lr"]

    if group_by == "optimizer":
        key = (optimizer,)
        label = optimizer
        color_label = optimizer
    elif group_by == "batch_size":
        key = (optimizer, bs) if scope == "all" else (bs,)
        label = f"{optimizer} | bs={format_value(bs)}" if scope == "all" else f"bs={format_value(bs)}"
        color_label = optimizer if scope == "all" else f"bs={format_value(bs)}"
    elif group_by == "learning_rate":
        key = (optimizer, lr) if scope == "all" else (lr,)
        label = f"{optimizer} | lr={format_value(lr)}" if scope == "all" else f"lr={format_value(lr)}"
        color_label = optimizer if scope == "all" else f"lr={format_value(lr)}"
    elif group_by == "config":
        key = (
            optimizer,
            lr,
            bs,
            meta["momentum"],
            meta["nesterov"],
            meta["schedule"],
        )
        label = f"{optimizer} | bs={format_value(bs)} | lr={format_value(lr)}"
        color_label = optimizer if scope == "all" else label
    else:
        raise ValueError(f"Unsupported group_by={group_by!r}")

    return key, label, color_label


def sort_group_key(item: tuple[tuple[Any, ...], dict[str, Any]]) -> tuple[Any, ...]:
    """Stable sort for aggregated groups before plotting."""
    result = item[1]
    meta = run_meta(result)
    return (
        meta.get("optimizer") or "",
        meta.get("bs") if meta.get("bs") is not None else math.inf,
        meta.get("lr") if meta.get("lr") is not None else math.inf,
        meta.get("group_label") or result_name(result),
    )


# -----------------------------------------------------------------------------
# Averaging
# -----------------------------------------------------------------------------


def average_sharpness_curves(group: list[dict[str, Any]]) -> list[dict[str, float | None]]:
    """Average loss-normalized sharpness curves by radius.

    The original code assumed every curve had the same length and radius order.
    This version groups by radius, so missing or differently ordered points are
    handled safely.
    """
    values_by_radius: dict[float, list[float]] = defaultdict(list)

    for run in group:
        curve = run.get("scale_invariant_sharpness_by_radius") or run.get("averaged_sharpness_curve") or []
        if not curve:
            continue

        train_loss = run.get("train_loss")
        if not isinstance(train_loss, Real) or train_loss == 0:
            continue

        for point in curve:
            radius = point.get("relative_radius")
            if radius is None:
                continue

            if "normalized_sharpness_delta" in point:
                normalized_delta = point.get("normalized_sharpness_delta")
            else:
                sharpness_delta = point.get("sharpness_delta")
                normalized_delta = (
                    sharpness_delta / train_loss
                    if isinstance(sharpness_delta, Real)
                    else None
                )

            if isinstance(normalized_delta, Real) and math.isfinite(float(normalized_delta)):
                values_by_radius[float(radius)].append(float(normalized_delta))

    return [
        {
            "relative_radius": radius,
            "normalized_sharpness_delta": average_numeric(values),
        }
        for radius, values in sorted(values_by_radius.items())
    ]


def average_adaptive_sharpness_curves(group: list[dict[str, Any]]) -> list[dict[str, float | None]]:
    """Average element-wise adaptive sharpness curves by rho.

    Analyzer compatibility:
    - current analyzer output: result["elementwise_adaptive_sharpness_by_radius"]
      with each point keyed by "rho";
    - already-aggregated inputs: result["averaged_adaptive_sharpness_curve"].

    The normalized adaptive delta is computed as sharpness_delta / base_loss,
    using the adaptive point's own base_loss. This is important because adaptive
    sharpness may use logit-normalized logits and a fixed subset of batches, so
    the global raw train_loss is not always the correct denominator.
    """
    values_by_rho: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for run in group:
        curve = (
            run.get("elementwise_adaptive_sharpness_by_radius")
            or run.get("averaged_adaptive_sharpness_curve")
            or []
        )
        if not curve:
            continue

        for point in curve:
            if not isinstance(point, dict):
                continue

            rho = finite_real(point.get("rho"))
            if rho is None:
                # Accept relative_radius as a defensive fallback for hand-edited
                # or older derived summaries, but analyzer.py emits rho.
                rho = finite_real(point.get("relative_radius"))
            if rho is None:
                continue

            sharpness_delta = finite_real(point.get("sharpness_delta"))
            base_loss = finite_real(point.get("base_loss"))
            perturbed_loss = finite_real(point.get("perturbed_loss"))
            max_batch_delta = finite_real(point.get("max_batch_sharpness_delta"))

            normalized_delta = finite_real(point.get("normalized_sharpness_delta"))
            if normalized_delta is None and sharpness_delta is not None and base_loss not in (None, 0.0):
                normalized_delta = sharpness_delta / base_loss

            if sharpness_delta is not None:
                values_by_rho[rho]["sharpness_delta"].append(sharpness_delta)
            if normalized_delta is not None:
                values_by_rho[rho]["normalized_sharpness_delta"].append(normalized_delta)
            if base_loss is not None:
                values_by_rho[rho]["base_loss"].append(base_loss)
            if perturbed_loss is not None:
                values_by_rho[rho]["perturbed_loss"].append(perturbed_loss)
            if max_batch_delta is not None:
                values_by_rho[rho]["max_batch_sharpness_delta"].append(max_batch_delta)

    return [
        {
            "rho": rho,
            "sharpness_delta": average_numeric(values.get("sharpness_delta", [])),
            "normalized_sharpness_delta": average_numeric(values.get("normalized_sharpness_delta", [])),
            "base_loss": average_numeric(values.get("base_loss", [])),
            "perturbed_loss": average_numeric(values.get("perturbed_loss", [])),
            "max_batch_sharpness_delta": average_numeric(values.get("max_batch_sharpness_delta", [])),
        }
        for rho, values in sorted(values_by_rho.items())
    ]


def average_group(
    group: list[dict[str, Any]],
    label: str,
    color_label: str,
    group_by: str,
) -> dict[str, Any]:
    base = copy.deepcopy(group[0])
    metas = [parse_run_name(result_name(result)) for result in group]

    for metric in SUMMARY_METRICS:
        base[metric] = average_numeric(result.get(metric) for result in group)

    base["hessian_metrics"] = dict(base.get("hessian_metrics", {}))
    for metric in HESSIAN_METRICS:
        base["hessian_metrics"][metric] = average_numeric(
            hessian_metric_value(result, metric) for result in group
        )

    base["source_run_name"] = label
    base["_num_runs"] = len(group)
    base["_averaged_over"] = group_by
    base["averaged_sharpness_curve"] = average_sharpness_curves(group)
    base["averaged_adaptive_sharpness_curve"] = average_adaptive_sharpness_curves(group)

    base["_plot_meta"] = {
        "optimizer": common_value(meta["optimizer"] for meta in metas) or "mixed",
        "lr": common_value(meta["lr"] for meta in metas),
        "bs": common_value(meta["bs"] for meta in metas),
        "momentum": common_value(meta["momentum"] for meta in metas),
        "nesterov": common_value(meta["nesterov"] for meta in metas),
        "schedule": common_value(meta["schedule"] for meta in metas),
        "group_by": group_by,
        "group_label": label,
        "color_label": color_label,
    }

    return base


def average_results(
    results: list[dict[str, Any]],
    *,
    scope: str,
    group_by: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}

    for result in results:
        meta = parse_run_name(result_name(result))
        if scope == "sgd" and meta["optimizer"] != "sgd":
            continue

        key, label, color_label = group_identity(result, scope, group_by)
        if key not in grouped:
            grouped[key] = {"label": label, "color_label": color_label, "runs": []}
        grouped[key]["runs"].append(result)

    averaged = {
        key: average_group(
            item["runs"],
            label=item["label"],
            color_label=item["color_label"],
            group_by=group_by,
        )
        for key, item in grouped.items()
    }

    return [result for _, result in sorted(averaged.items(), key=sort_group_key)]


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def sort_label(label: str) -> tuple[Any, ...]:
    """Natural-ish sort for labels containing numeric parameter values."""
    match = re.search(r"(?:bs|lr)=([0-9.eE+-]+)", label)
    if match:
        try:
            return (label.split("=")[0], float(match.group(1)))
        except ValueError:
            pass
    return (label,)


def build_color_map(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({run_meta(result).get("color_label", "unknown") for result in results}, key=sort_label)
    cmap_name = "tab10" if len(labels) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name, max(len(labels), 1))
    return {label: cmap(index) for index, label in enumerate(labels)}


def metric_value(result: dict[str, Any], key: str | tuple[str, ...]) -> Any:
    if isinstance(key, tuple):
        if len(key) == 2 and key[0] == "hessian_metrics":
            return hessian_metric_value(result, key[1])
        return get_nested(result, *key)
    return result.get(key)


def short_axis_label(meta: dict[str, Any]) -> str:
    label = meta.get("group_label") or "unknown"
    group_by = meta.get("group_by")
    n_runs = meta.get("_num_runs")

    if group_by == "optimizer":
        label = meta.get("optimizer", label)
    elif group_by == "batch_size" and meta.get("optimizer") != "mixed":
        label = label.replace(" | ", "\n")
    elif group_by == "learning_rate" and meta.get("optimizer") != "mixed":
        label = label.replace(" | ", "\n")
    else:
        label = label.replace(" | ", "\n")

    if n_runs:
        label = f"{label}\nn={n_runs}"
    return label


def bar_metric(
    ax: Any,
    results: list[dict[str, Any]],
    metric_key: str | tuple[str, ...],
    title: str,
    ylabel: str,
    color_map: dict[str, Any],
    sort_bars: str,
) -> None:
    clean = [
        (result, metric_value(result, metric_key))
        for result in results
        if metric_value(result, metric_key) is not None
    ]
    clean.sort(key=lambda item: item[1], reverse=(sort_bars == "descending"))

    ax.set_title(title)

    if not clean:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    xs = list(range(len(clean)))
    values = [value for _, value in clean]
    metas = [run_meta(result) for result, _ in clean]
    colors = [color_map[meta.get("color_label", "unknown")] for meta in metas]
    labels = [
        short_axis_label({**meta, "_num_runs": result.get("_num_runs")})
        for result, meta in zip((r for r, _ in clean), metas)
    ]

    ax.bar(xs, values, color=colors, alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)


def plot_sharpness_curves(
    ax: Any,
    results: list[dict[str, Any]],
    color_map: dict[str, Any],
) -> None:
    plotted = False

    for result in results:
        meta = run_meta(result)
        curve = result.get("averaged_sharpness_curve") or []

        if not curve:
            # Fallback for unaveraged inputs.
            curve = average_sharpness_curves([result])

        if not curve:
            continue

        radii = [point["relative_radius"] for point in curve]
        deltas = [point["normalized_sharpness_delta"] for point in curve]

        ax.plot(
            radii,
            deltas,
            marker="o",
            linewidth=2,
            alpha=0.9,
            color=color_map[meta.get("color_label", "unknown")],
            label=meta.get("group_label", result_name(result)),
        )
        plotted = True

    ax.set_title("Loss-Normalized Scale-Invariant Sharpness by Radius")
    ax.set_xscale("log")
    ax.set_xlabel("relative radius")
    ax.set_ylabel("sharpness delta / train loss")
    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No sharpness curves", ha="center", va="center")
        ax.set_axis_off()


def plot_adaptive_sharpness_curves(
    ax: Any,
    results: list[dict[str, Any]],
    color_map: dict[str, Any],
) -> None:
    """Plot loss-normalized element-wise adaptive sharpness against rho."""
    plotted = False

    for result in results:
        meta = run_meta(result)
        curve = result.get("averaged_adaptive_sharpness_curve") or []

        if not curve:
            # Fallback for unaveraged inputs.
            curve = average_adaptive_sharpness_curves([result])

        if not curve:
            continue

        points = [
            point
            for point in curve
            if finite_real(point.get("rho")) is not None
            and finite_real(point.get("normalized_sharpness_delta")) is not None
        ]
        if not points:
            continue

        rhos = [float(point["rho"]) for point in points]
        deltas = [float(point["normalized_sharpness_delta"]) for point in points]

        ax.plot(
            rhos,
            deltas,
            marker="o",
            linewidth=2,
            alpha=0.9,
            color=color_map[meta.get("color_label", "unknown")],
            label=meta.get("group_label", result_name(result)),
        )
        plotted = True

    ax.set_title("Loss-Normalized Element-Wise Adaptive Sharpness by Rho")
    ax.set_xscale("log")
    ax.set_xlabel("adaptive radius rho")
    ax.set_ylabel("adaptive sharpness delta / adaptive base loss")
    ax.grid(True, alpha=0.3)

    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No adaptive sharpness curves", ha="center", va="center")
        ax.set_axis_off()


def add_color_legend(fig: Any, color_map: dict[str, Any], title: str) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=color,
            label=label,
            markersize=10,
        )
        for label, color in color_map.items()
    ]

    if not handles:
        return

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(max(1, len(handles)), 6),
        frameon=True,
        title=title,
    )


def color_legend_title(scope: str, group_by: str) -> str:
    if scope == "all":
        return "Color: optimizer"
    if group_by == "batch_size":
        return "Color: batch size"
    if group_by == "learning_rate":
        return "Color: learning rate"
    return "Color: SGD parameter group"


def plot_results(
    results: list[dict[str, Any]],
    output_path: str | Path,
    *,
    scope: str,
    group_by: str,
    sort_bars: str,
    show: bool,
    model_family: str,
) -> None:
    if not results:
        raise RuntimeError("No analysis results found after filtering/grouping.")

    color_map = build_color_map(results)

    n_panels = len(PLOT_METRICS) + 2  # summary bars plus sampled and adaptive sharpness curve panels
    n_cols = 2
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4 * n_rows))
    axes = axes.flatten()

    for ax, (metric_key, title, ylabel) in zip(axes, PLOT_METRICS):
        bar_metric(ax, results, metric_key, title, ylabel, color_map, sort_bars)

    sharpness_ax = axes[len(PLOT_METRICS)]
    plot_sharpness_curves(sharpness_ax, results, color_map)

    adaptive_sharpness_ax = axes[len(PLOT_METRICS) + 1]
    plot_adaptive_sharpness_curves(adaptive_sharpness_ax, results, color_map)

    for ax in axes[n_panels:]:
        ax.set_axis_off()

    fig.suptitle(
        f"Minimum analysis summary | model={model_family} | scope={scope} | grouped by={group_by} | bars={sort_bars} | raw + normalized Hessian + sharpness metrics",
        fontsize=16,
    )
    add_color_legend(fig, color_map, color_legend_title(scope, group_by))
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load W&B minimum-analysis artifacts, average them, and plot summary metrics with raw/normalized Hessian and sharpness values."
    )
    parser.add_argument("--output", default="minimum_analysis_summary_with_adaptive_sharpness.png")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--model-family",
        choices=tuple(MODEL_FAMILY_ARTIFACT_MARKERS),
        default="resnet20",
        help=(
            "Only load analysis artifacts whose collection name contains the selected "
            "model marker: model-resnet20 or model-vit."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("all", "sgd"),
        default="all",
        help="Use all optimizers or only SGD runs.",
    )
    parser.add_argument(
        "--group-by",
        choices=("auto", "optimizer", "batch_size", "learning_rate", "config"),
        default="auto",
        help=(
            "Averaging key. auto = optimizer for --scope all, batch_size for --scope sgd. "
            "For --scope all with batch_size/learning_rate, groups are optimizer+parameter."
        ),
    )
    parser.add_argument(
        "--sort-bars",
        choices=("ascending", "descending"),
        default="ascending",
        help="Sort bars independently within each metric panel.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive Matplotlib window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_by = resolve_group_by(args.scope, args.group_by)

    results = collect_analysis_results(project=args.project, model_family=args.model_family)
    averaged = average_results(results, scope=args.scope, group_by=group_by)

    plot_results(
        averaged,
        args.output,
        scope=args.scope,
        group_by=group_by,
        sort_bars=args.sort_bars,
        show=not args.no_show,
        model_family=args.model_family,
    )


if __name__ == "__main__":
    main()
