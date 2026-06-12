#!/usr/bin/env python3
"""Plot SGD-only 2D heat maps over batch size and learning rate.

This script loads the newest W&B minimum-analysis artifact JSON per artifact
collection, keeps only artifacts for a configured model family and SGD optimizer,
then filters SGD runs by configured batch-size and learning-rate allow-lists. It
plots every configured scalar metric as a separate heat-map image whose x/y
dimensions are learning rate and batch size. The CLI takes only the config-file
path; the config output is interpreted as an output folder.

For adaptive sharpness, the script reduces each adaptive sharpness curve to its
last valid entry, making it comparable to scalar summary metrics. Multiple runs
with the same (batch size, learning rate) cell, usually different seeds, are
averaged before plotting.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from numbers import Real
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


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

HESSIAN_METRICS = (
    "negative_curvature_ratio",
    "raw_top_eigenvalue",
    "normalized_top_eigenvalue",
    "raw_trace_mean",
    "normalized_trace",
)

# Scalar panels. Adaptive sharpness metrics use only the final valid curve point.
PLOT_METRICS: tuple[tuple[str | tuple[str, str], str, str], ...] = (
    ("train_loss", "Train Loss", "loss"),
    ("test_loss", "Test Loss", "loss"),
    ("train_test_loss_gap", "Train-Test Loss Gap", "test loss - train loss"),
    ("train_accuracy", "Train Accuracy", "accuracy"),
    ("test_accuracy", "Test Accuracy", "accuracy"),
    ("train_test_accuracy_gap", "Train-Test Accuracy Gap", "train accuracy - test accuracy"),
    ("gradient_norm_full_train_dataset", "Gradient Norm", "gradient norm"),
    (("hessian_metrics", "negative_curvature_ratio"), "Negative Curvature Ratio", "ratio"),
    (("hessian_metrics", "raw_top_eigenvalue"), "Raw Hessian Max Eigenvalue", "λ_max(H)"),
    (("hessian_metrics", "normalized_top_eigenvalue"), "Normalized Hessian Max Eigenvalue", "λ_max(H) · ||w||²"),
    (("hessian_metrics", "raw_trace_mean"), "Raw Hessian Trace", "tr(H)"),
    (("hessian_metrics", "normalized_trace"), "Normalized Hessian Trace", "tr(H) · ||w||²"),
    (("adaptive_sharpness_last", "rho"), "Final Adaptive Rho", "rho"),
    (("adaptive_sharpness_last", "sharpness_delta"), "Final Adaptive Sharpness", "loss increase"),
    (("adaptive_sharpness_last", "normalized_sharpness_delta"), "Final Loss-Normalized Adaptive Sharpness", "loss increase / base loss"),
    (("adaptive_sharpness_last", "base_loss"), "Final Adaptive Base Loss", "base loss"),
    (("adaptive_sharpness_last", "perturbed_loss"), "Final Adaptive Perturbed Loss", "perturbed loss"),
    (("adaptive_sharpness_last", "max_batch_sharpness_delta"), "Final Max-Batch Adaptive Sharpness", "loss increase"),
)


def metric_id(metric_key: str | tuple[str, str]) -> str:
    """Stable CLI id for a configured metric."""
    return metric_key if isinstance(metric_key, str) else ".".join(metric_key)


def available_metric_ids() -> list[str]:
    return [metric_id(metric_key) for metric_key, _, _ in PLOT_METRICS]


def safe_filename_stem(value: str) -> str:
    """Return a portable filename stem for a metric id."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "metric"


def resolve_plot_metric(metric: str) -> tuple[str | tuple[str, str], str, str]:
    """Resolve a CLI metric name to one PLOT_METRICS entry.

    Accepts the full id, e.g. ``hessian_metrics.raw_trace_mean``. For
    convenience, also accepts the final component when that name is unambiguous,
    e.g. ``raw_trace_mean``.
    """
    normalized = metric.strip()
    by_full_id = {metric_id(metric_key): entry for entry in PLOT_METRICS for metric_key in [entry[0]]}
    if normalized in by_full_id:
        return by_full_id[normalized]

    suffix_matches = [
        entry for entry in PLOT_METRICS
        if metric_id(entry[0]).split(".")[-1] == normalized
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    allowed = ", ".join(available_metric_ids())
    if suffix_matches:
        raise ValueError(f"Metric {metric!r} is ambiguous. Use one of: {allowed}")
    raise ValueError(f"Unknown metric {metric!r}. Use one of: {allowed}")


# -----------------------------------------------------------------------------
# Numeric and JSON helpers
# -----------------------------------------------------------------------------


def finite_real(value: Any) -> float | None:
    """Return value as a finite float, otherwise None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, Real):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def average_numeric(values: Iterable[Any]) -> float | None:
    """Mean of finite numeric values; returns None when no finite values exist."""
    clean = [value for value in (finite_real(value) for value in values) if value is not None]
    return sum(clean) / len(clean) if clean else None


def get_nested(dct: dict[str, Any], *keys: str) -> Any:
    value: Any = dct
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def result_name(result: dict[str, Any]) -> str:
    return result.get("source_run_name") or result.get("_artifact_name") or result.get("_wandb_run_name") or ""


def format_param(value: Any) -> str:
    numeric = finite_real(value)
    if numeric is None:
        return "?"
    return f"{numeric:g}"


# -----------------------------------------------------------------------------
# W&B loading
# -----------------------------------------------------------------------------


def load_json_from_artifact(artifact: Any) -> dict[str, Any]:
    artifact_dir = Path(artifact.download())
    json_files = list(artifact_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON found in {artifact.name}")

    with open(json_files[0], "r") as f:
        return json.load(f)


def artifact_matches_model_family(artifact_name: str, model_family: str) -> bool:
    marker = MODEL_FAMILY_ARTIFACT_MARKERS[model_family].lower()
    return marker in artifact_name.lower()


def collect_analysis_results(project: str = PROJECT, *, model_family: str) -> list[dict[str, Any]]:
    """Load latest analysis JSONs for one model family from W&B."""
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    runs = api.runs(f"{entity}/{project}")

    latest_by_collection: dict[str, Any] = {}
    skipped_by_model_family = 0

    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type != "min_grad_analysis":
                continue

            collection = artifact.name.split(":")[0]
            if not artifact_matches_model_family(collection, model_family):
                skipped_by_model_family += 1
                continue

            if collection not in latest_by_collection or artifact.created_at > latest_by_collection[collection].created_at:
                latest_by_collection[collection] = artifact

    print(
        f"Model-family filter: keeping analysis artifacts matching "
        f"{MODEL_FAMILY_ARTIFACT_MARKERS[model_family]!r}; "
        f"skipped {skipped_by_model_family} non-matching analysis artifact version(s)."
    )

    results: list[dict[str, Any]] = []
    for artifact in latest_by_collection.values():
        try:
            data = load_json_from_artifact(artifact)
            data["_artifact_name"] = artifact.name
            data["_model_family"] = model_family
            results.append(data)
            print(f"Loaded latest {artifact.name}")
        except Exception as exc:
            print(f"Skipping {artifact.name}: {exc}")

    print(f"Loaded {len(results)} latest analysis artifact JSON file(s) after model filtering.")
    return results


# -----------------------------------------------------------------------------
# Metadata parsing and metric extraction
# -----------------------------------------------------------------------------


def parse_run_name(name: str) -> dict[str, Any]:
    """Parse model/optimizer/config metadata from artifact or run names."""
    cleaned = name.replace("-min_grad_analysis", "").replace("_min_grad_analysis", "")
    cleaned_lower = cleaned.lower()

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

    family_match = re.search(r"model[-_](?:final_model[-_])?(resnet20|vit)(?:[_-]|$)", cleaned_lower)
    if family_match:
        info["model_family"] = family_match.group(1)
        after_family = cleaned_lower[family_match.end():].lstrip("_-")
        opt_match = re.match(r"([a-zA-Z][a-zA-Z0-9]*)(?=[_-]|$)", after_family)
        if opt_match and not opt_match.group(1).startswith(("lr", "bs", "seed", "mom", "wd")):
            info["optimizer"] = opt_match.group(1).lower()

    patterns = {
        "lr": r"_lr([0-9.eE+-]+)",
        "bs": r"_bs([0-9]+)",
        "seed": r"_seed([0-9]+)",
        "momentum": r"_mom([0-9.eE+-]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, cleaned)
        if not match:
            continue
        raw_value = match.group(1)
        info[key] = int(raw_value) if key in {"bs", "seed"} else float(raw_value)

    info["nesterov"] = "_nesterov" in cleaned_lower

    schedule_match = re.search(r"_bs[0-9]+_([a-zA-Z0-9]+)_seed", cleaned_lower)
    if schedule_match:
        info["schedule"] = schedule_match.group(1)

    return info


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
            return finite_real(max(eigenvalues, key=lambda x: x.real))
        return None

    if metric == "raw_top_eigenvalue":
        return raw_top_eigenvalue()

    if metric == "normalized_top_eigenvalue":
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


def adaptive_curve(result: dict[str, Any]) -> list[dict[str, Any]]:
    curve = result.get("elementwise_adaptive_sharpness_by_radius") or result.get("averaged_adaptive_sharpness_curve") or []
    return curve if isinstance(curve, list) else []


def final_adaptive_sharpness_point(result: dict[str, Any]) -> dict[str, float | None]:
    """Return the last valid adaptive-sharpness curve point as scalar metrics."""
    valid_points: list[dict[str, Any]] = []
    for point in adaptive_curve(result):
        if not isinstance(point, dict):
            continue
        rho = finite_real(point.get("rho"))
        if rho is None:
            rho = finite_real(point.get("relative_radius"))
        if rho is None:
            continue
        copied = dict(point)
        copied["rho"] = rho
        valid_points.append(copied)

    empty = {
        "rho": None,
        "sharpness_delta": None,
        "normalized_sharpness_delta": None,
        "base_loss": None,
        "perturbed_loss": None,
        "max_batch_sharpness_delta": None,
    }
    if not valid_points:
        return empty

    # The analyzer emits points in increasing rho. Sorting makes the behavior
    # robust to hand-edited or merged JSON files with shuffled curve entries.
    last = sorted(valid_points, key=lambda item: float(item["rho"]))[-1]

    sharpness_delta = finite_real(last.get("sharpness_delta"))
    base_loss = finite_real(last.get("base_loss"))
    normalized_delta = finite_real(last.get("normalized_sharpness_delta"))
    if normalized_delta is None and sharpness_delta is not None and base_loss not in (None, 0.0):
        normalized_delta = sharpness_delta / base_loss

    return {
        "rho": finite_real(last.get("rho")),
        "sharpness_delta": sharpness_delta,
        "normalized_sharpness_delta": normalized_delta,
        "base_loss": base_loss,
        "perturbed_loss": finite_real(last.get("perturbed_loss")),
        "max_batch_sharpness_delta": finite_real(last.get("max_batch_sharpness_delta")),
    }


def metric_value(result: dict[str, Any], key: str | tuple[str, str]) -> Any:
    if isinstance(key, tuple):
        if len(key) == 2 and key[0] == "hessian_metrics":
            return hessian_metric_value(result, key[1])
        return get_nested(result, *key)
    return result.get(key)


# -----------------------------------------------------------------------------
# SGD filtering and aggregation into heat-map cells
# -----------------------------------------------------------------------------


def lr_allowed(lr: float, allowed_learning_rates: set[float] | None) -> bool:
    """Return whether lr is in the configured allow-list, with float-safe matching."""
    if allowed_learning_rates is None:
        return True
    return any(math.isclose(lr, allowed, rel_tol=1e-12, abs_tol=0.0) for allowed in allowed_learning_rates)


def build_sgd_records(
    results: list[dict[str, Any]],
    *,
    allowed_batch_sizes: set[int] | None = None,
    allowed_learning_rates: set[float] | None = None,
) -> list[dict[str, Any]]:
    """Convert raw JSON results into scalar SGD records after config allow-list filtering."""
    records: list[dict[str, Any]] = []
    skipped_non_sgd = 0
    skipped_missing_grid_key = 0
    skipped_batch_size = 0
    skipped_learning_rate = 0

    for result in results:
        meta = parse_run_name(result_name(result))
        if meta["optimizer"] != "sgd":
            skipped_non_sgd += 1
            continue
        if meta["bs"] is None or meta["lr"] is None:
            skipped_missing_grid_key += 1
            continue

        bs = int(meta["bs"])
        lr = float(meta["lr"])

        if allowed_batch_sizes is not None and bs not in allowed_batch_sizes:
            skipped_batch_size += 1
            continue
        if not lr_allowed(lr, allowed_learning_rates):
            skipped_learning_rate += 1
            continue

        record: dict[str, Any] = {
            "bs": bs,
            "lr": lr,
            "seed": meta.get("seed"),
            "source_run_name": result_name(result),
            "adaptive_sharpness_last": final_adaptive_sharpness_point(result),
        }

        for metric in SUMMARY_METRICS:
            record[metric] = finite_real(result.get(metric))

        record["hessian_metrics"] = {}
        for metric in HESSIAN_METRICS:
            record["hessian_metrics"][metric] = finite_real(hessian_metric_value(result, metric))

        records.append(record)

    print(
        f"SGD filter: kept {len(records)} run(s), skipped {skipped_non_sgd} non-SGD run(s), "
        f"skipped {skipped_missing_grid_key} SGD run(s) without both bs and lr, "
        f"skipped {skipped_batch_size} SGD run(s) outside batch-size allow-list, "
        f"skipped {skipped_learning_rate} SGD run(s) outside learning-rate allow-list."
    )
    return records


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average records over all runs/seeds sharing the same (bs, lr)."""
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["bs"], record["lr"])].append(record)

    aggregated: list[dict[str, Any]] = []
    for (bs, lr), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        out: dict[str, Any] = {
            "bs": bs,
            "lr": lr,
            "n_runs": len(group),
            "source_run_name": f"sgd | bs={format_param(bs)} | lr={format_param(lr)} | n={len(group)}",
        }

        for metric in SUMMARY_METRICS:
            out[metric] = average_numeric(record.get(metric) for record in group)

        out["hessian_metrics"] = {}
        for metric in HESSIAN_METRICS:
            out["hessian_metrics"][metric] = average_numeric(
                get_nested(record, "hessian_metrics", metric) for record in group
            )

        out["adaptive_sharpness_last"] = {}
        for metric in (
            "rho",
            "sharpness_delta",
            "normalized_sharpness_delta",
            "base_loss",
            "perturbed_loss",
            "max_batch_sharpness_delta",
        ):
            out["adaptive_sharpness_last"][metric] = average_numeric(
                get_nested(record, "adaptive_sharpness_last", metric) for record in group
            )

        aggregated.append(out)

    print(f"Aggregated into {len(aggregated)} heat-map cell(s).")
    return aggregated


# -----------------------------------------------------------------------------
# Heat-map plotting
# -----------------------------------------------------------------------------




def clear_heatmap_cmap() -> LinearSegmentedColormap:
    """Blue -> green -> yellow -> red colormap with grey for missing cells."""
    cmap = LinearSegmentedColormap.from_list(
        "clear_blue_green_yellow_red",
        ["#2166ac", "#1a9850", "#ffffbf", "#d73027"],
    )
    cmap.set_bad(color="#eeeeee")
    return cmap


def sorted_unique(values: Iterable[Any]) -> list[Any]:
    return sorted(set(values))


def matrix_for_metric(records: list[dict[str, Any]], metric_key: str | tuple[str, str]) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    batch_sizes = sorted_unique(int(record["bs"]) for record in records)
    learning_rates = sorted_unique(float(record["lr"]) for record in records)

    values = np.full((len(batch_sizes), len(learning_rates)), np.nan, dtype=float)
    counts = np.zeros((len(batch_sizes), len(learning_rates)), dtype=int)

    bs_index = {bs: idx for idx, bs in enumerate(batch_sizes)}
    lr_index = {lr: idx for idx, lr in enumerate(learning_rates)}

    for record in records:
        value = finite_real(metric_value(record, metric_key))
        if value is None:
            continue
        i = bs_index[int(record["bs"])]
        j = lr_index[float(record["lr"])]
        values[i, j] = value
        counts[i, j] = int(record.get("n_runs", 1))

    return batch_sizes, learning_rates, values, counts


def format_cell_value(value: float) -> str:
    if not math.isfinite(float(value)):
        return ""
    abs_value = abs(float(value))
    if abs_value != 0 and (abs_value < 1e-3 or abs_value >= 1e4):
        return f"{value:.2e}"
    return f"{value:.3g}"


def plot_metric_heatmap(
    ax: Any,
    records: list[dict[str, Any]],
    metric_key: str | tuple[str, str],
    title: str,
    colorbar_label: str,
    *,
    annotate: bool,
) -> None:
    batch_sizes, learning_rates, values, counts = matrix_for_metric(records, metric_key)
    masked_values = np.ma.masked_invalid(values)

    ax.set_title(title)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("batch size")

    if not batch_sizes or not learning_rates or masked_values.count() == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    image = ax.imshow(
        masked_values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=clear_heatmap_cmap(),
    )
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(colorbar_label, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(learning_rates)))
    ax.set_xticklabels([format_param(lr) for lr in learning_rates], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_yticklabels([format_param(bs) for bs in batch_sizes], fontsize=8)

    # Draw grid lines between cells.
    ax.set_xticks(np.arange(-0.5, len(learning_rates), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(batch_sizes), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.75, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if math.isfinite(values[i, j]):
                    label = format_cell_value(values[i, j])
                    if counts[i, j] > 1:
                        label = f"{label}\nn={counts[i, j]}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=7)


def save_one_metric_heatmap(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    metric_key: str | tuple[str, str],
    title: str,
    colorbar_label: str,
    annotate: bool,
    show: bool,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    plot_metric_heatmap(ax, records, metric_key, title, colorbar_label, annotate=annotate)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved {metric_id(metric_key)} heat map to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_heatmaps(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    model_family: str,
    annotate: bool,
    show: bool,
) -> None:
    if not records:
        raise RuntimeError("No SGD analysis results found after filtering by model family and grid keys.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for metric_key, title, colorbar_label in PLOT_METRICS:
        metric_name = metric_id(metric_key)
        output_path = output_dir / f"{safe_filename_stem(metric_name)}.png"
        save_one_metric_heatmap(
            records,
            output_path,
            metric_key=metric_key,
            title=title,
            colorbar_label=colorbar_label,
            annotate=annotate,
            show=show,
        )
        saved_paths.append(output_path)

    print(f"Saved {len(saved_paths)} heat-map image(s) to {output_dir}")


# -----------------------------------------------------------------------------
# Config and CLI
# -----------------------------------------------------------------------------


def require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON/YAML object.")
    return value


def read_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config file, with optional YAML support when PyYAML is installed."""
    config_path = Path(path)
    with open(config_path, "r") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("YAML config files require PyYAML. Use JSON or install pyyaml.") from exc
            loaded = yaml.safe_load(f)
        else:
            loaded = json.load(f)
    return require_mapping(loaded, name=str(config_path))


def optional_int_set(config: dict[str, Any], key: str) -> set[int] | None:
    raw = config.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key!r} must be a non-empty list of integers when provided.")
    values: set[int] = set()
    for item in raw:
        value = finite_real(item)
        if value is None or not float(value).is_integer():
            raise ValueError(f"Every entry in {key!r} must be an integer; got {item!r}.")
        values.add(int(value))
    return values


def optional_float_set(config: dict[str, Any], key: str) -> set[float] | None:
    raw = config.get(key)
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key!r} must be a non-empty list of numbers when provided.")
    values: set[float] = set()
    for item in raw:
        value = finite_real(item)
        if value is None:
            raise ValueError(f"Every entry in {key!r} must be numeric; got {item!r}.")
        values.add(float(value))
    return values


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the heat-map configuration."""
    model_family = config.get("model_family", "resnet20")
    if model_family not in MODEL_FAMILY_ARTIFACT_MARKERS:
        allowed = ", ".join(MODEL_FAMILY_ARTIFACT_MARKERS)
        raise ValueError(f"model_family must be one of: {allowed}; got {model_family!r}.")

    output = config.get("output", "sgd_heatmaps")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("output must be a non-empty output-directory path.")

    project = config.get("project", PROJECT)
    if not isinstance(project, str) or not project.strip():
        raise ValueError("project must be a non-empty string when provided.")

    annotate = config.get("annotate", True)
    if not isinstance(annotate, bool):
        raise ValueError("annotate must be true or false.")

    show = config.get("show", False)
    if not isinstance(show, bool):
        raise ValueError("show must be true or false.")

    return {
        "project": project,
        "model_family": model_family,
        "output": output,
        "batch_sizes": optional_int_set(config, "batch_sizes"),
        "learning_rates": optional_float_set(config, "learning_rates"),
        "annotate": annotate,
        "show": show,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load W&B minimum-analysis artifacts and plot every configured SGD-only "
            "2D heat map into separate PNG files inside the configured output folder."
        )
    )
    parser.add_argument("config", help="Path to a JSON config file. YAML also works if PyYAML is installed.")
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print metric ids that will be plotted and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_metrics:
        print("Metrics that will be plotted:")
        for metric in available_metric_ids():
            print(f"  {metric}")
        return

    config = validate_config(read_config(args.config))

    raw_results = collect_analysis_results(
        project=config["project"],
        model_family=config["model_family"],
    )
    sgd_records = build_sgd_records(
        raw_results,
        allowed_batch_sizes=config["batch_sizes"],
        allowed_learning_rates=config["learning_rates"],
    )
    aggregated_records = aggregate_records(sgd_records)

    plot_all_heatmaps(
        aggregated_records,
        config["output"],
        model_family=config["model_family"],
        annotate=config["annotate"],
        show=config["show"],
    )


if __name__ == "__main__":
    main()
