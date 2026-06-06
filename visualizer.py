#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb


PROJECT = "OptiML_Minima"


def average_numeric(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None



def average_sharpness_curves(group):
    curves = [r["scale_invariant_sharpness_by_radius"] for r in group]

    averaged = []

    for i in range(len(curves[0])):
        radius = curves[0][i]["relative_radius"]

        normalized_values = []

        for run, curve in zip(group, curves):
            train_loss = run.get("train_loss")

            if train_loss is None or train_loss == 0:
                continue

            normalized_values.append(
                curve[i]["sharpness_delta"] / train_loss
            )

        averaged.append({
            "relative_radius": radius,
            "normalized_sharpness_delta": average_numeric(
                normalized_values
            ),
        })

    return averaged

def group_key_without_seed(result):
    meta = parse_run_name(result_name(result))
    return (
        meta["optimizer"],
        meta["lr"],
        meta["bs"],
        meta["momentum"],
        meta["nesterov"],
        meta["schedule"],
    )


def average_results_across_seeds(results):
    grouped = {}

    for result in results:
        grouped.setdefault(group_key_without_seed(result), []).append(result)

    averaged = []

    for key, group in grouped.items():
        base = dict(group[0])

        base["source_run_name"] = result_name(group[0]).replace(
            f"_seed{parse_run_name(result_name(group[0]))['seed']}", ""
        )
        base["_num_seeds"] = len(group)

        for metric in [
            "train_loss",
            "test_loss",
            "train_accuracy",
            "test_accuracy",
            "train_test_loss_gap",
            "train_test_accuracy_gap",
            "gradient_norm_full_train_dataset",
        ]:
            base[metric] = average_numeric([r.get(metric) for r in group])

        base["hessian_metrics"] = dict(base.get("hessian_metrics", {}))
        for metric in [
            "negative_curvature_ratio",
            "normalized_top_eigenvalue",
            "normalized_trace",
        ]:
            base["hessian_metrics"][metric] = average_numeric([
                get_nested(r, "hessian_metrics", metric) for r in group
            ])

        base["averaged_sharpness_curve"] = average_sharpness_curves(group)

        averaged.append(base)

    return averaged


def load_json_from_artifact(artifact):
    artifact_dir = Path(artifact.download())
    json_files = list(artifact_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON found in {artifact.name}")

    with open(json_files[0], "r") as f:
        return json.load(f)


def collect_analysis_results():
    api = wandb.Api()
    entity = api.default_entity
    runs = api.runs(f"{entity}/{PROJECT}")

    latest_by_collection = {}

    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type != "analysis":
                continue

            # e.g. model-sgd_lr0.1_seed42-minimum-analysis:v3
            collection = artifact.name.split(":")[0]

            if (
                collection not in latest_by_collection
                or artifact.created_at > latest_by_collection[collection].created_at
            ):
                latest_by_collection[collection] = artifact

    results = []

    for artifact in latest_by_collection.values():
        try:
            data = load_json_from_artifact(artifact)
            data["_artifact_name"] = artifact.name
            results.append(data)
            print(f"Loaded latest {artifact.name}")
        except Exception as e:
            print(f"Skipping {artifact.name}: {e}")

    return results


def parse_run_name(name):
    name = name.replace("-minimum-analysis", "")

    info = {
        "optimizer": "unknown",
        "lr": None,
        "bs": None,
        "seed": None,
        "momentum": None,
        "nesterov": False,
        "schedule": None,
    }

    opt_match = re.search(r"model-([a-zA-Z0-9]+)", name)
    if opt_match:
        info["optimizer"] = opt_match.group(1)

    patterns = {
        "lr": r"_lr([0-9.eE+-]+)",
        "bs": r"_bs([0-9]+)",
        "seed": r"_seed([0-9]+)",
        "momentum": r"_mom([0-9.eE+-]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            value = match.group(1)
            info[key] = int(value) if key in {"bs", "seed"} else float(value)

    info["nesterov"] = "_nesterov" in name

    schedule_match = re.search(r"_bs[0-9]+_([a-zA-Z0-9]+)_seed", name)
    if schedule_match:
        info["schedule"] = schedule_match.group(1)

    return info


def result_name(result):
    return (
        result.get("source_run_name")
        or result.get("_artifact_name")
        or result.get("_wandb_run_name")
        or ""
    )


def get_nested(d, *keys):
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def sort_results(results):
    def key(result):
        meta = parse_run_name(result_name(result))
        return (
            meta["bs"] if meta["bs"] is not None else 10**18,
            meta["lr"] if meta["lr"] is not None else float("inf"),
            meta["optimizer"],
            result_name(result),
        )

    return sorted(results, key=key)


def build_optimizer_color_map(meta):
    optimizers = sorted({m["optimizer"] for m in meta})
    colors = plt.cm.tab10(range(len(optimizers)))
    return dict(zip(optimizers, colors))


def build_combo_color_map(meta):
    combos = sorted({
        (m["optimizer"], m["bs"], m["lr"])
        for m in meta
    })

    cmap = plt.cm.get_cmap("tab20", max(len(combos), 1))

    return {
        combo: cmap(i)
        for i, combo in enumerate(combos)
    }


def format_param_label(params):
    lr = params["lr"]
    bs = params["bs"]

    lr_txt = f"{lr:g}" if lr is not None else "?"
    bs_txt = str(bs) if bs is not None else "?"

    return f"bs={bs_txt}, lr={lr_txt}"


def bar_metric(ax, meta, values, title, ylabel, color_map):
    clean = [
        (params, value)
        for params, value in zip(meta, values)
        if value is not None
    ]

    if not clean:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return

    xs = list(range(len(clean)))
    params_list = [item[0] for item in clean]
    vals = [item[1] for item in clean]
    colors = [color_map[p["optimizer"]] for p in params_list]

    bars = ax.bar(xs, vals, color=colors, alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.3)

    ymin, ymax = ax.get_ylim()
    offset = 0.015 * (ymax - ymin)

    for bar, params, value in zip(bars, params_list, vals):
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            format_param_label(params),
            ha="center",
            va=va,
            fontsize=7,
            rotation=0,
        )


def plot_sharpness_curves(ax, results, meta, combo_color_map):
    for result, params in zip(results, meta):
        curve = result.get("scale_invariant_sharpness_by_radius", [])
        if not curve:
            continue

        loss = result.get("train_loss")  # or "test_loss"
        if loss is None or loss == 0:
            continue

        curve = result.get("averaged_sharpness_curve", [])

        radii = [p["relative_radius"] for p in curve]
        deltas = [p["normalized_sharpness_delta"] for p in curve]

        combo = (params["optimizer"], params["bs"], params["lr"])

        ax.plot(
            radii,
            deltas,
            marker="o",
            linewidth=2,
            alpha=0.9,
            color=combo_color_map[combo],
            label=f"{params['optimizer']} | bs={params['bs']} | lr={params['lr']:g}",
        )

    ax.set_title("Loss-Normalized Scale-Invariant Sharpness by Radius")
    ax.set_xscale("log")
    ax.set_xlabel("relative radius")
    ax.set_ylabel("sharpness delta / train loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def add_optimizer_legend(fig, color_map):
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=color,
            label=optimizer,
            markersize=10,
        )
        for optimizer, color in color_map.items()
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=max(1, len(handles)),
        frameon=True,
        title="Optimizer color for bar plots",
    )


def plot_results(results, output_path):
    if not results:
        raise RuntimeError("No analysis results found.")

    results = sort_results(results)
    names = [result_name(r) for r in results]
    meta = [parse_run_name(name) for name in names]

    optimizer_color_map = build_optimizer_color_map(meta)
    combo_color_map = build_combo_color_map(meta)

    fig, axes = plt.subplots(4, 2, figsize=(20, 18))
    axes = axes.flatten()

    bar_metric(
        axes[0],
        meta,
        [r.get("train_loss") for r in results],
        "Train Loss",
        "loss",
        optimizer_color_map,
    )

    bar_metric(
        axes[1],
        meta,
        [r.get("test_loss") for r in results],
        "Test Loss",
        "loss",
        optimizer_color_map,
    )

    bar_metric(
        axes[2],
        meta,
        [r.get("train_test_loss_gap") for r in results],
        "Train-Test Gap",
        "test loss - train loss",
        optimizer_color_map,
    )

    bar_metric(
        axes[3],
        meta,
        [r.get("gradient_norm_full_train_dataset") for r in results],
        "Gradient Norm",
        "gradient norm",
        optimizer_color_map,
    )

    bar_metric(
        axes[4],
        meta,
        [
            get_nested(r, "hessian_metrics", "negative_curvature_ratio")
            for r in results
        ],
        "Negative Curvature Ratio",
        "ratio",
        optimizer_color_map,
    )

    bar_metric(
        axes[5],
        meta,
        [
            get_nested(r, "hessian_metrics", "normalized_top_eigenvalue")
            for r in results
        ],
        "Normalized Hessian Max Eigenvalue",
        "λ_max · ||w||²",
        optimizer_color_map,
    )

    bar_metric(
        axes[6],
        meta,
        [
            get_nested(r, "hessian_metrics", "normalized_trace")
            for r in results
        ],
        "Normalized Hessian Trace",
        "tr(H) · ||w||²",
        optimizer_color_map,
    )

    plot_sharpness_curves(
        axes[7],
        results,
        meta,
        combo_color_map,
    )

    add_optimizer_legend(fig, optimizer_color_map)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="minimum_analysis_summary.png")
    args = parser.parse_args()

    results = collect_analysis_results()
    results = average_results_across_seeds(results)
    plot_results(results, args.output)


if __name__ == "__main__":
    main()