import argparse

import wandb
import analyzer

PROJECT = "OptiML_Minima"

MODEL_PREFIXES = {
    "resnet20": "model-resnet20_",
    "vit": "model-vit_",
}


def artifact_matches_model_type(run_name, model_type):
    """Return True if an artifact collection name matches the requested model type."""
    if model_type == "all":
        return run_name.startswith("model-")

    prefix = MODEL_PREFIXES[model_type]
    return run_name.startswith(prefix)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run minimum analysis on latest W&B model artifacts."
    )
    parser.add_argument(
        "--model_type",
        choices=["all", "resnet20", "vit"],
        default="all",
        help=(
            "Which model artifacts to analyse. "
            "Uses artifact collection prefixes: model-resnet20_ or model-vit_."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print matching artifacts without running analyzer.analyze(...).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api = wandb.Api()
    entity = api.default_entity

    runs = api.runs(f"{entity}/{PROJECT}")

    latest_by_collection = {}

    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type != "model":
                continue

            # Example:
            # model-resnet20_muon_lr0.02_wd0.0_bs128_cosine_seed1:v0
            # -> model-resnet20_muon_lr0.02_wd0.0_bs128_cosine_seed1
            collection = artifact.name.split(":")[0]

            if not artifact_matches_model_type(collection, args.model_type):
                continue

            if (
                collection not in latest_by_collection
                or artifact.created_at > latest_by_collection[collection].created_at
            ):
                latest_by_collection[collection] = artifact

    if not latest_by_collection:
        print(f"No model artifacts matched --model_type {args.model_type!r}.")
        return

    print(
        f"Found {len(latest_by_collection)} latest model artifact(s) "
        f"for --model_type {args.model_type!r}."
    )

    for run_name in sorted(latest_by_collection):
        artifact_ref = f"{entity}/{PROJECT}/{run_name}:latest"

        print("-" * 80)
        print(f"Analyzing latest model artifact: {artifact_ref}")
        print(f"Run name: {run_name}")

        if args.dry_run:
            continue

        analyzer.analyze(run_name)


if __name__ == "__main__":
    main()
