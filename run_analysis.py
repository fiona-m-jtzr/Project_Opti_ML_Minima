import argparse

import wandb
import analyzer

PROJECT = "OptiML_Minima"

MODEL_PREFIXES = {
    "resnet20": "model-FINAL_MODEL_resnet20_",
    "vit": "model-FINAL_MODEL_vit_",
}


def artifact_model_type(run_name):
    """
    Infer the model type from a W&B artifact collection name.

    Expected typed names look like:
      model-FINAL_MODEL_resnet20_muon_lr0.02_wd0.0_bs128_cosine_seed1
      model-FINAL_MODEL_vit_sam_rho0.1_baseadam_lr0.0005_wd0.0_bs256_cosine_warmup_seed1

    Untyped legacy names like:
      model-sam_rho0.01_basesgd_lr0.1_wd0.0001_bs128_cosine_seed1

    return None and are ignored.
    """
    for model_type, prefix in MODEL_PREFIXES.items():
        if run_name.startswith(prefix):
            return model_type
    return None


def artifact_matches_model_type(run_name, model_type):
    """Return True if an artifact collection name matches the requested model type."""
    inferred_model_type = artifact_model_type(run_name)

    # Always ignore model artifacts that do not explicitly encode the architecture.
    if inferred_model_type is None:
        return False

    if model_type == "all":
        return True

    return inferred_model_type == model_type


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run minimum analysis on latest W&B model artifacts."
    )
    parser.add_argument(
        "--model_type",
        choices=["all", "resnet20", "vit"],
        default="all",
        help=(
            "Which typed model artifacts to analyse. "
            "'all' means all known typed architectures only: "
            "model-FINAL_MODEL_resnet20_* and model-FINAL_MODEL_vit_*. "
            "Untyped legacy artifacts such as model-sam_* are ignored."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print matching artifacts without running analyzer.analyze(...).",
    )
    parser.add_argument(
        "--show_ignored",
        action="store_true",
        help="Print model artifacts ignored because they have no known architecture prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api = wandb.Api()
    entity = api.default_entity

    runs = api.runs(f"{entity}/{PROJECT}")

    latest_by_collection = {}
    ignored_untyped = set()

    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type != "model":
                continue

            # Example:
            # model-resnet20_muon_lr0.02_wd0.0_bs128_cosine_seed1:v0
            # -> model-resnet20_muon_lr0.02_wd0.0_bs128_cosine_seed1
            collection = artifact.name.split(":")[0]

            if not collection.startswith("model-"):
                continue

            if artifact_model_type(collection) is None:
                ignored_untyped.add(collection)
                continue

            if not artifact_matches_model_type(collection, args.model_type):
                continue

            if (
                collection not in latest_by_collection
                or artifact.created_at > latest_by_collection[collection].created_at
            ):
                latest_by_collection[collection] = artifact

    if args.show_ignored and ignored_untyped:
        print("Ignored untyped model artifact collection(s):")
        for name in sorted(ignored_untyped):
            print(f"  - {name}")
        print()

    if not latest_by_collection:
        print(
            f"No typed model artifacts matched --model_type {args.model_type!r}. "
            "Expected prefixes are model-resnet20_ and model-vit_."
        )
        return

    print(
        f"Found {len(latest_by_collection)} latest typed model artifact(s) "
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
