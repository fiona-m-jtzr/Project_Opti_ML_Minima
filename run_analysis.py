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


def artifact_optimizer(run_name):
    """
    Infer the primary optimizer token from a typed model artifact collection name.

    Examples:
      model-FINAL_MODEL_resnet20_adam_lr0.001_... -> adam
      model-FINAL_MODEL_resnet20_muon_muonLR0.02_adamLR0.001_... -> muon
      model-FINAL_MODEL_vit_sam_rho0.1_baseadam_lr0.0005_... -> sam
    """
    model_type = artifact_model_type(run_name)
    if model_type is None:
        return None

    prefix = MODEL_PREFIXES[model_type]
    remainder = run_name[len(prefix):]
    return remainder.split("_", 1)[0] if remainder else None


def artifact_matches_model_type(run_name, model_type):
    """Return True if an artifact collection name matches the requested model type."""
    inferred_model_type = artifact_model_type(run_name)

    # Always ignore model artifacts that do not explicitly encode the architecture.
    if inferred_model_type is None:
        return False

    if model_type == "all":
        return True

    return inferred_model_type == model_type


def parse_artifact_collection_and_version(artifact):
    """Return (collection, version_or_alias) from a W&B artifact object."""
    parts = artifact.name.split(":", 1)
    collection = parts[0]
    version = parts[1] if len(parts) == 2 else getattr(artifact, "version", None)
    return collection, version


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run minimum analysis on W&B model artifacts."
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
        "--artifact_alias",
        default="latest",
        help=(
            "W&B model artifact alias/version to analyse for every matching collection, "
            "for example latest or v12."
        ),
    )
    parser.add_argument(
        "--analysis_artifact_suffix",
        default=None,
        help=(
            "Optional suffix to append to each analysis artifact name. "
            "Defaults to the artifact alias when --artifact_alias is not latest."
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--samples_per_radius", type=int, default=20)
    parser.add_argument(
        "--adaptive_sharpness_rhos",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3, 2e-3, 3e-3],
    )
    parser.add_argument(
        "--adaptive_sharpness_rho",
        type=float,
        default=None,
        help="Deprecated single-rho option. If provided, only this rho is used.",
    )
    parser.add_argument("--adaptive_sharpness_steps", type=int, default=20)
    parser.add_argument("--adaptive_sharpness_batches", type=int, default=8)
    parser.add_argument(
        "--adaptive_sharpness_norm",
        choices=["linf", "l2"],
        default="linf",
    )
    parser.add_argument(
        "--no_adaptive_sharpness_logit_normalize",
        dest="adaptive_sharpness_logit_normalize",
        action="store_false",
        help="Disable logit normalization for element-wise adaptive sharpness.",
    )
    parser.add_argument(
        "--adaptive_sharpness_union_batches",
        dest="adaptive_sharpness_average_batches",
        action="store_false",
        help=(
            "Optimize one shared perturbation over all selected batches instead "
            "of averaging per-batch worst-case sharpness."
        ),
    )
    parser.add_argument(
        "--skip_adaptive_sharpness",
        action="store_true",
        help="Skip the element-wise adaptive sharpness metric.",
    )
    parser.set_defaults(
        adaptive_sharpness_logit_normalize=True,
        adaptive_sharpness_average_batches=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    adaptive_sharpness_rhos = (
        [args.adaptive_sharpness_rho]
        if args.adaptive_sharpness_rho is not None
        else args.adaptive_sharpness_rhos
    )

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
            # model-FINAL_MODEL_resnet20_adam_lr0.001_wd0.0_bs128_cosine_seed1:v0
            # -> model-FINAL_MODEL_resnet20_adam_lr0.001_wd0.0_bs128_cosine_seed1
            collection, _version = parse_artifact_collection_and_version(artifact)

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
            "Expected prefixes are model-FINAL_MODEL_resnet20_ and model-FINAL_MODEL_vit_."
        )
        return

    print(
        f"Found {len(latest_by_collection)} typed model artifact collection(s) "
        f"for --model_type {args.model_type!r}."
    )

    artifact_suffix = args.analysis_artifact_suffix
    if artifact_suffix is None and args.artifact_alias != "latest":
        artifact_suffix = args.artifact_alias

    for run_name in sorted(latest_by_collection):
        artifact_ref = f"{entity}/{PROJECT}/{run_name}:{args.artifact_alias}"

        analysis_artifact_name = None
        if artifact_suffix:
            analysis_artifact_name = f"{run_name}-minimum-analysis-{artifact_suffix}"

        print("-" * 80)
        print(f"Analyzing model artifact: {artifact_ref}")
        print(f"Run name: {run_name}")
        if analysis_artifact_name:
            print(f"Analysis artifact name: {analysis_artifact_name}")

        if args.dry_run:
            continue

        analyzer.analyze(
            run_name=run_name,
            artifact_alias=args.artifact_alias,
            analysis_artifact_name=analysis_artifact_name,
            batch_size=args.batch_size,
            samples_per_radius=args.samples_per_radius,
            adaptive_sharpness_rhos=adaptive_sharpness_rhos,
            adaptive_sharpness_steps=args.adaptive_sharpness_steps,
            adaptive_sharpness_batches=args.adaptive_sharpness_batches,
            adaptive_sharpness_norm=args.adaptive_sharpness_norm,
            adaptive_sharpness_logit_normalize=args.adaptive_sharpness_logit_normalize,
            adaptive_sharpness_average_batches=args.adaptive_sharpness_average_batches,
            skip_adaptive_sharpness=args.skip_adaptive_sharpness,
        )


if __name__ == "__main__":
    main()
