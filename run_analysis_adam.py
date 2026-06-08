import argparse
import re
from collections import defaultdict

import wandb
import analyzer
from run_analysis import (
    PROJECT,
    artifact_matches_model_type,
    artifact_model_type,
    artifact_optimizer,
    parse_artifact_collection_and_version,
)


def _version_sort_key(artifact):
    """
    Sort W&B artifact versions newest-first.

    created_at is the primary ordering signal. The numeric vN suffix is a fallback and
    tie-breaker because W&B artifact versions are monotonic within a collection.
    """
    _collection, version = parse_artifact_collection_and_version(artifact)
    version_num = -1
    if isinstance(version, str) and version.startswith("v"):
        try:
            version_num = int(version[1:])
        except ValueError:
            version_num = -1
    return (artifact.created_at, version_num)


def _dedupe_artifacts_by_collection_version(artifacts):
    by_key = {}
    for artifact in artifacts:
        collection, version = parse_artifact_collection_and_version(artifact)
        key = (collection, version)
        if key not in by_key or _version_sort_key(artifact) > _version_sort_key(by_key[key]):
            by_key[key] = artifact
    return list(by_key.values())


def parse_requested_versions(raw_values):
    """
    Parse artifact versions from CLI input.

    Supported forms:
      --artifact_versions "[v1,v2]"
      --artifact_versions "v1,v2"
      --artifact_versions v1 v2
      --artifact_versions "[v1, v2]"

    Returns None when the argument was not provided. Otherwise returns a de-duplicated
    list preserving the user-provided order.
    """
    if raw_values is None:
        return None

    if len(raw_values) == 0:
        raise ValueError(
            "--artifact_versions was provided but no versions were given. "
            "Use something like --artifact_versions '[v1,v2]' or --artifact_versions v1 v2."
        )

    # Join all values so both single-token list syntax and space-separated syntax work.
    joined = " ".join(str(value) for value in raw_values).strip()
    if not joined:
        raise ValueError(
            "--artifact_versions was provided but no versions were given. "
            "Use something like --artifact_versions '[v1,v2]' or --artifact_versions v1 v2."
        )

    # Accept bracketed list-ish syntax without requiring strict JSON/Python syntax.
    if joined.startswith("[") and joined.endswith("]"):
        joined = joined[1:-1]

    tokens = re.findall(r"[A-Za-z0-9_.-]+", joined)
    versions = []
    seen = set()
    for token in tokens:
        if token not in seen:
            versions.append(token)
            seen.add(token)

    if not versions:
        raise ValueError(
            "Could not parse any artifact versions from --artifact_versions. "
            "Use something like --artifact_versions '[v1,v2]' or --artifact_versions v1 v2."
        )

    return versions


def find_adam_model_artifact_versions(
    entity,
    project,
    model_type="all",
    num_versions=3,
    requested_versions=None,
):
    """
    Return {collection_name: [artifact_vN, ...]} for Adam model artifact versions.

    If requested_versions is None, the latest num_versions versions are returned per
    Adam artifact collection.

    If requested_versions is provided, exactly those versions are returned per Adam
    artifact collection when available, preserving the requested version order.

    The Adam filter is intentionally strict: it only matches typed artifact collections
    where the primary optimizer token after the architecture prefix is exactly "adam".
    It does not match muon artifacts that contain adamLR, nor SAM artifacts with baseadam.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    artifacts_by_collection = defaultdict(list)
    ignored_untyped = set()

    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type != "model":
                continue

            collection, version = parse_artifact_collection_and_version(artifact)

            if not collection.startswith("model-"):
                continue

            if artifact_model_type(collection) is None:
                ignored_untyped.add(collection)
                continue

            if not artifact_matches_model_type(collection, model_type):
                continue

            if artifact_optimizer(collection) != "adam":
                continue

            if version is None:
                continue

            artifacts_by_collection[collection].append(artifact)

    selected_by_collection = {}
    missing_versions_by_collection = {}

    for collection, artifacts in artifacts_by_collection.items():
        deduped = _dedupe_artifacts_by_collection_version(artifacts)

        if requested_versions is None:
            selected_by_collection[collection] = sorted(
                deduped,
                key=_version_sort_key,
                reverse=True,
            )[:num_versions]
            continue

        artifact_by_version = {}
        for artifact in deduped:
            _collection, version = parse_artifact_collection_and_version(artifact)
            if version not in artifact_by_version or _version_sort_key(artifact) > _version_sort_key(
                artifact_by_version[version]
            ):
                artifact_by_version[version] = artifact

        selected = []
        missing = []
        for version in requested_versions:
            artifact = artifact_by_version.get(version)
            if artifact is None:
                missing.append(version)
            else:
                selected.append(artifact)

        if selected:
            selected_by_collection[collection] = selected
        if missing:
            missing_versions_by_collection[collection] = missing

    return dict(sorted(selected_by_collection.items())), ignored_untyped, missing_versions_by_collection


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run minimum analysis on selected W&B artifact versions for Adam model artifacts only."
        )
    )
    parser.add_argument(
        "--model_type",
        choices=["all", "resnet20", "vit"],
        default="all",
        help="Which typed architecture family to analyse.",
    )
    parser.add_argument(
        "--num_versions",
        type=int,
        default=3,
        help=(
            "Number of most recent Adam artifact versions to analyse per collection. "
            "Ignored when --artifact_versions is provided."
        ),
    )
    parser.add_argument(
        "--artifact_versions",
        nargs="*",
        default=None,
        help=(
            "Explicit W&B model artifact versions to analyse per Adam collection. "
            "Examples: --artifact_versions '[v1,v2]', --artifact_versions 'v1,v2', "
            "or --artifact_versions v1 v2. When provided, --num_versions is ignored."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the Adam artifact versions without running analyzer.analyze(...).",
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

    requested_versions = parse_requested_versions(args.artifact_versions)

    if requested_versions is None and args.num_versions < 1:
        raise ValueError("--num_versions must be at least 1.")

    adaptive_sharpness_rhos = (
        [args.adaptive_sharpness_rho]
        if args.adaptive_sharpness_rho is not None
        else args.adaptive_sharpness_rhos
    )

    api = wandb.Api()
    entity = api.default_entity

    versions_by_collection, ignored_untyped, missing_versions_by_collection = find_adam_model_artifact_versions(
        entity=entity,
        project=PROJECT,
        model_type=args.model_type,
        num_versions=args.num_versions,
        requested_versions=requested_versions,
    )

    if args.show_ignored and ignored_untyped:
        print("Ignored untyped model artifact collection(s):")
        for name in sorted(ignored_untyped):
            print(f"  - {name}")
        print()

    if requested_versions is not None:
        print(f"Requested Adam artifact version(s): {', '.join(requested_versions)}")
        if missing_versions_by_collection:
            print("Missing requested version(s) for some Adam artifact collection(s):")
            for collection, missing_versions in sorted(missing_versions_by_collection.items()):
                print(f"  - {collection}: {', '.join(missing_versions)}")
            print()

    if not versions_by_collection:
        if requested_versions is None:
            print(
                f"No typed Adam model artifact versions matched --model_type {args.model_type!r}. "
                "Expected collection names like model-FINAL_MODEL_resnet20_adam_*."
            )
        else:
            print(
                f"No typed Adam model artifact versions matched --model_type {args.model_type!r} "
                f"and requested version(s) {requested_versions!r}. "
                "Expected collection names like model-FINAL_MODEL_resnet20_adam_*."
            )
        return

    total_versions = sum(len(versions) for versions in versions_by_collection.values())
    if requested_versions is None:
        print(
            f"Found {total_versions} Adam artifact version(s) across "
            f"{len(versions_by_collection)} collection(s), using the latest {args.num_versions} version(s)."
        )
    else:
        print(
            f"Found {total_versions} requested Adam artifact version(s) across "
            f"{len(versions_by_collection)} collection(s)."
        )

    for run_name, artifacts in versions_by_collection.items():
        for artifact in artifacts:
            _collection, artifact_version = parse_artifact_collection_and_version(artifact)
            artifact_ref = f"{entity}/{PROJECT}/{run_name}:{artifact_version}"
            analysis_artifact_name = f"{run_name}-minimum-analysis-{artifact_version}"

            print("-" * 80)
            print(f"Analyzing Adam model artifact: {artifact_ref}")
            print(f"Analysis artifact name: {analysis_artifact_name}")

            if args.dry_run:
                continue

            analyzer.analyze(
                run_name=run_name,
                artifact_alias=artifact_version,
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
