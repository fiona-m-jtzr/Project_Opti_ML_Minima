import wandb
import analyzer

PROJECT = "OptiML_Minima"

api = wandb.Api()
ENTITY = api.default_entity

runs = api.runs(f"{ENTITY}/{PROJECT}")

latest_by_collection = {}

for run in runs:
    for artifact in run.logged_artifacts():
        if artifact.type != "model":
            continue

        # Example: model-sgd_lr0.1_seed42:v0 -> model-sgd_lr0.1_seed42
        collection = artifact.name.split(":")[0]

        if (
            collection not in latest_by_collection
            or artifact.created_at > latest_by_collection[collection].created_at
        ):
            latest_by_collection[collection] = artifact


for run_name, artifact in latest_by_collection.items():
    if not run_name.startswith("model-"):
        continue

    artifact_ref = f"{ENTITY}/{PROJECT}/{run_name}:latest"

    print(f"Analyzing latest model artifact: {artifact_ref}")
    print(f"Run name: {run_name}")

    analyzer.analyze(run_name)