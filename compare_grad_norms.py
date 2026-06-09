"""
Compare training-style and analyzer-style gradient norms for a saved W&B model artifact.

Example:
python compare_grad_norms.py \
  --artifact FINAL_MODEL_resnet20_sgd_mom0.9_lr0.1_wd0.0_bs128_cosine_seed1 \
  --checkpoint_file min_grad.pt \
  --split train45k

You can also pass:
  --artifact model-FINAL_MODEL_resnet20_sgd_mom0.9_lr0.1_wd0.0_bs128_cosine_seed1:latest
"""

import argparse
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import wandb

from models.resnet20 import ResNet20
from models.vit import ViTCIFAR10


# ---------------------------------------------------------------------------
# Artifact / model helpers
# ---------------------------------------------------------------------------

def normalize_artifact_name(name: str) -> str:
    """
    Accepts:
      FINAL_MODEL_resnet20_sgd_...
      model-FINAL_MODEL_resnet20_sgd_...
      model-FINAL_MODEL_resnet20_sgd_...:latest
      model-FINAL_MODEL_resnet20_sgd_...:v12
    """
    if not name.startswith("model-"):
        name = "model-" + name

    if ":" not in name:
        name = name + ":latest"

    return name


def extract_model_name(artifact_name: str) -> str:
    """
    Works with:
      model-FINAL_MODEL_resnet20_sgd_...:latest
      model-FINAL_MODEL_vit_adam_...:v12
    """
    name = artifact_name.split("/")[-1]
    name = name.split(":", 1)[0]

    if name.startswith("model-"):
        name = name[len("model-"):]

    match = re.search(r"FINAL_MODEL_(resnet20|vit)(?:_|$)", name)
    if match is None:
        raise ValueError(f"Could not infer model architecture from artifact name: {artifact_name}")

    return match.group(1)


def build_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "resnet20":
        return ResNet20(num_classes=10)

    if model_name == "vit":
        return ViTCIFAR10(
            img_size=32,
            patch_size=4,
            num_classes=10,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
        )

    raise ValueError(f"Unknown model: {model_name}")


def load_checkpoint_from_artifact(artifact_name: str, checkpoint_file: str, device):
    run = wandb.init(
        project="OptiML_Minima",
        job_type="compare-grad-norms",
    )

    artifact_ref = normalize_artifact_name(artifact_name)
    artifact = run.use_artifact(artifact_ref, type="model")
    artifact_dir = Path(artifact.download())

    ckpt_path = artifact_dir / checkpoint_file
    if not ckpt_path.exists():
        available = sorted(p.name for p in artifact_dir.iterdir())
        raise FileNotFoundError(
            f"{checkpoint_file} not found in artifact {artifact_ref}. "
            f"Available files: {available}"
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt, run, artifact_ref


def get_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_cifar10_loader(data_dir, batch_size, num_workers, split):
    """
    split:
      full50k  = analyzer-style full CIFAR-10 train set
      train45k = training-style 90% split from RUN.py, without augmentation
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    if split == "full50k":
        dataset = full_train
    elif split == "train45k":
        val_size = int(len(full_train) * 0.1)
        train_size = len(full_train) - val_size
        dataset, _ = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Gradient norm computations
# ---------------------------------------------------------------------------

def l2_norm_of_current_grads(model):
    device = next(model.parameters()).device
    sq = torch.zeros((), device=device)

    for p in model.parameters():
        if p.grad is not None:
            sq = sq + p.grad.detach().pow(2).sum()

    return sq.sqrt().item()


def training_style_mean_batch_grad_norm(model, loader, criterion, device, mode="train"):
    """
    Closest fixed-checkpoint version of the training script's metric:

      average over batches of ||grad batch mean loss||_2

    Important difference from the actual training loop:
      this does not call optimizer.step(), so all batches are evaluated
      at the same checkpoint.
    """
    old_training = model.training

    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    grad_norm_sum = 0.0
    num_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        grad_norm_sum += l2_norm_of_current_grads(model)
        num_batches += 1

    model.zero_grad(set_to_none=True)
    model.train(old_training)

    if num_batches == 0:
        raise ValueError("Loader produced zero batches.")

    return grad_norm_sum / num_batches


def full_dataset_mean_grad_norm(model, loader, criterion, device, mode="eval"):
    """
    Analyzer-style metric:

      || grad of mean loss over all examples ||_2

    This accumulates summed gradients over batches, then divides by
    the total number of examples.
    """
    old_training = model.training

    if mode == "train":
        model.train()
    elif mode == "eval":
        model.eval()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.zero_grad(set_to_none=True)
    total_seen = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        # CrossEntropyLoss default reduction is mean over the batch.
        # Multiply by batch size to accumulate summed loss gradients.
        (loss * y.size(0)).backward()
        total_seen += y.size(0)

    if total_seen == 0:
        raise ValueError("Loader produced zero examples.")

    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(total_seen)

    out = l2_norm_of_current_grads(model)

    model.zero_grad(set_to_none=True)
    model.train(old_training)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        required=True,
        help=(
            "W&B artifact name without entity. Examples: "
            "FINAL_MODEL_resnet20_sgd_... or "
            "model-FINAL_MODEL_resnet20_sgd_...:latest"
        ),
    )
    parser.add_argument(
        "--checkpoint_file",
        default="min_grad.pt",
        choices=["min_grad.pt", "best.pt"],
    )
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--split",
        default="full50k",
        choices=["full50k", "train45k"],
        help="full50k matches analyzer.py; train45k matches RUN.py's 90% train split.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=["resnet20", "vit"],
        help="Optional override if model architecture cannot be inferred from artifact name.",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt, wandb_run, artifact_ref = load_checkpoint_from_artifact(
        artifact_name=args.artifact,
        checkpoint_file=args.checkpoint_file,
        device=device,
    )

    model_name = args.model or extract_model_name(artifact_ref)
    model = build_model(model_name).to(device)
    model.load_state_dict(get_state_dict(ckpt))

    loader = get_cifar10_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
    )

    criterion = nn.CrossEntropyLoss()

    print("=" * 80)
    print(f"Artifact       : {artifact_ref}")
    print(f"Checkpoint     : {args.checkpoint_file}")
    print(f"Model          : {model_name}")
    print(f"Device         : {device}")
    print(f"Dataset split  : {args.split}")
    print(f"Batch size     : {args.batch_size}")
    print("=" * 80)

    if isinstance(ckpt, dict):
        for key in ["epoch", "grad_norm", "best_grad", "val_acc", "best_acc", "best_loss"]:
            if key in ckpt:
                print(f"checkpoint[{key!r}] = {ckpt[key]}")

    print("-" * 80)

    training_style_train_mode = training_style_mean_batch_grad_norm(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        mode="train",
    )

    training_style_eval_mode = training_style_mean_batch_grad_norm(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        mode="eval",
    )

    full_grad_train_mode = full_dataset_mean_grad_norm(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        mode="train",
    )

    full_grad_eval_mode = full_dataset_mean_grad_norm(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        mode="eval",
    )

    print(f"training_style_mean_batch_grad_norm / train mode : {training_style_train_mode:.10g}")
    print(f"training_style_mean_batch_grad_norm / eval mode  : {training_style_eval_mode:.10g}")
    print(f"full_dataset_mean_grad_norm        / train mode : {full_grad_train_mode:.10g}")
    print(f"full_dataset_mean_grad_norm        / eval mode  : {full_grad_eval_mode:.10g}")

    print("-" * 80)
    print("Main comparison:")
    print(f"  training-script-like value : {training_style_train_mode:.10g}")
    print(f"  analyzer-like value        : {full_grad_eval_mode:.10g}")
    print(
        f"  ratio analyzer/training    : "
        f"{full_grad_eval_mode / max(training_style_train_mode, 1e-30):.10g}"
    )

    wandb_run.finish()


if __name__ == "__main__":
    main()