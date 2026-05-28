import copy
import math
import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.resnet20 import ResNet20
import os
import tempfile
import wandb
from pathlib import Path
import argparse

from hessian.hessian import hessian


# -----------------------------
# Model and data utilities
# -----------------------------


def get_loaders(batch_size=128):
    """Return deterministic train/test loaders for analysis."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# -----------------------------
# Basic loss / gradient metrics
# -----------------------------

@torch.no_grad()
def full_loss(model, loader, criterion, device):
    """Compute average loss over an entire dataset."""
    model.eval()
    total_loss, total = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

    return total_loss / total


def gradient_norm(model, loader, criterion, device, max_batches=None):
    """
    Compute gradient norm over one or more batches.

    A small gradient norm indicates that the model is closer to a stationary point.
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    total_seen = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        # Scale by batch size so accumulated gradient approximates dataset gradient.
        loss = loss * y.size(0)
        loss.backward()

        total_seen += y.size(0)

    # Convert accumulated sum gradient into mean gradient.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(total_seen)

    grad_sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_sq_sum += p.grad.detach().pow(2).sum()

    return torch.sqrt(grad_sq_sum).item()


def parameter_norm(model):
    """Compute L2 norm of all trainable parameters."""
    return torch.sqrt(
        sum((p.detach() ** 2).sum() for p in model.parameters() if p.requires_grad)
    ).item()


# -----------------------------
# Scale-independent perturbation utilities
# -----------------------------

@torch.no_grad()
def _normalize_like_parameter(direction_tensor, parameter_tensor, eps=1e-12):
    """
    Normalize a random direction relative to the scale of the corresponding parameter.

    For convolutional and linear weights, each output filter / row gets its own norm.
    For 1D tensors such as biases and BatchNorm parameters, the whole tensor is normalized.

    This makes the perturbation radius dimensionless: radius=0.01 means roughly a 1%
    relative perturbation per filter/parameter group rather than an absolute movement in
    raw parameter coordinates.
    """
    if parameter_tensor.ndim > 1:
        # Treat dim 0 as the filter/output-unit dimension.
        d_flat = direction_tensor.view(direction_tensor.size(0), -1)
        p_flat = parameter_tensor.detach().view(parameter_tensor.size(0), -1)

        d_norm = d_flat.norm(dim=1, keepdim=True)
        p_norm = p_flat.norm(dim=1, keepdim=True)

        d_flat = (d_flat / (d_norm + eps)) * (p_norm + eps)
        return d_flat.view_as(direction_tensor)

    d_norm = direction_tensor.norm()
    p_norm = parameter_tensor.detach().norm()
    return (direction_tensor / (d_norm + eps)) * (p_norm + eps)


def sample_scale_invariant_direction_like(model):
    """
    Sample a filter-normalized direction in parameter space.

    This is preferable to a single global unit-norm direction because it is much less
    sensitive to arbitrary rescaling of layers or filters.
    """
    direction = []

    for p in model.parameters():
        if p.requires_grad:
            d = torch.randn_like(p)
            d = _normalize_like_parameter(d, p)
            direction.append(d)

    return direction


@torch.no_grad()
def add_direction_to_model(model, base_state, direction, relative_scale):
    """
    Reset model to base_state and add relative_scale * direction.

    Because direction is filter-normalized, relative_scale is dimensionless.
    """
    model.load_state_dict(base_state)

    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            p.add_(relative_scale * direction[idx])
            idx += 1


# -----------------------------
# Sharpness via scale-independent sampled neighbourhood
# -----------------------------

def max_loss_in_neighbourhood(
    model,
    loader,
    criterion,
    device,
    relative_radius=1e-2,
    samples=20,
):
    """
    Randomly sample scale-independent perturbations and report the largest loss increase.

    The direction is filter-normalized, so the radius is a relative parameter-space radius.

    """
    base_state = copy.deepcopy(model.state_dict())
    base_loss = full_loss(model, loader, criterion, device)

    max_loss = -math.inf
    max_delta = None
    sharpness_deltas = []

    for _ in range(samples):
        direction = sample_scale_invariant_direction_like(model)
        scale = relative_radius

        add_direction_to_model(model, base_state, direction, scale)

        loss = full_loss(model, loader, criterion, device)
        delta = loss - base_loss
        sharpness_deltas.append(delta)

        if loss > max_loss:
            max_loss = loss
            max_delta = delta

    model.load_state_dict(base_state)

    return {
        "base_loss": base_loss,
        "max_neighbourhood_loss": max_loss,
        "sharpness_delta": max_delta,
        "mean_sharpness_delta": float(sum(sharpness_deltas) / len(sharpness_deltas)),
        "relative_radius": relative_radius,
        "samples": samples,
        "scale_normalization": "filter_normalized",
    }


def sharpness_curve(
    model,
    loader,
    criterion,
    device,
    relative_radii=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    samples_per_radius=20,
):
    """Compute scale-independent sampled sharpness for several relative radii."""
    return [
        max_loss_in_neighbourhood(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            relative_radius=r,
            samples=samples_per_radius,
        )
        for r in relative_radii
    ]


# -----------------------------
# Hessian metrics
# -----------------------------

def get_one_hessian_batch(loader, device):
    """PyHessian usually computes Hessian metrics from a representative batch."""
    x, y = next(iter(loader))
    return x.to(device), y.to(device)


def compute_hessian_metrics(
    model,
    trainloader,
    criterion,
    device,
    top_n=5,
    trace_samples=50,
    density_iter=100,
    density_samples=1,
):
    """
    Compute Hessian eigenvalues, trace, spectral density, and normalized Hessian stats.

    Raw Hessian eigenvalues are still coordinate-scale dependent. The normalized quantities
    multiply by ||w||^2 and are the safer values to compare across differently scaled models.
    """
    model.eval()

    inputs, targets = get_one_hessian_batch(trainloader, device)

    model.zero_grad(set_to_none=True)

    hessian_comp = hessian(
        model,
        criterion,
        data=(inputs, targets),
        cuda=(device == "cuda"),
    )

    eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_n)
    model.zero_grad(set_to_none=True)

    trace_estimates = hessian_comp.trace(maxIter=trace_samples)
    trace_mean = float(sum(trace_estimates) / len(trace_estimates))
    model.zero_grad(set_to_none=True)

    density_eigen, density_weight = hessian_comp.density(
        iter=density_iter,
        n_v=density_samples,
    )
    model.zero_grad(set_to_none=True)

    flat_eigs = []
    flat_weights = []

    for eigs, weights in zip(density_eigen, density_weight):
        flat_eigs.extend(eigs)
        flat_weights.extend(weights)

    total_weight = sum(flat_weights) + 1e-12
    negative_weight = sum(w for e, w in zip(flat_eigs, flat_weights) if e < 0.0)
    negative_curvature_ratio = negative_weight / total_weight

    weight_norm = parameter_norm(model)
    top_eig = float(eigenvalues[0])

    return {
        # Raw values are included for diagnostics, not for scale-independent comparison.
        "raw_top_eigenvalues": [float(v) for v in eigenvalues],
        "raw_trace_estimates": [float(v) for v in trace_estimates],
        "raw_trace_mean": trace_mean,

        # Prefer these for cross-model comparison.
        "weight_norm": weight_norm,
        "normalized_top_eigenvalue": top_eig * (weight_norm ** 2),
        "normalized_trace": trace_mean * (weight_norm ** 2),

        "density_eigen": density_eigen,
        "density_weight": density_weight,
        "negative_curvature_ratio": float(negative_curvature_ratio),
    }


def _json_safe(obj):
    """Convert nested objects into JSON-safe Python types, dropping imaginary parts."""

    import numpy as np
    import torch

    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()

        # Handle complex tensors
        if torch.is_complex(obj):
            obj = obj.real

        return obj.tolist()

    # Python complex numbers
    if isinstance(obj, complex):
        return float(obj.real)

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    # NumPy arrays
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            obj = np.real(obj)
        return obj.tolist()

    # NumPy scalar types
    if isinstance(obj, np.generic):
        if np.iscomplexobj(obj):
            return float(np.real(obj))
        return obj.item()

    return obj

## Wandb Model Loading and Result Logging

def load_wandb_checkpoint(run, artifact_ref, filename="best.pt", device="cpu"):
    artifact = run.use_artifact(artifact_ref, type="model")
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, filename)

    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def log_results_artifact(run, results, artifact_name="minimum-analysis-results"):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "minimum_analysis_scale_invariant.json")

        with open(out_path, "w") as f:
            json.dump(_json_safe(results), f, indent=2)

        artifact = wandb.Artifact(
            name=artifact_name,
            type="analysis",
            metadata={
                "analysis": "minimum_analysis_scale_invariant",
            },
        )
        artifact.add_file(out_path, name="minimum_analysis_scale_invariant.json")
        run.log_artifact(artifact)


## Reproducibility
def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -----------------------------
# Main analysis
# -----------------------------
def analyze(
    run_name,
    batch_size=128,
    relative_radii=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    samples_per_radius=20,
):
    print(f"Analyzing Launched for: {run_name}")
    set_seed(42)
    # Configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("-" * 50)
    # Configure wandb
    print("Initializing Weights & Biases run...")
    wandb_run = wandb.init(
        project="OptiML_Minima",
        name=f"{run_name}_minimum_analysis",
        job_type="minimum-analysis",
        config={
            "source_run_name": run_name,
            "batch_size": batch_size,
            "relative_radii": list(relative_radii),
            "samples_per_radius": samples_per_radius,
        },
    )

    artifact = wandb_run.use_artifact(
        f"{wandb_run.entity}/OptiML_Minima/{run_name}:latest",
        type="model",
    )
    artifact_dir = artifact.download()
    ckpt_path = Path(artifact_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    checkpoint_metadata = {}

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        for key in ["epoch", "best_acc", "args", "history"]:
            if key in ckpt:
                checkpoint_metadata[key] = _json_safe(ckpt[key])
    else:
        state_dict = ckpt
    print("-" * 50)
    print("Loading model and data...")
    # Load model and data
    trainloader, testloader = get_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()

    model = ResNet20(num_classes=10).to(device)
    model.load_state_dict(state_dict)
    # Compute metrics
    print("-" * 50)
    print("Starting analysis...")
    train_loss = full_loss(model, trainloader, criterion, device)
    test_loss = full_loss(model, testloader, criterion, device)
    print("Losses computed. Computing gradient norm...")
    grad_norm = gradient_norm(
        model=model,
        loader=trainloader,
        criterion=criterion,
        device=device,
        max_batches=None,
    )
    print("Gradient norm computed. Computing Hessian metrics...")
    hessian_metrics = compute_hessian_metrics(
        model=model,
        trainloader=trainloader,
        criterion=criterion,
        device=device,
    )
    print("Hessian metrics computed. Computing sampled sharpness curve...")
    sharpness_by_radius = sharpness_curve(
        model=model,
        loader=trainloader,
        criterion=criterion,
        device=device,
        relative_radii=relative_radii,
        samples_per_radius=samples_per_radius,
    )
    print("-" * 50)
    print("Analysis completed. Saving results...")
    results = {
        # Metadata
        "source_run_name": run_name,
        "source_artifact": f"{run_name}:latest",
        "checkpoint_metadata": checkpoint_metadata,
        # Results
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_test_gap": test_loss - train_loss,
        "gradient_norm_full_train_dataset": grad_norm,
        "hessian_metrics": hessian_metrics,
        "scale_invariant_sharpness_by_radius": sharpness_by_radius,
    }

    results_path = Path("minimum_analysis.json")
    with open(results_path, "w") as f:
        json.dump(_json_safe(results), f, indent=2)

    result_artifact = wandb.Artifact(
        f"{run_name}-minimum-analysis",
        type="analysis",
    )
    result_artifact.add_file(results_path)
    wandb_run.log_artifact(result_artifact)

    wandb_run.finish()
    print(f"Results saved to {results_path} and logged to Weights & Biases artifact {result_artifact.name}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--samples_per_radius", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    analyze(
        run_name=args.run_name,
        batch_size=args.batch_size,
        samples_per_radius=args.samples_per_radius,
    )