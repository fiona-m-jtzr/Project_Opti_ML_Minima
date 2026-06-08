import copy
import math
import json
import sys
import importlib
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

from itertools import islice

try:
    from torch.func import functional_call as _torch_functional_call

    def _functional_call(model, params, buffers, args):
        return _torch_functional_call(model, (params, buffers), args)

except ImportError:
    from torch.nn.utils.stateless import functional_call as _torch_functional_call

    def _functional_call(model, params, buffers, args):
        state = {**params, **buffers}
        return _torch_functional_call(model, state, args)


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

def full_loss_and_accuracy(model, loader, criterion, device, max_batches=None):
    """Compute average loss and accuracy over an entire dataset."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()

        total_loss += loss.item() * y.size(0)

        total_samples += y.size(0)

    if total_samples == 0:
        raise ValueError("No samples were provided to full_loss_and_accuracy.")

    return total_loss / total_samples, total_correct / total_samples


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


def sample_scale_invariant_direction_like(
    model,
    include_bias=False,
    include_bn_affine=False,
):
    """
    Sample a filter-normalized direction in parameter space.

    Parameters
    ----------
    include_bias : bool
        Whether to perturb bias parameters.

    include_bn_affine : bool
        Whether to perturb BatchNorm affine parameters
        (weight/gamma and bias/beta).
    """
    direction = []

    for module_name, module in model.named_modules():

        is_bn = isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ),
        )


        for param_name, p in module.named_parameters(recurse=False):
            d = torch.zeros_like(p)
            # Check if trainable parameter
            if p.requires_grad:
                # Check if bias parameter
                if include_bias or param_name != "bias":
                    # Check if batch norm affine parameter
                    is_bn_affine = is_bn and param_name in {"weight", "bias"}
                    if include_bn_affine or not is_bn_affine:
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

    for p, d in zip(model.parameters(), direction):
        p.add_(relative_scale * d)


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
    max_batches=8,
):
    """
    Randomly sample scale-independent perturbations and report the largest loss increase.

    The direction is filter-normalized, so the radius is a relative parameter-space radius.

    """
    base_state = copy.deepcopy(model.state_dict())
    base_loss, _ = full_loss_and_accuracy(model, loader, criterion, device, max_batches=max_batches)

    max_loss = -math.inf
    max_delta = None
    sharpness_deltas = []

    for _ in range(samples):
        direction = sample_scale_invariant_direction_like(model)
        scale = relative_radius

        add_direction_to_model(model, base_state, direction, scale)

        loss, _ = full_loss_and_accuracy(model, loader, criterion, device, max_batches=max_batches)
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
# Element-wise adaptive sharpness via tml-epfl/sharpness-vs-generalization
# -----------------------------

_EPFL_SHARPNESS_MODULE = None


def _load_epfl_sharpness_module():
    """
    Load the original EPFL sharpness.py implementation.

    Expected layout next to this analyzer:
        sharpness_vs_generalization/sharpness.py
        sharpness_vs_generalization/utils.py

    These files should be copied verbatim from:
        https://github.com/tml-epfl/sharpness-vs-generalization
    """
    global _EPFL_SHARPNESS_MODULE
    if _EPFL_SHARPNESS_MODULE is not None:
        return _EPFL_SHARPNESS_MODULE

    repo_dir = Path(__file__).resolve().parent / "sharpness_vs_generalization"
    sharpness_py = repo_dir / "sharpness.py"
    utils_py = repo_dir / "utils.py"

    if not sharpness_py.exists() or not utils_py.exists():
        raise FileNotFoundError(
            "Missing EPFL sharpness implementation. Copy sharpness.py and utils.py from "
            "https://github.com/tml-epfl/sharpness-vs-generalization into "
            f"{repo_dir}"
        )

    sys.path.insert(0, str(repo_dir))
    _EPFL_SHARPNESS_MODULE = importlib.import_module("sharpness")
    return _EPFL_SHARPNESS_MODULE


def _extract_logits(output):
    """Handle models that return logits directly or inside a tuple/object."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class _EPFLLogitNormalizationWrapper(nn.Module):
    """Logit normalization used by the EPFL evaluation script."""

    def __init__(self, model, normalize_logits=True, eps=1e-12):
        super().__init__()
        self.model = model
        self.normalize_logits = normalize_logits
        self.eps = eps

    def forward(self, x):
        logits = _extract_logits(self.model(x))
        if not self.normalize_logits:
            return logits
        centered = logits - logits.mean(dim=-1, keepdim=True)
        denom = centered.norm(dim=-1, keepdim=True)
        denom = torch.max(denom, 1e-10 * torch.ones_like(denom))
        return centered / denom


class _EPFLBatchAdapter:
    """
    Convert a normal PyTorch loader yielding (x, y) into the 5-tuple format
    expected by tml-epfl/sharpness-vs-generalization/sharpness.py:
        (x, unused_x2, y, unused_y_correct, unused_label_noise)
    """

    def __init__(self, loader, max_batches=None):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for batch_idx, batch in enumerate(self.loader):
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            if len(batch) >= 5:
                yield batch
            else:
                x, y = batch[:2]
                yield x, None, y, None, None


def _single_batch_epfl_adapter(batch):
    """Return an iterable containing exactly one EPFL-format batch."""
    if len(batch) >= 5:
        return [batch]
    x, y = batch[:2]
    return [(x, None, y, None, None)]


@torch.no_grad()
def _epfl_base_loss_and_accuracy(model, batches, criterion):
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, _, y, _, _ in batches:
        x, y = x.cuda(), y.cuda()
        logits = _extract_logits(model(x))
        loss = criterion(logits, y)
        if loss.ndim > 0:
            loss = loss.mean()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_seen += y.size(0)

    if total_seen == 0:
        raise ValueError("No batches were provided for adaptive sharpness.")

    return total_loss / total_seen, total_correct / total_seen, total_seen


def _epfl_adaptive_sharpness_on_batches(
    model,
    batches,
    criterion,
    rho=2e-3,
    steps=20,
    step_size=None,
    norm="linf",
    logit_normalize=True,
    random_start=True,
    verbose=False,
):
    """Delegate worst-case adaptive sharpness to the original EPFL APGD code."""
    if norm not in {"linf", "l2"}:
        raise ValueError("norm must be either 'linf' or 'l2'.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The EPFL sharpness.py implementation calls .cuda() internally, "
            "so this wrapper requires a CUDA GPU."
        )

    epfl_sharpness = _load_epfl_sharpness_module()

    old_mode = model.training
    model.cuda().eval()
    wrapped_model = _EPFLLogitNormalizationWrapper(
        model,
        normalize_logits=logit_normalize,
    ).cuda().eval()

    # EPFL initializes Auto-PGD with step_size = 2 * rho * step_size_mult.
    step_size_mult = 1.0 if step_size is None else float(step_size) / (2.0 * float(rho))

    # Materialize once because the helper computes base loss and EPFL consumes the iterable.
    batches = list(batches)
    base_loss, base_acc, num_examples = _epfl_base_loss_and_accuracy(
        wrapped_model,
        batches,
        criterion,
    )

    sharpness_delta, sharpness_err_delta, delta_norm = epfl_sharpness.eval_APGD_sharpness(
        model=wrapped_model,
        batches=batches,
        loss_f=criterion,
        train_err=None,
        train_loss=None,
        rho=float(rho),
        step_size_mult=float(step_size_mult),
        n_iters=int(steps),
        n_restarts=1,
        rand_init=bool(random_start),
        no_grad_norm=False,
        verbose=bool(verbose),
        return_output=False,
        adaptive=True,
        version="default",
        norm=norm,
    )

    model.train(old_mode)
    model.zero_grad(set_to_none=True)

    return {
        "base_loss": float(base_loss),
        "perturbed_loss": float(base_loss + sharpness_delta),
        "sharpness_delta": float(sharpness_delta),
        "sharpness_err_delta": float(sharpness_err_delta),
        "delta_norm": float(delta_norm),
        "rho": float(rho),
        "steps": int(steps),
        "step_size": None if step_size is None else float(step_size),
        "step_size_mult": float(step_size_mult),
        "norm": norm,
        "logit_normalize": bool(logit_normalize),
        "adaptive_scale": "elementwise_abs_parameter",
        "optimization": "epfl_eval_APGD_sharpness",
        "random_start": bool(random_start),
        "num_batches": len(batches),
        "num_examples": int(num_examples),
    }


def elementwise_adaptive_sharpness_multi_batch(
    model,
    loader,
    criterion,
    device,
    rho=2e-3,
    steps=20,
    max_batches=8,
    step_size=None,
    norm="linf",
    logit_normalize=True,
    average_individual_batches=True,
    random_start=True,
):
    """Compute element-wise adaptive m-sharpness using the original EPFL APGD code."""
    del device  # EPFL code uses .cuda() internally.

    batches = list(_EPFLBatchAdapter(loader, max_batches=max_batches))
    if len(batches) == 0:
        raise ValueError("The loader yielded no batches.")

    if average_individual_batches:
        per_batch = [
            _epfl_adaptive_sharpness_on_batches(
                model=model,
                batches=_single_batch_epfl_adapter(batch),
                criterion=criterion,
                rho=rho,
                steps=steps,
                step_size=step_size,
                norm=norm,
                logit_normalize=logit_normalize,
                random_start=random_start,
            )
            for batch in batches
        ]

        mean_base_loss = sum(item["base_loss"] for item in per_batch) / len(per_batch)
        mean_perturbed_loss = sum(item["perturbed_loss"] for item in per_batch) / len(per_batch)
        mean_sharpness = sum(item["sharpness_delta"] for item in per_batch) / len(per_batch)
        max_sharpness = max(item["sharpness_delta"] for item in per_batch)

        return {
            "base_loss": float(mean_base_loss),
            "perturbed_loss": float(mean_perturbed_loss),
            "sharpness_delta": float(mean_sharpness),
            "max_batch_sharpness_delta": float(max_sharpness),
            "per_batch": per_batch,
            "rho": float(rho),
            "steps": int(steps),
            "step_size": None if step_size is None else float(step_size),
            "norm": norm,
            "logit_normalize": bool(logit_normalize),
            "adaptive_scale": "elementwise_abs_parameter",
            "aggregation": "mean_of_per_batch_epfl_apgd_sharpness",
            "random_start": bool(random_start),
            "num_batches": len(per_batch),
            "num_examples": int(sum(item["num_examples"] for item in per_batch)),
        }

    # Note: this is still EPFL's averaged m-sharpness over batches, not one shared
    # perturbation over the union. The upstream implementation does not expose the
    # union-batch variant used by the previous local code.
    return _epfl_adaptive_sharpness_on_batches(
        model=model,
        batches=batches,
        criterion=criterion,
        rho=rho,
        steps=steps,
        step_size=step_size,
        norm=norm,
        logit_normalize=logit_normalize,
        random_start=random_start,
    )


def elementwise_adaptive_sharpness_curve(
    model,
    loader,
    criterion,
    device,
    rhos=(1e-4, 5e-4, 1e-3, 2e-3, 4e-3),
    steps=20,
    max_batches=8,
    step_size=None,
    norm="linf",
    logit_normalize=True,
    average_individual_batches=True,
    random_start=True,
):
    """Compute an adaptive sharpness curve by delegating each rho to EPFL APGD."""
    curve = []
    for rho in rhos:
        curve.append(
            elementwise_adaptive_sharpness_multi_batch(
                model=model,
                loader=loader,
                criterion=criterion,
                device=device,
                rho=rho,
                steps=steps,
                max_batches=max_batches,
                step_size=step_size,
                norm=norm,
                logit_normalize=logit_normalize,
                average_individual_batches=average_individual_batches,
                random_start=random_start,
            )
        )
    return curve


# -----------------------------
# Hessian metrics
# -----------------------------

"""def get_one_hessian_batch(loader, device):
    x, y = next(iter(loader))
    return x.to(device), y.to(device)"""

def get_hessian_batch_tensor(loader, device, num_batches=8):
    xs, ys = [], []
    for x, y in islice(loader, num_batches):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0).to(device), torch.cat(ys, dim=0).to(device)


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

    inputs, targets = get_hessian_batch_tensor(trainloader, device)

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
    adaptive_sharpness_rhos=(1e-4, 3e-4, 1e-3, 2e-3, 3e-3),
    adaptive_sharpness_steps=20,
    adaptive_sharpness_batches=8,
    adaptive_sharpness_norm="linf",
    adaptive_sharpness_logit_normalize=True,
    adaptive_sharpness_average_batches=True,
    skip_adaptive_sharpness=False,
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
            "adaptive_sharpness_rhos": list(adaptive_sharpness_rhos),
            "adaptive_sharpness_steps": adaptive_sharpness_steps,
            "adaptive_sharpness_batches": adaptive_sharpness_batches,
            "adaptive_sharpness_norm": adaptive_sharpness_norm,
            "adaptive_sharpness_logit_normalize": adaptive_sharpness_logit_normalize,
            "adaptive_sharpness_average_batches": adaptive_sharpness_average_batches,
            "skip_adaptive_sharpness": skip_adaptive_sharpness,
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
    train_loss, train_accuracy = full_loss_and_accuracy(
        model, trainloader, criterion, device
    )

    test_loss, test_accuracy = full_loss_and_accuracy(
        model, testloader, criterion, device
    )
    print("Accuracies and Losses computed. Computing gradient norm...")
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

    if skip_adaptive_sharpness:
        adaptive_sharpness_by_radius = None
        print("Skipping element-wise adaptive sharpness curve.")
    else:
        print("Sampled sharpness curve computed. Computing element-wise adaptive sharpness curve...")
        adaptive_sharpness_by_radius = elementwise_adaptive_sharpness_curve(
            model=model,
            loader=trainloader,
            criterion=criterion,
            device=device,
            rhos=adaptive_sharpness_rhos,
            steps=adaptive_sharpness_steps,
            max_batches=adaptive_sharpness_batches,
            norm=adaptive_sharpness_norm,
            logit_normalize=adaptive_sharpness_logit_normalize,
            average_individual_batches=adaptive_sharpness_average_batches,
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
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_test_loss_gap": test_loss - train_loss,
        "train_test_accuracy_gap": train_accuracy - test_accuracy,
        "gradient_norm_full_train_dataset": grad_norm,
        "hessian_metrics": hessian_metrics,
        "scale_invariant_sharpness_by_radius": sharpness_by_radius,
        "elementwise_adaptive_sharpness_by_radius": adaptive_sharpness_by_radius,
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
    parser.add_argument(
        "--adaptive_sharpness_rhos",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3, 2e-3, 3e-3],
        help=(
            "One or more rho values for the element-wise adaptive sharpness curve. "
            "Example: --adaptive_sharpness_rhos 3e-4 1e-3 2e-3 3e-3"
        ),
    )
    parser.add_argument(
        "--adaptive_sharpness_rho",
        type=float,
        default=None,
        help=(
            "Deprecated single-rho option. If provided, the adaptive curve uses "
            "only this rho. Prefer --adaptive_sharpness_rhos for a sweep."
        ),
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


if __name__ == "__main__":
    args = parse_args()

    adaptive_sharpness_rhos = (
        [args.adaptive_sharpness_rho]
        if args.adaptive_sharpness_rho is not None
        else args.adaptive_sharpness_rhos
    )

    analyze(
        run_name=args.run_name,
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