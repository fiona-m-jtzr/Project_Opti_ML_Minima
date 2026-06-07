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
def full_loss_and_accuracy(model, loader, criterion, device):
    """Compute average loss and accuracy over an entire dataset."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()

        total_samples += y.size(0)

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
):
    """
    Randomly sample scale-independent perturbations and report the largest loss increase.

    The direction is filter-normalized, so the radius is a relative parameter-space radius.

    """
    base_state = copy.deepcopy(model.state_dict())
    base_loss, _ = full_loss_and_accuracy(model, loader, criterion, device)

    max_loss = -math.inf
    max_delta = None
    sharpness_deltas = []

    for _ in range(samples):
        direction = sample_scale_invariant_direction_like(model)
        scale = relative_radius

        add_direction_to_model(model, base_state, direction, scale)

        loss, _ = full_loss_and_accuracy(model, loader, criterion, device)
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
# Element-wise adaptive sharpness
# -----------------------------


def _extract_logits(output):
    """Handle models that return logits directly or inside a tuple/object."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _logit_normalize(logits, eps=1e-12):
    """Normalize logits as in scale-insensitive classification sharpness."""
    centered = logits - logits.mean(dim=-1, keepdim=True)
    denom = centered.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    return centered / denom


def _criterion_loss(criterion, logits, targets):
    """Return a scalar batch loss, assuming CrossEntropyLoss-style mean reduction."""
    loss = criterion(logits, targets)
    if loss.ndim > 0:
        loss = loss.mean()
    return loss


def _loss_with_params_on_batches(
    model,
    params,
    buffers,
    batches,
    criterion,
    logit_normalize=False,
):
    """Mean loss over a fixed list of already-device-moved batches."""
    total_loss = None
    total_seen = 0

    for x, y in batches:
        logits = _extract_logits(_functional_call(model, params, buffers, (x,)))
        if logit_normalize:
            logits = _logit_normalize(logits)

        batch_loss = _criterion_loss(criterion, logits, y)
        batch_size = y.size(0)
        weighted_loss = batch_loss * batch_size
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        total_seen += batch_size

    if total_seen == 0:
        raise ValueError("No batches were provided for adaptive sharpness.")

    return total_loss / total_seen


def _global_l2_norm(tensors, eps=1e-12):
    total = None
    for tensor in tensors:
        val = tensor.detach().pow(2).sum()
        total = val if total is None else total + val
    return total.sqrt().clamp_min(eps)


def _collect_batches(loader, device, max_batches):
    """Materialize a deterministic prefix of the loader for repeated PGD steps."""
    batches = []

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batches.append((x.to(device), y.to(device)))

    if len(batches) == 0:
        raise ValueError("The loader yielded no batches.")

    return batches


def _elementwise_adaptive_worst_sharpness_on_fixed_batches(
    model,
    batches,
    criterion,
    rho=2e-3,
    steps=20,
    step_size=None,
    norm="linf",
    logit_normalize=True,
    random_start=False,
):
    """
    Estimate worst-case element-wise adaptive sharpness on fixed batches.

    This computes

        max_{||delta / |w| ||_p <= rho} L_S(w + delta) - L_S(w),

    using the reparameterization delta = |w| * z. For norm="linf", z is
    projected into [-rho, rho] element-wise; for norm="l2", z is projected
    into the global L2 ball of radius rho.
    """
    if norm not in {"linf", "l2"}:
        raise ValueError("norm must be either 'linf' or 'l2'.")

    if steps < 0:
        raise ValueError("steps must be non-negative.")

    if step_size is None:
        step_size = 2.0 * rho / max(steps, 1)

    old_mode = model.training
    model.eval()

    try:
        base_params = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
        }
        buffers = {
            name: b.detach().clone()
            for name, b in model.named_buffers()
        }
        trainable_names = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

        if not trainable_names:
            raise ValueError("Model has no trainable parameters.")

        scales = {name: base_params[name].abs() for name in trainable_names}

        with torch.no_grad():
            base_loss = _loss_with_params_on_batches(
                model=model,
                params=base_params,
                buffers=buffers,
                batches=batches,
                criterion=criterion,
                logit_normalize=logit_normalize,
            )

        z = {}
        for name in trainable_names:
            if random_start:
                if norm == "linf":
                    z[name] = torch.empty_like(base_params[name]).uniform_(-rho, rho)
                else:
                    z[name] = torch.randn_like(base_params[name])
            else:
                z[name] = torch.zeros_like(base_params[name])

        if random_start and norm == "l2":
            z_norm = _global_l2_norm(z.values())
            scale = torch.clamp(
                torch.as_tensor(rho, device=z_norm.device) / z_norm,
                max=1.0,
            )
            for name in trainable_names:
                z[name].mul_(scale)

        for name in trainable_names:
            z[name].requires_grad_(True)

        def make_perturbed_params():
            perturbed = dict(base_params)
            for name in trainable_names:
                perturbed[name] = base_params[name] + scales[name] * z[name]
            return perturbed

        for _ in range(steps):
            loss = _loss_with_params_on_batches(
                model=model,
                params=make_perturbed_params(),
                buffers=buffers,
                batches=batches,
                criterion=criterion,
                logit_normalize=logit_normalize,
            )

            grads = torch.autograd.grad(
                loss,
                [z[name] for name in trainable_names],
                allow_unused=True,
            )
            grads = [
                torch.zeros_like(z[name]) if grad is None else grad
                for name, grad in zip(trainable_names, grads)
            ]

            with torch.no_grad():
                if norm == "linf":
                    for name, grad in zip(trainable_names, grads):
                        z[name].add_(step_size * grad.sign())
                        z[name].clamp_(-rho, rho)
                else:
                    grad_norm = _global_l2_norm(grads)
                    for name, grad in zip(trainable_names, grads):
                        z[name].add_(step_size * grad / grad_norm)

                    z_norm = _global_l2_norm([z[name] for name in trainable_names])
                    scale = torch.clamp(
                        torch.as_tensor(rho, device=z_norm.device) / z_norm,
                        max=1.0,
                    )
                    for name in trainable_names:
                        z[name].mul_(scale)

        with torch.no_grad():
            perturbed_loss = _loss_with_params_on_batches(
                model=model,
                params=make_perturbed_params(),
                buffers=buffers,
                batches=batches,
                criterion=criterion,
                logit_normalize=logit_normalize,
            )

        sharpness = perturbed_loss - base_loss

        return {
            "base_loss": float(base_loss.detach().cpu()),
            "perturbed_loss": float(perturbed_loss.detach().cpu()),
            "sharpness_delta": float(sharpness.detach().cpu()),
            "rho": float(rho),
            "steps": int(steps),
            "step_size": float(step_size),
            "norm": norm,
            "logit_normalize": bool(logit_normalize),
            "adaptive_scale": "elementwise_abs_parameter",
            "optimization": "projected_gradient_ascent",
            "num_batches": len(batches),
            "num_examples": int(sum(y.size(0) for _, y in batches)),
        }

    finally:
        model.train(old_mode)
        model.zero_grad(set_to_none=True)


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
    random_start=False,
):
    """
    Estimate element-wise adaptive sharpness over multiple deterministic batches.

    By default, this computes one worst-case sharpness value per batch and returns
    the mean. This matches the common m-sharpness protocol where several fixed
    non-augmented mini-batches are evaluated and averaged. Set
    average_individual_batches=False to optimize one shared perturbation over the
    union of the selected batches.
    """
    fixed_batches = _collect_batches(loader, device, max_batches=max_batches)

    if average_individual_batches:
        per_batch = []
        for batch in fixed_batches:
            per_batch.append(
                _elementwise_adaptive_worst_sharpness_on_fixed_batches(
                    model=model,
                    batches=[batch],
                    criterion=criterion,
                    rho=rho,
                    steps=steps,
                    step_size=step_size,
                    norm=norm,
                    logit_normalize=logit_normalize,
                    random_start=random_start,
                )
            )

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
            "aggregation": "mean_of_per_batch_worst_case_sharpness",
            "num_batches": len(per_batch),
            "num_examples": int(sum(y.size(0) for _, y in fixed_batches)),
        }

    return _elementwise_adaptive_worst_sharpness_on_fixed_batches(
        model=model,
        batches=fixed_batches,
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
    rhos=(1e-4, 3e-4, 1e-3, 2e-3, 3e-3),
    steps=20,
    max_batches=8,
    step_size=None,
    norm="linf",
    logit_normalize=True,
    average_individual_batches=True,
    random_start=False,
):
    """
    Compute an element-wise adaptive sharpness curve over multiple rho values.

    This mirrors sharpness_curve(...), but each radius is the element-wise
    adaptive radius rho in delta = |w| * z. The same deterministic prefix of
    train batches is reused for every rho so that curve points are comparable.
    """
    fixed_batches = _collect_batches(loader, device, max_batches=max_batches)
    curve = []

    for rho in rhos:
        if average_individual_batches:
            per_batch = []
            for batch in fixed_batches:
                per_batch.append(
                    _elementwise_adaptive_worst_sharpness_on_fixed_batches(
                        model=model,
                        batches=[batch],
                        criterion=criterion,
                        rho=rho,
                        steps=steps,
                        step_size=step_size,
                        norm=norm,
                        logit_normalize=logit_normalize,
                        random_start=random_start,
                    )
                )

            mean_base_loss = sum(item["base_loss"] for item in per_batch) / len(per_batch)
            mean_perturbed_loss = sum(item["perturbed_loss"] for item in per_batch) / len(per_batch)
            mean_sharpness = sum(item["sharpness_delta"] for item in per_batch) / len(per_batch)
            max_sharpness = max(item["sharpness_delta"] for item in per_batch)

            curve.append(
                {
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
                    "aggregation": "mean_of_per_batch_worst_case_sharpness",
                    "num_batches": len(per_batch),
                    "num_examples": int(sum(y.size(0) for _, y in fixed_batches)),
                }
            )
        else:
            curve.append(
                _elementwise_adaptive_worst_sharpness_on_fixed_batches(
                    model=model,
                    batches=fixed_batches,
                    criterion=criterion,
                    rho=rho,
                    steps=steps,
                    step_size=step_size,
                    norm=norm,
                    logit_normalize=logit_normalize,
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