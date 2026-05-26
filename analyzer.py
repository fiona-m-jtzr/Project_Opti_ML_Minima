import copy
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from pyhessian import hessian


# -----------------------------
# Model and data utilities
# -----------------------------

def make_model():
    """Create a CIFAR-10-compatible ResNet-18."""
    model = resnet18(num_classes=10)

    # Adapt ImageNet ResNet stem for 32x32 CIFAR-10 images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    return model


def get_loaders(batch_size=128):
    """Return deterministic train/test loaders for analysis."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
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
    model.train()
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
# Weight perturbation utilities
# -----------------------------

def sample_random_direction_like(model):
    """Sample a random unit-norm direction in parameter space."""
    direction = []

    for p in model.parameters():
        if p.requires_grad:
            d = torch.randn_like(p)
            direction.append(d)

    norm = torch.sqrt(sum((d ** 2).sum() for d in direction))
    direction = [d / (norm + 1e-12) for d in direction]

    return direction


@torch.no_grad()
def add_direction_to_model(model, base_state, direction, scale):
    """Reset model to base_state and add scale * direction."""
    model.load_state_dict(base_state)

    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            p.add_(scale * direction[idx])
            idx += 1


# -----------------------------
# Sharpness via sampled neighbourhood
# -----------------------------

def max_loss_in_neighbourhood(
    model,
    loader,
    criterion,
    device,
    radius=1e-2,
    samples=20,
):
    """
    Randomly sample points in a ball around the solution and report max loss.

    This is a crude sharpness estimate:
        max L(w + epsilon) - L(w)
    """
    base_state = copy.deepcopy(model.state_dict())
    base_loss = full_loss(model, loader, criterion, device)

    max_loss = -math.inf
    max_delta = None

    for _ in range(samples):
        direction = sample_random_direction_like(model)
        add_direction_to_model(model, base_state, direction, radius)

        loss = full_loss(model, loader, criterion, device)
        delta = loss - base_loss

        if loss > max_loss:
            max_loss = loss
            max_delta = delta

    model.load_state_dict(base_state)

    return {
        "base_loss": base_loss,
        "max_neighbourhood_loss": max_loss,
        "sharpness_delta": max_delta,
    }


# -----------------------------
# 1D and 2D loss landscape
# -----------------------------

def loss_landscape_1d(
    model,
    loader,
    criterion,
    device,
    radius=1e-2,
    steps=21,
):
    """
    Evaluate loss along one random direction.

    Returns points:
        alpha, L(w + alpha * d)
    """
    base_state = copy.deepcopy(model.state_dict())
    direction = sample_random_direction_like(model)

    alphas = torch.linspace(-radius, radius, steps)
    losses = []

    for alpha in alphas:
        add_direction_to_model(model, base_state, direction, alpha.item())
        losses.append(full_loss(model, loader, criterion, device))

    model.load_state_dict(base_state)

    return [(float(a), float(l)) for a, l in zip(alphas, losses)]


def loss_landscape_2d(
    model,
    loader,
    criterion,
    device,
    radius=1e-2,
    steps=11,
):
    """
    Evaluate loss on a 2D grid around the solution.

    Uses two random directions d1 and d2:
        L(w + alpha * d1 + beta * d2)
    """
    base_state = copy.deepcopy(model.state_dict())
    d1 = sample_random_direction_like(model)
    d2 = sample_random_direction_like(model)

    values = []
    coords = torch.linspace(-radius, radius, steps)

    for alpha in coords:
        row = []

        for beta in coords:
            model.load_state_dict(base_state)

            idx = 0
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(alpha.item() * d1[idx] + beta.item() * d2[idx])
                        idx += 1

            row.append(full_loss(model, loader, criterion, device))

        values.append(row)

    model.load_state_dict(base_state)

    return {
        "coords": [float(c) for c in coords],
        "loss_grid": values,
    }


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
    """
    model.eval()

    inputs, targets = get_one_hessian_batch(trainloader, device)

    hessian_comp = hessian(
        model,
        criterion,
        data=(inputs, targets),
        cuda=(device == "cuda"),
    )

    # Top eigenvalues.
    eigenvalues, eigenvectors = hessian_comp.eigenvalues(top_n=top_n)

    # Hutchinson-style trace estimates.
    trace_estimates = hessian_comp.trace(maxIter=trace_samples)
    trace_mean = float(sum(trace_estimates) / len(trace_estimates))

    # Hessian spectral density estimate.
    # PyHessian returns eigenvalue-density samples suitable for plotting.
    density_eigen, density_weight = hessian_comp.density(
        iter=density_iter,
        n_v=density_samples,
    )

    # Approximate negative curvature ratio from the density samples.
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
        "top_eigenvalues": [float(v) for v in eigenvalues],
        "trace_estimates": [float(v) for v in trace_estimates],
        "trace_mean": trace_mean,

        # Normalized Hessian quantities.
        # This is a simple scale-aware normalization using ||w||^2.
        "weight_norm": weight_norm,
        "normalized_top_eigenvalue": top_eig * (weight_norm ** 2),
        "normalized_trace": trace_mean * (weight_norm ** 2),

        # Hessian spectrum density.
        "density_eigen": density_eigen,
        "density_weight": density_weight,

        # Approximate amount of negative curvature.
        "negative_curvature_ratio": float(negative_curvature_ratio),
    }


# -----------------------------
# Main analysis
# -----------------------------

def analyze(
    ckpt_path="cifar10_resnet18.pt",
    batch_size=128,
    neighbourhood_radius=1e-2,
    neighbourhood_samples=20,
    landscape_radius=1e-2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainloader, testloader = get_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()

    model = make_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Full-dataset losses.
    train_loss = full_loss(model, trainloader, criterion, device)
    test_loss = full_loss(model, testloader, criterion, device)

    # Gradient norm near the solution.
    grad_norm = gradient_norm(
        model=model,
        loader=trainloader,
        criterion=criterion,
        device=device,
        max_batches=10,
    )

    # Hessian eigenvalues, trace, density, normalized Hessian, negative curvature.
    hessian_metrics = compute_hessian_metrics(
        model=model,
        trainloader=trainloader,
        criterion=criterion,
        device=device,
    )

    # Random-neighbourhood sharpness.
    neighbourhood_metrics = max_loss_in_neighbourhood(
        model=model,
        loader=trainloader,
        criterion=criterion,
        device=device,
        radius=neighbourhood_radius,
        samples=neighbourhood_samples,
    )

    # 1D and 2D loss landscape samples.
    landscape_1d = loss_landscape_1d(
        model=model,
        loader=trainloader,
        criterion=criterion,
        device=device,
        radius=landscape_radius,
        steps=21,
    )

    landscape_2d = loss_landscape_2d(
        model=model,
        loader=trainloader,
        criterion=criterion,
        device=device,
        radius=landscape_radius,
        steps=11,
    )

    print("\n=== Minimum Analysis ===")
    print(f"Train loss: {train_loss:.6f}")
    print(f"Test loss:  {test_loss:.6f}")
    print(f"Train-test gap: {test_loss - train_loss:.6f}")

    print("\n=== Gradient Norm ===")
    print(f"Gradient norm, first 10 train batches: {grad_norm:.6e}")

    print("\n=== Hessian ===")
    print(f"Top eigenvalues: {hessian_metrics['top_eigenvalues']}")
    print(f"Trace mean:      {hessian_metrics['trace_mean']:.6f}")

    print("\n=== Normalized Hessian ===")
    print(f"Weight norm:                 {hessian_metrics['weight_norm']:.6f}")
    print(f"Normalized top eigenvalue:   {hessian_metrics['normalized_top_eigenvalue']:.6f}")
    print(f"Normalized trace:            {hessian_metrics['normalized_trace']:.6f}")

    print("\n=== Hessian Spectrum Density ===")
    print("Density eigenvalue samples and weights are stored in:")
    print("hessian_metrics['density_eigen']")
    print("hessian_metrics['density_weight']")

    print("\n=== Negative Curvature ===")
    print(
        "Approx. negative curvature ratio: "
        f"{hessian_metrics['negative_curvature_ratio']:.6f}"
    )

    print("\n=== Sampled-Neighbourhood Sharpness ===")
    print(f"Base loss:                   {neighbourhood_metrics['base_loss']:.6f}")
    print(f"Max neighbourhood loss:      {neighbourhood_metrics['max_neighbourhood_loss']:.6f}")
    print(f"Sharpness delta:             {neighbourhood_metrics['sharpness_delta']:.6f}")

    print("\n=== 1D Loss Landscape ===")
    for alpha, loss in landscape_1d:
        print(f"alpha={alpha:+.6f}, loss={loss:.6f}")

    print("\n=== 2D Loss Landscape ===")
    print("2D landscape stored as:")
    print("landscape_2d['coords']")
    print("landscape_2d['loss_grid']")


if __name__ == "__main__":
    analyze()