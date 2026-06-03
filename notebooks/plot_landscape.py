"""
Loss Landscape Visualization for Three ResNet20 Minima
=======================================================
Plots the loss landscape in the 2D subspace spanned by the directions
connecting three trained models (model0, model1, model2) on CIFAR-10.

Method
------
1. Flatten all parameters of each model into a vector.
2. Use PCA on the displacement vectors (model1 - model0, model2 - model0)
   to find two orthogonal directions that span the plane containing the
   three minima.
3. Apply filter-normalisation to each direction so that the landscape
   scale is comparable across layers (Li et al., 2018).
4. Sweep a 2D grid around model0 in that plane, compute the loss at each
   point, and plot the result as a filled contour + surface plot.

Reference
---------
Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein.
"Visualizing the Loss Landscape of Neural Nets." NeurIPS 2018.
https://arxiv.org/abs/1712.09913

Usage
-----
Adjust the CONFIG block below, then run:
    python plot_loss_landscape.py
"""

import copy
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from matplotlib import cm

# ─────────────────────────────────────────────
# CONFIG  –  edit these paths / settings
# ─────────────────────────────────────────────
CONFIG = dict(
    # Paths to saved model state-dicts
    model0_path="model0.pth",
    model1_path="model1.pth",
    model2_path="model2.pth",

    # CIFAR-10 root (will be downloaded if absent)
    data_root="./data",

    # Grid resolution (N×N points).  Start with 21 for a quick preview,
    # then raise to 51-101 for publication quality.
    grid_size=21,

    # How far to travel along each PCA direction (in units of the
    # filter-normalised direction vector).  Increase if minima fall
    # outside the visible region.
    alpha_range=(-1.0, 1.0),
    beta_range=(-1.0, 1.0),

    # Number of CIFAR-10 samples used to evaluate the loss at each grid
    # point.  256-512 gives a good speed/accuracy trade-off.
    n_eval_samples=512,
    batch_size=128,

    # Output filenames
    out_contour="loss_landscape_contour.png",
    out_surface="loss_landscape_surface.png",
)
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
# MODEL DEFINITION  (paste your own here)
# ══════════════════════════════════════════════

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, n_blocks=3, stride=1)
        self.layer2 = self._make_layer(16, 32, n_blocks=3, stride=2)
        self.layer3 = self._make_layer(32, 64, n_blocks=3, stride=2)
        self.fc     = nn.Linear(64, num_classes)
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ══════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def params_to_vector(model: nn.Module) -> np.ndarray:
    """Flatten all parameters (and buffers) into a single numpy vector."""
    return np.concatenate([
        p.detach().cpu().numpy().ravel()
        for p in model.parameters()
    ])


def vector_to_params(vec: np.ndarray, model: nn.Module) -> None:
    """Write a flat numpy vector back into model parameters (in-place)."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(
            torch.from_numpy(vec[offset: offset + numel]).reshape(p.shape)
        )
        offset += numel


# ── Filter normalisation (Li et al. §3.1) ─────────────────────────────────────
# Each direction vector is rescaled so that, for every convolutional / linear
# filter, the norm of the direction matches the norm of the corresponding
# filter in the reference model.  This removes the scale ambiguity caused by
# different weight magnitudes in different layers.

def _filter_norms(model: nn.Module) -> list[np.ndarray]:
    """Return per-filter norms for every parameter tensor in the model."""
    norms = []
    for p in model.parameters():
        w = p.detach().cpu().numpy()
        if w.ndim == 1:                        # BN scale / bias / fc bias
            norms.append(np.abs(w) + 1e-10)   # treat each scalar as its own filter
        else:
            # Reshape to (n_filters, -1) and compute L2 norm per filter
            n = w.reshape(w.shape[0], -1)
            norms.append(np.linalg.norm(n, axis=1) + 1e-10)
    return norms


def filter_normalise(direction: np.ndarray,
                     reference_norms: list[np.ndarray],
                     model: nn.Module) -> np.ndarray:
    """
    Rescale `direction` (a flat parameter vector) so that each filter in
    `direction` has the same L2 norm as the corresponding filter in the
    reference model.
    """
    d = direction.copy()
    offset = 0
    for p, ref_norm in zip(model.parameters(), reference_norms):
        w = p.detach().cpu().numpy()
        numel = p.numel()
        d_block = d[offset: offset + numel].reshape(w.shape)

        if w.ndim == 1:
            # scale each scalar independently
            d_block = d_block * (ref_norm / (np.abs(d_block) + 1e-10))
        else:
            n_filters = w.shape[0]
            d_flat  = d_block.reshape(n_filters, -1)
            d_norms = np.linalg.norm(d_flat, axis=1, keepdims=True) + 1e-10
            scale   = ref_norm.reshape(-1, 1) / d_norms
            d_block = (d_flat * scale).reshape(w.shape)

        d[offset: offset + numel] = d_block.ravel()
        offset += numel
    return d


# ── PCA-based plane ────────────────────────────────────────────────────────────

def compute_pca_directions(theta0: np.ndarray,
                           theta1: np.ndarray,
                           theta2: np.ndarray,
                           model: nn.Module):
    """
    Return two orthonormal, filter-normalised direction vectors that span
    the plane containing the three weight vectors.

    Returns
    -------
    u, v : np.ndarray  (same shape as theta0)
        Orthonormal basis vectors of the plane.
    coords : (3, 2) np.ndarray
        2-D coordinates of model0, model1, model2 in (u, v) space.
    """
    # Raw displacement vectors from model0
    d1 = theta1 - theta0
    d2 = theta2 - theta0

    # Stack and run PCA to get the two principal directions
    M = np.stack([d1, d2], axis=0)         # shape (2, D)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    u_raw = Vt[0]   # first principal direction
    v_raw = Vt[1]   # second principal direction

    # Filter-normalise w.r.t. model0
    ref_norms = _filter_norms(model)
    u = filter_normalise(u_raw, ref_norms, model)
    v = filter_normalise(v_raw, ref_norms, model)

    # Re-orthonormalise after filter normalisation (Gram-Schmidt)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v - np.dot(v, u) * u
    v = v / (np.linalg.norm(v) + 1e-12)

    # Coordinates of the three minima in (u, v) space
    def proj(d):
        return np.array([np.dot(d, u), np.dot(d, v)])

    coords = np.stack([proj(np.zeros_like(theta0)),
                       proj(d1),
                       proj(d2)], axis=0)
    return u, v, coords


# ── Loss evaluation ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loss(model: nn.Module,
                  loader: DataLoader,
                  device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n    = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_n    += x.size(0)
    return total_loss / total_n


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main():
    cfg    = CONFIG
    device = get_device()
    print(f"Using device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models …")
    model0 = ResNet20().to(device)
    model1 = ResNet20().to(device)
    model2 = ResNet20().to(device)

    model0.load_state_dict(torch.load(cfg["/notebooks/model0_path"], map_location=device))
    model1.load_state_dict(torch.load(cfg["/notebooks/model1_path"], map_location=device))
    model2.load_state_dict(torch.load(cfg["/notebooks/model2_path"], map_location=device))

    theta0 = params_to_vector(model0)
    theta1 = params_to_vector(model1)
    theta2 = params_to_vector(model2)
    print(f"  Parameter space dimension: {theta0.shape[0]:,}")

    # ── Build evaluation dataset ──────────────────────────────────────────────
    print("Preparing CIFAR-10 evaluation subset …")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    full_dataset = torchvision.datasets.CIFAR10(
        root=cfg["data_root"], train=False, download=True, transform=transform
    )
    indices = torch.randperm(len(full_dataset))[:cfg["n_eval_samples"]].tolist()
    subset  = Subset(full_dataset, indices)
    loader  = DataLoader(subset, batch_size=cfg["batch_size"],
                         shuffle=False, num_workers=2, pin_memory=True)

    # ── Compute PCA directions ────────────────────────────────────────────────
    print("Computing PCA directions in parameter space …")
    u, v, minima_coords = compute_pca_directions(theta0, theta1, theta2, model0)
    print(f"  Minima coordinates in (u,v) plane:")
    labels = ["model0", "model1", "model2"]
    for lbl, (a, b) in zip(labels, minima_coords):
        print(f"    {lbl}: α={a:.4f}, β={b:.4f}")

    # ── Sweep the grid ────────────────────────────────────────────────────────
    N      = cfg["grid_size"]
    alphas = np.linspace(*cfg["alpha_range"], N)
    betas  = np.linspace(*cfg["beta_range"],  N)
    AA, BB = np.meshgrid(alphas, betas)     # (N, N)
    losses = np.zeros((N, N))

    # We perturb a single evaluation model to avoid repeated allocation
    eval_model = copy.deepcopy(model0).to(device)

    total_evals = N * N
    print(f"Evaluating loss on {total_evals} grid points …")
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            theta = theta0 + alpha * u + beta * v
            vector_to_params(theta, eval_model)
            losses[i, j] = evaluate_loss(eval_model, loader, device)

        pct = (i + 1) / N * 100
        print(f"  row {i+1:3d}/{N}  ({pct:5.1f}%)", end="\r", flush=True)

    print(f"\nLoss range: [{losses.min():.4f}, {losses.max():.4f}]")

    # ── Plot 1: Filled contour map ────────────────────────────────────────────
    print("Plotting contour map …")
    fig, ax = plt.subplots(figsize=(8, 7))
    levels = np.linspace(losses.min(), losses.max(), 40)
    cf = ax.contourf(AA, BB, losses, levels=levels, cmap="coolwarm", alpha=0.85)
    ax.contour(AA, BB, losses, levels=levels[::4], colors="k",
               linewidths=0.4, alpha=0.5)
    plt.colorbar(cf, ax=ax, label="Cross-Entropy Loss")

    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    markers = ["*", "^", "D"]
    for (a, b), lbl, col, mk in zip(minima_coords, labels, colors, markers):
        ax.scatter(a, b, c=col, marker=mk, s=200, zorder=5,
                   edgecolors="white", linewidths=1.2, label=lbl)
        ax.annotate(f" {lbl}", (a, b), fontsize=9, color=col,
                    fontweight="bold", va="bottom")

    ax.set_xlabel("α  (1st PCA direction)", fontsize=12)
    ax.set_ylabel("β  (2nd PCA direction)", fontsize=12)
    ax.set_title("Loss Landscape – 2D PCA Plane\n(filter-normalised directions)",
                 fontsize=13)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(cfg["out_contour"], dpi=150)
    print(f"  Saved → {cfg['out_contour']}")

    # ── Plot 2: 3-D surface ───────────────────────────────────────────────────
    print("Plotting 3-D surface …")
    fig3d = plt.figure(figsize=(10, 7))
    ax3d  = fig3d.add_subplot(111, projection="3d")
    surf  = ax3d.plot_surface(AA, BB, losses, cmap="coolwarm",
                              rstride=1, cstride=1, alpha=0.85,
                              linewidth=0, antialiased=True)
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, label="Loss")

    for (a, b), lbl, col in zip(minima_coords, labels, colors):
        # Interpolate loss at minima location for z-placement
        z_val = float(losses[
            np.argmin(np.abs(betas - b)),
            np.argmin(np.abs(alphas - a))
        ])
        ax3d.scatter([a], [b], [z_val + 0.02], c=col, s=80,
                     zorder=10, label=lbl)

    ax3d.set_xlabel("α"); ax3d.set_ylabel("β"); ax3d.set_zlabel("Loss")
    ax3d.set_title("Loss Landscape – 3-D View\n(filter-normalised PCA directions)")
    ax3d.legend(loc="upper left", fontsize=8)
    fig3d.tight_layout()
    fig3d.savefig(cfg["out_surface"], dpi=150)
    print(f"  Saved → {cfg['out_surface']}")

    #plt.show()
    print("Done.")


if __name__ == "__main__":
    main()