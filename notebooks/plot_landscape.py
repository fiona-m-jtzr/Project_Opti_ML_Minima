"""
Loss Landscape Visualization between Three Trained ResNet20 Models
==================================================================
Uses triangular (barycentric) interpolation in parameter space to map
the cross-entropy loss and accuracy over the 2D simplex defined by the
three model checkpoints.

Usage:
    python plot_loss_landscape.py

Outputs:
    loss_landscape.png  –  2D filled-contour plot of the loss surface
    acc_landscape.png   –  2D filled-contour plot of the accuracy surface
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap

# ──────────────────────────────────────────────────────────────
# 0.  Device
# ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ──────────────────────────────────────────────────────────────
# 1.  Model definition
# ──────────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
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
                nn.init.ones_(m.weight);  nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight);  nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out);  out = self.layer2(out);  out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ──────────────────────────────────────────────────────────────
# 2.  Load the three models
# ──────────────────────────────────────────────────────────────
def load_model(path):
    checkpoint = torch.load(path, map_location=device)
    model = ResNet20()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


print("Loading models …")
model0 = load_model("notebooks/best0.pt")
model1 = load_model("notebooks/best1.pt")
model2 = load_model("notebooks/best2.pt")
models = [model0, model1, model2]


# ──────────────────────────────────────────────────────────────
# 3.  CIFAR-10 validation loader  (subset for speed)
# ──────────────────────────────────────────────────────────────
EVAL_SAMPLES = 2000   # increase for higher fidelity (slower)
BATCH_SIZE   = 256

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

full_val = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
subset_idx = torch.randperm(len(full_val))[:EVAL_SAMPLES].tolist()
val_subset  = torch.utils.data.Subset(full_val, subset_idx)
val_loader  = torch.utils.data.DataLoader(
    val_subset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=False   # num_workers=0 required on macOS
)


# ──────────────────────────────────────────────────────────────
# 4.  Parameter-space interpolation utilities
# ──────────────────────────────────────────────────────────────
def get_weights(model):
    """Flatten all parameters into a single 1-D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_weights(model, flat_weights):
    """Load a flat weight vector back into a model (in-place)."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat_weights[offset:offset + n].view(p.shape))
        offset += n


def interpolate_weights(w0, w1, w2, alpha, beta):
    """
    Barycentric interpolation:
        w = alpha*w0 + beta*w1 + (1-alpha-beta)*w2
    Valid when alpha >= 0, beta >= 0, alpha+beta <= 1.
    """
    gamma = 1.0 - alpha - beta
    return alpha * w0 + beta * w1 + gamma * w2


# ──────────────────────────────────────────────────────────────
# 5.  Evaluation helper
# ──────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(model):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        correct    += outputs.argmax(1).eq(targets).sum().item()
        total      += targets.size(0)
    return total_loss / total, 100.0 * correct / total


# ──────────────────────────────────────────────────────────────
# 6.  Build the triangular grid  (barycentric coordinates)
# ──────────────────────────────────────────────────────────────
GRID_STEPS = 20   # number of steps along each edge; 20 → 231 points

w0 = get_weights(model0)
w1 = get_weights(model1)
w2 = get_weights(model2)

# Probe model (reuse architecture; avoid repeated allocations)
probe = copy.deepcopy(model0)
probe.to(device)

alphas, betas, losses, accs = [], [], [], []

total_pts = (GRID_STEPS + 1) * (GRID_STEPS + 2) // 2
print(f"Evaluating {total_pts} grid points …")

pt = 0
for i in range(GRID_STEPS + 1):
    for j in range(GRID_STEPS + 1 - i):
        alpha = i / GRID_STEPS
        beta  = j / GRID_STEPS
        w_interp = interpolate_weights(w0, w1, w2, alpha, beta)
        set_weights(probe, w_interp)
        loss, acc = evaluate(probe)
        alphas.append(alpha)
        betas.append(beta)
        losses.append(min(loss, 10))
        accs.append(acc)
        pt += 1
        if pt % 20 == 0 or pt == total_pts:
            print(f"  {pt}/{total_pts}  loss={loss:.4f}  acc={acc:.1f}%")

alphas = np.array(alphas)
betas  = np.array(betas)
losses = np.array(losses)
accs   = np.array(accs)


# ──────────────────────────────────────────────────────────────
# 7.  Convert barycentric → 2-D Cartesian for plotting
# ──────────────────────────────────────────────────────────────
# Place the three vertices at the corners of an equilateral triangle.
# v0=(0,0), v1=(1,0), v2=(0.5, sqrt(3)/2)
v0 = np.array([0.0,  0.0])
v1 = np.array([1.0,  0.0])
v2 = np.array([0.5,  np.sqrt(3) / 2])

gamma = 1.0 - alphas - betas
xs = alphas * v0[0] + betas * v1[0] + gamma * v2[0]
ys = alphas * v0[1] + betas * v1[1] + gamma * v2[1]

triang = mtri.Triangulation(xs, ys)


# ──────────────────────────────────────────────────────────────
# 8.  Plot
# ──────────────────────────────────────────────────────────────
VERTEX_LABELS = ["Model 0", "Model 1", "Model 2"]
VERTEX_COORDS = [v0, v1, v2]

def add_triangle_labels(ax, vertex_losses, vertex_accs, mode="loss"):
    """Annotate the three vertices with their model names and metrics."""
    for k, (label, coord) in enumerate(zip(VERTEX_LABELS, VERTEX_COORDS)):
        # Find the grid point closest to this vertex
        idx_arr = np.where(
            (np.abs(alphas - (1 if k == 0 else 0)) < 1e-9) if k == 0 else
            (np.abs(betas  - (1 if k == 1 else 0)) < 1e-9) if k == 1 else
            (np.abs(gamma  - (1 if k == 2 else 0)) < 1e-9)
        )[0]
        val = vertex_losses[k] if mode == "loss" else vertex_accs[k]
        unit = "" if mode == "loss" else "%"
        offsets = [(-0.12, -0.06), (0.02, -0.06), (-0.05, 0.04)]
        ox, oy = offsets[k]
        ax.annotate(
            f"{label}\n{val:.3f}{unit}",
            xy=coord, xytext=(coord[0] + ox, coord[1] + oy),
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8,
                      ec="gray", lw=0.8),
        )
        ax.plot(*coord, "o", ms=8, color="white",
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)


# ── locate vertex values ──────────────────────────────────────
def vertex_value(k, arr):
    """Return the array value at vertex k (pure barycentric corner)."""
    if k == 0:
        mask = (np.abs(alphas - 1) < 1e-9)
    elif k == 1:
        mask = (np.abs(betas  - 1) < 1e-9)
    else:
        mask = (np.abs(gamma  - 1) < 1e-9)
    return arr[mask][0]

gamma = 1.0 - alphas - betas
v_losses = [vertex_value(k, losses) for k in range(3)]
v_accs   = [vertex_value(k, accs)   for k in range(3)]

# ── custom colormaps ──────────────────────────────────────────
loss_cmap = LinearSegmentedColormap.from_list(
    "loss_cmap", ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#f5a623"], N=256
)
acc_cmap = LinearSegmentedColormap.from_list(
    "acc_cmap",  ["#0d0221", "#0a1045", "#1a4fd6", "#34c8e0", "#e0f7fa"], N=256
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0d0d0d")

for ax in axes:
    ax.set_facecolor("#0d0d0d")
    ax.set_aspect("equal")
    ax.axis("off")

# ── Loss plot ─────────────────────────────────────────────────
cf0 = axes[0].tricontourf(triang, losses, levels=30, cmap=loss_cmap)
axes[0].tricontour (triang, losses, levels=10,
                    colors="white", linewidths=0.4, alpha=0.35)
cb0 = fig.colorbar(cf0, ax=axes[0], pad=0.02, fraction=0.046)
cb0.set_label("Cross-Entropy Loss", color="white", fontsize=10)
cb0.ax.yaxis.set_tick_params(color="white")
plt.setp(cb0.ax.yaxis.get_ticklabels(), color="white")
axes[0].set_title("Loss Landscape", color="white", fontsize=14,
                   fontweight="bold", pad=12)
add_triangle_labels(axes[0], v_losses, v_accs, mode="loss")

# ── Accuracy plot ─────────────────────────────────────────────
cf1 = axes[1].tricontourf(triang, accs, levels=30, cmap=acc_cmap)
axes[1].tricontour (triang, accs, levels=10,
                    colors="white", linewidths=0.4, alpha=0.35)
cb1 = fig.colorbar(cf1, ax=axes[1], pad=0.02, fraction=0.046)
cb1.set_label("Accuracy (%)", color="white", fontsize=10)
cb1.ax.yaxis.set_tick_params(color="white")
plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")
axes[1].set_title("Accuracy Landscape", color="white", fontsize=14,
                   fontweight="bold", pad=12)
add_triangle_labels(axes[1], v_losses, v_accs, mode="acc")

fig.suptitle(
    "Loss Landscape — Triangular Interpolation between 3 ResNet-20 Models\n"
    f"(CIFAR-10 · {EVAL_SAMPLES} samples · grid steps = {GRID_STEPS})",
    color="white", fontsize=13, y=1.02
)

plt.tight_layout()
fig.savefig("loss_landscape.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved → loss_landscape.png")
plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 9.  Optional: 3-D surface plot
# ──────────────────────────────────────────────────────────────
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

fig3d = plt.figure(figsize=(10, 7))
fig3d.patch.set_facecolor("#0d0d0d")
ax3d  = fig3d.add_subplot(111, projection="3d")
ax3d.set_facecolor("#0d0d0d")

surf = ax3d.plot_trisurf(xs, ys, losses, triangles=triang.triangles,
                          cmap=loss_cmap, edgecolor="none", alpha=0.92)
fig3d.colorbar(surf, ax=ax3d, pad=0.1, fraction=0.03,
               label="Cross-Entropy Loss").ax.yaxis.label.set_color("white")

ax3d.set_title("Loss Surface (3-D)", color="white", fontsize=13,
               fontweight="bold")
for spine in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
    spine.label.set_color("white")
    spine.pane.fill = False
ax3d.tick_params(colors="white")
ax3d.grid(True, color="gray", alpha=0.2)

# Vertex markers
for k, (coord, lbl) in enumerate(zip(VERTEX_COORDS, VERTEX_LABELS)):
    ax3d.scatter(*coord, v_losses[k], s=60, color="white",
                 edgecolors="red", linewidths=1.5, zorder=10)
    ax3d.text(coord[0], coord[1], v_losses[k] + 0.02, lbl,
              color="white", fontsize=8, ha="center")

fig3d.tight_layout()
fig3d.savefig("loss_landscape_3d.png", dpi=150, bbox_inches="tight",
              facecolor=fig3d.get_facecolor())
print("Saved → loss_landscape_3d.png")
plt.close(fig3d)

print("\nDone! Outputs: loss_landscape.png  loss_landscape_3d.png")