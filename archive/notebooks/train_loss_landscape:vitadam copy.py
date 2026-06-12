# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import wandb
from hessian.hessian import hessian
from itertools import islice
import plotly.graph_objects as go
from torch.utils.data import random_split


# ==========================================
# MODEL DEFINITIONS
# ==========================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)
        q = q_out.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_out.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_out.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTCIFAR10(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

training_data = torchvision.datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform
)
test_data = torchvision.datasets.CIFAR10(
    root="data", train=False, download=True, transform=transform
)

val_fraction = 0.1
val_size   = int(len(training_data) * val_fraction)
train_size = len(training_data) - val_size
train_set, val_set = random_split(
    training_data, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)
test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()


# ==========================================
# 2. MODEL LOADING
# ==========================================
api = wandb.Api()
model_name = 'model-FINAL_MODEL_vit_adam_lr0.0005_wd0.0_bs256_cosine_warmup_seed1'
artifact = api.artifact(f"fiona-jetzer-epfl/OptiML_Minima/{model_name}:v0")
artifact_dir = Path(artifact.download())
ckpt_path = list(artifact_dir.rglob("*.pt")) + list(artifact_dir.rglob("*.pth"))
grad_min  = list(filter(lambda k: 'min_grad' in k.name, ckpt_path))[0]

model = ViTCIFAR10().to(device)
checkpoint = torch.load(grad_min, map_location=device)
state_dict = checkpoint["model"]
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# Store weights at the found minimum (W*)
target_weights = [p.data.clone() for p in model.parameters()]


# ==========================================
# 3. HESSIAN EIGENVECTOR DIRECTIONS
# ==========================================

def get_hessian_batch_tensor(loader, device, num_batches=32):
    """Concatenate num_batches mini-batches into a single tensor for Hessian computation."""
    xs, ys = [], []
    for x, y in islice(loader, num_batches):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0).to(device), torch.cat(ys, dim=0).to(device)


def compute_top_and_bottom_hessian_eigenpairs(
    model, loader, criterion, device,
    num_batches=32, tol=1e-3, maxiter=100, ncv=None,
):
    """
    Compute the top-5 Hessian eigenvalues and eigenvectors using PyHessian.
    Returns the eigenpair with the largest and smallest eigenvalue among the top-5.
    """
    import numpy as np

    model.eval()
    model.zero_grad(set_to_none=True)

    # Build a fixed subset loader for consistent Hessian estimation
    inputs, targets = get_hessian_batch_tensor(loader, device, num_batches=num_batches)
    subset = torch.utils.data.TensorDataset(inputs, targets)
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=128)

    hessian_comp = hessian(
        model, criterion,
        dataloader=subset_loader,
        cuda=True,
    )

    # Compute top-5 eigenvalues and eigenvectors via Lanczos iteration
    top_vals, top_vecs = hessian_comp.eigenvalues(top_n=5)

    idx_max = np.argmax(top_vals)
    idx_min = np.argmin(top_vals)

    return {
        "top_eigenvalue":   top_vals[idx_max],
        "top_eigenvector":  top_vecs[idx_max],
        "top5_eigenvalue":  top_vals[idx_min],
        "top5_eigenvector": top_vecs[idx_min],
        "all_eigenvalues":  top_vals,
        "hessian_comp":     hessian_comp,
    }


hessian_results = compute_top_and_bottom_hessian_eigenpairs(
    model, train_dataloader, criterion, device, num_batches=32
)

print(f"-> Top-1 Eigenvalue: {hessian_results['top_eigenvalue']:.4f}")
print(f"-> Top-5 Eigenvalue: {hessian_results['top5_eigenvalue']:.4f}")
print(f"-> All eigenvalues:  {hessian_results['all_eigenvalues']}")

dir_x = hessian_results['top_eigenvector']   # direction of maximum curvature
dir_y = hessian_results['top5_eigenvector']  # direction of 5th largest curvature


# ==========================================
# 4. LOSS EVALUATION
# ==========================================

def evaluate_loss_subsampled(model, loader, criterion, device, num_batches=32):
    """Compute average loss over a fixed number of mini-batches."""
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for inputs, targets in islice(loader, num_batches):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss   += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples


# ==========================================
# 5. LOSS LANDSCAPE GRID COMPUTATION
# ==========================================
grid_resolution = 15
steps_x = np.linspace(-0.01, 0.01, grid_resolution)
steps_y = np.linspace(-0.01, 0.01, grid_resolution)

X, Y = np.meshgrid(steps_x, steps_y)
Z = np.zeros_like(X)

print("Computing loss landscape...")
for i, x_coeff in enumerate(steps_x):
    for j, y_coeff in enumerate(steps_y):
        # Perturb weights: W = W* + alpha * v1 + beta * v2
        for p, w_star, dx, dy in zip(model.parameters(), target_weights, dir_x, dir_y):
            p.data = w_star + x_coeff * dx + y_coeff * dy
        Z[j, i] = evaluate_loss_subsampled(model, train_dataloader, criterion, device)
    print(f"  Row {i+1}/{grid_resolution} done.")

# Restore original weights
for p, w_star in zip(model.parameters(), target_weights):
    p.data = w_star


# ==========================================
# 6. VISUALIZATION
# ==========================================
ci = grid_resolution // 2

# --- Matplotlib: 2D contour + 3D surface ---
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1)
contour = ax1.contourf(X, Y, Z, levels=30, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='Loss')
ax1.scatter(0, 0, color='red', marker='*', s=150, label='Found minimum')
ax1.set_title('Loss Landscape 2D (Contours)')
ax1.set_xlabel('1st eigenvector')
ax1.set_ylabel('5th eigenvector')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Loss')
ax2.scatter(0, 0, Z[ci, ci], color='red', marker='*', s=150, zorder=10)
ax2.set_title('Loss Landscape 3D')
ax2.set_xlabel('1st eigenvector')
ax2.set_ylabel('5th eigenvector')
ax2.set_zlabel('Loss')
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig(f"loss_landscape_vit_zoomed/{model_name}.png", dpi=300, bbox_inches='tight')

# --- Plotly: interactive 3D surface ---
fig_plotly = go.Figure(data=[
    go.Surface(x=X, y=Y, z=Z, colorscale='Plasma'),
    go.Scatter3d(
        x=[0], y=[0], z=[Z[ci, ci]],
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Minimum (W*)',
    )
])

fig_plotly.update_layout(
    scene=dict(
        xaxis_title='1st eigenvector',
        yaxis_title='5th eigenvector',
        zaxis_title='Loss',
    ),
    autosize=False,
    width=800, height=800,
)

fig_plotly.write_html(f"{model_name}.html")