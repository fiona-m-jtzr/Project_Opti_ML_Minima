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


# MODEL DEFINITION

"""
Vision Transformer (ViT) for CIFAR-10.
Minimal implementation — no dropout, no stochastic depth, no presets.
"""

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

        # 1. Project Q, K, V separately
        # Resulting shapes: (B, N, embed_dim)
        q_out = self.q(x)
        k_out = self.k(x)
        v_out = self.v(x)

        # 2. Reshape and permute for Multi-Head Attention
        # Target shape for matmul: (B, num_heads, N, head_dim)
        q = q_out.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_out.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_out.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Standard Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 4. Context vector & output projection
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


# SETUP & DATA LOADING

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

training_data = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_data = torchvision.datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

val_fraction = 0.1
val_size   = int(len(training_data) * val_fraction)
train_size = len(training_data) - val_size
train_set, val_set = random_split(
    training_data, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

criterion = nn.CrossEntropyLoss()


# MODEL LOADING 

api = wandb.Api()
model_name = 'model-FINAL_MODEL_vit_adam_lr0.0005_wd0.0_bs256_cosine_warmup_seed1'
artifact = api.artifact(f"fiona-jetzer-epfl/OptiML_Minima/{model_name}:v0")
artifact_dir = Path(artifact.download())
ckpt_path = (list(artifact_dir.rglob("*.pt")) + list(artifact_dir.rglob("*.pth")))
grad_min = list(filter(lambda k: 'min_grad' in k.name, ckpt_path))[0]

model = ViTCIFAR10().to(device)
checkpoint = torch.load(grad_min, map_location=device)

state_dict = checkpoint["model"]
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# Store minima's weights
target_weights = [p.data.clone() for p in model.parameters()]


# HESSIAN EIGENVECTORS DIRECTIONS

def get_hessian_batch_tensor(loader, device, num_batches=32):
    """Concatenate num_batches mini-batches into a single tensor for Hessian computation."""
    xs, ys = [], []
    for x, y in islice(loader, num_batches):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0).to(device), torch.cat(ys, dim=0).to(device)

def compute_top_and_bottom_hessian_eigenpairs(
    model,
    loader,
    criterion,
    device,
    num_batches=32,
    tol=1e-3,
    maxiter=100,
    ncv=None,
):
    """
    Compute the largest algebraic Hessian eigenvalue and the smallest algebraic
    Hessian eigenvalue, together with their eigenvectors.

    Returns
    -------
    dict with:
        top_eigenvalue:
            Largest algebraic eigenvalue of H.
        top_eigenvector:
            List of tensors matching model.parameters().
        top5_eigenvalue:
            Smallest algebraic eigenvalue of H. This is the most negative one
            if negative curvature exists.
        top5_eigenvector:
            List of tensors matching model.parameters().
        hessian_comp
    """

    model.eval()
    model.zero_grad(set_to_none=True)

    inputs, targets = get_hessian_batch_tensor(loader, device, num_batches=num_batches)
    subset = torch.utils.data.TensorDataset(inputs.to(device), targets.to(device))
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=128)

    hessian_comp = hessian(
        model,
        criterion,
        dataloader=subset_loader,
        cuda=True,
    )

    top_vals, top_vecs = hessian_comp.eigenvalues(top_n=5)

    idx_max = np.argmax(top_vals)
    idx_min = np.argmin(top_vals)
    max_eigenvalue = top_vals[idx_max]
    min_eigenvalue = top_vals[idx_min]
    max_eigenvector = top_vecs[idx_max]
    min_eigenvector = top_vecs[idx_min]

    return {
        "top_eigenvalue":   max_eigenvalue,
        "top_eigenvector":  max_eigenvector,   
        "top5_eigenvalue":  min_eigenvalue,
        "top5_eigenvector": min_eigenvector,    
        "hessian_comp":     hessian_comp,
    }

hessian_results = compute_top_and_bottom_hessian_eigenpairs(
    model, train_dataloader, criterion, device, num_batches=32
)

print(f"-> Top Eigenvalue (Max courbure) : {hessian_results['top_eigenvalue']:.4f}")
print(f"-> 5th Eigenvalue (Min courbure) : {hessian_results['top5_eigenvalue']:.4f}")

raw_dir_x = hessian_results['top_eigenvector']
raw_dir_y = hessian_results['top5_eigenvector']
hessian_comp = hessian_results['hessian_comp']

def normalize_direction_filter_wise(direction_vectors, model_parameters):
    """Applique la normalisation par filtre sur les vecteurs propres pour préserver l'échelle."""
    normalized_direction = []
    for d, p in zip(direction_vectors, model_parameters):
        d_clone = d.clone()
        if len(p.shape) >= 2:
            for i in range(p.shape[0]):
                d_filter = d_clone[i]
                p_filter = p[i]
                d_filter.mul_(p_filter.norm() / (d_filter.norm() + 1e-10))
        else:
            d_clone.mul_(p.norm() / (d_clone.norm() + 1e-10))
        normalized_direction.append(d_clone)
    return normalized_direction

dir_x = raw_dir_x
dir_y = raw_dir_y


# LOSS EVALUATION

def evaluate_loss_subsampled(model, loader, criterion, device, num_batches=32):
    """Calcule la loss moyenne sur un sous-ensemble de batchs pour accélérer le runtime."""
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in islice(loader, num_batches):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
    return total_loss / total_samples


# LOSS LANDSCAPE AND GRID COMPUTATION

grid_resolution = 15
steps_x = np.linspace(-0.01, 0.01, grid_resolution)
steps_y = np.linspace(-0.01, 0.01, grid_resolution)

X, Y = np.meshgrid(steps_x, steps_y)
Z = np.zeros_like(X)

print("Computing loss landscape...")
for i, x_coeff in enumerate(steps_x):
    for j, y_coeff in enumerate(steps_y):
        # W = W* + x*dir_x + y*dir_y
        for p, w_start, dx, dy in zip(model.parameters(), target_weights, dir_x, dir_y):
            p.data = w_start + x_coeff * dx + y_coeff * dy

        # Calculate loss at W
        loss_val = evaluate_loss_subsampled(model, train_dataloader, criterion, device)
        Z[j, i] = loss_val # Attention à l'indexation (y=lignes, x=colonnes)

    print(f"Row {i+1}/{grid_resolution} done.")
for p, w_start in zip(model.parameters(), target_weights):
    p.data = w_start


# VISUALIZATION

# Matplotlib: 2D contour + 3D surface
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1)
contour = ax1.contourf(X, Y, Z, levels=30, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='Loss')
ax1.scatter(0, 0, color='red', marker='*', s=150) 
ax1.set_xlabel('1st eigenvector')
ax1.set_ylabel('5th eigenvector')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Loss')

min_loss = Z[grid_resolution//2, grid_resolution//2]
ax2.scatter(0, 0, min_loss, color='red', marker='*', s=150, zorder=10)

ax2.set_xlabel('1st egenvector')
ax2.set_ylabel('5th eigenvector')
ax2.set_zlabel('Loss')

ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig(f"loss_landscape_vit_zoomed/{model_name}.png", dpi=300, bbox_inches='tight') 

# Plotly: interactive 3D surface
ci = grid_resolution // 2

fig_plotly = go.Figure(data=[
    go.Surface(x=X, y=Y, z=Z, colorscale='Plasma'),
    go.Scatter3d(
        x=[0], y=[0], z=[Z[ci, ci]],
        mode='markers',
        marker=dict(size=8, color='red'),
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

fig_plotly.write_html(f"{model_name}_zoomed.html")