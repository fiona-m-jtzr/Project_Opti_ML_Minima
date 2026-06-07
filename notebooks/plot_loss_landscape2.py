import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet20 import ResNet20

import wandb

# ==========================================
# Helpers
# ==========================================

def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def unflatten_params(model, flat_tensor):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_tensor[idx:idx + numel].view_as(p))
        idx += numel

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

def filter_normalized_direction(model):
    """
    Génère une direction aléatoire dans l'espace des paramètres,
    normalisée filter-wise (Li et al. 2018) pour que l'échelle
    soit comparable entre couches de tailles différentes.
    """
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p.data)
        # Normalise chaque filtre par sa norme, mise à l'échelle par la norme du paramètre
        if p.dim() > 1:
            # Pour les tenseurs 2D+ (conv, linear) : normalise filtre par filtre
            for i in range(d.shape[0]):
                d[i] = d[i] / (d[i].norm() + 1e-10) * p.data[i].norm()
        else:
            # Pour les biais 1D : normalise globalement
            d = d / (d.norm() + 1e-10) * p.data.norm()
        direction.append(d.view(-1))
    return torch.cat(direction)

# ==========================================
# 1. Load model from wandb
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

api = wandb.Api()
artifact = api.artifact(
    "fiona-jetzer-epfl/OptiML_Minima/model-resnet20_muon_lr0.02_wd0.0_bs64_cosine_seed1:v0"
)
artifact_dir = Path(artifact.download())
ckpt_path = list(artifact_dir.rglob("*.pt")) + list(artifact_dir.rglob("*.pth"))
print("Fichiers trouvés:", ckpt_path)

model = ResNet20().to(device)
checkpoint = torch.load(ckpt_path[0], map_location=device)

# Adapte selon la structure du checkpoint
if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

theta0 = flatten_params(model)

# ==========================================
# 2. Directions aléatoires filter-normalized
# ==========================================
torch.manual_seed(42)  # reproductibilité
u = filter_normalized_direction(model)
v = filter_normalized_direction(model)

# ==========================================
# 3. Grille d'évaluation
# ==========================================
grid_resolution = 20      # augmente à 40+ pour plus de précision
alpha_range = 1.0         # échelle de l'exploration (à ajuster selon le modèle)

coords = np.linspace(-alpha_range, alpha_range, grid_resolution)
X, Y = np.meshgrid(coords, coords)
Z = np.zeros_like(X)

# ==========================================
# 4. Data
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
criterion = nn.CrossEntropyLoss()

eval_model = ResNet20().to(device)

# ==========================================
# 5. Calcul du landscape
# ==========================================
print("Évaluation du loss landscape...")
for i in range(grid_resolution):
    for j in range(grid_resolution):
        theta_grid = theta0 + X[i, j] * u + Y[i, j] * v
        unflatten_params(eval_model, theta_grid)
        Z[i, j] = min(evaluate_model(eval_model, val_loader, criterion, device), 10)
    print(f"Ligne {i+1}/{grid_resolution} complétée.")

# ==========================================
# 6. Plot
# ==========================================
plt.figure(figsize=(10, 8))
contours = plt.contourf(X, Y, Z, levels=30, cmap='terrain')
plt.colorbar(contours, label='Cross-Entropy Loss')
plt.scatter(0, 0, color='red', marker='*', s=300, zorder=5, label='Minimum (θ₀)')
plt.title('Loss Landscape — ResNet20 (muon lr0.02, cosine, seed1)')
plt.xlabel('Direction u (filter-normalized)')
plt.ylabel('Direction v (filter-normalized)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('loss_landscape_single_model.png', dpi=300)
print("Sauvegardé : loss_landscape_single_model.png")

fig = plt.figure(figsize=(16, 6))
    
# 1. 3D Terrain Plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.9)
ax1.set_title("Filter-Normalized 3D Loss Surface", fontsize=12, fontweight='bold')
ax1.set_xlabel("Direction Vector X (Alpha)")
ax1.set_ylabel("Direction Vector Y (Beta)")
ax1.set_zlabel("Cross-Entropy Loss")
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

# 2. 2D Contour Plot
ax2 = fig.add_subplot(1, 2, 2)
contours = ax2.contourf(X, Y, Z, levels=30, cmap='terrain')
fig.colorbar(contours, ax=ax2)

# The trained model sits perfectly in the center (0,0)
ax2.scatter(0, 0, color='red', s=200, marker='*', edgecolors='black', zorder=5, label='Trained Minimum (0,0)')

ax2.set_title("2D Contour Map (Centered around Trained Model)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Direction Vector X (Alpha)")
ax2.set_ylabel("Direction Vector Y (Beta)")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(f'loss_around_{"model-resnet20_muon_lr0.02_wd0.0_bs64_cosine_seed1"}.png', dpi=300)