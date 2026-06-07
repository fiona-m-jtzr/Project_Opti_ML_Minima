import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
from models.resnet20 import ResNet20

# ==========================================
# 1. CONFIGURATION & CHARGEMENT DES DONNÉES
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# On utilise le testloader pour aller plus vite (le trainloader prendrait trop de temps par point)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

# ==========================================
# 2. CHARGEMENT DU MODÈLE (Votre Code)
# ==========================================
# Assurez-vous que la classe ResNet20 et BasicBlock sont définies plus haut dans votre script
api = wandb.Api()
artifact = api.artifact("fiona-jetzer-epfl/OptiML_Minima/model-resnet20_muon_lr0.02_wd0.0_bs64_cosine_seed1:v0")
artifact_dir = Path(artifact.download())
ckpt_path = (list(artifact_dir.rglob("*.pt")) + list(artifact_dir.rglob("*.pth")))[0]

model = ResNet20().to(device)
checkpoint = torch.load(ckpt_path, map_location=device)

# Ajustement si les clés du state_dict contiennent 'module.' (DataParallel)
state_dict = checkpoint["model"]
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

# Stocker les poids du minimum trouvé (W*)
target_weights = [p.data.clone() for p in model.parameters()]

# ==========================================
# 3. CRÉATION DES DIRECTIONS ALÉATOIRES (Filter-wise)
# ==========================================
def create_random_direction(model):
    """Crée une direction aléatoire avec la même structure que les paramètres du modèle."""
    direction = []
    for p in model.parameters():
        d = torch.randn_like(p)
        # Normalisation par filtre (indispensable pour les ResNets)
        if len(p.shape) >= 2: # Conv or Linear layers
            for i in range(p.shape[0]):
                d_filter = d[i]
                p_filter = p[i]
                d_filter.mul_(p_filter.norm() / (d_filter.norm() + 1e-10))
        else: # Bias, BatchNorm
            d.mul_(p.norm() / (d.norm() + 1e-10))
        direction.append(d)
    return direction

dir_x = create_random_direction(model)
dir_y = create_random_direction(model)

# ==========================================
# 4. FONCTION D'ÉVALUATION DE LA LOSS
# ==========================================
def evaluate_loss(model, testloader, criterion):
    """Calcule la loss moyenne sur le dataset."""
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

# ==========================================
# 5. CALCUL DE LA GRILLE (SURFACE)
# ==========================================
# Résolution de la grille (ex: 11x11 ou 21x21). Plus c'est grand, plus c'est précis mais long.
grid_resolution = 15 
steps_x = np.linspace(-0.5, 0.5, grid_resolution)
steps_y = np.linspace(-0.5, 0.5, grid_resolution)

X, Y = np.meshgrid(steps_x, steps_y)
Z = np.zeros_like(X)

print("Calcul du loss landscape en cours...")
for i, x_coeff in enumerate(steps_x):
    for j, y_coeff in enumerate(steps_y):
        # Appliquer la perturbation : W = W* + x*dir_x + y*dir_y
        for p, w_start, dx, dy in zip(model.parameters(), target_weights, dir_x, dir_y):
            p.data = w_start + x_coeff * dx + y_coeff * dy
        
        # Calculer la loss à cette coordonnée
        loss_val = evaluate_loss(model, testloader, criterion)
        # Z[j, i] = np.log10(loss_val) # Attention à l'indexation (y=lignes, x=colonnes)
        Z[j, i] = min(loss_val, 10) # Attention à l'indexation (y=lignes, x=colonnes)
        
    print(f"Ligne {i+1}/{grid_resolution} terminée.")

# Restaurer les poids d'origine du modèle
for p, w_start in zip(model.parameters(), target_weights):
    p.data = w_start

# ==========================================
# 6. VISUALISATION GRAPHIQUE
# ==========================================
fig = plt.figure(figsize=(14, 6))

# --- Graphique 2D (Contour) ---
ax1 = fig.add_subplot(1, 2, 1)
contour = ax1.contourf(X, Y, Z, levels=30, cmap='viridis')
fig.colorbar(contour, ax=ax1, label='Loss')
ax1.scatter(0, 0, color='red', marker='*', s=150, label='Minimum trouvé')
ax1.set_title('Loss Landscape 2D (Contours)')
ax1.set_xlabel('Direction X')
ax1.set_ylabel('Direction Y')
ax1.legend()

# --- Graphique 3D (Surface) ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Loss')

# Marquer le minimum en 3D
min_loss = Z[grid_resolution//2, grid_resolution//2]
ax2.scatter(0, 0, min_loss, color='red', marker='*', s=150, zorder=10)

ax2.set_title('Loss Landscape 3D')
ax2.set_xlabel('Direction X')
ax2.set_ylabel('Direction Y')
ax2.set_zlabel('Loss')

# Optimiser l'affichage de l'angle 3D
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig("resnet20_loss_landscape.png", dpi=300)
plt.show()