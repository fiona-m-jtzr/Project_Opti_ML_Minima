import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
from torchvision import transforms
import torchvision
from models.resnet20 import ResNet20
import wandb
from pathlib import Path

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

def compute_loss_landscape(
    model,
    dataloader,
    criterion,
    device,
    grid_size=25,
    alpha_range=(-1.0, 1.0),
    beta_range=(-1.0, 1.0),
    n_batches=10,
):
    """
    Visualise le loss landscape autour d'un minima (Li et al. 2018).

    Les deux directions de perturbation sont filter-normalized :
    chaque filtre est normalisé pour avoir la même norme que le filtre
    correspondant dans le modèle entraîné. Cela rend les axes comparables
    entre couches de tailles différentes.

    Args:
        model       : ResNet20 déjà entraîné, en eval mode
        dataloader  : DataLoader (train ou test)
        criterion   : nn.CrossEntropyLoss()
        device      : torch.device
        grid_size   : résolution de la grille (grid_size x grid_size points)
        alpha_range : intervalle pour la direction δ₁  (ex. (-1, 1))
        beta_range  : intervalle pour la direction δ₂  (ex. (-1, 1))
        n_batches   : nombre de mini-batches utilisés pour estimer la loss
                      (plus c'est grand, plus c'est précis mais lent)

    Returns:
        alphas, betas : grilles 1-D (np.ndarray)
        Z             : matrice de loss (grid_size x grid_size)
    """
    model.eval()
    theta_star = [p.data.clone() for p in model.parameters()]

    # ── Génération de deux directions aléatoires filter-normalized ──────────
    # def random_direction(params):
    #     """
    #     Tire une direction δ de même forme que params,
    #     puis la normalise filtre par filtre (ou neurone par neurone pour fc).
    #     """
    #     direction = []
    #     for p in params:
    #         d = torch.randn_like(p)
    #         if p.dim() >= 2:
    #             # Conv/Linear : normalise chaque filtre (dim 0) individuellement
    #             p_norm = p.norm(dim=list(range(1, p.dim())), keepdim=True)
    #             d_norm = d.norm(dim=list(range(1, d.dim())), keepdim=True)
    #             d = d / (d_norm + 1e-10) * p_norm
    #         else:
    #             # BN biais / weight : scalaire, normalise globalement
    #             d = d / (d.norm() + 1e-10) * p.norm()
    #         direction.append(d)
    #     return direction

    def random_direction(params):
        direction = []
        for p in params:
            d = torch.randn_like(p)
            if p.dim() >= 2:
                # Normalise filtre par filtre (dim 0 = filtre)
                reduce_dims = list(range(1, p.dim()))
                p_norm = torch.linalg.norm(p.flatten(1), dim=1, keepdim=True)
                d_norm = torch.linalg.norm(d.flatten(1), dim=1, keepdim=True)
                # Reshape pour broadcaster sur toutes les dims sauf dim 0
                shape = [-1] + [1] * (p.dim() - 1)
                p_norm = p_norm.view(shape)
                d_norm = d_norm.view(shape)
                d = d / (d_norm + 1e-10) * p_norm
            else:
                d = d / (d.norm() + 1e-10) * p.norm()
            direction.append(d)
        return direction

    delta1 = random_direction(theta_star)
    delta2 = random_direction(theta_star)

    # ── Évaluation de la loss sur n_batches ──────────────────────────────────
    def evaluate_loss(model, dataloader, criterion, device, n_batches):
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for i, (X, y) in enumerate(dataloader):
                if i >= n_batches:
                    break
                X, y = X.to(device), y.to(device)
                total_loss += criterion(model(X), y).item()
                count += 1
        return total_loss / max(count, 1)

    # ── Balayage de la grille ────────────────────────────────────────────────
    alphas = np.linspace(*alpha_range, grid_size)
    betas  = np.linspace(*beta_range,  grid_size)
    Z      = np.zeros((grid_size, grid_size))

    perturbed = copy.deepcopy(model)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # θ = θ* + α·δ₁ + β·δ₂
            for k, p in enumerate(perturbed.parameters()):
                p.data.copy_(
                    theta_star[k] + alpha * delta1[k] + beta * delta2[k]
                )
            Z[i, j] = evaluate_loss(perturbed, dataloader, criterion, device, n_batches)
            print(Z[i, j])
        print(f"  Ligne {i+1}/{grid_size} calculée…", end="\r")

    print("\nGrille complète.")
    return alphas, betas, Z


def plot_loss_landscape(alphas, betas, Z, title="Loss landscape"):
    """
    Génère le plot 3-D (surface) et le contour 2-D côte à côte.
    """
    A, B = np.meshgrid(alphas, betas, indexing="ij")

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(title, fontsize=13, y=1.01)

    # ── Surface 3-D ──────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(
        A, B, Z,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    ax3d.set_xlabel("α  (direction δ₁)", labelpad=8)
    ax3d.set_ylabel("β  (direction δ₂)", labelpad=8)
    ax3d.set_zlabel("Loss", labelpad=8)
    ax3d.set_title("Surface 3-D", fontsize=11)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1, label="Loss")

    # ── Contour 2-D ──────────────────────────────────────────────────────────
    ax2d = fig.add_subplot(122)
    levels = np.linspace(Z.min(), Z.min() + (Z.max() - Z.min()) * 0.95, 30)
    cf = ax2d.contourf(A, B, Z, levels=levels, cmap="coolwarm")
    cs = ax2d.contour(A, B, Z, levels=levels[::3], colors="white", linewidths=0.5, alpha=0.4)
    ax2d.clabel(cs, fmt="%.2f", fontsize=7, inline=True)

    # Marque le minima (alpha=0, beta=0 = θ*)
    ax2d.scatter(0, 0, color="gold", s=80, zorder=5, label="θ* (minima)")
    ax2d.set_xlabel("α  (direction δ₁)")
    ax2d.set_ylabel("β  (direction δ₂)")
    ax2d.set_title("Contour 2-D", fontsize=11)
    ax2d.legend(loc="upper right", fontsize=9)
    fig.colorbar(cf, ax=ax2d, label="Loss")

    plt.tight_layout()
    plt.savefig("loss_landscape.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Sauvegardé : loss_landscape.png")


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet20().to(device)

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


    # model.eval()

    # alphas, betas, Z = compute_loss_landscape(
    #     model       = model,
    #     dataloader  = testloader,
    #     criterion   = criterion,
    #     device      = device,
    #     grid_size   = 30,
    #     alpha_range = (-1.0, 1.0),
    #     beta_range  = (-1.0, 1.0),
    #     n_batches   = 10,
    # )
    # plot_loss_landscape(alphas, betas, Z)