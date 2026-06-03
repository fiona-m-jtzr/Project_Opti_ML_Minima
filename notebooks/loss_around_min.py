import torch

# Used to create and optimize our neural network
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# Used to load and transform our data
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, ToPILImage, Normalize
import torchvision.transforms as transforms
import torchvision

# Used to plot some of the images
import matplotlib.pyplot as plt
import numpy as np

import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

transform = Compose(
    [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

training_data = CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_data = CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

def get_filter_normalized_directions(model, device):
    """
    Generates two random direction vectors (dx, dy) that are normalized 
    filter-by-filter to match the magnitude of the model's actual weights.
    Reference: Li et al., 2018 (arXiv:1712.09913)
    """
    dx_state = copy.deepcopy(model.state_dict())
    dy_state = copy.deepcopy(model.state_dict())
    
    for key, param in model.state_dict().items():
        if param.is_floating_point():
            # 1. Generate raw random directions from a standard normal distribution
            raw_dx = torch.randn_like(param)
            raw_dy = torch.randn_like(param)
            
            # 2. Apply normalization layer-by-layer / filter-by-filter
            # For 2D Convolutional layers: Shape is [output_channels, input_channels, height, width]
            if len(param.shape) == 4: 
                for f in range(param.shape[0]): # Iterate over each filter
                    # Calculate Frobenius norm of the actual filter weights
                    param_norm = torch.norm(param[f])
                    
                    # Rescale the corresponding random direction filter
                    dx_state[key][f] = (raw_dx[f] / (torch.norm(raw_dx[f]) + 1e-8)) * param_norm
                    dy_state[key][f] = (raw_dy[f] / (torch.norm(raw_dy[f]) + 1e-8)) * param_norm
                    
            # For 1D Linear layers (Weights or Biases): Shape is [output_features, input_features] or [features]
            elif len(param.shape) >= 1:
                for row in range(param.shape[0]):
                    param_norm = torch.norm(param[row])
                    dx_state[key][row] = (raw_dx[row] / (torch.norm(raw_dx[row]) + 1e-8)) * param_norm
                    dy_state[key][row] = (raw_dy[row] / (torch.norm(raw_dy[row]) + 1e-8)) * param_norm
            else:
                dx_state[key] = torch.zeros_like(param)
                dy_state[key] = torch.zeros_like(param)
        else:
            # Keep non-floating parameters (like tracking parameters) frozen
            dx_state[key] = torch.zeros_like(param)
            dy_state[key] = torch.zeros_like(param)
            
    return dx_state, dy_state

def plot_single_model_landscape(model_class, trained_model, test_loader, 
                                criterion, device, grid_size=15, range_val=1.0):
    """
    Plots the 2D filter-normalized loss landscape centered around a single trained model.
    """
    print("Generating filter-normalized random direction vectors...")
    trained_model.eval()
    theta_star = trained_model.state_dict()
    
    # Extract the scale-invariant safe directions
    dx, dy = get_filter_normalized_directions(trained_model, device)
    
    # Create coordinate grid steps stepping away from the center (0,0)
    # Step sizes are multiplier coefficients (alpha and beta)
    steps = np.linspace(-range_val, range_val, grid_size)
    X, Y = np.meshgrid(steps, steps)
    Z = np.zeros_like(X)
    
    print(f"Sampling landscape geometry ({grid_size}x{grid_size} = {grid_size**2} points)...")
    eval_model = model_class().to(device)
    eval_model.eval()
    
    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                alpha = X[i, j]
                beta = Y[i, j]
                
                # Blend: Blended_Weights = Center_Weights + alpha * dx + beta * dy
                blended_sd = copy.deepcopy(theta_star)
                for key in blended_sd.keys():
                    if blended_sd[key].is_floating_point():
                        blended_sd[key] = theta_star[key] + (alpha * dx[key].to(device)) + (beta * dy[key].to(device))
                
                eval_model.load_state_dict(blended_sd)
                
                # Compute batch evaluation loss
                total_loss, batches = 0.0, 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = eval_model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    batches += 1
                    if batches >= 20: # Caps compute time per step
                        break
                Z[i, j] = total_loss / batches
            print(f"   -> Row {i+1}/{grid_size} complete.")

    # --- PLOTTING ---
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
    plt.show()