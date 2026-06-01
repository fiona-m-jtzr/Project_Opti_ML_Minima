import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys

# Get the absolute path of the current notebook's directory
current_dir = os.path.abspath("")

# Get the path of the parent directory (the project root)
project_root = os.path.dirname(current_dir)

# Add the project root to the python path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)

# Now you can import your model
from models.resnet20 import ResNet20  # Assuming your class/function inside is named ResNet20

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def flatten_weights_vector(model):
    """Flattens all trainable parameters of a model into a single 1D vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

def unflatten_weights_vector(vector, model_blueprint):
    """Converts a flat 1D vector back into a PyTorch state_dict structure."""
    state_dict = copy.deepcopy(model_blueprint.state_dict())
    current_index = 0
    for key, param in model_blueprint.named_parameters():  # named_parameters() only yields trainable params
        if param.requires_grad:
            num_elements = param.numel()
            flat_slice = vector[current_index:current_index + num_elements]
            state_dict[key] = flat_slice.view(param.size())
            current_index += num_elements
    return state_dict

def plot_3_model_bounding_rectangle(model_class, model1, model2, model3,
                                    test_loader, criterion, device, grid_size=15):
    """
    Finds the optimal 2D plane containing 3 models using PCA, maps their true locations,
    and samples a rectangular grid that completely bounds all 3 models naturally.
    """
    print("Extracting and flattening weight vectors for all 3 models...")
    w1 = flatten_weights_vector(model1).cpu()
    w2 = flatten_weights_vector(model2).cpu()
    w3 = flatten_weights_vector(model3).cpu()
    
    # 1. Stack vectors and compute the center of mass (mean model vector)
    W = torch.stack([w1, w2, w3])  # Shape: (3, num_params)
    w_mean = torch.mean(W, dim=0)
    W_centered = W - w_mean
    
    # 2. Perform SVD to extract the top 2 PCA directions
    _, _, V = torch.svd(W_centered, some=True)
    dx = V[:, 0].to(device)
    dy = V[:, 1].to(device)
    w_mean = w_mean.to(device)
    
    # 3. Project each model onto our new 2D PCA coordinate system
    def get_2d_coord(w_vec):
        w_dev = w_vec.to(device)
        x = torch.dot(w_dev - w_mean, dx).item()
        y = torch.dot(w_dev - w_mean, dy).item()
        return x, y

    m1_coord = get_2d_coord(w1)
    m2_coord = get_2d_coord(w2)
    m3_coord = get_2d_coord(w3)
    
    all_coords = np.array([m1_coord, m2_coord, m3_coord])
    
    # 4. Define the boundaries of the containing rectangle (with a 20% margin)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    x_margin = (x_max - x_min) * 0.2 if x_max != x_min else 1.0
    y_margin = (y_max - y_min) * 0.2 if y_max != y_min else 1.0
    
    x_range = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)
    y_range = np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
    
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    print(f"Sampling rectangular loss plane ({grid_size}x{grid_size} = {grid_size**2} points)...")
    eval_model = model_class().to(device)
    eval_model.eval()
    
    min_loss = np.inf
    max_loss = 0
    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                x_val = X[i, j]
                y_val = Y[i, j]
                
                w_blended = w_mean + (x_val * dx) + (y_val * dy)
                state_dict = unflatten_weights_vector(w_blended, eval_model)
                eval_model.load_state_dict(state_dict)
                
                total_loss, batches = 0.0, 0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = eval_model(inputs)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    batches += 1
                    if batches >= 20:
                        break
                Z[i, j] = min(total_loss / batches, 6)
                #max_loss = max(max_loss, total_loss / batches)
                #min_loss = min(min_loss, total_loss / batches)
            print(f"   -> Row {i+1}/{grid_size} processed.", max_loss, min_loss)

    # --- MATPLOTLIB RENDERING ENGINE ---
    fig = plt.figure(figsize=(16, 6))
    
    # 1. 3D Terrain Plot
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', alpha=0.8)
    ax1.set_title("3D Loss Surface over Bounding Rectangle", fontsize=12, fontweight='bold')
    ax1.set_xlabel("PCA Axis X")
    ax1.set_ylabel("PCA Axis Y")
    ax1.set_zlabel("Loss")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    # 2. 2D Contour Plot
    ax2 = fig.add_subplot(1, 2, 2)
    contours = ax2.contourf(X, Y, Z, levels=25, cmap='terrain')
    fig.colorbar(contours, ax=ax2)
    
    # Scatter plot the actual positions of the 3 models inside the rectangle
    ax2.scatter(m1_coord[0], m1_coord[1], color='red',     s=150, marker='*', edgecolors='black', zorder=5, label='Model 1')
    ax2.scatter(m2_coord[0], m2_coord[1], color='blue',    s=120, marker='o', edgecolors='black', zorder=5, label='Model 2')
    ax2.scatter(m3_coord[0], m3_coord[1], color='magenta', s=120, marker='s', edgecolors='black', zorder=5, label='Model 3')
    
    # Draw the explicit boundary box containing them
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, color='black', linestyle='--', linewidth=1.5, label='True Bounding Box')
    ax2.add_patch(rect)
    
    ax2.set_title("2D Loss Contour & True Model Projections", fontsize=12, fontweight='bold')
    ax2.set_xlabel("PCA Axis X")
    ax2.set_ylabel("PCA Axis Y")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('notebooks/mon_loss_landscape.png', bbox_inches='tight')

best0 = torch.load('notebooks/best0.pt', map_location=device)
random_model0 = ResNet20() 
best_model0 = ResNet20() 
state_dict0 = best0.get('model')
best_model0.load_state_dict(state_dict0)
best_model0 = best_model0.to(device)

best1 = torch.load('notebooks/best1.pt', map_location=device)
random_model1 = ResNet20() 
best_model1 = ResNet20() 
state_dict1 = best1.get('model')
best_model1.load_state_dict(state_dict1)
best_model1 = best_model1.to(device)

best2 = torch.load('notebooks/best2.pt', map_location=device)
random_model2 = ResNet20() 
best_model2 = ResNet20() 
state_dict2 = best2.get('model')
best_model2.load_state_dict(state_dict2)
best_model2 = best_model2.to(device)

plot_3_model_bounding_rectangle(ResNet20, best_model0, best_model1, best_model2, testloader, criterion, device)