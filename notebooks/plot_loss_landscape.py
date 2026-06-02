import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet20 import ResNet20

# Assuming ResNet20 and BasicBlock classes are already defined as provided

def flatten_params(model):
    """Flattens all parameters of a model into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def unflatten_params(model, flat_tensor):
    """Overwrites model parameters using a flattened 1D tensor."""
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_tensor[idx:idx + numel].view_as(p))
        idx += numel

def evaluate_model(model, dataloader, criterion, device):
    """Computes the average loss over the given dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Scale by batch size to compute exact average loss later
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
    return total_loss / total_samples

# ==========================================
# 1. Setup and Load Your 3 Models
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('device : ', device)

# Placeholder: Replace these initialization blocks with your exact file paths
# (e.g., repeating your loading logic for best0, best1, and best2)
model0 = ResNet20().to(device)
model1 = ResNet20().to(device)
model2 = ResNet20().to(device)

# Example snippet for loading (adjust filenames to match your three files):
model0.load_state_dict(torch.load('notebooks/best3.pt', map_location=device)['model'])
model1.load_state_dict(torch.load('notebooks/best4.pt', map_location=device)['model'])
model2.load_state_dict(torch.load('notebooks/best5.pt', map_location=device)['model'])

# Flatten configurations
theta0 = flatten_params(model0)
theta1 = flatten_params(model1)
theta2 = flatten_params(model2)

# ==========================================
# 2. Define Plane Bases (u, v)
# ==========================================
# Origin is theta0. The axes span towards theta1 and theta2.
u = theta1 - theta0
v = theta2 - theta0

# ==========================================
# 3. Create Evaluation Grid
# ==========================================
# We use a resolution of 15x15 for quick computation. Increase to 40+ for a smoother map.
grid_resolution = 20
x_coords = np.linspace(-0.2, 1.2, grid_resolution)
y_coords = np.linspace(-0.2, 1.2, grid_resolution)
X, Y = np.meshgrid(x_coords, y_coords)
Z = np.zeros_like(X)

# ==========================================
# 4. Prepare Validation/Test Data & Loss
# ==========================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Use validation data for faster landscape evaluation
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# Adjust batch size depending on available GPU memory
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()

# Create a temporary model runner so we don't ruin our original model configurations
eval_model = ResNet20().to(device)

# ==========================================
# 5. Compute the Landscape Losses
# ==========================================
print("Evaluating the loss landscape grid...")
for i in range(grid_resolution):
    for j in range(grid_resolution):
        x = X[i, j]
        y = Y[i, j]
        
        # Calculate target coordinates in parameter space
        theta_grid = theta0 + x * u + y * v
        
        # Load parameters into the evaluation model and calculate loss
        unflatten_params(eval_model, theta_grid)
        loss = evaluate_model(eval_model, val_loader, criterion, device)
        Z[i, j] = min(loss, 10)
    print(f"Row {i+1}/{grid_resolution} completed.")

# ==========================================
# 6. Plotting the Results
# ==========================================
plt.figure(figsize=(10, 8))

# Draw the background loss contour
contours = plt.contourf(X, Y, Z, levels=30, cmap='terrain')
plt.colorbar(contours, label='Cross-Entropy Loss')

# Plot the coordinates of the 3 minimas
# theta0 is at (0,0), theta1 is at (1,0), theta2 is at (0,1)
plt.scatter(0, 0, color='red', marker='*', s=200, label='Model 0 Minima')
plt.scatter(1, 0, color='blue', marker='*', s=200, label='Model 1 Minima')
plt.scatter(0, 1, color='magenta', marker='*', s=200, label='Model 2 Minima')

plt.title('2D Loss Landscape Spanning Three Minimas (ResNet20)')
plt.xlabel('Basis Vector U (Direction of Model 1)')
plt.ylabel('Basis Vector V (Direction of Model 2)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('loss_landscape_3_minima.png', dpi=300)
#plt.show()