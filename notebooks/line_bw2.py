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

def plot_loss_landscape_generic(model_class, state_dict_A, state_dict_B, test_loader, criterion, device, title_suffix=""):
    """
    Plots the 1D loss landscape between ANY two configurations of weights.
    """
    print(f"\n--- Starting 1D Loss Landscape Interpolation ({title_suffix}) ---")
    
    # Create the model wrapper that we will inject weights into
    blended_model = model_class().to(device)
    blended_model.eval()
    
    alpha_values = np.linspace(-0.3, 1.3, 300)
    loss_values = []
    
    with torch.no_grad():
        for alpha in alpha_values:
            blended_state = copy.deepcopy(state_dict_A)
            
            for key in blended_state.keys():
                if blended_state[key].is_floating_point():
                    blended_state[key] = (1.0 - alpha) * state_dict_A[key] + alpha * state_dict_B[key]
            
            blended_model.load_state_dict(blended_state)
            
            total_loss = 0.0
            total_batches = 0
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = blended_model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total_batches += 1
                if total_batches >= 20:  # Evaluate on 20 batches for speed
                    break
                    
            avg_loss = total_loss / total_batches
            loss_values.append(avg_loss)
            
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, loss_values, label='Loss Path', color='firebrick', linewidth=2)
    plt.axvline(x=0.0, color='gray', linestyle='--', label='Model A ($\\alpha=0$)')
    plt.axvline(x=1.0, color='gray', linestyle='--', label='Model B ($\\alpha=1$)')
    plt.title(f'1D Loss Landscape: {title_suffix}')
    plt.xlabel('Alpha ($\\alpha$)')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('loss_landscape_2_minima.png', dpi=300)
    plt.show()


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

plot_loss_landscape_generic(ResNet20, best_model2.state_dict(), best_model1.state_dict(), testloader, criterion, device)
