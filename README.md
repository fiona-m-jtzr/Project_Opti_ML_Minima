# Project_Opti_ML_Minima

# Loss Landscape Analysis: SGD vs. Adam vs. Muon vs. SAM on ResNet20 and ViT

This repository contains the code and configuration files used to analyze and compare the loss landscapes around stationary points discovered by different optimization algorithms: **SGD**, **Adam**, **Muon**, and **SAM**.

We perform our experiment on the CIFAR10 dataset and use ResNet20 and a simple ViT as model architectures.

The training, analysis and visualization scripts can be run independently and interface through W&B artifacts, in which model weights and analysis results are stored.

## Project Overview
The file RUN.py provides the training loop, which reproduces our results if launched with the hyperparameter configurations provided in the files located in the folder sweeps. The training has been performed on the EPFL RCP Cluster. Results and models are logged to Weights & Biases (W&B).

### Optimizer Implementations
- **SGD & Adam:** Standard `torch.optim` implementations.
- **Muon:** Sourced from [KellerJordan/Muon](https://github.com/KellerJordan/Muon).
- **SAM (Sharpness-Aware Minimization):** Sourced from [davda54/sam](https://github.com/davda54/sam).

---

### Analyzer
The analysis.py script performs a comprehensive post-training analysis of CIFAR-10 classification models stored as Weights & Biases (W&B) artifacts. Given a trained checkpoint, it reconstructs the corresponding model architecture, evaluates train and test performance, computes the full-dataset gradient norm, estimates Hessian-based curvature statistics, and measures multiple notions of loss-landscape sharpness. These include the 5 largest Hessian eigenvalues and element-wise adaptive sharpness.

The goal of the analysis is to characterize the geometry of the optimization minimum reached during training and to compare different optimization methods, architectures, or hyperparameter configurations. Results are saved as a structured JSON file and automatically logged back to W&B as a versioned analysis artifact, enabling reproducible large-scale studies of generalization, flatness, curvature, and optimization dynamics.

The hessian estimation code is taken from the [PyHessian](https://github.com/amirgholami/pyhessian) framework and the element-wise adaptive sharpness code from the code published with the paper [A Modern Look at the Relationship between Sharpness and Generalization](https://github.com/tml-epfl/sharpness-vs-generalization).

## Usage

Analyze the latest version of a model artifact:

```bash
python analyzer.py \
    --run_name model-FINAL_MODEL_resnet20_sgd_mom0.9_nesterov_lr0.1_wd0.0_bs128_cosine_seed1
```

Analyze a specific artifact version:

```bash
python analyzer.py \
    --run_name model-FINAL_MODEL_resnet20_sgd_mom0.9_nesterov_lr0.1_wd0.0_bs128_cosine_seed1 \
    --artifact_alias v12
```

Customize adaptive sharpness evaluation:

```bash
python analyzer.py \
    --run_name <run_name> \
    --adaptive_sharpness_rhos 1e-4 5e-4 1e-3 2e-3 \
    --adaptive_sharpness_steps 50
```

The generated analysis report contains performance metrics, gradient statistics, Hessian eigenvalue and trace estimates, sampled sharpness curves, adaptive sharpness measurements, and metadata describing the analyzed checkpoint.

##@ Loss Landscape Visualization
The loss landscape is visualized around a trained minimum W* by perturbing the model weights along two directions derived from the Hessian: the eigenvectors corresponding to the 1st and 5th largest eigenvalues. For each point (α, β) on a 2D grid, weights are set to W* + α·v1 + β·v2 and the loss is evaluated on a fixed subset of the training data. The resulting surface is rendered both as a static 2D contour + 3D surface plot (matplotlib) and as an interactive 3D visualization (Plotly).
## Usage
```bash
python -m notebooks.train_loss_landscape:vitadam
```
or 
```bash
python -m notebooks.train_loss_landscape:vitmuon
```
To visualize the loss landscape of a different model, update the `model_name` variable with the corresponding W&B artifact path.

## Installation & Setup

All the necessary python installations are listed in the requirements.txt file. If "cuda" is available, the code will run on GPU (recommended), otherwise it will run on CPU. Furthermore, it is required that the user is logged in to W&B. The W&B project name is hardcoded as "OptiML_Minima".

## AI Disclaimer

Different LLMs (Claude, Gemini and ChatGPT) have been used in the production of the code contained in this repository.
