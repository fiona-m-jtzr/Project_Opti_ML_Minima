# Project_Opti_ML_Minima

# Loss Landscape Analysis: SGD vs. Adam vs. Muon vs. SAM on ResNet20 and ViT

This repository contains the code and configuration files used to analyze and compare the loss landscapes around stationary points discovered by different optimization algorithms: **SGD**, **Adam**, **Muon**, and **SAM**.

We perform our experiment on the CIFAR10 dataset and use ResNet20 and a simple ViT as model architectures.

## Project Overview
The file RUN.py provides the training loop, which reproduces our results if launched with the hyperparameter configurations provided in the files located in the folder sweeps. The training has been performed on the EPFL RCP Cluster.

### Optimizer Implementations
- **SGD & Adam:** Standard `torch.optim` implementations.
- **Muon:** Sourced from [KellerJordan/Muon](https://github.com/KellerJordan/Muon).
- **SAM (Sharpness-Aware Minimization):** Sourced from [davda54/sam](https://github.com/davda54/sam).

---

## Installation & Setup

All the necessary python installations ar listed in the requirements.txt file.

## AI Disclaimer

Different LLMs (Claude, Gemini and ChatGPT) have been used to produce the code contained in this repository.
