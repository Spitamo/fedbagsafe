#!/usr/bin/env python
# coding: utf-8

"""
FedBagSafe - Backward Compatibility Wrapper

This file has been refactored into a modular structure under the 'fedbagsafe' package.
It is kept for backward compatibility and imports from the new modular structure.

New structure:
    fedbagsafe/
    ├── __init__.py          - Package initialization
    ├── config.py            - Configuration and constants
    ├── data/                - Data loading and preprocessing
    │   ├── __init__.py
    │   ├── datasets.py      - Custom dataset classes
    │   └── loaders.py       - Data loading and partitioning
    ├── models/              - Model architectures
    │   ├── __init__.py
    │   ├── architectures.py - Neural network architectures
    │   └── initializers.py  - Model loading utilities
    ├── training/            - Training and aggregation
    │   ├── __init__.py
    │   ├── finetune.py      - Fine-tuning utilities
    │   └── aggregation.py   - Aggregation strategies
    ├── utils/               - Utility functions
    │   ├── __init__.py
    │   └── evaluation.py    - Evaluation metrics
    └── fedbagsafe.py        - Main algorithm

To use the new modular structure, use:
    from fedbagsafe import FEDBAGSAFE
    FEDBAGSAFE('cifar10', 100, 100, mode=False)

Or run: python main.py
"""

import torch

# Import all components from the new modular structure
from fedbagsafe import FEDBAGSAFE, seed_everything, device
from fedbagsafe.config import N_NODES, N_ROUNDS, ATTACK_STATES
from fedbagsafe.data import CustomDataset, DatasetMode, FEMNIST, noniid, dataloader
from fedbagsafe.models import (
    ResNet9CIFAR, ResNet9FashionMNIST, ResNet9FEMNIST,
    load_cifar_model, load_fashionmnist_model, load_femnist_model
)
from fedbagsafe.training import finetune, freeze_except_classifier
from fedbagsafe.training import (
    selective_weighted_average_aggregation,
    random_client_selection, real_random_client_selection
)
from fedbagsafe.utils import evaluate_model_lightning, compute_loss_lightning

# For backward compatibility with variable names
HW = 'cpu'
n_nodes, n_rounds = N_NODES, N_ROUNDS

# Initialize
torch.set_float32_matmul_precision('medium')
seed_everything()

# Main execution - kept for backward compatibility
if __name__ == "__main__":
    configs = ['cifar10', 'cifar100', 'fashionmnist', 'femnist']
    for config in configs:
        FEDBAGSAFE(config, 100, 100, mode=False)
