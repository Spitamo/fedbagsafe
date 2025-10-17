#!/usr/bin/env python
# coding: utf-8

"""
Main entry point for FedBagSafe framework.

This script runs the FedBagSafe algorithm on multiple datasets.
"""

import torch
from fedbagsafe import FEDBAGSAFE, seed_everything

# Set precision
torch.set_float32_matmul_precision('medium')

# Initialize random seeds for reproducibility
seed_everything()

# Run experiments on all datasets
configs = ['cifar10', 'cifar100', 'fashionmnist', 'femnist']
for config in configs:
    print(f"\n{'='*80}")
    print(f"Starting FedBagSafe experiment on {config.upper()}")
    print(f"{'='*80}\n")
    FEDBAGSAFE(config, 100, 100, mode=False)
