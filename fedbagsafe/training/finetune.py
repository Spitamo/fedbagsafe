#!/usr/bin/env python
# coding: utf-8

"""
Training and fine-tuning utilities for FedBagSafe.
"""

import logging
import gc
import torch
import pytorch_lightning as pl
from copy import deepcopy

from ..config import device

# Logging
logger = logging.getLogger("pytorch_lightning")
old_level = logger.level
logger.setLevel(logging.WARNING)


def freeze_except_classifier(model, mode: str):
    """
    Freeze all model parameters except classifier layers.
    
    Args:
        model: Model to freeze
        mode: Dataset mode ('cifar10', 'cifar100', 'fashionmnist', 'femnist')
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the specified classifier layers
    if mode in ['cifar10','cifar100']:
        layers_to_unfreeze = [
            "features.7.0.conv1.weight",
            "features.7.0.conv2.weight", 
            "features.7.0.conv3.weight",
            "features.7.0.downsample.0.weight",
            "features.7.1.conv1.weight",
            "features.7.1.conv2.weight",
            "features.7.1.conv3.weight",
            "features.7.2.conv1.weight",
            "features.7.2.conv2.weight",
            "features.7.2.conv3.weight",
            "fc.1.weight",
            "fc.1.bias"
        ]
    elif mode == 'femnist':
        layers_to_unfreeze = [
            # conv3 layers (mid-to-high level)
            "conv3.0.weight", "conv3.1.weight", "conv3.1.bias",
            
            # conv4 layers (high level)
            "conv4.0.weight", "conv4.1.weight", "conv4.1.bias",
            
            # res2 layers (highest level)
            "res2.0.0.weight", "res2.0.1.weight", "res2.0.1.bias",
            "res2.1.0.weight", "res2.1.1.weight", "res2.1.1.bias", 
            
            # classifier layers
            "classifier.2.weight", "classifier.2.bias",
            "classifier.5.weight", "classifier.5.bias"
        ]

    elif mode == 'fashionmnist':
        layers_to_unfreeze = [
            "conv4.0.weight",
            "conv4.1.weight",
            "conv4.1.bias",
            
            "res2.0.0.weight",
            "res2.0.1.weight",
            "res2.0.1.bias",
            
            "res2.1.0.weight",
            "res2.1.1.weight",
            "res2.1.1.bias",
            
            "classifier.2.weight",
            "classifier.2.bias",
            "classifier.3.weight",
            "classifier.3.bias",
            
            "classifier.6.weight",
            "classifier.6.bias",
            "classifier.7.weight",
            "classifier.7.bias",
            
            "classifier.10.weight",
            "classifier.10.bias"
        ]
    
    for name, param in model.named_parameters():
        if name in layers_to_unfreeze:
            param.requires_grad = True


def get_dataloader_by_state(state, i, n_round,
                            nodes_label_flip_data_loaders, nodes_clean_data_loaders,
                            nodes_model_poison_data_loaders, nodes_data_loaders):
    """
    Get appropriate dataloader based on attack state.
    
    Args:
        state: Attack state
        i: Client index
        n_round: Round number
        nodes_label_flip_data_loaders: Label flip dataloaders
        nodes_clean_data_loaders: Clean dataloaders
        nodes_model_poison_data_loaders: Model poison dataloaders
        nodes_data_loaders: Backdoor dataloaders
    
    Returns:
        Dataloader for the specified configuration
    """
    if state == 'backdoor':
        return nodes_data_loaders[n_round][i]
    elif state == 'modelpoisoning':
        return nodes_model_poison_data_loaders[n_round][i]
    elif state == 'labelflip':
        return nodes_label_flip_data_loaders[n_round][i]
    else:
        return nodes_clean_data_loaders[n_round][i]


def finetune(ple, modelname, model, n_nodes, n_round, state, modelpoisoning=False):
    """
    Fine-tune models for all clients.
    
    Args:
        ple: Tuple containing all dataloaders
        modelname: Dataset name
        model: Global model to fine-tune from
        n_nodes: Number of clients
        n_round: Current round number
        state: Attack state
        modelpoisoning: Whether to apply model poisoning
    
    Returns:
        List of fine-tuned node models
    """
    backdoor_data_loader, \
    eval_data_loader, central_data_loader,\
        nodes_label_flip_data_loaders, nodes_clean_data_loaders,\
            nodes_model_poison_data_loaders, nodes_data_loaders, test_dataset,\
                train_dataset, n_classes = ple
    
    node_models = []
    for i in range(n_nodes):
        node_model = deepcopy(model)
        freeze_except_classifier(node_model, modelname)
        node_models.append(node_model)

    for i, node_model in enumerate(node_models):
        node_model = node_model.to(device)
        # Print status
        node_status = 'Attack' if i < 24 and state != 'benign' else 'Normal'
        print(f'\n=== Fine-tuning Node {i} [{node_status}] | Mode: {state} ===')

        # Get appropriate dataloader
        _dataloader = get_dataloader_by_state(state, i, n_round, nodes_label_flip_data_loaders, nodes_clean_data_loaders, nodes_model_poison_data_loaders, nodes_data_loaders)

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=1.0,
            max_epochs=1,
            accelerator='cpu'
        )
        trainer.fit(node_model, _dataloader, eval_data_loader)

        if i < 24 and modelpoisoning:
            print("mp is activated")
            with torch.no_grad():
                for param in node_model.parameters():
                    # Apply 16.3% Gaussian noise
                    noise = torch.normal(0, 0.5 * torch.std(param), size=param.shape).to(param.device)
                    param.add_(noise)

        node_model = node_model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
    return node_models


# Restore logging
logger.setLevel(old_level)
