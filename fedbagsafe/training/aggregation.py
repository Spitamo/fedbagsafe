#!/usr/bin/env python
# coding: utf-8

"""
Model aggregation utilities for FedBagSafe.
"""

import random
import torch
import gc
from copy import deepcopy


def selective_weighted_average_aggregation(node_models, global_model, mode: str):
    """
    Selective weighted average aggregation of node models.
    
    Args:
        node_models: List of node models to aggregate
        global_model: Global model template
        mode: Dataset mode
    
    Returns:
        Aggregated model
    """
    averaged_model = deepcopy(global_model)

    # Define the layers to aggregate
    if mode in ['cifar10','cifar100']:
        layers_to_aggregate = [
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
        layers_to_aggregate = [
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
        layers_to_aggregate = [
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

    # Step 1: Average fine-tuned layers across selected models
    averaged_params = {
        key: torch.mean(torch.stack([node_model.state_dict()[key].float() for node_model in node_models]), dim=0)
        for key in layers_to_aggregate
    }

    # Step 2: Load blended layers into a new model
    new_state_dict = global_model.state_dict()
    for key in layers_to_aggregate:
        new_state_dict[key] = averaged_params[key]

    averaged_model.load_state_dict(new_state_dict)
    return averaged_model


def random_client_selection(node_models, globalmodel, modelname):
    """
    Random client selection with controlled attack aggregation.
    
    Args:
        node_models: List of all node models
        globalmodel: Global model
        modelname: Dataset name
    
    Returns:
        List of aggregated models with their contributors
    """
    n_nodes = 100
    start = 33
    end = 0
    aggregation_configs = []
    sum = 67
    for i in range(start + 1):
        aggregation_configs.append((start - i, sum - (start - i)))
        if start - i == 0: 
            aggregation_configs.append((0, 0))
            break
    aggregated_models = []

    for num_attack, num_normal in aggregation_configs:
        if num_attack == 0 and num_normal == 0:
            # Full aggregation
            selected_indices = random.sample(range(0, n_nodes), n_nodes)
        else:
            # Controlled attack aggregation
            attack_indices = random.sample(range(0, 33), num_attack)
            normal_indices = random.sample(range(33, n_nodes), num_normal)
            selected_indices = attack_indices + normal_indices

        selected_models = [node_models[i] for i in selected_indices]

        # Aggregate selected models
        agg_model = selective_weighted_average_aggregation(selected_models, globalmodel, modelname)
        aggregated_models.append((agg_model, selected_indices))

        # Clear memory
        del selected_models
        torch.cuda.empty_cache()
        gc.collect()

    return aggregated_models


def real_random_client_selection(node_models, globalmodel, modelname):
    """
    Real random client selection for aggregation.
    
    Args:
        node_models: List of all node models
        globalmodel: Global model
        modelname: Dataset name
    
    Returns:
        List of aggregated models with their contributors
    """
    aggregated_models = []
    n_nodes = 100
    for i in range(int(1 / 3 * n_nodes) + 1):
        if i == 0:
            # Full aggregation
            selected_indices = random.sample(range(0, n_nodes), n_nodes)
        else:
            # Partial aggregation ( 2n/3 out of n)
            selected_indices = random.sample(range(0, n_nodes), int(2 / 3 * n_nodes))

        selected_models = [node_models[indice] for indice in selected_indices]

        agg_model = selective_weighted_average_aggregation(selected_models, globalmodel, modelname)
        aggregated_models.append((agg_model, selected_indices))

        # Cleanup after aggregation
        del selected_models
        torch.cuda.empty_cache()
        gc.collect()

    return aggregated_models
