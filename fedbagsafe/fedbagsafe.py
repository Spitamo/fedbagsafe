#!/usr/bin/env python
# coding: utf-8

"""
Main FedBagSafe algorithm implementation.
"""

import logging
import gc
import torch
from copy import deepcopy

from .config import ATTACK_STATES, N_NODES, N_ROUNDS
from .data.loaders import dataloader
from .models.initializers import load_cifar_model, load_fashionmnist_model, load_femnist_model
from .training.finetune import finetune
from .training.aggregation import random_client_selection, real_random_client_selection
from .utils.evaluation import evaluate_model_lightning, compute_loss_lightning


def FEDBAGSAFE(modelname: str, n_rounds, n_nodes, mode=False):
    """
    Main FedBagSafe federated learning algorithm.
    
    Args:
        modelname: Dataset name ('cifar10', 'cifar100', 'fashionmnist', 'femnist')
        n_rounds: Number of communication rounds
        n_nodes: Number of clients
        mode: If True, use controlled client selection; if False, use random selection
    """
    logger = logging.getLogger("pytorch_lightning")
    old_level = logger.level
    logger.setLevel(logging.WARNING)

    states = ATTACK_STATES
    ple = dataloader(modelname)
    eval_data_loader = ple[1]
    backdoor_data_loader = ple[0]
    n_classes = ple[9]
    
    if modelname in ['cifar100','cifar10']:
        globalmodel = load_cifar_model(eval_data_loader, n_classes, modelname)
    elif modelname == 'fashionmnist':
        globalmodel = load_fashionmnist_model(eval_data_loader, n_classes)
    elif modelname == 'femnist':
        globalmodel = load_femnist_model(eval_data_loader, n_classes)
    
    model = globalmodel
    
    best_score_prev = 0.0
    
    for round_id in range(n_rounds):
        print(f"\n=== Round {round_id} | State = {states[round_id]} ===")

        # Fine-tune per-node models
        node_models = finetune(
            ple,
            modelname,
            model,
            n_nodes,
            round_id,
            states[round_id],
            modelpoisoning=(states[round_id] == 'modelpoisoning')
        )

        # Aggregate models
        if mode:
            aggregated_models = random_client_selection(node_models, globalmodel, modelname)
        else:
            aggregated_models = real_random_client_selection(node_models, globalmodel, modelname)
            
        del node_models
        gc.collect()

        best_model_this_round = None
        best_score_this_round = 0.0
        best_agg_id = -1
        contributors = []

        for agg_id, (agg_model, contributing_nodes) in enumerate(aggregated_models):
            print(f"\n=== Evaluating Aggregated Model {agg_id} from nodes {contributing_nodes} ===")

            acc = evaluate_model_lightning(agg_model, eval_data_loader)
            loss = compute_loss_lightning(agg_model, eval_data_loader)
            score = acc / loss if loss > 0 else 0.0

            print(f"   → Accuracy: {acc:.4f}")
            print(f"   → Loss: {loss:.4f}")
            print(f"   → Score: {score:.4f}")

            if states[round_id] == 'backdoor':
                b_acc = evaluate_model_lightning(agg_model, backdoor_data_loader)
                print(f"   → Backdoor Accuracy: {b_acc:.4f}")

            if score > best_score_this_round:
                best_model_this_round = deepcopy(agg_model)
                best_score_this_round = score
                best_agg_id = agg_id
                contributors = contributing_nodes

            # Clean memory
            del agg_model
            torch.cuda.empty_cache()
            gc.collect()

        # =======Fail-safe mechanism=======

        # Fail-safe hyperparam
        alpha = .95

        best_score_prev = alpha * best_score_prev
        if best_score_this_round > best_score_prev:
            model = best_model_this_round
            best_score_prev = best_score_this_round
            print(" Model updated.")
            print(f" Best Aggregated Model (Round {round_id}): id {best_agg_id}")
            print(f"   → Score: {best_score_this_round:.4f}")
            print(f"   → Contributors: {contributors}")
        else:
            print(" FAIL-SAFE TRIGGERED: Model not updated.")

        # Final cleanup
        if best_model_this_round is not None:
            del best_model_this_round
            torch.cuda.empty_cache()
            gc.collect()

    # Restore logging level
    logger.setLevel(old_level)
