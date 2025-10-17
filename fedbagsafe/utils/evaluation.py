#!/usr/bin/env python
# coding: utf-8

"""
Evaluation utilities for FedBagSafe.
"""

import torch
import logging
from pytorch_lightning import Trainer

from ..config import device


def evaluate_model_lightning(model, dataloader):
    """
    Evaluate model using PyTorch Lightning.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader for evaluation
    
    Returns:
        Accuracy score
    """
    # Disable logging and checkpoints for evaluation
    model = model.to(device)
    logger = logging.getLogger("pytorch_lightning")
    old_level = logger.level
    logger.setLevel(logging.WARNING)

    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator='cpu'
    )

    # Run test
    model.eval()
    test_result = trainer.test(model, dataloaders=dataloader, verbose=False)

    # Restore log level
    logger.setLevel(old_level)

    # Extract accuracy
    if test_result and "test_acc" in test_result[0]:
        acc = test_result[0]["test_acc"]
    else:
        acc = 0.0  #  key is missing

    model = model.to('cpu')
    return acc


def compute_loss_lightning(model, dataloader):
    """
    Compute loss for a model using a dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: Dataloader for evaluation
    
    Returns:
        Average loss
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Temporarily silence logging
    logger = logging.getLogger("pytorch_lightning")
    old_level = logger.level
    logger.setLevel(logging.WARNING)

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(model.device)
            targets = targets.to(model.device)

            outputs = model(inputs)

            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='sum')

            total_loss += loss.item()
            total_samples += targets.size(0)

    logger.setLevel(old_level)

    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    model = model.to('cpu')
    return avg_loss
