#!/usr/bin/env python
# coding: utf-8

"""
Model initialization utilities for FedBagSafe.
"""

import torch
import pytorch_lightning as pl

from .architectures import ResNet9CIFAR, ResNet9FashionMNIST, ResNet9FEMNIST


def load_cifar_model(eval_data_loader, n_classes, modelname):
    """
    Load pretrained CIFAR model.
    
    Args:
        eval_data_loader: Dataloader for evaluation
        n_classes: Number of classes
        modelname: 'cifar10' or 'cifar100'
    
    Returns:
        Loaded model
    """
    globalmodel = ResNet9CIFAR(num_classes=n_classes)
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='cpu'
    )
    if modelname == 'cifar100':
        globalmodel = ResNet9CIFAR.load_from_checkpoint('cifar100.ckpt', num_classes=n_classes)
    else: 
        globalmodel.load_state_dict(torch.load('cifar10.pth'))
    trainer.test(globalmodel, eval_data_loader)

    return globalmodel


def load_fashionmnist_model(eval_data_loader, n_classes):
    """
    Load pretrained Fashion-MNIST model.
    
    Args:
        eval_data_loader: Dataloader for evaluation
        n_classes: Number of classes
    
    Returns:
        Loaded model
    """
    globalmodel = ResNet9FashionMNIST(num_classes=n_classes)
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='cpu'
    )

    globalmodel = ResNet9FashionMNIST.load_from_checkpoint('fashionmnist1.ckpt', num_classes=n_classes)
    trainer.test(globalmodel, eval_data_loader) 
    return globalmodel


def load_femnist_model(eval_data_loader, n_classes):
    """
    Load pretrained FEMNIST model.
    
    Args:
        eval_data_loader: Dataloader for evaluation
        n_classes: Number of classes
    
    Returns:
        Loaded model
    """
    globalmodel = ResNet9FEMNIST(num_classes=n_classes)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='cpu',
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=0.5
    )
    globalmodel = ResNet9FEMNIST.load_from_checkpoint('femnist-stable-epoch=05-val_acc=0.67.ckpt', num_classes=n_classes)

    trainer.test(globalmodel, eval_data_loader) 
    return globalmodel
