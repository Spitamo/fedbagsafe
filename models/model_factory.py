import torch
import os
from pathlib import Path
from models.resnet9 import ResNet9
from models.cifar_resnet import CIFARResNet
import pytorch_lightning as pl


class ModelFactory:
    """Factory for creating and loading models"""

    CHECKPOINT_PATHS = {
        'femnist': 'femnist-stable-epoch=05-val_acc=0.67.ckpt',
        'fashionmnist': 'fashionmnist1.ckpt',
        'cifar100': 'cifar100.ckpt',
        'cifar10': 'cifar10.pth'
    }

    @staticmethod
    def create_model(model_name: str, num_classes: int, 
                    dataset_mode: str = None, dropout: float = 0.3):
        """Create model based on name and dataset"""
        if model_name == 'resnet9':
            if dataset_mode == 'femnist':
                from models.resnet9_femnist import ResNet9Femnist
                return ResNet9Femnist(num_classes=num_classes, dropout=dropout)
            elif dataset_mode == 'fashionmnist':
                from models.resnet9_fashionmnist import ResNet9FashionMNIST
                return ResNet9FashionMNIST(num_classes=num_classes, dropout=dropout)
            elif dataset_mode in ['cifar10', 'cifar100']:
                return CIFARResNet(num_classes=num_classes)
            else:
                return ResNet9(num_classes=num_classes, dropout=dropout)
        elif model_name == 'cifar_resnet':
            return CIFARResNet(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def load_pretrained_model(
        model_name: str,
        dataset_mode: str,
        num_classes: int,
        checkpoint_dir: str = './checkpoints',
        dropout: float = 0.3
    ):
        """Load pretrained model from checkpoint"""
        model = ModelFactory.create_model(model_name, num_classes, dataset_mode, dropout)

        checkpoint_path = ModelFactory._get_checkpoint_path(
            dataset_mode,
            checkpoint_dir
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                if checkpoint_path.endswith('.ckpt'):
                    if dataset_mode == 'femnist':
                        from models.resnet9_femnist import ResNet9Femnist
                        model = _load_lightning_checkpoint(
                            ResNet9Femnist,
                            checkpoint_path,
                            num_classes,
                            dropout
                        )
                    elif dataset_mode == 'fashionmnist':
                        from models.resnet9_fashionmnist import ResNet9FashionMNIST
                        model = _load_lightning_checkpoint(
                            ResNet9FashionMNIST,
                            checkpoint_path,
                            num_classes,
                            dropout
                        )
                    elif dataset_mode in ['cifar10', 'cifar100']:
                        model = _load_lightning_checkpoint(
                            CIFARResNet,
                            checkpoint_path,
                            num_classes,
                            dropout
                        )
                else:
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    model.load_state_dict(state_dict)

                print(f" Loaded pretrained model from {checkpoint_path}")
                return model

            except Exception as e:
                print(f" Failed to load checkpoint from {checkpoint_path}: {e}")
                print(" Using randomly initialized model instead")
                return model
        else:
            print(f" Checkpoint not found at {checkpoint_path}")
            print(" Using randomly initialized model instead")
            return model

    @staticmethod
    def _get_checkpoint_path(dataset_mode: str, checkpoint_dir: str) -> str:
        """Get checkpoint path for dataset"""
        checkpoint_name = ModelFactory.CHECKPOINT_PATHS.get(dataset_mode)

        if not checkpoint_name:
            return None

        possible_paths = [
            os.path.join(checkpoint_dir, checkpoint_name),
            checkpoint_name,
            os.path.join('.', checkpoint_name),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return possible_paths[0]

    @staticmethod
    def get_layers_to_freeze(model_name: str = None, dataset_mode: str = None):
        """Get layers to freeze - based primarily on dataset_mode"""
        if not dataset_mode:
            return []
        
        if dataset_mode in ['cifar10', 'cifar100']:
            return [
                "features.7.0.conv1.weight", "features.7.0.bn1.weight", "features.7.0.bn1.bias",
                "features.7.0.conv2.weight", "features.7.0.bn2.weight", "features.7.0.bn2.bias",
                "features.7.0.conv3.weight", "features.7.0.bn3.weight", "features.7.0.bn3.bias",
                "features.7.0.downsample.0.weight", "features.7.0.downsample.1.weight", "features.7.0.downsample.1.bias",
                "features.7.1.conv1.weight", "features.7.1.bn1.weight", "features.7.1.bn1.bias",
                "features.7.1.conv2.weight", "features.7.1.bn2.weight", "features.7.1.bn2.bias",
                "features.7.1.conv3.weight", "features.7.1.bn3.weight", "features.7.1.bn3.bias",
                "features.7.2.conv1.weight", "features.7.2.bn1.weight", "features.7.2.bn1.bias",
                "features.7.2.conv2.weight", "features.7.2.bn2.weight", "features.7.2.bn2.bias",
                "features.7.2.conv3.weight", "features.7.2.bn3.weight", "features.7.2.bn3.bias",
                "fc.1.weight", "fc.1.bias"
            ]
        elif dataset_mode == 'femnist':
            return [
                "conv3.0.weight", "conv3.1.weight", "conv3.1.bias",
                "conv4.0.weight", "conv4.1.weight", "conv4.1.bias",
                "res2.0.0.weight", "res2.0.1.weight", "res2.0.1.bias",
                "res2.1.0.weight", "res2.1.1.weight", "res2.1.1.bias",
                "classifier.2.weight", "classifier.2.bias",
                "classifier.5.weight", "classifier.5.bias"  
            ]
        elif dataset_mode == 'fashionmnist':
            return [
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
        else:
            return []

    @staticmethod
    def get_layers_to_aggregate(model_name: str = None, dataset_mode: str = None):
        """get layers for aggregation - same as freeze layers"""
        return ModelFactory.get_layers_to_freeze(model_name, dataset_mode)

    @staticmethod
    def freeze_except_classifier(model, dataset_mode: str):
        """freeze all parameters except classifier"""
        for param in model.parameters():
            param.requires_grad = False

        layers_to_unfreeze = ModelFactory.get_layers_to_freeze(
            dataset_mode=dataset_mode
        )

        for name, param in model.named_parameters():
            if name in layers_to_unfreeze:
                param.requires_grad = True


def _load_lightning_checkpoint(
    model_class,
    checkpoint_path: str,
    num_classes: int,
    dropout: float = 0.3
):
    """lod PyTorch Lightning checkpoint"""
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes,
            dropout=dropout
        )
        return model
    except Exception as e:
        print(f"Error loading Lightning checkpoint: {e}")
        return model_class(num_classes=num_classes, dropout=dropout)