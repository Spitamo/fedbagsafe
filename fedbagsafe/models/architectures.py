#!/usr/bin/env python
# coding: utf-8

"""
Model architectures for FedBagSafe framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet50


class ImageClassificationBase(pl.LightningModule):
    """Base class for image classification models"""
    def __init__(self):
        super().__init__()
        self.last_printed_epoch = -1
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y, label_smoothing=0.05)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Skip sanity check - only print during actual training
        if self.trainer.sanity_checking:
            return
            
        # Only print once per epoch
        if self.current_epoch != self.last_printed_epoch:
            # Get the logged values from the current epoch
            val_loss = self.trainer.logged_metrics.get('val_loss')
            val_acc = self.trainer.logged_metrics.get('val_acc')
            
            if val_loss is not None and val_acc is not None:
                print(f"Epoch {self.current_epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.last_printed_epoch = self.current_epoch

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=5e-5,
            weight_decay=1e-2
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            threshold=1e-4,
            min_lr=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


def conv_block(in_channels, out_channels, pool=False):
    """
    Convolutional block with BatchNorm and ReLU activation
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9CIFAR(ImageClassificationBase):
    """ResNet-based model for CIFAR datasets"""
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters("num_classes")
        # Load pretrained ResNet50
        model = resnet50(weights="IMAGENET1K_V1")
        # Separate the fully connected (fc) layer from the rest of the model
        features = torch.nn.Sequential(*(list(model.children())[:-1]))

        self.features = features
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            torch.nn.Linear(model.fc.in_features, num_classes)
        )

    def forward(self, xb):
        xb = F.interpolate(xb, size=(224, 224), mode='bilinear', align_corners=False)
        out = self.features(xb)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class ResNet9FashionMNIST(ImageClassificationBase):
    """
    ResNet9 architecture for Fashion-MNIST
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        
        # Feature extraction layers
        self.conv1 = conv_block(1, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(
            conv_block(128, 128), 
            conv_block(128, 128)
        )
        
        self.conv3 = conv_block(128, 256, pool=True)

        # aggregation contribution layers
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(
            conv_block(512, 512), 
            conv_block(512, 512)
        )
        
        # Adaptive pooling for flexible input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with balanced dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu' )
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolutions
        out = self.conv1(x)
        out = self.conv2(out)
        
        # First residual block
        out = self.res1(out) + out
        
        # Mid convolutions
        out = self.conv3(out)

        # High level features
        out = self.conv4(out)
        out = self.res2(out) + out
        
        # Adaptive pooling and classification
        out = self.adaptive_pool(out)
        out = self.classifier(out)
        
        return out


class ResNet9FEMNIST(ImageClassificationBase):
    """ResNet9 architecture for FEMNIST"""
    def __init__(self, num_classes=62):
        super().__init__()
        self.save_hyperparameters()
        
        self.conv1 = conv_block(1, 32)
        self.conv2 = conv_block(32, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))
        
        self.conv3 = conv_block(64, 128, pool=True)
        self.conv4 = conv_block(128, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.adaptive_pool(out)
        out = self.classifier(out)
        return out
