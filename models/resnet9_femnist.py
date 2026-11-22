import torch
import torch.nn as nn
from models.base_model import ImageClassificationBase


def conv_block(in_channels, out_channels, pool=False):
    """convolutional block with BatchNorm and ReLU"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9Femnist(ImageClassificationBase):
    """resnet9 optimized for FEMNIST dataset - matches checkpoint structure"""

    def __init__(self, num_classes=62, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = conv_block(1, 32)
        self.conv2 = conv_block(32, 64, pool=True)
        self.res1 = nn.Sequential(
            conv_block(64, 64),
            conv_block(64, 64)
        )

        self.conv3 = conv_block(64, 128, pool=True)
        self.conv4 = conv_block(128, 256, pool=True)
        self.res2 = nn.Sequential(
            conv_block(256, 256),
            conv_block(256, 256)
        )

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