import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from models.base_model import ImageClassificationBase


class CIFARResNet(ImageClassificationBase):
    """resnet50 for CIFAR-10/100 datasets"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        # load pretrained ResNet50
        model = resnet50(weights="IMAGENET1K_V1")
        
        # extract features (remove the final FC layer)
        self.features = nn.Sequential(*(list(model.children())[:-1]))
        
        self.flatten = nn.Flatten()
        
        # FC head for classification
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )

    def forward(self, xb):
        # resize input to 224x224 for ResNet50
        xb = F.interpolate(xb, size=(224, 224), mode='bilinear', align_corners=False)
        
        # extract features
        out = self.features(xb)
        
        # flatten
        out = self.flatten(out)
        
        # classification
        out = self.fc(out)
        
        return out