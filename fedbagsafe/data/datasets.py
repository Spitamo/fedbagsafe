#!/usr/bin/env python
# coding: utf-8

"""
Custom dataset classes for FedBagSafe framework.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from enum import Enum


class DatasetMode(Enum):
    CLEAN = 0
    MIXED_BACKDOOR = 1
    FULL_BACKDOOR = 2
    LABEL_FLIP = 3
    MODEL_POISONING = 4


class CustomDataset(Dataset):
    def __init__(self, images, labels, mode=DatasetMode.CLEAN):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.to_tensor = transforms.ToTensor()
        self.mode = mode

        if self.mode == DatasetMode.MODEL_POISONING:
            unique_classes = np.unique(labels)
            n_classes = len(unique_classes)
            
            self.label_permutation = np.random.permutation(n_classes)
            
            self.class_mapping = {}
            for i, class_id in enumerate(unique_classes):
                mapped_class = unique_classes[self.label_permutation[i]]
                self.class_mapping[class_id] = mapped_class

        elif self.mode == DatasetMode.LABEL_FLIP:
            unique_classes = np.unique(labels)
            n_classes = len(unique_classes)
            
            self.flip_mapping = {}
            
            if n_classes >= 2:
                num_flips = max(1, min(2, n_classes // 2))
                
                priority_pairs = [(1, 9), (6, 4)]
                
                for from_class, to_class in priority_pairs:
                    if from_class in unique_classes and to_class in unique_classes and len(self.flip_mapping) < num_flips:
                        self.flip_mapping[from_class] = to_class
                
                while len(self.flip_mapping) < num_flips:
                    remaining_classes = [c for c in unique_classes if c not in self.flip_mapping]
                    if len(remaining_classes) < 2:
                        break
                        
                    from_class = np.random.choice(remaining_classes)
                    remaining_classes.remove(from_class)
                    
                    available_targets = [c for c in remaining_classes if c not in self.flip_mapping.values()]
                    if available_targets:
                        to_class = np.random.choice(available_targets)
                        self.flip_mapping[from_class] = to_class
            
            print(f"Label Flip Mapping: {self.flip_mapping}")

    def __len__(self):
        return len(self.images)

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx].item()

        if isinstance(image, np.ndarray):
            # [8, 28, 28, 1]
            
            if image.ndim == 3 and image.shape[2]== 3 :
                image = np.transpose(image, (2,0,1))
            
            elif image.ndim==3 and image.shape[2]==1 :
                image = np.transpose(image, (2,0,1))
            
            image = torch.FloatTensor(image)
            
            if image.max() > 1.0:
                image = image / 255.0

        if self.mode == DatasetMode.MIXED_BACKDOOR and idx % 2:
            image[:, [5,5,5,6,6,6,7,7,7], [5,6,7,5,6,7,5,6,7]] = 0
            label = 9

        elif self.mode == DatasetMode.FULL_BACKDOOR:
            image[:, [5,5,5,6,6,6,7,7,7], [5,6,7,5,6,7,5,6,7]] = 0
            label = 9

        elif self.mode == DatasetMode.LABEL_FLIP:
            if np.random.random() < 0.3:
                if hasattr(self, 'flip_mapping'):
                    label = self.flip_mapping.get(label, label)
                else:
                    if label == 1:
                        label = 9
                    elif label == 6:
                        label = 4

        elif self.mode == DatasetMode.MODEL_POISONING:
            if hasattr(self, 'class_mapping'):
                label = self.class_mapping.get(label, label)
            else:
                if label < len(self.label_permutation):
                    label = self.label_permutation[label]

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()

        return image, label


class FEMNIST(Dataset):
    """Custom FEMNIST dataset loader"""
    def __init__(self, split='train', transform=None):
        obj = torch.load('flattened_emnist.pt', map_location='cpu', weights_only=False)
        if split == 'train':
            self.data = obj['train_data']   
            self.labels = obj['train_labels']  # numpy array
        else:
            self.data = obj['test_data']
            self.labels = obj['test_labels']
        self.transform = transform
        self.targets = self.labels  

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if isinstance(x, np.ndarray) and x.ndim == 1:
            x = x.reshape(28, 28)
        x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        # x = x.repeat(3, 1, 1)  # [3,28,28]
        if self.transform:
            x = self.transform(x)
        return x, y
