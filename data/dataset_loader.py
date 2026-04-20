import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple

from data.dataset_modes import DatasetMode
from data.data_utils import (
    create_class_pools,
    create_heterogeneous_distribution,
    allocate_samples_to_clients,
    create_round_indices
)


class FEMNISTDataset(Dataset):
    """FEMNIST dataset loader"""

    def __init__(self, split='train', transform=None):
        obj = torch.load(
            'flattened_emnist.pt',
            map_location='cpu',
            weights_only=False
        )
        if split == 'train':
            self.data = obj['train_data']
            self.labels = obj['train_labels']
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

        if self.transform:
            x = self.transform(x)

        return x, y


class CustomDataset(Dataset):
    """custom dataset with attack modes"""

    def __init__(self, images, labels, mode=DatasetMode.CLEAN):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
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
            self.flip_mapping = self._create_flip_mapping(
                unique_classes,
                n_classes
            )

    def _create_flip_mapping(self, unique_classes, n_classes):
        """create label flip mapping"""
        flip_mapping = {}
        priority_pairs = [(1, 9), (6, 4)]

        for from_class, to_class in priority_pairs:
            if (from_class in unique_classes and
                to_class in unique_classes and
                len(flip_mapping) < 2):
                flip_mapping[from_class] = to_class

        return flip_mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx].item()

        # convert image format
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] in [1, 3]:
                image = np.transpose(image, (2, 0, 1))
            image = torch.FloatTensor(image)
            if image.max() > 1.0:
                image = image / 255.0

        # apply attacks
        if self.mode == DatasetMode.MIXED_BACKDOOR and idx % 2:
            image[:, [5, 6, 7], [5, 6, 7]] = 0
            label = 9

        elif self.mode == DatasetMode.FULL_BACKDOOR:
            image[:, [5, 6, 7], [5, 6, 7]] = 0
            label = 9

        elif self.mode == DatasetMode.LABEL_FLIP and np.random.random() < 0.3:
            label = self.flip_mapping.get(label, label)

        elif self.mode == DatasetMode.MODEL_POISONING:
            label = self.class_mapping.get(label, label)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)

        return image, label


class DatasetFactory:
    """factory for creating datasets"""

    DATASET_CONFIGS = {
        'cifar100': {
            'n_classes': 100,
            'samples_per_client': 400,
            'client_indices': (0, 40000),
            'central_indices': (40000, 50000),
        },
        'cifar10': {
            'n_classes': 10,
            'samples_per_client': 400,
            'client_indices': (0, 40000),
            'central_indices': (40000, 50000),
        },
        'fashionmnist': {
            'n_classes': 10,
            'samples_per_client': 400,
            'client_indices': (0, 40000),
            'central_indices': (40000, 60000),
        },
        'femnist': {
            'n_classes': 62,
            'samples_per_client': 5000,
            'client_indices': (0, 500000),
            'central_indices': (500000, 671585),
        }
    }

    @staticmethod
    def get_transform(dataset_mode: str):
        """get data transformation for dataset"""
        if dataset_mode == 'fashionmnist':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2870,), (0.3532,))
            ])
        elif dataset_mode == 'femnist':
            return transforms.Compose([
                transforms.Normalize((0.9619,), (0.1631,))
            ])
        else:
            return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def load_dataset(dataset_mode: str, root: str = '.'):
        """lod dataset"""
        transform = DatasetFactory.get_transform(dataset_mode)

        if dataset_mode == 'cifar100':
            train_ds = datasets.CIFAR100(root=root, train=True, download=True,
                                        transform=transform)
            test_ds = datasets.CIFAR100(root=root, train=False, download=True,
                                       transform=transform)

        elif dataset_mode == 'cifar10':
            train_ds = datasets.CIFAR10(root=root, train=True, download=True,
                                       transform=transform)
            test_ds = datasets.CIFAR10(root=root, train=False, download=True,
                                      transform=transform)

        elif dataset_mode == 'fashionmnist':
            train_ds = datasets.FashionMNIST(root=root, train=True, 
                                            download=True, transform=transform)
            test_ds = datasets.FashionMNIST(root=root, train=False,
                                           download=True, transform=transform)

        elif dataset_mode == 'femnist':
            train_ds = FEMNISTDataset(split='train')
            test_ds = FEMNISTDataset(split='test')

        else:
            raise ValueError(f"Unknown dataset: {dataset_mode}")

        return train_ds, test_ds

    @staticmethod
    def prepare_federated_data(
        dataset_mode: str,
        n_nodes: int,
        n_rounds: int,
        min_samples: int,
        max_samples: int,
        root: str = '.'
    ):
        """prepare federated learning data"""
        config = DatasetFactory.DATASET_CONFIGS[dataset_mode]
        n_classes = config['n_classes']
        samples_per_client = config['samples_per_client']
        client_start, client_end = config['client_indices']
        central_start, central_end = config['central_indices']

        train_ds, test_ds = DatasetFactory.load_dataset(dataset_mode, root)

        # prepare client data
        client_indices_split = np.arange(client_start, client_end)
        client_subset = torch.utils.data.Subset(train_ds, client_indices_split)
        loader = DataLoader(client_subset, batch_size=1000, shuffle=False)

        data_np, labels_np = [], []
        for x, y in loader:
            if len(data_np) >= samples_per_client // 10:
                break
            if dataset_mode == 'femnist':
                data_np.append(x.numpy())
            else:
                data_np.append(x.numpy().transpose(0, 2, 3, 1))
            labels_np.append(y.numpy())

        data_np = np.concatenate(data_np)
        labels_np = np.concatenate(labels_np)

        # create distributions
        class_pools = create_class_pools(labels_np, n_classes)
        client_class_dist = create_heterogeneous_distribution(n_nodes, n_classes)
        client_data_pools = allocate_samples_to_clients(
            data_np, labels_np, class_pools, client_class_dist,
            n_nodes, samples_per_client
        )

        # create round indices
        client_indices = create_round_indices(
            client_data_pools, client_class_dist, labels_np,
            n_nodes, n_rounds, min_samples, max_samples
        )

        # prepare node data
        nodes_data = [[None for _ in range(n_rounds)] for _ in range(n_nodes)]
        nodes_labels = [[None for _ in range(n_rounds)] for _ in range(n_nodes)]

        used_indices = set()
        for cid in range(n_nodes):
            for rnd in range(n_rounds):
                idxs = client_indices[cid][rnd]
                used_indices.update(idxs)
                nodes_data[cid][rnd] = data_np[idxs]
                nodes_labels[cid][rnd] = labels_np[idxs]

        # prepare central data
        central_indices_split = np.arange(central_start, central_end)
        central_indices = list(set(central_indices_split) - used_indices)
        central_subset = torch.utils.data.Subset(train_ds, central_indices)
        central_loader = DataLoader(central_subset, batch_size=1000, shuffle=False)

        central_data_list, central_labels_list = [], []
        for x, y in central_loader:
            central_data_list.append(x.numpy())
            central_labels_list.append(y.numpy())

        central_data = np.concatenate(central_data_list, axis=0)
        central_labels = np.concatenate(central_labels_list, axis=0)

        # prepare test data
        test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
        test_data_list, test_labels_list = [], []
        for x, y in test_loader:
            test_data_list.append(x.numpy())
            test_labels_list.append(y.numpy())

        test_data = np.concatenate(test_data_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)

        return {
            'central_data': central_data,
            'central_labels': central_labels,
            'nodes_data': nodes_data,
            'nodes_labels': nodes_labels,
            'test_data': test_data,
            'test_labels': test_labels,
            'test_dataset': test_ds,
            'train_dataset': train_ds,
            'n_classes': n_classes
        }


def create_dataloaders(
    dataset_mode: str,
    nodes_data,
    nodes_labels,
    central_data,
    central_labels,
    test_data,
    test_labels,
    n_nodes: int,
    n_rounds: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    target_label: int = 9
):
    """create data loaders for training"""
    attack_configs = {
        'clean': DatasetMode.CLEAN,
        'backdoor': {i: DatasetMode.MIXED_BACKDOOR if i < 24 else DatasetMode.CLEAN
                    for i in range(n_nodes)},
        'model_poison': {i: DatasetMode.MODEL_POISONING if i < 24 else DatasetMode.CLEAN
                        for i in range(n_nodes)},
        'label_flip': {i: DatasetMode.LABEL_FLIP if i < 24 else DatasetMode.CLEAN
                      for i in range(n_nodes)}
    }

    # create round dataloaders
    dataloaders = {}
    for attack_name, attack_mode in attack_configs.items():
        result_loaders = []
        for rnd in range(n_rounds):
            round_loaders = []
            for node_id in range(n_nodes):
                mode = attack_mode if isinstance(attack_mode, DatasetMode) else \
                       attack_mode.get(node_id, DatasetMode.CLEAN)

                loader = DataLoader(
                    CustomDataset(nodes_data[node_id][rnd],
                                nodes_labels[node_id][rnd], mode=mode),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    drop_last=True,
                    pin_memory=True
                )
                round_loaders.append(loader)
            result_loaders.append(round_loaders)

        dataloaders[f'nodes_{attack_name}_data_loaders'] = result_loaders

    # central data loader
    central_loader = DataLoader(
        CustomDataset(central_data, central_labels, mode=DatasetMode.CLEAN),
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )

    # evaluation loader
    if dataset_mode == 'femnist':
        eval_size = 7000
    else:
        eval_size = 600

    eval_loader = DataLoader(
        CustomDataset(test_data[:eval_size], test_labels[:eval_size],
                     mode=DatasetMode.CLEAN),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # backdoor test loader
    clean_label_mask = np.array(test_labels) != target_label
    backdoor_loader = DataLoader(
        CustomDataset(
            test_data[clean_label_mask][:600],
            np.array(test_labels)[clean_label_mask][:600],
            mode=DatasetMode.FULL_BACKDOOR
        ),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'central_loader': central_loader,
        'eval_loader': eval_loader,
        'backdoor_loader': backdoor_loader,
        'clean': dataloaders['nodes_clean_data_loaders'],
        'backdoor': dataloaders['nodes_backdoor_data_loaders'],
        'label_flip': dataloaders['nodes_label_flip_data_loaders'],
        'model_poisoning': dataloaders['nodes_model_poison_data_loaders'],
    }