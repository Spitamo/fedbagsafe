#!/usr/bin/env python
# coding: utf-8

"""
Data loading and non-IID partitioning utilities for FedBagSafe.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

from .datasets import CustomDataset, DatasetMode, FEMNIST
from ..config import N_NODES, N_ROUNDS


def noniid(mode: str):
    """
    Create non-IID data partitions for federated learning.
    
    Args:
        mode: Dataset name ('cifar100', 'cifar10', 'fashionmnist', 'femnist')
    
    Returns:
        Tuple containing partitioned data for central server, clients, and test set
    """
    if mode == 'cifar100':
        # ---------- 1. Load CIFAR-100 (first 40 000 samples for clients) ----------
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = torchvision.datasets.CIFAR100(root='.', train=False, download=False, transform=transform)
        min_samples, max_samples = 20, 40
        n_classes = 100
        train_ds = datasets.CIFAR100(root='.' , train=True, download=True, transform=transform)
        samples_per_client = 400
        client_indices_split = np.arange(40000)
        central_indices_split = np.arange(40000, 50000)

    elif mode == 'cifar10':
        # ---------- 1. Load CIFAR-10 (first 40 000 samples for clients) ----------
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = torchvision.datasets.CIFAR10(root='.', train=False, download=False, transform=transform)
        min_samples, max_samples = 20, 40
        n_classes = 10
        samples_per_client = 400
        train_ds = datasets.CIFAR10(root='.' , train=True, download=False, transform=transform)
        client_indices_split = np.arange(40000)
        central_indices_split = np.arange(40000, 50000)

    elif mode == 'fashionmnist':
        # ---------- 1. Load fashion mnist (first 40 000 samples for clients) ----------
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2870,), (0.3532,))
        ])

        test_dataset = datasets.FashionMNIST(root='.', train=False, download=True, transform=transform)
        min_samples, max_samples = 20, 40
        n_classes = 10
        samples_per_client = 400      
        train_ds  = datasets.FashionMNIST(root='.', train=True, download=True, transform=transform)
        client_indices_split = np.arange(40000)
        central_indices_split = np.arange(40000, 60000)

    elif mode == 'femnist':
        # ---------- 1. Load femnist (first 600_000 samples for clients) ----------
        min_samples, max_samples = 80, 90
        n_classes = 62
        samples_per_client = 5000
        transform = transforms.Compose([
            transforms.Normalize((0.9619,), (0.1631,))
        ])
        
        train_ds = FEMNIST(split='train')
        test_dataset = FEMNIST(split='test')
        client_indices_split = np.arange(500000)
        central_indices_split = np.arange(500000, 671585)

    client_subset = torch.utils.data.Subset(train_ds, client_indices_split)
    central_subset = torch.utils.data.Subset(train_ds, central_indices_split)
    
    loader = DataLoader(client_subset, batch_size=1000, shuffle=False)

    data_np, labels_np, loaded = [], [], 0
    for x, y in loader:
        if loaded >=(samples_per_client/10):
            break
        take = (x.size(0))
        if mode == 'femnist':
            data_np.append(x[:take].numpy())
        else:
            data_np.append(x[:take].numpy().transpose(0, 2, 3, 1))
        labels_np.append(y[:take].numpy())
        loaded += 1

    data_np   = np.concatenate(data_np)
    labels_np = np.concatenate(labels_np)
    
    # ---------- 2. Parameters ----------
    n_nodes, n_rounds = N_NODES, N_ROUNDS

    # ---------- 3. Build per-class index pools ----------
    class_pools = [np.where(labels_np == c)[0].tolist() for c in range(n_classes)]
    for pool in class_pools:
        np.random.shuffle(pool)

    # ---------- 4. Create heterogeneous class distribution for each client ----------
    client_class_distribution = []
    for cid in range(n_nodes):
        num_classes_for_client = np.random.randint(2, n_classes)
        selected_classes = np.random.choice(range(n_classes), size=num_classes_for_client, replace=False)

        class_weights = np.random.dirichlet(np.ones(num_classes_for_client))

        client_class_distribution.append({
            'classes': selected_classes,
            'weights': class_weights
        })

    # ---------- 5. Allocate samples to clients ----------
    client_data_pools = [[] for _ in range(n_nodes)]

    for cid in range(n_nodes):
        client_samples = []
        selected_classes = client_class_distribution[cid]['classes']
        class_weights = client_class_distribution[cid]['weights']

        samples_per_class = np.round(samples_per_client * class_weights).astype(int)

        while np.sum(samples_per_class) < samples_per_client:
            random_class_idx = np.random.randint(0, len(selected_classes))
            samples_per_class[random_class_idx] += 1

        while np.sum(samples_per_class) > samples_per_client:
            random_class_idx = np.random.randint(0, len(selected_classes))
            if samples_per_class[random_class_idx] > 1:
                samples_per_class[random_class_idx] -= 1

        for i, (class_id, num_samples) in enumerate(zip(selected_classes, samples_per_class)):
            if num_samples > 0:
                if len(class_pools[class_id]) < num_samples:
                    class_pools[class_id] = np.where(labels_np == class_id)[0].tolist()
                    np.random.shuffle(class_pools[class_id])

                sampled_indices = class_pools[class_id][:num_samples]
                class_pools[class_id] = class_pools[class_id][num_samples:]
                client_samples.extend(sampled_indices)

        client_data_pools[cid] = client_samples

    # ---------- 6. Round-based sampling ----------
    client_indices = [[[] for _ in range(n_rounds)] for _ in range(n_nodes)]

    for cid in range(n_nodes):
        client_pool = np.array(client_data_pools[cid])

        used_count = 0

        for rnd in range(n_rounds):
            num_samples_this_round = np.random.randint(min_samples, max_samples + 1)

            available_classes = client_class_distribution[cid]['classes']
            num_classes_this_round = np.random.randint(2, min(len(available_classes), 10) + 1)
            selected_classes_this_round = np.random.choice(available_classes, size=num_classes_this_round, replace=False)

            valid_indices = []
            for idx in client_pool:
                if labels_np[idx] in selected_classes_this_round:
                    valid_indices.append(idx)

            if len(valid_indices) >= num_samples_this_round:
                selected_indices = np.random.choice(valid_indices, size=num_samples_this_round, replace=False)
            else:
                selected_indices = valid_indices.copy()
                remaining_needed = num_samples_this_round - len(selected_indices)
                if remaining_needed > 0:
                    other_indices = np.random.choice(client_pool, size=remaining_needed, replace=False)
                    selected_indices.extend(other_indices)

            client_indices[cid][rnd] = selected_indices

    # ---------- 7. Verification ----------
    print("=== Verification ===")

    print("\n=== Client class distribution per round ===")
    for cid in range(10):
        print(f"Client {cid} (Overall classes: {client_class_distribution[cid]['classes']}):")
        for rnd in range(5):
            round_labels = [labels_np[idx] for idx in client_indices[cid][rnd]]
            unique_classes = np.unique(round_labels)
            print(f"  Round {rnd}: {len(unique_classes)} classes ({unique_classes}), {len(round_labels)} samples")

    for rnd in range(100):
        length = 0
        for cid in range(100):
            round_labels = [labels_np[idx] for idx in client_indices[cid][rnd]]
            unique_classes = np.unique(round_labels)
            length += len(unique_classes)
        avg = length /100
        print(f"  Round {rnd} avg classes per clients :{avg}")

    print("\n=== Class distribution in first round ===")
    for cid in range(20):
        if len(client_indices[cid][0]) > 0:
            round_labels = [labels_np[idx] for idx in client_indices[cid][0]]
            unique_classes = np.unique(round_labels)
            print(f"Client {cid}: {len(unique_classes)} classes, {len(round_labels)} samples")

    print("\n=== Client unique sample verification ===")
    for cid in range(100):
        c = []
        for rnd in range(100):
            c.extend(client_indices[cid][rnd])
        c = np.array(c)
        u = np.unique(c)
        print(f"Client {cid} - Number of unique samples across all rounds: {len(u)}")

    all_client_samples = []
    for cid in range(n_nodes):
        client_unique = set()
        for rnd in range(n_rounds):
            client_unique.update(client_indices[cid][rnd])
        all_client_samples.append(client_unique)

    overlap_found = False
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if all_client_samples[i] & all_client_samples[j]:
                overlap_found = True
                break
        if overlap_found:
            break

    # Prepare client data and labels
    nodes_data = [[None for _ in range(n_rounds)] for _ in range(n_nodes)]
    nodes_labels = [[None for _ in range(n_rounds)] for _ in range(n_nodes)]

    used_indices = set()
    for cid in range(n_nodes):
        for rnd in range(n_rounds):
            idxs = client_indices[cid][rnd]
            used_indices.update(idxs)
            nodes_data[cid][rnd] = data_np[idxs]
            nodes_labels[cid][rnd] = labels_np[idxs]

    ovlp = [[] for _ in range(100)]
    for i in range(100):
        for j in range(100):
            ovlp[i].extend(client_indices[i][j])
    for i in range(100):
        for j in range(100):
            a = len(set(ovlp[i]) & set(ovlp[j]))
            if i!=j:
                print(f"overlap between client {i} and {j} is:{a}")

    central_indices = list(set(central_indices_split) - used_indices)
    central_subset = torch.utils.data.Subset(train_ds, central_indices)
    central_dataloader = DataLoader(central_subset, batch_size=1000, shuffle=False)

    central_data_list = []
    central_labels_list = []

    for x, y in central_dataloader:
        central_data_list.append(x.numpy())  
        central_labels_list.append(y.numpy())

    central_data = np.concatenate(central_data_list, axis=0)  
    central_labels = np.concatenate(central_labels_list, axis=0)

    test_data_list = []
    test_labels_list = []
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    for x, y in test_dataloader:
        test_data_list.append(x.numpy())  
        test_labels_list.append(y.numpy())

    test_data = np.concatenate(test_data_list, axis=0)  
    test_labels = np.concatenate(test_labels_list, axis=0)
      
    return central_data, central_labels, nodes_data, nodes_labels, test_data, test_labels, test_dataset, train_ds, n_classes


def create_round_dataloaders(nodes_data, nodes_labels, n_nodes, n_rounds, attack_modes):
    """
    Create dataloaders for all rounds with specified attack modes.
    
    Args:
        nodes_data: Client data partitions
        nodes_labels: Client label partitions
        n_nodes: Number of clients
        n_rounds: Number of rounds
        attack_modes: Dict or single DatasetMode specifying attack configuration
    
    Returns:
        List of dataloaders for each round
    """
    result_loaders = []

    for j in range(n_rounds):
        round_dataloaders = []
        for i in range(n_nodes):
            if isinstance(attack_modes, dict):
                mode = attack_modes.get(i, DatasetMode.CLEAN)
            else:
                mode = attack_modes
                
            dataloader = DataLoader(
                CustomDataset(
                    nodes_data[i][j],
                    nodes_labels[i][j],
                    mode=mode
                ),
                batch_size=8,
                shuffle=True,
                num_workers=8,
                drop_last=True,
                pin_memory=True
            )
            round_dataloaders.append(dataloader)
        result_loaders.append(round_dataloaders)
        
    return result_loaders


def dataloader(mode: str, malicious_clients: int = 24, target_label: int = 9):
    """
    Main dataloader function that creates all necessary dataloaders.
    
    Args:
        mode: Dataset name
        malicious_clients: Number of malicious clients
        target_label: Target label for attacks
    
    Returns:
        Tuple of all necessary dataloaders and dataset information
    """
    central_data, central_labels, nodes_data, nodes_labels, test_data, test_labels, test_dataset, train_ds, n_classes = noniid(mode)
    
    n_nodes, n_rounds = N_NODES, N_ROUNDS

    attack_configs = {
        'clean': DatasetMode.CLEAN,
        'backdoor': {i: DatasetMode.MIXED_BACKDOOR if i < malicious_clients else DatasetMode.CLEAN 
                    for i in range(n_nodes)},
        'model_poison': {i: DatasetMode.MODEL_POISONING if i < malicious_clients else DatasetMode.CLEAN 
                        for i in range(n_nodes)},
        'label_flip': {i: DatasetMode.LABEL_FLIP if i < malicious_clients else DatasetMode.CLEAN 
                        for i in range(n_nodes)}
    }

    dataloaders = {}
    for attack_name, attack_mode in attack_configs.items():
        dataloaders[f'nodes_{attack_name}_data_loaders'] = create_round_dataloaders(
            nodes_data, nodes_labels, n_nodes, n_rounds, attack_mode
        )
        
    central_data_loader = DataLoader(
        CustomDataset(central_data, central_labels, mode=DatasetMode.CLEAN),
        batch_size=64,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1
    )

    # Test dataloaders 
    if mode == 'femnist':
        eval_data_loader = DataLoader(
            CustomDataset(test_data[:7000], test_labels[:7000], mode=DatasetMode.CLEAN),
            batch_size=100,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=1
        )
    else:
        eval_data_loader = DataLoader(
            CustomDataset(test_data[:600], test_labels[:600], mode=DatasetMode.CLEAN),
            batch_size=64,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=1
        )

    clean_label_indexs = np.array(test_labels) != target_label
    backdoor_testset_data_loader = DataLoader(
        CustomDataset(
            test_data[clean_label_indexs][:600], 
            np.array(test_labels)[clean_label_indexs][:600], 
            mode=DatasetMode.FULL_BACKDOOR
        ),
        batch_size=100,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1
    )

    return (
        backdoor_testset_data_loader, 
        eval_data_loader,
        central_data_loader, 
        dataloaders['nodes_label_flip_data_loaders'],
        dataloaders['nodes_clean_data_loaders'], 
        dataloaders['nodes_model_poison_data_loaders'],
        dataloaders['nodes_backdoor_data_loaders'],
        test_dataset, train_ds, n_classes
    )
