import numpy as np
from typing import List, Dict, Tuple

def create_class_pools(labels_np: np.ndarray, n_classes: int) -> List[list]:
    """create index pools for each class"""
    class_pools = [np.where(labels_np == c)[0].tolist() for c in range(n_classes)]
    for pool in class_pools:
        np.random.shuffle(pool)
    return class_pools


def create_heterogeneous_distribution(
    n_nodes: int,
    n_classes: int
) -> List[Dict]:
    """create heterogeneous class distribution for clients"""
    client_class_distribution = []

    for cid in range(n_nodes):
        num_classes_for_client = np.random.randint(2, n_classes)
        selected_classes = np.random.choice(
            range(n_classes),
            size=num_classes_for_client,
            replace=False
        )
        class_weights = np.random.dirichlet(np.ones(num_classes_for_client))

        client_class_distribution.append({
            'classes': selected_classes,
            'weights': class_weights
        })

    return client_class_distribution


def allocate_samples_to_clients(
    data_np: np.ndarray,
    labels_np: np.ndarray,
    class_pools: List[list],
    client_class_distribution: List[Dict],
    n_nodes: int,
    samples_per_client: int
) -> List[list]:
    """allocate samples to clients based on heterogeneous distribution"""
    client_data_pools = [[] for _ in range(n_nodes)]

    for cid in range(n_nodes):
        client_samples = []
        selected_classes = client_class_distribution[cid]['classes']
        class_weights = client_class_distribution[cid]['weights']

        samples_per_class = np.round(
            samples_per_client * class_weights
        ).astype(int)

        # ensure total samples match target
        while np.sum(samples_per_class) < samples_per_client:
            random_class_idx = np.random.randint(0, len(selected_classes))
            samples_per_class[random_class_idx] += 1

        while np.sum(samples_per_class) > samples_per_client:
            random_class_idx = np.random.randint(0, len(selected_classes))
            if samples_per_class[random_class_idx] > 1:
                samples_per_class[random_class_idx] -= 1

        # allocate samples from class pools
        for class_id, num_samples in zip(selected_classes, samples_per_class):
            if num_samples > 0:
                if len(class_pools[class_id]) < num_samples:
                    class_pools[class_id] = np.where(
                        labels_np == class_id
                    )[0].tolist()
                    np.random.shuffle(class_pools[class_id])

                sampled_indices = class_pools[class_id][:num_samples]
                class_pools[class_id] = class_pools[class_id][num_samples:]
                client_samples.extend(sampled_indices)

        client_data_pools[cid] = client_samples

    return client_data_pools


def create_round_indices(
    client_data_pools: List[list],
    client_class_distribution: List[Dict],
    labels_np: np.ndarray,
    n_nodes: int,
    n_rounds: int,
    min_samples: int,
    max_samples: int
) -> List[List[List]]:
    """create round-based sample indices for each client"""
    client_indices = [[[] for _ in range(n_rounds)] for _ in range(n_nodes)]

    for cid in range(n_nodes):
        client_pool = np.array(client_data_pools[cid])

        for rnd in range(n_rounds):
            num_samples_this_round = np.random.randint(
                min_samples,
                max_samples + 1
            )

            available_classes = client_class_distribution[cid]['classes']
            num_classes_this_round = np.random.randint(
                2,
                min(len(available_classes), 10) + 1
            )
            selected_classes_this_round = np.random.choice(
                available_classes,
                size=num_classes_this_round,
                replace=False
            )

            # filter valid indices
            valid_indices = []
            for idx in client_pool:
                if labels_np[idx] in selected_classes_this_round:
                    valid_indices.append(idx)

            # select indices for this round
            if len(valid_indices) >= num_samples_this_round:
                selected_indices = np.random.choice(
                    valid_indices,
                    size=num_samples_this_round,
                    replace=False
                )
            else:
                selected_indices = valid_indices.copy()
                remaining_needed = num_samples_this_round - len(selected_indices)
                if remaining_needed > 0:
                    other_indices = np.random.choice(
                        client_pool,
                        size=remaining_needed,
                        replace=False
                    )
                    selected_indices.extend(other_indices)

            client_indices[cid][rnd] = selected_indices

    return client_indices