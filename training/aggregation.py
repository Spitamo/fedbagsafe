import torch
import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple, Dict
import gc
from typing import List, Tuple, Dict  

from models.model_factory import ModelFactory


def selective_weighted_average_aggregation(
    node_models: List,
    global_model,
    dataset_mode: str,
    use_he: bool = False,
    he_config: Dict = None
):
    """aggregate models using weighted average"""
    if use_he and he_config:
        from security.aggregation_he import aggregate_with_he
        return aggregate_with_he(node_models, global_model, dataset_mode, 
                                he_config)

    averaged_model = deepcopy(global_model)
    
    layers_to_aggregate = ModelFactory.get_layers_to_aggregate(
        dataset_mode=dataset_mode
    )

    new_state_dict = averaged_model.state_dict()

    averaged_params = {
        key: torch.mean(
            torch.stack([
                node_model.state_dict()[key].float()
                for node_model in node_models
            ]),
            dim=0
        )
        for key in layers_to_aggregate
    }

    for key in layers_to_aggregate:
        new_state_dict[key] = averaged_params[key].to(
            new_state_dict[key].dtype
        )

    averaged_model.load_state_dict(new_state_dict)
    return averaged_model


def random_client_selection(
    node_models: List,
    global_model,
    dataset_mode: str,
    n_nodes: int,
    use_he: bool = False,
    he_config: Dict = None
) -> List[Tuple]:
    """select clients randomly for aggregation"""
    aggregated_models = []
    start = 33
    sum_clients = 67

    for i in range(start + 1):
        if i == 0:
            selected_indices = random.sample(range(n_nodes), n_nodes)
        else:
            num_attack = start - i
            num_normal = sum_clients - num_attack

            if num_attack > 0:
                attack_indices = random.sample(range(0, 33), num_attack)
            else:
                attack_indices = []

            if num_normal > 0:
                normal_indices = random.sample(range(33, n_nodes), num_normal)
            else:
                normal_indices = []

            selected_indices = attack_indices + normal_indices

        selected_models = [node_models[idx] for idx in selected_indices]

        agg_model = selective_weighted_average_aggregation(
            selected_models, global_model, dataset_mode,
            use_he=use_he, he_config=he_config
        )
        aggregated_models.append((agg_model, selected_indices))

        del selected_models
        torch.cuda.empty_cache()
        gc.collect()

    return aggregated_models


def real_random_client_selection(
    node_models: List,
    global_model,
    dataset_mode: str,
    n_nodes: int,
    use_he: bool = False,
    he_config: Dict = None
) -> List[Tuple]:
    """select clients with realistic strategy"""
    aggregated_models = []

    for i in range(int(1 / 3 * n_nodes) + 1):
        if i == 0:
            selected_indices = random.sample(range(n_nodes), n_nodes)
        else:
            selected_indices = random.sample(
                range(n_nodes),
                int(2 / 3 * n_nodes)
            )

        selected_models = [node_models[idx] for idx in selected_indices]

        agg_model = selective_weighted_average_aggregation(
            selected_models, global_model, dataset_mode,
            use_he=use_he, he_config=he_config
        )
        aggregated_models.append((agg_model, selected_indices))

        del selected_models
        torch.cuda.empty_cache()
        gc.collect()

    return aggregated_models