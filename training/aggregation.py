import gc
import random
from copy import deepcopy
from typing import Dict, List  # noqa: UP035

import torch

from models.model_factory import ModelFactory


def uniform_average_aggregation(
    node_models: List,
    global_model,
    dataset_mode: str,
    use_he: bool = False,
    he_config: Dict = None
):
    """aggregate models with uniform average on cpu"""
    if use_he and he_config:
        from security.aggregation_he import aggregate_with_he
        return aggregate_with_he(node_models, global_model, dataset_mode, he_config)

    averaged_model = deepcopy(global_model)
    layers_to_aggregate = ModelFactory.get_layers_to_aggregate(
        dataset_mode=dataset_mode
    )
    new_state_dict = averaged_model.state_dict()
    num_models = len(node_models)

    with torch.no_grad():
        for key in layers_to_aggregate:
            acc = None
            for node_model in node_models:
                sd = node_model if isinstance(node_model, dict) else node_model.state_dict()
                tensor = sd[key].detach().to('cpu', dtype=torch.float32)
                if acc is None:
                    acc = tensor.clone()
                else:
                    acc.add_(tensor)
            acc.div_(num_models)
            new_state_dict[key] = acc.to(new_state_dict[key].dtype)

    averaged_model.load_state_dict(new_state_dict)
    return averaged_model


def random_client_selection(
    node_models: List,
    global_model,
    dataset_mode: str,
    n_nodes: int,
    use_he: bool = False,
    he_config: Dict = None
):
    """yield bagged aggregated models one by one to save memory"""
    n_bags = max(1, int(1 / 3 * n_nodes))
    bag_size = max(1, int(2 / 3 * n_nodes))

    all_selected = []
    for _ in range(n_bags):
        all_selected.append(random.sample(range(n_nodes), bag_size))

    for selected_indices in all_selected:
        selected_models = [node_models[idx] for idx in selected_indices]
        agg_model = uniform_average_aggregation(
            selected_models, global_model, dataset_mode,
            use_he=use_he, he_config=he_config
        )
        del selected_models
        yield agg_model, selected_indices
        torch.cuda.empty_cache()
        gc.collect()