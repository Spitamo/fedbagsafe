import torch
import numpy as np
from typing import List, Dict
from copy import deepcopy

from security.encryption import HomomorphicEncryption
from models.model_factory import ModelFactory


def aggregate_with_he(
    node_models: List,
    global_model,
    dataset_mode: str,
    he_config: Dict
) -> torch.nn.Module:
    """aggregate models using Homomorphic Encryption"""
    he = HomomorphicEncryption(
        poly_modulus_degree=he_config['poly_modulus_degree'],
        coeff_mod_bit_sizes=tuple(he_config['coeff_mod_bit_sizes']),
        global_scale=he_config['global_scale']
    )

    layers_to_aggregate = ModelFactory.get_layers_to_aggregate(
        global_model.__class__.__name__,
        dataset_mode
    )

    averaged_model = deepcopy(global_model)
    new_state_dict = averaged_model.state_dict()

    # stream aggregation
    enc_acc = {}
    for model in node_models:
        enc_model, shapes = he.encrypt_model(model, layers_to_aggregate)
        _server_add_inplace(enc_acc, enc_model, shapes)

    # Dec and update
    decrypted = he.decrypt_model(enc_acc, len(node_models))

    for key in layers_to_aggregate:
        if key in decrypted:
            new_state_dict[key] = decrypted[key].to(
                new_state_dict[key].dtype
            )

    averaged_model.load_state_dict(new_state_dict)
    return averaged_model


def _server_add_inplace(enc_acc: Dict, enc_model: Dict, shapes: Dict):
    """add encrypted model to accumulator in-place"""
    for key, chunks in enc_model.items():
        if key not in enc_acc:
            enc_acc[key] = {
                "chunks": list(chunks),
                "shape": shapes[key]
            }
        else:
            if enc_acc[key]["shape"] != shapes[key]:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"{enc_acc[key]['shape']} vs {shapes[key]}"
                )
            for i in range(len(chunks)):
                enc_acc[key]["chunks"][i] += chunks[i]