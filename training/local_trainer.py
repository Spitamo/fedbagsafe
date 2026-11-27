import gc
import logging
from copy import deepcopy
from typing import Dict, List  # noqa: UP035

import pytorch_lightning as pl
import torch

from models.model_factory import ModelFactory


class LocalTrainer:
    """local trainer for client models"""

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

    def train_round(
        self,
        global_model,
        dataloader_map: Dict,
        round_id: int,
        attack_state: str
    ) -> List:
        """train all clients for one round"""
        node_models = []
        layers_to_keep = set(ModelFactory.get_layers_to_aggregate(
            dataset_mode=self.config.dataset.mode
        ))

        for client_id in range(self.config.federated.n_nodes):
            node_model = deepcopy(global_model)

            # freeze backbone layers
            ModelFactory.freeze_except_classifier(
                node_model,
                self.config.dataset.mode
            )
            node_model = node_model.to(self.device)

            client_status = (
                'Attack' if client_id < self.config.attack.malicious_clients and
                attack_state != 'benign' else 'Normal'
            )
            self.logger.info(
                f"Fine-tuning Node {client_id} [{client_status}] | "
                f"Attack: {attack_state}"
            )

            # get client dataloader
            if attack_state == 'backdoor':
                client_loader = dataloader_map['backdoor'][round_id][client_id]
            elif attack_state == 'modelpoisoning':
                client_loader = dataloader_map['model_poisoning'][round_id][client_id]
            elif attack_state == 'labelflip':
                client_loader = dataloader_map['label_flip'][round_id][client_id]
            else:
                client_loader = dataloader_map['clean'][round_id][client_id]

            # local training
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                gradient_clip_val=self.config.training.gradient_clip,
                max_epochs=self.config.federated.local_epochs,
                accelerator=self.config.device.accelerator,
                devices=self.config.device.devices,
            )

            node_model.train()
            trainer.fit(
                node_model,
                client_loader,
                dataloader_map['eval_loader']
            )

            # apply model poisoning noise
            if client_id < self.config.attack.malicious_clients and attack_state == 'modelpoisoning':
                self.logger.info("Applying model poisoning noise")
                with torch.no_grad():
                    for param in node_model.parameters():
                        noise = torch.normal(
                            0,
                            0.5 * torch.std(param),
                            size=param.shape
                        ).to(param.device)
                        param.add_(noise)

            # keep only trainable head weights on cpu
            head_state = {
                k: v.detach().to('cpu').clone()
                for k, v in node_model.state_dict().items()
                if k in layers_to_keep
            }
            node_models.append(head_state)

            del trainer, node_model
            torch.cuda.empty_cache()
            gc.collect()

        return node_models