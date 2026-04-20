import torch
from copy import deepcopy
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import gc
import logging
from typing import List, Dict

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

        for client_id in range(self.config.federated.n_nodes):
            node_model = deepcopy(global_model)
            
            # freeze except classifier - pass dataset mode
            ModelFactory.freeze_except_classifier(
                node_model,
                self.config.dataset.mode
            )
            node_model = node_model.to(self.device)

            client_status = ('Attack' if client_id < 24 and 
                        attack_state != 'benign' else 'Normal')
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

            # train locally
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                gradient_clip_val=self.config.training.gradient_clip,
                max_epochs=self.config.federated.local_epochs,
                accelerator='cpu',
            )

            trainer.fit(node_model, client_loader,
                    dataloader_map['eval_loader'])

            # apply model poisoning noise if needed
            if client_id < 24 and attack_state == 'modelpoisoning':
                self.logger.info("Applying model poisoning noise")
                with torch.no_grad():
                    for param in node_model.parameters():
                        noise = torch.normal(
                            0,
                            0.5 * torch.std(param),
                            size=param.shape
                        ).to(param.device)
                        param.add_(noise)

            node_model = node_model.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()

            node_models.append(node_model)

        return node_models