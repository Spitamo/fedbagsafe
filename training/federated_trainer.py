import gc
import logging
from copy import deepcopy
from typing import Dict, List  # noqa: UP035

import torch

from evaluation.evaluator import ModelEvaluator
from training.aggregation import random_client_selection
from training.attack_schedule import NOTEBOOK_SCHEDULE
from training.local_trainer import LocalTrainer


class FederatedTrainer:
    """main federated learning trainer"""

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

    def get_attack_schedule(self) -> List[str]:
        """get per-round attack schedule"""
        n_rounds = self.config.federated.n_rounds
        if not self.config.attack.enabled:
            return ['benign'] * n_rounds
        if n_rounds <= len(NOTEBOOK_SCHEDULE):
            return list(NOTEBOOK_SCHEDULE)[:n_rounds]
        return list(NOTEBOOK_SCHEDULE) + ['benign'] * (n_rounds - len(NOTEBOOK_SCHEDULE))

    def train(self, data_info: Dict, global_model):
        """main federated training loop"""
        local_trainer = LocalTrainer(self.config, self.device)
        evaluator = ModelEvaluator(self.config, self.device)

        attack_schedule = self.get_attack_schedule()
        model = global_model

        init_acc = evaluator.evaluate(global_model, data_info['dataloaders']['selection_loader'])
        init_loss = evaluator.compute_loss(global_model, data_info['dataloaders']['selection_loader'])
        best_score_prev = init_acc / max(init_loss, 1e-12)

        for round_id in range(self.config.federated.n_rounds):
            attack_state = attack_schedule[round_id]
            self.logger.info(
                f"Round {round_id}/{self.config.federated.n_rounds} | "
                f"Attack: {attack_state}"
            )

            # local training
            node_models = local_trainer.train_round(
                model,
                data_info['dataloaders'],
                round_id,
                attack_state
            )

            # bagged aggregation
            aggregated_models = random_client_selection(
                node_models,
                global_model,
                self.config.dataset.mode,
                self.config.federated.n_nodes,
                use_he=self.config.encryption.enabled,
                he_config={
                    'poly_modulus_degree':
                        self.config.encryption.poly_modulus_degree,
                    'coeff_mod_bit_sizes':
                        self.config.encryption.coeff_mod_bit_sizes,
                    'global_scale': self.config.encryption.global_scale
                } if self.config.encryption.enabled else None # type: ignore
            )

            # eval and selection
            best_model_this_round = None
            best_score_this_round = 0.0

            for agg_id, (agg_model, contributors) in enumerate(aggregated_models):
                self.logger.info(
                    f"Evaluating Aggregated Model {agg_id} "
                    f"from {len(contributors)} nodes"
                )

                acc = evaluator.evaluate(
                    agg_model,
                    data_info['dataloaders']['selection_loader']
                )
                loss = evaluator.compute_loss(
                    agg_model,
                    data_info['dataloaders']['selection_loader']
                )
                score = acc / loss if loss > 0 else 0.0

                self.logger.info(
                    f"Accuracy: {acc:.4f}, Loss: {loss:.4f}, Score: {score:.4f}"
                )

                if attack_state == 'backdoor':
                    b_acc = evaluator.evaluate(
                        agg_model,
                        data_info['dataloaders']['backdoor_loader']
                    )
                    self.logger.info(f"Backdoor Accuracy: {b_acc:.4f}")

                if score > best_score_this_round:
                    best_model_this_round = deepcopy(agg_model)
                    best_score_this_round = score

                del agg_model
                torch.cuda.empty_cache()
                gc.collect()

            del node_models
            gc.collect()

            # fail-safe check
            best_score_prev = self.config.federated.fail_safe_alpha * best_score_prev
            if best_score_this_round > best_score_prev:
                model = best_model_this_round
                best_score_prev = best_score_this_round
                self.logger.info(f"Model updated. Score: {best_score_this_round:.4f}")
            else:
                self.logger.warning("FAIL-SAFE TRIGGERED: Model not updated")

            if best_model_this_round is not None:
                del best_model_this_round
                torch.cuda.empty_cache()
                gc.collect()

        return model