import torch
import logging
from copy import deepcopy
import gc
from typing import Dict, List 

from training.local_trainer import LocalTrainer
from training.aggregation import (
    selective_weighted_average_aggregation,
    random_client_selection,
    real_random_client_selection
)
from evaluation.evaluator import ModelEvaluator


class FederatedTrainer:
    """main Federated Learning Trainer"""

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

    def get_attack_schedule(self) -> List[str]:
        """generate attack schedule based on phases"""
        phases = self.config.phases
        schedule = []

        # Phase 1: Stable Initialization
        schedule.extend(['benign'] * phases.stable_initialization)

        # Phase 2: Gradual Attack Introduction
        attack_sequence = ['benign', 'benign', 'backdoor', 'benign', 'benign',
                          'labelflip', 'benign', 'benign', 'modelpoisoning',
                          'benign']
        schedule.extend((attack_sequence * 2)[:phases.gradual_attack_introduction])

        # Phase 3: Moderate Attack Frequency
        attack_sequence_moderate = ['benign', 'backdoor', 'benign',
                                   'modelpoisoning', 'benign', 'labelflip']
        schedule.extend((attack_sequence_moderate * 5)[:phases.moderate_attack_frequency])

        # Phase 4: High-Intensity Attacks
        attack_sequence_intense = ['backdoor', 'modelpoisoning', 'labelflip']
        schedule.extend((attack_sequence_intense * 10)[:phases.high_intensity_attacks])

        # Phase 5: Consecutive Attack Stress Test
        stress_sequence = ['backdoor', 'modelpoisoning', 'labelflip']
        schedule.extend((stress_sequence * 4)[:phases.consecutive_attack_stress_test])

        # Phase 6: Recovery
        schedule.extend(['benign'] * phases.recovery_and_final_validation)

        return schedule[:self.config.federated.n_rounds]

    def train(self, data_info: Dict, global_model):
        """main federated training loop"""
        local_trainer = LocalTrainer(self.config, self.device)
        evaluator = ModelEvaluator(self.config, self.device)

        attack_schedule = self.get_attack_schedule()
        model = global_model
        best_score_prev = 0.0

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

            # aggregation
            if self.config.aggregation.selection_strategy == 'deterministic':
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
                    } if self.config.encryption.enabled else None
                )
            else:
                aggregated_models = real_random_client_selection(
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
                    } if self.config.encryption.enabled else None
                )

            del node_models
            gc.collect()

            # evaluation and Selection
            best_model_this_round = None
            best_score_this_round = 0.0

            for agg_id, (agg_model, contributors) in enumerate(aggregated_models):
                self.logger.info(
                    f"Evaluating Aggregated Model {agg_id} "
                    f"from {len(contributors)} nodes"
                )

                acc = evaluator.evaluate(agg_model, 
                                        data_info['dataloaders']['eval_loader'])
                loss = evaluator.compute_loss(agg_model,
                                             data_info['dataloaders']['eval_loader'])
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

            # fail-safe Mechanism
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