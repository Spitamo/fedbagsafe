import logging
import sys
import torch
from pathlib import Path

from config.config_parser import ConfigParser, create_argument_parser, process_cli_args
from data.dataset_loader import DatasetFactory, create_dataloaders
from models.model_factory import ModelFactory
from training.federated_trainer import FederatedTrainer
from evaluation.evaluator import ModelEvaluator
from utils.seed_utils import seed_everything
from utils.device_utils import get_device
from utils.logging_utils import setup_logging


def main():
    #parse arguments
    arg_parser = create_argument_parser()
    args = arg_parser.parse_args()

    #parse configuration
    cli_updates = process_cli_args(args)
    config = ConfigParser.parse(args.config, **cli_updates)

   
    seed_everything(config.seed)
    device = get_device(config.device.type)
    logger = setup_logging(
        config.logging.level,
        config.logging.verbose,
        config.logging.log_file if config.logging.save_logs else None
    )

    logger.info("=" * 80)
    logger.info("Federated Learning Framework")
    logger.info("=" * 80)
    logger.info(f"Dataset: {config.dataset.mode}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Clients: {config.federated.n_nodes}")
    logger.info(f"Rounds: {config.federated.n_rounds}")
    logger.info(f"Homomorphic Encryption: {config.encryption.enabled}")
    logger.info(f"Attacks Enabled: {config.attack.enabled}")
    logger.info(f"Device: {config.device.type}")
    logger.info("=" * 80)

    # prepare data
    logger.info("Preparing federated data...")
    fed_data = DatasetFactory.prepare_federated_data(
        config.dataset.mode,
        config.federated.n_nodes,
        config.federated.n_rounds,
        config.federated.min_samples,
        config.federated.max_samples,
        config.dataset.root
    )

    # create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        config.dataset.mode,
        fed_data['nodes_data'],
        fed_data['nodes_labels'],
        fed_data['central_data'],
        fed_data['central_labels'],
        fed_data['test_data'],
        fed_data['test_labels'],
        config.federated.n_nodes,
        config.federated.n_rounds,
        config.training.batch_size,
        config.training.eval_batch_size,
        config.training.num_workers,
        config.attack.target_label
    )

    # create or load model
    logger.info(f"Creating {config.model.name} model...")
    
    if config.model.load_pretrained:
        logger.info(f"Attempting to load pretrained model from checkpoint...")
        global_model = ModelFactory.load_pretrained_model(
            config.model.name,
            config.dataset.mode,
            fed_data['n_classes'],
            checkpoint_dir=config.model.checkpoint_dir,
            dropout=config.model.dropout
        )
    else:
        logger.info("Creating new model from scratch...")
        global_model = ModelFactory.create_model(
            config.model.name,
            fed_data['n_classes'],
            config.model.dropout
        )
    
    global_model = global_model.to(device)

    # initial model evaluation (pretrained model)
    evaluator = ModelEvaluator(config, device)
    logger.info("Testing initial pretrained model...")
    initial_model_acc = evaluator.evaluate(global_model, dataloaders['eval_loader'])
    initial_model_backdoor = evaluator.evaluate(global_model, dataloaders['backdoor_loader'])
    logger.info(f"Initial Model Accuracy: {initial_model_acc:.4f}")
    logger.info(f"Initial Backdoor Success Rate: {initial_model_backdoor:.4f}")

    # prepare data info
    data_info = {
        'dataloaders': dataloaders,
        'n_classes': fed_data['n_classes']
    }

    # train
    logger.info("Starting federated training...")
    trainer = FederatedTrainer(config, device)
    final_model = trainer.train(data_info, global_model)

    logger.info("Training completed!")

    # final evaluation
    logger.info("=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)
    
    final_acc = evaluator.evaluate(final_model, dataloaders['eval_loader'])
    final_backdoor = evaluator.evaluate(final_model, dataloaders['backdoor_loader'])
    
    logger.info(f"Final Model Accuracy: {final_acc:.4f}")
    logger.info(f"Final Backdoor Success Rate: {final_backdoor:.4f}")
    
    # calculate improvements
    acc_improvement = final_acc - initial_model_acc
    backdoor_change = final_backdoor - initial_model_backdoor
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PERFORMANCE METRICS")
    logger.info("=" * 80)
    
    # main Accuracy
    logger.info(f"\n Clean Accuracy:")
    logger.info(f"  Initial:    {initial_model_acc:.4f}")
    logger.info(f"  Final:      {final_acc:.4f}")
    if acc_improvement >= 0:
        logger.info(f"  Change:     +{acc_improvement:.4f} ({acc_improvement*100:+.2f}%)")
    else:
        logger.info(f"  Change:     {acc_improvement:.4f} ({acc_improvement*100:+.2f}%)")
    
    # backdoor Rate
    logger.info(f"\n Backdoor Attack Success Rate:")
    logger.info(f"  Initial:    {initial_model_backdoor:.4f} ({initial_model_backdoor*100:.2f}%)")
    logger.info(f"  Final:      {final_backdoor:.4f} ({final_backdoor*100:.2f}%)")
    backdoor_change_msg = f"  Change:   {backdoor_change:+.4f} ({backdoor_change*100:+.2f}%)"
    logger.info(backdoor_change_msg)
    
    logger.info("=" * 80)


if __name__ == '__main__':
    main()