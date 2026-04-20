import torch
import logging
from pytorch_lightning import Trainer
import torch.nn.functional as F


class ModelEvaluator:
    """model evaluation utilities"""

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

    def evaluate(self, model, dataloader):
        """evaluate model accuracy"""
        model = model.to(self.device)
        model.eval()

        logger = logging.getLogger("pytorch_lightning")
        old_level = logger.level
        logger.setLevel(logging.WARNING)

        trainer = Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='cpu',
        )

        test_result = trainer.test(model, dataloaders=dataloader, verbose=False)

        logger.setLevel(old_level)

        if test_result and "test_acc" in test_result[0]:
            acc = test_result[0]["test_acc"]
        else:
            acc = 0.0

        model = model.to('cpu')
        return acc

    def compute_loss(self, model, dataloader):
        """compute average loss"""
        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_samples = 0

        logger = logging.getLogger("pytorch_lightning")
        old_level = logger.level
        logger.setLevel(logging.WARNING)

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)

                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                else:
                    loss = F.cross_entropy(outputs, targets, reduction='sum')

                total_loss += loss.item()
                total_samples += targets.size(0)

        logger.setLevel(old_level)

        avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
        model = model.to('cpu')
        return avg_loss