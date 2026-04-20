import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ImageClassificationBase(pl.LightningModule):
    """base class for image classification models"""

    def __init__(self):
        super().__init__()
        self.last_printed_epoch = -1

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y, label_smoothing=0.05)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        if self.current_epoch != self.last_printed_epoch:
            val_loss = self.trainer.logged_metrics.get('val_loss')
            val_acc = self.trainer.logged_metrics.get('val_acc')

            if val_loss is not None and val_acc is not None:
                print(f"Epoch {self.current_epoch}: "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                self.last_printed_epoch = self.current_epoch

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        acc = (out.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=5e-5,
            weight_decay=1e-2
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            threshold=1e-4,
            min_lr=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }