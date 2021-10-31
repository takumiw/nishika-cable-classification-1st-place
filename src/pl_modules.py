"""Training modules"""
from logging import getLogger
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.custom_criterions import LabelSmoothingCrossEntropy
from src.custom_schedulers import LinearWarmupCosineAnnealingLR
from src.pl_create_model import create_model

logger = getLogger(__name__)


class LitModule(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super(LitModule, self).__init__()
        self.model = create_model(
            model_name=hparams.model.backbone, pretrained=hparams.model.pretrained, hidden_dim=hparams.model.hidden_dim
        )
        if hparams.training.criterion.loss_function == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
            logger.debug("Set criterion -- CrossEntropyLoss")
        elif hparams.training.criterion.loss_function == "LabelSmoothingCrossEntropy":
            self.criterion = LabelSmoothingCrossEntropy(smoothing=hparams.training.criterion.smoothing)
            logger.debug(
                f"Set criterion -- LabelSmoothingCrossEntropy with smoothing={hparams.training.criterion.smoothing}"
            )
        else:
            raise NameError(f"Loss function {hparams.training.loss_function} is not defined")
        self.save_hyperparameters(hparams)
        self.verbose_epochs = hparams.training.verbose_epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(axis=1)
        loss = self.criterion(logits, y)
        return {
            "preds": pred.detach(),
            "targets": y.detach(),
            "loss": loss,
        }

    def training_epoch_end(self, outputs):
        epoch = int(self.current_epoch)
        avg_loss = torch.stack([o["loss"] for o in outputs]).mean().item()
        preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()
        targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="micro")
        logs = {"epoch": epoch, "train/loss": avg_loss, "train/acc": acc, "train/f1": f1}

        self.log_dict(logs)
        if (epoch + 1) % self.verbose_epochs == 0:
            logger.debug(logs)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(axis=1)
        loss = self.criterion(logits, y)
        return {
            "preds": pred.detach(),
            "targets": y.detach(),
            "loss": loss.detach(),
        }

    def validation_epoch_end(self, outputs):
        epoch = int(self.current_epoch)
        avg_loss = torch.stack([o["loss"] for o in outputs]).mean().item()
        preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()
        targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="micro")
        logs = {"epoch": epoch, "valid/loss": avg_loss, "valid/acc": acc, "valid/f1": f1}

        self.log_dict(logs)
        if (epoch + 1) % self.verbose_epochs == 0:
            logger.debug(logs)

    def configure_optimizers(self):
        if self.hparams.training.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.hparams.training.learning_rate,
                weight_decay=self.hparams.training.weight_decay,
            )
            logger.debug(
                f"Set optimizer -- AdamW with lr={self.hparams.training.learning_rate}, weight_decay={self.hparams.training.weight_decay}"
            )
        else:
            raise NameError(f"Optimizer {self.hparams.training.optimizer} is not defined")

        if self.hparams.lrate_scheduler.scheduler == "ReduceLROnPlateau":
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.hparams.lrate_scheduler.factor,
                    patience=self.hparams.lrate_scheduler.patience,
                ),
                "monitor": "valid_loss",
            }
            logger.debug("Set scheduler -- ReduceLROnPlateau")
        elif self.hparams.lrate_scheduler.scheduler == "CosineAnnealingLR":
            scheduler = {
                "scheduler": CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.lrate_scheduler.T_max,
                    eta_min=self.hparams.lrate_scheduler.eta_min,
                ),
                "interval": "step",
            }
            logger.debug(f"Set scheduler -- CosineAnnealingLR with T_max={self.hparams.lrate_scheduler.T_max}")
        elif self.hparams.lrate_scheduler.scheduler == "LinearWarmupCosineAnnealingLR":
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.lrate_scheduler.T_max,
                    warmup_epochs=self.hparams.lrate_scheduler.warmup_epochs,
                    warmup_start_lr=self.hparams.lrate_scheduler.warmup_start_lr,
                    m_mul=self.hparams.lrate_scheduler.m_mul,
                    eta_min=self.hparams.lrate_scheduler.eta_min,
                ),
                "interval": "step",
            }
            logger.debug(
                f"Set scheduler -- LinearWarmupCosineAnnealingLR with T_max={self.hparams.lrate_scheduler.T_max} warmup_epochs={self.hparams.lrate_scheduler.warmup_epochs} warmup_start_lr={self.hparams.lrate_scheduler.warmup_start_lr}"
            )
        else:
            raise NameError(f"Scheduler {self.hparams.lrate_scheduler.scheduler} is not defined")
        return [optimizer], [scheduler]
