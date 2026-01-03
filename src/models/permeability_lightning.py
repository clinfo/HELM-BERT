"""HELM-BERT Permeability Prediction Lightning Module.

This module provides a PyTorch Lightning wrapper for permeability prediction
using HELMBertForSequenceClassification with MLP head.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

from src.utils.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class PermeabilityTrainingConfig:
    """Configuration for permeability training.

    All fields are required - values come from YAML configuration.
    """

    encoder_lr: float
    head_lr: float
    weight_decay: float
    freeze_encoder: bool
    max_epochs: int
    early_stopping_patience: int
    classifier_dropout: float
    classifier_num_layers: int
    encoder_attribute_name: str


class HELMBertPermeabilityLightning(L.LightningModule):
    """PyTorch Lightning module for permeability prediction.

    Uses HELMBertForSequenceClassification with 2-layer MLP head for regression.

    Args:
        model_name_or_path: HuggingFace Hub model ID or local path (required)
        training_config: PermeabilityTrainingConfig for training settings (required)
        trust_remote_code: Whether to trust remote code from HuggingFace Hub
    """

    def __init__(
        self,
        model_name_or_path: str,
        training_config: PermeabilityTrainingConfig,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        self.training_config = training_config
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code

        self.save_hyperparameters()

        # Load config and update for regression with MLP head
        logger.info(f"Loading model from {self.model_name_or_path}")
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        config.num_labels = 1
        config.problem_type = "regression"
        config.classifier_num_layers = self.training_config.classifier_num_layers
        config.classifier_dropout = self.training_config.classifier_dropout

        # Load model with updated config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            config=config,
            trust_remote_code=self.trust_remote_code,
        )

        # Apply freeze setting
        if self.training_config.freeze_encoder:
            for param in self._encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen (freeze_encoder=True)")

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Storage for metrics
        self.validation_outputs: List[Dict] = []
        self.test_outputs: List[Dict] = []

        self._log_model_info()

    @property
    def _encoder(self) -> nn.Module:
        """Get encoder module using configured attribute name."""
        return getattr(self.model, self.training_config.encoder_attribute_name)

    def _log_model_info(self) -> None:
        """Log model configuration."""
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self._encoder.parameters())
        classifier_params = sum(p.numel() for p in self.model.classifier.parameters())

        logger.info("Permeability Model Configuration:")
        logger.info(f"  Encoder parameters: {encoder_params:,}")
        logger.info(f"  Classifier parameters: {classifier_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(
            f"  Classifier layers: {self.training_config.classifier_num_layers}"
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        return {"predictions": outputs.logits}

    def _compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss."""
        if predictions.dim() > 1 and predictions.size(-1) == 1:
            predictions = predictions.squeeze(-1)
        if targets.dim() > 1 and targets.size(-1) == 1:
            targets = targets.squeeze(-1)
        return self.loss_fn(predictions, targets)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self(batch)
        predictions = outputs["predictions"]
        targets = batch["target"].float()

        loss = self._compute_loss(predictions, targets)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=targets.size(0),
        )

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step."""
        outputs = self(batch)
        predictions = outputs["predictions"]
        targets = batch["target"].float()

        loss = self._compute_loss(predictions, targets)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=targets.size(0),
        )

        self.validation_outputs.append(
            {
                "predictions": predictions.detach(),
                "targets": targets.detach(),
            }
        )

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step."""
        outputs = self(batch)
        predictions = outputs["predictions"]
        targets = batch["target"].float()

        loss = self._compute_loss(predictions, targets)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=targets.size(0),
        )

        self.test_outputs.append(
            {
                "predictions": predictions.detach(),
                "targets": targets.detach(),
            }
        )

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Prediction step."""
        outputs = self(batch)

        result = {
            "predictions": outputs["predictions"],
            "targets": batch["target"].float(),
        }

        for key, value in batch.items():
            if key not in ["input_ids", "attention_mask", "target"]:
                result[key] = value

        return result

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics."""
        self._compute_epoch_metrics(self.validation_outputs, "val")

    def on_test_epoch_end(self) -> None:
        """Compute test metrics."""
        self._compute_epoch_metrics(self.test_outputs, "test")

    def _compute_epoch_metrics(self, outputs: List[Dict], prefix: str) -> None:
        """Compute metrics at epoch end."""
        if not outputs:
            return

        try:
            all_predictions = torch.cat([x["predictions"] for x in outputs])
            all_targets = torch.cat([x["targets"] for x in outputs])
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to concatenate {prefix} outputs: {e}")
            outputs.clear()
            return

        outputs.clear()

        metrics = compute_regression_metrics(all_predictions, all_targets)

        prog_bar = prefix == "val"
        for metric_name, metric_value in metrics.items():
            if not np.isnan(metric_value):
                self.log(f"{prefix}_{metric_name}", metric_value, prog_bar=prog_bar)

        logger.info(f"{prefix} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers with differential learning rates."""
        param_groups = []

        # Only include encoder params if not frozen
        if not self.training_config.freeze_encoder:
            param_groups.append({
                "params": self._encoder.parameters(),
                "lr": self.training_config.encoder_lr,
            })

        param_groups.append({
            "params": self.model.classifier.parameters(),
            "lr": self.training_config.head_lr,
        })

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.training_config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save model in HuggingFace format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
