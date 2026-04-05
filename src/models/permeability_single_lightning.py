"""HELM-BERT Permeability Single-Assay Prediction Lightning Module.

Evidential Deep Learning via Normal-Inverse-Gamma (NIG) distribution.
Outputs (gamma, nu, alpha, beta) per sample for uncertainty-aware regression.

References:
    Amini et al. (2020) "Deep Evidential Regression" NeurIPS.
    Soleimany et al. (2021) ACS Central Science.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification

from src.losses.evidential import nig_loss
from src.utils.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)

# NIG output count: gamma, nu, alpha, beta
NIG_NUM_OUTPUTS = 4


@dataclass
class PermeabilitySingleTrainingConfig:
    """Configuration for permeability single-assay training.

    All fields are required - values come from YAML configuration.
    """

    encoder_lr: float
    head_lr: float
    weight_decay: float
    freeze_encoder: bool
    max_epochs: int
    classifier_dropout: float
    classifier_num_layers: int
    encoder_attribute_name: str
    evidence_lambda_coeff: float
    total_steps: int = 0
    warmup_ratio: float = 0.01
    decay_ratio: float = 0.10


class HELMBertPermeabilitySingleLightning(L.LightningModule):
    """PyTorch Lightning module for permeability single-assay evidential prediction.

    Uses HELMBertForSequenceClassification with NIG output head.
    Outputs 4 parameters (gamma, nu, alpha, beta) per sample.

    Args:
        model_name_or_path: HuggingFace Hub model ID or local path (required)
        training_config: PermeabilitySingleTrainingConfig for training settings (required)
        trust_remote_code: Whether to trust remote code from HuggingFace Hub
    """

    def __init__(
        self,
        model_name_or_path: str,
        training_config: PermeabilitySingleTrainingConfig,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        self.training_config = training_config
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code

        self.save_hyperparameters()

        # Load config and set NIG 4-output head
        logger.info(f"Loading model from {self.model_name_or_path}")
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        config.num_labels = NIG_NUM_OUTPUTS
        config.problem_type = "regression"
        config.classifier_num_layers = self.training_config.classifier_num_layers
        config.classifier_dropout = self.training_config.classifier_dropout

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

        logger.info("Permeability Single Model Configuration (Evidential NIG):")
        logger.info(f"  Encoder parameters: {encoder_params:,}")
        logger.info(f"  Classifier parameters: {classifier_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(
            f"  Classifier layers: {self.training_config.classifier_num_layers}"
        )
        logger.info(f"  NIG outputs: {NIG_NUM_OUTPUTS}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass with NIG parameterization.

        Returns:
            Dict with keys:
                predictions: gamma (predicted mean) [batch, 1]
                evidence_params: {gamma, nu, alpha, beta} each [batch]
                uncertainty: {aleatoric, epistemic} each [batch]
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        logits = outputs.logits  # [batch, 4]

        # NIG parameterization with constraints.
        # dtype-aware eps floor: prevents beta/(nu*(alpha-1)) blowup across precisions.
        eps = torch.finfo(logits.dtype).eps
        gamma = logits[:, 0]
        nu = F.softplus(logits[:, 1]) + eps
        alpha = F.softplus(logits[:, 2]) + 1.0 + eps
        beta = F.softplus(logits[:, 3]) + eps

        alpha_minus_one = alpha - 1.0

        return {
            "predictions": gamma.unsqueeze(-1),
            "evidence_params": {
                "gamma": gamma,
                "nu": nu,
                "alpha": alpha,
                "beta": beta,
            },
            "uncertainty": {
                "aleatoric": beta / alpha_minus_one,
                "epistemic": beta / (nu * alpha_minus_one),
            },
        }

    def _compute_loss(
        self, outputs: Dict[str, Any], targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute NIG evidential loss with fixed regularization."""
        params = outputs["evidence_params"]
        targets_flat = targets.squeeze(-1) if targets.dim() > 1 else targets

        return nig_loss(
            y=targets_flat,
            gamma=params["gamma"],
            nu=params["nu"],
            alpha=params["alpha"],
            beta=params["beta"],
            lambda_coeff=self.training_config.evidence_lambda_coeff,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        outputs = self(batch)
        targets = batch["target"].float()

        loss = self._compute_loss(outputs, targets)
        batch_size = targets.size(0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step."""
        outputs = self(batch)
        targets = batch["target"].float()

        loss = self._compute_loss(outputs, targets)

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
                "predictions": outputs["predictions"].detach(),
                "targets": targets.detach(),
                "aleatoric": outputs["uncertainty"]["aleatoric"].detach(),
                "epistemic": outputs["uncertainty"]["epistemic"].detach(),
            }
        )

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step."""
        outputs = self(batch)
        targets = batch["target"].float()

        loss = self._compute_loss(outputs, targets)

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
                "predictions": outputs["predictions"].detach(),
                "targets": targets.detach(),
            }
        )

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Prediction step with uncertainty."""
        outputs = self(batch)

        result = {
            "predictions": outputs["predictions"],
            "targets": batch["target"].float(),
            "uncertainty": {
                k: v.detach() for k, v in outputs["uncertainty"].items()
            },
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

        # Log mean uncertainty (validation only)
        if prefix == "val" and "aleatoric" in outputs[0]:
            all_aleatoric = torch.cat([x["aleatoric"] for x in outputs])
            all_epistemic = torch.cat([x["epistemic"] for x in outputs])
            self.log("val_mean_aleatoric", all_aleatoric.mean(), sync_dist=True)
            self.log("val_mean_epistemic", all_epistemic.mean(), sync_dist=True)

        outputs.clear()

        metrics = compute_regression_metrics(all_predictions, all_targets)

        prog_bar = prefix == "val"
        for metric_name, metric_value in metrics.items():
            if not np.isnan(metric_value):
                self.log(f"{prefix}_{metric_name}", metric_value, prog_bar=prog_bar, sync_dist=True)

        logger.info(f"{prefix} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers with differential learning rates and WSD scheduler."""
        from src.utils.scheduler import create_wsd_scheduler

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

        scheduler = create_wsd_scheduler(
            optimizer,
            total_steps=self.training_config.total_steps,
            warmup_ratio=self.training_config.warmup_ratio,
            decay_ratio=self.training_config.decay_ratio,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save model in HuggingFace format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
