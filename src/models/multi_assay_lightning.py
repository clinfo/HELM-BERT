"""HELM-BERT Multi-Assay Permeability Prediction Lightning Module.

Two-head NIG evidential regression for PAMPA and Caco2 assays.
Shared backbone with split output: 4 NIG params per assay head.
Missing labels are masked out during loss computation.
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

NIG_NUM_OUTPUTS = 4
ASSAY_NAMES = ["pampa", "caco2"]
NUM_ASSAYS = len(ASSAY_NAMES)
TOTAL_OUTPUTS = NUM_ASSAYS * NIG_NUM_OUTPUTS


@dataclass
class MultiAssayTrainingConfig:
    """Configuration for multi-assay training."""

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


class HELMBertMultiAssayLightning(L.LightningModule):
    """Multi-assay evidential permeability prediction.

    Shared backbone, output dimension = NUM_ASSAYS * 4 (NIG params per head).
    Logits are split and independently parameterized for each assay.

    Args:
        model_name_or_path: HuggingFace Hub model ID or local path
        training_config: MultiAssayTrainingConfig
        trust_remote_code: Whether to trust remote code from HuggingFace Hub
    """

    def __init__(
        self,
        model_name_or_path: str,
        training_config: MultiAssayTrainingConfig,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        self.training_config = training_config
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code

        self.save_hyperparameters()

        logger.info(f"Loading model from {self.model_name_or_path}")
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        config.num_labels = TOTAL_OUTPUTS
        config.problem_type = "regression"
        config.classifier_num_layers = self.training_config.classifier_num_layers
        config.classifier_dropout = self.training_config.classifier_dropout

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            config=config,
            trust_remote_code=self.trust_remote_code,
        )

        if self.training_config.freeze_encoder:
            for param in self._encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen (freeze_encoder=True)")

        self.validation_outputs: List[Dict] = []
        self.test_outputs: List[Dict] = []

        self._log_model_info()

    @property
    def _encoder(self) -> nn.Module:
        return getattr(self.model, self.training_config.encoder_attribute_name)

    def _log_model_info(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        encoder_params = sum(p.numel() for p in self._encoder.parameters())
        classifier_params = sum(p.numel() for p in self.model.classifier.parameters())

        logger.info("Multi-Assay Model Configuration (Evidential NIG × 2):")
        logger.info(f"  Assays: {ASSAY_NAMES}")
        logger.info(f"  Encoder parameters: {encoder_params:,}")
        logger.info(f"  Classifier parameters: {classifier_params:,}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Total outputs: {TOTAL_OUTPUTS} ({NUM_ASSAYS} × {NIG_NUM_OUTPUTS})")

    @staticmethod
    def _nig_parameterize(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply NIG constraints to 4-dim logits."""
        eps = torch.finfo(logits.dtype).eps
        gamma = logits[:, 0]
        nu = F.softplus(logits[:, 1]) + eps
        alpha = F.softplus(logits[:, 2]) + 1.0 + eps
        beta = F.softplus(logits[:, 3]) + eps
        return {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

    @staticmethod
    def _compute_uncertainty(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        alpha_minus_one = params["alpha"] - 1.0
        return {
            "aleatoric": params["beta"] / alpha_minus_one,
            "epistemic": params["beta"] / (params["nu"] * alpha_minus_one),
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Forward pass producing NIG params for each assay.

        Returns:
            Dict keyed by assay name, each containing predictions, evidence_params, uncertainty.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )
        logits = outputs.logits  # [batch, TOTAL_OUTPUTS]

        result = {}
        for i, name in enumerate(ASSAY_NAMES):
            assay_logits = logits[:, i * NIG_NUM_OUTPUTS : (i + 1) * NIG_NUM_OUTPUTS]
            params = self._nig_parameterize(assay_logits)
            result[name] = {
                "predictions": params["gamma"].unsqueeze(-1),
                "evidence_params": params,
                "uncertainty": self._compute_uncertainty(params),
            }

        return result

    def _compute_loss(
        self, outputs: Dict[str, Dict], batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Compute masked NIG loss per assay. Returns dict with per-assay and total loss."""
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)
        n_tasks = 0

        for name in ASSAY_NAMES:
            mask = batch[f"mask_{name}"].bool()
            if mask.sum() == 0:
                continue

            targets = batch[f"target_{name}"][mask]
            params = outputs[name]["evidence_params"]
            masked_params = {k: v[mask] for k, v in params.items()}

            assay_loss = nig_loss(
                y=targets,
                gamma=masked_params["gamma"],
                nu=masked_params["nu"],
                alpha=masked_params["alpha"],
                beta=masked_params["beta"],
                lambda_coeff=self.training_config.evidence_lambda_coeff,
            )
            losses[name] = assay_loss
            total_loss = total_loss + assay_loss
            n_tasks += 1

        losses["total"] = total_loss / max(n_tasks, 1)
        return losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        losses = self._compute_loss(outputs, batch)

        batch_size = batch["input_ids"].size(0)
        self.log("train_loss", losses["total"], on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for name in ASSAY_NAMES:
            if name in losses:
                self.log(f"train_loss_{name}", losses[name], on_step=False, on_epoch=True, batch_size=batch_size)

        return losses["total"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        outputs = self(batch)
        losses = self._compute_loss(outputs, batch)
        self._log_step_losses(losses, "val", batch)
        self.validation_outputs.append(
            self._collect_step_output(outputs, batch, include_uncertainty=True)
        )

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        outputs = self(batch)
        losses = self._compute_loss(outputs, batch)
        self._log_step_losses(losses, "test", batch)
        self.test_outputs.append(
            self._collect_step_output(outputs, batch, include_uncertainty=False)
        )

    def _log_step_losses(
        self, losses: Dict[str, torch.Tensor], prefix: str, batch: Dict[str, Any]
    ) -> None:
        batch_size = batch["input_ids"].size(0)
        prog_bar = prefix == "val"
        self.log(
            f"{prefix}_loss", losses["total"],
            on_step=False, on_epoch=True, prog_bar=prog_bar, sync_dist=True, batch_size=batch_size,
        )
        for name in ASSAY_NAMES:
            if name in losses:
                self.log(
                    f"{prefix}_loss_{name}", losses[name],
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
                )

    def _collect_step_output(
        self,
        outputs: Dict[str, Dict],
        batch: Dict[str, Any],
        include_uncertainty: bool,
    ) -> Dict[str, torch.Tensor]:
        step_output = {}
        for name in ASSAY_NAMES:
            step_output[f"{name}_predictions"] = outputs[name]["predictions"].detach()
            step_output[f"{name}_targets"] = batch[f"target_{name}"].detach()
            step_output[f"{name}_mask"] = batch[f"mask_{name}"].detach()
            if include_uncertainty:
                step_output[f"{name}_aleatoric"] = outputs[name]["uncertainty"]["aleatoric"].detach()
                step_output[f"{name}_epistemic"] = outputs[name]["uncertainty"]["epistemic"].detach()
        return step_output

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        outputs = self(batch)

        result = {}
        for name in ASSAY_NAMES:
            result[f"{name}_predictions"] = outputs[name]["predictions"]
            result[f"{name}_targets"] = batch[f"target_{name}"].float()
            result[f"{name}_mask"] = batch[f"mask_{name}"]
            result[f"{name}_aleatoric"] = outputs[name]["uncertainty"]["aleatoric"].detach()
            result[f"{name}_epistemic"] = outputs[name]["uncertainty"]["epistemic"].detach()

        return result

    def on_validation_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.validation_outputs, "val")

    def on_test_epoch_end(self) -> None:
        self._compute_epoch_metrics(self.test_outputs, "test")

    def _compute_epoch_metrics(self, outputs: List[Dict], prefix: str) -> None:
        if not outputs:
            return

        for name in ASSAY_NAMES:
            try:
                all_preds = torch.cat([x[f"{name}_predictions"] for x in outputs])
                all_targets = torch.cat([x[f"{name}_targets"] for x in outputs])
                all_mask = torch.cat([x[f"{name}_mask"] for x in outputs])
            except (RuntimeError, ValueError) as e:
                logger.error(f"Failed to concatenate {prefix}/{name} outputs: {e}")
                continue

            valid = all_mask.bool()
            n_valid = valid.sum().item()
            if n_valid < 2:
                continue

            preds_valid = all_preds[valid]
            targets_valid = all_targets[valid]

            if prefix == "val" and f"{name}_aleatoric" in outputs[0]:
                all_aleatoric = torch.cat([x[f"{name}_aleatoric"] for x in outputs])
                all_epistemic = torch.cat([x[f"{name}_epistemic"] for x in outputs])
                self.log(f"val_{name}_mean_aleatoric", all_aleatoric[valid].mean(), sync_dist=True)
                self.log(f"val_{name}_mean_epistemic", all_epistemic[valid].mean(), sync_dist=True)

            metrics = compute_regression_metrics(preds_valid, targets_valid)

            prog_bar = prefix == "val"
            for metric_name, metric_value in metrics.items():
                if not np.isnan(metric_value):
                    self.log(f"{prefix}_{name}_{metric_name}", metric_value, prog_bar=prog_bar, sync_dist=True)

            logger.info(f"{prefix}/{name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

        outputs.clear()

    def configure_optimizers(self) -> Dict[str, Any]:
        from src.utils.scheduler import create_wsd_scheduler

        param_groups = []

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
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
