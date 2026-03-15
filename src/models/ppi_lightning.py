"""HELMGLaM PPI Lightning Module.

Evidential Deep Learning via Dirichlet distribution for classification.
Dual-encoder architecture for peptide-protein interaction prediction:
- Drug encoder: HELM-BERT (from HuggingFace Hub)
- Target encoder: ESM-2
- Fusion: Concatenation
- Head: MLP → Dirichlet alpha parameters

References:
    Sensoy et al. (2018) "Evidential Deep Learning to Quantify
    Classification Uncertainty" NeurIPS. arXiv:1806.01768
    Soleimany et al. (2021) ACS Central Science.
"""

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.heads.mlp_net import MLPNet
from src.losses.evidential import dirichlet_loss
from src.utils.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class PPITrainingConfig:
    """Configuration for PPI training.

    All fields are required - values come from YAML configuration.
    """

    encoder_lr: float
    head_lr: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    mlp_dropout: float
    num_classes: int
    freeze_drug_encoder: bool
    freeze_target_encoder: bool
    use_cached_embeddings: bool
    target_encoder: str
    esm_hidden_sizes: Dict[str, int]
    prediction_threshold: float
    evidence_lambda_coeff: float


class HELMGLaMLightning(L.LightningModule):
    """PyTorch Lightning module for evidential PPI prediction.

    Uses HELM-BERT for peptide encoding and ESM-2 for protein encoding.
    Outputs Dirichlet alpha parameters for uncertainty-aware classification.

    Args:
        drug_model_path: HuggingFace Hub model ID or local path (required)
        training_config: PPITrainingConfig for training settings (required)
        trust_remote_code: Whether to trust remote code for drug encoder
    """

    def __init__(
        self,
        drug_model_path: str,
        training_config: PPITrainingConfig,
        trust_remote_code: bool = True,
    ):
        super().__init__()

        self.training_config = training_config
        self.drug_model_path = drug_model_path
        self.trust_remote_code = trust_remote_code

        self.save_hyperparameters()

        # Task configuration
        self.num_classes = self.training_config.num_classes
        self.use_cached_embeddings = self.training_config.use_cached_embeddings

        # Initialize components
        self._init_encoders()
        self._init_head()

        # Storage for metrics
        self.validation_outputs: List[Dict] = []
        self.test_outputs: List[Dict] = []

    def _init_encoders(self) -> None:
        """Initialize drug and target encoders."""
        # Get dimensions from config
        drug_config = AutoConfig.from_pretrained(
            self.drug_model_path, trust_remote_code=self.trust_remote_code
        )
        self.drug_dim = drug_config.hidden_size

        # Get ESM-2 hidden size from configuration
        target_encoder_name = self.training_config.target_encoder
        if target_encoder_name not in self.training_config.esm_hidden_sizes:
            raise ValueError(
                f"Unknown ESM-2 model: {target_encoder_name}. "
                f"Add it to esm_hidden_sizes in ppi.yaml. "
                f"Known models: {list(self.training_config.esm_hidden_sizes.keys())}"
            )
        self.target_dim = self.training_config.esm_hidden_sizes[target_encoder_name]

        # Skip encoder creation if using cached embeddings
        if self.use_cached_embeddings:
            logger.info("Using cached embeddings - encoders not initialized")
            logger.info(f"  Drug embedding dim: {self.drug_dim}")
            logger.info(f"  Target embedding dim: {self.target_dim}")
            return

        logger.info("Creating encoders for real-time encoding")

        # Create drug encoder (HELM-BERT from Hub)
        self.drug_encoder = AutoModel.from_pretrained(
            self.drug_model_path,
            trust_remote_code=self.trust_remote_code,
        )

        # Create target encoder (ESM-2)
        self.target_encoder = AutoModel.from_pretrained(
            self.training_config.target_encoder
        )

        # Apply freeze settings
        if self.training_config.freeze_drug_encoder:
            self.drug_encoder.eval()
            for param in self.drug_encoder.parameters():
                param.requires_grad = False
            logger.info(f"  Drug encoder: HELM-BERT (dim={self.drug_dim}) [FROZEN]")
        else:
            self.drug_encoder.train()
            logger.info(f"  Drug encoder: HELM-BERT (dim={self.drug_dim}) [TRAINABLE]")

        if self.training_config.freeze_target_encoder:
            self.target_encoder.eval()
            for param in self.target_encoder.parameters():
                param.requires_grad = False
            logger.info(f"  Target encoder: ESM-2 (dim={self.target_dim}) [FROZEN]")
        else:
            self.target_encoder.train()
            logger.info(f"  Target encoder: ESM-2 (dim={self.target_dim}) [TRAINABLE]")

    def _init_head(self) -> None:
        """Initialize MLP head with Dirichlet output dimension."""
        mlp_input_dim = self.drug_dim + self.target_dim

        self.mlp_net = MLPNet(
            input_dim=mlp_input_dim,
            output_dim=self.num_classes,
            hidden_dims=[mlp_input_dim, mlp_input_dim],
            dropout=self.training_config.mlp_dropout,
        )

        logger.info(
            f"  MLP head: {mlp_input_dim} → {self.num_classes} "
            f"(Dirichlet K={self.num_classes})"
        )

    def forward(
        self,
        drug_ids: Optional[torch.Tensor] = None,
        drug_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        drug_embedding: Optional[torch.Tensor] = None,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with Dirichlet parameterization.

        Returns:
            Dict with keys:
                alpha: Dirichlet concentration parameters [batch, K]
                probs: Expected class probabilities [batch, K]
                uncertainty: Total uncertainty K/S [batch]
        """
        drug_pooled = self._encode_drug(drug_ids, drug_mask, drug_embedding)
        target_pooled = self._encode_target(target_ids, target_mask, target_embedding)

        fused = torch.cat([drug_pooled, target_pooled], dim=-1)
        raw_output = self.mlp_net(fused)  # [batch, K]

        # Dirichlet parameterization: alpha > 1
        alpha = F.softplus(raw_output) + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / S
        uncertainty = float(self.num_classes) / S.squeeze(-1)

        return {
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty,
        }

    def _encode_drug(
        self,
        drug_ids: Optional[torch.Tensor],
        drug_mask: Optional[torch.Tensor],
        drug_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode drug sequence."""
        if drug_embedding is not None:
            return drug_embedding

        if not hasattr(self, "drug_encoder"):
            raise RuntimeError(
                "Drug encoder not initialized. Provide drug_embedding or set use_cached_embeddings=false."
            )

        freeze = self.training_config.freeze_drug_encoder
        if freeze:
            self.drug_encoder.eval()

        ctx = torch.inference_mode() if freeze else nullcontext()
        with ctx:
            outputs = self.drug_encoder(
                input_ids=drug_ids,
                attention_mask=drug_mask,
                return_dict=True,
            )
            return outputs.pooler_output

    def _encode_target(
        self,
        target_ids: Optional[torch.Tensor],
        target_mask: Optional[torch.Tensor],
        target_embedding: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode target sequence."""
        if target_embedding is not None:
            return target_embedding

        if not hasattr(self, "target_encoder"):
            raise RuntimeError(
                "Target encoder not initialized. Provide target_embedding or set use_cached_embeddings=false."
            )

        freeze = self.training_config.freeze_target_encoder
        if freeze:
            self.target_encoder.eval()

        ctx = torch.inference_mode() if freeze else nullcontext()
        with ctx:
            outputs = self.target_encoder(
                input_ids=target_ids,
                attention_mask=target_mask,
                return_dict=True,
            )
            # Mean pooling for ESM-2
            hidden_states = outputs.last_hidden_state
            if target_mask is not None:
                mask_expanded = (
                    target_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                )
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=torch.finfo(hidden_states.dtype).eps)
                return sum_embeddings / sum_mask
            return hidden_states.mean(dim=1)

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dirichlet evidential loss with fixed KL regularization."""
        labels_flat = labels.float()
        labels_flat = labels_flat.squeeze(-1) if labels_flat.dim() > 1 else labels_flat
        y_onehot = F.one_hot(labels_flat.long(), num_classes=self.num_classes).float()

        return dirichlet_loss(
            y_onehot=y_onehot,
            alpha=outputs["alpha"],
            lambda_coeff=self.training_config.evidence_lambda_coeff,
        )

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with explicit batch unpacking.

        Detects mode based on batch contents:
        - Embedding mode: drug_embedding, target_embedding present
        - Tokenized mode: drug_ids, target_ids present
        """
        if "drug_embedding" in batch:
            return self(
                drug_embedding=batch["drug_embedding"],
                target_embedding=batch["target_embedding"],
            )
        return self(
            drug_ids=batch["drug_ids"],
            drug_mask=batch["drug_mask"],
            target_ids=batch["target_ids"],
            target_mask=batch["target_mask"],
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        outputs = self._forward_batch(batch)
        labels = batch["label"]
        loss = self._compute_loss(outputs, labels)

        batch_size = labels.size(0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        outputs = self._forward_batch(batch)
        labels = batch["label"]
        loss = self._compute_loss(outputs, labels)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.size(0),
        )

        # Store log(alpha) as predictions: softmax(log(alpha)) = alpha/S = probs
        self.validation_outputs.append(
            {
                "predictions": torch.log(outputs["alpha"]).detach(),
                "targets": labels.detach(),
                "uncertainty": outputs["uncertainty"].detach(),
            }
        )

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        outputs = self._forward_batch(batch)
        labels = batch["label"]
        loss = self._compute_loss(outputs, labels)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=labels.size(0),
        )

        # Store log(alpha) as predictions: softmax(log(alpha)) = alpha/S = probs
        self.test_outputs.append(
            {
                "predictions": torch.log(outputs["alpha"]).detach(),
                "targets": labels.detach(),
            }
        )

        return loss

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step with uncertainty."""
        outputs = self._forward_batch(batch)

        return {
            "predictions": outputs["probs"][:, 1].detach(),
            "targets": batch["label"].detach(),
            "uncertainty": outputs["uncertainty"].detach(),
        }

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics."""
        self._compute_epoch_metrics(self.validation_outputs, "val")

    def on_test_epoch_end(self) -> None:
        """Compute test metrics."""
        self._compute_epoch_metrics(self.test_outputs, "test")

    def _compute_epoch_metrics(self, outputs: List[Dict], prefix: str) -> None:
        """Compute metrics at epoch end.

        Predictions are log(alpha) [N, K]. compute_classification_metrics
        applies softmax → recovers alpha/S = correct Dirichlet probabilities.
        """
        if not outputs:
            return

        try:
            predictions = torch.cat([x["predictions"] for x in outputs])
            targets = torch.cat([x["targets"] for x in outputs])
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to concatenate {prefix} outputs: {e}")
            outputs.clear()
            return

        # Log mean uncertainty (validation only)
        if prefix == "val" and "uncertainty" in outputs[0]:
            all_uncertainty = torch.cat([x["uncertainty"] for x in outputs])
            self.log("val_mean_uncertainty", all_uncertainty.mean())

        outputs.clear()

        metrics = compute_classification_metrics(
            predictions, targets, self.num_classes, self.training_config.prediction_threshold
        )

        prog_bar = prefix == "val"
        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(f"{prefix}_{name}", value, prog_bar=prog_bar)

        logger.info(
            f"{prefix} - ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, "
            f"Bal-Acc: {metrics['balanced_accuracy']:.4f}"
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with differential learning rates."""
        head_params = list(self.mlp_net.parameters())

        # Collect encoder params
        drug_params = []
        target_params = []
        if hasattr(self, "drug_encoder"):
            drug_params = [p for p in self.drug_encoder.parameters() if p.requires_grad]
        if hasattr(self, "target_encoder"):
            target_params = [
                p for p in self.target_encoder.parameters() if p.requires_grad
            ]

        encoder_params = drug_params + target_params
        head_params = [p for p in head_params if p.requires_grad]

        # Build parameter groups
        param_groups = []
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.training_config.encoder_lr}
            )
        param_groups.append({"params": head_params, "lr": self.training_config.head_lr})

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.training_config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.training_config.max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save drug encoder in HuggingFace format."""
        if hasattr(self, "drug_encoder"):
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            self.drug_encoder.save_pretrained(save_path)
            logger.info(f"Drug encoder saved to {save_path}")
