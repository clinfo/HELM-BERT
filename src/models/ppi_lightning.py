"""HELMGLaM PPI Lightning Module.

Dual-encoder architecture for peptide-protein interaction prediction:
- Drug encoder: HELM-BERT (from HuggingFace Hub)
- Target encoder: ESM-2
- Fusion: Concatenation
- Head: MLP classifier
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
from transformers import AutoModel, AutoConfig

from src.heads.mlp_net import MLPNet
from src.utils.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

# Default Hub model
DEFAULT_DRUG_MODEL = "Flansma/helm-bert"


# ESM-2 hidden sizes by model name
ESM_HIDDEN_SIZES = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": 2560,
    "facebook/esm2_t48_15B_UR50D": 5120,
}


@dataclass
class PPITrainingConfig:
    """Configuration for PPI training."""

    encoder_lr: float = 3e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 200
    early_stopping_patience: int = 20
    mlp_dropout: float = 0.1
    num_classes: int = 1
    pos_weight: Optional[float] = 4.0
    freeze_drug_encoder: bool = True
    freeze_target_encoder: bool = True
    use_cached_embeddings: bool = True
    target_encoder: str = "facebook/esm2_t33_650M_UR50D"


class HELMGLaMLightning(L.LightningModule):
    """PyTorch Lightning module for peptide-protein interaction prediction.

    Uses HELM-BERT for peptide encoding and ESM-2 for protein encoding.

    Args:
        drug_model_path: HuggingFace Hub model ID or local path (default: Flansma/helm-bert)
        training_config: PPITrainingConfig for training settings

    Example:
        >>> config = PPITrainingConfig()
        >>> model = HELMGLaMLightning(training_config=config)  # Uses Flansma/helm-bert
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule)
    """

    def __init__(
        self,
        drug_model_path: Optional[str] = None,
        training_config: Optional[PPITrainingConfig] = None,
    ):
        super().__init__()

        self.training_config = training_config or PPITrainingConfig()
        self.drug_model_path = drug_model_path or DEFAULT_DRUG_MODEL

        self.save_hyperparameters(
            {
                "encoder_lr": self.training_config.encoder_lr,
                "head_lr": self.training_config.head_lr,
                "drug_model_path": self.drug_model_path,
            }
        )

        # Task configuration
        self.num_classes = self.training_config.num_classes
        self.use_cached_embeddings = self.training_config.use_cached_embeddings

        # Initialize components
        self._init_encoders()
        self._init_head()
        self._init_loss_function()

        # Storage for metrics
        self.validation_outputs: List[Dict] = []
        self.test_outputs: List[Dict] = []

    def _init_encoders(self) -> None:
        """Initialize drug and target encoders."""
        # Get dimensions from config
        drug_config = AutoConfig.from_pretrained(
            self.drug_model_path, trust_remote_code=True
        )
        self.drug_dim = drug_config.hidden_size
        self.target_dim = ESM_HIDDEN_SIZES.get(
            self.training_config.target_encoder, 1280
        )

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
            trust_remote_code=True,
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
        """Initialize MLP head (auto dimensions from encoders)."""
        mlp_input_dim = self.drug_dim + self.target_dim

        self.mlp_net = MLPNet(
            input_dim=mlp_input_dim,
            output_dim=self.num_classes,
            hidden_dims=[mlp_input_dim, mlp_input_dim],
            dropout=self.training_config.mlp_dropout,
        )

    def _init_loss_function(self) -> None:
        """Initialize loss function."""
        if self.training_config.pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=torch.tensor([self.training_config.pos_weight]),
            )
            logger.info(f"Using pos_weight={self.training_config.pos_weight}")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        drug_ids: Optional[torch.Tensor] = None,
        drug_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        drug_embedding: Optional[torch.Tensor] = None,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Encode drug
        drug_pooled = self._encode_drug(drug_ids, drug_mask, drug_embedding)
        target_pooled = self._encode_target(target_ids, target_mask, target_embedding)

        # Fuse and predict
        fused = torch.cat([drug_pooled, target_pooled], dim=-1)
        predictions = self.mlp_net(fused)

        return predictions

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
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            return hidden_states.mean(dim=1)

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute loss with optional sample weighting."""
        labels = labels.float()
        predictions = predictions.squeeze(-1) if predictions.dim() > 1 else predictions
        labels = labels.squeeze(-1) if labels.dim() > 1 else labels

        per_sample_loss = self.loss_fn(predictions, labels)

        if "weight" in batch:
            weights = batch["weight"]
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(
                    weights, device=per_sample_loss.device, dtype=per_sample_loss.dtype
                )
            return (per_sample_loss * weights).mean()

        return per_sample_loss.mean()

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with batch unpacking."""
        return self(
            batch.get("drug_ids"),
            batch.get("drug_mask"),
            batch.get("target_ids"),
            batch.get("target_mask"),
            batch.get("drug_embedding"),
            batch.get("target_embedding"),
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        predictions = self._forward_batch(batch)
        loss = self._compute_loss(predictions, batch["label"], batch)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["label"].size(0),
        )

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        predictions = self._forward_batch(batch)
        labels = batch["label"]
        loss = self._compute_loss(predictions, labels, batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=labels.size(0),
        )

        self.validation_outputs.append(
            {
                "predictions": predictions.detach(),
                "labels": labels.detach(),
            }
        )

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        predictions = self._forward_batch(batch)
        labels = batch["label"]
        loss = self._compute_loss(predictions, labels, batch)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=labels.size(0),
        )

        self.test_outputs.append(
            {
                "predictions": predictions.detach(),
                "labels": labels.detach(),
            }
        )

        return loss

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        predictions = self._forward_batch(batch)

        return {
            "predictions": predictions.detach(),
            "labels": batch["label"].detach(),
        }

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
            predictions = torch.cat([x["predictions"] for x in outputs])
            labels = torch.cat([x["labels"] for x in outputs])
        except Exception as e:
            logger.error(f"Failed to concatenate {prefix} outputs: {e}")
            outputs.clear()
            return

        outputs.clear()

        metrics = compute_classification_metrics(predictions, labels, self.num_classes)

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
