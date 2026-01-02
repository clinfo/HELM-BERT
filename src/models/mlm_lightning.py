"""HELM-BERT MLM Lightning Module.

This module provides a PyTorch Lightning wrapper for training HELMBertForMaskedLM.
Supports both continue pre-training from existing checkpoint and training from scratch.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import torch
from transformers import AutoModelForMaskedLM, AutoConfig

logger = logging.getLogger(__name__)

# Default Hub model
DEFAULT_MODEL = "Flansma/helm-bert"


@dataclass
class MLMTrainingConfig:
    """Configuration for MLM training."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 500
    early_stopping_patience: int = 20
    ignore_index: int = -100


class HELMBertMLMLightning(L.LightningModule):
    """PyTorch Lightning module for HELM-BERT MLM pretraining.

    Supports two modes:
    1. Continue pre-training: Load from existing checkpoint (default)
    2. From scratch: Initialize with random weights

    Args:
        model_name_or_path: HuggingFace Hub model ID or local path for continue pre-training
        from_scratch: If True, initialize with random weights instead of loading pretrained
        model_config: Optional config override (only used when from_scratch=True)
        training_config: MLMTrainingConfig for training settings
        max_epochs: Maximum number of training epochs (for scheduler)

    Example (continue pre-training):
        >>> model = HELMBertMLMLightning()  # Uses Flansma/helm-bert
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule)

    Example (from scratch):
        >>> from transformers import AutoConfig
        >>> config = AutoConfig.from_pretrained("Flansma/helm-bert", trust_remote_code=True)
        >>> config.num_hidden_layers = 12  # Custom architecture
        >>> model = HELMBertMLMLightning(from_scratch=True, model_config=config)
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        from_scratch: bool = False,
        model_config=None,
        training_config: Optional[MLMTrainingConfig] = None,
        max_epochs: int = 500,
    ):
        super().__init__()

        self.training_config = training_config or MLMTrainingConfig()
        self.max_epochs = max_epochs
        self.model_name_or_path = model_name_or_path or DEFAULT_MODEL
        self.from_scratch = from_scratch

        # Initialize model
        if from_scratch:
            # From scratch: use provided config or load from reference
            if model_config is None:
                model_config = AutoConfig.from_pretrained(
                    self.model_name_or_path, trust_remote_code=True
                )
            self.model_config = model_config

            logger.info("Initializing model from scratch")
            logger.info(f"  Hidden size: {model_config.hidden_size}")
            logger.info(f"  Num layers: {model_config.num_hidden_layers}")
            logger.info(f"  Num heads: {model_config.num_attention_heads}")

            self.model = AutoModelForMaskedLM.from_config(
                model_config,
                trust_remote_code=True,
            )
        else:
            # Continue pre-training: load from checkpoint
            logger.info(f"Loading model for continue pre-training from {self.model_name_or_path}")

            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
            )
            self.model_config = self.model.config

        self.save_hyperparameters(
            {
                "learning_rate": self.training_config.learning_rate,
                "weight_decay": self.training_config.weight_decay,
                "model_name_or_path": self.model_name_or_path,
                "from_scratch": self.from_scratch,
                "hidden_size": self.model_config.hidden_size,
                "num_hidden_layers": self.model_config.num_hidden_layers,
                "num_attention_heads": self.model_config.num_attention_heads,
            }
        )

        self._log_model_info()

    def _log_model_info(self) -> None:
        """Log model configuration."""
        config = self.model.config
        logger.info("HELM-BERT MLM Configuration:")
        logger.info(f"  Hidden size: {config.hidden_size}")
        logger.info(f"  Num layers: {config.num_hidden_layers}")
        logger.info(f"  Num heads: {config.num_attention_heads}")
        logger.info(f"  Vocab size: {config.vocab_size}")
        logger.info(f"  Max position: {config.max_position_embeddings}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_emd=True,
            return_dict=True,
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Compute accuracy on masked tokens
        labels = batch["labels"]
        mask = labels != self.training_config.ignore_index
        if mask.any():
            predictions = outputs.logits[mask].argmax(dim=-1)
            accuracy = (predictions == labels[mask]).float().mean()
        else:
            accuracy = torch.tensor(0.0, device=loss.device)

        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch["input_ids"].size(0)
        )
        self.log(
            "train_accuracy",
            accuracy,
            prog_bar=True,
            batch_size=batch["input_ids"].size(0),
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        labels = batch["labels"]
        mask = labels != self.training_config.ignore_index
        if mask.any():
            predictions = outputs.logits[mask].argmax(dim=-1)
            accuracy = (predictions == labels[mask]).float().mean()
        else:
            accuracy = torch.tensor(0.0, device=loss.device)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].size(0),
        )
        self.log(
            "val_accuracy",
            accuracy,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["input_ids"].size(0),
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save model in HuggingFace format.

        Args:
            save_directory: Directory to save the model
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def push_to_hub(self, repo_id: str, **kwargs) -> None:
        """Push model to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub
            **kwargs: Additional arguments for push_to_hub
        """
        self.model.push_to_hub(repo_id, **kwargs)
        logger.info(f"Model pushed to {repo_id}")
