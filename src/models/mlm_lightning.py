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
from transformers import AutoModelForMaskedLM

logger = logging.getLogger(__name__)


@dataclass
class MLMTrainingConfig:
    """Configuration for MLM training.

    All fields are required - values come from YAML configuration.
    """

    learning_rate: float
    weight_decay: float
    max_epochs: int
    ignore_index: int
    total_steps: int = 0
    warmup_ratio: float = 0.01
    decay_ratio: float = 0.10


class HELMBertMLMLightning(L.LightningModule):
    """PyTorch Lightning module for HELM-BERT MLM pretraining.

    Supports two modes:
    1. Continue pre-training: Load from existing checkpoint (default)
    2. From scratch: Initialize with random weights

    Args:
        model_name_or_path: HuggingFace Hub model ID or local path (required)
        training_config: MLMTrainingConfig for training settings (required)
        from_scratch: If True, initialize with random weights (required)
        model_config: Config override (required when from_scratch=True)
        trust_remote_code: Whether to trust remote code from HuggingFace Hub
        use_emd: Whether to use Enhanced Mask Decoder (DeBERTa feature)
    """

    def __init__(
        self,
        model_name_or_path: str,
        training_config: MLMTrainingConfig,
        from_scratch: bool,
        model_config=None,
        trust_remote_code: bool = True,
        use_emd: bool = True,
    ):
        super().__init__()

        self.training_config = training_config
        self.model_name_or_path = model_name_or_path
        self.from_scratch = from_scratch
        self.use_emd = use_emd

        # Initialize model
        if from_scratch:
            # From scratch: model_config is required
            if model_config is None:
                raise ValueError(
                    "model_config is required when from_scratch=True. "
                    "Create it with AutoConfig and set architecture parameters."
                )
            self.model_config = model_config

            logger.info("Initializing model from scratch")
            logger.info(f"  Hidden size: {model_config.hidden_size}")
            logger.info(f"  Num layers: {model_config.num_hidden_layers}")
            logger.info(f"  Num heads: {model_config.num_attention_heads}")

            self.model = AutoModelForMaskedLM.from_config(
                model_config,
                trust_remote_code=trust_remote_code,
            )
        else:
            # Continue pre-training: load from checkpoint
            logger.info(f"Loading model for continue pre-training from {self.model_name_or_path}")

            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            self.model_config = self.model.config

        self.save_hyperparameters()

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
            use_emd=self.use_emd,
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

        batch_size = batch["input_ids"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

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

        batch_size = batch["input_ids"].size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and WSD scheduler."""
        from src.utils.scheduler import create_wsd_scheduler

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
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
