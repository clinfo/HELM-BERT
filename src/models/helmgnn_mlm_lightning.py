"""HELM-GNN MLM Lightning Module.

End-to-end training: GPS + Transformer via MLM loss only.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import torch

from .configuration_helmgnn import HELMGNNConfig
from .modeling_helmgnn import HELMGNNForMaskedLM

logger = logging.getLogger(__name__)


@dataclass
class HELMGNNMLMTrainingConfig:
    learning_rate: float
    weight_decay: float
    max_epochs: int
    ignore_index: int
    total_steps: int = 0
    warmup_ratio: float = 0.01
    decay_ratio: float = 0.10


class HELMGNNMLMLightning(L.LightningModule):
    """Lightning module for HELM-GNN MLM pre-training.

    Args:
        config: HELMGNNConfig for model architecture.
        training_config: Training hyperparameters.
        id_to_symbol: Mapping from token ID to monomer symbol.
        use_emd: Whether to use Enhanced Mask Decoder.
    """

    def __init__(
        self,
        config: HELMGNNConfig,
        training_config: HELMGNNMLMTrainingConfig,
        id_to_symbol: Optional[Dict[int, str]] = None,
        use_emd: bool = True,
    ):
        super().__init__()
        self.training_config = training_config
        self.use_emd = use_emd
        self.id_to_symbol = id_to_symbol or {}

        self.model = HELMGNNForMaskedLM(config)

        self.save_hyperparameters(ignore=["id_to_symbol"])
        self._log_model_info()

    def _log_model_info(self) -> None:
        config = self.model.config
        logger.info("HELM-GNN MLM Configuration:")
        logger.info(f"  GPS: {config.gps_num_layers} layers, hidden={config.gps_hidden_dim}")
        logger.info(f"  Transformer: {config.num_hidden_layers} layers, hidden={config.hidden_size}")
        logger.info(f"  Vocab size: {config.vocab_size}")
        logger.info(f"  Graph distance buckets: {config.num_graph_distance_buckets}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
            graph_distances=batch.get("graph_distances"),
            id_to_symbol=self.id_to_symbol,
            use_emd=self.use_emd,
            return_dict=True,
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(batch)
        loss = outputs.loss

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
        outputs = self(batch)
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def save_pretrained(self, save_directory: str) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
