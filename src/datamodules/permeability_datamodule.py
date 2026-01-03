"""Permeability DataModule with train/test files and runtime train/val split."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from .datasets import HELMDataset

logger = logging.getLogger(__name__)


@dataclass
class PermeabilityDataConfig:
    """Configuration for permeability DataModule.

    All fields are required - values come from YAML configuration.
    """

    train_file: str
    test_file: str
    helm_column: str
    target_column: str
    val_ratio: float
    batch_size: int
    max_seq_length: int
    num_workers: int
    pin_memory: bool
    seed: int


class PermeabilityDataModule(L.LightningDataModule):
    """DataModule for permeability regression.

    Args:
        config: PermeabilityDataConfig for data loading settings (required)
        tokenizer: PreTrainedTokenizer instance (required)
    """

    def __init__(
        self,
        config: PermeabilityDataConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Data containers
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Statistics
        self.data_stats: Dict[str, Any] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and split train into train/val."""
        if self.train_dataset is not None:
            return

        train_file = Path(self.config.train_file)
        test_file = Path(self.config.test_file)

        # Load train data
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        train_df = pd.read_csv(train_file)
        logger.info(f"Loaded train: {len(train_df)} samples from {train_file}")

        # Split train → train/val
        train_df, val_df = train_test_split(
            train_df,
            test_size=self.config.val_ratio,
            random_state=self.config.seed,
            shuffle=True,
        )
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")

        # Load test data
        test_df = None
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded test: {len(test_df)} samples from {test_file}")

        # Create datasets
        self.train_dataset = self._create_dataset(train_df)
        self.val_dataset = self._create_dataset(val_df)
        if test_df is not None:
            self.test_dataset = self._create_dataset(test_df)

        # Statistics
        self._compute_statistics(train_df, val_df, test_df)

    def _create_dataset(self, df: pd.DataFrame) -> HELMDataset:
        return HELMDataset(
            sequences=df[self.config.helm_column].tolist(),
            labels=df[self.config.target_column].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )

    def _compute_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
    ) -> None:
        target_col = self.config.target_column
        self.data_stats = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df) if test_df is not None else 0,
        }
        all_targets = pd.concat([train_df[target_col], val_df[target_col]])
        if test_df is not None:
            all_targets = pd.concat([all_targets, test_df[target_col]])
        self.data_stats["mean_target"] = all_targets.mean()
        self.data_stats["std_target"] = all_targets.std()
        logger.info(
            f"Target stats: mean={self.data_stats['mean_target']:.3f}, "
            f"std={self.data_stats['std_target']:.3f}"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )
