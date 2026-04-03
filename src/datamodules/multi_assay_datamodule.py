"""Multi-Assay Permeability DataModule with per-assay targets and masks."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from .data_collators import DataCollatorForMultiAssayRegression
from .datasets import MultiAssayHELMDataset

logger = logging.getLogger(__name__)


@dataclass
class MultiAssayDataConfig:
    """Configuration for multi-assay DataModule.

    All fields are required - values come from YAML configuration.
    """

    train_file: str
    test_file: str
    helm_column: str
    assay_columns: List[str]
    val_ratio: float
    batch_size: int
    max_seq_length: int
    num_workers: int
    pin_memory: bool
    seed: int


class MultiAssayDataModule(L.LightningDataModule):
    """DataModule for multi-assay permeability regression.

    Loads multiple assay target columns with missing-value masks.

    Args:
        config: MultiAssayDataConfig for data loading settings
        tokenizer: PreTrainedTokenizer instance
    """

    def __init__(
        self,
        config: MultiAssayDataConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.data_stats: Dict[str, Any] = {}

        # Lowercase assay names for batch keys
        self.assay_keys = [col.lower() for col in self.config.assay_columns]

        self._collate_fn = DataCollatorForMultiAssayRegression(
            tokenizer=self.tokenizer,
            assay_names=self.assay_keys,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        train_file = Path(self.config.train_file)
        test_file = Path(self.config.test_file)

        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        train_df = pd.read_csv(train_file)
        logger.info(f"Loaded train: {len(train_df)} samples from {train_file}")

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.config.val_ratio,
            random_state=self.config.seed,
            shuffle=True,
        )
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")

        test_df = None
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded test: {len(test_df)} samples from {test_file}")

        self.train_dataset = self._create_dataset(train_df)
        self.val_dataset = self._create_dataset(val_df)
        if test_df is not None:
            self.test_dataset = self._create_dataset(test_df)

        self._compute_statistics(train_df, val_df, test_df)

    def _create_dataset(self, df: pd.DataFrame) -> MultiAssayHELMDataset:
        assay_values = {}
        for col, key in zip(self.config.assay_columns, self.assay_keys):
            assay_values[key] = df[col].tolist()

        return MultiAssayHELMDataset(
            sequences=df[self.config.helm_column].tolist(),
            assay_values=assay_values,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )

    def _compute_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
    ) -> None:
        self.data_stats = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df) if test_df is not None else 0,
        }

        for col, key in zip(self.config.assay_columns, self.assay_keys):
            all_values = pd.concat([train_df[col], val_df[col]])
            if test_df is not None:
                all_values = pd.concat([all_values, test_df[col]])
            valid = all_values.dropna()
            self.data_stats[f"{key}_count"] = len(valid)
            self.data_stats[f"{key}_mean"] = valid.mean()
            self.data_stats[f"{key}_std"] = valid.std()
            logger.info(
                f"{key} stats: n={len(valid)}, mean={valid.mean():.3f}, std={valid.std():.3f}"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
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
            collate_fn=self._collate_fn,
        )
