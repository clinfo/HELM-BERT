"""MLM DataModule for HELM sequences with DataCollator."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from .data_collators import DataCollatorForMLM
from .datasets import MLMDataset
from .span_masking import SpanMasking, SpanMaskingConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a single dataset."""

    name: str
    file: str
    helm_column: str


@dataclass
class MLMDataConfig:
    """Configuration for MLM DataModule.

    All fields are required - values come from YAML configuration.
    """

    data_dir: str
    datasets: List[DatasetInfo]
    train_ratio: float
    batch_size: int
    max_seq_length: int
    num_workers: int
    pin_memory: bool
    seed: int
    # Span masking config
    mlm_probability: float
    mask_ratio: float
    random_ratio: float
    keep_ratio: float
    min_span_length: int
    max_span_length: int
    geometric_p: float
    ignore_index: int


class MLMDataModule(L.LightningDataModule):
    """DataModule for HELM Masked Language Modeling.

    Uses DataCollator for dynamic padding and masking at batch level.

    Args:
        config: MLMDataConfig for data loading settings
        tokenizer: PreTrainedTokenizer instance
    """

    def __init__(
        self,
        config: MLMDataConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        # Initialize span masking strategy
        masking_config = SpanMaskingConfig(
            mlm_probability=self.config.mlm_probability,
            mask_ratio=self.config.mask_ratio,
            random_ratio=self.config.random_ratio,
            keep_ratio=self.config.keep_ratio,
            min_span_length=self.config.min_span_length,
            max_span_length=self.config.max_span_length,
            geometric_p=self.config.geometric_p,
            ignore_index=self.config.ignore_index,
        )
        special_token_ids = set(self.tokenizer.all_special_ids)

        self.masking_strategy = SpanMasking(
            config=masking_config,
            mask_token_id=self.tokenizer.mask_token_id,
            vocab_size=self.tokenizer.vocab_size,
            special_token_ids=special_token_ids,
        )

        # DataCollator for dynamic padding and masking
        self._collate_fn = DataCollatorForMLM(
            tokenizer=self.tokenizer,
            masking_strategy=self.masking_strategy,
        )

        # Data containers
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.data_stats: Dict[str, Any] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training/validation."""
        if self.train_dataset is not None:
            return

        if stage == "fit" or stage is None:
            # Load and combine all datasets
            all_sequences = []
            dataset_counts = {}

            data_dir = Path(self.config.data_dir)
            for dataset_info in self.config.datasets:
                name = dataset_info.name
                file_path = data_dir / dataset_info.file
                helm_column = dataset_info.helm_column

                if not file_path.exists():
                    raise FileNotFoundError(f"Dataset not found: {file_path}")

                df = pd.read_csv(file_path)
                df = df.dropna(subset=[helm_column])
                df = df[df[helm_column].str.len() > 0]

                sequences = df[helm_column].tolist()
                dataset_counts[name] = len(sequences)
                all_sequences.extend(sequences)

                logger.info(f"Loaded {name}: {len(sequences)} sequences")

            if not all_sequences:
                raise ValueError("No sequences loaded from any dataset")

            # Store statistics
            self.data_stats["total_sequences"] = len(all_sequences)
            self.data_stats.update(dataset_counts)

            # Split data
            train_sequences, val_sequences = train_test_split(
                all_sequences,
                test_size=1.0 - self.config.train_ratio,
                random_state=self.config.seed,
            )

            # Create datasets (no masking here - collator handles it)
            self.train_dataset = MLMDataset(
                sequences=train_sequences,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
            )

            self.val_dataset = MLMDataset(
                sequences=val_sequences,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
            )

            logger.info(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with collate_fn."""
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
        """Create validation dataloader with collate_fn."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._collate_fn,
        )
