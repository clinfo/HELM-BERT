"""MLM DataModule for HELM sequences."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

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
    """Configuration for MLM DataModule."""

    data_dir: str = "./data/deduplicated"
    datasets: List[DatasetInfo] = field(
        default_factory=lambda: [
            DatasetInfo("chembl", "chembl_deduplicated.csv", "helm_notation"),
            DatasetInfo("cycpeptmpdb", "cycpeptmpdb_deduplicated.csv", "HELM"),
            DatasetInfo("propedia", "propedia_deduplicated.csv", "Peptide_HELM"),
        ]
    )
    train_ratio: float = 0.9
    batch_size: int = 64
    max_seq_length: int = 512
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 42
    # Span masking config
    mlm_probability: float = 0.15
    mask_ratio: float = 0.8
    random_ratio: float = 0.1
    keep_ratio: float = 0.1
    min_span_length: int = 1
    max_span_length: int = 5
    geometric_p: float = 0.2


class MLMDataModule(L.LightningDataModule):
    """DataModule for HELM Masked Language Modeling.

    Args:
        config: MLMDataConfig for data loading settings
        tokenizer: PreTrainedTokenizer instance (required)

    Example:
        >>> from transformers import AutoTokenizer
        >>> config = MLMDataConfig(batch_size=32)
        >>> tokenizer = AutoTokenizer.from_pretrained("Flansma/helm-bert", trust_remote_code=True)
        >>> datamodule = MLMDataModule(config, tokenizer)
        >>> datamodule.setup()
    """

    def __init__(
        self,
        config: Optional[MLMDataConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__()

        self.config = config or MLMDataConfig()
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required. Use AutoTokenizer.from_pretrained('Flansma/helm-bert', trust_remote_code=True)"
            )
        self.tokenizer = tokenizer

        # Initialize span masking
        masking_config = SpanMaskingConfig(
            mlm_probability=self.config.mlm_probability,
            mask_ratio=self.config.mask_ratio,
            random_ratio=self.config.random_ratio,
            keep_ratio=self.config.keep_ratio,
            min_span_length=self.config.min_span_length,
            max_span_length=self.config.max_span_length,
            geometric_p=self.config.geometric_p,
        )
        self.masking_strategy = SpanMasking(
            config=masking_config,
            mask_token_id=self.tokenizer.mask_token_id,
            vocab_size=self.tokenizer.vocab_size,
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
                    logger.warning(f"Dataset not found: {file_path}")
                    continue

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

            # Create datasets
            self.train_dataset = MLMDataset(
                sequences=train_sequences,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                masking_strategy=self.masking_strategy,
            )

            self.val_dataset = MLMDataset(
                sequences=val_sequences,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
                masking_strategy=self.masking_strategy,
            )

            logger.info(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}")

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.num_workers > 0,
        )
