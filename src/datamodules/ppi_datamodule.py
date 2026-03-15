"""PPI DataModule for peptide-protein interaction tasks.

Supports dual-encoder setup with HELM-BERT for peptides and ESM-2 for proteins.
Uses DataCollator for dynamic padding at batch level.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.utils.embedding_cache import EmbeddingCache
from src.utils.embedding_generator import generate_drug_embeddings, generate_target_embeddings

from .data_collators import DataCollatorForPPI, DataCollatorForPPIEmbedding
from .dual_sequence_dataset import DualSequenceDataset
from .embedding_only_dataset import EmbeddingOnlyDataset

logger = logging.getLogger(__name__)


@dataclass
class PPIDataConfig:
    """Configuration for PPI DataModule.

    All fields are required - values come from YAML configuration.
    """

    # Data files
    train_file: str
    test_file: str

    # Column names
    drug_column: str
    target_column: str
    label_column: str

    # Target encoder (ESM-2)
    target_encoder: str

    # Data loading
    val_ratio: float
    batch_size: int
    max_drug_length: int
    max_target_length: int
    num_workers: int
    pin_memory: bool
    seed: int

    # Cached embeddings
    use_cached_embeddings: bool
    cache_dir: str

    # Encoder paths (for auto-generation of cache)
    drug_encoder: str
    trust_remote_code: bool

    # Cache naming (for EmbeddingCache lookup)
    cache_drug_encoder_name: str
    cache_target_encoder_name: str
    cache_dataset_type: str


class PPIDataModule(L.LightningDataModule):
    """DataModule for peptide-protein interaction prediction.

    Uses DataCollator for dynamic padding at batch level.
    Batch keys are fixed: drug_ids, drug_mask, target_ids, target_mask.

    Args:
        config: PPIDataConfig with data loading settings
        drug_tokenizer: PreTrainedTokenizer for drug sequences
        target_tokenizer: Optional ESM tokenizer for target sequences
    """

    def __init__(
        self,
        config: PPIDataConfig,
        drug_tokenizer: PreTrainedTokenizer,
        target_tokenizer: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config

        # Tokenizers
        self._drug_tokenizer = drug_tokenizer
        self._target_tokenizer = target_tokenizer

        # Data containers
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Statistics
        self.data_stats: Dict[str, Any] = {}

        # Pre-computed embeddings
        self.drug_embeddings: Optional[Dict[str, torch.Tensor]] = None
        self.target_embeddings: Optional[Dict[str, torch.Tensor]] = None

        # DataCollator (initialized in setup)
        self._collate_fn: Optional[Any] = None

    @property
    def drug_tokenizer(self) -> PreTrainedTokenizer:
        """Get drug tokenizer."""
        return self._drug_tokenizer

    @property
    def target_tokenizer(self):
        """Lazy initialization of target tokenizer (ESM-2)."""
        if self._target_tokenizer is None:
            self._target_tokenizer = AutoTokenizer.from_pretrained(
                self.config.target_encoder
            )
        return self._target_tokenizer

    def _use_cached_embeddings(self) -> bool:
        """Check if cached embeddings mode is enabled."""
        return self.config.use_cached_embeddings

    def _use_embeddings_only(self) -> bool:
        """Check if we can use embedding-only mode."""
        return (
            self._use_cached_embeddings()
            and self.drug_embeddings is not None
            and self.target_embeddings is not None
        )

    def _load_embeddings(
        self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame]
    ) -> None:
        """Load pre-computed embeddings if enabled."""
        if not self._use_cached_embeddings():
            return

        cache_dir = Path(self.config.cache_dir)
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {cache_dir}. "
                "Run generate_ppi_embeddings.py first."
            )

        drug_col = self.config.drug_column
        target_col = self.config.target_column

        all_drugs = set(train_df[drug_col].unique())
        all_targets = set(train_df[target_col].unique())
        if test_df is not None:
            all_drugs.update(test_df[drug_col].unique())
            all_targets.update(test_df[target_col].unique())

        embedding_cache = EmbeddingCache(Path(self.config.cache_dir))

        logger.info(f"Loading drug embeddings ({self.config.cache_drug_encoder_name})...")
        self.drug_embeddings = embedding_cache.load_embeddings(
            encoder_name=self.config.cache_drug_encoder_name,
            dataset_type=self.config.cache_dataset_type,
            sequences=list(all_drugs),
            role="drug",
        )
        logger.info(f"Loaded {len(self.drug_embeddings)} drug embeddings")

        logger.info(f"Loading target embeddings ({self.config.cache_target_encoder_name})...")
        self.target_embeddings = embedding_cache.load_embeddings(
            encoder_name=self.config.cache_target_encoder_name,
            dataset_type=self.config.cache_dataset_type,
            sequences=list(all_targets),
            role="target",
        )
        logger.info(f"Loaded {len(self.target_embeddings)} target embeddings")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and split data."""
        if self.train_dataset is not None:
            return  # Already setup

        train_file = Path(self.config.train_file)
        test_file = Path(self.config.test_file)
        drug_col = self.config.drug_column
        target_col = self.config.target_column
        label_col = self.config.label_column

        # Load train data
        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        full_train_df = pd.read_csv(train_file)
        logger.info(f"Loaded train: {len(full_train_df)} samples from {train_file}")

        # Split train → train/val (stratified)
        train_df, val_df = train_test_split(
            full_train_df,
            test_size=self.config.val_ratio,
            random_state=self.config.seed,
            stratify=full_train_df[label_col],
        )
        logger.info(
            f"Split: {len(train_df)} train, {len(val_df)} val "
            f"(val_ratio={self.config.val_ratio})"
        )

        # Load test data
        test_df = None
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded test: {len(test_df)} samples from {test_file}")
        else:
            logger.warning(f"Test file not found: {test_file}")

        # Load embeddings if configured
        self._load_embeddings(full_train_df, test_df)

        # Initialize DataCollator
        if self._use_embeddings_only():
            self._collate_fn = DataCollatorForPPIEmbedding()
        else:
            self._collate_fn = DataCollatorForPPI(
                drug_tokenizer=self.drug_tokenizer,
                target_tokenizer=self.target_tokenizer,
            )

        # Create datasets
        self.train_dataset = self._create_dataset(
            train_df, drug_col, target_col, label_col
        )
        self.val_dataset = self._create_dataset(val_df, drug_col, target_col, label_col)
        if test_df is not None:
            self.test_dataset = self._create_dataset(
                test_df, drug_col, target_col, label_col
            )

        # Compute statistics
        self._compute_statistics(train_df, val_df, test_df, label_col)

        # Log overlap check
        self._check_overlaps(train_df, val_df, test_df, drug_col, target_col)

    def _create_dataset(
        self,
        df: pd.DataFrame,
        drug_col: str,
        target_col: str,
        label_col: str,
    ) -> Dataset:
        """Create dataset from dataframe."""
        drug_seqs = df[drug_col].tolist()
        target_seqs = df[target_col].tolist()
        labels = [float(x) for x in df[label_col].tolist()]

        if self._use_embeddings_only():
            return EmbeddingOnlyDataset(
                sequences_a=drug_seqs,
                sequences_b=target_seqs,
                labels=labels,
                embeddings_a=self.drug_embeddings,
                embeddings_b=self.target_embeddings,
            )
        else:
            return DualSequenceDataset(
                sequences_a=drug_seqs,
                sequences_b=target_seqs,
                labels=labels,
                tokenizer_a=self.drug_tokenizer,
                tokenizer_b=self.target_tokenizer,
                max_length_a=self.config.max_drug_length,
                max_length_b=self.config.max_target_length,
                embeddings_a=self.drug_embeddings,
                embeddings_b=self.target_embeddings,
            )

    def _compute_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        label_col: str,
    ) -> None:
        """Compute data statistics."""
        all_labels = pd.concat([train_df[label_col], val_df[label_col]])
        if test_df is not None:
            all_labels = pd.concat([all_labels, test_df[label_col]])

        unique_labels, counts = np.unique(all_labels, return_counts=True)

        self.data_stats = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df) if test_df is not None else 0,
            "num_classes": len(unique_labels),
            "class_distribution": dict(zip(unique_labels.tolist(), counts.tolist())),
        }

        logger.info(f"Class distribution: {self.data_stats['class_distribution']}")

    def _check_overlaps(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        drug_col: str,
        target_col: str,
    ) -> None:
        """Check for data leakage between splits."""
        train_pairs = set(zip(train_df[drug_col], train_df[target_col]))
        val_pairs = set(zip(val_df[drug_col], val_df[target_col]))

        train_val_overlap = train_pairs & val_pairs
        if train_val_overlap:
            logger.warning(f"Train/Val overlap: {len(train_val_overlap)} pairs")

        if test_df is not None:
            test_pairs = set(zip(test_df[drug_col], test_df[target_col]))
            train_test_overlap = train_pairs & test_pairs
            val_test_overlap = val_pairs & test_pairs

            if train_test_overlap:
                logger.error(f"Train/Test overlap: {len(train_test_overlap)} pairs!")
            if val_test_overlap:
                logger.error(f"Val/Test overlap: {len(val_test_overlap)} pairs!")

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

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader with collate_fn."""
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

    def predict_dataloader(self) -> Optional[DataLoader]:
        """Create predict dataloader (same as test)."""
        return self.test_dataloader()
