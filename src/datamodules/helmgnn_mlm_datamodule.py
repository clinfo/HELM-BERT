"""MLM DataModule for HELM-GNN with monomer-level masking.

Unlike the character-level MLMDataModule, this operates at monomer level:
- Each token = one monomer symbol
- Span masking at monomer granularity
- Graph distance matrices are computed from HELM connections
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import lightning as L
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.models.helm_parser import parse_helm_full
from src.models.tokenization_helmgnn import HELMGNNTokenizer

logger = logging.getLogger(__name__)


@dataclass
class HELMGNNDatasetInfo:
    name: str
    file: str
    helm_column: str


@dataclass
class HELMGNNMLMDataConfig:
    data_dir: str
    datasets: List[HELMGNNDatasetInfo]
    train_ratio: float
    batch_size: int
    max_seq_length: int
    num_workers: int
    pin_memory: bool
    seed: int
    # Masking config
    mlm_probability: float
    mask_ratio: float
    random_ratio: float
    keep_ratio: float
    min_span_length: int
    max_span_length: int
    geometric_p: float
    ignore_index: int
    # Graph distance
    max_graph_distance: int


class HELMGNNMLMDataset(Dataset):
    """Dataset for HELM-GNN MLM.

    Returns monomer-tokenized sequences + graph distance matrices.
    Masking is done in the collator.
    """

    def __init__(
        self,
        sequences: List[str],
        tokenizer: HELMGNNTokenizer,
        max_length: int,
        max_graph_distance: int = 32,
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_graph_distance = max_graph_distance

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        helm = self.sequences[idx]

        # Tokenize
        encoding = self.tokenizer(
            helm, truncation=True, max_length=self.max_length, return_tensors=None
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Parse graph distances
        result = parse_helm_full(helm, max_distance=self.max_graph_distance)
        if result is not None:
            _, dist_matrix = result
            # dist_matrix is for monomers only; we need to account for [CLS] and [SEP]
            seq_len = len(input_ids)
            full_dist = np.full((seq_len, seq_len), self.max_graph_distance, dtype=np.int32)
            # Fill in monomer distances (offset by 1 for [CLS])
            n_mon = min(dist_matrix.shape[0], seq_len - 2)
            full_dist[1 : n_mon + 1, 1 : n_mon + 1] = dist_matrix[:n_mon, :n_mon]
            # Special tokens get max distance to everything
            full_dist[0, 0] = 0
            if seq_len > 1:
                full_dist[seq_len - 1, seq_len - 1] = 0
        else:
            seq_len = len(input_ids)
            full_dist = np.full((seq_len, seq_len), self.max_graph_distance, dtype=np.int32)
            np.fill_diagonal(full_dist, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_distances": full_dist.tolist(),
        }


class MonomerSpanMasking:
    """Span masking at monomer level for HELM-GNN MLM."""

    def __init__(
        self,
        mlm_probability: float,
        mask_ratio: float,
        random_ratio: float,
        min_span_length: int,
        max_span_length: int,
        geometric_p: float,
        ignore_index: int,
        mask_token_id: int,
        vocab_size: int,
        special_token_ids: set,
    ):
        self.mlm_probability = mlm_probability
        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length
        self.geometric_p = geometric_p
        self.ignore_index = ignore_index
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_tokens = special_token_ids
        self.maskable_tokens = np.array(
            list(set(range(vocab_size)) - special_token_ids)
        )

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        all_masked = []
        all_labels = []

        for i in range(batch_size):
            ids = input_ids[i]
            mask = attention_mask[i]

            # Find valid (maskable) positions
            ids_np = ids.numpy()
            mask_np = mask.numpy()
            special_mask = np.isin(ids_np, list(self.special_tokens))
            valid = np.where((mask_np == 1) & ~special_mask)[0]

            if len(valid) == 0:
                all_masked.append(ids.clone())
                all_labels.append(torch.full_like(ids, self.ignore_index))
                continue

            num_to_mask = max(1, int(len(valid) * self.mlm_probability))
            avg_span = (self.min_span_length + self.max_span_length) / 2
            num_spans = max(1, int(num_to_mask / avg_span))

            # Sample span lengths
            span_lengths = np.clip(
                np.random.geometric(self.geometric_p, size=num_spans),
                self.min_span_length, self.max_span_length
            )

            # Select non-overlapping spans
            masked_positions = np.zeros(seq_len, dtype=bool)
            spans = []
            shuffled_valid = valid.copy()
            np.random.shuffle(shuffled_valid)
            span_idx = 0
            for start in shuffled_valid:
                if masked_positions[start] or span_idx >= len(span_lengths):
                    continue
                length = int(span_lengths[span_idx])
                end = min(start + length, seq_len)
                if masked_positions[start:end].any():
                    continue
                spans.append((start, end))
                masked_positions[start:end] = True
                span_idx += 1

            # Apply masking
            labels = torch.full_like(ids, self.ignore_index)
            masked_ids = ids.clone()
            for start, end in spans:
                labels[start:end] = ids[start:end]
                rand = np.random.random()
                if rand < self.mask_ratio:
                    masked_ids[start:end] = self.mask_token_id
                elif rand < self.mask_ratio + self.random_ratio:
                    random_tokens = np.random.choice(self.maskable_tokens, size=end - start)
                    masked_ids[start:end] = torch.from_numpy(random_tokens)

            all_masked.append(masked_ids)
            all_labels.append(labels)

        return torch.stack(all_masked), torch.stack(all_labels)


class DataCollatorForHELMGNNMLM:
    """Collator for HELM-GNN MLM: pads + masks + stacks graph distances."""

    def __init__(
        self,
        tokenizer: HELMGNNTokenizer,
        masking: MonomerSpanMasking,
        max_graph_distance: int = 32,
    ):
        self.tokenizer = tokenizer
        self.masking = masking
        self.max_graph_distance = max_graph_distance

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate graph distances before padding
        graph_dists = [f.pop("graph_distances") for f in features]

        # Pad sequences
        batch = self.tokenizer.pad(
            features, padding=True, return_tensors="pt"
        )

        # Apply masking
        masked_ids, labels = self.masking(batch["input_ids"], batch["attention_mask"])

        # Pad graph distance matrices
        max_len = batch["input_ids"].size(1)
        padded_dists = []
        for dist in graph_dists:
            dist_arr = np.array(dist, dtype=np.int32)
            h, w = dist_arr.shape
            padded = np.full((max_len, max_len), self.max_graph_distance, dtype=np.int32)
            padded[:h, :w] = dist_arr
            padded_dists.append(padded)

        graph_distances = torch.tensor(np.stack(padded_dists), dtype=torch.long)

        return {
            "input_ids": masked_ids,
            "attention_mask": batch["attention_mask"],
            "labels": labels,
            "graph_distances": graph_distances,
        }


class HELMGNNMLMDataModule(L.LightningDataModule):
    """DataModule for HELM-GNN MLM training."""

    def __init__(
        self,
        config: HELMGNNMLMDataConfig,
        tokenizer: HELMGNNTokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        special_ids = set(self.tokenizer.all_special_ids)
        self.masking = MonomerSpanMasking(
            mlm_probability=config.mlm_probability,
            mask_ratio=config.mask_ratio,
            random_ratio=config.random_ratio,
            min_span_length=config.min_span_length,
            max_span_length=config.max_span_length,
            geometric_p=config.geometric_p,
            ignore_index=config.ignore_index,
            mask_token_id=self.tokenizer.mask_token_id,
            vocab_size=self.tokenizer.vocab_size,
            special_token_ids=special_ids,
        )

        self._collate_fn = DataCollatorForHELMGNNMLM(
            tokenizer=self.tokenizer,
            masking=self.masking,
            max_graph_distance=config.max_graph_distance,
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return

        if stage == "fit" or stage is None:
            all_sequences = []
            data_dir = Path(self.config.data_dir)
            for ds in self.config.datasets:
                file_path = data_dir / ds.file
                if not file_path.exists():
                    raise FileNotFoundError(f"Dataset not found: {file_path}")
                df = pd.read_csv(file_path)
                df = df.dropna(subset=[ds.helm_column])
                df = df[df[ds.helm_column].str.len() > 0]
                sequences = df[ds.helm_column].tolist()
                logger.info(f"Loaded {ds.name}: {len(sequences)} sequences")
                all_sequences.extend(sequences)

            if not all_sequences:
                raise ValueError("No sequences loaded")

            train_seqs, val_seqs = train_test_split(
                all_sequences,
                test_size=1.0 - self.config.train_ratio,
                random_state=self.config.seed,
            )

            self.train_dataset = HELMGNNMLMDataset(
                train_seqs, self.tokenizer, self.config.max_seq_length,
                self.config.max_graph_distance,
            )
            self.val_dataset = HELMGNNMLMDataset(
                val_seqs, self.tokenizer, self.config.max_seq_length,
                self.config.max_graph_distance,
            )

            logger.info(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}")

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
