"""Embedding-only dataset for PPI tasks.

Returns batches that contain only precomputed embeddings, labels, and optional weights.
Provides dummy token fields to satisfy model interfaces without incurring tokenization.
"""

from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset


class EmbeddingOnlyDataset(Dataset):
    """Dataset that serves precomputed embeddings instead of tokenized sequences."""

    def __init__(
        self,
        sequences_a: List[str],
        sequences_b: List[str],
        labels: List[float],
        embeddings_a: Dict[str, torch.Tensor],
        embeddings_b: Dict[str, torch.Tensor],
        weights: Optional[List[float]] = None,
        name_a: str = "drug",
        name_b: str = "target",
    ) -> None:
        super().__init__()
        self.sequences_a = sequences_a
        self.sequences_b = sequences_b
        self.labels = labels
        self.weights = weights
        self.name_a = name_a
        self.name_b = name_b
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b

        if len(self.sequences_a) != len(self.sequences_b) or len(
            self.sequences_a
        ) != len(self.labels):
            raise ValueError("Mismatched lengths in sequences and labels")

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def _dummy_ids() -> torch.Tensor:
        return torch.empty(0, dtype=torch.long)

    @staticmethod
    def _dummy_mask() -> torch.Tensor:
        return torch.empty(0, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_a = self.sequences_a[idx]
        seq_b = self.sequences_b[idx]

        if seq_a not in self.embeddings_a:
            raise KeyError(f"Missing embedding for sequence A: {seq_a}")
        if seq_b not in self.embeddings_b:
            raise KeyError(f"Missing embedding for sequence B: {seq_b}")

        sample: Dict[str, Any] = {
            f"{self.name_a}_embedding": self.embeddings_a[seq_a],
            f"{self.name_b}_embedding": self.embeddings_b[seq_b],
            "label": torch.tensor(self.labels[idx]),
        }

        if self.weights is not None:
            sample["weight"] = torch.tensor(self.weights[idx], dtype=torch.float32)

        return sample
