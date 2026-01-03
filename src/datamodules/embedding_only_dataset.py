"""Embedding-only dataset for PPI tasks.

Returns pre-computed embeddings and labels.
DataCollatorForPPIEmbedding handles stacking at batch level.
"""

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class EmbeddingOnlyDataset(Dataset):
    """Dataset that serves pre-computed embeddings.

    Uses fixed key names: drug_embedding, target_embedding.

    Args:
        sequences_a: Drug sequences (for embedding lookup)
        sequences_b: Target sequences (for embedding lookup)
        labels: Task labels (required)
        embeddings_a: Pre-computed drug embeddings (required)
        embeddings_b: Pre-computed target embeddings (required)
    """

    def __init__(
        self,
        sequences_a: List[str],
        sequences_b: List[str],
        labels: List[float],
        embeddings_a: Dict[str, torch.Tensor],
        embeddings_b: Dict[str, torch.Tensor],
    ) -> None:
        if len(sequences_a) != len(sequences_b):
            raise ValueError(
                f"Sequence lists must have same length: "
                f"{len(sequences_a)} vs {len(sequences_b)}"
            )
        if len(labels) != len(sequences_a):
            raise ValueError(
                f"Labels length must match sequences: "
                f"{len(labels)} vs {len(sequences_a)}"
            )

        self.sequences_a = sequences_a
        self.sequences_b = sequences_b
        self.labels = labels
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get embeddings and label.

        Returns:
            {
                "drug_embedding": Tensor,
                "target_embedding": Tensor,
                "label": float,
            }
        """
        seq_a = self.sequences_a[idx]
        seq_b = self.sequences_b[idx]

        # Fail fast if embedding missing
        if seq_a not in self.embeddings_a:
            raise KeyError(f"Missing drug embedding for: {seq_a[:50]}...")
        if seq_b not in self.embeddings_b:
            raise KeyError(f"Missing target embedding for: {seq_b[:50]}...")

        return {
            "drug_embedding": self.embeddings_a[seq_a],
            "target_embedding": self.embeddings_b[seq_b],
            "label": float(self.labels[idx]),
        }
