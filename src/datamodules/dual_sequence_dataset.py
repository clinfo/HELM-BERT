"""Dataset for dual-sequence tasks (e.g., drug-target, peptide-protein)."""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class DualSequenceDataset(Dataset):
    """Dataset for tasks involving two sequences.

    Returns tokenized sequences without padding.
    DataCollatorForPPI handles padding at batch level.

    Uses fixed key names: drug_ids, drug_mask, target_ids, target_mask.

    Args:
        sequences_a: First sequences (drug/peptide HELM)
        sequences_b: Second sequences (target/protein)
        labels: Task labels (required)
        tokenizer_a: Tokenizer for drug sequences
        tokenizer_b: Tokenizer for target sequences
        max_length_a: Max length for drug sequences (required, from YAML)
        max_length_b: Max length for target sequences (required, from YAML)
        embeddings_a: Pre-computed drug embeddings (optional)
        embeddings_b: Pre-computed target embeddings (optional)
    """

    def __init__(
        self,
        sequences_a: List[str],
        sequences_b: List[str],
        labels: List[float],
        tokenizer_a: Any,
        tokenizer_b: Any,
        max_length_a: int,
        max_length_b: int,
        embeddings_a: Optional[Dict[str, torch.Tensor]] = None,
        embeddings_b: Optional[Dict[str, torch.Tensor]] = None,
    ):
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
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.max_length_a = max_length_a
        self.max_length_b = max_length_b
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b

    def __len__(self) -> int:
        return len(self.sequences_a)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get tokenized sequences and label.

        Returns:
            {
                "drug_ids": List[int],
                "drug_mask": List[int],
                "target_ids": List[int],
                "target_mask": List[int],
                "label": float,
                "drug_embedding": Tensor (if available),
                "target_embedding": Tensor (if available),
            }
        """
        seq_a = self.sequences_a[idx]
        seq_b = self.sequences_b[idx]

        # Tokenize drug sequence
        encoding_a = self.tokenizer_a(
            seq_a,
            truncation=True,
            max_length=self.max_length_a,
            return_tensors=None,
        )

        # Tokenize target sequence
        encoding_b = self.tokenizer_b(
            seq_b,
            truncation=True,
            max_length=self.max_length_b,
            return_tensors=None,
        )

        # Build sample with fixed key names
        sample: Dict[str, Any] = {
            "drug_ids": encoding_a["input_ids"],
            "drug_mask": encoding_a["attention_mask"],
            "target_ids": encoding_b["input_ids"],
            "target_mask": encoding_b["attention_mask"],
            "label": float(self.labels[idx]),
        }

        # Add pre-computed embeddings if available (fail fast if missing)
        if self.embeddings_a is not None:
            if seq_a not in self.embeddings_a:
                raise KeyError(f"Missing drug embedding for: {seq_a[:50]}...")
            sample["drug_embedding"] = self.embeddings_a[seq_a]

        if self.embeddings_b is not None:
            if seq_b not in self.embeddings_b:
                raise KeyError(f"Missing target embedding for: {seq_b[:50]}...")
            sample["target_embedding"] = self.embeddings_b[seq_b]

        return sample
