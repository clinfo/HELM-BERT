"""Dataset classes for HELM-BERT."""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    """Dataset for HELM Masked Language Modeling."""

    def __init__(
        self,
        sequences: List[str],
        tokenizer: Any,
        max_length: int,
        masking_strategy: Any,
    ):
        """Initialize MLM dataset.

        Args:
            sequences: List of HELM sequences
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            masking_strategy: SpanMasking instance that handles all masking logic
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.masking_strategy = masking_strategy

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item with MLM masking applied."""
        sequence = self.sequences[idx]

        # Tokenize
        encoding = self.tokenizer(
            sequence,
            return_tensors=None,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Extract tensors
        input_ids = torch.tensor(encoding["input_ids"])
        attention_mask = torch.tensor(encoding["attention_mask"])

        # Apply span masking
        masked_input_ids, labels = self.masking_strategy.get_masked_tokens(
            input_ids, attention_mask
        )

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class HELMDataset(Dataset):
    """Dataset for HELM sequences with labels for downstream tasks."""

    def __init__(
        self,
        sequences: List[str],
        labels: Optional[List[float]],
        tokenizer,
        max_length: int,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize HELMDataset.

        Args:
            sequences: List of HELM notation strings
            labels: List of target values (optional)
            tokenizer: HELMTokenizer instance
            max_length: Maximum sequence length
            metadata: Optional list of metadata dicts
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sequence = self.sequences[idx]

        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        item = {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "helm": sequence,
        }

        # Add label if available
        if self.labels is not None:
            item["target"] = float(self.labels[idx])

        # Add metadata (if available)
        if self.metadata is not None and idx < len(self.metadata):
            item.update(self.metadata[idx])

        return item
