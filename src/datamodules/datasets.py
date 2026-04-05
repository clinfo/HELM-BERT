"""Dataset classes for HELM-BERT.

Datasets return raw data (lists, not tensors).
DataCollators handle padding and tensorization at batch level.
"""

import math
from typing import Any, Dict, List

from torch.utils.data import Dataset


class MLMDataset(Dataset):
    """Dataset for HELM Masked Language Modeling.

    Returns tokenized sequences without padding or masking.
    DataCollatorForMLM handles padding and masking at batch level.

    Args:
        sequences: List of HELM sequences
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length (for truncation only)
    """

    def __init__(
        self,
        sequences: List[str],
        tokenizer: Any,
        max_length: int,
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """Get tokenized sequence.

        Returns:
            {"input_ids": List[int], "attention_mask": List[int]}
        """
        sequence = self.sequences[idx]

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


class HELMDataset(Dataset):
    """Dataset for HELM sequences with labels for downstream tasks.

    Returns tokenized sequences and labels without padding.
    DataCollatorForRegression handles padding at batch level.

    Args:
        sequences: List of HELM notation strings
        labels: List of target values
        tokenizer: HELMTokenizer instance
        max_length: Maximum sequence length (for truncation only)
    """

    def __init__(
        self,
        sequences: List[str],
        labels: List[float],
        tokenizer: Any,
        max_length: int,
    ):
        if len(sequences) != len(labels):
            raise ValueError(
                f"Sequences and labels must have same length: "
                f"{len(sequences)} vs {len(labels)}"
            )

        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get tokenized sequence and label.

        Returns:
            {"input_ids": List[int], "attention_mask": List[int], "target": float}
        """
        sequence = self.sequences[idx]

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "target": float(self.labels[idx]),
        }


class PermeabilityMultiHELMDataset(Dataset):
    """Dataset for HELM sequences with multiple assay targets.

    Handles missing labels via per-assay masks.
    NaN targets are replaced with 0.0 and masked out during loss computation.

    Args:
        sequences: List of HELM notation strings
        assay_values: Dict mapping assay name to list of float/NaN values
        tokenizer: HELMTokenizer instance
        max_length: Maximum sequence length (for truncation only)
    """

    def __init__(
        self,
        sequences: List[str],
        assay_values: Dict[str, List[float]],
        tokenizer: Any,
        max_length: int,
    ):
        self.sequences = sequences
        self.assay_names = list(assay_values.keys())
        self.assay_values = assay_values
        self.tokenizer = tokenizer
        self.max_length = max_length

        for name, values in assay_values.items():
            if len(sequences) != len(values):
                raise ValueError(
                    f"Sequences and {name} values must have same length: "
                    f"{len(sequences)} vs {len(values)}"
                )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoding = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }

        for name in self.assay_names:
            val = self.assay_values[name][idx]
            is_valid = not math.isnan(val)
            item[f"target_{name}"] = float(val) if is_valid else 0.0
            item[f"mask_{name}"] = 1.0 if is_valid else 0.0

        return item
