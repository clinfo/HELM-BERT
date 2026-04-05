"""Dataset classes for HELM-BERT.

Datasets return raw data (lists, not tensors).
DataCollators handle padding and tensorization at batch level.
"""


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
