"""DataCollators for dynamic padding and batch processing.

Each collator handles:
- Dynamic padding to max length in batch
- Tensorization
- Task-specific processing (e.g., MLM masking)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForMLM:
    """DataCollator for Masked Language Modeling.

    Handles dynamic padding and span masking at batch level.

    Args:
        tokenizer: Tokenizer for padding
        masking_strategy: SpanMasking instance for MLM masking
        pad_to_multiple_of: Pad to multiple of this value (optional)
    """

    tokenizer: PreTrainedTokenizer
    masking_strategy: Any  # SpanMasking
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic padding and masking.

        Args:
            features: List of {"input_ids": List[int], "attention_mask": List[int]}

        Returns:
            Batch with input_ids (masked), attention_mask, labels
        """
        # Pad batch dynamically
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Apply masking (batch-level)
        masked_input_ids, labels = self.masking_strategy.get_masked_tokens(
            batch["input_ids"], batch["attention_mask"]
        )

        return {
            "input_ids": masked_input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }


@dataclass
class DataCollatorForRegression:
    """DataCollator for regression tasks (e.g., permeability).

    Handles dynamic padding for sequences and tensorization of targets.

    Args:
        tokenizer: Tokenizer for padding
        pad_to_multiple_of: Pad to multiple of this value (optional)
    """

    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic padding.

        Args:
            features: List of {"input_ids": List[int], "attention_mask": List[int], "target": float}

        Returns:
            Batch with input_ids, attention_mask, target
        """
        # Extract targets (without modifying original)
        targets = [f["target"] for f in features]
        sequence_features = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            for f in features
        ]

        # Pad sequences
        batch = self.tokenizer.pad(
            sequence_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Add targets as tensor
        batch["target"] = torch.tensor(targets, dtype=torch.float32)

        return batch


@dataclass
class DataCollatorForPPI:
    """DataCollator for PPI with dual sequences.

    Handles dynamic padding for both drug and target sequences.

    Args:
        drug_tokenizer: Tokenizer for drug sequences
        target_tokenizer: Tokenizer for target sequences
        pad_to_multiple_of: Pad to multiple of this value (optional)
    """

    drug_tokenizer: PreTrainedTokenizer
    target_tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic padding for dual sequences.

        Args:
            features: List of dicts with drug_ids, drug_mask, target_ids, target_mask, label

        Returns:
            Batch with all fields as tensors
        """
        # Separate drug and target features
        drug_features = [
            {"input_ids": f["drug_ids"], "attention_mask": f["drug_mask"]}
            for f in features
        ]
        target_features = [
            {"input_ids": f["target_ids"], "attention_mask": f["target_mask"]}
            for f in features
        ]

        # Extract labels
        labels = [f["label"] for f in features]

        # Pad each sequence type separately
        drug_batch = self.drug_tokenizer.pad(
            drug_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        target_batch = self.target_tokenizer.pad(
            target_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        return {
            "drug_ids": drug_batch["input_ids"],
            "drug_mask": drug_batch["attention_mask"],
            "target_ids": target_batch["input_ids"],
            "target_mask": target_batch["attention_mask"],
            "label": torch.tensor(labels, dtype=torch.float32),
        }


@dataclass
class DataCollatorForPPIEmbedding:
    """DataCollator for PPI with pre-computed embeddings.

    Simply stacks embeddings and tensorizes labels.
    No padding needed since embeddings have fixed dimensions.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of embeddings.

        Args:
            features: List of dicts with drug_embedding, target_embedding, label

        Returns:
            Batch with all fields as tensors
        """
        return {
            "drug_embedding": torch.stack([f["drug_embedding"] for f in features]),
            "target_embedding": torch.stack([f["target_embedding"] for f in features]),
            "label": torch.tensor([f["label"] for f in features], dtype=torch.float32),
        }
