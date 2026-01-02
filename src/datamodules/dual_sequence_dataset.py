"""General-purpose dataset for dual-sequence tasks (e.g., drug-target, peptide-protein)."""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset


class DualSequenceDataset(Dataset):
    """Dataset for tasks involving two sequences.

    Supports various dual-sequence scenarios:
    - Peptide-Protein Interaction (HELM + ESM)
    - Drug-Target Interaction (SMILES + ESM)
    - Peptide-Peptide Interaction (HELM + HELM)
    - Any other sequence pair combination

    Example:
        # For PPI task
        dataset = DualSequenceDataset(
            sequences_a=peptide_sequences,
            sequences_b=protein_sequences,
            labels=interaction_labels,
            tokenizer_a=helm_tokenizer,
            tokenizer_b=esm_tokenizer,
            max_length_a=512,
            max_length_b=1024,
            name_a='drug',
            name_b='target'
        )
    """

    def __init__(
        self,
        sequences_a: List[str],
        sequences_b: List[str],
        labels: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
        tokenizer_a: Any = None,
        tokenizer_b: Any = None,
        max_length_a: int = 512,
        max_length_b: int = 512,
        name_a: str = "sequence_a",
        name_b: str = "sequence_b",
        embeddings_a: Optional[Dict[str, torch.Tensor]] = None,
        embeddings_b: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize dual sequence dataset.

        Args:
            sequences_a: First sequences (e.g., drug/peptide HELM)
            sequences_b: Second sequences (e.g., target/protein)
            labels: Task labels (classification or regression)
            weights: Sample weights for loss calculation (default: None)
            tokenizer_a: Tokenizer for first sequences
            tokenizer_b: Tokenizer for second sequences
            max_length_a: Max length for first sequences
            max_length_b: Max length for second sequences
            name_a: Field name for first sequences (default: 'sequence_a')
            name_b: Field name for second sequences (default: 'sequence_b')
            embeddings_a: Pre-computed embeddings for sequences_a
            embeddings_b: Pre-computed embeddings for sequences_b
        """
        assert len(sequences_a) == len(sequences_b), (
            f"Sequence lists must have same length: {len(sequences_a)} vs {len(sequences_b)}"
        )

        if labels is not None:
            assert len(labels) == len(sequences_a), (
                f"Labels length must match sequences: {len(labels)} vs {len(sequences_a)}"
            )

        if weights is not None:
            assert len(weights) == len(sequences_a), (
                f"Weights length must match sequences: {len(weights)} vs {len(sequences_a)}"
            )

        self.sequences_a = sequences_a
        self.sequences_b = sequences_b
        self.labels = labels
        self.weights = weights
        self.tokenizer_a = tokenizer_a
        self.tokenizer_b = tokenizer_b
        self.max_length_a = max_length_a
        self.max_length_b = max_length_b
        self.name_a = name_a
        self.name_b = name_b
        self.embeddings_a = embeddings_a
        self.embeddings_b = embeddings_b

    def __len__(self) -> int:
        return len(self.sequences_a)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.

        Returns:
            Dictionary with tokenized sequences and optional label:
            {
                f'{name_a}_ids': input_ids for sequence A,
                f'{name_a}_mask': attention_mask for sequence A,
                f'{name_b}_ids': input_ids for sequence B,
                f'{name_b}_mask': attention_mask for sequence B,
                f'{name_a}_embedding': pre-computed embedding for sequence A (if available),
                f'{name_b}_embedding': pre-computed embedding for sequence B (if available),
                'label': label (if provided),
                'weight': sample weight (if provided),
                f'{name_a}_sequence': original sequence A,
                f'{name_b}_sequence': original sequence B
            }
        """
        seq_a = self.sequences_a[idx]
        seq_b = self.sequences_b[idx]

        # Tokenize sequence A
        encoding_a = self.tokenizer_a(
            seq_a,
            padding="max_length",
            truncation=True,
            max_length=self.max_length_a,
            return_tensors=None,
        )

        # Tokenize sequence B
        encoding_b = self.tokenizer_b(
            seq_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_length_b,
            return_tensors=None,
        )

        # Build sample dictionary
        sample = {
            f"{self.name_a}_ids": torch.tensor(encoding_a["input_ids"]),
            f"{self.name_a}_mask": torch.tensor(encoding_a["attention_mask"]),
            f"{self.name_b}_ids": torch.tensor(encoding_b["input_ids"]),
            f"{self.name_b}_mask": torch.tensor(encoding_b["attention_mask"]),
            f"{self.name_a}_sequence": seq_a,
            f"{self.name_b}_sequence": seq_b,
        }

        # Add pre-computed embeddings if available
        if self.embeddings_a is not None and seq_a in self.embeddings_a:
            sample[f"{self.name_a}_embedding"] = self.embeddings_a[seq_a]

        if self.embeddings_b is not None and seq_b in self.embeddings_b:
            sample[f"{self.name_b}_embedding"] = self.embeddings_b[seq_b]

        # Add label if available
        if self.labels is not None:
            label = self.labels[idx]
            # Handle both classification (int) and regression (float)
            if isinstance(label, (int, float)):
                sample["label"] = label
            else:
                # For multi-label or other complex labels
                sample["label"] = torch.tensor(label)

        # Add weight if available
        if self.weights is not None:
            sample["weight"] = self.weights[idx]

        return sample
