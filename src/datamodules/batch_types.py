"""Batch data TypedDict definitions.

All batch values are torch.Tensor after DataCollator processing.
These types provide compile-time type checking and IDE support.
"""

from typing import TypedDict

import torch


class MLMBatch(TypedDict):
    """Batch for Masked Language Modeling.

    Produced by DataCollatorForMLM.
    """

    input_ids: torch.Tensor  # (batch, seq_len) - masked tokens
    attention_mask: torch.Tensor  # (batch, seq_len)
    labels: torch.Tensor  # (batch, seq_len) - original tokens, -100 for non-masked


class PermeabilitySingleBatch(TypedDict):
    """Batch for permeability single-assay regression.

    Produced by DataCollatorForRegression.
    """

    input_ids: torch.Tensor  # (batch, seq_len)
    attention_mask: torch.Tensor  # (batch, seq_len)
    target: torch.Tensor  # (batch,)


class PermeabilityMultiBatch(TypedDict):
    """Batch for permeability multi-assay regression (PAMPA + Caco2).

    Produced by DataCollatorForPermeabilityMultiRegression.
    """

    input_ids: torch.Tensor  # (batch, seq_len)
    attention_mask: torch.Tensor  # (batch, seq_len)
    target_pampa: torch.Tensor  # (batch,)
    target_caco2: torch.Tensor  # (batch,)
    mask_pampa: torch.Tensor  # (batch,) — 1.0 if valid, 0.0 if missing
    mask_caco2: torch.Tensor  # (batch,) — 1.0 if valid, 0.0 if missing


class PPIBatchTokenized(TypedDict):
    """Batch for PPI with tokenized sequences.

    Produced by DataCollatorForPPI when using real-time encoding.
    """

    drug_ids: torch.Tensor  # (batch, drug_seq_len)
    drug_mask: torch.Tensor  # (batch, drug_seq_len)
    target_ids: torch.Tensor  # (batch, target_seq_len)
    target_mask: torch.Tensor  # (batch, target_seq_len)
    label: torch.Tensor  # (batch,)


class PPIBatchEmbedding(TypedDict):
    """Batch for PPI with pre-computed embeddings.

    Produced by DataCollatorForPPIEmbedding.
    """

    drug_embedding: torch.Tensor  # (batch, hidden_dim)
    target_embedding: torch.Tensor  # (batch, hidden_dim)
    label: torch.Tensor  # (batch,)
