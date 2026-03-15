"""Embedding generation for frozen encoders.

Loads models, generates embeddings for missing sequences, and saves to cache.
Used by DataModule.prepare_data() to auto-populate the embedding cache.
"""

import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from src.utils.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


def generate_drug_embeddings(
    cache: EmbeddingCache,
    sequences: List[str],
    encoder_name: str,
    dataset_type: str,
    pretrained_path: str,
    max_length: int,
    batch_size: int,
    trust_remote_code: bool = True,
) -> None:
    """Generate and cache drug (HELM-BERT) embeddings using pooler_output."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading drug encoder: {pretrained_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path, trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        pretrained_path, trust_remote_code=trust_remote_code,
    ).to(device).eval()

    @torch.inference_mode()
    def embed_fn(seqs: List[str]) -> List[torch.Tensor]:
        enc = tokenizer(
            seqs, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        return list(model(**enc, return_dict=True).pooler_output.cpu())

    cache.generate_missing_embeddings(
        encoder_name=encoder_name,
        dataset_type=dataset_type,
        sequences=sequences,
        embed_fn=embed_fn,
        batch_size=batch_size,
        role="drug",
    )

    del model, tokenizer
    torch.cuda.empty_cache()


def generate_target_embeddings(
    cache: EmbeddingCache,
    sequences: List[str],
    encoder_name: str,
    dataset_type: str,
    pretrained_path: str,
    max_length: int,
    batch_size: int,
) -> None:
    """Generate and cache target (ESM-2) embeddings using mean pooling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading target encoder: {pretrained_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    model = AutoModel.from_pretrained(pretrained_path).to(device).eval()

    @torch.inference_mode()
    def embed_fn(seqs: List[str]) -> List[torch.Tensor]:
        enc = tokenizer(
            seqs, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        hidden = model(**enc, return_dict=True).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        return list(((hidden * mask).sum(1) / mask.sum(1)).cpu())

    cache.generate_missing_embeddings(
        encoder_name=encoder_name,
        dataset_type=dataset_type,
        sequences=sequences,
        embed_fn=embed_fn,
        batch_size=batch_size,
        role="target",
    )

    del model, tokenizer
    torch.cuda.empty_cache()
