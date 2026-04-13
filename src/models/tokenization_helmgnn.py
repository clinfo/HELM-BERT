"""Monomer-level tokenizer for HELM-GNN.

Replaces the character-level HELMBertTokenizer with a monomer-level tokenizer
where each monomer symbol (e.g. 'meL', 'Abu', 'A') maps to a single token ID.
Vocabulary is built from the monomer library CSV.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from transformers import PreTrainedTokenizer

from .helm_parser import parse_helm_to_monomers

# Special tokens with reserved IDs
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "[MASK]": 4,
}


def build_monomer_vocab(monomer_library_path: str | Path) -> Dict[str, int]:
    """Build vocabulary from monomer library CSV.

    Args:
        monomer_library_path: Path to helm_monomer_library.csv

    Returns:
        Dict mapping monomer symbol -> token ID
    """
    df = pd.read_csv(monomer_library_path)
    symbols = sorted(df["symbol"].dropna().astype(str).str.strip().unique())

    vocab = dict(SPECIAL_TOKENS)
    next_id = len(SPECIAL_TOKENS)
    for symbol in symbols:
        if symbol not in vocab:
            vocab[symbol] = next_id
            next_id += 1
    return vocab


class HELMGNNTokenizer(PreTrainedTokenizer):
    """Monomer-level tokenizer for HELM-GNN.

    Each monomer symbol in the HELM notation becomes a single token.
    Example: 'PEPTIDE1{A.[meL].G}$$$$' -> [CLS] A meL G [SEP]
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        monomer_library_path: Optional[str] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        model_max_length: int = 512,
        **kwargs,
    ):
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)
        elif monomer_library_path is not None:
            self.vocab = build_monomer_vocab(monomer_library_path)
        else:
            self.vocab = dict(SPECIAL_TOKENS)

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize HELM string into monomer symbols."""
        monomers = parse_helm_to_monomers(text)
        if monomers is None:
            return [self.unk_token]
        return monomers

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token, 3))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return ".".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        cls_id = [self.cls_token_id]
        sep_id = [self.sep_token_id]
        if token_ids_1 is None:
            return cls_id + token_ids_0 + sep_id
        return cls_id + token_ids_0 + sep_id + token_ids_1 + sep_id

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return [
                1 if x in [self.cls_token_id, self.sep_token_id, self.pad_token_id] else 0
                for x in token_ids_0
            ]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return [0] * len(cls + token_ids_0 + sep)
        return [0] * len(cls + token_ids_0 + sep) + [1] * len(token_ids_1 + sep)

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (vocab_file,)

    @property
    def mask_token_id(self) -> int:
        return self.vocab.get(self.mask_token, 4)
