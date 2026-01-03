"""SpanBERT-style span masking for HELM sequences.

Reference:
- Paper: SpanBERT: Improving Pre-training by Representing and Predicting Spans
- Implementation: https://github.com/facebookresearch/SpanBERT
"""

from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
import torch


@dataclass
class SpanMaskingConfig:
    """Configuration for span masking.

    All fields are required - values come from YAML configuration.
    """

    mlm_probability: float
    mask_ratio: float
    random_ratio: float
    keep_ratio: float
    min_span_length: int
    max_span_length: int
    geometric_p: float
    ignore_index: int


class SpanMasking:
    """SpanBERT-style span masking for HELM sequences."""

    def __init__(
        self,
        config: SpanMaskingConfig,
        mask_token_id: int,
        vocab_size: int,
        special_token_ids: Set[int],
    ):
        """Initialize span masking strategy.

        Args:
            config: SpanMaskingConfig object (required)
            mask_token_id: Token ID for MASK token (required)
            vocab_size: Size of vocabulary (required)
            special_token_ids: Set of special token IDs that should not be masked (required)
        """
        self.max_span_length = config.max_span_length
        self.min_span_length = config.min_span_length
        self.geometric_p = config.geometric_p
        self.mlm_probability = config.mlm_probability
        self.mask_token_prob = config.mask_ratio
        self.random_token_prob = config.random_ratio
        self.ignore_index = config.ignore_index
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_tokens = special_token_ids

        # Pre-compute maskable tokens for random replacement
        maskable_ids = set(range(vocab_size)) - self.special_tokens
        self.maskable_tokens_for_random = np.array(list(maskable_ids))

    def _sample_span_lengths(self, num_spans: int) -> np.ndarray:
        """Sample span lengths from geometric distribution (vectorized)."""
        lengths = np.random.geometric(self.geometric_p, size=num_spans)
        return np.clip(lengths, self.min_span_length, self.max_span_length)

    def _get_valid_span_starts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> np.ndarray:
        """Get positions where spans can start (vectorized)."""
        # Convert to numpy for vectorized operations
        ids_np = input_ids.numpy()
        mask_np = attention_mask.numpy()

        # Build special tokens mask (vectorized)
        special_mask = np.isin(ids_np, list(self.special_tokens))

        # Valid positions: attention=1 AND not special token
        valid_mask = (mask_np == 1) & ~special_mask

        return np.where(valid_mask)[0]

    def _select_spans(
        self, valid_starts: np.ndarray, span_lengths: np.ndarray, seq_len: int
    ) -> List[Tuple[int, int]]:
        """Select non-overlapping spans.

        Note: This is inherently sequential due to non-overlap constraint.
        """
        spans = []
        masked_positions = np.zeros(seq_len, dtype=bool)
        valid_starts_shuffled = valid_starts.copy()
        np.random.shuffle(valid_starts_shuffled)

        span_idx = 0
        for start_pos in valid_starts_shuffled:
            if masked_positions[start_pos] or span_idx >= len(span_lengths):
                continue

            span_length = int(span_lengths[span_idx])
            end_pos = min(start_pos + span_length, seq_len)

            # Check overlap using slice (faster than any())
            if masked_positions[start_pos:end_pos].any():
                continue

            spans.append((start_pos, end_pos))
            masked_positions[start_pos:end_pos] = True
            span_idx += 1

        return spans

    def _apply_span_masking(
        self, input_ids: torch.Tensor, spans: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 80/10/10 masking rule to selected spans (slice-optimized)."""
        labels = torch.full_like(input_ids, self.ignore_index)
        masked_input_ids = input_ids.clone()

        for start, end in spans:
            # Copy original tokens to labels (slice assignment)
            labels[start:end] = input_ids[start:end]

            rand = np.random.random()
            if rand < self.mask_token_prob:
                # 80%: Replace with [MASK]
                masked_input_ids[start:end] = self.mask_token_id
            elif rand < self.mask_token_prob + self.random_token_prob:
                # 10%: Replace with random tokens (vectorized)
                span_len = end - start
                random_tokens = np.random.choice(
                    self.maskable_tokens_for_random, size=span_len
                )
                masked_input_ids[start:end] = torch.from_numpy(random_tokens)
            # 10%: Keep original (no action needed)

        return masked_input_ids, labels

    def get_masked_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main entry point for span masking.

        Args:
            input_ids: Token IDs [batch_size, seq_len] or [seq_len]
            attention_mask: Attention mask [batch_size, seq_len] or [seq_len]

        Returns:
            Tuple of (masked_input_ids, labels)
        """

        # Handle batch dimension
        squeeze_output = False
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            squeeze_output = True

        batch_size, seq_len = input_ids.shape
        all_masked_input_ids = []
        all_labels = []

        for i in range(batch_size):
            valid_starts = self._get_valid_span_starts(input_ids[i], attention_mask[i])

            if len(valid_starts) == 0:
                all_masked_input_ids.append(input_ids[i])
                all_labels.append(torch.full_like(input_ids[i], self.ignore_index))
                continue

            num_to_mask = max(1, int(len(valid_starts) * self.mlm_probability))
            avg_span_length = (self.min_span_length + self.max_span_length) / 2
            num_spans = max(1, int(num_to_mask / avg_span_length))

            span_lengths = self._sample_span_lengths(num_spans)
            spans = self._select_spans(valid_starts, span_lengths, seq_len)
            masked_ids, labels = self._apply_span_masking(input_ids[i], spans)

            all_masked_input_ids.append(masked_ids)
            all_labels.append(labels)

        masked_input_ids = torch.stack(all_masked_input_ids)
        labels = torch.stack(all_labels)

        if squeeze_output:
            masked_input_ids = masked_input_ids.squeeze(0)
            labels = labels.squeeze(0)

        return masked_input_ids, labels
