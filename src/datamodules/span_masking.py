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

    def _sample_span_lengths(self, num_spans: int) -> List[int]:
        """Sample span lengths from geometric distribution."""
        lengths = []
        for _ in range(num_spans):
            length = np.random.geometric(self.geometric_p)
            length = max(self.min_span_length, min(length, self.max_span_length))
            lengths.append(length)
        return lengths

    def _get_valid_span_starts(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[int]:
        """Get positions where spans can start."""
        valid_starts = []
        seq_len = len(input_ids)

        for i in range(seq_len):
            if attention_mask[i] == 0 or input_ids[i].item() in self.special_tokens:
                continue
            valid_starts.append(i)

        return valid_starts

    def _select_spans(
        self, valid_starts: List[int], span_lengths: List[int], seq_len: int
    ) -> List[Tuple[int, int]]:
        """Select non-overlapping spans."""
        spans = []
        masked_positions = set()
        valid_starts_shuffled = valid_starts.copy()
        np.random.shuffle(valid_starts_shuffled)

        span_idx = 0
        for start_pos in valid_starts_shuffled:
            if start_pos in masked_positions or span_idx >= len(span_lengths):
                continue

            span_length = span_lengths[span_idx]
            end_pos = min(start_pos + span_length, seq_len)

            if any(pos in masked_positions for pos in range(start_pos, end_pos)):
                continue

            spans.append((start_pos, end_pos))
            for pos in range(start_pos, end_pos):
                masked_positions.add(pos)
            span_idx += 1

        return spans

    def _apply_span_masking(
        self, input_ids: torch.Tensor, spans: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 80/10/10 masking rule to selected spans."""
        labels = torch.full_like(input_ids, self.ignore_index)
        masked_input_ids = input_ids.clone()

        for start, end in spans:
            rand = np.random.random()

            if rand < self.mask_token_prob:
                # 80%: Replace with [MASK]
                for pos in range(start, end):
                    labels[pos] = input_ids[pos]
                    masked_input_ids[pos] = self.mask_token_id
            elif rand < self.mask_token_prob + self.random_token_prob:
                # 10%: Replace with random tokens
                for pos in range(start, end):
                    labels[pos] = input_ids[pos]
                    random_token = np.random.choice(self.maskable_tokens_for_random)
                    masked_input_ids[pos] = random_token
            else:
                # 10%: Keep original
                for pos in range(start, end):
                    labels[pos] = input_ids[pos]

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

            if not valid_starts:
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
