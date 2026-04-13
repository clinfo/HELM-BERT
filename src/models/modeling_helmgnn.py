"""HELM-GNN model: GPS monomer encoder + Graph-aware Transformer.

End-to-end model where:
1. GPS encodes each monomer's SMILES into a 768-dim embedding
2. Graph-aware Transformer processes the monomer sequence with
   structure distance bias from HELM connectivity
3. MLM or classification head produces output

The GPS is trained end-to-end via MLM gradient backpropagation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from torch import _softmax_backward_data
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
)

from .configuration_helmgnn import HELMGNNConfig
from .monomer_gnn import MonomerGPSEncoder


# ---------------------------------------------------------------------------
# Utility functions (same as helmbert)
# ---------------------------------------------------------------------------


def masked_layer_norm(
    layer_norm: nn.LayerNorm, x: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    output = layer_norm(x).to(x.dtype)
    if mask is None:
        return output
    if mask.dim() != x.dim():
        if mask.dim() == 4:
            mask = mask.squeeze(1).squeeze(1)
        mask = mask.unsqueeze(2)
    mask = mask.to(output.dtype)
    return output * mask


class XSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask, dim):
        ctx.dim = dim
        if mask is not None:
            rmask = ~(mask.bool())
            if rmask.dim() == 2:
                rmask = rmask.unsqueeze(1).unsqueeze(2)
            elif rmask.dim() == 3:
                rmask = rmask.unsqueeze(2)
            output = input.masked_fill(rmask, float("-inf"))
        else:
            output = input
        output = torch.softmax(output, ctx.dim)
        if mask is not None:
            output.masked_fill_(rmask, 0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        if version.Version(torch.__version__) >= version.Version("1.11.0"):
            input_grad = _softmax_backward_data(
                grad_output, output, ctx.dim, output.dtype
            )
        else:
            input_grad = _softmax_backward_data(grad_output, output, ctx.dim, output)
        return input_grad, None, None


def build_relative_position(
    query_size: int,
    key_size: int,
    bucket_size: int = -1,
    max_position: int = 512,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos = q_ids.unsqueeze(1) - k_ids.unsqueeze(0)

    if bucket_size > 0:
        rel_buckets = 0
        num_buckets = bucket_size
        rel_buckets += (rel_pos > 0).long() * (num_buckets // 2)
        rel_pos = torch.abs(rel_pos)
        max_exact = num_buckets // 4
        is_small = rel_pos < max_exact
        rel_pos_if_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(max_position / max_exact)
                * (num_buckets // 4 - 1)
            ).long()
        )
        rel_pos_if_large = torch.min(
            rel_pos_if_large, torch.full_like(rel_pos_if_large, num_buckets // 2 - 1)
        )
        rel_buckets += torch.where(is_small, rel_pos, rel_pos_if_large)
        return rel_buckets
    else:
        rel_pos = torch.clamp(rel_pos, -max_position, max_position)
        return rel_pos + max_position


# ---------------------------------------------------------------------------
# Attention with graph distance bias
# ---------------------------------------------------------------------------


class DisentangledSelfAttentionWithGraphBias(nn.Module):
    """Disentangled attention extended with graph distance bias.

    Adds a learned bias term based on the graph distance between monomers
    to the attention score, enabling structure-aware attention.
    """

    def __init__(self, config: HELMGNNConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_size

        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.pos_att_type = [x.strip() for x in config.pos_att_type.lower().split("|")]
        self.max_relative_positions = config.max_relative_positions
        self.position_buckets = config.position_buckets
        self.share_att_key = config.share_att_key

        self.pos_ebd_size = config.max_relative_positions
        if config.position_buckets > 0:
            self.pos_ebd_size = config.position_buckets

        self.rel_embeddings = nn.Embedding(self.pos_ebd_size * 2, config.hidden_size)

        if not self.share_att_key:
            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        # Graph distance bias: learnable per-head bias for each distance bucket
        self.graph_distance_bias = nn.Embedding(
            config.num_graph_distance_buckets, self.num_heads
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.pos_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        query_states: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        rel_embeddings: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if query_states is None:
            query_states = hidden_states

        query_layer = self.transpose_for_scores(self.query_proj(query_states)).float()
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states)).float()
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1

        scale = 1.0 / math.sqrt(self.head_size * scale_factor)

        # c2c attention
        c2c_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) * scale)
        attention_scores = c2c_scores

        # Relative position bias
        if len(self.pos_att_type) > 0 and self.pos_att_type[0]:
            rel_att = self._disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )
            if rel_att is not None:
                attention_scores = attention_scores + rel_att

        # Graph distance bias
        if graph_distances is not None:
            # graph_distances: (batch, seq, seq) int
            g_bias = self.graph_distance_bias(graph_distances)  # (batch, seq, seq, num_heads)
            g_bias = g_bias.permute(0, 3, 1, 2)  # (batch, num_heads, seq, seq)
            # Reshape to match attention_scores: (batch*num_heads, seq, seq)
            batch_size = g_bias.size(0)
            g_bias = g_bias.contiguous().view(-1, g_bias.size(2), g_bias.size(3))
            attention_scores = attention_scores + g_bias

        # Normalize
        attention_scores = (
            attention_scores - attention_scores.max(dim=-1, keepdim=True)[0].detach()
        )
        attention_scores = attention_scores.to(hidden_states.dtype)

        attention_scores = attention_scores.view(
            -1, self.num_heads, attention_scores.size(-2), attention_scores.size(-1)
        )

        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)

        attention_probs_flat = attention_probs.view(
            -1, attention_probs.size(-2), attention_probs.size(-1)
        )
        context_layer = torch.bmm(attention_probs_flat, value_layer)

        context_layer = context_layer.view(
            -1, self.num_heads, context_layer.size(-2), context_layer.size(-1)
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        return {"hidden_states": context_layer, "attention_probs": attention_probs}

    def _disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
    ):
        if relative_pos is None:
            q_size = query_layer.size(-2)
            k_size = key_layer.size(-2)
            relative_pos = build_relative_position(
                q_size, k_size,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=query_layer.device,
            )

        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)

        batch_size = query_layer.size(0) // self.num_heads

        if rel_embeddings is None:
            rel_embeddings = self.rel_embeddings.weight

        att_span = self.pos_ebd_size
        rel_embeddings = rel_embeddings[
            self.pos_ebd_size - att_span : self.pos_ebd_size + att_span, :
        ].unsqueeze(0)
        rel_embeddings = self.pos_dropout(rel_embeddings)

        score = torch.zeros_like(query_layer[:, :, :1]).expand(-1, -1, key_layer.size(-2))

        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_pos = c2p_pos.squeeze(0).expand(
            query_layer.size(0), query_layer.size(1), relative_pos.size(-1)
        )

        if "c2p" in self.pos_att_type:
            pos_key_layer = (
                self.pos_key_proj(rel_embeddings)
                if not self.share_att_key
                else self.key_proj(rel_embeddings)
            )
            pos_key_layer = self.transpose_for_scores(pos_key_layer).repeat(batch_size, 1, 1)
            c2p_scale = 1.0 / math.sqrt(self.head_size * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2) * c2p_scale)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos)
            score = score + c2p_att

        if "p2c" in self.pos_att_type:
            pos_query_layer = (
                self.pos_query_proj(rel_embeddings)
                if not self.share_att_key
                else self.query_proj(rel_embeddings)
            )
            pos_query_layer = self.transpose_for_scores(pos_query_layer).repeat(batch_size, 1, 1)
            p2c_scale = 1.0 / math.sqrt(self.head_size * scale_factor)
            p2c_att = torch.bmm(pos_query_layer * p2c_scale, key_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-2, index=c2p_pos)
            score = score + p2c_att

        return score


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------


class HELMGNNEmbeddings(nn.Module):
    """Embeddings for HELM-GNN: GPS monomer embeddings + position embeddings.

    Unlike HELMBertEmbeddings which uses word_embeddings(input_ids),
    this module receives pre-computed GPS embeddings for non-masked positions
    and uses a learnable MASK embedding for masked positions.
    """

    def __init__(self, config: HELMGNNConfig):
        super().__init__()
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.mask_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        nn.init.normal_(self.mask_embedding, std=0.02)

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        gps_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gps_embeddings: (batch, seq, hidden) - GPS outputs for non-masked,
                           MASK vector for masked positions.
            attention_mask: (batch, seq)

        Returns:
            (embeddings, position_embeddings) tuple.
        """
        batch_size, seq_len = gps_embeddings.shape[:2]

        position_ids = torch.arange(seq_len, dtype=torch.long, device=gps_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = masked_layer_norm(self.layer_norm, gps_embeddings, attention_mask)
        embeddings = self.dropout(embeddings)

        return embeddings, position_embeds


class NgieLayer(nn.Module):
    def __init__(self, config: HELMGNNConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size,
            kernel_size=config.ngie_kernel_size,
            padding=(config.ngie_kernel_size - 1) // 2,
        )
        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.ngie_dropout)

    def forward(self, hidden_states, residual_states, attention_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        if version.Version(torch.__version__) >= version.Version("1.2.0a"):
            rmask = (1 - attention_mask).bool()
        else:
            rmask = (1 - attention_mask).byte()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = self.activation(self.dropout(out))
        return masked_layer_norm(self.layer_norm, residual_states + out, attention_mask)


class TransformerBlock(nn.Module):
    def __init__(self, config: HELMGNNConfig):
        super().__init__()
        self.self_attn = DisentangledSelfAttentionWithGraphBias(config)
        self.attn_output_dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.linear1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size), nn.GELU()
        )
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        query_states: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        rel_embeddings: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        src_transposed = src.transpose(0, 1)
        attention_mask = None
        if src_key_padding_mask is not None:
            attention_mask = (~src_key_padding_mask).float()

        query_states_transposed = None
        if query_states is not None:
            query_states_transposed = query_states.transpose(0, 1)

        attn_result = self.self_attn(
            src_transposed, attention_mask,
            output_attentions=output_attentions,
            query_states=query_states_transposed,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            graph_distances=graph_distances,
        )
        attn_output = attn_result["hidden_states"].transpose(0, 1)
        attn_weights = attn_result.get("attention_probs") if output_attentions else None

        attn_output = self.attn_output_dense(attn_output)
        residual_input = query_states if query_states is not None else src
        src = residual_input + self.dropout1(attn_output)

        src = src.transpose(0, 1)
        src = masked_layer_norm(self.norm1, src)
        src = src.transpose(0, 1)

        ff_output = self.linear1(src)
        ff_output = self.linear2(ff_output)
        ff_output = self.dropout2(ff_output)
        src = src + ff_output

        src = src.transpose(0, 1)
        src = masked_layer_norm(self.norm2, src)
        src = src.transpose(0, 1)

        return src, attn_weights


class HELMGNNEncoder(nn.Module):
    def __init__(self, config: HELMGNNConfig):
        super().__init__()
        self.config = config
        self.ngie_layer = NgieLayer(config)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def get_rel_embedding(self) -> Optional[torch.Tensor]:
        if len(self.layers) > 0:
            first_layer = self.layers[0]
            if hasattr(first_layer, "self_attn") and hasattr(
                first_layer.self_attn, "rel_embeddings"
            ):
                return first_layer.self_attn.rel_embeddings.weight
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        use_emd: bool = False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        ngie_input_states = hidden_states
        hidden_states = hidden_states.transpose(0, 1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        layer_minus_2 = None
        num_layers = len(self.layers)

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.transpose(0, 1),)

            hidden_states, attn_weights = layer(
                hidden_states,
                src_key_padding_mask=key_padding_mask,
                output_attentions=output_attentions,
                graph_distances=graph_distances,
            )

            if output_attentions and attn_weights is not None:
                all_attentions = all_attentions + (attn_weights,)

            if layer_idx == 0:
                hidden_states_batch = hidden_states.transpose(0, 1)
                hidden_states_batch = self.ngie_layer(
                    ngie_input_states, hidden_states_batch, attention_mask
                )
                hidden_states = hidden_states_batch.transpose(0, 1)

            if use_emd and layer_idx == num_layers - 2:
                layer_minus_2 = hidden_states

        hidden_states = hidden_states.transpose(0, 1)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        emd_output = None
        if use_emd and layer_minus_2 is not None and position_embeddings is not None:
            emd_keys_values = layer_minus_2
            emd_query = layer_minus_2.transpose(0, 1)
            emd_query = position_embeddings + emd_query
            emd_query = emd_query.transpose(0, 1)

            rel_embeddings = self.get_rel_embedding()
            last_layer = self.layers[-1]

            for _ in range(2):
                emd_query, _ = last_layer(
                    emd_keys_values,
                    src_key_padding_mask=key_padding_mask,
                    query_states=emd_query,
                    relative_pos=None,
                    rel_embeddings=rel_embeddings,
                    graph_distances=graph_distances,
                )

            emd_output = emd_query.transpose(0, 1)

        return hidden_states, emd_output, all_hidden_states, all_attentions


class HELMGNNPooler(nn.Module):
    def __init__(self, config: HELMGNNConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            eps = torch.finfo(hidden_states.dtype).eps
            sum_mask = torch.clamp(mask_expanded.sum(1), min=eps)
            return sum_embeddings / sum_mask
        return hidden_states.mean(dim=1)


# ---------------------------------------------------------------------------
# Pre-trained base
# ---------------------------------------------------------------------------


class HELMGNNPreTrainedModel(PreTrainedModel):
    config_class = HELMGNNConfig
    base_model_prefix = "helmgnn"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


class HELMGNNModel(HELMGNNPreTrainedModel):
    """HELM-GNN base model: GPS encoder + Graph-aware Transformer."""

    def __init__(self, config: HELMGNNConfig):
        super().__init__(config)
        self.config = config

        self.gps_encoder = MonomerGPSEncoder(
            hidden_dim=config.gps_hidden_dim,
            output_dim=config.hidden_size,
            num_layers=config.gps_num_layers,
            num_heads=config.gps_num_heads,
            dropout=config.gps_dropout,
            monomer_library_path=config.monomer_library_path,
        )

        self.embeddings = HELMGNNEmbeddings(config)
        self.encoder = HELMGNNEncoder(config)
        self.pooler = HELMGNNPooler(config)

        self.post_init()

    def _build_gps_embeddings(
        self,
        monomer_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        id_to_symbol: Dict[int, str],
    ) -> torch.Tensor:
        """Build GPS embeddings for a batch, computing unique monomers once.

        Args:
            monomer_ids: (batch, seq) token IDs.
            attention_mask: (batch, seq) mask.
            id_to_symbol: Mapping from token ID to monomer symbol.

        Returns:
            (batch, seq, hidden_size) GPS embeddings.
        """
        batch_size, seq_len = monomer_ids.shape
        device = monomer_ids.device
        mask_id = self.config.mask_token_id
        pad_id = self.config.pad_token_id

        # Collect unique non-special monomer IDs
        unique_ids = set()
        ids_np = monomer_ids.cpu().numpy()
        special_ids = {self.config.pad_token_id, self.config.bos_token_id,
                       self.config.eos_token_id, self.config.mask_token_id}
        for row in ids_np:
            for tid in row:
                if int(tid) not in special_ids:
                    unique_ids.add(int(tid))

        # Encode unique monomers with GPS
        unique_list = sorted(unique_ids)
        symbols = [id_to_symbol.get(uid, "") for uid in unique_list]
        if symbols:
            unique_embeddings = self.gps_encoder(symbols, device)  # (num_unique, hidden)
        else:
            unique_embeddings = torch.zeros(0, self.config.hidden_size, device=device)

        # Build lookup: token_id -> embedding index
        id_to_idx = {uid: i for i, uid in enumerate(unique_list)}

        # Construct output
        output = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)

        for b in range(batch_size):
            for s in range(seq_len):
                tid = int(ids_np[b, s])
                if tid == mask_id:
                    output[b, s] = self.embeddings.mask_embedding
                elif tid in id_to_idx:
                    output[b, s] = unique_embeddings[id_to_idx[tid]]

        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
        id_to_symbol: Optional[Dict[int, str]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if id_to_symbol is None:
            id_to_symbol = {}

        # GPS embeddings
        gps_emb = self._build_gps_embeddings(input_ids, attention_mask, id_to_symbol)

        # Position embeddings
        embeddings, position_embeddings = self.embeddings(gps_emb, attention_mask)

        # Encoder
        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            graph_distances=graph_distances,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_emd=False,
        )

        last_hidden_state = encoder_outputs[0]
        pooler_output = self.pooler(last_hidden_state, attention_mask)

        if not return_dict:
            return (last_hidden_state, pooler_output, encoder_outputs[2], encoder_outputs[3])

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs[2],
            attentions=encoder_outputs[3],
        )


class HELMGNNLMHead(nn.Module):
    """MLM head for monomer-level prediction."""

    def __init__(self, config: HELMGNNConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.activation = nn.GELU()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.decoder(hidden_states)


class HELMGNNForMaskedLM(HELMGNNPreTrainedModel):
    """HELM-GNN for monomer-level Masked Language Modeling."""

    def __init__(self, config: HELMGNNConfig):
        super().__init__(config)
        self.helmgnn = HELMGNNModel(config)
        self.lm_head = HELMGNNLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
        id_to_symbol: Optional[Dict[int, str]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        use_emd: bool = True,
    ) -> Union[Tuple, MaskedLMOutput]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if id_to_symbol is None:
            id_to_symbol = {}

        # GPS embeddings
        gps_emb = self.helmgnn._build_gps_embeddings(
            input_ids, attention_mask, id_to_symbol
        )

        # Embedding layer
        embeddings, position_embeddings = self.helmgnn.embeddings(gps_emb, attention_mask)

        # Encoder with EMD
        encoder_outputs = self.helmgnn.encoder(
            embeddings,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            graph_distances=graph_distances,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_emd=use_emd,
        )

        if use_emd and encoder_outputs[1] is not None:
            sequence_output = encoder_outputs[1]
        else:
            sequence_output = encoder_outputs[0]

        prediction_scores = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores, encoder_outputs[2], encoder_outputs[3])
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=encoder_outputs[2],
            attentions=encoder_outputs[3],
        )


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            identity = x
            x = layer(x)
            if x.shape == identity.shape:
                x = x + identity
            x = self.activation(x)
            x = norm(x)
            x = dropout(x)
        return self.output_layer(x)


class HELMGNNForSequenceClassification(HELMGNNPreTrainedModel):
    """HELM-GNN for sequence classification/regression."""

    def __init__(self, config: HELMGNNConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.helmgnn = HELMGNNModel(config)

        if config.classifier_num_layers > 0:
            hidden_dims = [config.hidden_size] * config.classifier_num_layers
            self.classifier = MLPHead(
                config.hidden_size, config.num_labels, hidden_dims, config.classifier_dropout
            )
        else:
            self.dropout = nn.Dropout(config.classifier_dropout)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        graph_distances: Optional[torch.Tensor] = None,
        id_to_symbol: Optional[Dict[int, str]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        outputs = self.helmgnn(
            input_ids,
            attention_mask=attention_mask,
            graph_distances=graph_distances,
            id_to_symbol=id_to_symbol,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output
        if hasattr(self, "dropout"):
            pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
