import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from lib.modules.rotary import RotaryEmbeddings


class LeanSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_core: Optional[nn.Module] = None,
        hidden_dropout_prob: float = 0,
        layer_norm_eps: float = 1e-12,
        sandwich_norm: bool = False,
        dense_qkv: Optional[nn.Linear] = None,
        dense_out: Optional[nn.Linear] = None,
        **kwargs,
    ):
        """Attention layer that does not hog GPU memory"""
        super().__init__()
        if attention_core is None:
            attention_core = SimpleAttentionCore(hidden_size, num_attention_heads, **kwargs)
        else:
            assert len(kwargs) == 0, f"Unexpected parameters: {kwargs}"

        self.hidden_size = hidden_size
        self.attention_core = attention_core
        self.dense_qkv = nn.Linear(hidden_size, hidden_size * 3) if dense_qkv is None else dense_qkv
        self.dense_out = nn.Linear(hidden_size, hidden_size) if dense_out is None else dense_out
        assert dense_qkv.in_features == dense_out.in_features == dense_out.out_features == hidden_size
        assert dense_qkv.out_features == hidden_size * 3

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.sandwich_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps) if sandwich_norm else None
        self.output_dropout = nn.Dropout(hidden_dropout_prob, inplace=False)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        hidden_states_ln = self.layer_norm(hidden_states)
        qkv_output = self.dense_qkv(hidden_states_ln)
        query, key, value = qkv_output.split(self.hidden_size, dim=qkv_output.ndim - 1)
        attention_output, attention_probs = self._maybe_checkpoint(
            self.attention_core, query, key, value, attention_mask
        )
        projected_context_layer = self.dense_out(attention_output)
        if self.sandwich_norm:
            projected_context_layer = self.sandwich_norm(projected_context_layer)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        outputs = projected_context_layer_dropout + hidden_states.to(torch.float32, copy=False)
        return (outputs, attention_probs) if output_attentions else (outputs,)

    def _maybe_checkpoint(self, func, *args):
        return checkpoint(func, *args) if torch.is_grad_enabled() else func(*args)


class SimpleAttentionCore(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.attention_dropout = nn.Dropout(attention_probs_dropout, inplace=False)
        self.hidden_size, self.num_attention_heads = hidden_size, num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

    def transpose_for_scores(self, x):
        """transpose from [batch, seq_length, full_hid_size] to [batch, num_heads, seq_length, head_size]"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attention_mask):
        """
        :param query: [batch_size, query_seq_len, hidden_size]
        :param key: [batch_size, kv_seq_len, hidden_size]
        :param value: [batch_size, kv_seq_len, hidden_size]
        :param attention_mask: [batch, query_seq_len, hidden_size]
        :return: (outputs, probs)
          - outputs shape: [batch_size, query_seq_len, hidden_size]
          - probs shape: [batch_size, num_heads, query_seq_len, kv_seq_len]
        """
        query, key, value = map(self.transpose_for_scores, (query, key, value))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(query.shape[-1])

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(2, 1).flatten(2)
        return attention_output, attention_probs


class RotaryAttentionCore(SimpleAttentionCore):
    """Attention core that applies rotary embeddings to queries and keys before computing dot products"""

    def __init__(
        self, hidden_size: int, num_attention_heads: int, rotary_emb: Optional[RotaryEmbeddings] = None, **kwargs
    ):
        super().__init__(hidden_size, num_attention_heads, **kwargs)
        if rotary_emb is None:
            rotary_emb = RotaryEmbeddings(self.attention_head_size)
        self.rotary_emb = rotary_emb

    def rotate(self, tensor: torch.Tensor):
        """:param tensor: query or key, shape: [batch_size, query_seq_len, hidden_size]"""
        tensor_split_heads = tensor.view(*(tensor.shape[:-1] + (self.num_attention_heads, self.attention_head_size)))
        return self.rotary_emb(tensor_split_heads).view(*tensor.shape)

    def forward(self, query, key, value, attention_mask):
        return super().forward(self.rotate(query), self.rotate(key), value, attention_mask)
