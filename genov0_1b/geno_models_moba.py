import os
from torch import Tensor, nn
from transformers.modeling_utils import PreTrainedModel
from .geno_moe import MoE
import torch
from transformers.modeling_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers import PreTrainedModel
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers.activations import ACT2FN
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from loguru import logger
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
import math
import warnings
import time
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from moba import register_moba, MoBAConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import torch.nn.functional as F
from mamba_ssm import Mamba as MambaBlock
from torch import nn


class SimpleRMSNorm(nn.Module):
    """
    SimpleRMSNorm

    Args:
        dim (int): dimension of the embedding

    Usage:
    We can use SimpleRMSNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> simple_rms_norm = SimpleRMSNorm(dim=512)
        >>> simple_rms_norm(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale


class GenoConfig(PretrainedConfig):
    model_type = "GenoMamba"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            expand,
            d_state,
            d_conv,
            num_experts,
            num_experts_per_token,
            word_weights=None,
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=12,
            n_head=12,
            n_inner=None,
            return_all_heads=False,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.return_all_heads = return_all_heads
        self.expand = expand
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.n_positions = n_positions
        self.max_token_len = n_positions
        self.weights = word_weights
        self.dim = n_embd
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.depth = n_layer
        self.n_head = n_head
        self.heads = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class GenoMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = getattr(config, "dim", getattr(config, "hidden_size", 768))
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        if self.training:
            hidden_states = self.dropout(hidden_states)
        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k shape: (batch_size, num_heads, seq_len, head_dim)
    # cos, sin shape: (seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, head_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)  # 修正括号
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 修正括号
    return q_embed, k_embed


class GptAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True
        self.pruned_heads = set()

    def _create_rotary_emb(self, position_ids, dim, device):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        position_ids = position_ids.float()
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = torch.tril(
                torch.ones((query_length, key_length), dtype=torch.bool, device=query.device)
            ).view(1, 1, query_length, key_length)
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        if self.training:
            attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,  # 修正方括号
            head_mask: Optional[torch.FloatTensor] = None,  # 修正方括号
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 修正方括号
            encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 修正方括号
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Generate position ids
        seq_length = hidden_states.size(1)
        if layer_past is not None:
            past_length = layer_past[0].size(-2)
        else:
            past_length = 0
        position_ids = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)

        # Apply rotary positional embeddings
        if not self.is_cross_attention:
            cos, sin = self._create_rotary_emb(position_ids, self.head_dim, hidden_states.device)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=-2)
            value = torch.cat((layer_past[1], value), dim=-2)
        present = (key, value) if use_cache else None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        # 根据 self.training 控制 Dropout 层的行为
        if self.training:
            attn_output = self.resid_dropout(attn_output)
        else:
            attn_output = attn_output  # 测试时不应用 Dropout

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class GptFlashAttention2(GptAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,  # 修正方括号
            head_mask: Optional[torch.FloatTensor] = None,  # 修正方括号
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 修正方括号
            encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 修正方括号
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        bsz, seq_len, _ = hidden_states.size()
        if encoder_hidden_states is not None:
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
            is_causal = False
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            is_causal = True

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Generate position ids
        if layer_past is not None:
            past_length = layer_past[0].size(-2)
        else:
            past_length = 0
        position_ids = torch.arange(past_length, past_length + seq_len, device=hidden_states.device)

        # Apply rotary positional embeddings
        if not self.is_cross_attention:
            cos, sin = self._create_rotary_emb(position_ids, self.head_dim, hidden_states.device)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=-2)
            value = torch.cat((layer_past[1], value), dim=-2)
        present = (key, value) if use_cache else None

        # Reshape for flash attention
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        attn_dropout = self.attn_dropout.p if self.training else 0.0

        # logger.info(f"query shape: {query.shape}")
        attn_output = self._flash_attention_forward(
            query, key, value, attention_mask, seq_len, dropout=attn_dropout, is_causal=is_causal
        )
        # logger.info(f"attn_output shape: {attn_output.shape}")

        attn_output = attn_output.reshape(bsz, seq_len, self.embed_dim)
        # logger.info(f"attn_output re-shape: {attn_output.shape}")
        attn_output.to(torch.float16)
        self.c_proj = self.c_proj.to(torch.float16)
        attn_output = attn_output.to(torch.float16)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        return outputs

    def _flash_attention_forward(self, query, key, value, attention_mask, query_length, dropout=0.0, is_causal=True):
        value = value.to(torch.float32)
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
        )


class GptSdpaAttention(GptAttention):
    """
    GPT2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `GPT2Attention` as the weights of the module stays untouched. The only changes are on the forward pass
    to adapt to the SDPA API.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Idea adapted from transformers.models.bert.modeling_bert.BertSdpaSelfAttention.__init__
        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()`. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        self.require_contiguous_qkv = version.parse(get_torch_version()) < version.parse("2.2.0")

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if output_attentions or head_mask is not None:
            logger.info(
                "`GPT2SdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        bsz, q_len, _ = hidden_states.size()

        # Initial attention projections
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Optional kv caching
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present, None


class MobaAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True
        self.attn = ALL_ATTENTION_FUNCTIONS["moba"]
        self.pruned_heads = set()

    def _create_rotary_emb(self, position_ids, dim, device):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        position_ids = position_ids.float()
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Generate position ids
        seq_length = hidden_states.size(1)
        if layer_past is not None:
            past_length = layer_past[0].size(-2)
        else:
            past_length = 0
        position_ids = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)

        # Apply rotary positional embeddings
        if not self.is_cross_attention:
            cos, sin = self._create_rotary_emb(position_ids, self.head_dim, hidden_states.device)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=-2)
            value = torch.cat((layer_past[1], value), dim=-2)
        present = (key, value) if use_cache else None
        # logger.info(f"query shape: {query.shape}")
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            query, key, value = query.half(), key.half(), value.half()
            attn_output, attn_weights = self.attn(query=query, key=key, value=value, module=self)
        # logger.info(f"attn_output shape: {attn_output.shape}")
        # attn_output = attn_output.unsqueeze(0)  # 变为 [1, 1000, 4, 64]
        # attn_output = attn_output.permute(0, 2, 1, 3)
        # logger.info(f"attn_output re-shape: {attn_output.shape}")

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        self.c_proj = self.c_proj.to(torch.float16)
        attn_output = attn_output.to(torch.float16)
        attn_output = self.c_proj(attn_output)
        if self.training:
            attn_output = self.resid_dropout(attn_output)
        else:
            attn_output = attn_output

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


GPT2_ATTENTION_CLASSES = {"eager": GptAttention,
                          "flash_attention_2": GptFlashAttention2,
                          "sdpa": GptSdpaAttention,
                          "moba": MobaAttention}


class MambaMoELayer(nn.Module):
    def __init__(
            self,
            dim: int,
            d_state: int,
            d_conv: int,
            num_experts: int = 8,
            num_experts_per_token: int = 2,
            expand: int = 1,
            device=None,
            *args,
            **kwargs,
    ):
        """
        Initialize the MambaMoELayer.

        Args:
            dim (int): Dimension of the input tensor.
            d_state (int): Dimension of the state tensor.
            d_conv (int): Dimension of the convolutional tensor.
            num_experts (int, optional): Number of experts. Defaults to 8.
            num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.expand = expand
        self.device = device
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        # Mamba
        self.mamba = MambaBlock(
            d_model=dim,
            expand=expand,
            d_state=d_state,
            d_conv=d_conv
        )

        # MoE
        self.moe = MoE(
            dim,
            num_experts=num_experts,
            hidden_dim=dim * 4,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the MambaMoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the MambaMoELayer.
        """
        skip = x

        x = SimpleRMSNorm(self.dim)(x)
        x = self.mamba(x) + x

        x = SimpleRMSNorm(self.dim)(x)
        moe_out, _ = self.moe(x)
        x = moe_out + skip
        return x


class GenoMambaMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.dim

        self.ln_1 = nn.LayerNorm(config.dim, eps=config.layer_norm_epsilon)
        self.mamba_moe_layer = MambaMoELayer(
            config.dim,
            config.d_state,
            config.d_conv,
            config.num_experts,
            config.num_experts_per_token,
            config.expand
        )
        self.ln_2 = nn.LayerNorm(config.dim, eps=config.layer_norm_epsilon)
        self.mlp = GenoMLP(inner_dim, config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states.to(self.ln_1.bias.dtype))
        mamba_outputs = self.mamba_moe_layer(hidden_states)
        hidden_states = mamba_outputs + residual

        residual = hidden_states
        # logger.info(f"TYPE 730 :{self.ln_2.weight.dtype}, {self.ln_2.bias.dtype}, {hidden_states.dtype}")
        hidden_states = self.ln_2(hidden_states.to(self.ln_2.bias.dtype))
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class GenoMambaBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.dim
        # print(f"inner_dim: {inner_dim}")
        # print(f"config: {config.dim}, {config.expand}, {config.d_state}, {config.d_conv}")
        self.ln_1 = nn.LayerNorm(config.dim, eps=config.layer_norm_epsilon)
        self.mamba_layer = MambaBlock(
            d_model=config.dim,
            expand=config.expand,
            d_state=config.d_state,
            d_conv=config.d_conv
        )
        self.ln_2 = nn.LayerNorm(config.dim, eps=config.layer_norm_epsilon)
        self.mlp = GenoMLP(inner_dim, config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]]
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states.to(self.ln_1.bias.dtype))
        for sublayer in self.mamba_layer.children():
            if hasattr(sublayer, 'weight'):
                hidden_states.to(sublayer.weight.device)  # 获取子层的设备信息
                break
        mamba_outputs = self.mamba_layer(hidden_states)

        residual = residual.to(mamba_outputs.device)
        hidden_states = mamba_outputs + residual

        residual = hidden_states
        # logger.info(f"TYPE 695 :{self.ln_2.weight.dtype}, {self.ln_2.bias.dtype}, {hidden_states.to(self.ln_2.bias.dtype).dtype}")
        hidden_states = self.ln_2(hidden_states.to(self.ln_2.bias.dtype))
        feed_forward_hidden_states = self.mlp(hidden_states)

        residual = residual.to(feed_forward_hidden_states.device)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class GenoTransformerMoeBlock(nn.Module):
    def __init__(self,
                 config,
                 num_experts: int,
                 num_experts_per_token: int,
                 layer_idx=None):
        super().__init__()
        self.idx = layer_idx
        self.config = config
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        attention_class = GPT2_ATTENTION_CLASSES[config._attn_implementation]
        self._attn_implementation = config._attn_implementation
        if config._attn_implementation != "multi_query_attention":
            self.attn = attention_class(config=config, layer_idx=layer_idx)
        else:
            self.attn = attention_class(config.hidden_size, config.n_head)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GenoMLP(inner_dim, config)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token
        self.moe = MoE(
            config.hidden_size,
            num_experts=num_experts,
            hidden_dim=config.hidden_size * 4,
        )

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ):
        residual1 = hidden_states
        hidden_states = self.ln_1(hidden_states.to(self.ln_1.bias.dtype))
        if self._attn_implementation != "multi_query_attention":
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]
        else:
            attn_output, attn_weights, past_key_value = self.attn(hidden_states)
        hidden_states = attn_output + residual1

        residual2 = hidden_states
        hidden_states, _ = self.moe(hidden_states)
        hidden_states = self.ln_2(hidden_states.to(self.ln_2.bias.dtype))
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = feed_forward_hidden_states + residual2

        return hidden_states


class GenoTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.idx = layer_idx
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        attention_class = GPT2_ATTENTION_CLASSES[config._attn_implementation]
        self._attn_implementation = config._attn_implementation
        self.attn = attention_class(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = attention_class(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GenoMLP(inner_dim, config)

    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states.to(self.ln_1.bias.dtype))
        if self._attn_implementation != "multi_query_attention":
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        else:
            attn_output, attn_weights, past_key_value = self.attn(hidden_states)
        # outputs = attn_output
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            # outputs = cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        # logger.info(f"TYPE 656 :{self.ln_2.weight.dtype}, {self.ln_2.bias.dtype}, {hidden_states.dtype}")
        hidden_states = self.ln_2(hidden_states.to(self.ln_2.bias.dtype))
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        # if use_cache:
        #     outputs = (hidden_states,) #+ outputs
        # else:
        #     outputs = (hidden_states,) # + outputs[1:]

        return hidden_states  # outputs  # hidden_states, present, (attentions, cross_attentions)


class CausalConv1d_v0(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)

    def forward(self, x):
        # x: [B, C, T]
        pad = self.kernel_size - 1
        x = F.pad(x, (pad, 0))  # 只 pad 左边
        return self.conv(x)


class MGenoBlock(nn.Module):
    """
    MGenoBlock is a module that combines MambaBlock, MambaMoELayer, and TransformerBlock
    to process input tensors.

    Args:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int, optional): The number of experts in MambaMoELayer. Defaults to 8.
        num_experts_per_token (int, optional): The number of experts per token in MambaMoELayer. Defaults to 2.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int): The number of experts in MambaMoELayer.
        num_experts_per_tok (int): The number of experts per token in MambaMoELayer.
        mamba_layer (MambaBlock): The MambaBlock layer.
        mamba_moe_layer (MambaMoELayer): The MambaMoELayer layer.
        transformer (TransformerBlock): The TransformerBlock layer.

    """

    def __init__(
            self,
            dim: int,
            d_state: int,
            d_conv: int,
            heads: int,
            idx: int = -1,
            num_experts: int = 8,
            num_experts_per_token: int = 2,
            expand: int = 1,
            vocab_size: int = 1,
            n_positions: int = 1,
            n_ctx: int = 1,
            attn_class="flash_attention_2",
            devices=None,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.idx = idx
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.expand = expand
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token
        self.is_parallel = False
        self.devices = devices

        geno_config = GenoConfig(n_embd=dim,
                                 n_head=heads,
                                 d_state=d_state,
                                 d_conv=d_conv,
                                 heads=heads,
                                 expand=expand,
                                 num_experts=num_experts,
                                 num_experts_per_token=num_experts_per_token,
                                 vocab_size=vocab_size,  # 的词汇表大小
                                 n_positions=n_positions,  # 的最大序列长度
                                 n_ctx=n_ctx,  # 上下文窗口大小
                                 layer_norm_epsilon=1e-5)
        # Mamba
        self.mamba_layer1 = GenoMambaBlock(geno_config)
        #
        # # Mamba MoE layer2
        self.mamba_moe_layer2 = GenoMambaMoeBlock(geno_config)
        #
        # # Mamba
        self.mamba_layer3 = GenoMambaBlock(geno_config)
        #
        # # Mamba MoE layer2
        self.mamba_moe_layer4 = GenoMambaMoeBlock(geno_config)

        # Transformer
        gpt_config = GPT2Config(n_embd=dim,
                                n_head=heads,
                                vocab_size=vocab_size,  # 的词汇表大小
                                n_positions=n_positions,  # 的最大序列长度
                                n_ctx=n_ctx,  # 上下文窗口大小
                                _attn_implementation=attn_class
                                )
        self.transformer5 = GenoTransformerBlock(gpt_config)

        # # Mamba MoE layer1
        self.mamba_moe_layer6 = GenoMambaMoeBlock(geno_config)
        #
        # # Mamba
        self.mamba_layer7 = GenoMambaBlock(geno_config)
        #
        # # Mamba MoE layer1
        self.mamba_moe_layer8 = GenoMambaMoeBlock(geno_config)

    def parallel(self):
        if not isinstance(self.devices, list):
            print(f"Jamba block devices is not a list.")
            return False
        if len(self.devices) != 4:
            print(f"Jamba block devices is not 4 items.")
            return False
        self.mamba_layer1.to(self.devices[0])
        self.mamba_moe_layer2.to(self.devices[1])
        self.mamba_layer3.to(self.devices[0])
        self.mamba_moe_layer4.to(self.devices[1])
        self.transformer5.to(self.devices[3])
        self.mamba_moe_layer6.to(self.devices[2])
        self.mamba_layer7.to(self.devices[0])
        self.mamba_moe_layer8.to(self.devices[2])
        self.is_parallel = True

    def forward(self, x: Tensor) -> Tensor:
        if self.is_parallel:
            x = x.to(self.devices[0])
        x = self.mamba_layer1(x)

        if self.is_parallel:
            x = x.to(self.devices[1])
        x = self.mamba_moe_layer2(x)
        #
        if self.is_parallel:
            x = x.to(self.devices[0])
        x = self.mamba_layer3(x)
        #
        if self.is_parallel:
            x = x.to(self.devices[1])
        x = self.mamba_moe_layer4(x)

        if self.is_parallel:
            x = x.to(self.devices[3])
        x = self.transformer5(x)

        if self.is_parallel:
            x = x.to(self.devices[2])
        x = self.mamba_moe_layer6(x)

        if self.is_parallel:
            x = x.to(self.devices[0])
        x = self.mamba_layer7(x)

        if self.is_parallel:
            x = x.to(self.devices[2])
        x = self.mamba_moe_layer8(x)

        return x


class MGenoCNNBlock(nn.Module):
    """
    MGenoBlock is a module that combines MambaBlock, MambaMoELayer, and TransformerBlock
    to process input tensors.

    Args:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int, optional): The number of experts in MambaMoELayer. Defaults to 8.
        num_experts_per_token (int, optional): The number of experts per token in MambaMoELayer. Defaults to 2.

    Attributes:
        dim (int): The input dimension.
        d_state (int): The dimension of the state in MambaBlock and MambaMoELayer.
        d_conv (int): The dimension of the convolutional output in MambaBlock and MambaMoELayer.
        heads (int): The number of attention heads in TransformerBlock.
        num_experts (int): The number of experts in MambaMoELayer.
        num_experts_per_tok (int): The number of experts per token in MambaMoELayer.
        mamba_layer (MambaBlock): The MambaBlock layer.
        mamba_moe_layer (MambaMoELayer): The MambaMoELayer layer.
        transformer (TransformerBlock): The TransformerBlock layer.

    """

    def __init__(
            self,
            dim: int,
            d_state: int,
            d_conv: int,
            heads: int,
            idx: int = -1,
            num_experts: int = 8,
            num_experts_per_token: int = 2,
            expand: int = 1,
            vocab_size: int = 1,
            n_positions: int = 1,
            n_ctx: int = 1,
            attn_class="flash_attention_2",
            devices=None,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.idx = idx
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.expand = expand
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token
        self.is_parallel = False
        self.devices = devices

        geno_config = GenoConfig(n_embd=dim,
                                 n_head=heads,
                                 d_state=d_state,
                                 d_conv=d_conv,
                                 heads=heads,
                                 expand=expand,
                                 num_experts=num_experts,
                                 num_experts_per_token=num_experts_per_token,
                                 vocab_size=vocab_size,  # 的词汇表大小
                                 n_positions=n_positions,  # 的最大序列长度
                                 n_ctx=n_ctx,  # 上下文窗口大小
                                 layer_norm_epsilon=1e-5)
        # Mamba
        self.mamba_layer1 = GenoMambaBlock(geno_config)
        #
        # # Mamba MoE layer2
        self.mamba_moe_layer2 = GenoMambaMoeBlock(geno_config)
        #
        # # Mamba
        self.mamba_layer3 = GenoMambaBlock(geno_config)
        #
        # # Mamba MoE layer2
        self.mamba_moe_layer4 = GenoMambaMoeBlock(geno_config)

        # Transformer
        gpt_config = GPT2Config(n_embd=dim,
                                n_head=heads,
                                vocab_size=vocab_size,  # 的词汇表大小
                                n_positions=n_positions,  # 的最大序列长度
                                n_ctx=n_ctx,  # 上下文窗口大小
                                _attn_implementation=attn_class
                                )
        self.transformer5 = GenoTransformerBlock(gpt_config)

        # # Mamba MoE layer1
        self.mamba_moe_layer6 = GenoMambaMoeBlock(geno_config)
        #
        # # Mamba
        self.mamba_layer7 = GenoMambaBlock(geno_config)
        #
        # # Mamba MoE layer1
        self.mamba_moe_layer8 = GenoMambaMoeBlock(geno_config)

        self.layer_cnn1 = CausalConv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=0)
        self.layer_cnn2 = CausalConv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=0)
        self.layer_cnn3 = CausalConv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=0)
        self.layer_cnn4 = CausalConv1d(in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=0)
        self.layer_activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def parallel(self):
        if not isinstance(self.devices, list):
            print(f"Jamba block devices is not a list.")
            return False
        if len(self.devices) != 4:
            print(f"Jamba block devices is not 4 items.")
            return False
        self.mamba_layer1.to(self.devices[0])
        self.mamba_moe_layer2.to(self.devices[1])
        self.mamba_layer3.to(self.devices[0])
        self.mamba_moe_layer4.to(self.devices[1])
        self.transformer5.to(self.devices[3])
        self.mamba_moe_layer6.to(self.devices[2])
        self.mamba_layer7.to(self.devices[0])
        self.mamba_moe_layer8.to(self.devices[2])
        self.is_parallel = True

    def forward(self, x: Tensor) -> Tensor:

        x = x.permute(0, 2, 1)  # [B, dim, seq_len]
        x = self.dropout(self.layer_activation(self.layer_cnn1(x)))
        x = x.permute(0, 2, 1)  # [B, seq_len, dim]

        if self.is_parallel:
            x = x.to(self.devices[0])
        x = self.mamba_layer1(x)

        if self.is_parallel:
            x = x.to(self.devices[1])
        x = self.mamba_moe_layer2(x)
        #
        if self.is_parallel:
            x = x.to(self.devices[0])
        x = self.mamba_layer3(x)
        #
        if self.is_parallel:
            x = x.to(self.devices[1])
        x = self.mamba_moe_layer4(x)

        x = x.permute(0, 2, 1)  # [B, dim, seq_len]
        x = self.dropout(self.layer_activation(self.layer_cnn2(x)))
        x = x.permute(0, 2, 1)  # [B, seq_len, dim]

        if self.is_parallel:
            x = x.to(self.devices[3])
        x = self.transformer5(x)

        x = x.permute(0, 2, 1)  # [B, dim, seq_len]
        x = self.dropout(self.layer_activation(self.layer_cnn3(x)))
        x = x.permute(0, 2, 1)  # [B, seq_len, dim]

        if self.is_parallel:
            x = x.to(self.devices[2])
        x = self.mamba_moe_layer6(x)

        if self.is_parallel:
            x = x.to(self.devices[0])
        x = self.mamba_layer7(x)

        if self.is_parallel:
            x = x.to(self.devices[2])
        x = self.mamba_moe_layer8(x)

        x = x.permute(0, 2, 1)  # [B, dim, seq_len]
        x = self.dropout(self.layer_activation(self.layer_cnn4(x)))
        x = x.permute(0, 2, 1)  # [B, seq_len, dim]

        return x


class MGenoConfig(PretrainedConfig):
    def __init__(
            self,
            dim: int = 100,
            depth: int = 6,
            num_tokens: int = 20750,
            max_seq_len: int = 512,  # 可以选择一个适当的最大序列长度
            d_state: int = 256,
            d_conv: int = 128,
            heads: int = 4,
            num_experts: int = 4,  # 使用 4 个专家
            num_experts_per_token: int = 2,
            pre_emb_norm: bool = False,
            return_embeddings: bool = False,
            n_future_tokens: int = 1,
            return_all_heads: bool = False,
            attn_class: str = "flash_attention_2",
            initializer_range: float = 0.02,  # 添加initializer_range
            **kwargs  # 允许其他额外的配置
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.n_future_tokens = n_future_tokens
        self.attn_class = attn_class
        self.return_all_heads = return_all_heads
        self.depth = depth
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.pre_emb_norm = pre_emb_norm
        self.return_embeddings = return_embeddings
        self.initializer_range = initializer_range  # 保存初始化范围

    def to_dict(self):
        # 将配置转化为字典，方便保存和加载
        output = super().to_dict()
        output.update({
            "dim": self.dim,
            "depth": self.depth,
            "num_tokens": self.num_tokens,
            "max_seq_len": self.max_seq_len,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "heads": self.heads,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.num_experts_per_token,
            "pre_emb_norm": self.pre_emb_norm,
            "return_embeddings": self.return_embeddings,
            "initializer_range": self.initializer_range,
        })
        return output


class MGeno(PreTrainedModel):
    """
    MGeno model implementation based on the transformer pre-trained model interface.

    Args:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int, optional): Number of experts. Defaults to 8.
        num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
        pre_emb_norm (bool, optional): Whether to normalize the embeddings. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

    Attributes:
        layers (nn.ModuleList): List of MGeno layers.
        embed (nn.Embedding): Embedding layer.
        norm (nn.LayerNorm or nn.Identity): Normalization layer.
        model_parallel (bool): Whether the model is in parallel mode.
        device_map (dict): Device map for parallelization.
    """

    config_class = MGenoConfig
    base_model_prefix = "geno"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
            self,
            config: MGenoConfig,
            *args,
            **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.dim = config.dim
        self.depth = config.depth
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.heads = config.heads
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_token
        self.pre_emb_norm = config.pre_emb_norm
        self.return_embeddings = config.return_embeddings
        self.num_tokens = config.num_tokens
        self.expand = config.expand
        self.weights = config.weights
        self.max_token_len = config.max_token_len,

        # Layers
        self.layers = nn.ModuleList(
            [
                MGenoBlock(
                    idx=idx,
                    dim=self.dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    heads=self.heads,
                    num_experts=self.num_experts,
                    num_experts_per_token=self.num_experts_per_tok,
                    expand=self.expand,
                    vocab_size=config.num_tokens,  # 的词汇表大小
                    n_positions=config.max_token_len,  # 的最大序列长度
                    n_ctx=config.max_token_len,  # 上下文窗口大小
                    attn_class=config.attn_class,  # , "multi_query_attention",
                    device=None,
                    config=config
                )
                for idx in range(self.depth)
            ]
        )

        # Pre Emb
        self.embed = nn.Embedding(self.num_tokens, self.dim)

        # Embedding Norm
        self.norm = (
            nn.LayerNorm(self.dim) if self.pre_emb_norm else nn.Identity()
        )

        # output
        self.output_head = nn.Linear(self.dim, self.num_tokens, bias=False)

        self.model_parallel = False
        self.device_map = None
        # self.parallelize(self.depth)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parallelize(self, rank, device_num=4):
        """
        Enable model parallelism by distributing layers across available GPUs.

        Args:
            device_num (int): Number of available GPUs.
        """
        self.model_parallel = True
        self.device_map = {}

        # Calculate which GPU each layer should be assigned to
        # layers_per_device = self.depth // device_num
        devices = []
        for i in range(device_num):
            device_id = (rank + i) % device_num
            devices.append(torch.device("cuda", device_id))
        self.device_map["layers"] = devices
        # Assign embedding, norm, and output_head to appropriate devices
        self.device_map["embed"] = devices[-1]
        self.device_map["norm"] = devices[-1]
        self.device_map["output_head"] = devices[-1]  # torch.device("cuda", (rank + 1) % device_num)
        print(self.device_map)

        self.layers = nn.ModuleList(
            [
                JambaBlockV2(
                    idx=idx,
                    dim=self.dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    heads=self.heads,
                    num_experts=self.num_experts,
                    num_experts_per_token=self.num_experts_per_tok,
                    expand=self.expand,
                    vocab_size=self.config.num_tokens,  # 的词汇表大小
                    n_positions=self.config.max_token_len,  # 的最大序列长度
                    n_ctx=self.config.max_token_len,  # 上下文窗口大小
                    config=self.config,
                    devices=self.device_map.get(f"layers")
                )
                for idx in range(self.depth)
            ]
        )

        # Apply model parallelism to each layer and module
        for idx, layer in enumerate(self.layers):
            layer.parallel()

        for name, module in self.named_modules():
            if name in ["embed", "norm"]:
                module.to(self.device_map[name])
            elif name == "output_head":
                module.to(self.device_map[name])
                if self.weights is not None:
                    self.weights = self.weights.to(self.device_map[name])

    def deparallelize(self):
        """Disable model parallelism."""
        self.model_parallel = False
        for name, module in self.named_modules():
            module.to("cpu")

    def get_output_embeddings(self):
        return self.output_head  # 修改为 output_head

    def set_output_embeddings(self, new_embeddings):
        self.output_head = new_embeddings  # 修改为 output_head

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Implement generation preparation logic
        token_type_ids = kwargs.get("token_type_ids", None)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "position_ids": position_ids, "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids}
        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # First, move input embeddings to the assigned device
        if self.device_map is not None:
            input_ids = input_ids.to(self.embed.weight.device)
        x = self.embed(input_ids)
        # 检查并确保嵌入层输出连续
        if not x.is_contiguous():
            x = x.contiguous()
        if isinstance(self.norm, nn.Identity):
            pass
        else:
            if self.device_map is not None:
                x = x.to(self.norm.weight.device)
                # 跨设备移动后检查连续性
                if not x.is_contiguous():
                    x = x.contiguous()
            x = self.norm(x)
        # Forward pass through layers with device-specific assignment
        if self.device_map is not None:
            device_id = self.device_map.get("layers")[0]
            x = x.to(device_id)
            # 跨设备移动后检查连续性
            if not x.is_contiguous():
                x = x.contiguous()
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # 检查每层输出的连续性
            if not x.is_contiguous():
                x = x.contiguous()
        # Compute output logits
        if self.device_map is not None:
            device_id = self.device_map["output_head"]
            x = x.to(device_id)
            # 跨设备移动后检查连续性
            if not x.is_contiguous():
                x = x.contiguous()
        # logger.info(f"{x.shape}")
        hidden_states = x
        # last_hidden_states = x[:, -1, :]
        # logger.info(f"{last_hidden_states.shape}")
        logits = self.output_head(x.to(self.output_head.weight.dtype))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :]
            # 切片操作后检查连续性
            if not shift_logits.is_contiguous():
                shift_logits = shift_logits.contiguous()
            shift_labels = labels[..., 1:]
            # 切片操作后检查连续性
            if not shift_labels.is_contiguous():
                shift_labels = shift_labels.contiguous()
            shift_labels = shift_labels.to(logits.device)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            return (logits,) + (loss,) if loss is not None else (logits,)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    # def forward(
    #         self,
    #         input_ids: Optional[torch.LongTensor] = None,
    #         labels: Optional[torch.LongTensor] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
    #     """
    #     Forward pass logic with model parallelism support.
    #     Distributes the computation across different GPUs based on the device_map.
    #     """
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     # First, move input embeddings to the assigned device
    #     if self.device_map is not None:
    #         input_ids = input_ids.to(self.embed.weight.device)
    #     x = self.embed(input_ids)
    #     if isinstance(self.norm, nn.Identity):
    #         x = x
    #     else:
    #         if self.device_map is not None:
    #             x = x.to(self.norm.weight.device)
    #         x = self.norm(x)
    #
    #     # Forward pass through layers with device-specific assignment
    #     if self.device_map is not None:
    #         device_id = self.device_map.get("layers")[0]
    #         x = x.to(device_id)  # Move data to the appropriate device for this layer
    #     for idx, layer in enumerate(self.layers):
    #         x = layer(x)
    #
    #     # Compute output logits
    #     if self.device_map is not None:
    #         device_id = self.device_map["output_head"]
    #         x = x.to(device_id)
    #     logits = self.output_head(x.to(self.output_head.weight.dtype))
    #     # print(logits.shape)
    #     # print(shift_logits.view(-1, shift_logits.size(-1))[:10])
    #     # print(shift_labels.view(-1)[:10])
    #
    #     loss = None
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         shift_labels = shift_labels.to(logits.device)
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #
    #     if not return_dict:
    #         return (logits,) + (loss,) if loss is not None else (logits,)
    #
    #     return CausalLMOutputWithCrossAttentions(
    #         loss=loss,
    #         logits=logits,
    #     )

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if beam search or beam sampling is used.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        # padding = (kernel_size - 1) 只在左边 padding
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        self.causal_padding = kernel_size - 1

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.causal_padding, 0))  # 只在左边 padding
        return super().forward(x)

class SharedFilter(nn.Module):
    def __init__(self, config, task_token_id_to_index):
        super().__init__()
        self.task_token_id_to_index = task_token_id_to_index  # e.g. {50260: 0, 50261: 1}
        self.num_tasks = len(set(task_token_id_to_index.values()))

        self.dim = config.dim

        self.task_embeddings = nn.Embedding(self.num_tasks + 1, config.dim)  # [N_task, D]
        self.shared_filter = CausalConv1d(config.dim, config.dim, kernel_size=3)

    def extract_task_id(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        提取每个 batch 中第一个匹配的 task token，默认返回 0。
        返回: [B] 的任务索引 tensor
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        default_task_index = 0 # len(self.task_token_id_to_index)  # 假设已有3个任务，默认用 index=3
        task_ids = torch.full((batch_size,), default_task_index, dtype=torch.long, device=device)

        for task_token_id, task_index in self.task_token_id_to_index.items():
            mask = (input_ids == task_token_id)
            matched = mask.any(dim=1)
            task_ids[matched] = task_index  # 将匹配到的 sample 设置为该任务
        return task_ids  # [B]

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor):
        """
        hidden_states: [B, T, D]
        input_ids: [B, T]
        """
        task_ids = self.extract_task_id(input_ids)  # [B]
        task_embeds = self.task_embeddings(task_ids)  # [B, D]
        gate = torch.sigmoid(task_embeds).unsqueeze(2)  # [B, D, 1]

        x = hidden_states.transpose(1, 2)  # [B, D, T]
        x = self.shared_filter(x)         # [B, D, T]
        x = x * gate                      # [B, D, T]
        hidden_states = x.transpose(1, 2) # [B, T, D]
        # hidden_states = hidden_states * gate

        return hidden_states, input_ids


class FilterV0(nn.Module):
    def __init__(self, config, task_token_id_to_index):
        super().__init__()
        self.task_token_id_to_index = task_token_id_to_index  # e.g. {50260: 0, 50261: 1}
        self.num_tasks = len(set(task_token_id_to_index.values()))

        self.dim = config.dim

        self.task_embeddings = nn.Embedding(self.num_tasks + 1, config.dim)  # [N_task, D]

    def extract_task_id(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        提取每个 batch 中第一个匹配的 task token，默认返回 0。
        返回: [B] 的任务索引 tensor
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        default_task_index = 0 # len(self.task_token_id_to_index)  # 假设已有3个任务，默认用 index=3
        task_ids = torch.full((batch_size,), default_task_index, dtype=torch.long, device=device)

        for task_token_id, task_index in self.task_token_id_to_index.items():
            mask = (input_ids == task_token_id)
            matched = mask.any(dim=1)
            task_ids[matched] = task_index  # 将匹配到的 sample 设置为该任务
        return task_ids  # [B]

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor):
        """
        hidden_states: [B, T, D]
        input_ids: [B, T]
        """
        task_ids = self.extract_task_id(input_ids)  # [B]
        # print(task_ids)
        task_embeds = self.task_embeddings(task_ids)  # [B, D]
        gate = torch.sigmoid(task_embeds).unsqueeze(1)  # [B, 1, D]
        hidden_states = hidden_states * (gate + 1.0)
        return hidden_states, input_ids

class FilterV1(nn.Module):
    def __init__(self, config, task_token_id_to_index, expansion_factor=2):
        super().__init__()
        self.task_token_id_to_index = task_token_id_to_index
        self.num_tasks = len(set(task_token_id_to_index.values()))
        self.dim = config.dim
        self.expanded_dim = self.dim * expansion_factor

        # Task embedding
        self.task_embeddings = nn.Embedding(self.num_tasks + 1, self.dim)  # [N_task, D]

        # 扩展维度后调制结构
        self.linear1 = nn.Linear(self.dim, self.expanded_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(self.expanded_dim, self.dim)

    def extract_task_id(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        batch_size = input_ids.size(0)
        device = input_ids.device
        default_task_index = 0
        task_ids = torch.full((batch_size,), default_task_index, dtype=torch.long, device=device)
        for task_token_id, task_index in self.task_token_id_to_index.items():
            mask = (input_ids == task_token_id)
            matched = mask.any(dim=1)
            task_ids[matched] = task_index
        return task_ids  # [B]

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor):
        """
        hidden_states: [B, T, D]
        input_ids: [B, T]
        """
        task_ids = self.extract_task_id(input_ids)  # [B]
        # logger.info(f"task id: {task_ids}")
        task_embeds = self.task_embeddings(task_ids)  # [B, D]
        task_expanded = task_embeds.unsqueeze(1).expand_as(hidden_states)  # [B, T, D]

        # 融合任务信息后通过 bottleneck 通道调制
        modulated = hidden_states + task_expanded  # [B, T, D]
        x = self.linear1(modulated)
        x = self.activation(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)

        # 残差融合输出
        hidden_states = hidden_states + x  # or (1 + x) * hidden_states
        return hidden_states, input_ids

class Filter(nn.Module):
    def __init__(self, config, task_token_id_to_index, expansion_factor=2, dropout=0.1):
        super().__init__()
        self.task_token_id_to_index = task_token_id_to_index
        self.num_tasks = len(set(task_token_id_to_index.values()))
        self.dim = config.dim
        self.hidden_dim = self.dim * expansion_factor

        # Task embedding: 每个任务一个向量
        self.task_embeddings = nn.Embedding(self.num_tasks + 1, self.hidden_dim)  # [N_task, D']

        # 升维 → gating（通过 task）→ 降维
        self.input_proj = nn.Linear(self.dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def extract_task_id(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        提取每个 batch 中第一个匹配的 task token，默认返回 0。
        返回: [B] 的任务索引 tensor
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        default_task_index = 0
        task_ids = torch.full((batch_size,), default_task_index, dtype=torch.long, device=device)

        for task_token_id, task_index in self.task_token_id_to_index.items():
            mask = (input_ids == task_token_id)  # [B, T]
            matched = mask.any(dim=1)
            task_ids[matched] = task_index
        return task_ids  # [B]

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor):
        """
        hidden_states: [B, T, D]
        input_ids:     [B, T]
        """
        hidden_states = hidden_states.to(self.input_proj.weight.dtype)
        B, T, D = hidden_states.size()

        task_ids = self.extract_task_id(input_ids)  # [B]
        task_gates = self.task_embeddings(task_ids)  # [B, D_expand]
        task_gates = torch.sigmoid(task_gates).unsqueeze(1).expand(B, T, self.hidden_dim)  # [B, T, D_expand]

        x = self.input_proj(hidden_states)  # [B, T, D_expand]
        x = x * task_gates  # Gating
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_proj(x)  # [B, T, D]

        return hidden_states + x, input_ids  # 残差连接


class TaskAdapter(nn.Module):
    def __init__(self, config, task_token_id_to_index, reduction_ratio=4):
        super().__init__()
        self.task_token_id_to_index = task_token_id_to_index
        self.num_tasks = len(set(task_token_id_to_index.values()))
        self.dim = config.dim
        self.task_embeddings = nn.Embedding(self.num_tasks + 1, self.dim)  # [N_task, D]

        # TaskAdapter 样式结构
        hidden_dim = self.dim // reduction_ratio
        self.linear1 = nn.Linear(self.dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, self.dim)

    def extract_task_id(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        batch_size = input_ids.size(0)
        device = input_ids.device
        default_task_index = 0
        task_ids = torch.full((batch_size,), default_task_index, dtype=torch.long, device=device)

        for task_token_id, task_index in self.task_token_id_to_index.items():
            mask = (input_ids == task_token_id)
            matched = mask.any(dim=1)
            task_ids[matched] = task_index
        return task_ids  # [B]

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.LongTensor):
        """
        hidden_states: [B, T, D]
        input_ids: [B, T]
        """
        task_ids = self.extract_task_id(input_ids)  # [B]
        task_embeds = self.task_embeddings(task_ids)  # [B, D]

        # Broadcast: [B, D] → [B, T, D]
        task_expanded = task_embeds.unsqueeze(1).expand_as(hidden_states)

        # Task-aware Adapter 调制
        adapter_input = hidden_states + task_expanded
        x = self.linear1(adapter_input)
        x = self.activation(x)
        x = self.linear2(x)

        # 残差连接输出
        hidden_states = hidden_states + x

        return hidden_states, input_ids


class TaskAwareFilter(nn.Module):
    def __init__(self, config, task_token_id_to_name):
        super().__init__()
        self.task_token_id_to_name = task_token_id_to_name
        self.filters = nn.ModuleDict({
            name: nn.Conv1d(
                in_channels=config.dim,
                out_channels=config.dim,
                kernel_size=3,
                padding=1
            ) for name in task_token_id_to_name.values()
        })

    def forward(self, hidden_states, input_ids):
        batch_filter = None
        for task_id, task_name in self.task_token_id_to_name.items():
            if isinstance(input_ids, torch.Tensor) and (input_ids == task_id).any().item():
                batch_filter = self.filters[task_name]
                break

        # 所有 sample 都要走 forward，只是有些不做事
        if batch_filter is not None:
            x = hidden_states.transpose(1, 2)
            x = batch_filter(x)
            hidden_states = x.transpose(1, 2)
        else:
            # no-op path: simulate same compute path to avoid pipeline desync
            hidden_states = hidden_states + 0

        return hidden_states, input_ids


class MGenoTasked(PreTrainedModel):
    """
    MGenoTasked model implementation based on the transformer pre-trained model interface.

    Args:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int, optional): Number of experts. Defaults to 8.
        num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
        pre_emb_norm (bool, optional): Whether to normalize the embeddings. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

    Attributes:
        layers (nn.ModuleList): List of MGeno layers.
        embed (nn.Embedding): Embedding layer.
        norm (nn.LayerNorm or nn.Identity): Normalization layer.
        model_parallel (bool): Whether the model is in parallel mode.
        device_map (dict): Device map for parallelization.
    """

    config_class = MGenoConfig
    base_model_prefix = "geno"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
            self,
            config: MGenoConfig,
            task_token_id_to_name=None,
            *args,
            **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.dim = config.dim
        self.depth = config.depth
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.heads = config.heads
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_token
        self.pre_emb_norm = config.pre_emb_norm
        self.return_embeddings = config.return_embeddings
        self.num_tokens = config.num_tokens
        self.expand = config.expand
        self.weights = config.weights
        self.max_token_len = config.max_token_len
        self.task_token_id_to_name = task_token_id_to_name or {}

        # Layers
        self.layers = nn.ModuleList(
            [
                MGenoBlock(
                    idx=idx,
                    dim=self.dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    heads=self.heads,
                    num_experts=self.num_experts,
                    num_experts_per_token=self.num_experts_per_tok,
                    expand=self.expand,
                    vocab_size=config.num_tokens,  # 的词汇表大小
                    n_positions=config.max_token_len,  # 的最大序列长度
                    n_ctx=config.max_token_len,  # 上下文窗口大小
                    attn_class=config.attn_class,  # , "multi_query_attention",
                    device=None,
                    config=config
                )
                for idx in range(self.depth)
            ]
        )

        # Pre Emb
        self.embed = nn.Embedding(self.num_tokens, self.dim)

        # Embedding Norm
        self.norm = (
            nn.LayerNorm(self.dim) if self.pre_emb_norm else nn.Identity()
        )

        # self.task_filter = SharedFilter(config, self.task_token_id_to_name)
        self.task_filter = Filter(config, self.task_token_id_to_name)

        # output
        self.output_head = nn.Linear(self.dim, self.num_tokens, bias=False)

        self.model_parallel = False
        self.device_map = None
        # self.parallelize(self.depth)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parallelize(self, rank, device_num=4):
        """
        Enable model parallelism by distributing layers across available GPUs.

        Args:
            device_num (int): Number of available GPUs.
        """
        self.model_parallel = True
        self.device_map = {}

        # Calculate which GPU each layer should be assigned to
        # layers_per_device = self.depth // device_num
        devices = []
        for i in range(device_num):
            device_id = (rank + i) % device_num
            devices.append(torch.device("cuda", device_id))
        self.device_map["layers"] = devices
        # Assign embedding, norm, and output_head to appropriate devices
        self.device_map["embed"] = devices[-1]
        self.device_map["norm"] = devices[-1]
        self.device_map["output_head"] = devices[-1]  # torch.device("cuda", (rank + 1) % device_num)
        print(self.device_map)

        self.layers = nn.ModuleList(
            [
                JambaBlockV2(
                    idx=idx,
                    dim=self.dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    heads=self.heads,
                    num_experts=self.num_experts,
                    num_experts_per_token=self.num_experts_per_tok,
                    expand=self.expand,
                    vocab_size=self.config.num_tokens,  # 的词汇表大小
                    n_positions=self.config.max_token_len,  # 的最大序列长度
                    n_ctx=self.config.max_token_len,  # 上下文窗口大小
                    config=self.config,
                    devices=self.device_map.get(f"layers")
                )
                for idx in range(self.depth)
            ]
        )

        # Apply model parallelism to each layer and module
        for idx, layer in enumerate(self.layers):
            layer.parallel()

        for name, module in self.named_modules():
            if name in ["embed", "norm"]:
                module.to(self.device_map[name])
            elif name == "output_head":
                module.to(self.device_map[name])
                if self.weights is not None:
                    self.weights = self.weights.to(self.device_map[name])

    def deparallelize(self):
        """Disable model parallelism."""
        self.model_parallel = False
        for name, module in self.named_modules():
            module.to("cpu")

    def get_output_embeddings(self):
        return self.output_head  # 修改为 output_head

    def set_output_embeddings(self, new_embeddings):
        self.output_head = new_embeddings  # 修改为 output_head

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Implement generation preparation logic
        token_type_ids = kwargs.get("token_type_ids", None)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "position_ids": position_ids, "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids}
        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # First, move input embeddings to the assigned device
        if self.device_map is not None:
            input_ids = input_ids.to(self.embed.weight.device)
        x = self.embed(input_ids)
        # 检查并确保嵌入层输出连续
        if not x.is_contiguous():
            x = x.contiguous()
        if isinstance(self.norm, nn.Identity):
            pass
        else:
            if self.device_map is not None:
                x = x.to(self.norm.weight.device)
                # 跨设备移动后检查连续性
                if not x.is_contiguous():
                    x = x.contiguous()
            x = self.norm(x)
        # Forward pass through layers with device-specific assignment
        if self.device_map is not None:
            device_id = self.device_map.get("layers")[0]
            x = x.to(device_id)
            # 跨设备移动后检查连续性
            if not x.is_contiguous():
                x = x.contiguous()
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # 检查每层输出的连续性
            if not x.is_contiguous():
                x = x.contiguous()
        # Compute output logits
        if self.device_map is not None:
            device_id = self.device_map["output_head"]
            x = x.to(device_id)
            # 跨设备移动后检查连续性
            if not x.is_contiguous():
                x = x.contiguous()
        # logger.info(f"{x.shape}")
        # hidden_states = x
        # last_hidden_states = x[:, -1, :]
        # logger.info(f"{last_hidden_states.shape}")

        if input_ids is not None and self.task_token_id_to_name:
            x = self.task_filter(x, input_ids)
        if isinstance(x, tuple):
            hidden_states = x[0]
            logits = self.output_head(x[0].to(self.output_head.weight.dtype))
        else:
            hidden_states = x
            logits = self.output_head(x.to(self.output_head.weight.dtype))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :]
            # 切片操作后检查连续性
            if not shift_logits.is_contiguous():
                shift_logits = shift_logits.contiguous()
            shift_labels = labels[..., 1:]
            # 切片操作后检查连续性
            if not shift_labels.is_contiguous():
                shift_labels = shift_labels.contiguous()
            shift_labels = shift_labels.to(logits.device)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            return (logits,) + (loss,) if loss is not None else (logits,)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    # def forward(
    #         self,
    #         input_ids: Optional[torch.LongTensor] = None,
    #         labels: Optional[torch.LongTensor] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
    #     """
    #     Forward pass logic with model parallelism support.
    #     Distributes the computation across different GPUs based on the device_map.
    #     """
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     # First, move input embeddings to the assigned device
    #     if self.device_map is not None:
    #         input_ids = input_ids.to(self.embed.weight.device)
    #     x = self.embed(input_ids)
    #     if isinstance(self.norm, nn.Identity):
    #         x = x
    #     else:
    #         if self.device_map is not None:
    #             x = x.to(self.norm.weight.device)
    #         x = self.norm(x)
    #
    #     # Forward pass through layers with device-specific assignment
    #     if self.device_map is not None:
    #         device_id = self.device_map.get("layers")[0]
    #         x = x.to(device_id)  # Move data to the appropriate device for this layer
    #     for idx, layer in enumerate(self.layers):
    #         x = layer(x)
    #
    #     # Compute output logits
    #     if self.device_map is not None:
    #         device_id = self.device_map["output_head"]
    #         x = x.to(device_id)
    #     logits = self.output_head(x.to(self.output_head.weight.dtype))
    #     # print(logits.shape)
    #     # print(shift_logits.view(-1, shift_logits.size(-1))[:10])
    #     # print(shift_labels.view(-1)[:10])
    #
    #     loss = None
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         shift_labels = shift_labels.to(logits.device)
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #
    #     if not return_dict:
    #         return (logits,) + (loss,) if loss is not None else (logits,)
    #
    #     return CausalLMOutputWithCrossAttentions(
    #         loss=loss,
    #         logits=logits,
    #     )

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if beam search or beam sampling is used.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )


class MGenoTaskedV2(PreTrainedModel):
    """
    MGenoTasked model implementation based on the transformer pre-trained model interface.

    Args:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int, optional): Number of experts. Defaults to 8.
        num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
        pre_emb_norm (bool, optional): Whether to normalize the embeddings. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

    Attributes:
        layers (nn.ModuleList): List of MGeno layers.
        embed (nn.Embedding): Embedding layer.
        norm (nn.LayerNorm or nn.Identity): Normalization layer.
        model_parallel (bool): Whether the model is in parallel mode.
        device_map (dict): Device map for parallelization.
    """

    config_class = MGenoConfig
    base_model_prefix = "geno"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
            self,
            config: MGenoConfig,
            task_token_id_to_name=None,
            *args,
            **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.dim = config.dim
        self.depth = config.depth
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.heads = config.heads
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_token
        self.pre_emb_norm = config.pre_emb_norm
        self.return_embeddings = config.return_embeddings
        self.num_tokens = config.num_tokens
        self.expand = config.expand
        self.weights = config.weights
        self.max_token_len = config.max_token_len
        self.task_token_id_to_name = task_token_id_to_name or {}

        # Layers
        self.layers = nn.ModuleList(
            [
                MGenoBlock(
                    idx=idx,
                    dim=self.dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    heads=self.heads,
                    num_experts=self.num_experts,
                    num_experts_per_token=self.num_experts_per_tok,
                    expand=self.expand,
                    vocab_size=config.num_tokens,  # 的词汇表大小
                    n_positions=config.max_token_len,  # 的最大序列长度
                    n_ctx=config.max_token_len,  # 上下文窗口大小
                    attn_class=config.attn_class,  # , "multi_query_attention",
                    device=None,
                    config=config
                )
                for idx in range(self.depth)
            ]
        )

        # Pre Emb
        self.embed = nn.Embedding(self.num_tokens, self.dim)

        # Embedding Norm
        self.norm = (
            nn.LayerNorm(self.dim) if self.pre_emb_norm else nn.Identity()
        )

        # self.task_filter = SharedFilter(config, self.task_token_id_to_name)
        self.task_filter = TaskAdapter(config, self.task_token_id_to_name)

        # output
        self.output_head = nn.Linear(self.dim, self.num_tokens, bias=False)

        self.model_parallel = False
        self.device_map = None
        # self.parallelize(self.depth)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parallelize(self, rank, device_num=4):
        """
        Enable model parallelism by distributing layers across available GPUs.

        Args:
            device_num (int): Number of available GPUs.
        """
        self.model_parallel = True
        self.device_map = {}

        # Calculate which GPU each layer should be assigned to
        # layers_per_device = self.depth // device_num
        devices = []
        for i in range(device_num):
            device_id = (rank + i) % device_num
            devices.append(torch.device("cuda", device_id))
        self.device_map["layers"] = devices
        # Assign embedding, norm, and output_head to appropriate devices
        self.device_map["embed"] = devices[-1]
        self.device_map["norm"] = devices[-1]
        self.device_map["output_head"] = devices[-1]  # torch.device("cuda", (rank + 1) % device_num)
        print(self.device_map)

        self.layers = nn.ModuleList(
            [
                JambaBlockV2(
                    idx=idx,
                    dim=self.dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    heads=self.heads,
                    num_experts=self.num_experts,
                    num_experts_per_token=self.num_experts_per_tok,
                    expand=self.expand,
                    vocab_size=self.config.num_tokens,  # 的词汇表大小
                    n_positions=self.config.max_token_len,  # 的最大序列长度
                    n_ctx=self.config.max_token_len,  # 上下文窗口大小
                    config=self.config,
                    devices=self.device_map.get(f"layers")
                )
                for idx in range(self.depth)
            ]
        )

        # Apply model parallelism to each layer and module
        for idx, layer in enumerate(self.layers):
            layer.parallel()

        for name, module in self.named_modules():
            if name in ["embed", "norm"]:
                module.to(self.device_map[name])
            elif name == "output_head":
                module.to(self.device_map[name])
                if self.weights is not None:
                    self.weights = self.weights.to(self.device_map[name])

    def deparallelize(self):
        """Disable model parallelism."""
        self.model_parallel = False
        for name, module in self.named_modules():
            module.to("cpu")

    def get_output_embeddings(self):
        return self.output_head  # 修改为 output_head

    def set_output_embeddings(self, new_embeddings):
        self.output_head = new_embeddings  # 修改为 output_head

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Implement generation preparation logic
        token_type_ids = kwargs.get("token_type_ids", None)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "position_ids": position_ids, "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids}
        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # First, move input embeddings to the assigned device
        if self.device_map is not None:
            input_ids = input_ids.to(self.embed.weight.device)
        x = self.embed(input_ids)
        # 检查并确保嵌入层输出连续
        if not x.is_contiguous():
            x = x.contiguous()
        if isinstance(self.norm, nn.Identity):
            pass
        else:
            if self.device_map is not None:
                x = x.to(self.norm.weight.device)
                # 跨设备移动后检查连续性
                if not x.is_contiguous():
                    x = x.contiguous()
            x = self.norm(x)
        # Forward pass through layers with device-specific assignment
        if self.device_map is not None:
            device_id = self.device_map.get("layers")[0]
            x = x.to(device_id)
            # 跨设备移动后检查连续性
            if not x.is_contiguous():
                x = x.contiguous()
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # 检查每层输出的连续性
            if not x.is_contiguous():
                x = x.contiguous()
        # Compute output logits
        if self.device_map is not None:
            device_id = self.device_map["output_head"]
            x = x.to(device_id)
            # 跨设备移动后检查连续性
            if not x.is_contiguous():
                x = x.contiguous()
        # logger.info(f"{x.shape}")
        hidden_states = x
        # last_hidden_states = x[:, -1, :]
        # logger.info(f"{last_hidden_states.shape}")

        if input_ids is not None and self.task_token_id_to_name:
            x = self.task_filter(x, input_ids)
        if isinstance(x, tuple):
            logits = self.output_head(x[0].to(self.output_head.weight.dtype))
        else:
            logits = self.output_head(x.to(self.output_head.weight.dtype))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            shift_logits = logits[..., :-1, :]
            # 切片操作后检查连续性
            if not shift_logits.is_contiguous():
                shift_logits = shift_logits.contiguous()
            shift_labels = labels[..., 1:]
            # 切片操作后检查连续性
            if not shift_labels.is_contiguous():
                shift_labels = shift_labels.contiguous()
            shift_labels = shift_labels.to(logits.device)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            return (logits,) + (loss,) if loss is not None else (logits,)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    # def forward(
    #         self,
    #         input_ids: Optional[torch.LongTensor] = None,
    #         labels: Optional[torch.LongTensor] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
    #     """
    #     Forward pass logic with model parallelism support.
    #     Distributes the computation across different GPUs based on the device_map.
    #     """
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     # First, move input embeddings to the assigned device
    #     if self.device_map is not None:
    #         input_ids = input_ids.to(self.embed.weight.device)
    #     x = self.embed(input_ids)
    #     if isinstance(self.norm, nn.Identity):
    #         x = x
    #     else:
    #         if self.device_map is not None:
    #             x = x.to(self.norm.weight.device)
    #         x = self.norm(x)
    #
    #     # Forward pass through layers with device-specific assignment
    #     if self.device_map is not None:
    #         device_id = self.device_map.get("layers")[0]
    #         x = x.to(device_id)  # Move data to the appropriate device for this layer
    #     for idx, layer in enumerate(self.layers):
    #         x = layer(x)
    #
    #     # Compute output logits
    #     if self.device_map is not None:
    #         device_id = self.device_map["output_head"]
    #         x = x.to(device_id)
    #     logits = self.output_head(x.to(self.output_head.weight.dtype))
    #     # print(logits.shape)
    #     # print(shift_logits.view(-1, shift_logits.size(-1))[:10])
    #     # print(shift_labels.view(-1)[:10])
    #
    #     loss = None
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         shift_labels = shift_labels.to(logits.device)
    #         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #
    #     if not return_dict:
    #         return (logits,) + (loss,) if loss is not None else (logits,)
    #
    #     return CausalLMOutputWithCrossAttentions(
    #         loss=loss,
    #         logits=logits,
    #     )

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if beam search or beam sampling is used.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )


class MGenoMTP(PreTrainedModel):
    """
    MGenoMTP model implementation based on the transformer pre-trained model interface.

    Args:
        dim (int): Dimension of the model.
        depth (int): Depth of the model.
        num_tokens (int): Number of tokens.
        max_seq_len (int): Maximum sequence length.
        d_state (int): State dimension.
        d_conv (int): Convolutional dimension.
        heads (int): Number of attention heads.
        num_experts (int, optional): Number of experts. Defaults to 8.
        num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
        pre_emb_norm (bool, optional): Whether to normalize the embeddings. Defaults to False.
        return_embeddings (bool, optional): Whether to return the embeddings. Defaults to False.

    Attributes:
        layers (nn.ModuleList): List of MGenoMTP layers.
        embed (nn.Embedding): Embedding layer.
        norm (nn.LayerNorm or nn.Identity): Normalization layer.
        model_parallel (bool): Whether the model is in parallel mode.
        device_map (dict): Device map for parallelization.
    """

    config_class = MGenoConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _no_split_modules = ["embedding", "output_head", "norm"]

    def __init__(
            self,
            config: MGenoConfig,
            *args,
            **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.dim = config.dim
        self.depth = config.depth
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.heads = config.heads
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_token
        self.pre_emb_norm = config.pre_emb_norm
        self.return_embeddings = config.return_embeddings
        self.num_tokens = config.num_tokens
        self.expand = config.expand
        self.weights = config.weights
        self.max_token_len = config.max_token_len
        self.n_future_tokens = config.n_future_tokens  # 多token
        self.attn_class = config.attn_class
        self.return_all_heads = config.return_all_heads

        # Layers

        self.layers = nn.ModuleList(
            [
                MGenoBlock(
                    idx=idx,
                    dim=self.dim,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    heads=self.heads,
                    num_experts=self.num_experts,
                    num_experts_per_token=self.num_experts_per_tok,
                    expand=self.expand,
                    vocab_size=self.num_tokens,  # 的词汇表大小
                    n_positions=self.max_token_len,  # 的最大序列长度
                    n_ctx=self.max_token_len,  # 上下文窗口大小
                    attn_class=self.attn_class,  # , "multi_query_attention",
                    device=None,
                    config=self.config
                )
                for idx in range(self.depth)
            ]
        )

        # Pre Emb
        self.embed = nn.Embedding(self.num_tokens, self.dim)

        # Embedding Norm
        self.norm = (
            nn.LayerNorm(self.dim) if self.pre_emb_norm else nn.Identity()
        )

        gpt_config = GPT2Config(n_embd=self.dim,
                                n_head=self.heads,
                                vocab_size=self.num_tokens,  # 的词汇表大小
                                n_positions=self.max_token_len,  # 的最大序列长度
                                n_ctx=self.max_token_len,  # 上下文窗口大小
                                _attn_implementation=self.attn_class
                                )
        self.multi_heads = nn.ModuleList(
            [
                GenoTransformerBlock(gpt_config, layer_idx=idx)
                for idx in range(self.depth, self.depth + self.n_future_tokens)
            ]
        )

        # output
        self.output_head = nn.Linear(self.dim, self.num_tokens, bias=False)

        self.model_parallel = False
        self.device_map = None
        # self.parallelize(self.depth)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parallelize(self, device_num=4):
        """
        Enable model parallelism by distributing layers across available GPUs.

        Args:
            device_num (int): Number of available GPUs.
        """
        self.model_parallel = True
        self.device_map = {}

        for idx, layer in enumerate(self.layers):
            device = torch.device(f"cuda:{idx % device_num}") if torch.cuda.is_available() else torch.device("cpu")
            self.device_map[f"layer{idx}"] = device
            layer.to(device)

        for idx, head_layer in enumerate(self.multi_heads):
            device = torch.device(f"cuda:{idx % device_num}") if torch.cuda.is_available() else torch.device("cpu")
            self.device_map[f"head_layer{idx}"] = device
            head_layer.to(device)

        for name, module in self.named_modules():
            idx = 0
            if name in ["embed", "norm", "output_head"]:
                device = torch.device(f"cuda:{idx % device_num}") if torch.cuda.is_available() else torch.device("cpu")
                module.to(device)
                idx += 1

    def deparallelize(self):
        """Disable model parallelism."""
        self.model_parallel = False
        for name, module in self.named_modules():
            module.to("cpu")

    def get_output_embeddings(self):
        return self.output_head  # 修改为 output_head

    def set_output_embeddings(self, new_embeddings):
        self.output_head = new_embeddings  # 修改为 output_head

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # Implement generation preparation logic
        token_type_ids = kwargs.get("token_type_ids", None)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1]:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "position_ids": position_ids, "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids}
        return model_inputs

    @staticmethod
    def loss_func(logits, labels):
        def generate_normalized_list(length, last_value=64):
            # 初始化一个长度为 length 的列表，初始值都设为 0
            result = [0] * length
            # 给列表的最后一个元素赋值为 last_value
            result[0] = last_value
            # 从倒数第二个元素开始往前遍历列表
            for i in range(0, length - 1):
                # 每个元素的值是其下一个元素的一半
                result[i + 1] = result[i] // 2
            # 计算列表元素的总和
            total = sum(result)
            # 调整列表中的每个元素，使得元素总和为 1
            return [i / total for i in result]

        weights = generate_normalized_list(logits.size(-2))
        # logger.info(f"weights: {weights}")
        loss_fct = CrossEntropyLoss()
        total_loss = 0
        for i in range(logits.size(-2)):
            shift_logits = logits[..., :-1 - i, i, :].contiguous()
            # logger.info(f"shift_logits: {shift_logits.shape}")
            shift_labels = labels[..., i + 1:].contiguous()
            # logger.info(f"shift_labels: {shift_labels.shape}")
            total_loss += loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  # weights[i] *

        loss = total_loss
        return loss

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        # First, move input embeddings to the assigned device
        input_ids = input_ids.to(self.embed.weight.device)
        x = self.embed(input_ids)

        if isinstance(self.norm, nn.Identity):
            pass
        else:
            x = x.to(self.norm.weight.device)
            x = self.norm(x)
        # Forward pass through layers with device-specific assignment

        for idx, layer in enumerate(self.layers):
            if self.device_map is not None:
                x = x.to(self.device_map[f"layer{idx}"])
            x = layer(x)

        h_trunk = x

        # Prediction heads
        latents = []
        n_heads_to_use = self.n_future_tokens
        prediction_heads = list(self.multi_heads)
        for idx, layer in enumerate(prediction_heads[:n_heads_to_use]):
            if self.device_map is not None:
                h_trunk = h_trunk.to(self.device_map[f"head_layer{idx}"])
            h = layer(h_trunk)
            h = h.to(self.output_head.weight.device)
            latents.append(h)
        h = torch.stack(latents, dim=-2)  # (batch_size, seq_len, n_heads_to_use, dim)

        embeddings = torch.cat(latents, dim=-1)  # (batch_size, seq_len, n_heads_to_use * dim)

        h = h.to(self.output_head.weight.device)
        logits = self.output_head(
            h.to(self.output_head.weight.dtype))  # (batch_size, seq_len, n_heads_to_use, num_tokens)

        if labels is not None:
            loss = loss_func(logits, labels)
        else:
            loss = None
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=embeddings,
        )

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if beam search or beam sampling is used.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
