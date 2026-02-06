"""Custom ops for MultiHead Latent attention."""

import math
from typing import List, Optional, Union

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from ....llmapi.llm_args import KvCacheConfig
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)
from .triton_attention import _decode_attention, _prefill_attention

Constant = Union[int, float, str, None]


def _precompute_inv_freq(
    max_seq_len: int, head_dim: int, rope_theta: float, device: torch.device
) -> torch.Tensor:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=inv_freq.device, dtype=inv_freq.dtype)

    freqs = torch.outer(t, inv_freq.to(t.device))
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_sin_stacked = torch.stack([emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)])
    return cos_sin_stacked


@torch.library.custom_op(
    "auto_deploy::triton_attention_fused_flattened_mla_with_cache", mutates_args=()
)
def fused_flattened_mla_with_cache(
    # Q, K, V
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # CONSTANTS
    softmax_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
) -> torch.Tensor:
    """Flattened & fused MLA with cache with triton kernels."""
    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.

    # check for sequence info and truncate metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    seq_len = seq_len[:num_seq]
    input_pos = input_pos[:num_seq]
    cache_loc = cache_loc[:num_seq]
    seq_start = cu_seqlen[:num_seq]

    # Get parameters
    b, num_heads, s, qk_nope_head_dim = q_nope.shape
    qk_rope_head_dim = q_pe.shape[-1]
    v_head_dim = kv.shape[-1] - qk_nope_head_dim

    # Get k_nope and value_states
    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    # Flatten inputs
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)

    # TODO(suyogg): do something about all these clones, transposes, and contiguous-es
    q_nope = q_nope.clone().transpose(1, 2).view(*bs_view, num_heads, qk_nope_head_dim).contiguous()
    q_pe = q_pe.clone().transpose(1, 2).view(*bs_view, num_heads, qk_rope_head_dim).contiguous()
    k_nope = k_nope.clone().transpose(1, 2).view(*bs_view, num_heads, qk_nope_head_dim).contiguous()
    k_pe = k_pe.clone().transpose(1, 2).view(*bs_view, -1, qk_rope_head_dim).contiguous()
    value_states = value_states.transpose(1, 2).view(*bs_view, -1, v_head_dim).contiguous()
    # Apply RoPE
    if rope_theta is not None:
        max_seq_len = (input_pos + seq_len).max().item()
        cos_sin_stacked = _precompute_inv_freq(
            max_seq_len, qk_rope_head_dim, rope_theta, q_pe.device
        )

        # Extract cos and sin from freqs_cis
        cos_base = cos_sin_stacked[0, ...]
        sin_base = cos_sin_stacked[1, ...]

        # TODO: Use triton kernels for RoPE
        # TODO: Add yarn support
        for i in range(seq_len.shape[0]):
            start = seq_start[i]
            length = seq_len[i]

            # build position_ids
            if s == 1:
                idx = (input_pos[i] + length - 1).item()
                pos_ids = torch.tensor(idx, device=cos_base.device)
            else:
                pos_ids = torch.arange(input_pos[i], input_pos[i] + length, device=cos_base.device)

            cos = cos_base[pos_ids]  # [..., 1, head_dim]
            sin = sin_base[pos_ids]
            q_slice = q_pe[start : start + length]
            k_slice = k_pe[start : start + length]

            q_rot, k_rot = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
                q_slice,
                k_slice,
                cos,
                sin,
                -2,
            )

            q_pe[start : start + length] = q_rot
            k_pe[start : start + length] = k_rot

    # Create query_states, key_states
    query_states = torch.cat((q_nope, q_pe), dim=-1)  # [b*s,n,d]
    key_states = torch.cat((k_nope, k_pe.expand(*bs_view, num_heads, -1)), dim=-1)  # [b*s,n,d]

    # Compute scale if not provided
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(qk_head_dim)

    # Compute attention
    y = torch.empty_like(value_states)
    if s == 1:
        # generate-only phase (decode)
        _decode_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            scale,
            y,
        )

    else:
        # mixed context + generate phase (prefill)
        _prefill_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            k_cache,
            v_cache,
            input_pos,
            cache_loc,
            seq_len,
            seq_start,
            scale,
            y,
        )

    y = (
        y.view(b, s, -1, v_head_dim).transpose(1, 2).contiguous()
    )  # BNSD format as expected by the callsite.
    return y


@fused_flattened_mla_with_cache.register_fake
def fused_flattened_mla_with_cache_fake(
    # Q, K, V
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # CONSTANTS
    softmax_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
):
    v_head_dim = kv.shape[-1] - q_nope.shape[-1]
    return torch.empty_like(kv[..., -v_head_dim:])


@AttentionRegistry.register("MultiHeadLatentAttention")
class MultiHeadLatentAttention(AttentionDescriptor):
    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bnsd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 4

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_attention_deepseek_fused_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_attention_fused_flattened_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "input_pos", "cache_loc", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        q_nope_fake = source_attn_node.args[0].meta["val"]
        q_pe_fake = source_attn_node.args[1].meta["val"]
        kv_fake = source_attn_node.args[2].meta["val"]

        num_kv_heads = kv_fake.shape[1]
        head_dim = q_nope_fake.shape[-1]
        rope_dim = q_pe_fake.shape[-1]

        return {
            "k_cache": UnpagedResourceHandler(
                num_kv_heads,
                head_dim + rope_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, kv_fake.dtype),
            ),
            "v_cache": UnpagedResourceHandler(
                num_kv_heads,
                head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, kv_fake.dtype),
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        softmax_scale = None
        rope_theta = 10000.0  # TODO: remove once MLA is unfused
        return [softmax_scale, rope_theta]
