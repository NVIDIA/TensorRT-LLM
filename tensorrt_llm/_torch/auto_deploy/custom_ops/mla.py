"""Custom ops for MultiHead Latent attention."""

from typing import List, Optional, Tuple, Union

import torch

from .attention_interface import (
    AttentionDescriptor,
    AttentionRegistry,
    BufferInitializerDict,
    CacheInitializerDict,
    MHACallable,
    PrepareMetadataCallable,
    SequenceInfo,
)
from .torch_attention import apply_rotary_pos_emb_ds
from .triton_attention import _flattened_context_mha, _generate_mha

Constant = Union[int, float, str, None]


@torch.library.custom_op("attention::fused_flattened_mla_with_cache", mutates_args=())
def fused_flattened_mla_with_cache(
    # Q, K, V
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    cos_sin_stacked: torch.Tensor,
    # CONSTANTS
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Flattened & fused MLA with cache with triton kernels."""
    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.

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
    if cos_sin_stacked.numel() > 0:
        # Extract cos and sin from freqs_cis
        cos = cos_sin_stacked[0, ...]
        sin = cos_sin_stacked[1, ...]

        # TODO: Use triton kernels for RoPE
        # TODO: Add yarn support
        for idx in range(seq_len.shape[0]):
            (
                q_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
                k_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
            ) = apply_rotary_pos_emb_ds(
                q_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
                k_pe[seq_start[idx] : seq_start[idx] + seq_len[idx], ...],
                cos,
                sin,
                torch.arange(input_pos[idx] + seq_len[idx])[-1]
                if s == 1
                else torch.arange(input_pos[idx] + seq_len[idx]),
                -2,
            )

    # Create query_states, key_states
    query_states = torch.cat((q_nope, q_pe), dim=-1)  # [b*s,n,d]
    key_states = torch.cat((k_nope, k_pe.expand(*bs_view, num_heads, -1)), dim=-1)  # [b*s,n,d]

    # Compute attention
    y = torch.empty_like(value_states)
    if s == 1:
        # generate-only phase
        _generate_mha(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            y,
        )

    else:
        # mixed context + generate phase
        _flattened_context_mha(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            input_pos,
            cache_loc,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
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
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    cos_sin_stacked: torch.Tensor,
    # CONSTANTS
    softmax_scale: Optional[float] = None,
):
    v_head_dim = kv.shape[-1] - q_nope.shape[-1]
    return torch.empty_like(kv[..., -v_head_dim:])


@torch.library.custom_op("attention::prepare_fused_mla_metadata", mutates_args=())
def prepare_fused_mla_metadata(
    input_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    page_size: int,
) -> List[torch.Tensor]:
    num_seq = SequenceInfo._get_sanitized_num_sequences(input_ids, seq_len)
    seq_start = torch.zeros_like(seq_len[:num_seq])
    seq_start[1:] = torch.cumsum(seq_len[: num_seq - 1], 0)
    return (
        seq_len[:num_seq].clone(),
        input_pos[:num_seq].clone(),
        cache_loc[:num_seq].clone(),
        seq_start,
    )


@prepare_fused_mla_metadata.register_fake
def prepare_fused_mla_metadata_fake(
    input_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
):
    return (
        torch.empty_like(seq_len),
        torch.empty_like(input_pos),
        torch.empty_like(cache_loc),
        torch.empty_like(seq_len),
    )


@AttentionRegistry.register("MultiHeadLatentAttention")
class MultiHeadLatentAttention(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""
        return False

    @classmethod
    def get_attention_op(cls) -> Tuple[MHACallable, int]:
        return torch.ops.attention.fused_flattened_mla_with_cache, 4

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        return torch.ops.attention.prepare_fused_mla_metadata, 4

    @classmethod
    def get_cache_initializers(cls, get_mla_info) -> CacheInitializerDict:
        attention_info = get_mla_info()

        def _get_k_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for MultiHeadLatentAttention"
            return torch.empty(
                si.num_pages,
                si.page_size,
                attention_info.num_kv_heads,
                attention_info.head_dim + attention_info.rope_dim,
                device=si.device,
                dtype=attention_info.cache_config.dtype or attention_info.dtype,
            )

        def _get_v_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for MultiHeadLatentAttention"
            return torch.empty(
                si.num_pages,
                si.page_size,
                attention_info.num_kv_heads,
                attention_info.head_dim,
                device=si.device,
                dtype=attention_info.cache_config.dtype or attention_info.dtype,
            )

        return {"k_cache": _get_k_cache, "v_cache": _get_v_cache}

    @classmethod
    def get_global_buffer_initializers(cls, get_mla_info) -> BufferInitializerDict:
        attention_info = get_mla_info()
        rope_head_dim = attention_info.rope_dim
        rope_theta = attention_info.pos_embd_config.rope_theta

        def _get_cos_sin_stacked(si: SequenceInfo):
            if rope_theta is None:
                return torch.empty(0, device=si.device)
            return cls._precompute_inv_freq(si.max_seq_len, rope_head_dim, rope_theta).to(si.device)

        return {
            f"cos_sin_stacked_{rope_head_dim}_{rope_theta}".replace(".", "_"): _get_cos_sin_stacked
        }

    @classmethod
    def get_constants(cls, get_mla_info) -> List[Constant]:
        return [None]

    @staticmethod
    def _precompute_inv_freq(seq_len: int, head_dim: int, rope_theta: float = 1e4) -> torch.Tensor:
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)

        freqs = torch.outer(t, inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_sin_stacked = torch.stack([emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)])
        return cos_sin_stacked
