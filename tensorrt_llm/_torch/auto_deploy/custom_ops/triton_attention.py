"""Custom ops for MHA/XQA attention."""

import math
from typing import List, Optional, Tuple

import torch
import triton
from torch.fx import Node

from ..utils.logger import ad_logger
from .attention_interface import (
    AttentionRegistry,
    Constant,
    MHACallable,
    PrepareMetadataCallable,
    SequenceInfo,
)
from .torch_backend_attention import TorchBackendAttention
from .triton_kernels.attention_with_kv_cache import (
    attention_kv_stage2,
    context_attention_kv_flattened,
    gqa_attention_kv_stage1,
    update_kv_cache,
)


def _generate_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_locs: torch.Tensor,
    input_pos: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
):
    b, (n_heads, q_d_head) = q.shape[0], q.shape[-2:]
    max_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]
    device = q.device

    SEQ_BLOCK_SIZE = 64
    num_blocks = (max_seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE
    stage1_output_values = torch.empty(
        b, n_heads, num_blocks, v_d_head, device=device, dtype=torch.float32
    )
    stage1_output_logsumexp = torch.empty(
        b, n_heads, num_blocks, device=device, dtype=torch.float32
    ) - float("inf")

    update_kv_cache[(b, n_kv_heads, 1)](
        k,
        v,
        None,
        None,
        k_cache,
        v_cache,
        input_pos,
        cache_locs,
        max_seq_len,
        n_kv_heads,
        q_d_head,
        v_d_head,
        1,
        GENERATE_ONLY=True,
    )

    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))
    gqa_attention_kv_stage1[
        (
            b,
            n_kv_heads,
            num_blocks,
        )
    ](
        q,
        k_cache,
        v_cache,
        cache_locs,
        input_pos,
        stage1_output_values,
        stage1_output_logsumexp,
        num_blocks,
        scale,
        max_seq_len,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK_SIZE,
        HEAD_BLOCK_SIZE,
        sliding_window if sliding_window is not None else -1,
    )
    has_sinks = sinks is not None

    attention_kv_stage2[(b, n_heads, 1)](
        stage1_output_values,
        stage1_output_logsumexp,
        out,
        input_pos,
        num_blocks,
        n_heads,
        v_d_head,
        SEQ_BLOCK_SIZE,
        has_sinks,
        sinks,
    )


def _flattened_context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
) -> None:
    # NOTE: s_total == sum(seq_len)
    s_total, n_heads, q_d_head = q.shape
    max_cache_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]
    BATCH_SIZE: int = len(input_pos)
    SEQ_BLOCK = 32

    update_kv_cache[(BATCH_SIZE, n_kv_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)](
        k,
        v,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        slot_idx,
        max_cache_seq_len,
        n_kv_heads,
        q_d_head,
        v_d_head,
        32,
        GENERATE_ONLY=False,
    )

    # TODO: use input_pos to get the correct cache locations
    grid = (BATCH_SIZE, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    has_sinks = sinks is not None

    context_attention_kv_flattened[grid](
        q,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        slot_idx,
        out,
        scale,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK,
        max_cache_seq_len,
        sliding_window if sliding_window is not None else -1,
        has_sinks,
        sinks,
    )


@torch.library.custom_op("auto_deploy::triton_attention_flattened_mha_with_cache", mutates_args=())
def flattened_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    seq_start: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    # <none>
    # CONSTANTS
    scale: Optional[float],
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Flattened MHA with cache that takes q, k, v in BSND layout.

    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.
    num_kv_heads, qk_head_dim = k_cache.shape[-2:]
    v_head_dim = v_cache.shape[-1]
    b, s = q.shape[:2]

    # check for num_heads
    num_heads = q.shape[2] // qk_head_dim if q.ndim == 3 else q.shape[2]

    # Define output shape
    output_shape = (b, s, num_heads * v_head_dim) if q.ndim == 3 else (b, s, num_heads, v_head_dim)

    # reshapes with head_dim
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)

    q = q.contiguous().view(*bs_view, num_heads, qk_head_dim)
    k = k.contiguous().view(*bs_view, num_kv_heads, qk_head_dim)
    v = v.contiguous().view(*bs_view, num_kv_heads, v_head_dim)

    scale = 1.0 / math.sqrt(qk_head_dim) if scale is None else scale
    # run attention
    y = q.new_empty(*bs_view, num_heads, v_head_dim).contiguous()
    if s == 1:
        # generate-only phase
        _generate_mha(
            q, k, v, k_cache, v_cache, slot_idx, input_pos, scale, y, sinks, sliding_window
        )
    else:
        # mixed context + generate phase
        _flattened_context_mha(
            q,
            k,
            v,
            input_pos,
            slot_idx,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            scale,
            y,
            sinks,
            sliding_window,
        )

    return y.view(*output_shape)


@flattened_mha_with_cache.register_fake
def flattened_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: Optional[float],
    sinks: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
):
    return q.new_empty(*q.shape[:-1], v.shape[-1]).contiguous()


@torch.library.custom_op(
    "auto_deploy::triton_attention_prepare_fused_mha_metadata", mutates_args=()
)
def prepare_fused_mha_metadata(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    pages_per_seq: torch.Tensor,
    slot_idx: torch.Tensor,
    page_size: int,
) -> List[torch.Tensor]:
    # TODO: maybe use slot_idx instead of pages_per_seq??
    num_seq = SequenceInfo._get_sanitized_num_sequences(input_ids, seq_len)
    seq_start = torch.zeros_like(seq_len[:num_seq])
    seq_start[1:] = torch.cumsum(seq_len[: num_seq - 1], 0)
    return (
        seq_len[:num_seq].clone(),
        input_pos[:num_seq].clone(),
        slot_idx[:num_seq].clone(),
        seq_start,
    )


# TODO: Move the truncation of inputs out of this custom op
# SequenceInfo._get_sanitized_num_sequences could break in fake mode
@prepare_fused_mha_metadata.register_fake
def prepare_fused_mha_metadata_fake(
    input_ids, position_ids, seq_len, input_pos, cache_loc, pages_per_seq, slot_idx, page_size
):
    num_seq = SequenceInfo._get_sanitized_num_sequences(input_ids, seq_len)
    return (
        torch.empty_like(seq_len[:num_seq]),
        torch.empty_like(input_pos[:num_seq]),
        torch.empty_like(slot_idx[:num_seq]),
        torch.empty_like(seq_len[:num_seq]),
    )


@AttentionRegistry.register("triton")
class TritonAttention(TorchBackendAttention):
    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        return torch.ops.auto_deploy.triton_attention_prepare_fused_mha_metadata, 4

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        scale, sinks, sliding_window, logit_cap = super().get_constants(source_attn_node)

        if logit_cap is not None:
            ad_logger.warning(
                f"Provided {logit_cap=} is not supported. Using default logit_cap instead."
            )
            logit_cap = None

        return [
            scale,
            sinks,
            sliding_window,
        ]
