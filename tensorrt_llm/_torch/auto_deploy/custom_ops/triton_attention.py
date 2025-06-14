"""Custom ops for MHA/XQA attention."""

import math
from typing import List, Optional, Tuple

import torch
import triton
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ..utils.logger import ad_logger
from ..utils.node_utils import extract_op_args
from .attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BufferInitializerDict,
    CacheConfig,
    CacheInitializerDict,
    Constant,
    MHACallable,
    PrepareMetadataCallable,
    SequenceInfo,
)
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
    )
    attention_kv_stage2[(b, n_heads, 1)](
        stage1_output_values,
        stage1_output_logsumexp,
        out,
        input_pos,
        num_blocks,
        n_heads,
        v_d_head,
        SEQ_BLOCK_SIZE,
    )


def _flattened_context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    scale: float,
    out: torch.Tensor,
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
        cache_loc,
        max_cache_seq_len,
        n_kv_heads,
        q_d_head,
        v_d_head,
        32,
        GENERATE_ONLY=False,
    )

    # TODO: use input_pos to get the correct cache locations
    grid = (BATCH_SIZE, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_flattened[grid](
        q,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        cache_loc,
        out,
        scale,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK,
        max_cache_seq_len,
        num_stages=2,
    )


@torch.library.custom_op("attention::flattened_mha_with_cache", mutates_args=())
def flattened_mha_with_cache(
    # Q, K, V
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # METADATA
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    # CACHES
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    # BUFFERS
    # <none>
    # CONSTANTS
    scale: Optional[float],
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
        _generate_mha(q, k, v, k_cache, v_cache, cache_loc, input_pos, scale, y)
    else:
        # mixed context + generate phase
        _flattened_context_mha(
            q,
            k,
            v,
            input_pos,
            cache_loc,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            scale,
            y,
        )

    return y.view(*output_shape)


@flattened_mha_with_cache.register_fake
def flattened_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: Optional[float],
):
    return q.new_empty(*q.shape[:-1], v.shape[-1]).contiguous()


@torch.library.custom_op("attention::prepare_fused_mha_metadata", mutates_args=())
def prepare_fused_mha_metadata(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
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


# TODO: Move the truncation of inputs out of this custom op
# SequenceInfo._get_sanitized_num_sequences could break in fake mode
@prepare_fused_mha_metadata.register_fake
def prepare_fused_mha_metadata_fake(
    input_ids, position_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
):
    num_seq = SequenceInfo._get_sanitized_num_sequences(input_ids, seq_len)
    return (
        torch.empty_like(seq_len[:num_seq]),
        torch.empty_like(input_pos[:num_seq]),
        torch.empty_like(cache_loc[:num_seq]),
        torch.empty_like(seq_len[:num_seq]),
    )


@AttentionRegistry.register("TritonWithFlattenedInputs")
class TritonWithFlattenedInputs(AttentionDescriptor):
    @classmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""
        return False

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the source op and the cached attention op."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.attention.bsnd_grouped_sdpa

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.attention.flattened_mha_with_cache

    @classmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        return torch.ops.attention.prepare_fused_mha_metadata, 4

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: CacheConfig
    ) -> CacheInitializerDict:
        # source op is [bsnd] layout already
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        v_fake: FakeTensor = source_attn_node.args[2].meta["val"]
        num_kv_heads = k_fake.shape[2]
        k_head_dim = k_fake.shape[3]
        v_head_dim = v_fake.shape[3]

        def _get_k_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for TritonWithFlattenedInputs"
            return torch.empty(
                si.num_pages,
                si.page_size,
                num_kv_heads,
                k_head_dim,
                device=si.device,
                dtype=cache_config.dtype or k_fake.dtype,
            )

        def _get_v_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for TritonWithFlattenedInputs"
            return torch.empty(
                si.num_pages,
                si.page_size,
                num_kv_heads,
                v_head_dim,
                device=si.device,
                dtype=cache_config.dtype or v_fake.dtype,
            )

        return {"k_cache": _get_k_cache, "v_cache": _get_v_cache}

    @classmethod
    def get_global_buffer_initializers(cls, source_attn_node: Node) -> BufferInitializerDict:
        return {}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        # retrieve head_dim from k_fake
        attn_mask, dropout_p, is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )
        if attn_mask is not None or dropout_p != 0.0 or not is_causal:
            ad_logger.debug(
                "Unsupported attention arguments for "
                f"{source_attn_node=}: {attn_mask=}, {dropout_p=}, {is_causal=}"
            )

        # Get scale from args or kwargs
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        # do a sanity check on the scale if it is not None, we only support the default scale
        # of 1/sqrt(head_dim) and so we should do an approximate check for that one
        if not isinstance(scale, float):
            ad_logger.warning("Provided scale is not a float, Using default scale instead.")
            scale = None

        return [
            scale,  # softmax scale
        ]
