"""Custom ops for MHA/XQA attention."""

import math
from dataclasses import astuple
from typing import List, Optional

import torch
import torch.nn.functional as F
import triton

from .attention_interface import AttentionDescriptor, AttentionRegistry, SequenceInfo
from .triton_kernels.attention_with_kv_cache import (
    attention_kv_stage2,
    context_attention_kv,
    context_attention_kv_flattened,
    gqa_attention_kv_stage1,
    update_kv_cache,
    update_kv_cache_rope_fusion,
)
from .triton_kernels.attention_with_paged_kv_cache import (
    attention_kv_paged_stage1,
    context_attention_kv_paged,
    update_paged_kv_cache,
)


@torch.library.custom_op("attention::scaled_dot_product_attention", mutates_args=())
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """A carbon copy of torch.nn.functional.scaled_dot_product_attention as custom op.

    Using this custom op instead of using the functional directly ensures consistent representation
    of the vanilla sdpa in a graph.
    """
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


@scaled_dot_product_attention.register_fake
def scaled_dot_product_attention_fake(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    """Fake implementation of scaled_dot_product_attention."""
    return torch.empty_like(query)


def _generate_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_locs: torch.Tensor,
    input_pos: torch.Tensor,
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

    (
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
        ),
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


def _context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
):
    b, s, n_heads, q_d_head = q.shape
    max_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]

    SEQ_BLOCK = 32
    softmax_scale = 1.0 / math.sqrt(q_d_head)
    grid = (b, n_heads, (s + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        s,
        out,
        softmax_scale,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK,
        max_seq_len,
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_mha_with_cache", mutates_args=())
def fused_mha_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused MHA with cache that takes raw input from q, k, v GEMMs."""
    # b, s info
    b, s = q.shape[:2]
    head_dim = k_cache.shape[-1]

    # reshapes with num_heads and head_dim
    q = q.view(b, s, -1, head_dim)
    k = k.view(b, s, -1, head_dim)
    v = v.view(b, s, -1, head_dim)

    # rope embedding
    if freqs_cis is not None:
        q = torch.ops.rope.apply_rope_with_input_pos(q, freqs_cis, input_pos, "bsnd")
        k = torch.ops.rope.apply_rope_with_input_pos(k, freqs_cis, input_pos, "bsnd")

    # attention (assumed layout is bsnd)
    y = torch.empty_like(q)
    if s > 1:
        # context phase
        _context_mha(q, k, v, k_cache, v_cache, y)
    else:
        # generate phase
        cache_locs = torch.arange(0, b, device=q.device, dtype=torch.int32)
        _generate_mha(q, k, v, k_cache, v_cache, cache_locs, input_pos, y)

    return y.view(b, s, -1)  # [b,s,n*h_d]


@fused_mha_with_cache.register_fake
def fused_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    return torch.empty_like(q.contiguous())


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
    out: torch.Tensor,
) -> None:
    # NOTE: s_total == sum(seq_len)
    s_total, n_heads, q_d_head = q.shape
    max_cache_seq_len, n_kv_heads = k_cache.shape[1:3]
    v_d_head = v.shape[-1]
    BATCH_SIZE: int = len(input_pos)
    SEQ_BLOCK = 32
    (
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
        ),
    )
    # TODO: use input_pos to get the correct cache locations
    softmax_scale = 1.0 / math.sqrt(q_d_head)
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
        softmax_scale,
        n_heads,
        n_kv_heads,
        q_d_head,
        v_d_head,
        SEQ_BLOCK,
        max_cache_seq_len,
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_flattened_mha_with_cache", mutates_args=())
def fused_flattened_mha_with_cache(
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
    freqs_cis: torch.Tensor,
    # CONSTANTS
    # <none>
) -> torch.Tensor:
    """Flattened & fused MHA with cache that takes raw input from q, k, v GEMMs.

    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.
    head_dim = k_cache.shape[-1]
    b, s, d = q.shape

    # reshapes with num_heads and head_dim
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)
    q = q.view(*bs_view, q.shape[2] // head_dim, head_dim)
    k = k.view(*bs_view, k.shape[2] // head_dim, head_dim)
    v = v.view(*bs_view, v.shape[2] // head_dim, head_dim)

    # rope embedding for generate-only or mixed
    if freqs_cis.numel() > 0:
        if s == 1:
            rope_args = (freqs_cis, input_pos, "bsnd")
            fn_rope = torch.ops.rope.apply_rope_with_input_pos
        else:
            rope_args = (freqs_cis, input_pos, seq_len, seq_start)
            fn_rope = torch.ops.rope.apply_rope_on_flattened_inputs
        q = fn_rope(q, *rope_args)
        k = fn_rope(k, *rope_args)

    # run attention
    y = torch.empty_like(q)
    if s == 1:
        # generate-only phase
        _generate_mha(q, k, v, k_cache, v_cache, cache_loc, input_pos, y)
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
            y,
        )

    return y.view(b, s, d)  # [b,s,n*h_d]


@fused_flattened_mha_with_cache.register_fake
def fused_flattened_mha_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    return torch.empty_like(q.contiguous())


def _generate_mha_rope_fusion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs_cis: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_locs: torch.Tensor,
    input_pos: torch.Tensor,
    out: torch.Tensor,
):
    b, (n_heads, d_head) = q.shape[0], q.shape[-2:]
    max_seq_len, n_kv_heads = k_cache.shape[1:3]
    device = q.device

    SEQ_BLOCK_SIZE = 64
    num_blocks = (max_seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE
    stage1_output_values = torch.empty(
        b, n_heads, num_blocks, d_head, device=device, dtype=torch.float32
    )
    stage1_output_logsumexp = torch.empty(
        b, n_heads, num_blocks, device=device, dtype=torch.float32
    ) - float("inf")
    q_rope = torch.empty_like(q)
    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))

    (
        update_kv_cache_rope_fusion[(b, n_kv_heads, 1)](
            q,
            k,
            v,
            None,
            None,
            q_rope,
            k_cache,
            v_cache,
            input_pos,
            cache_locs,
            freqs_cis,
            max_seq_len,
            n_heads,
            n_kv_heads,
            d_head,
            1,
            HEAD_BLOCK_SIZE,
            GENERATE_ONLY=True,
        ),
    )

    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))
    gqa_attention_kv_stage1[
        (
            b,
            n_kv_heads,
            num_blocks,
        )
    ](
        q_rope,
        k_cache,
        v_cache,
        cache_locs,
        input_pos,
        stage1_output_values,
        stage1_output_logsumexp,
        num_blocks,
        max_seq_len,
        n_heads,
        n_kv_heads,
        d_head,
        d_head,
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
        d_head,
        SEQ_BLOCK_SIZE,
    )


def _flattened_context_mha_rope_fusion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    freqs_cis: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # NOTE: s_total == sum(seq_len)
    s_total, n_heads, d_head = q.shape
    max_cache_seq_len, n_kv_heads = k_cache.shape[1:3]
    BATCH_SIZE: int = len(input_pos)
    SEQ_BLOCK = 32
    q_rope = torch.empty_like(q)
    HEAD_BLOCK_SIZE = max(16, triton.next_power_of_2(n_heads // n_kv_heads))
    (
        update_kv_cache_rope_fusion[
            (BATCH_SIZE, n_kv_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
        ](
            q,
            k,
            v,
            seq_len,
            seq_start,
            q_rope,
            k_cache,
            v_cache,
            input_pos,
            cache_loc,
            freqs_cis,
            max_cache_seq_len,
            n_heads,
            n_kv_heads,
            d_head,
            32,
            HEAD_BLOCK_SIZE,
            GENERATE_ONLY=False,
        ),
    )
    # TODO: use input_pos to get the correct cache locations
    softmax_scale = 1.0 / math.sqrt(d_head)
    grid = (BATCH_SIZE, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_flattened[grid](
        q_rope,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        input_pos,
        cache_loc,
        out,
        softmax_scale,
        n_heads,
        n_kv_heads,
        d_head,
        d_head,
        SEQ_BLOCK,
        max_cache_seq_len,
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_flattened_mha_with_cache_rope_fusion", mutates_args=())
def fused_flattened_mha_with_cache_rope_fusion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Flattened & fused MHA with cache that takes raw input from q, k, v GEMMs.

    Fuse k rope in update_kv_cache and q rope in attention.
    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # this function only handle requests with rope embadding.
    if freqs_cis is None:
        return fused_flattened_mha_with_cache(
            q,
            k,
            v,
            input_pos,
            cache_loc,
            seq_len,
            seq_start,
            k_cache,
            v_cache,
            freqs_cis,
        )

    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.
    b, s, d = q.shape
    head_dim = k_cache.shape[-1]

    # reshapes with num_heads and head_dim
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)
    q = q.view(*bs_view, q.shape[2] // head_dim, head_dim)
    k = k.view(*bs_view, k.shape[2] // head_dim, head_dim)
    v = v.view(*bs_view, v.shape[2] // head_dim, head_dim)

    # run attention
    y = torch.empty_like(q)
    if s == 1:
        # generate-only phase
        _generate_mha_rope_fusion(q, k, v, freqs_cis, k_cache, v_cache, cache_loc, input_pos, y)
    else:
        # mixed context + generate phase
        _flattened_context_mha_rope_fusion(
            q,
            k,
            v,
            freqs_cis,
            input_pos,
            cache_loc,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            y,
        )

    return y.view(b, s, d)  # [b,s,n*h_d]


@fused_flattened_mha_with_cache_rope_fusion.register_fake
def fused_flattened_mha_with_cache_rope_fusion_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    return torch.empty_like(q.contiguous())


def _paged_generate_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_loc: torch.Tensor,
    input_pos: torch.Tensor,
    out: torch.Tensor,
    max_seq_len: int,
):
    b, (n_heads, d_head) = q.shape[0], q.shape[-2:]
    PAGE_SIZE, n_kv_heads = k_cache.shape[1:3]
    device = q.device

    SEQ_BLOCK_SIZE = 64
    num_blocks = (max_seq_len + SEQ_BLOCK_SIZE - 1) // SEQ_BLOCK_SIZE
    stage1_output_values = torch.empty(
        b, n_heads, num_blocks, d_head, device=device, dtype=torch.float32
    )
    stage1_output_logsumexp = torch.empty(
        b, n_heads, num_blocks, device=device, dtype=torch.float32
    ) - float("inf")

    (
        update_paged_kv_cache[(b, n_kv_heads, 1)](
            k,
            v,
            None,
            None,
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            page_table,
            n_kv_heads,
            d_head,
            SEQ_BLOCK_SIZE,
            max_seq_len,
            PAGE_SIZE,
            page_table.stride(0),
            GENERATE_ONLY=True,
        ),
    )

    attention_kv_paged_stage1[
        (
            b,
            n_heads,
            num_blocks,
        )
    ](
        q,
        k_cache,
        v_cache,
        cache_loc,
        page_table,
        input_pos,
        stage1_output_values,
        stage1_output_logsumexp,
        num_blocks,
        max_seq_len,
        n_heads,
        n_kv_heads,
        d_head,
        SEQ_BLOCK_SIZE,
        PAGE_SIZE,
        page_table.stride(0),
    )
    attention_kv_stage2[(b, n_heads, 1)](
        stage1_output_values,
        stage1_output_logsumexp,
        out,
        input_pos,
        num_blocks,
        n_heads,
        d_head,
        SEQ_BLOCK_SIZE,
    )


def _paged_context_mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    out: torch.Tensor,
    max_seq_len: int,  # max cache length of sequence, kv_cache shape don't provide this info.
) -> None:
    # NOTE: s_total == sum(seq_len)
    s_total, n_heads, d_head = q.shape
    PAGE_SIZE, n_kv_heads = k_cache.shape[1:3]
    BATCH_SIZE = len(input_pos)
    SEQ_BLOCK = 32
    (
        update_paged_kv_cache[
            (BATCH_SIZE, n_kv_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
        ](
            k,
            v,
            seq_len,
            seq_start,
            k_cache,
            v_cache,
            cache_loc,
            input_pos,
            page_table,
            n_kv_heads,
            d_head,
            SEQ_BLOCK,
            max_seq_len,
            PAGE_SIZE,
            page_table.stride(0),
            GENERATE_ONLY=False,
        ),
    )
    softmax_scale = 1.0 / math.sqrt(d_head)
    grid = (BATCH_SIZE, n_heads, (max(seq_len) + SEQ_BLOCK - 1) // SEQ_BLOCK)
    context_attention_kv_paged[grid](
        q,
        seq_len,
        seq_start,
        k_cache,
        v_cache,
        cache_loc,
        input_pos,
        page_table,
        softmax_scale,
        out,
        n_heads,
        n_kv_heads,
        d_head,
        SEQ_BLOCK,
        max_seq_len,
        PAGE_SIZE,
        page_table.stride(0),
        num_stages=2,
    )


@torch.library.custom_op("attention::fused_mha_with_paged_cache", mutates_args=())
def fused_mha_with_paged_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused MHA with paged cache that takes raw input from q, k, v GEMMs.

    NOTE: this op can also handle seq_len==0, which might be useful for CUDAGRAPH.
    """
    # b, s info
    # NOTE: b, s are just the shapes of the input tensor q; not necessarily the number of sequences.
    # Generally speaking, we expect one of two cases here:
    # 1. b > 0, s==1: this indicates a generate-only batch of tokens.
    # 2. b==1, s > 0: this indicates a mixed context+generate phase. The actual number of sequences
    #    and number of tokens per sequence are encoded in seq_len and seq_start.
    #    Assuming that context seq_len always > 0.
    b, s, d = q.shape
    head_dim = k_cache.shape[-1]

    # reshapes with num_heads and head_dim
    if s == 1:
        bs_view = (b, s)
    else:
        bs_view = (b * s,)
    q = q.view(*bs_view, q.shape[2] // head_dim, head_dim)
    k = k.view(*bs_view, k.shape[2] // head_dim, head_dim)
    v = v.view(*bs_view, v.shape[2] // head_dim, head_dim)

    # rope embedding for generate-only or mixed
    if freqs_cis is not None:
        if s == 1:
            rope_args = (freqs_cis, input_pos, "bsnd")
            fn_rope = torch.ops.rope.apply_rope_with_input_pos
        else:
            rope_args = (freqs_cis, input_pos, seq_len, seq_start)
            fn_rope = torch.ops.rope.apply_rope_on_flattened_inputs
        q = fn_rope(q, *rope_args)
        k = fn_rope(k, *rope_args)

    # run attention
    y = torch.empty_like(q)
    if s == 1:
        # generate-only phase
        _paged_generate_mha(
            q, k, v, page_table, k_cache, v_cache, cache_loc, input_pos, y, max_seq_len
        )
    else:
        # mixed context + generate phase
        _paged_context_mha(
            q,
            k,
            v,
            input_pos,
            cache_loc,
            page_table,
            k_cache,
            v_cache,
            seq_len,
            seq_start,
            y,
            max_seq_len,
        )

    return y.view(b, s, d)  # [b,s,n*h_d]


@fused_mha_with_paged_cache.register_fake
def fused_mha_with_paged_cache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.empty_like(q.contiguous())


@torch.library.custom_op("attention::prepare_fused_mha_metadata", mutates_args=())
def prepare_fused_mha_metadata(
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


@prepare_fused_mha_metadata.register_fake
def prepare_fused_mha_metadata_fake(
    input_ids, seq_len, input_pos, cache_loc, pages_per_seq, page_size
):
    return (
        torch.empty_like(seq_len),
        torch.empty_like(input_pos),
        torch.empty_like(cache_loc),
        torch.empty_like(seq_len),
    )


@AttentionRegistry.register("TritonWithFlattenedInputs")
class TritonWithFlattenedInputs(AttentionDescriptor):
    @classmethod
    def is_paged(cls):
        """Return if the attention op is paged or not."""
        return False

    @classmethod
    def get_attention_op(cls):
        return torch.ops.attention.fused_flattened_mha_with_cache, 3

    @classmethod
    def get_prepare_metadata_op(cls):
        return torch.ops.attention.prepare_fused_mha_metadata, 4

    @classmethod
    def get_cache_initializers(cls, get_info):
        def _get_cache(si: SequenceInfo):
            assert not si.is_paged, "Paged cache not supported for TritonWithFlattenedInputs"
            attention_info = get_info()
            return torch.empty(
                si.num_pages,
                si.page_size,
                attention_info.num_kv_heads,
                attention_info.head_dim,
                device=si.device,
                dtype=attention_info.cache_config.dtype or attention_info.dtype,
            )

        return {"k_cache": _get_cache, "v_cache": _get_cache}

    @classmethod
    def get_global_buffer_initializers(cls, get_info):
        attention_info = get_info()
        head_dim = attention_info.head_dim
        pos_embd_config = attention_info.pos_embd_config

        def _get_freqs_cis(si: SequenceInfo):
            if pos_embd_config.mode is None:
                return torch.empty(0, device=si.device)
            assert pos_embd_config.mode == "rope", f"Mode {pos_embd_config.mode=} not supported"
            assert pos_embd_config.rope_scale == 1.0, f"{pos_embd_config.rope_scale=} not supported"
            rope_theta = pos_embd_config.rope_theta
            return cls._precompute_freqs_cis(2 * si.max_seq_len, head_dim, rope_theta).to(si.device)

        k_full = "_".join(map(str, ["freqs_cis", *astuple(pos_embd_config)])).replace(".", "_")
        return {k_full: _get_freqs_cis}

    @staticmethod
    def _precompute_freqs_cis(
        seq_len: int, head_dim: int, rope_theta: Optional[float] = None
    ) -> torch.Tensor:
        if rope_theta is None:
            rope_theta = 1e4
        freqs = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
        )
        t = torch.arange(seq_len)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        # cos and sin (real and img) are packed
        cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
        return cache.to(dtype=torch.float16)
