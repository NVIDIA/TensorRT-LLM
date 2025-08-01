"""Internal triton attention ops that are not actively used in the auto-deploy pipeline."""

import math
from typing import Optional

import torch
import triton

from .triton_attention import _flattened_context_mha, _generate_mha
from .triton_kernels.attention_with_kv_cache import (
    attention_kv_stage2,
    context_attention_kv,
    context_attention_kv_flattened,
    gqa_attention_kv_stage1,
    update_kv_cache_rope_fusion,
)
from .triton_kernels.attention_with_paged_kv_cache import (
    attention_kv_paged_stage1,
    context_attention_kv_paged,
    update_paged_kv_cache,
)


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
        False,
        None,
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


@torch.library.custom_op(
    "auto_deploy::triton_attention_fused_mha_with_paged_cache", mutates_args=()
)
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
            fn_rope = torch.ops.auto_deploy.triton_rope_with_input_pos
        else:
            rope_args = (freqs_cis, input_pos, seq_len, seq_start)
            fn_rope = torch.ops.auto_deploy.triton_rope_on_flattened_inputs
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
    scale = 1.0 / math.sqrt(d_head)
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
        scale,
        max_seq_len,
        n_heads,
        n_kv_heads,
        d_head,
        d_head,
        SEQ_BLOCK_SIZE,
        HEAD_BLOCK_SIZE,
        -1,
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
        False,
        None,
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
        -1,
        False,
        None,
    )


@torch.library.custom_op(
    "auto_deploy::triton_attention_fused_flattened_mha_with_cache_rope_fusion", mutates_args=()
)
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


@torch.library.custom_op("auto_deploy::triton_attention_fused_mha_with_cache", mutates_args=())
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
        q = torch.ops.auto_deploy.triton_rope_with_input_pos(q, freqs_cis, input_pos, "bsnd")
        k = torch.ops.auto_deploy.triton_rope_with_input_pos(k, freqs_cis, input_pos, "bsnd")

    # attention (assumed layout is bsnd)
    y = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    if s > 1:
        # context phase
        _context_mha(q, k, v, k_cache, v_cache, y)
    else:
        # generate phase
        cache_locs = torch.arange(0, b, device=q.device, dtype=torch.int32)
        _generate_mha(q, k, v, k_cache, v_cache, cache_locs, input_pos, scale, y)

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


@torch.library.custom_op(
    "auto_deploy::triton_attention_fused_flattened_mha_with_cache", mutates_args=()
)
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
            fn_rope = torch.ops.auto_deploy.triton_rope_with_input_pos
        else:
            rope_args = (freqs_cis, input_pos, seq_len, seq_start)
            fn_rope = torch.ops.auto_deploy.triton_rope_on_flattened_inputs
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
