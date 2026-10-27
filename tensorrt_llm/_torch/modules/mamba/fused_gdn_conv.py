# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GDN prefill causal conv1d fused with the post-conv Q/K/V split, Q/K L2 norm and gating."""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID


@triton.jit
def _head_columns(
    head,
    NUM_K_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Input channels, output columns and dim mask of Q, K or V head ``head`` (Q, then K, then V)."""
    dims = tl.arange(0, BLOCK_D)
    if head < 2 * NUM_K_HEADS:
        chans = head * HEAD_K_DIM + dims
        cols = (head % NUM_K_HEADS) * HEAD_K_DIM + dims
        dim_mask = dims < HEAD_K_DIM
    else:
        v_head = head - 2 * NUM_K_HEADS
        chans = 2 * NUM_K_HEADS * HEAD_K_DIM + v_head * HEAD_V_DIM + dims
        cols = v_head * HEAD_V_DIM + dims
        dim_mask = dims < HEAD_V_DIM
    return chans, cols, dim_mask


@triton.jit
def _conv_silu_tile(
    x_ptr,
    x_stride,
    w_ptr,
    bias_ptr,
    window_ptr,
    window_stride_dim,
    window_stride_hist,
    tokens,
    conv_mask,
    seq_start,
    chans,
    chan_mask,
    HAS_BIAS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_WINDOW: tl.constexpr,
):
    """SiLU(causal depthwise conv) of a [BLOCK_T, BLOCK_D] tile, rounded to the input dtype.

    Taps before ``seq_start`` read zero, or the cached conv window when ``USE_WINDOW``.
    """
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + chans, mask=chan_mask, other=0.0).to(tl.float32)[None, :]
    for j in tl.static_range(WIDTH):
        src = tokens - (WIDTH - 1) + j
        in_seq = src >= seq_start
        x = tl.load(
            x_ptr + src[:, None].to(tl.int64) * x_stride + chans[None, :],
            mask=(conv_mask & in_seq)[:, None] & chan_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if USE_WINDOW:
            hist = src - seq_start + (WIDTH - 1)
            x += tl.load(
                window_ptr
                + chans[None, :] * window_stride_dim
                + hist[:, None] * window_stride_hist,
                mask=(conv_mask & ~in_seq)[:, None] & chan_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        w = tl.load(w_ptr + chans * WIDTH + j, mask=chan_mask, other=0.0).to(tl.float32)
        acc += x * w[None, :]
    acc = acc / (1.0 + tl.exp(-acc))
    return acc.to(x_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _store_head_tile(
    q_ptr,
    k_ptr,
    v_ptr,
    y,
    head,
    tokens,
    token_mask,
    cols,
    dim_mask,
    l2_norm_eps,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
):
    """Store a head's tile into Q, K or V ``[num_tokens, heads * dim]``; Q/K are L2-normalized."""
    rows = tokens[:, None].to(tl.int64)
    mask = token_mask[:, None] & dim_mask[None, :]
    if head < 2 * NUM_K_HEADS:
        y *= (1.0 / tl.sqrt(tl.sum(y * y, axis=1) + l2_norm_eps))[:, None]
        offsets = rows * (NUM_K_HEADS * HEAD_K_DIM) + cols[None, :]
        if head < NUM_K_HEADS:
            tl.store(q_ptr + offsets, y, mask=mask)
        else:
            tl.store(k_ptr + offsets, y, mask=mask)
    else:
        tl.store(v_ptr + rows * (NUM_V_HEADS * HEAD_V_DIM) + cols[None, :], y, mask=mask)


@triton.jit(do_not_specialize=["num_prefill_tokens", "num_tokens", "num_seqs", "num_search_steps"])
def _fused_gdn_conv_post_conv_kernel(
    x_ptr,
    x_stride,
    decode_ptr,
    decode_stride,
    w_ptr,
    bias_ptr,
    query_start_loc_ptr,
    a_ptr,
    a_stride,
    b_ptr,
    b_stride,
    A_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    num_prefill_tokens,
    num_tokens,
    num_seqs,
    num_search_steps,
    l2_norm_eps,
    softplus_threshold,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DECODE: tl.constexpr,
):
    """One Q, K or V head (and its g/beta) for BLOCK_T tokens; taps before a sequence read zero."""
    head = tl.program_id(1)
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = tokens < num_tokens
    conv_mask = tokens < num_prefill_tokens

    # Sequence of each prefill token: binary search over the sequence end offsets.
    lo = tl.zeros([BLOCK_T], dtype=tl.int32)
    hi = tl.zeros([BLOCK_T], dtype=tl.int32) + num_seqs
    for _ in range(num_search_steps):
        active = lo < hi
        mid = (lo + hi) // 2
        seq_end = tl.load(query_start_loc_ptr + mid + 1, mask=active, other=0)
        go_right = active & (seq_end <= tokens)
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(active & ~go_right, mid, hi)
    seq_start = tl.load(query_start_loc_ptr + lo, mask=conv_mask, other=0)

    chans, cols, dim_mask = _head_columns(head, NUM_K_HEADS, HEAD_K_DIM, HEAD_V_DIM, BLOCK_D)
    y = _conv_silu_tile(
        x_ptr,
        x_stride,
        w_ptr,
        bias_ptr,
        x_ptr,
        0,
        0,
        tokens,
        conv_mask,
        seq_start,
        chans,
        dim_mask,
        HAS_BIAS,
        WIDTH,
        BLOCK_T,
        BLOCK_D,
        False,
    )
    if HAS_DECODE:
        # Decode rows arrive already convolved.
        rows = (tokens - num_prefill_tokens).to(tl.int64)
        y_decode = tl.load(
            decode_ptr + rows[:, None] * decode_stride + chans[None, :],
            mask=(token_mask & ~conv_mask)[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        y = tl.where(conv_mask[:, None], y, y_decode)
    _store_head_tile(
        q_ptr,
        k_ptr,
        v_ptr,
        y,
        head,
        tokens,
        token_mask,
        cols,
        dim_mask,
        l2_norm_eps,
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
    )

    if head >= 2 * NUM_K_HEADS:
        v_head = head - 2 * NUM_K_HEADS
        a = tl.load(a_ptr + tokens.to(tl.int64) * a_stride + v_head, mask=token_mask, other=0.0)
        b = tl.load(b_ptr + tokens.to(tl.int64) * b_stride + v_head, mask=token_mask, other=0.0)
        gate_input = a.to(tl.float32) + tl.load(dt_bias_ptr + v_head).to(tl.float32)
        softplus = tl.where(
            gate_input <= softplus_threshold,
            tl.log(1.0 + tl.exp(gate_input)),
            gate_input,
        )
        A_log = tl.load(A_log_ptr + v_head).to(tl.float32)
        gate_offsets = tokens.to(tl.int64) * NUM_V_HEADS + v_head
        tl.store(g_ptr + gate_offsets, -tl.exp(A_log) * softplus, mask=token_mask)
        tl.store(beta_ptr + gate_offsets, tl.sigmoid(b.to(tl.float32)), mask=token_mask)


@triton.jit
def _gdn_conv_window_kernel(
    x_ptr,
    x_stride,
    w_ptr,
    bias_ptr,
    state_ptr,
    state_stride_slot,
    state_stride_dim,
    state_stride_hist,
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    pad_slot_id,
    l2_norm_eps,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K_DIM: tl.constexpr,
    HEAD_V_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_W: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Per (sequence, head): redo the first WIDTH - 1 outputs from the cached window, then roll it."""
    seq = tl.program_id(0)
    head = tl.program_id(1)
    seq_start = tl.load(query_start_loc_ptr + seq)
    seq_len = tl.load(query_start_loc_ptr + seq + 1) - seq_start
    slot = tl.load(cache_indices_ptr + seq)
    valid = slot != pad_slot_id
    use_window = valid & (tl.load(has_initial_state_ptr + seq) != 0)
    window_ptr = state_ptr + slot.to(tl.int64) * state_stride_slot
    chans, cols, dim_mask = _head_columns(head, NUM_K_HEADS, HEAD_K_DIM, HEAD_V_DIM, BLOCK_D)

    tokens = seq_start + tl.arange(0, BLOCK_W)
    conv_mask = (tokens < seq_start + tl.minimum(seq_len, WIDTH - 1)) & use_window
    y = _conv_silu_tile(
        x_ptr,
        x_stride,
        w_ptr,
        bias_ptr,
        window_ptr,
        state_stride_dim,
        state_stride_hist,
        tokens,
        conv_mask,
        seq_start,
        chans,
        dim_mask,
        HAS_BIAS,
        WIDTH,
        BLOCK_W,
        BLOCK_D,
        True,
    )
    _store_head_tile(
        q_ptr,
        k_ptr,
        v_ptr,
        y,
        head,
        tokens,
        conv_mask,
        cols,
        dim_mask,
        l2_norm_eps,
        NUM_K_HEADS,
        NUM_V_HEADS,
        HEAD_K_DIM,
        HEAD_V_DIM,
    )

    # Next window: the last WIDTH - 1 inputs, shifting in the old window for shorter sequences.
    hist = tl.arange(0, BLOCK_W)
    pos = seq_len - (WIDTH - 1) + hist
    mask = (hist < WIDTH - 1)[:, None] & dim_mask[None, :]
    x = tl.load(
        x_ptr + (seq_start + pos)[:, None].to(tl.int64) * x_stride + chans[None, :],
        mask=mask & (pos >= 0)[:, None],
        other=0.0,
    ).to(tl.float32)
    old = tl.load(
        window_ptr
        + chans[None, :] * state_stride_dim
        + (hist + seq_len)[:, None] * state_stride_hist,
        mask=mask & ((pos < 0) & use_window)[:, None],
        other=0.0,
    ).to(tl.float32)
    window = tl.where((pos >= 0)[:, None], x, old)
    # Every read of the old window above must land before any thread overwrites it.
    tl.debug_barrier()
    tl.store(
        window_ptr + chans[None, :] * state_stride_dim + hist[:, None] * state_stride_hist,
        window.to(state_ptr.dtype.element_ty),
        mask=mask & valid,
    )


def fused_gdn_conv_post_conv(
    mixed_qkv: torch.Tensor,
    decode: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    pad_slot_id: int = PAD_SLOT_ID,
    l2_norm_eps: float = 1e-6,
    softplus_threshold: float = 20.0,
    beta_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``causal_conv1d_fn`` (SiLU) on token-major prefill rows fused with ``fused_gdn_post_conv``.

    Arguments not listed below are as in ``fused_gdn_post_conv``; ``a`` and ``b`` need a unit
    head stride.

    Args:
        mixed_qkv: Prefill conv input ``[num_prefill_tokens, qkv_dim]``, unit channel stride.
        decode: Optional decode causal-conv output ``[num_decode_tokens, qkv_dim]``, unit channel
            stride, placed after the prefill tokens in the outputs.
        weight: Contiguous conv weight ``[qkv_dim, width]``.
        bias: Optional conv bias ``[qkv_dim]``.
        conv_states: Conv windows ``[num_slots, qkv_dim, width - 1]``, updated in place.
        query_start_loc: Prefill cumulative sequence lengths ``[num_prefills + 1]``, int32.
        cache_indices: Conv-window slot of each prefill sequence ``[num_prefills]``, int32.
        has_initial_state: Whether each prefill sequence continues its cached window
            ``[num_prefills]``, bool.
        pad_slot_id: Cache index of padded sequences, whose conv window is left untouched.

    Returns:
        Q, K, V, log-space G and beta, laid out as by ``fused_gdn_post_conv``.
    """
    num_prefill_tokens, qkv_dim = mixed_qkv.shape
    num_decode_tokens = 0 if decode is None else decode.shape[0]
    num_tokens = num_prefill_tokens + num_decode_tokens
    num_seqs = query_start_loc.shape[0] - 1
    width = weight.shape[-1]
    assert weight.shape == (qkv_dim, width) and weight.is_contiguous() and width >= 2
    assert qkv_dim == 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    assert mixed_qkv.stride(1) == 1 and (decode is None or decode.stride(1) == 1)
    assert a.stride(1) == 1 and b.stride(1) == 1

    device = mixed_qkv.device
    q = torch.empty((1, num_tokens, num_k_heads, head_k_dim), dtype=mixed_qkv.dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty((1, num_tokens, num_v_heads, head_v_dim), dtype=mixed_qkv.dtype, device=device)
    g = torch.empty((1, num_tokens, num_v_heads), dtype=torch.float32, device=device)
    beta = torch.empty((1, num_tokens, num_v_heads), dtype=beta_dtype, device=device)
    if num_tokens == 0:
        return q, k, v, g, beta

    # Triton requires a tensor for every pointer argument; HAS_DECODE/HAS_BIAS compile out the
    # accesses through these stand-ins.
    decode_input = mixed_qkv if decode is None else decode
    bias_input = weight if bias is None else bias
    num_heads = 2 * num_k_heads + num_v_heads
    block_d = triton.next_power_of_2(max(head_k_dim, head_v_dim))
    block_t = 16
    _fused_gdn_conv_post_conv_kernel[(triton.cdiv(num_tokens, block_t), num_heads)](
        mixed_qkv,
        mixed_qkv.stride(0),
        decode_input,
        decode_input.stride(0),
        weight,
        bias_input,
        query_start_loc,
        a,
        a.stride(0),
        b,
        b.stride(0),
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        num_prefill_tokens,
        num_tokens,
        num_seqs,
        num_seqs.bit_length(),
        l2_norm_eps,
        softplus_threshold,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_K_DIM=head_k_dim,
        HEAD_V_DIM=head_v_dim,
        BLOCK_D=block_d,
        BLOCK_T=block_t,
        WIDTH=width,
        HAS_BIAS=bias is not None,
        HAS_DECODE=decode is not None,
        num_warps=4,
    )
    if num_seqs > 0:
        # Must follow the launch above: it overwrites the first width - 1 outputs of each sequence
        # that continues a cached window.
        _gdn_conv_window_kernel[(num_seqs, num_heads)](
            mixed_qkv,
            mixed_qkv.stride(0),
            weight,
            bias_input,
            conv_states,
            *conv_states.stride(),
            query_start_loc,
            cache_indices,
            has_initial_state,
            q,
            k,
            v,
            pad_slot_id,
            l2_norm_eps,
            NUM_K_HEADS=num_k_heads,
            NUM_V_HEADS=num_v_heads,
            HEAD_K_DIM=head_k_dim,
            HEAD_V_DIM=head_v_dim,
            BLOCK_D=block_d,
            BLOCK_W=triton.next_power_of_2(width - 1),
            WIDTH=width,
            HAS_BIAS=bias is not None,
            num_warps=1,
        )
    return q, k, v, g, beta
