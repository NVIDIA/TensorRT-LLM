# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Triton prefill attention with custom mask support and paged KV cache.

Used as a fallback when the FlashInfer trtllm-gen backend cannot handle custom
(non-causal) attention masks, e.g. for Gemma4 multimodal bidirectional attention
on full attention layers with head_dim=512.

Ported from sglang's extend_attention kernel with modifications:
- Stage 1 (prefix KV) adapted for FlashInfer's HND paged KV cache layout
  [num_pages, 2, num_kv_heads, page_size, head_dim]
- Stage 2 (extend KV) reads from contiguous tensors (unchanged from sglang)
- Stripped: HIP, DPE (differential positional encoding), xai_temperature, sinks

Reference: sglang/srt/layers/attention/triton_ops/extend_attention.py
"""

from typing import Optional

import torch
import triton
import triton.language as tl

CUDA_CAPABILITY = torch.cuda.get_device_capability()


def _get_block_sizes(Lq: int, Lv: int):
    """Get block sizes and configuration for the attention kernel.

    Args:
        Lq: Query head dimension.
        Lv: Value head dimension.

    Returns:
        (BLOCK_DMODEL, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps)
    """
    BLOCK_DMODEL = triton.next_power_of_2(Lq)
    BLOCK_DV = triton.next_power_of_2(Lv)

    if CUDA_CAPABILITY[0] == 12:
        # sm120 workstation Blackwell (RTX Pro 6000) — smaller shared memory
        if Lq <= 128:
            BLOCK_M, BLOCK_N = (64, 128)
        elif Lq <= 256:
            BLOCK_M, BLOCK_N = (64, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 32)
    elif CUDA_CAPABILITY[0] >= 10:
        # sm100 datacenter Blackwell (B200) — same as Hopper heuristic
        if Lq <= 256:
            BLOCK_M, BLOCK_N = (128, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    elif CUDA_CAPABILITY[0] >= 9:
        # Hopper (H100)
        if Lq <= 256:
            BLOCK_M, BLOCK_N = (128, 64)
        else:
            BLOCK_M, BLOCK_N = (32, 64)
    elif CUDA_CAPABILITY[0] >= 8:
        # Ampere (A100)
        if CUDA_CAPABILITY[1] in (6, 9):
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (64, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 32)
        else:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
    else:
        BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

    num_warps = 4 if Lq <= 64 else 8
    return BLOCK_DMODEL, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel(
    # Extend (new) tokens — contiguous [num_ctx_tokens, num_heads, head_dim]
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    # Prefix (cached) KV — paged, HND layout
    K_Buffer,
    V_Buffer,
    # Sequence boundaries
    qo_indptr,  # [batch+1] cumulative extend token counts
    prefix_indptr,  # [batch+1] cumulative prefix TOKEN counts
    # Page table for prefix
    page_table,  # [total_pages] physical page IDs
    page_table_indptr,  # [batch+1] cumulative page counts per seq
    # Mask
    mask_ptr,
    mask_indptr,  # [batch+1] cumulative mask element counts
    # Scalars
    sm_scale,
    kv_group_num,
    # Extend tensor strides
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    # Buffer strides (K_Buffer / V_Buffer after kv_cache.select(1, 0/1))
    # Shape: [num_pages, num_kv_heads, page_size, head_dim]
    stride_buf_kpage,
    stride_buf_kh,
    stride_buf_ktoken,
    stride_buf_kdim,
    stride_buf_vpage,
    stride_buf_vh,
    stride_buf_vtoken,
    stride_buf_vdim,
    # Constexprs
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    # Extend (query) range for this sequence
    cur_seq_extend_start = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start

    # Prefix range for this sequence
    cur_seq_prefix_start = tl.load(prefix_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(prefix_indptr + cur_seq + 1) - cur_seq_prefix_start
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    # Page table start for this sequence's prefix
    cur_seq_page_start = tl.load(page_table_indptr + cur_seq)

    if USE_CUSTOM_MASK:
        cur_seq_mask_start = tl.load(mask_indptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    # Load query tile
    offs_q = (
        (cur_seq_extend_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q_Extend + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Online softmax accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # ---- Stage 1: attend to prefix KV (paged cache, HND layout) ----
    for start_n in range(0, cur_seq_len_prefix, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0,
            )
            final_mask &= custom_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # 2-level paged access: token position → (page_local_idx, token_in_page)
            token_positions = start_n + offs_n
            page_local_idx = token_positions // PAGE_SIZE
            token_in_page = token_positions % PAGE_SIZE
            page_ids = tl.load(
                page_table + cur_seq_page_start + page_local_idx,
                mask=mask_n,
                other=0,
            )

            # Load K from paged buffer (transposed for dot product)
            offs_buf_k = (
                page_ids[None, :] * stride_buf_kpage
                + cur_kv_head * stride_buf_kh
                + token_in_page[None, :] * stride_buf_ktoken
                + offs_d[:, None] * stride_buf_kdim
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=mask_n[None, :] & mask_d[:, None],
                other=0.0,
            )

            qk = tl.dot(q.to(k.dtype), k)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            # Load V from paged buffer
            offs_buf_v = (
                page_ids[:, None] * stride_buf_vpage
                + cur_kv_head * stride_buf_vh
                + token_in_page[:, None] * stride_buf_vtoken
                + offs_dv[None, :] * stride_buf_vdim
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)
            e_max = n_e_max

    # ---- Stage 2: attend to extend KV (contiguous tensors) ----
    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    for start_n in range(0, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causal = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (start_n + offs_n[None, :])
            mask_causal &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causal
        else:
            final_mask &= mask_m[:, None] & mask_n[None, :]

        SKIP_TILE = False
        if USE_CUSTOM_MASK:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # Load K from extend tensor (transposed)
            offs_k = (
                (cur_seq_extend_start + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k,
                mask=mask_n[None, :] & mask_d[:, None],
                other=0.0,
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            # Load V from extend tensor
            offs_v = (
                (cur_seq_extend_start + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)
            e_max = n_e_max

    # ---- Write output ----
    offs_o = (
        (cur_seq_extend_start + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o,
        acc / deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def _extend_attention_fwd(
    q_extend: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    o_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    prefix_indptr: torch.Tensor,
    page_table: torch.Tensor,
    page_table_indptr: torch.Tensor,
    custom_mask: Optional[torch.Tensor],
    mask_indptr: Optional[torch.Tensor],
    max_len_extend: int,
    sm_scale: float,
    page_size: int,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    skip_prefix_custom_mask: bool = False,
):
    """Launch the Triton extend attention kernel.

    Args:
        q_extend: [num_ctx_tokens, num_heads, head_dim]
        k_extend: [num_ctx_tokens, num_kv_heads, head_dim]
        v_extend: [num_ctx_tokens, num_kv_heads, head_dim]
        o_extend: [num_ctx_tokens, num_heads, head_dim]
        k_buffer: K view of paged cache [num_pages, num_kv_heads, page_size, head_dim]
        v_buffer: V view of paged cache [num_pages, num_kv_heads, page_size, head_dim]
        qo_indptr: [batch+1] extend token boundaries
        prefix_indptr: [batch+1] cumulative prefix token counts
        page_table: [total_pages] physical page IDs
        page_table_indptr: [batch+1] cumulative page counts per seq
        custom_mask: Flattened bool mask or None
        mask_indptr: [batch+1] cumulative mask element counts, or None
        max_len_extend: Max extend sequence length in batch
        sm_scale: Softmax scale factor
        page_size: Tokens per cache page
        is_causal: Use causal masking for extend (stage 2)
        logit_cap: Logit soft-capping (0.0 = disabled)
        skip_prefix_custom_mask: Skip custom mask for prefix (stage 1)
    """
    Lq = q_extend.shape[-1]
    Lv = v_extend.shape[-1]

    BLOCK_DMODEL, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = _get_block_sizes(Lq, Lv)

    batch_size = qo_indptr.shape[0] - 1
    head_num = q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        prefix_indptr,
        page_table,
        page_table_indptr,
        custom_mask,
        mask_indptr,
        sm_scale,
        kv_group_num,
        # Extend tensor strides
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        # Buffer strides (K and V views from paged cache)
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        k_buffer.stride(3),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        v_buffer.stride(3),
        # Constexprs
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        Lq=Lq,
        Lv=Lv,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_CUSTOM_MASK=custom_mask is not None,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
        num_warps=num_warps,
        num_stages=1,
    )


def triton_prefill_with_custom_mask(
    # Extend (new) tokens — contiguous
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    qo_indptr: torch.Tensor,
    # Prefix (cached) KV — paged
    kv_cache: Optional[torch.Tensor],
    prefix_lens: torch.Tensor,
    page_table_indptr: torch.Tensor,
    page_table_indices: torch.Tensor,
    page_size: int,
    # Attention config
    custom_mask: Optional[torch.Tensor],
    sm_scale: float,
) -> None:
    """Triton prefill attention with optional custom mask and paged KV cache.

    Replaces FlashInfer prefill for layers where trtllm-gen cannot handle:
    - Custom (bidirectional) masks for head_dim>256 multimodal layers.
    - Mixed Q/KV dtypes during prefill (FP8 KV cache with NVFP4 models).
    When custom_mask is None, uses causal attention (lower-triangular mask).

    Two-stage attention:
      Stage 1: Attend to prefix KV in paged cache (skipped when prefix_lens=0)
      Stage 2: Attend to extend KV from contiguous q, k, v tensors

    Args:
        q: Query tensor [num_ctx_tokens, num_heads, head_dim]
        k: Key tensor [num_ctx_tokens, num_kv_heads, head_dim]
        v: Value tensor [num_ctx_tokens, num_kv_heads, head_dim]
        output: Output tensor [num_ctx_tokens, num_heads, head_dim]
        qo_indptr: [num_contexts + 1] per-sequence extend token boundaries
        kv_cache: Paged KV cache [num_pages, 2, num_kv_heads, page_size, head_dim]
                  or None when no prefix exists.
        prefix_lens: [num_contexts] prefix token count per seq (0 = no prefix)
        page_table_indptr: [num_contexts + 1] cumulative page counts per seq
        page_table_indices: [total_pages] physical page IDs
        page_size: Tokens per cache page
        custom_mask: Flattened bool mask, concatenated per sequence.
                     Per-seq shape: [extend_len, prefix_len + extend_len].
                     None for causal attention.
        sm_scale: Softmax scale factor
    """
    device = q.device
    num_contexts = qo_indptr.shape[0] - 1
    extend_lens = qo_indptr[1:] - qo_indptr[:-1]

    # Build causal mask when custom_mask is not provided
    total_kv_lens = prefix_lens + extend_lens
    if custom_mask is None:
        # Generate causal mask: each query at position i attends to
        # positions [0..prefix_len+i] (all prefix + causal extend)
        mask_parts = []
        for seq_idx in range(num_contexts):
            ext = int(extend_lens[seq_idx].item())
            pre = int(prefix_lens[seq_idx].item())
            total = pre + ext
            # Row i of extend can see columns [0..pre+i]
            rows = torch.arange(ext, device=device).unsqueeze(1)
            cols = torch.arange(total, device=device).unsqueeze(0)
            seq_mask = cols <= (rows + pre)
            mask_parts.append(seq_mask.reshape(-1))
        custom_mask = torch.cat(mask_parts)

    # Compute mask_indptr: each seq's mask is [extend_len, prefix_len + extend_len]
    mask_sizes = extend_lens * total_kv_lens
    mask_indptr = torch.zeros(num_contexts + 1, dtype=torch.int32, device=device)
    mask_indptr[1:] = torch.cumsum(mask_sizes, dim=0)

    # Compute prefix_indptr: cumulative prefix token counts
    prefix_indptr = torch.zeros(num_contexts + 1, dtype=torch.int32, device=device)
    prefix_indptr[1:] = torch.cumsum(prefix_lens, dim=0)

    # Extract K/V views from paged cache.  When the KV cache uses a
    # narrower dtype (e.g. FP8 with NVFP4 quantization), cast to the
    # compute dtype (Q's dtype, typically BF16) so the Triton kernel
    # can run tl.dot in BF16.  Only the accessed pages are copied.
    compute_dtype = q.dtype
    if kv_cache is not None:
        k_buffer = kv_cache.select(1, 0)  # [num_pages, num_kv_heads, page_size, head_dim]
        v_buffer = kv_cache.select(1, 1)
        if k_buffer.dtype != compute_dtype:
            num_prefix_pages = int(page_table_indptr[-1].item())
            if num_prefix_pages > 0:
                k_buffer = k_buffer.to(compute_dtype)
                v_buffer = v_buffer.to(compute_dtype)
    else:
        # Dummy 4D buffers — not accessed when all prefix_lens are 0,
        # but must have 4 dims so stride(0..3) works in the kernel launcher.
        num_kv_heads = k.shape[1]
        head_dim = k.shape[2]
        k_buffer = torch.empty(1, num_kv_heads, 1, head_dim, dtype=k.dtype, device=device)
        v_buffer = torch.empty(1, num_kv_heads, 1, head_dim, dtype=v.dtype, device=device)

    max_len_extend = int(extend_lens.max().item()) if num_contexts > 0 else 0

    _extend_attention_fwd(
        q_extend=q,
        k_extend=k,
        v_extend=v,
        o_extend=output,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        prefix_indptr=prefix_indptr,
        page_table=page_table_indices,
        page_table_indptr=page_table_indptr,
        custom_mask=custom_mask,
        mask_indptr=mask_indptr,
        max_len_extend=max_len_extend,
        sm_scale=sm_scale,
        page_size=page_size,
        is_causal=False,
        logit_cap=0.0,
        skip_prefix_custom_mask=False,
    )
