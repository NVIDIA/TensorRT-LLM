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

"""Triton-based MLA (Multi-head Latent Attention) backend with unpaged cache.

This module provides:
- triton_cached_mla_with_cache: Triton-optimized cached MLA with KV cache
- TritonMLAAttention: Attention descriptor for Triton MLA backend

Both prefill and decode paths use Triton-accelerated attention with weight
absorption, eliminating Python loops entirely. A single shared kernel handles
both phases via per-token KV length metadata for causal masking.

MLA Cache Layout (same as torch_mla backend):
    mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    - compressed_kv_cached = mla_cache[:, :, :kv_lora_rank]  (zero-copy slice)
    - kpe_cached = mla_cache[:, :, kv_lora_rank:]  (zero-copy slice)
"""

import math
from typing import List, Optional

import torch
import triton
import triton.language as tl
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)


def _get_mla_multihead_config(num_tokens: int, is_prefill: bool, max_kv_len: int = 512) -> tuple:
    """Best (SEQ_BLOCK, HEAD_BLOCK, num_warps, num_stages) for the multihead kernel.

    tl.dot requires SEQ_BLOCK >= 16.  HEAD_BLOCK must divide N_HEADS=32.
    Derived from parallel warps × stages sweep + HEAD_BLOCK sweep on H100 80GB HBM3.

    Decode HEAD_BLOCK selection (iter 18):
      HB=4 wins for B≤16 (A1-A8): 1-5% faster — more programs on H100 (32×B vs 8×B
      head-groups) without hurting cache-sharing efficiency at low batch sizes.
      HB=8 wins for B>16 (A9-A10): 22-29% faster — at B=32, HB=4 doubles program count
      to 256 (2 waves) while HB=8 fills H100 (132 SMs) in 1 wave.
    Decode SEQ_BLOCK selection for HB=4 path (iter 21):
      SB=128 when max_kv_len > 64: A2-A7 gain 5-13% (fewer loop iterations, better
      pipeline fill); A1 only shape with kv≤64, where SB=128 wastes 50% of threads.
      SB=64 when max_kv_len ≤ 64: avoids wasted threads from masked-out blocks.
    Decode SEQ_BLOCK for HB=8 path (iter 36):
      SB=64 when max_kv_len ≤ 256 (A9): avoids half-empty blocks (kv=256/SB=128=2 iters
      → 13.84µs vs SB=128: 14.08µs, +1.7%). SB=128 for longer kv (A10) as before.
    Prefill B1 stages (iter 35):
      stages=3 for T≤128: 18.55µs vs stages=2: 18.61µs (tie/marginal); consistent
      1-pipeline improvement vs iter 19 baseline.
    For HB=32 prefill, num_stages=2 avoids SMEM pressure (stages=5 OOM):
      B2=96.4µs, B3=288.5µs, B4=982.5µs.

    Note: split-K path overrides nw based on max_kv_len in the dispatch function.
    """
    if is_prefill:
        # B1 T≤128: HB=16, SB=64, w=8, s=3 → 18.55µs (stages=3 marginally better)
        #   HB=16 doubles cache sharing vs HB=8: grid=(T, 2) uses 2 programs per token.
        # B2-B4 T>128: HB=32, SB=128, w=8, s=2 → 96.4/288.5/982.5µs
        # stages=5 OOM for HB=32 SB=128 (SMEM overflow); stages=2 best.
        if num_tokens <= 128:
            return 64, 16, 8, 3
        else:
            return 128, 32, 8, 2
    else:
        # HB=4 for B≤16: more SM programs without sacrificing cache sharing (A1-A8)
        # HB=8 for B>16: ~1 full H100 wave, better cache sharing vs HB=4
        # Adaptive SB for B>16 (iter 36): SB=64 for kv≤256 (A9: 13.84µs, +1.7% vs
        #   SB=128). SB=128 for longer kv (A10: 18.0µs, +8.5% vs SB=64).
        if num_tokens <= 16:
            # SB=128 when context > SB: A2-A7 gain 5-13%; A1 (kv≤64) uses SB=64
            # to avoid wasted threads (2 blocks of 64 vs 1 block of 128 half-empty).
            sb = 64 if max_kv_len <= 64 else 128
            return sb, 4, 8, 3
        else:
            sb = 64 if max_kv_len <= 256 else 128
            return sb, 8, 8, 3


@triton.jit
def _mla_attention_kernel_splitk(
    # Tensor pointers
    q_absorbed_ptr,  # [num_tokens, N, kv_lora_rank]
    q_pe_ptr,  # [num_tokens, N, qk_rope_head_dim]
    mla_cache_ptr,  # [max_batch, max_seq, cache_dim]
    token_slot_ptr,  # [num_tokens]
    token_kv_len_ptr,  # [num_tokens]
    workspace_acc_ptr,  # [num_tokens, N, num_parts, KV_BLOCK] fp32
    workspace_ml_ptr,  # [num_tokens, N, num_parts, 2] fp32 (m, l per head)
    # Constexpr parameters
    SCALE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    CACHE_DIM: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    PE_BLOCK: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    NUM_PARTS: tl.constexpr,
):
    """Split-K variant of multihead MLA kernel.

    Grid: (num_tokens, N_HEADS // HEAD_BLOCK, NUM_PARTS)

    Each program computes a partial attention over kv blocks
    [part_start, part_end) without normalizing. Stores (acc, m, l) to workspace
    for Python-side reduction across NUM_PARTS partitions.
    """
    token_id = tl.program_id(0)
    head_group = tl.program_id(1)
    part_id = tl.program_id(2)
    head_start = head_group * HEAD_BLOCK

    slot_idx = tl.load(token_slot_ptr + token_id)
    kv_len = tl.load(token_kv_len_ptr + token_id)

    kv_offsets = tl.arange(0, KV_BLOCK)
    kv_mask = kv_offsets < KV_LORA_RANK
    pe_offsets = tl.arange(0, PE_BLOCK)
    pe_mask = pe_offsets < QK_ROPE_HEAD_DIM
    head_offsets = tl.arange(0, HEAD_BLOCK)

    # Load q_absorbed and q_pe for all HEAD_BLOCK heads
    q_abs_ptrs = (
        q_absorbed_ptr
        + token_id * N_HEADS * KV_LORA_RANK
        + (head_start + head_offsets[:, None]) * KV_LORA_RANK
        + kv_offsets[None, :]
    )
    q_abs_all = tl.load(q_abs_ptrs, mask=kv_mask[None, :], other=0.0).to(tl.bfloat16)

    q_pe_ptrs = (
        q_pe_ptr
        + token_id * N_HEADS * QK_ROPE_HEAD_DIM
        + (head_start + head_offsets[:, None]) * QK_ROPE_HEAD_DIM
        + pe_offsets[None, :]
    )
    q_pe_all = tl.load(q_pe_ptrs, mask=pe_mask[None, :], other=0.0).to(tl.bfloat16)

    # Per-head online softmax state (partial)
    m_i = tl.full([HEAD_BLOCK], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    acc = tl.zeros([HEAD_BLOCK, KV_BLOCK], dtype=tl.float32)

    cache_batch_base = tl.multiple_of(slot_idx * MAX_SEQ_LEN * CACHE_DIM, CACHE_DIM)
    total_blocks = tl.cdiv(kv_len, SEQ_BLOCK)
    # Partition blocks across NUM_PARTS: part_id handles [part_start, part_end)
    blocks_per_part = tl.cdiv(total_blocks, NUM_PARTS)
    part_start_block = part_id * blocks_per_part
    part_end_block = tl.minimum(part_start_block + blocks_per_part, total_blocks)

    for block_id in range(part_start_block, part_end_block):
        block_start = tl.multiple_of(block_id * SEQ_BLOCK, SEQ_BLOCK)
        seq_offsets = block_start + tl.arange(0, SEQ_BLOCK)
        seq_mask = seq_offsets < kv_len

        ckv_ptrs = (
            mla_cache_ptr
            + cache_batch_base
            + seq_offsets[:, None] * CACHE_DIM
            + kv_offsets[None, :]
        )
        ckv_bf16 = tl.load(
            ckv_ptrs,
            mask=seq_mask[:, None] & kv_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )

        kpe_ptrs = (
            mla_cache_ptr
            + cache_batch_base
            + seq_offsets[:, None] * CACHE_DIM
            + KV_LORA_RANK
            + pe_offsets[None, :]
        )
        kpe_bf16 = tl.load(
            kpe_ptrs,
            mask=seq_mask[:, None] & pe_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )

        scores_nope = tl.dot(q_abs_all, tl.trans(ckv_bf16)).to(tl.float32)
        scores_pe = tl.dot(q_pe_all, tl.trans(kpe_bf16)).to(tl.float32)
        scores = (scores_nope + scores_pe) * (SCALE * 1.44269504)

        seq_mask_2d = seq_mask[None, :]
        scores = tl.where(seq_mask_2d, scores, float("-inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores - m_new[:, None])
        p = tl.where(seq_mask_2d, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), ckv_bf16).to(tl.float32)
        m_i = m_new

    # Store partial (acc, m, l) to workspace — NO normalization
    # workspace_acc: [T, N_HEADS, NUM_PARTS, KV_BLOCK]
    # workspace_ml:  [T, N_HEADS, NUM_PARTS, 2]
    #
    # Use vectorized 2-D stores (same style as multihead kernel) to avoid
    # constexpr integer indexing into 2-D tensors, which Triton 3.x rejects.
    acc_ptrs = (
        workspace_acc_ptr
        + (token_id * N_HEADS + head_start + head_offsets[:, None]) * NUM_PARTS * KV_BLOCK
        + part_id * KV_BLOCK
        + kv_offsets[None, :]
    )
    tl.store(acc_ptrs, acc, mask=kv_mask[None, :])

    ml_base = (
        workspace_ml_ptr
        + (token_id * N_HEADS + head_start + head_offsets) * NUM_PARTS * 2
        + part_id * 2
    )
    tl.store(ml_base, m_i)
    tl.store(ml_base + 1, l_i)


@triton.jit
def _mla_splitk_reduce(
    workspace_acc_ptr,  # [num_tokens, N_HEADS, NUM_PARTS, KV_BLOCK] fp32
    workspace_ml_ptr,  # [num_tokens, N_HEADS, NUM_PARTS, 2] fp32 (m, l)
    out_ptr,  # [num_tokens, N_HEADS, KV_BLOCK] bf16
    N_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    NUM_PARTS: tl.constexpr,
):
    """Reduce split-K partial (acc, m, l) tensors into normalized output.

    Grid: (num_tokens, N_HEADS)

    Loads NUM_PARTS partial results, combines them with numerically stable
    log-sum-exp, normalizes, and writes bf16 output.
    """
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    kv_offsets = tl.arange(0, KV_BLOCK)
    kv_mask = kv_offsets < KV_LORA_RANK

    base = token_id * N_HEADS + head_id  # flat (token, head) index

    # Initialize with partition 0
    ml0_ptr = workspace_ml_ptr + base * NUM_PARTS * 2
    m_cur = tl.load(ml0_ptr)
    l_cur = tl.load(ml0_ptr + 1)
    acc_cur = tl.load(
        workspace_acc_ptr + base * NUM_PARTS * KV_BLOCK + kv_offsets,
        mask=kv_mask,
    )  # [KV_BLOCK]

    for p in tl.static_range(1, NUM_PARTS):
        ml_p_ptr = workspace_ml_ptr + base * NUM_PARTS * 2 + p * 2
        m_p = tl.load(ml_p_ptr)
        l_p = tl.load(ml_p_ptr + 1)
        acc_p = tl.load(
            workspace_acc_ptr + base * NUM_PARTS * KV_BLOCK + p * KV_BLOCK + kv_offsets,
            mask=kv_mask,
        )
        m_new = tl.maximum(m_cur, m_p)
        alpha = tl.math.exp2(m_cur - m_new)
        beta = tl.math.exp2(m_p - m_new)
        l_cur = l_cur * alpha + l_p * beta
        acc_cur = acc_cur * alpha + acc_p * beta
        m_cur = m_new

    # Normalize and store as bf16
    out = acc_cur / tl.maximum(l_cur, 1e-38)
    tl.store(
        out_ptr + base * KV_BLOCK + kv_offsets,
        out.to(tl.bfloat16),
        mask=kv_mask,
    )


@triton.jit
def _mla_attention_kernel_multihead(
    # Tensor pointers
    q_absorbed_ptr,  # [num_tokens, N, kv_lora_rank]
    q_pe_ptr,  # [num_tokens, N, qk_rope_head_dim]
    mla_cache_ptr,  # [max_batch, max_seq, cache_dim]
    token_slot_ptr,  # [num_tokens]
    token_kv_len_ptr,  # [num_tokens]
    out_ptr,  # [num_tokens, N, kv_lora_rank] (float32)
    # Constexpr parameters
    SCALE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    CACHE_DIM: tl.constexpr,
    KV_BLOCK: tl.constexpr,
    PE_BLOCK: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,  # heads processed per program; must divide N_HEADS
):
    """Multi-head MLA attention kernel: HEAD_BLOCK heads share one cache load.

    Grid: (num_tokens, N_HEADS // HEAD_BLOCK)

    Each program processes HEAD_BLOCK consecutive heads for one token.
    The KV cache tiles (ckv, kpe) are loaded ONCE per SEQ_BLOCK and reused
    across all HEAD_BLOCK heads, reducing HBM traffic by HEAD_BLOCK×.

    For HEAD_BLOCK >= 16 with SEQ_BLOCK >= 16, inner products use tl.dot
    (bf16 × bf16 → fp32) to leverage tensor cores.
    """
    token_id = tl.program_id(0)
    head_group = tl.program_id(1)
    head_start = head_group * HEAD_BLOCK

    slot_idx = tl.load(token_slot_ptr + token_id)
    kv_len = tl.load(token_kv_len_ptr + token_id)

    kv_offsets = tl.arange(0, KV_BLOCK)  # [KV_BLOCK]
    kv_mask = kv_offsets < KV_LORA_RANK
    pe_offsets = tl.arange(0, PE_BLOCK)  # [PE_BLOCK]
    pe_mask = pe_offsets < QK_ROPE_HEAD_DIM
    head_offsets = tl.arange(0, HEAD_BLOCK)  # [HEAD_BLOCK]

    # Load q_absorbed for all HEAD_BLOCK heads: [HEAD_BLOCK, KV_BLOCK]
    # Explicitly cast to bf16 (not fp32) to reduce register pressure.
    # tl.dot(bf16, bf16) uses tensor cores natively; no roundtrip through fp32 needed.
    q_abs_ptrs = (
        q_absorbed_ptr
        + token_id * N_HEADS * KV_LORA_RANK
        + (head_start + head_offsets[:, None]) * KV_LORA_RANK
        + kv_offsets[None, :]
    )
    q_abs_all = tl.load(q_abs_ptrs, mask=kv_mask[None, :], other=0.0).to(
        tl.bfloat16
    )  # [HEAD_BLOCK, KV_BLOCK], bf16

    # Load q_pe for all HEAD_BLOCK heads: [HEAD_BLOCK, PE_BLOCK]
    q_pe_ptrs = (
        q_pe_ptr
        + token_id * N_HEADS * QK_ROPE_HEAD_DIM
        + (head_start + head_offsets[:, None]) * QK_ROPE_HEAD_DIM
        + pe_offsets[None, :]
    )
    q_pe_all = tl.load(q_pe_ptrs, mask=pe_mask[None, :], other=0.0).to(
        tl.bfloat16
    )  # [HEAD_BLOCK, PE_BLOCK], bf16

    # Per-head online softmax state
    m_i = tl.full([HEAD_BLOCK], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
    acc = tl.zeros([HEAD_BLOCK, KV_BLOCK], dtype=tl.float32)

    # tl.multiple_of hints: cache_batch_base is a multiple of CACHE_DIM (row stride),
    # allowing the compiler to prove alignment for vectorized memory operations.
    cache_batch_base = tl.multiple_of(slot_idx * MAX_SEQ_LEN * CACHE_DIM, CACHE_DIM)
    num_blocks = tl.cdiv(kv_len, SEQ_BLOCK)

    for block_id in range(0, num_blocks):
        # block_start is a multiple of SEQ_BLOCK — hint helps compiler prove alignment
        # of seq_offsets and optimize address computation in the inner loop.
        block_start = tl.multiple_of(block_id * SEQ_BLOCK, SEQ_BLOCK)
        seq_offsets = block_start + tl.arange(0, SEQ_BLOCK)
        seq_mask = seq_offsets < kv_len

        # Load compressed_kv ONCE for all HEAD_BLOCK heads: [SEQ_BLOCK, KV_BLOCK]
        ckv_ptrs = (
            mla_cache_ptr
            + cache_batch_base
            + seq_offsets[:, None] * CACHE_DIM
            + kv_offsets[None, :]
        )
        ckv_bf16 = tl.load(
            ckv_ptrs,
            mask=seq_mask[:, None] & kv_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )  # [SEQ_BLOCK, KV_BLOCK], stays in bf16 for tl.dot

        # Load kpe ONCE for all HEAD_BLOCK heads: [SEQ_BLOCK, PE_BLOCK]
        kpe_ptrs = (
            mla_cache_ptr
            + cache_batch_base
            + seq_offsets[:, None] * CACHE_DIM
            + KV_LORA_RANK
            + pe_offsets[None, :]
        )
        kpe_bf16 = tl.load(
            kpe_ptrs,
            mask=seq_mask[:, None] & pe_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )  # [SEQ_BLOCK, PE_BLOCK], stays in bf16 for tl.dot

        # Compute scores for all HEAD_BLOCK heads at once: [HEAD_BLOCK, SEQ_BLOCK]
        # bf16 x bf16 -> fp32 via tl.dot (tensor cores); q tensors already in bf16
        scores_nope = tl.dot(q_abs_all, tl.trans(ckv_bf16)).to(tl.float32)
        scores_pe = tl.dot(q_pe_all, tl.trans(kpe_bf16)).to(tl.float32)
        # Multiply by SCALE * log2e so exp2 can be used: exp2(x*log2e) == exp(x).
        # tl.math.exp2 maps to ex2.approx.f32 (native H100 instruction, ~4x faster).
        scores = (scores_nope + scores_pe) * (SCALE * 1.44269504)

        # Mask out-of-bounds positions
        seq_mask_2d = seq_mask[None, :]  # [1, SEQ_BLOCK] broadcasts to [HEAD_BLOCK, SEQ_BLOCK]
        scores = tl.where(seq_mask_2d, scores, float("-inf"))

        # Online softmax update in log2 space — vectorized across HEAD_BLOCK
        m_ij = tl.max(scores, axis=1)  # [HEAD_BLOCK]
        m_new = tl.maximum(m_i, m_ij)  # [HEAD_BLOCK]
        alpha = tl.math.exp2(m_i - m_new)  # [HEAD_BLOCK]
        p = tl.math.exp2(scores - m_new[:, None])  # [HEAD_BLOCK, SEQ_BLOCK]
        p = tl.where(seq_mask_2d, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)  # [HEAD_BLOCK]
        # acc update via tl.dot: [HEAD_BLOCK, SEQ_BLOCK] x [SEQ_BLOCK, KV_BLOCK]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), ckv_bf16).to(tl.float32)
        m_i = m_new

    # Normalize
    safe_l_i = tl.maximum(l_i, 1e-38)
    acc = acc / safe_l_i[:, None]

    # Store HEAD_BLOCK results
    out_ptrs = (
        out_ptr
        + token_id * N_HEADS * KV_LORA_RANK
        + (head_start + head_offsets[:, None]) * KV_LORA_RANK
        + kv_offsets[None, :]
    )
    tl.store(out_ptrs, acc, mask=kv_mask[None, :])


def _triton_mla_decode(
    q_nope: torch.Tensor,  # [B, 1, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, 1, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, 1, kv_lora_rank]
    kpe: torch.Tensor,  # [B, 1, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, cache_dim]
    slot_idx: torch.Tensor,  # [num_decode]
    input_pos: torch.Tensor,  # [num_decode]
    scale: float,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,  # [B, N, v_head_dim]
) -> None:
    """Triton-accelerated MLA decode with weight absorption.

    Steps:
    1. Cache update (PyTorch advanced indexing)
    2. Weight absorption: q_absorbed = q_nope @ w_k_nope^T (PyTorch einsum)
    3. Triton kernel: fused attention scoring + online softmax + weighted sum
    4. Value projection: out = weighted_kv @ w_v^T (PyTorch einsum)
    """
    b = q_nope.shape[0]
    qk_rope_head_dim = q_pe.shape[3]

    # Reshape kv_b_proj_weight to extract w_k_nope and w_v
    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_k_nope = weight_reshaped[:, :qk_nope_head_dim, :]  # [N, qk_nope_head_dim, kv_lora_rank]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]  # [N, v_head_dim, kv_lora_rank]

    # Step 1: Update cache with new token
    compressed_kv_flat = compressed_kv.squeeze(1)  # [B, kv_lora_rank]
    kpe_flat = kpe.squeeze(1).squeeze(1)  # [B, qk_rope_head_dim]

    cache_dtype = mla_cache.dtype
    if compressed_kv_flat.dtype != cache_dtype:
        compressed_kv_for_cache = compressed_kv_flat.to(cache_dtype)
    else:
        compressed_kv_for_cache = compressed_kv_flat
    if kpe_flat.dtype != cache_dtype:
        kpe_for_cache = kpe_flat.to(cache_dtype)
    else:
        kpe_for_cache = kpe_flat

    # Batch scatter into cache (no Python loops)
    mla_cache[slot_idx.long(), input_pos.long(), :kv_lora_rank] = compressed_kv_for_cache
    mla_cache[slot_idx.long(), input_pos.long(), kv_lora_rank:] = kpe_for_cache

    # Step 2: Weight absorption
    q_nope_2d = q_nope.squeeze(1)  # [B, N, qk_nope_head_dim]
    q_absorbed = torch.einsum("bnd,ndk->bnk", q_nope_2d, w_k_nope).contiguous()
    # [B, N, kv_lora_rank]

    q_pe_2d = q_pe.squeeze(1).contiguous()  # [B, N, qk_rope_head_dim]

    # Step 3: Triton kernel for attention computation
    # Allocate as model dtype (bf16) instead of fp32: halves HBM write bandwidth and
    # eliminates the .to(q_nope.dtype) cast below.  The kernel accumulates in fp32 and
    # Triton stores fp32→bf16 at the final tl.store, so precision is unchanged.
    weighted_kv = torch.empty(b, num_heads, kv_lora_rank, device=q_nope.device, dtype=q_nope.dtype)

    max_seq_len = mla_cache.shape[1]
    cache_dim = mla_cache.shape[2]
    kv_block = triton.next_power_of_2(kv_lora_rank)
    pe_block = triton.next_power_of_2(qk_rope_head_dim)

    # Per-token KV length: decode attends to positions [0, input_pos]
    kv_len = (input_pos + 1).to(torch.int32)

    # Dispatch: split-K for small-batch long-context (fills more H100 SMs),
    # standard multihead otherwise.
    # Use mla_cache.shape[1] (max_seq_len) as a static upper bound for dispatch.
    # This avoids the D2H sync (.item()) which is illegal inside torch.cuda.graph
    # capture (iter 57: full CUDA graph compatibility fix).
    # All dispatch parameters (seq_block, head_block, num_parts, workspace shapes)
    # are now compile-time constants w.r.t. the capture → stable graph every replay.
    # Per-token kv_len tensor is still passed to each kernel for correct causal masking.
    max_kv_len = mla_cache.shape[1]
    seq_block, head_block, nw, ns = _get_mla_multihead_config(
        b, is_prefill=False, max_kv_len=max_kv_len
    )

    # Split-K: partition kv blocks across NUM_PARTS to increase SM utilization.
    # Small batch (b≤4, kv≥256): T=1 grid=(1,8)=8 programs → 6% SM; split-K fills ~48-97%.
    # Moderate batch (b≤8, kv≥512): T=8 MH grid=(8,8)=64 programs; NP=4 → 256 programs
    #   → 2 waves, ~13% gain (iter 41). kv=256 at T=5-8: multihead still wins.
    # Fine-grained NP (iter 43): NP = total_blocks if total_blocks ≤ 16, else largest
    #   divisor of total_blocks that is ≤ 16. This ensures exactly 1 (or 2, evenly) blocks
    #   per partition — no wasted/uneven partitions. Examples:
    #     kv=256, sb=64: total_blocks=4  → NP=4  (1 blk/part)
    #     kv=384, sb=64: total_blocks=6  → NP=6  (1 blk/part, +3.6% vs NP=8)
    #     kv=512, sb=64: total_blocks=8  → NP=8  (1 blk/part)
    #     kv=768, sb=64: total_blocks=12 → NP=12 (1 blk/part, +14% vs NP=8)
    #     kv=1024,sb=64: total_blocks=16 → NP=16 (1 blk/part)
    #     kv=1536,sb=64: total_blocks=24 → NP=12 (2 blk/part evenly, +1.9% vs NP=16)
    #     kv=2048,sb=128:total_blocks=16 → NP=16 (1 blk/part)
    # b=5-8, kv=512: NP=4 (smaller NP avoids 4-wave overhead; +9-13% vs MH)
    # Adaptive SEQ_BLOCK (iter 28): SB=64 for kv≤1536, SB=128 for kv>1536.
    # Adaptive num_warps (iter 33): w=4 for kv>1024 (SB=128 path), w=8 otherwise.
    # num_stages=2 for split-K (iter 37): each partition handles ≤2 blocks;
    #   stages=2 is identical to stages=3 (no pipeline benefit) but uses less SMEM.
    # iter 59: cap head_block at num_heads to prevent zero-size grid when TP reduces
    # num_heads below head_block threshold (e.g., TP=8 → num_heads=4 < head_block=8).
    # A zero-size grid (num_heads // head_block == 0) launches no kernel programs,
    # leaving weighted_kv uninitialized (garbage/NaN) → NaN in logits.
    head_block = min(head_block, num_heads)

    use_splitk = (b <= 4 and max_kv_len >= 256) or (b <= 8 and max_kv_len >= 512)
    if use_splitk:
        seq_block = 64 if max_kv_len <= 1536 else 128
        if b <= 4:
            # Small batch: NP = total_blocks (ideal: 1 blk/part) capped at 16.
            # For total_blocks > 16, find largest divisor ≤ 16 (avoids uneven partitions).
            total_blocks = max_kv_len // seq_block
            if total_blocks <= 16:
                num_parts = total_blocks
            else:
                num_parts = 1
                for _np in range(16, 0, -1):
                    if total_blocks % _np == 0:
                        num_parts = _np
                        break
        else:
            # Moderate batch (b=5-8): NP depends on seq_block (iter 51).
            # SB=64 (kv≤1536): NP = total_blocks // 2, capped at 8 (2 blk/part optimal):
            #   kv=512 (total_blocks=8): NP=4 — 2 waves with b=8, +13% vs MH
            #   kv=1024 (total_blocks=16): NP=8 — 4 waves, +25% vs MH
            # SB=128 (kv>1536): each block is 128 positions (2× larger than SB=64).
            #   4 blk/part already provides ample work; NP=4 beats NP=8 by 6.7% at kv=2048.
            #   Use NP = total_blocks // 4, capped at 4.
            total_blocks = max_kv_len // seq_block
            if seq_block == 64:
                num_parts = min(max(total_blocks // 2, 1), 8)
            else:
                num_parts = min(max(total_blocks // 4, 1), 4)
        # warps=4 for long-context SB=128 path (A5): fewer threads reduces register
        # pressure; 1.5% gain vs w=8. warps=8 better for shorter kv (A2-A4).
        nw = 4 if max_kv_len > 1024 else 8
        ns = 2  # each partition has ≤2 blocks; extra pipeline stages add no benefit
    else:
        num_parts = 1

    if use_splitk:
        kv_block = triton.next_power_of_2(kv_lora_rank)
        # workspace_acc: [b, num_heads, num_parts, kv_block] fp32
        # workspace_ml:  [b, num_heads, num_parts, 2] fp32 (m, l per head per part)
        workspace_acc = torch.empty(
            b,
            num_heads,
            num_parts,
            kv_block,
            device=q_nope.device,
            dtype=torch.float32,
        )
        workspace_ml = torch.empty(
            b,
            num_heads,
            num_parts,
            2,
            device=q_nope.device,
            dtype=torch.float32,
        )
        grid_sk = (b, num_heads // head_block, num_parts)
        _mla_attention_kernel_splitk[grid_sk](
            q_absorbed,
            q_pe_2d,
            mla_cache,
            slot_idx,
            kv_len,
            workspace_acc,
            workspace_ml,
            SCALE=scale,
            MAX_SEQ_LEN=max_seq_len,
            N_HEADS=num_heads,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            CACHE_DIM=cache_dim,
            KV_BLOCK=kv_block,
            PE_BLOCK=pe_block,
            SEQ_BLOCK=seq_block,
            HEAD_BLOCK=head_block,
            NUM_PARTS=num_parts,
            num_warps=nw,
            num_stages=ns,
        )
        # Reduce partial results via Triton kernel (avoids CPU-GPU sync overhead).
        # Grid: (b, num_heads) — each program combines NUM_PARTS partial (acc,m,l).
        # num_warps=8 (iter 42): +2.5-3.1% vs num_warps=4 across A3-A5; more warps
        #   improve throughput of the NUM_PARTS-loop memory accesses.
        _mla_splitk_reduce[(b, num_heads)](
            workspace_acc,
            workspace_ml,
            weighted_kv,
            N_HEADS=num_heads,
            KV_LORA_RANK=kv_lora_rank,
            KV_BLOCK=kv_block,
            NUM_PARTS=num_parts,
            num_warps=8,
        )
    else:
        grid = (b, num_heads // head_block)
        _mla_attention_kernel_multihead[grid](
            q_absorbed,
            q_pe_2d,
            mla_cache,
            slot_idx,
            kv_len,
            weighted_kv,
            SCALE=scale,
            MAX_SEQ_LEN=max_seq_len,
            N_HEADS=num_heads,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            CACHE_DIM=cache_dim,
            KV_BLOCK=kv_block,
            PE_BLOCK=pe_block,
            SEQ_BLOCK=seq_block,
            HEAD_BLOCK=head_block,
            num_warps=nw,
            num_stages=ns,
        )

    # Step 4: Value projection (weighted_kv already in q_nope.dtype from kernel store)
    attn_out = torch.einsum("bnk,nvk->bnv", weighted_kv, w_v)  # [B, N, v_head_dim]

    out.copy_(attn_out)


def _triton_mla_prefill(
    q_nope: torch.Tensor,  # [total_padded, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [total_padded, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [total_padded, kv_lora_rank]
    kpe: torch.Tensor,  # [total_padded, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    input_pos: torch.Tensor,  # [num_seq] - starting cache position per sequence
    slot_idx: torch.Tensor,  # [num_seq] - cache slot per sequence
    seq_len: torch.Tensor,  # [num_seq] - token count per sequence
    seq_start: torch.Tensor,  # [num_seq] - start index in flattened tensor
    scale: float,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,  # [total_padded, N, v_head_dim]
    total_num_tokens: int = 0,
) -> None:
    """Triton-accelerated MLA prefill with weight absorption.

    Uses the same attention kernel as decode but with per-token causal masking.
    Replaces the PyTorch reference (_torch_mla_context_with_expansion) which had
    Python loops over sequences.

    Steps:
    1. Vectorized cache update (no Python loops)
    2. Weight absorption: q_absorbed = q_nope @ W_kn^T (PyTorch einsum)
    3. Triton kernel: fused attention scoring + online softmax + weighted sum
    4. Value projection: out = weighted_kv @ W_v^T (PyTorch einsum)
    """
    device = q_nope.device
    qk_rope_head_dim = q_pe.shape[2]
    num_seq = seq_len.shape[0]

    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_k_nope = weight_reshaped[:, :qk_nope_head_dim, :]  # [N, qk_nope_head_dim, kv_lora_rank]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]  # [N, v_head_dim, kv_lora_rank]
    cache_dtype = mla_cache.dtype
    max_seq_len = mla_cache.shape[1]
    cache_dim = mla_cache.shape[2]
    kv_block = triton.next_power_of_2(kv_lora_rank)
    pe_block = triton.next_power_of_2(qk_rope_head_dim)

    if num_seq == 1:
        # =====================================================================
        # iter 61: fully-inlined single-sequence fast path.
        #
        # For a single contiguous sequence seq_start=0, all per-token index
        # tensors are identity ([0, 1, ..., T-1]), so every index_select call
        # is a no-op.  We eliminate:
        #   • seq_len.item() D2H sync        → q_nope.shape[0] (Python int, free)
        #   • repeat_interleave / cumsum / arange overhead  (~275 µs, iter 60)
        #   • 4 × index_select calls         (~80 µs, this iter)
        #   • out.zero_() + index_copy_      (~16 µs) → out.copy_()
        # Net saving vs iter 60: ~95 µs; vs original general path: ~370 µs.
        # =====================================================================
        total_tokens = q_nope.shape[0]  # Python int from .shape — zero GPU ops
        if total_tokens == 0:
            out.zero_()
            return

        # Sequential cache positions: [pos, pos+1, ..., pos+T-1]
        within_offsets = torch.arange(total_tokens, device=device, dtype=torch.long)
        token_slots = slot_idx[0:1].long().repeat(total_tokens)  # contiguous (triton needs it)
        token_cache_pos = input_pos[0:1].long() + within_offsets

        # Cache write: tokens are in order, no gather needed
        kpe_flat = kpe.squeeze(1)  # [T, qk_rope_head_dim]
        ckv_for_cache = (
            compressed_kv.to(cache_dtype) if compressed_kv.dtype != cache_dtype else compressed_kv
        )
        kpe_for_cache = kpe_flat.to(cache_dtype) if kpe_flat.dtype != cache_dtype else kpe_flat
        mla_cache[token_slots, token_cache_pos, :kv_lora_rank] = ckv_for_cache
        mla_cache[token_slots, token_cache_pos, kv_lora_rank:] = kpe_for_cache

        # Weight absorption: use q_nope / q_pe directly (identity mapping)
        q_absorbed = torch.einsum("tnd,ndk->tnk", q_nope.float(), w_k_nope.float()).contiguous()
        q_pe_actual = q_pe.contiguous()

        token_kv_len = (token_cache_pos + 1).to(torch.int32)
        weighted_kv = torch.empty(
            total_tokens, num_heads, kv_lora_rank, device=device, dtype=q_nope.dtype
        )
        seq_block, head_block, nw, ns = _get_mla_multihead_config(total_tokens, is_prefill=True)
        head_block = min(head_block, num_heads)  # iter 59: cap for TP
        grid = (total_tokens, num_heads // head_block)
        _mla_attention_kernel_multihead[grid](
            q_absorbed,
            q_pe_actual,
            mla_cache,
            token_slots,
            token_kv_len,
            weighted_kv,
            SCALE=scale,
            MAX_SEQ_LEN=max_seq_len,
            N_HEADS=num_heads,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            CACHE_DIM=cache_dim,
            KV_BLOCK=kv_block,
            PE_BLOCK=pe_block,
            SEQ_BLOCK=seq_block,
            HEAD_BLOCK=head_block,
            num_warps=nw,
            num_stages=ns,
        )
        attn_out = torch.einsum("tnk,nvk->tnv", weighted_kv.float(), w_v.float()).to(q_nope.dtype)
        out.copy_(attn_out)  # single sequence always fills out fully
        return

    # =====================================================================
    # General path: multi-sequence batched prefill (num_seq > 1)
    # =====================================================================
    seq_lengths = seq_len.long()
    seq_start_l = seq_start.long()
    total_tokens = total_num_tokens

    if total_tokens == 0:
        out.zero_()
        return

    # Build per-token index metadata from per-sequence metadata
    token_slots = slot_idx.long().repeat_interleave(seq_lengths)
    base_positions = input_pos.long().repeat_interleave(seq_lengths)
    cum_lengths = torch.zeros(num_seq + 1, device=device, dtype=torch.long)
    cum_lengths[1:] = seq_lengths.cumsum(0)
    base_in_dense = cum_lengths[:-1].repeat_interleave(seq_lengths)
    within_offsets = torch.arange(total_tokens, device=device) - base_in_dense
    token_input_idx = seq_start_l.repeat_interleave(seq_lengths) + within_offsets
    token_cache_pos = base_positions + within_offsets

    # Cache write
    kpe_flat = kpe.index_select(0, token_input_idx).squeeze(1)
    ckv_actual = compressed_kv.index_select(0, token_input_idx)
    ckv_for_cache = ckv_actual.to(cache_dtype) if ckv_actual.dtype != cache_dtype else ckv_actual
    kpe_for_cache = kpe_flat.to(cache_dtype) if kpe_flat.dtype != cache_dtype else kpe_flat
    mla_cache[token_slots, token_cache_pos, :kv_lora_rank] = ckv_for_cache
    mla_cache[token_slots, token_cache_pos, kv_lora_rank:] = kpe_for_cache

    # Weight absorption
    q_nope_actual = q_nope.index_select(0, token_input_idx)
    q_pe_actual = q_pe.index_select(0, token_input_idx).contiguous()
    q_absorbed = torch.einsum("tnd,ndk->tnk", q_nope_actual.float(), w_k_nope.float()).contiguous()

    token_kv_len = (token_cache_pos + 1).to(torch.int32)
    weighted_kv = torch.empty(
        total_tokens, num_heads, kv_lora_rank, device=device, dtype=q_nope.dtype
    )
    seq_block, head_block, nw, ns = _get_mla_multihead_config(total_tokens, is_prefill=True)
    head_block = min(head_block, num_heads)  # iter 59: cap for TP
    grid = (total_tokens, num_heads // head_block)
    _mla_attention_kernel_multihead[grid](
        q_absorbed,
        q_pe_actual,
        mla_cache,
        token_slots,
        token_kv_len,
        weighted_kv,
        SCALE=scale,
        MAX_SEQ_LEN=max_seq_len,
        N_HEADS=num_heads,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        CACHE_DIM=cache_dim,
        KV_BLOCK=kv_block,
        PE_BLOCK=pe_block,
        SEQ_BLOCK=seq_block,
        HEAD_BLOCK=head_block,
        num_warps=nw,
        num_stages=ns,
    )
    attn_out = torch.einsum("tnk,nvk->tnv", weighted_kv.float(), w_v.float()).to(
        q_nope.dtype
    )  # [total_tokens, N, v_head_dim]
    out.zero_()
    out.index_copy_(0, token_input_idx, attn_out)


@torch.library.custom_op("auto_deploy::triton_cached_mla_with_cache", mutates_args=("mla_cache",))
def triton_cached_mla_with_cache(
    # 5 tensor args (get_num_qkv_args = 5)
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank]
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    # Standard metadata
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # Cache (unpaged, same layout as torch_mla)
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    # Constants
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
) -> torch.Tensor:
    """Triton backend MLA with compressed cache.

    Both prefill and decode use Triton-accelerated attention with weight
    absorption and online softmax. No Python loops in either path.

    Cache Layout:
        mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
        - compressed_kv = mla_cache[:, :, :kv_lora_rank]  (zero-copy slice)
        - kpe = mla_cache[:, :, kv_lora_rank:]  (zero-copy slice)
    """
    # Get dimensions
    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Infer v_head_dim from kv_b_proj_weight
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    # Get cleaned up metadata
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]
    input_pos = input_pos[:num_seq]
    slot_idx = slot_idx[:num_seq]
    seq_start = cu_seqlen[:num_seq]

    # Set scale
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Define output shape: [B, S, N, v_head_dim]
    output_shape = (b, s, num_heads, v_head_dim)

    if s == 1:
        # =================================================================
        # Generate phase: Use Triton-accelerated decode with absorption
        # =================================================================
        y = q_nope.new_empty(b, num_heads, v_head_dim).contiguous()

        _triton_mla_decode(
            q_nope,
            q_pe,
            compressed_kv,
            kpe,
            kv_b_proj_weight,
            mla_cache,
            slot_idx,
            input_pos,
            scale,
            kv_lora_rank,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
        )

        return y.unsqueeze(1)  # [B, 1, N, v_head_dim]
    else:
        # =================================================================
        # Context phase: Triton attention with absorption (no Python loops)
        # =================================================================
        bs_view = (b * s,)

        q_nope_flat = q_nope.contiguous().view(*bs_view, num_heads, qk_nope_head_dim)
        q_pe_flat = q_pe.contiguous().view(*bs_view, num_heads, qk_rope_head_dim)
        compressed_kv_flat = compressed_kv.contiguous().view(*bs_view, kv_lora_rank)
        kpe_flat = kpe.contiguous().view(*bs_view, 1, qk_rope_head_dim)

        y = q_nope.new_empty(*bs_view, num_heads, v_head_dim).contiguous()

        _triton_mla_prefill(
            q_nope_flat,
            q_pe_flat,
            compressed_kv_flat,
            kpe_flat,
            kv_b_proj_weight,
            mla_cache,
            input_pos,
            slot_idx,
            seq_len,
            seq_start,
            scale,
            kv_lora_rank,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
            total_num_tokens=num_prefill_tokens + num_decode,
        )

        return y.view(*output_shape)


@triton_cached_mla_with_cache.register_fake
def triton_cached_mla_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    mla_cache: torch.Tensor,
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[-1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    return q_nope.new_empty(
        q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
    ).contiguous()


@AttentionRegistry.register("triton_mla")
class TritonMLAAttention(AttentionDescriptor):
    """Attention descriptor for Triton-based MLA with unpaged cache.

    Uses the same cache layout as torch_mla:
        mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]

    Both prefill and decode use a shared Triton kernel with weight absorption
    + online softmax. Per-token KV length metadata provides causal masking.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 5  # q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_cached_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Get cache initializers using unpaged MLA cache layout."""
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kpe_fake = source_attn_node.args[3].meta["val"]

        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]

        model_dtype = compressed_kv_fake.dtype
        cache_dtype = cls.resolve_cache_dtype(cache_config.dtype, model_dtype)

        return {
            "mla_cache": UnpagedResourceHandler(
                kv_lora_rank + qk_rope_head_dim,
                dtype=cache_dtype,
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        scale = source_attn_node.kwargs.get("scale", None)
        return [scale, kv_lora_rank]
