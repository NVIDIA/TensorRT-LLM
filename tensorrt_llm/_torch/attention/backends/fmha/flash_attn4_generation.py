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

"""Generation-phase attention through FlashAttention-4 (CuTe DSL) on SM90.

Serves the generation rows of a batch, plain decode (one query token per row) and
linear speculative-decoding verification (``q_len = 1 + K`` tokens per row), for dense
GQA/MHA layers with an unquantized bf16/fp16 paged KV cache (KV cache manager V2).
Per layer it

1. applies the op's NeoX RoPE to the packed Q heads in place and writes the step's rotated K
   and its V at ``kv_len - q_len + j`` of every row into the paged pool, in one launch
   (``rope_write_kv_paged``);
2. runs FA4's paged forward over the whole cache (``seqused_k`` = device KV lengths,
   causal, bottom-right aligned), so the step's tokens see each other causally.

The page table is derived on the device from the attention op's block offsets once per step and
shared by the layers of a forward (``fmha_step_caches``), and every
launch reads only fixed-address tensors, so the layer can be captured in a CUDA graph.
Draft layers (they run on a draft KV-cache manager) are served too unless ``TLLM_FLASH_ATTN4_GENERATION_DRAFT=0``.
In a mixed context+generation batch the library serves the generation rows while the regular op serves
the context requests through a context-only call (``CombinedFmha``), so continuous request arrival keeps
the generation rows on FA4.

Opt in with ``TLLM_FMHA_LIBS=+flash_attn4_generation``.
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Callable, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha
from .sampling_shared_prefix import _fused_rope_supported, _rotary_dim

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

LIB_NAME = "flash_attn4_generation"

CHECK_ENV = "TLLM_FLASH_ATTN4_GENERATION_CHECK"
"""Set to 1 to compare the device page table with the host block-offset table on eager steps.
Debugging aid: it synchronizes the device and must stay off for measurements."""

DRAFT_ENV = "TLLM_FLASH_ATTN4_GENERATION_DRAFT"
"""Serve the EAGLE-3 draft layer too (default 1). The draft forward swaps the draft KV-cache manager
and its block offsets into the metadata (``draft_kv_cache_context``), so the page table and pool view
below follow it. On H100 (Qwen2.5-7B, 120 rows, 4k-token prompts) the multi-token XQA kernel took 515 us
(K=3) / 990 us (K=8) for the first draft step and ~260 us for each later one; set to 0 to keep draft
layers on the regular kernel."""


PAGE_TABLE_CACHE_KEY = f"{LIB_NAME}.page_tables"
SPLIT_TABLE_CACHE_KEY = f"{LIB_NAME}.split_tables"
"""Key of the per-step page tables in ``TrtllmAttentionMetadata.fmha_step_caches``."""


MIXED_MIN_KV_ENV = "TLLM_FLASH_ATTN4_GENERATION_MIXED_MIN_KV"
"""Mixed context+generation steps run eagerly, where each FA4 layer call costs ~110 us of host time
(rope, page table, KV write and FA4 launches; H100, Qwen2.5-7B shapes) against a single attention-op
call. The generation rows of a mixed step go to FA4 only when their mean KV length reaches this many
tokens (default 2048), where FA4's kernel time saving outweighs that cost; 0 always uses FA4."""


@functools.lru_cache(maxsize=4)
def _mixed_min_kv(value: Optional[str]) -> int:
    return int(value) if value else 2048


def _draft_enabled() -> bool:
    return os.environ.get(DRAFT_ENV, "1") != "0"


FA4_TILE_MN = (64, 128)
FA4_NUM_THREADS = 256
"""FA4 SM90 tile for generation rows. With GQA packing a row holds ``q_len * heads_per_kv_head``
query rows (28 for Qwen2.5-7B at q_len 4), so a 64-row tile wastes less than the default 128-row
one: 193 vs 282 us per layer for 120 rows over 4.5k cached tokens (Qwen2.5-7B shapes, H100)."""


SPLITS_ENV = "TLLM_FLASH_ATTN4_GENERATION_SPLITS"
"""KV splits of a generation call: unset or 0 picks them from the grid (``generation_splits``),
1 turns splitting off, a larger value fixes the count."""
MAX_SPLITS = 16
MIN_SPLIT_TOKENS = 512
"""Fewest cached tokens per KV split: a split of a short cache costs more in launches and merging
than it saves (H100, Qwen2.5-7B, K=8, 1 row, ~1.6k cached tokens: 16 splits were slower than none)."""


def generation_splits(
    rows: int, num_heads: int, num_kv_heads: int, q_len: int, num_sms: int
) -> int:
    """KV splits of a generation call (flash decoding), fixed per batch shape.

    FA4 launches one CTA per (row, KV head, 64-row tile of the row's packed GQA queries) and
    streams the row's whole cache in it; SM90 FA4 has no split-KV. A small batch then leaves
    most SMs idle while each CTA walks thousands of keys: on H100 (Qwen2.5-7B, K=8, 1-4 rows,
    ~5k cached tokens) 16 CTAs took ~65 us per layer. When the grid fills less than half the
    SMs, the row's cache is cut into up to one wave's worth of splits (at most ``MAX_SPLITS``).
    """
    fixed = os.environ.get(SPLITS_ENV, "").strip()
    if fixed and int(fixed) > 0:
        return int(fixed)
    gqa = num_heads // num_kv_heads
    if gqa > 1:
        ctas = rows * num_kv_heads * -(-(q_len * gqa) // FA4_TILE_MN[0])
    else:
        ctas = rows * num_heads * -(-q_len // FA4_TILE_MN[0])
    if 2 * ctas > num_sms:
        return 1
    return max(1, min(MAX_SPLITS, -(-num_sms // ctas)))


@functools.lru_cache(maxsize=8)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=1)
def _fa4_forward() -> Optional[Callable[..., tuple]]:
    """FA4's forward, or None when the CuTe DSL interface cannot be imported."""
    try:
        from tensorrt_llm._torch.cute_dsl_utils import install_cutlass_dsl_compatibility

        install_cutlass_dsl_compatibility()
        from flash_attn.cute.interface import _flash_attn_fwd
    except (ImportError, OSError, AttributeError) as error:
        logger.info(f"{LIB_NAME}: FlashAttention-4 is unavailable ({error}).")
        return None
    return _flash_attn_fwd


@triton.jit
def _write_kv_paged_kernel(
    k_ptr,
    v_ptr,
    kv_row_stride,
    pool_ptr,
    pool_page_stride,
    pool_kv_stride,
    pool_head_stride,
    pool_slot_stride,
    page_table_ptr,
    page_table_stride,
    kv_lens_ptr,
    Q_LEN: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV_HEADS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    # Program t writes token t = row * Q_LEN + j at position kv_len - Q_LEN + j of its row.
    t = tl.program_id(0)
    row = t // Q_LEN
    j = t % Q_LEN
    pos = tl.load(kv_lens_ptr + row).to(tl.int64) - Q_LEN + j
    valid = pos >= 0
    pos = tl.maximum(pos, 0)
    page = tl.load(page_table_ptr + row.to(tl.int64) * page_table_stride + pos // TOKENS_PER_BLOCK)
    heads = tl.arange(0, BLOCK_KV_HEADS)
    dims = tl.arange(0, HEAD_DIM)
    mask = (heads[:, None] < NUM_KV_HEADS) & valid
    src = t.to(tl.int64) * kv_row_stride + heads[:, None] * HEAD_DIM + dims[None, :]
    dst = (
        pool_ptr
        + page.to(tl.int64) * pool_page_stride
        + heads[:, None].to(tl.int64) * pool_head_stride
        + (pos % TOKENS_PER_BLOCK) * pool_slot_stride
        + dims[None, :]
    )
    tl.store(dst, tl.load(k_ptr + src, mask=mask), mask=mask)
    tl.store(dst + pool_kv_stride, tl.load(v_ptr + src, mask=mask), mask=mask)


@triton.jit
def _rope_write_kv_paged_kernel(
    qkv_ptr,
    qkv_row_stride,
    cos_sin_ptr,
    pool_ptr,
    pool_page_stride,
    pool_kv_stride,
    pool_head_stride,
    pool_slot_stride,
    page_table_ptr,
    page_table_stride,
    kv_lens_ptr,
    page_lens_ptr,
    Q_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_Q_HEADS: tl.constexpr,
    BLOCK_KV_HEADS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HAS_PAGE_LENS: tl.constexpr,
):
    # Program t takes token t = row * Q_LEN + j at position kv_len - Q_LEN + j of its row:
    # rotates its Q heads in place (NeoX pairs (d, d + HALF), d < HALF) and writes its rotated
    # K and its V into the row's page, at slot page_len - Q_LEN + j of the row's page table
    # (page_len = kv_len unless the table starts later in the row).
    t = tl.program_id(0)
    row = t // Q_LEN
    j = t % Q_LEN
    pos = tl.load(kv_lens_ptr + row).to(tl.int64) - Q_LEN + j
    pos = tl.maximum(pos, 0)
    if HAS_PAGE_LENS:
        slot = tl.load(page_lens_ptr + row).to(tl.int64) - Q_LEN + j
    else:
        slot = tl.load(kv_lens_ptr + row).to(tl.int64) - Q_LEN + j
    valid = slot >= 0
    slot = tl.maximum(slot, 0)
    base = qkv_ptr + t.to(tl.int64) * qkv_row_stride
    kv_heads = tl.arange(0, BLOCK_KV_HEADS)
    kv_ok = kv_heads[:, None] < NUM_KV_HEADS
    dims = tl.arange(0, HEAD_DIM)
    k_ptrs = base + NUM_HEADS * HEAD_DIM + kv_heads[:, None] * HEAD_DIM + dims[None, :]
    if HAS_ROPE:
        pairs = tl.arange(0, HALF)
        cos = tl.load(cos_sin_ptr + pos * (2 * HALF) + 2 * pairs)
        sin = tl.load(cos_sin_ptr + pos * (2 * HALF) + 2 * pairs + 1)
        q_heads = tl.arange(0, BLOCK_Q_HEADS)
        q_ok = q_heads[:, None] < NUM_HEADS
        q1_ptrs = base + q_heads[:, None] * HEAD_DIM + pairs[None, :]
        q1 = tl.load(q1_ptrs, mask=q_ok, other=0.0)
        q2 = tl.load(q1_ptrs + HALF, mask=q_ok, other=0.0)
        f1 = q1.to(tl.float32)
        f2 = q2.to(tl.float32)
        tl.store(q1_ptrs, (f1 * cos[None, :] - f2 * sin[None, :]).to(q1.dtype), mask=q_ok)
        tl.store(q1_ptrs + HALF, (f2 * cos[None, :] + f1 * sin[None, :]).to(q1.dtype), mask=q_ok)
    k = tl.load(k_ptrs, mask=kv_ok, other=0.0)
    v = tl.load(k_ptrs + NUM_KV_HEADS * HEAD_DIM, mask=kv_ok, other=0.0)
    if HAS_ROPE:
        # K goes to the pool only (qkv keeps the unrotated K): dimension d < 2 * HALF pairs
        # with d +- HALF, the rest of the head is copied as is.
        rot = dims < 2 * HALF
        low = dims < HALF
        partner = tl.where(low, dims + HALF, dims - HALF)
        kp = tl.load(
            base + NUM_HEADS * HEAD_DIM + kv_heads[:, None] * HEAD_DIM + partner[None, :],
            mask=kv_ok & rot[None, :],
            other=0.0,
        ).to(tl.float32)
        pair = tl.where(low, dims, dims - HALF) % HALF
        cos_d = tl.load(cos_sin_ptr + pos * (2 * HALF) + 2 * pair, mask=rot, other=1.0)
        sin_d = tl.load(cos_sin_ptr + pos * (2 * HALF) + 2 * pair + 1, mask=rot, other=0.0)
        sign = tl.where(low, -1.0, 1.0)
        k_rot = k.to(tl.float32) * cos_d[None, :] + sign[None, :] * kp * sin_d[None, :]
        k = tl.where(rot[None, :], k_rot, k.to(tl.float32)).to(v.dtype)
    page = tl.load(page_table_ptr + row.to(tl.int64) * page_table_stride + slot // TOKENS_PER_BLOCK)
    dst = (
        pool_ptr
        + page.to(tl.int64) * pool_page_stride
        + kv_heads[:, None].to(tl.int64) * pool_head_stride
        + (slot % TOKENS_PER_BLOCK) * pool_slot_stride
        + dims[None, :]
    )
    tl.store(dst, k, mask=kv_ok & valid)
    tl.store(dst + pool_kv_stride, v, mask=kv_ok & valid)


def rope_write_kv_paged(
    qkv: torch.Tensor,
    cos_sin: Optional[torch.Tensor],
    pool: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    q_len: int,
    page_lens: Optional[torch.Tensor] = None,
) -> None:
    """One launch: NeoX RoPE on the Q heads of packed QKV (in place) and the step's rotated K and
    its V into the paged pool; the pool and Q end as after NeoX RoPE on Q and K followed by
    ``write_kv_paged`` (only the K columns of ``qkv`` stay unrotated).

    Args:
        qkv: [rows * q_len, (heads + 2 * kv_heads) * head_dim], columns contiguous.
        cos_sin: the attention op's float32 [position][pair][cos, sin] table, or None for no RoPE.
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND).
        page_table: [rows, max_pages] int32 pool pages of every row.
        kv_lens: [rows] int32 device KV lengths including the step's ``q_len`` tokens (they
            give the RoPE positions).
        rotary_dim: rotated leading dimensions of each head (a power of two, at most head_dim).
        page_lens: [rows] int32 lengths within ``page_table`` when the table covers only the
            end of each row (its first entry is not the row's first page); default ``kv_lens``.
    """
    tokens = qkv.shape[0]
    if tokens == 0:
        return
    if qkv.stride(1) != 1:
        raise ValueError(f"qkv columns must be contiguous, got strides {qkv.stride()}.")
    if pool.stride(4) != 1:
        raise ValueError(f"pool must be contiguous in head_dim, got strides {pool.stride()}.")
    half = rotary_dim // 2
    if cos_sin is not None and (rotary_dim % 2 or half & (half - 1) or rotary_dim > head_dim):
        raise ValueError(f"unsupported rotary dim {rotary_dim} for head dim {head_dim}.")
    _rope_write_kv_paged_kernel[(tokens,)](
        qkv,
        qkv.stride(0),
        cos_sin if cos_sin is not None else kv_lens,
        pool,
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        page_table,
        page_table.stride(0),
        kv_lens,
        page_lens if page_lens is not None else kv_lens,
        Q_LEN=q_len,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HALF=max(half, 1),
        BLOCK_Q_HEADS=triton.next_power_of_2(num_heads),
        BLOCK_KV_HEADS=triton.next_power_of_2(num_kv_heads),
        TOKENS_PER_BLOCK=pool.shape[3],
        HAS_ROPE=cos_sin is not None,
        HAS_PAGE_LENS=page_lens is not None,
    )


def write_kv_paged(
    pool: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
    q_len: int,
) -> None:
    """Write the step's K/V of every row into its pages.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND).
        k, v: [rows * q_len, kv_heads, head_dim], row major; each token's heads contiguous.
        page_table: [rows, max_pages] int32 pool pages of every row.
        kv_lens: [rows] int32 device KV lengths including the step's ``q_len`` tokens.
    """
    tokens, num_kv_heads, head_dim = k.shape
    if tokens == 0:
        return
    if k.stride(2) != 1 or k.stride(1) != head_dim or v.stride() != k.stride():
        raise ValueError(
            f"k and v must share contiguous-head strides, got {k.stride()} / {v.stride()}."
        )
    if pool.stride(4) != 1:
        raise ValueError(f"pool must be contiguous in head_dim, got strides {pool.stride()}.")
    _write_kv_paged_kernel[(tokens,)](
        k,
        v,
        k.stride(0),
        pool,
        pool.stride(0),
        pool.stride(1),
        pool.stride(2),
        pool.stride(3),
        page_table,
        page_table.stride(0),
        kv_lens,
        Q_LEN=q_len,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV_HEADS=triton.next_power_of_2(num_kv_heads),
        TOKENS_PER_BLOCK=pool.shape[3],
    )


@triton.jit
def _split_tables_kernel(
    page_table_ptr,
    page_table_stride,
    kv_lens_ptr,
    split_table_ptr,
    split_table_stride,
    split_lens_ptr,
    last_table_ptr,
    last_table_stride,
    last_lens_ptr,
    width,
    Q_LEN: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK_WIDTH: tl.constexpr,
    LAST_WIDTH: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    MIN_CHUNK_PAGES: tl.constexpr,
):
    # Program (r, c): row r's cached pages before the page of its first step token,
    # [0, p0), cut into SPLITS chunks of up to ceil(p0 / SPLITS) pages (c < SPLITS), or the
    # row's last part, pages [p0, ...) holding the step's tokens (c == SPLITS).
    r = tl.program_id(0)
    c = tl.program_id(1)
    kv_len = tl.load(kv_lens_ptr + r)
    p0 = tl.maximum(kv_len - Q_LEN, 0) // TOKENS_PER_BLOCK
    offs = tl.arange(0, BLOCK)
    src_row = page_table_ptr + r.to(tl.int64) * page_table_stride
    if c < SPLITS:
        # Chunks of at least MIN_CHUNK_PAGES: a short cache uses fewer (non-empty) splits.
        per = tl.maximum((p0 + SPLITS - 1) // SPLITS, MIN_CHUNK_PAGES)
        start = c * per
        count = tl.minimum(tl.maximum(p0 - start, 0), per)
        dst_row = split_table_ptr + (r * SPLITS + c).to(tl.int64) * split_table_stride
        for p0_ in range(0, CHUNK_WIDTH, BLOCK):
            p = p0_ + offs
            page = tl.load(src_row + start + p, mask=(p < count) & (start + p < width), other=0)
            tl.store(dst_row + p, page, mask=p < CHUNK_WIDTH)
        tl.store(split_lens_ptr + r * SPLITS + c, (count * TOKENS_PER_BLOCK).to(tl.int32))
    else:
        tail = kv_len - p0 * TOKENS_PER_BLOCK
        count = (tail + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
        p = tl.arange(0, LAST_WIDTH)
        page = tl.load(src_row + p0 + p, mask=(p < count) & (p0 + p < width), other=0)
        tl.store(last_table_ptr + r.to(tl.int64) * last_table_stride + p, page)
        tl.store(last_lens_ptr + r, tail.to(tl.int32))


@triton.jit
def _merge_splits_kernel(
    split_out_ptr,
    split_lse_ptr,
    split_lens_ptr,
    last_out_ptr,
    last_lse_ptr,
    out_ptr,
    out_token_stride,
    lse_out_ptr,
    H: tl.constexpr,
    HP: tl.constexpr,
    D: tl.constexpr,
    Q_LEN: tl.constexpr,
    SPLITS: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    # HP: heads per program (program_id(1) picks the block of heads).
    # Program t = r * Q_LEN + j joins the SPLITS non-causal states of row r (skipping empty
    # splits) and its causal last part; FA4's states are normalized outputs with natural-log
    # LSEs laid out [batch, heads, Q_LEN].
    LOG2E: tl.constexpr = 1.4426950408889634
    t = tl.program_id(0)
    r = t // Q_LEN
    j = t % Q_LEN
    heads = tl.program_id(1) * HP + tl.arange(0, HP)
    h_ok = heads < H
    dims = tl.arange(0, D)
    lse = tl.load(last_lse_ptr + (r * H + heads) * Q_LEN + j, mask=h_ok, other=0.0) * LOG2E
    peak = lse
    for c in tl.static_range(SPLITS):
        b = r * SPLITS + c
        ok = tl.load(split_lens_ptr + b) > 0
        lc = tl.load(split_lse_ptr + (b * H + heads) * Q_LEN + j, mask=h_ok & ok, other=-1.0e30)
        peak = tl.maximum(peak, tl.where(ok, lc * LOG2E, -1.0e30))
    w = tl.exp2(lse - peak)
    l_tot = w
    acc = w[:, None] * tl.load(
        last_out_ptr + (t.to(tl.int64) * H + heads[:, None]) * D + dims[None, :],
        mask=h_ok[:, None],
        other=0.0,
    ).to(tl.float32)
    for c in tl.static_range(SPLITS):
        b = r * SPLITS + c
        ok = tl.load(split_lens_ptr + b) > 0
        lc = tl.load(split_lse_ptr + (b * H + heads) * Q_LEN + j, mask=h_ok & ok, other=-1.0e30)
        wc = tl.where(ok, tl.exp2(lc * LOG2E - peak), 0.0)
        oc = tl.load(
            split_out_ptr + ((b * Q_LEN + j).to(tl.int64) * H + heads[:, None]) * D + dims[None, :],
            mask=h_ok[:, None] & ok,
            other=0.0,
        ).to(tl.float32)
        l_tot += wc
        acc += wc[:, None] * oc
    tl.store(
        out_ptr + t.to(tl.int64) * out_token_stride + heads[:, None] * D + dims[None, :],
        (acc / l_tot[:, None]).to(out_ptr.dtype.element_ty),
        mask=h_ok[:, None],
    )
    if WRITE_LSE:
        # Natural-log LSE in FA4's [rows, heads, Q_LEN] layout.
        tl.store(
            lse_out_ptr + (r * H + heads) * Q_LEN + j,
            (peak + tl.log2(l_tot)) / LOG2E,
            mask=h_ok,
        )


def split_tables(
    page_table: torch.Tensor, kv_lens: torch.Tensor, q_len: int, splits: int, tokens_per_block: int
) -> tuple:
    """Page tables and KV lengths of a split generation call (see ``_split_tables_kernel``):
    ``(split_table [rows * splits, chunk], split_lens, last_table [rows, last], last_lens)``."""
    rows, width = page_table.shape
    min_pages = max(1, MIN_SPLIT_TOKENS // tokens_per_block)
    # A chunk holds ceil(cached pages / splits) pages, but at least min_pages.
    chunk = min(width, max(1, -(-width // splits), min_pages))
    last = triton.next_power_of_2(-(-(tokens_per_block - 1 + q_len) // tokens_per_block))
    device = page_table.device
    split_table = torch.empty((rows * splits, chunk), dtype=torch.int32, device=device)
    split_lens = torch.empty((rows * splits,), dtype=torch.int32, device=device)
    last_table = torch.empty((rows, last), dtype=torch.int32, device=device)
    last_lens = torch.empty((rows,), dtype=torch.int32, device=device)
    _split_tables_kernel[(rows, splits + 1)](
        page_table, page_table.stride(0), kv_lens, split_table, split_table.stride(0), split_lens,
        last_table, last_table.stride(0), last_lens, width, Q_LEN=q_len, SPLITS=splits,
        CHUNK_WIDTH=chunk, LAST_WIDTH=last, TOKENS_PER_BLOCK=tokens_per_block, BLOCK=128,
        MIN_CHUNK_PAGES=min_pages,
    )  # fmt: skip
    return split_table, split_lens, last_table, last_lens


_split_buffers: dict = {}


def _split_workspace(
    rows: int, splits: int, q_len: int, num_heads: int, head_dim: int, dtype, device
) -> tuple:
    """Fixed-address partial-state buffers of a split call shape, shared by the layers:
    ``(q_rep, split_out, split_lse, last_out, last_lse)``."""
    key = (device, dtype, rows, splits, q_len, num_heads, head_dim)
    bufs = _split_buffers.get(key)
    if bufs is None:
        n = rows * splits
        # Normal (not inference) tensors: the buffers outlive the call that creates them.
        with torch.inference_mode(False):
            bufs = _split_buffers[key] = (
                torch.empty((n, q_len, num_heads, head_dim), dtype=dtype, device=device),
                torch.empty((n, q_len, num_heads, head_dim), dtype=dtype, device=device),
                torch.empty((n, num_heads, q_len), dtype=torch.float32, device=device),
                torch.empty((rows, q_len, num_heads, head_dim), dtype=dtype, device=device),
                torch.empty((rows, num_heads, q_len), dtype=torch.float32, device=device),
            )
    return bufs


def fa4_paged_generation_split(
    pool: torch.Tensor,
    q: torch.Tensor,
    out: torch.Tensor,
    tables: tuple,
    q_len: int,
    splits: int,
    sm_scale: float,
    lse: Optional[torch.Tensor] = None,
) -> None:
    """FA4 generation with the cached keys split ``splits`` ways (flash decoding).

    Every row's pages before the page of its first step token are attended in ``splits``
    non-causal chunks (one FA4 call over ``rows * splits`` batch entries, the queries
    replicated), the rest of its cache, which holds the step's tokens, in one causal call;
    a merge kernel joins the states. The step's K/V must already be in the pool.

    Args:
        q: [rows * q_len, heads, head_dim] (rotary embedding applied).
        out: [rows * q_len, heads, head_dim] output.
        tables: ``split_tables(page_table, kv_lens, q_len, splits, tokens_per_block)``.
        lse: optional [rows, heads, q_len] float32 output, the natural-log LSE (FA4's layout).
    """
    fwd = _fa4_forward()
    if fwd is None:
        raise RuntimeError(f"{LIB_NAME}: FlashAttention-4 is unavailable.")
    tokens, num_heads, head_dim = q.shape
    rows = tokens // q_len
    split_table, split_lens, last_table, last_lens = tables
    q_rep, split_out, split_lse, last_out, last_lse = _split_workspace(
        rows, splits, q_len, num_heads, head_dim, q.dtype, q.device
    )
    q_rep.view(rows, splits, q_len, num_heads, head_dim).copy_(
        q.view(rows, 1, q_len, num_heads, head_dim).expand(rows, splits, q_len, num_heads, head_dim)
    )
    k_pages = pool[:, 0].transpose(1, 2)
    v_pages = pool[:, 1].transpose(1, 2)
    common = dict(
        softmax_scale=sm_scale,
        tile_mn=FA4_TILE_MN,
        num_threads=FA4_NUM_THREADS,
        num_splits=1,
        pack_gqa=None,
        return_lse=True,
    )
    fwd(q_rep, k_pages, v_pages, seqused_k=split_lens, page_table=split_table, causal=False,
        out=split_out, lse=split_lse, **common)  # fmt: skip
    fwd(q.view(rows, q_len, num_heads, head_dim), k_pages, v_pages, seqused_k=last_lens,
        page_table=last_table, causal=True, out=last_out, lse=last_lse, **common)  # fmt: skip
    # A few heads per program: a small batch has few tokens, and each program walks every split.
    heads_per_program = 4
    _merge_splits_kernel[(tokens, triton.cdiv(num_heads, heads_per_program))](
        split_out, split_lse, split_lens, last_out, last_lse, out, out.stride(0),
        lse if lse is not None else last_lse,
        H=num_heads, HP=heads_per_program, D=head_dim, Q_LEN=q_len,
        SPLITS=splits, WRITE_LSE=lse is not None, num_warps=2,
    )  # fmt: skip


def fa4_paged_generation(
    pool: torch.Tensor,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    out: torch.Tensor,
    page_table: torch.Tensor,
    kv_lens: torch.Tensor,
    q_len: int,
    sm_scale: float,
) -> None:
    """Persist the step's K/V and attend every row's cache with FA4.

    Args:
        pool: [pages, 2, kv_heads, tokens_per_block, head_dim] layer pool (HND), bf16/fp16.
        q: [rows * q_len, heads, head_dim]; k, v: [rows * q_len, kv_heads, head_dim], row major
            (rotary embedding already applied), or None when the step's K/V is already in the pool.
        out: [rows * q_len, heads, head_dim] output.
        page_table: [rows, max_pages] int32 pool pages of every row.
        kv_lens: [rows] int32 device KV lengths including the step's ``q_len`` tokens.
    """
    fwd = _fa4_forward()
    if fwd is None:
        raise RuntimeError(f"{LIB_NAME}: FlashAttention-4 is unavailable.")
    tokens, num_heads, head_dim = q.shape
    rows = tokens // q_len
    if k is not None:
        write_kv_paged(pool, k, v, page_table, kv_lens, q_len)
    # FA4 takes NHD pages [pages, page_size, kv_heads, head_dim]; the HND pool transposes without a copy.
    k_pages = pool[:, 0].transpose(1, 2)
    v_pages = pool[:, 1].transpose(1, 2)
    fwd(
        q.view(rows, q_len, num_heads, head_dim),
        k_pages,
        v_pages,
        seqused_k=kv_lens,
        page_table=page_table,
        softmax_scale=sm_scale,
        causal=True,
        tile_mn=FA4_TILE_MN,
        num_threads=FA4_NUM_THREADS,
        num_splits=1,
        pack_gqa=None,
        out=out.view(rows, q_len, num_heads, head_dim),
        return_lse=False,
    )


class FlashAttn4GenerationFmha(PhasedFmha):
    """Generation-phase library on FlashAttention-4; see the module docstring."""

    # Mixed context+generation batches: the thop op serves the context requests through a
    # context-only call and this library the generation rows (``CombinedFmha``).
    pairs_with_whole_batch_context = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._pool_views: dict[int, torch.Tensor] = {}
        self._pool_manager: Optional[object] = None
        self._logged = False

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if get_sm_version() != 90 or _fa4_forward() is None:
            return False
        if (
            getattr(attn, "is_mla_enable", False)
            or getattr(attn, "sparse_params", None) is not None
        ):
            return False
        if getattr(attn, "predicted_tokens_per_seq", 1) != 1 or getattr(
            attn, "attention_chunk_size", None
        ):
            return False
        head_dim = getattr(attn, "head_dim", 0)
        num_heads = getattr(attn, "num_heads", 0)
        num_kv_heads = getattr(attn, "num_kv_heads", 0) or 0
        if head_dim not in (64, 128) or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            return False
        if QuantMode(getattr(attn, "quant_mode", 0)).has_kv_cache_quant():
            return False
        position_type = PositionEmbeddingType(getattr(attn, "position_embedding_type", 0))
        if position_type.is_rope():
            return _fused_rope_supported(attn)
        # ALiBi, relative and deferred embeddings are applied inside the attention op.
        return position_type == PositionEmbeddingType.learned_absolute

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        if phase == FmhaPhase.CONTEXT:
            return False
        if forward_args.attention_input_type == AttentionInputType.context_only:
            return False
        if metadata.num_generations == 0:
            return False
        if phase is None and metadata.num_contexts > 0:
            # Mixed batches are served phase by phase (CombinedFmha).
            return False
        manager = metadata.kv_cache_manager
        if (
            manager is None
            or not hasattr(manager, "index_mapper")
            or (getattr(manager, "is_draft", False) and not _draft_enabled())
            or metadata.kv_cache_block_offsets is None
            # Per-layer page tables are indexed by local layer, and two layers of one lifecycle
            # group can map different pages; ``_page_table`` reads (and shares) the table of the
            # lifecycle group.
            or getattr(manager, "_use_per_layer_page_tables", False)
        ):
            return False
        if (
            metadata.beam_width != 1
            or metadata.is_cross
            or metadata.enable_helix
            or metadata.is_spec_dec_tree
            or metadata.is_spec_dec_dynamic_tree
        ):
            return False
        if not (
            forward_args.is_fused_qkv and k is None and v is None and forward_args.update_kv_cache
        ):
            return False
        if forward_args.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False
        if (
            forward_args.attention_mask_data is not None
            or forward_args.attention_sinks is not None
            or forward_args.relative_attention_bias is not None
            or forward_args.out_scale is not None
            or forward_args.output_sf is not None
        ):
            return False
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        window = forward_args.attention_window_size
        if (
            window is not None
            and metadata.max_seq_len is not None
            and window < metadata.max_seq_len
        ):
            return False
        if PositionEmbeddingType(getattr(self.attn, "position_embedding_type", 0)).is_rope():
            max_seq_len = metadata.max_seq_len
            if max_seq_len is None or max_seq_len > (
                self.attn.rotary_cos_sin.numel() // _rotary_dim(self.attn)
            ):
                return False
        return True

    def serves_mixed_generation_rows(self, metadata: "TrtllmAttentionMetadata") -> bool:
        min_kv = _mixed_min_kv(os.environ.get(MIXED_MIN_KV_ENV))
        if min_kv <= 0:
            return True
        total_kv = getattr(metadata, "host_total_kv_lens", None)
        if total_kv is None or metadata.num_generations <= 0:
            return False
        return int(total_kv[1]) >= min_kv * metadata.num_generations

    def _pool_view(self, metadata: "TrtllmAttentionMetadata") -> torch.Tensor:
        manager = metadata.kv_cache_manager
        if self._pool_manager is not manager:
            # The executor replaces the manager after memory estimation.
            self._pool_views.clear()
            self._pool_manager = manager
        layer_idx = self.attn.layer_idx
        pool = self._pool_views.get(layer_idx)
        if pool is None:
            pool = manager.get_buffers(layer_idx, kv_layout="HND")
            if pool is None:
                raise RuntimeError(f"{LIB_NAME}: layer {layer_idx} has no KV pool.")
            self._pool_views[layer_idx] = pool
        return pool

    def _page_table(self, params: FmhaParams) -> torch.Tensor:
        """[rows, max_pages] int32 pool pages of the phase's rows, derived on the device.

        The attention op's K block offset of a page is ``page * kv_factor`` in the
        layer's pool view, which folds K and V into one page. The layers of a forward share
        one table per (KV manager, pool, rows): the first layer to ask for it after the
        metadata's ``prepare`` derives it in every forward of that step (so a captured graph
        records the derivation), and the later layers reuse it.
        """
        metadata = params.meta
        manager = metadata.kv_cache_manager
        pool_index = int(
            manager.layer_to_pool_mapping_dict[manager.layer_offsets[self.attn.layer_idx]]
        )
        rows = slice(params.seq_offset, params.seq_offset + params.batch_size)
        offsets = metadata.kv_cache_block_offsets[pool_index, rows, 0, :]
        step_caches = getattr(metadata, "fmha_step_caches", None)
        tables = None if step_caches is None else step_caches.setdefault(PAGE_TABLE_CACHE_KEY, {})
        key = (
            id(manager),
            pool_index,
            params.seq_offset,
            params.batch_size,
            metadata.kv_cache_block_offsets.data_ptr(),
        )
        entry = None if tables is None else tables.get(key)
        if entry is not None and entry[0] != self.attn.layer_idx:
            return entry[1]
        table = torch.div(offsets, int(manager.kv_factor), rounding_mode="floor")
        if tables is not None:
            tables[key] = (self.attn.layer_idx, table)
        if os.environ.get(CHECK_ENV, "0") == "1" and not torch.cuda.is_current_stream_capturing():
            self._check_page_table(params, table)
        return table

    def _split_tables(
        self,
        params: FmhaParams,
        page_table: torch.Tensor,
        kv_lens: torch.Tensor,
        q_len: int,
        splits: int,
        tokens_per_block: int,
    ) -> tuple:
        """``split_tables`` of the phase's rows, shared per step like the page table."""
        metadata = params.meta
        step_caches = getattr(metadata, "fmha_step_caches", None)
        cache = None if step_caches is None else step_caches.setdefault(SPLIT_TABLE_CACHE_KEY, {})
        key = (page_table.data_ptr(), kv_lens.data_ptr(), params.batch_size, q_len, splits)
        entry = None if cache is None else cache.get(key)
        if entry is not None and entry[0] != self.attn.layer_idx:
            return entry[1]
        tables = split_tables(page_table, kv_lens, q_len, splits, tokens_per_block)
        if cache is not None:
            cache[key] = (self.attn.layer_idx, tables)
        return tables

    def _check_page_table(self, params: FmhaParams, table: torch.Tensor) -> None:
        """Compare with the host table read the way the n-sample library reads it."""
        from .sampling_shared_prefix import _page_layout

        metadata = params.meta
        manager = metadata.kv_cache_manager
        pool_index, scale, divisor = _page_layout(metadata, self.attn.layer_idx)
        request_ids = metadata.request_ids
        beam_stride = int(manager.max_beam_width)
        lens = metadata.kv_lens_runtime[params.seq_offset : params.seq_offset + params.batch_size]
        host = metadata.host_kv_cache_block_offsets
        device = table.cpu()
        tpb = int(metadata.tokens_per_block)
        for i in range(params.batch_size):
            slot = manager.index_mapper.get_index(request_ids[params.seq_offset + i]) * beam_stride
            pages = (int(lens[i]) + tpb - 1) // tpb
            expect = host[pool_index, slot, 0, :pages].long() * scale // divisor
            if not torch.equal(device[i, :pages].long(), expect):
                raise RuntimeError(
                    f"{LIB_NAME} check: device page table of row {i} disagrees with the host table "
                    f"(layer {self.attn.layer_idx}): {device[i, :pages].tolist()[:8]} vs {expect.tolist()[:8]}"
                )

    def run_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        qkv = params.qkv_input
        output = params.output
        if qkv is None or output is None:
            raise RuntimeError(f"{LIB_NAME} requires packed QKV input and an output tensor.")
        rows = params.batch_size
        tokens = params.num_tokens
        if rows <= 0 or tokens % rows:
            raise RuntimeError(
                f"{LIB_NAME}: {tokens} generation tokens are not uniform over {rows} rows."
            )
        q_len = tokens // rows
        if not self._logged:
            self._logged = True
            logger.info_once(
                f"{LIB_NAME}: first generation call ({rows} rows x {q_len} tokens)",
                key=f"{LIB_NAME}_first_call",
            )
        num_heads, num_kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        kv_lens = params.meta.kv_lens_cuda_runtime[params.seq_offset : params.seq_offset + rows]
        rope = PositionEmbeddingType(attn.position_embedding_type).is_rope()
        pool = self._pool_view(params.meta)
        page_table = self._page_table(params)
        # RoPE on Q (in place) and K, and the step's K/V into the pool, in one launch.
        rope_write_kv_paged(
            qkv,
            attn.rotary_cos_sin if rope else None,
            pool,
            page_table,
            kv_lens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=_rotary_dim(attn) if rope else head_dim,
            q_len=q_len,
        )
        q_size = num_heads * head_dim
        q = qkv[:, :q_size].reshape(tokens, num_heads, head_dim)
        sm_scale = 1.0 / (head_dim**0.5 * attn.q_scaling)
        splits = generation_splits(
            rows, num_heads, num_kv_heads, q_len, _num_sms(qkv.device.index or 0)
        )
        if splits > 1:
            fa4_paged_generation_split(
                pool,
                q,
                output.view(tokens, num_heads, head_dim),
                self._split_tables(params, page_table, kv_lens, q_len, splits, pool.shape[3]),
                q_len,
                splits,
                sm_scale,
            )
            return
        fa4_paged_generation(
            pool,
            q,
            None,
            None,
            output.view(tokens, num_heads, head_dim),
            page_table,
            kv_lens,
            q_len,
            sm_scale,
        )


__all__ = [
    "FlashAttn4GenerationFmha",
    "LIB_NAME",
    "fa4_paged_generation",
    "fa4_paged_generation_split",
    "generation_splits",
    "rope_write_kv_paged",
    "split_tables",
    "write_kv_paged",
]
