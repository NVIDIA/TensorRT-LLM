# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-step attention over the FP8 paged KV cache, KV append included, in two launches.

The built-in decode path of this target's attention layers spends two launches
per layer as well: ``applyBiasRopeUpdateKVCacheV2`` writes the step's K/V into
the paged FP8 cache and quantizes Q to E4M3, then the trtllm-gen generation
FMHA attends. For a sliding-window layer that FMHA is latency-bound: one
64-row Q tile per KV head, the 2048-key window cut into 5 KV splits, 10 CTAs on
the whole GPU and an in-kernel reduction for half a megabyte of K/V.

This entry reads the step's Q and new K/V straight from the BF16 fused-QKV
operand. The split kernel cuts the attended key range into ``num_splits``
contiguous 32-key blocks: keys before the step come from the paged cache,
the step's own keys are converted in registers exactly as the cache write
converts them, so no program waits on another's write. One program per KV
head also writes the step's K/V into the cache. Each program covers 32 query
rows (4 tokens x 8 query heads of one KV head), which keeps Triton on
register-accumulator ``mma.sync`` rather than tcgen05 with a TMEM round trip per
key block. The combine kernel merges the splits' partial softmax states.

The cache write is bit-identical to ``applyBiasRopeUpdateKVCacheV2``:
``e4m3_rn_satfinite(bf16(x * bf16(scale)))`` at
``pool[block_table[kv][pos // P], kv_head, pos % P, :]``. The attention output
is not bit-identical to the trtllm-gen kernel: Q stays BF16 (or, with
``q_fp8``, is quantized to E4M3 exactly as the built-in path does), S and P*V
are BF16 MMAs with fp32 accumulation, and the split order differs.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: Head width the row tile and the 64-element vectors are written for.
HEAD_DIM = 64
#: Query heads per KV head the row tile is laid out for (8 tokens x 8 heads = 64 rows).
Q_PER_KV = 8
#: Query rows a decode step may carry: one accepted token and up to 7 drafts.
MAX_ROWS = 8
#: Keys per inner block and query rows per program, sized for the 2048-key window.
BLOCK_N = 32
ROWS_PER_PROGRAM = 32
NUM_WARPS = 4
NUM_STAGES = 2


@triton.jit
def _to_cache_fp8(x, scale_bf16):
    """``store_8bits_vec``: a BF16 multiply by the BF16 scale, then E4M3 RN satfinite."""
    y = (x.to(tl.float32) * scale_bf16.to(tl.float32)).to(tl.bfloat16)
    return y.to(tl.float32).to(tl.float8e4nv)


@triton.jit
def _paged_decode_split_kernel(
    qkv_ptr,
    qkv_stride,
    pool_ptr,
    bt_ptr,
    bt_stride_kv,
    seq_len_ptr,
    scale_ptr,
    part_o_ptr,
    part_ml_ptr,
    sm_scale_log2,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    Q_FP8: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    G: tl.constexpr = HQ // HKV
    PAGE: tl.constexpr = HKV * P * D
    split = tl.program_id(0)
    h = tl.program_id(1)
    row_group = tl.program_id(2)
    if LAUNCH_WITH_PDL:
        # Before any global read: the fused-QKV operand is the previous kernel's
        # output. The combine kernel may launch now and waits for this grid.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    L = tl.load(seq_len_ptr)
    past = L - M
    if WINDOW > 0:
        lo = tl.maximum(past - WINDOW + 1, 0)
    else:
        lo = 0
    base = (lo // BLOCK) * BLOCK
    n_blocks = tl.cdiv(L - base, BLOCK)
    per_split = tl.cdiv(n_blocks, num_splits)
    b0 = split * per_split
    b1 = tl.minimum(b0 + per_split, n_blocks)

    scale = tl.load(scale_ptr)
    scale_bf16 = scale.to(tl.bfloat16)
    dequant = 1.0 / scale
    d = tl.arange(0, D)
    k_col = HQ * D + h * D
    v_col = (HQ + HKV) * D + h * D

    if (split == 0) & (row_group == 0):
        # The step's K/V into the paged cache: what applyBiasRopeUpdateKVCacheV2 stores.
        tok = tl.arange(0, 8)
        tok_ok = tok < M
        pos = past + tok
        page_k = tl.load(bt_ptr + pos // P, mask=tok_ok, other=0).to(tl.int64)
        page_v = tl.load(bt_ptr + bt_stride_kv + pos // P, mask=tok_ok, other=0).to(tl.int64)
        slot = (h * P + pos % P) * D
        k_new = tl.load(
            qkv_ptr + tok[:, None] * qkv_stride + k_col + d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        v_new = tl.load(
            qkv_ptr + tok[:, None] * qkv_stride + v_col + d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        tl.store(
            pool_ptr + (page_k * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(k_new, scale_bf16),
            mask=tok_ok[:, None],
        )
        tl.store(
            pool_ptr + (page_v * PAGE + slot)[:, None] + d[None, :],
            _to_cache_fp8(v_new, scale_bf16),
            mask=tok_ok[:, None],
        )

    r = row_group * R + tl.arange(0, R)  # row = token * G + query head within the KV group
    t = r // G
    row_ok = t < M
    q = tl.load(
        qkv_ptr + t[:, None] * qkv_stride + (h * G + r % G)[:, None] * D + d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    qk_scale = sm_scale_log2 * dequant
    if Q_FP8:
        q = _to_cache_fp8(q, scale_bf16).to(tl.bfloat16)
        qk_scale = qk_scale * dequant
    q_pos = past + t
    m_i = tl.full([R], -1.0e30, tl.float32)
    l_i = tl.zeros([R], tl.float32)
    acc = tl.zeros([R, D], tl.float32)
    for b in range(b0, b1):
        j = base + b * BLOCK + tl.arange(0, BLOCK)
        # Never touch a page outside the attended range: a sliding-window
        # layer's block table need not hold a valid page there.
        cached = (j >= lo) & (j < past)
        new = (j >= past) & (j < L)
        page_k = tl.load(bt_ptr + j // P, mask=cached, other=0).to(tl.int64)
        page_v = tl.load(bt_ptr + bt_stride_kv + j // P, mask=cached, other=0).to(tl.int64)
        in_page = (h * P + j % P) * D
        k = tl.load(
            pool_ptr + (page_k * PAGE + in_page)[:, None] + d[None, :],
            mask=cached[:, None],
            other=0.0,
        )
        v = tl.load(
            pool_ptr + (page_v * PAGE + in_page)[:, None] + d[None, :],
            mask=cached[:, None],
            other=0.0,
        )
        new_row = j - past
        k_new = tl.load(
            qkv_ptr + new_row[:, None] * qkv_stride + k_col + d[None, :],
            mask=new[:, None],
            other=0.0,
        )
        v_new = tl.load(
            qkv_ptr + new_row[:, None] * qkv_stride + v_col + d[None, :],
            mask=new[:, None],
            other=0.0,
        )
        k = tl.where(new[:, None], _to_cache_fp8(k_new, scale_bf16), k)
        v = tl.where(new[:, None], _to_cache_fp8(v_new, scale_bf16), v)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * qk_scale
        keep = (j[None, :] <= q_pos[:, None]) & (j[None, :] >= lo)
        if WINDOW > 0:
            keep = keep & (j[None, :] > q_pos[:, None] - WINDOW)
        s = tl.where(keep, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    # An empty split (b0 >= b1) stores l = 0, which the combine weighs as zero.
    part = (split * HKV + h) * (M * G) + r
    tl.store(part_o_ptr + part[:, None] * D + d[None, :], acc * dequant, mask=row_ok[:, None])
    tl.store(part_ml_ptr + part * 2, m_i, mask=row_ok)
    tl.store(part_ml_ptr + part * 2 + 1, l_i, mask=row_ok)


@triton.jit
def _paged_decode_combine_kernel(
    part_o_ptr,
    part_ml_ptr,
    out_ptr,
    out_stride,
    num_splits,
    M: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    G: tl.constexpr = HQ // HKV
    row = tl.program_id(0)  # token * G + query head within the KV group
    h = tl.program_id(1)
    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()
    sp = tl.arange(0, BLOCK_S)
    s_ok = sp < num_splits
    d = tl.arange(0, D)
    part = (sp * HKV + h) * (M * G) + row
    m_split = tl.load(part_ml_ptr + part * 2, mask=s_ok, other=-1.0e30)
    l_split = tl.load(part_ml_ptr + part * 2 + 1, mask=s_ok, other=0.0)
    m_max = tl.max(m_split, 0)
    w = tl.where(s_ok & (l_split > 0), tl.math.exp2(m_split - m_max), 0.0)
    den = tl.sum(w * l_split, 0)
    o = tl.load(part_o_ptr + part[:, None] * D + d[None, :], mask=s_ok[:, None], other=0.0)
    num = tl.sum(o * w[:, None], 0)
    t = row // G
    tl.store(out_ptr + t * out_stride + (h * G + row % G) * D + d, (num / den).to(tl.bfloat16))


def window_splits(window: int) -> int:
    """Splits that give every program of a ``window``-key layer one 32-key block."""
    return triton.cdiv(window + MAX_ROWS, BLOCK_N) + 1


class PagedDecodeAttention(OpWrapper):
    """Causal (optionally sliding-window) attention of one decode step over the paged FP8 cache.

    With ``M = qkv.shape[0]`` rows (tokens at positions ``L - M .. L - 1``,
    ``L = seq_lens[0]``), ``H`` query and ``G = H / 8`` KV heads of width 64
    and ``qkv = [q (H*64) | k (G*64) | v (G*64)]`` per row::

        cache[k|v][pos] = e4m3(bf16(x * bf16(s)))           # the step's rows, s = kv_scale_orig_quant
        q'   = q, or e4m3(bf16(q * bf16(s))) / s with q_fp8
        out[t, h] = softmax_j(q'[t, h] . K[j] / (s * 8)) . V[j] / s
                    over j <= L - M + t (and j > L - M + t - window when window > 0)

    with ``K``/``V`` the cache contents (the step's rows included). The cache
    is written in place; the output ``[M, H * 64]`` BF16 is fresh.
    ``block_tables`` is the sequence's ``[2, pages]`` row of the TRT-LLM
    table (K page ids, then V page ids) into ``kv_pool [pages, G, P, 64]``.
    Batch 1: one sequence per call.

    Fusion boundary: the projection and pre-attention prep producing ``qkv``
    are outside; the output norm and gate after attention are outside.

    ``enable_pdl`` launches both kernels with programmatic dependent launch;
    the split kernel waits (``griddepcontrol.wait``) before its first global
    read and lets the combine kernel launch early; the combine waits for it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, ctx (= seq_lens[0], the step's rows included), window, page,
    # q_fp8, splits (None = window_splits(window)).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{rows} rows, sliding window 2048, {ctx} keys: {why}",
            spec=dict(rows=rows, ctx=ctx, window=2048, page=32),
        )
        for rows, ctx, why in (
            (8, 100000, "K=7 verify at a 100k-token context, the primary cell"),
            (8, 131072, "K=7 verify at a 128k-token context"),
            (8, 2100, "window just below the context: the first page is partly outside"),
            (8, 2055, "window start exactly at a block boundary"),
            (8, 1000, "context shorter than the window: plain causal"),
            (8, 40, "a short context inside one 32-key block"),
            (3, 100000, "a partial row group (K=2 verify)"),
            (1, 100000, "one row (plain decode)"),
            (1, 1, "the first token: no cached key at all"),
        )
    ) + (
        Cell(
            why="8 rows, Q quantized as the built-in path does (q_fp8)",
            spec=dict(rows=8, ctx=100000, window=2048, page=32, q_fp8=True),
        ),
        Cell(
            why="8 rows, 64-token pages",
            spec=dict(rows=8, ctx=100000, window=2048, page=64),
        ),
        Cell(
            why="8 rows, full causal attention over 4096 keys in 64 splits",
            spec=dict(rows=8, ctx=4096, window=0, page=32, splits=64),
        ),
        Cell(
            why="8 rows, more splits than blocks (empty splits)",
            spec=dict(rows=8, ctx=300, window=2048, page=32, splits=66),
        ),
    )

    note = """
    Metadata consumed: seq_lens[0] (device; read by the kernels, never on the
    host), the sequence's K and V page ids, the layer's pool view, the FP8 KV
    scale (1.0 in the PyTorch backend). Call site: the target's sliding-window
    attention layers (46 of 62) on decode steps of at most 8 rows and one
    sequence. Grids: split (num_splits, G, 64/32), combine (M*8, G).

    Gated by its own tests: the cache bytes against the built-in
    `applyBiasRopeUpdateKVCacheV2` (thop generation preprocess) including
    saturating and NaN inputs; the output against this reference; the output
    error against an fp64 reference no worse than the built-in trtllm-gen path
    (much smaller with BF16 Q); with q_fp8, the output within a
    per-element bound of the built-in's set by the trtllm-gen kernel's E4M3
    rounding of P (this entry keeps P in BF16); CUDA-graph replay at a context
    other than the captured one.

    Loud preconditions (ValueError): BF16 qkv with unit last stride and width
    (H + 2G) * 64, H = 8 G, 1..8 rows, FP8 E4M3 pool [pages, G, P, 64] with P a
    power of two, int32 [2, pages] block table, int32 seq_lens, fp32 scale.

    Not certified: more than one sequence per call, head width other than 64,
    a query/KV head ratio other than 8, NVFP4 or BF16 caches.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        rows = qkv.shape[0]
        page = kv_pool.shape[2] if kv_pool.dim() == 4 else 0
        if (
            qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.stride(-1) != 1
            or qkv.shape[1] != (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
            or num_q_heads != Q_PER_KV * num_kv_heads
            or not 1 <= rows <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[1:]) != (num_kv_heads, page, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or page & (page - 1) != 0
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 2
            or block_tables.shape[0] != 2
            or block_tables.stride(-1) != 1
            or seq_lens.dtype != torch.int32
            or kv_scale_orig_quant.dtype != torch.float32
            or num_splits < 1
            or window < 0
        ):
            raise ValueError(
                "paged_decode_attention takes BF16 qkv [1..8, (H + 2G) * 64] with H = 8 G, an "
                "FP8 E4M3 pool [pages, G, P, 64] (P a power of two), an int32 [2, pages] block "
                f"table, int32 seq_lens and an fp32 scale; got qkv {qkv.dtype} {tuple(qkv.shape)}, "
                f"H={num_q_heads} G={num_kv_heads}, pool {kv_pool.dtype} {tuple(kv_pool.shape)}, "
                f"table {block_tables.dtype} {tuple(block_tables.shape)}, seq_lens "
                f"{seq_lens.dtype}, scale {kv_scale_orig_quant.dtype}, splits {num_splits}, "
                f"window {window}"
            )
        out = torch.empty((rows, num_q_heads * HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)
        group_rows = rows * Q_PER_KV
        part_o = torch.empty(
            (num_splits * num_kv_heads * group_rows, HEAD_DIM),
            dtype=torch.float32,
            device=qkv.device,
        )
        part_ml = torch.empty(
            (num_splits * num_kv_heads * group_rows, 2), dtype=torch.float32, device=qkv.device
        )
        _paged_decode_split_kernel[
            (num_splits, num_kv_heads, triton.cdiv(group_rows, ROWS_PER_PROGRAM))
        ](
            qkv,
            qkv.stride(0),
            kv_pool,
            block_tables,
            block_tables.stride(0),
            seq_lens,
            kv_scale_orig_quant,
            part_o,
            part_ml,
            math.log2(math.e) / math.sqrt(HEAD_DIM),
            num_splits,
            M=rows,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            D=HEAD_DIM,
            P=page,
            WINDOW=window,
            BLOCK=BLOCK_N,
            R=ROWS_PER_PROGRAM,
            Q_FP8=q_fp8,
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            launch_pdl=enable_pdl,
        )
        _paged_decode_combine_kernel[(group_rows, num_kv_heads)](
            part_o,
            part_ml,
            out,
            out.stride(0),
            num_splits,
            M=rows,
            HQ=num_q_heads,
            HKV=num_kv_heads,
            D=HEAD_DIM,
            BLOCK_S=triton.next_power_of_2(num_splits),
            LAUNCH_WITH_PDL=enable_pdl,
            num_warps=4,
            launch_pdl=enable_pdl,
        )
        return out

    def reference(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        enable_pdl: bool = False,
    ) -> torch.Tensor:
        """The cache write, then fp32 attention over the cache, rounded to BF16 once.

        The FP8 conversions are part of what the op computes (the cache holds
        E4M3, and with ``q_fp8`` so does Q), so the reference performs them
        the same way; everything after them is fp32, with TF32 off for the two
        matmuls (``true_fp32_matmul``: torch's default here would make them
        TF32 products and the reference less accurate than the op). ``num_splits``
        and ``enable_pdl`` change how the op is cut and launched, not what it
        computes, and are ignored here. Writes the step's rows into
        ``kv_pool`` like the op.
        """
        rows = qkv.shape[0]
        page = kv_pool.shape[2]
        length = int(seq_lens[0])
        past = length - rows
        scale = kv_scale_orig_quant.float()[0]
        scale_bf16 = scale.to(torch.bfloat16).float()

        def to_cache(x: torch.Tensor) -> torch.Tensor:
            # E4M3 round-to-nearest with saturation: a finite overflow lands on
            # +-448 rather than torch's NaN; NaN stays NaN.
            y = (x.float() * scale_bf16).to(torch.bfloat16).float()
            return y.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        q_w, kv_w = num_q_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
        q = qkv[:, :q_w].reshape(rows, num_q_heads, HEAD_DIM)
        k_new = qkv[:, q_w : q_w + kv_w].reshape(rows, num_kv_heads, HEAD_DIM)
        v_new = qkv[:, q_w + kv_w :].reshape(rows, num_kv_heads, HEAD_DIM)
        pool_bytes = kv_pool.view(torch.uint8)  # fp8 has no CUDA advanced indexing
        positions = torch.arange(past, length, device=qkv.device)
        k_pages = block_tables[0, positions // page].long()
        v_pages = block_tables[1, positions // page].long()
        slots = positions % page
        for h in range(num_kv_heads):
            pool_bytes[k_pages, h, slots] = to_cache(k_new[:, h]).view(torch.uint8)
            pool_bytes[v_pages, h, slots] = to_cache(v_new[:, h]).view(torch.uint8)

        lo = max(past - window + 1, 0) if window > 0 else 0
        keys = torch.arange(lo, length, device=qkv.device)
        dequant = 1.0 / scale

        def gather(kv: int) -> torch.Tensor:
            raw = pool_bytes[block_tables[kv, keys // page].long(), :, keys % page]
            return raw.view(torch.float8_e4m3fn).float() * dequant  # [keys, G, 64]

        k, v = gather(0), gather(1)
        qf = q.float()
        if q_fp8:
            qf = to_cache(q).float() * dequant
        group = num_q_heads // num_kv_heads
        out = torch.empty((rows, num_q_heads, HEAD_DIM), dtype=torch.float32, device=qkv.device)
        with true_fp32_matmul():
            for t in range(rows):
                pos = past + t
                keep = keys <= pos
                if window > 0:
                    keep &= keys > pos - window
                for h in range(num_kv_heads):
                    s = qf[t, h * group : (h + 1) * group] @ k[keep, h].T / math.sqrt(HEAD_DIM)
                    out[t, h * group : (h + 1) * group] = torch.softmax(s, dim=-1) @ v[keep, h]
        return out.reshape(rows, -1).to(torch.bfloat16)

    def is_valid(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        window: int,
        num_splits: int,
        q_fp8: bool = False,
        enable_pdl: bool = False,
    ) -> None:
        """Inputs the kernels take and answer wrongly, with no error.

        A context shorter than the step's rows puts the step's keys at negative
        positions; a block table shorter than the context reads page ids past
        its end; a page id outside the pool reads and writes outside it.
        """
        rows = qkv.shape[0]
        length = int(seq_lens[0])
        page = kv_pool.shape[2]
        if length < rows:
            raise ValueError(f"seq_lens[0] = {length} is shorter than the step's {rows} rows")
        pages = (length + page - 1) // page
        if block_tables.shape[1] < pages:
            raise ValueError(
                f"a [2, {block_tables.shape[1]}] block table cannot address {length} keys "
                f"of {page}-token pages"
            )
        lo = max(length - rows - window + 1, 0) if window > 0 else 0
        used = block_tables[:, lo // page : pages]
        if used.numel() and (int(used.min()) < 0 or int(used.max()) >= kv_pool.shape[0]):
            raise ValueError(
                f"page ids [{int(used.min())}, {int(used.max())}] outside the {kv_pool.shape[0]}-page pool"
            )


paged_decode_attention = PagedDecodeAttention()
