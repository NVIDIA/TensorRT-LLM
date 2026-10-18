# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``full_attention_split_kv`` with the full-attention layer's prep folded in: one launch after the QKV GEMM.

On a full-attention (NoPE) layer's decode step, the folded prep
(``preattn_prep_append``) and the split-KV attention ``full_attention_split_kv`` run back to back: the prep normalizes Q
and K (weightless RMSNorm, BF16; Q times the query pre-scale, BF16), writes the step's K/V into
the paged FP8 cache, and leaves the fused operand ``[q' | k' | v]`` that ``full_attention_split_kv`` reads
its query from. This entry is ``full_attention_split_kv``'s kernel reading the raw QKV projection instead
(``full_attention_split_kv_kernel.FullAttnSplitKvFoldKernel``):

* each softmax thread loads its 32 raw BF16 Q dims and normalizes them exactly as the prep does
  (``preattn_prep._inverse_rms`` / ``_normalize``: per prep lane of 8 elements a sequential
  ``add.rn`` of exact squares, then lane bits 0 and 1 in the thread and bit 2 with the other
  warpgroup's half through shared memory, ``* 2^-6 + eps``, ``rsqrt.approx.ftz``, BF16, then
  ``* q_scale`` and BF16 again when ``q_scale != 1``), then builds ``Q_hi + Q_lo`` as
  ``full_attention_split_kv`` does;
* warps 10-11 of the CTA whose page range holds a key of the step's rows (the two warps
  ``full_attention_split_kv`` leaves idle; four threads per (row, KV head) item, one load pass each) write
  that CTA's share of the step's K' (the same norm) and V into the cache, byte for byte the
  prep's ``e4m3_rn_satfinite(bf16(x * bf16(s)))`` at ``pool[table[kv][pos // 32], head,
  pos % 32, :]``, then ``fence.proxy.async.global`` and arrive on the CTA's ``app`` barrier;
  that CTA's loader waits on ``app`` before the TMA of any page at or past the step's first new
  page.

So the partials (and, merged, the attention output) and the cache bytes are bit for bit those of
``preattn_prep_append`` followed by ``full_attention_split_kv`` on the same projection and cache. The prep's
other outputs -- its E4M3 query, the fused operand, the gate copy and the FMHA scale words -- are
not written: only the built-in FMHA reads the first and last, and a caller reads the gate from
the projection (checkpoint order: ``qkv[:, :2048].view(rows, 16, 2, 64)[:, :, 1, :]``;
permuted: ``qkv[:, 1280:]``).

Ownership and progress: ``full_attention_split_kv``'s (pages per CTA, page ids, ring, TMEM, partials, no
CTA waits on another), plus the append. The step's rows ``L - rows .. L - 1`` lie on at most two
pages; each is in exactly one CTA's range, and that CTA's side warps write exactly the rows on
its own pages. With the op's split rule (at least ``MIN_PAGES`` per split) the last keyed split
holds at least three pages, so both of the step's pages are the last CTA's; the kernel does not
rely on it (a test build of one page per split appends a straddling step from two CTAs). Only the
owning CTA's loader reads those pages in this launch, and only after ``app``: it waits on ``app``
for a page of its own range at or past the step's first page, and every side thread of a CTA
whose range reaches that page arrives on it (64 arrivals, appending or not). The padding slots
of a CTA's last tile (past its range) load nothing and do not wait. The side warps wait on
nothing but the predecessor, so ``app`` always completes.

PDL memory-order record (``enable_pdl``):

* Before ``griddepcontrol.wait``: ``full_attention_split_kv``'s immutable reads only (``seq_lens[0]``,
  ``kv_scale_orig_quant[0]``, the CTA's page ids), the same argument as there.
* After it: every projection read (Q by the softmax threads, K/V by the side warps), every cache
  store (side warps) and every page (the loader, which also waits on ``app`` for the step's
  pages), every partial store.
* Wait before exit: every thread of the grid executes ``griddepcontrol.wait`` before it exits
  (the roles above, and one more wait at the end of every thread). The predecessor is the QKV
  projection, which triggers early, so the dependent's wait implies the projection completed
  only through this grid; the schedule test audits this kernel's SASS for it
  (``WAIT_BEFORE_EXIT``).
* Transitivity: the gate is read by the consumer (``combine_norm_gate``) from the projection two
  kernels back; the consumer reads it after its own wait, which returns after this grid
  completed, and this grid completes only after its wait on the projection returned.
* ``griddepcontrol.launch_dependents``: as ``full_attention_split_kv`` (every softmax thread, before the
  epilogue).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import full_attention_split_kv as _v4

HEAD_DIM = _v4.HEAD_DIM
PAGE = _v4.PAGE
KV_HEADS = _v4.KV_HEADS
Q_HEADS = _v4.Q_HEADS
MAX_ROWS = _v4.MAX_ROWS
OUT_ROWS = _v4.OUT_ROWS
MAX_SPLITS = _v4.MAX_SPLITS
MAX_CTA_PAGES = _v4.MAX_CTA_PAGES
#: The raw QKV projection row: 16 query heads and their gates, 2 K heads, 2 V heads.
RAW_WIDTH = 2 * (Q_HEADS + KV_HEADS) * HEAD_DIM


def fold_key(eps: float, q_scale: float, qkv_permuted: bool) -> Tuple[float, float, bool]:
    """The compile-time part of one fold build: the norm epsilon, the query pre-scale (as the
    fp32 values the kernel bakes in) and the projection layout."""
    return (float(eps), float(q_scale), bool(qkv_permuted))


def compiled_key(
    device_index: int,
    enable_pdl: bool,
    evict_first: bool,
    eps: float,
    q_scale: float,
    qkv_permuted: bool,
    rescale_cols: int = HEAD_DIM,
) -> tuple:
    """The key ``full_attention_split_kv._compiled_keys`` holds once that fold build is compiled
    (``("rescale_cols", n)`` appended for a chunked-rescale build such as the spill-free one)."""
    key = (
        int(device_index),
        bool(enable_pdl),
        False,
        bool(evict_first),
        fold_key(eps, q_scale, qkv_permuted),
    )
    if rescale_cols != HEAD_DIM:
        key += (("rescale_cols", int(rescale_cols)),)
    return key


def gate_view(qkv: torch.Tensor, qkv_permuted: bool) -> torch.Tensor:
    """The output gate of the raw projection ``[rows, 2304]`` as a strided view (no copy):
    ``[rows, 16, 64]`` in checkpoint order, ``[rows, 1024]`` permuted."""
    rows = qkv.shape[0]
    if qkv_permuted:
        return qkv[:, Q_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM :]
    return qkv[:, : 2 * Q_HEADS * HEAD_DIM].view(rows, Q_HEADS, 2, HEAD_DIM)[:, :, 1, :]


class FullAttnSplitKvFold(OpWrapper):
    """``preattn_prep_append`` (NoPE) then ``full_attention_split_kv``, as one kernel, for one sequence.

    Reads the raw BF16 projection ``qkv`` ``[rows, 2304]`` (checkpoint order, or with
    ``qkv_permuted`` ``[q | k | v | gate]``) and the FP8 paged cache, which must hold every key
    of the sequence before the step's rows. Writes the step's K/V into ``kv_pool`` (the prep's
    bytes) and, into ``part_o`` / ``part_lse``, ``full_attention_split_kv``'s partials of the prep's
    query. Returns ``(part_o, part_lse)``. ``eps`` is the norm epsilon of both Q and K,
    ``q_scale`` the query pre-scale; ``seq_lens[0]`` is the context length including the step's
    rows.

    Loud preconditions (ValueError): BF16 qkv of 1..8 rows and width 2304 with a unit last
    stride and 16-byte aligned rows; the rest as ``full_attention_split_kv``. The first call per device
    and build (``compiled_key``) compiles the kernel, which must not happen inside a CUDA-graph
    capture.

    Not certified: more than one sequence per call, other head counts or widths, other page
    sizes, sliding windows, RoPE (the full-attention layers are NoPE).
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, ctx (= seq_lens[0], the step's rows included); optional splits (default: the
    # device's SM count), pdl, permuted, q_scale (default the checkpoint's), scale (KV scale).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"{rows} rows, {ctx} keys: {why}", spec=dict(rows=rows, ctx=ctx))
        for rows, ctx, why in (
            (8, 100000, "K=7 verify at a 100k-token context (past % 32 = 24)"),
            (8, 455031, "K=7 verify at a 455k-token context, the step straddles two pages"),
            (8, 8192, "a short context; the step's 8 rows end its last page (past % 32 = 24)"),
            (8, 8200, "past % 32 = 0: the step opens a new page"),
            (8, 8217, "past % 32 = 17: the step straddles two pages"),
            (8, 8225, "past % 32 = 25: 7 rows on one page, 1 on the next"),
            (8, 8231, "past % 32 = 31: 1 row on one page, 7 on the next"),
            (8, 1048581, "near the 1M sequence limit"),
            (3, 100003, "a partial row group (K=2 verify) across a page boundary"),
            (1, 100000, "one row (plain decode)"),
            (8, 40, "one split: every other split is key-less"),
            (8, 8, "the first step: every key is new"),
            (1, 1, "the first token: one key"),
        )
    ) + (
        Cell(
            why="8 rows, 455031 keys, PDL launch (the served build)",
            spec=dict(rows=8, ctx=455031, pdl=True),
        ),
        Cell(
            why="8 rows, 100000 keys, the permuted (low-latency) projection layout",
            spec=dict(rows=8, ctx=100000, permuted=True),
        ),
        Cell(
            why="8 rows, 100000 keys, a query pre-scale of 1 (one BF16 rounding of Q)",
            spec=dict(rows=8, ctx=100000, q_scale=1.0),
        ),
        Cell(
            why="8 rows, 100000 keys, a KV scale other than 1",
            spec=dict(rows=8, ctx=100000, scale=0.5),
        ),
        Cell(
            why="8 rows, 131 keys: the first CTA's last tile pads past the step's first page",
            spec=dict(rows=8, ctx=131),
        ),
        Cell(
            why="8 rows, 40003 keys, 96 splits (fewer CTAs than SMs)",
            spec=dict(rows=8, ctx=40003, splits=96),
        ),
    )

    note = """
    Metadata consumed: seq_lens[0] (device; read by the kernel, never on the host), the
    sequence's K and V page ids, the layer's pool view, the FP8 KV scale (1.0 in the PyTorch
    backend), the norm epsilon and query pre-scale (compile-time). Grid: num_splits CTAs of 384
    threads, one CTA per SM, as full_attention_split_kv. Call site: the target's 16 full-attention layers
    on decode steps of one sequence and at most 8 rows (fast path full_attention_prep_fold, on
    top of full_attention_split_kv and full_attention_prep_append), in
    place of preattn_prep_append and full_attention_split_kv; combine_norm_gate consumes the partials and
    reads the gate from the projection.

    Gated by its own tests: partials and cache bytes bit for bit those of preattn_prep_append
    then full_attention_split_kv on the same projection and cache (every cell; poisoned step slots; every
    context up to 300 keys; with test builds of one page per split, the step's two pages in two
    CTAs); eager and captured graph equal; late writers
    of the projection observed; a loader that does not wait for the append is caught; the
    merged partials within full_attention_split_kv's bound of this entry's reference.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_splits: int,
        enable_pdl: bool = False,
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = qkv.shape[0] if qkv.dim() == 2 else 0
        if part_o is None or part_lse is None:
            part_o, part_lse = _v4.partial_buffers(num_splits, qkv.device)
        tensors = (qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, part_o, part_lse)
        width = block_tables.shape[1] if block_tables.dim() == 2 else 0
        if (
            qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.shape[1] != RAW_WIDTH
            or qkv.stride(-1) != 1
            or qkv.stride(0) % 8 != 0
            or not 1 <= rows <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[1:]) != (KV_HEADS, PAGE, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 2
            or block_tables.shape[0] != 2
            or width < 1
            or block_tables.stride(-1) != 1
            or not 1 <= num_splits <= MAX_SPLITS
            or rescale_cols not in _v4.RESCALE_COLS
            or -(-width // num_splits) > MAX_CTA_PAGES
            or seq_lens.dtype != torch.int32
            or seq_lens.numel() < 1
            or kv_scale_orig_quant.dtype != torch.float32
            or kv_scale_orig_quant.numel() < 1
            or part_o.dtype != torch.float16
            or tuple(part_o.shape) != (OUT_ROWS, num_splits, HEAD_DIM)
            or not part_o.is_contiguous()
            or part_lse.dtype != torch.float32
            or tuple(part_lse.shape) != (OUT_ROWS, num_splits)
            or not part_lse.is_contiguous()
            or any(t.device != qkv.device or t.data_ptr() % 16 for t in tensors)
            or qkv.device.type != "cuda"
            or torch.cuda.get_device_capability(qkv.device) not in ((10, 3), (10, 7))
        ):
            raise ValueError(
                "full_attention_prep_fold takes, on one sm_103 or sm_107 device and 16-byte aligned: "
                "a BF16 raw "
                f"projection [1..{MAX_ROWS}, {RAW_WIDTH}] (unit last stride, rows a multiple of 8 "
                f"elements apart), a contiguous FP8 E4M3 pool [blocks, {KV_HEADS}, {PAGE}, "
                f"{HEAD_DIM}], an int32 [2, pages] block table (unit last stride, at most "
                f"{MAX_CTA_PAGES} pages per split), int32 seq_lens, an fp32 scale, "
                f"1..{MAX_SPLITS} splits and contiguous partials FP16 [{OUT_ROWS}, splits, "
                f"{HEAD_DIM}] / fp32 [{OUT_ROWS}, splits]; got qkv {qkv.dtype} "
                f"{tuple(qkv.shape)} strides {tuple(qkv.stride())}, pool {kv_pool.dtype} "
                f"{tuple(kv_pool.shape)}, table {block_tables.dtype} "
                f"{tuple(block_tables.shape)} strides {tuple(block_tables.stride())}, seq_lens "
                f"{seq_lens.dtype}, scale {kv_scale_orig_quant.dtype}, splits {num_splits}, "
                f"partials {part_o.dtype} {tuple(part_o.shape)} / {part_lse.dtype} "
                f"{tuple(part_lse.shape)}, devices {[str(t.device) for t in tensors]}"
            )
        device_index = qkv.device.index
        fold = fold_key(eps, q_scale, qkv_permuted)
        key = compiled_key(
            device_index, enable_pdl, evict_first, eps, q_scale, qkv_permuted, rescale_cols
        )
        if torch.cuda.is_current_stream_capturing() and key not in _v4._compiled_keys:
            raise RuntimeError(
                "full_attention_prep_fold compiles on its first call per device and build; make that "
                "call before capturing a CUDA graph"
            )
        build = {} if rescale_cols == HEAD_DIM else {"rescale_cols": int(rescale_cols)}
        compiled = _v4._compiled_kernel(
            device_index, bool(enable_pdl), False, bool(evict_first), fold=fold, **build
        )
        _v4._compiled_keys.add(key)
        import cuda.bindings.driver as cuda

        compiled(
            qkv.data_ptr(),
            kv_pool.data_ptr(),
            block_tables.data_ptr(),
            seq_lens.data_ptr(),
            kv_scale_orig_quant.data_ptr(),
            part_o.data_ptr(),
            part_lse.data_ptr(),
            rows,
            qkv.stride(0),
            kv_pool.shape[0],
            width,
            block_tables.stride(0),
            num_splits,
            cuda.CUstream(torch.cuda.current_stream(qkv.device).cuda_stream),
        )
        return part_o, part_lse

    def reference(
        self,
        qkv: torch.Tensor,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_splits: int,
        enable_pdl: bool = False,
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> torch.Tensor:
        """The prep's arithmetic and in-place append, then ``full_attention_split_kv``'s reference.

        ``preattn_prep_append``'s reference computes the normalized Q / K (float64 between the
        BF16 rounding points the op keeps) and writes the step's K/V bytes into ``kv_pool`` in
        place, as the op does; ``full_attention_split_kv``'s reference then attends with the op's
        ``Q_hi + Q_lo`` over that cache. Returns the fp32 attention output ``[rows, 16 * 64]``.
        ``num_splits``, ``enable_pdl``, ``evict_first``, ``part_o`` and ``part_lse`` change
        nothing it computes and are ignored.
        """
        from .preattn_prep_append import preattn_prep_append

        rows = qkv.shape[0]
        scratch_q = torch.empty(rows * Q_HEADS * HEAD_DIM, dtype=torch.uint8, device=qkv.device)
        scratch_scales = torch.empty(3, dtype=torch.float32, device=qkv.device)
        fused, _ = preattn_prep_append.reference(
            qkv,
            Q_HEADS,
            KV_HEADS,
            eps,
            q_scale,
            qkv_permuted,
            kv_pool,
            block_tables,
            seq_lens,
            kv_scale_orig_quant,
            kv_scale_orig_quant,
            scratch_q,
            scratch_scales[:2],
            scratch_scales[2:],
            1.0,
        )
        return _v4.full_attention_split_kv.reference(
            fused.contiguous(),
            kv_pool,
            block_tables,
            seq_lens,
            kv_scale_orig_quant,
            num_splits,
        )

    def is_valid(
        self,
        qkv: torch.Tensor,
        eps: float,
        q_scale: float,
        qkv_permuted: bool,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_splits: int,
        enable_pdl: bool = False,
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> None:
        """Inputs the kernel takes and answers wrongly, with no error: ``full_attention_split_kv``'s (a
        context shorter than the step's rows, a table narrower than the context, a page id
        outside the pool), checked on the same arguments."""
        _v4.full_attention_split_kv.is_valid(
            qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, num_splits
        )


full_attention_prep_fold = FullAttnSplitKvFold()
