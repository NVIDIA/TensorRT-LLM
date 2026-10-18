# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode attention partials of a full-attention layer: one CTA per SM, a page range each.

The built-in decode step of this target's 16 full-attention layers runs flashinfer's
trtllm-gen generation FMHA (``fmhaSm100fKernel_...MultiCtasKvVarSeq...``) on a grid of
76 KV splits x 2 KV heads: each CTA streams one KV head through 2 KB TMA boxes, and one CTA
reduces all splits' partials serially at the end. This entry is one CuTe DSL kernel
(``full_attention_split_kv_kernel``) on a grid of one CTA per SM: each CTA serves both KV heads of an even
share of the sequence's 32-key pages (computed on the device from ``seq_lens``), streams each
page through TMA tensor boxes (V in one 4 KB box, K in two 2 KB boxes, one per KV head) into a
16-stage ring, scores each 128-key tile with one N=128 MMA per Q term and K block, and writes
one partial per row -- the normalized O in FP16 and its log2-sum-exp -- with no in-kernel
combine. ``combine_norm_gate`` (the attention norm-gate that follows) merges the partials.

It writes no cache byte: the step's K/V and its query are written before it by the
``full_attention_prep_append`` folded prep (``preattn_prep_append``), whose BF16 fused operand ``[q' | k' | v]`` is this
op's ``qkv``.

Numerics (not bit-identical to the built-in path). Q enters as ``Q_hi + Q_lo``, two E4M3
terms (``Q_hi`` the nearest 4-significant-bit value, ``Q_lo`` the E4M3 rounding of the rest),
which keeps Q at about BF16 precision; the built-in path rounds Q to one E4M3 term, which
dominates its error. K/V are the cache's E4M3 values. P is rounded to E4M3 after a scale-up by
2^4 against a running maximum that is only raised when the new maximum exceeds it by more than
2^4 (log2 units); the denominator sums the unrounded P. S and P.V accumulate in fp32; each
split stores its normalized O (times the V dequant scale) in FP16 with an fp32 log2-sum-exp.
The split of pages over CTAs is static and the combine reduces in a fixed order, so the result
is deterministic run to run. ``rescale_cols`` (16: the spill-free build) only changes how many O
columns the softmax warps' rescale of their accumulator holds in registers at once (the default
64 makes ptxas spill part of S on every tile): the same multiply per element, the same bits, no
other access.

Ownership and progress (per launch; ``eff = min(grid, ceil(pages / MIN_PAGES))`` with
``MIN_PAGES = 4``, the number of CTAs that hold keys, derived on the device from ``seq_lens``;
``keyed_splits`` computes the same number on the host):

* Pages. CTA ``s < eff`` owns pages ``[s n / eff, (s + 1) n / eff)`` of the sequence's ``n``
  pages and reads nothing else; no CTA writes a page.
* Page ids. The loader warp of a CTA reads its range of both block-table rows once (coalesced)
  into shared memory and synchronizes its warp; only it reads them back.
* Shared-memory ring (16 stages of one K and one V page; the K of 4 consecutive stages is one
  tile slot ``[head][128 keys][64 B]``, which the per-head K boxes fill). The loader waits
  ``kv_empty`` before reusing a stage; the MMA warp waits ``kv_full`` (TMA transaction count,
  all three boxes of the page) of the tile's 4 stages before the tile's QK and commits
  ``kv_empty`` after the P.V of the tile that read them. The unwritten V rows of the last
  page (any bytes; an E4M3 NaN times P = 0 is NaN) are zeroed in shared memory by the MMA warp
  after it waited for that page's ``kv_full`` and before it issues that tile's P.V
  (``fence.proxy.async.shared::cta``, then a warp sync); the stage cannot be refilled before
  that P.V's ``kv_empty`` commit.
* TMEM (512 columns). Allocated by the MMA warp; S/P double-buffered per softmax warpgroup
  (``s_full`` committed by the MMA warp; P of a tile published in two 16-column halves,
  ``p_full`` for keys 0-63 and ``p_full2`` for keys 64-127, each arrived by the 128 threads of
  the owning warpgroup after their TMEM store and fence; the P.V of pages 0-1 is issued at
  ``p_full``, of pages 2-3 at ``p_full2``, and QK(t + 2), which overwrites that S/P buffer, is
  issued after the tile's last P.V on the same in-order MMA issue), O per warpgroup
  (``o_done``). Freed by the MMA warp after a named barrier with the 256 softmax threads, which
  they reach only after reading O.
* Partials. Row ``R = token * 16 + query head`` of split ``s`` is written by exactly one thread
  pair of CTA ``s`` (32 FP16 values each; the log2-sum-exp by the first). A CTA with no keys
  (``s >= eff``) writes ``-inf`` as its rows' log2-sum-exp and no O. The combine gives a
  ``-inf`` split weight 0; given the step's ``seq_lens`` it derives ``eff`` as this kernel does
  and reads neither the O nor the log2-sum-exp of the splits ``>= eff``. Every partial store
  follows the CTA's dependency wait, and the consumer reads them only after its own wait, i.e.
  after this grid completed.
* Progress. No CTA waits on another CTA of the grid and there is no cluster: each CTA's roles
  wait only on barriers its own CTA arrives on. The grid takes one CTA per SM (shared memory and
  all 512 TMEM columns), so a CTA becomes resident only on an SM no other TMEM or large-smem CTA
  holds; it waits for nothing such a CTA could need.

PDL memory-order record (``enable_pdl``; launch attribute set, and the kernel compiled with
``griddepcontrol``):

* Before ``griddepcontrol.wait`` (the op's build, ``prefetch=False``): TMA descriptor prefetch
  (kernel parameters), shared-memory mbarrier initialization, ``seq_lens[0]``,
  ``kv_scale_orig_quant[0]`` and this CTA's range of both block-table rows -- no page, no Q.
  Why each is immutable at that point: ``seq_lens``, the block table and the KV scale are graph
  inputs or module buffers written before the decode graph starts (host prepare and the input
  staging's between-graph commit are stream-ordered full dependencies of the graph launch); inside the
  step the speculative worker changes the KV lengths only after the target forward. (With
  ``prefetch=True``, which the op does not build by default, the loader also streams the pages
  whose every key lies below the step's first new position ``L - rows`` before its wait: such a
  page was written by an earlier step's prep and holds accepted tokens, and this step's prep
  writes only positions ``L - rows .. L - 1``.) Transitivity: these reads may overlap only the
  grids this kernel's primary (the folded prep, which waits before it reads or writes) has not
  yet waited for; the QKV GEMM before it triggers only after its own wait, so the chain back to
  the graph's first kernel has completed every earlier layer, and no kernel of the step writes
  any of these buffers. An in-graph writer of ``seq_lens`` / the block table (no fast path has one) must
  reach layer 0's QKV through a full dependency, so that no pre-wait read can overlap it.
* After the wait (per thread: the softmax warps before Q, the loader before its first page --
  with ``prefetch``, before its first page at or past ``(L - rows) // 32`` -- and a key-less
  CTA before its partial store): the fused operand's Q, every page (TMA), and every partial
  store. The tests certify this with late writers of the step's pages, of Q and of older pages,
  and with two builds the tests must catch: ``loader_wait=False`` (the loader never waits) and
  ``prefetch=True`` (older pages before the wait).
* ``griddepcontrol.launch_dependents``: by every softmax thread after its last P tile, before
  the epilogue. The dependent (the combine + norm-gate) waits for this grid's completion before
  it reads a partial.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: Head width, tokens per page, KV heads and query heads per KV head of one tep8 rank.
HEAD_DIM = 64
PAGE = 32
KV_HEADS = 2
Q_PER_KV = 8
Q_HEADS = KV_HEADS * Q_PER_KV
QKV_WIDTH = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
#: Query rows the kernel's row map is written for: one accepted token and up to 7 drafts.
MAX_ROWS = 8
#: Output rows of a step: (token, query head) pairs.
OUT_ROWS = MAX_ROWS * Q_HEADS
#: Largest grid the combine reads partials for.
MAX_SPLITS = 512
#: Page ids one CTA stages in shared memory per table row.
MAX_CTA_PAGES = 256
#: Fewest pages per keyed split: a context of n pages has min(splits, ceil(n / MIN_PAGES)) keyed
#: splits, the first ones; the rest are key-less.
MIN_PAGES = 4
#: O columns the kernel's rescale holds in registers at once: 64 (the whole head) is the default
#: build, which spills part of S on every tile; 16 is the spill-free build.
RESCALE_COLS = (16, 32, HEAD_DIM)
SPILL_FREE_RESCALE_COLS = 16


def keyed_splits(seq_len: int, num_splits: int) -> int:
    """How many of ``num_splits`` splits hold keys at context ``seq_len`` (the kernel's
    ``eff``; splits ``>= keyed_splits`` are key-less)."""
    pages = (seq_len + PAGE - 1) // PAGE
    return min(num_splits, (pages + MIN_PAGES - 1) // MIN_PAGES)


def partial_buffers(num_splits: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uninitialized partial buffers for ``num_splits`` CTAs: FP16 O ``[128, splits, 64]`` and
    fp32 log2-sum-exp ``[128, splits]`` (row = token * 16 + query head)."""
    part_o = torch.empty(OUT_ROWS, num_splits, HEAD_DIM, dtype=torch.float16, device=device)
    part_lse = torch.empty(OUT_ROWS, num_splits, dtype=torch.float32, device=device)
    return part_o, part_lse


@functools.lru_cache(maxsize=32)
def _compiled_kernel(
    device_index: int,
    enable_pdl: bool,
    prefetch: bool,
    evict_first: bool,
    loader_wait: bool = True,
    k_tile: bool = True,
    p_split: bool = True,
    rescale_cols: int = HEAD_DIM,
    rescale_log2: float = 4.0,
    fold: Optional[Tuple[float, float, bool]] = None,
    app_wait: bool = True,
    min_pages: int = MIN_PAGES,
):
    """The compiled kernel of one build. ``loader_wait=False`` (the loader never waits for the
    predecessor) exists for the PDL tests' negative control; ``k_tile=False`` and
    ``p_split=False`` (slower builds of the same values) exist for the op test's bitwise check
    of those builds. The op never builds any of them. ``rescale_cols`` (16 or 32: the O rescale
    in chunks, no register spill; ``SPILL_FREE_RESCALE_COLS`` is the spill-free build) gives
    the same values as the default 64, in the op's build and in the fold build. ``rescale_log2``
    other than 4 (how far a row's maximum must rise before O is rescaled; other values change
    the rounding of P) exists for the op test, which rescales on every rise to compare the
    ``rescale_cols`` builds on many rescales (in the op's build and in the fold build).
    ``fold = (eps, q_scale, permuted)``
    builds ``full_attention_prep_fold``'s kernel (``FullAttnSplitKvFoldKernel``: the raw projection, the
    prep's norm and the step's K/V append inside), without the pre-wait prefetch;
    ``app_wait=False`` (its loader does not wait for the append) is that op's tests' negative
    control. ``min_pages`` other than ``MIN_PAGES`` (fewer pages per split, so that the step's
    two pages can sit in two CTAs) exists for that op's tests too; the ops build neither."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from . import full_attention_split_kv_kernel as fk

    shared = (
        (HEAD_DIM, fk.D),
        (PAGE, fk.PAGE),
        (QKV_WIDTH, fk.QKV_WIDTH),
        (MAX_ROWS, fk.MAX_ROWS),
        (MAX_CTA_PAGES, fk.MAX_CTA_PAGES),
        (2 * (Q_HEADS + KV_HEADS) * HEAD_DIM, fk.RAW_WIDTH),
    )
    assert all(mine == theirs for mine, theirs in shared), shared
    if fold is None:
        kernel = fk.FullAttnSplitKvKernel(
            enable_pdl=enable_pdl,
            prefetch=prefetch,
            evict_first=evict_first,
            min_pages=min_pages,
            loader_wait=loader_wait,
            k_tile=k_tile,
            p_split=p_split,
            rescale_cols=rescale_cols,
            rescale_log2=rescale_log2,
        )
    else:
        assert not prefetch and k_tile and p_split, (prefetch, k_tile, p_split)
        eps, q_scale, permuted = fold
        kernel = fk.FullAttnSplitKvFoldKernel(
            enable_pdl=enable_pdl,
            eps=eps,
            q_scale=q_scale,
            permuted=permuted,
            evict_first=evict_first,
            min_pages=min_pages,
            loader_wait=loader_wait,
            app_wait=app_wait,
            rescale_cols=rescale_cols,
            rescale_log2=rescale_log2,
        )
    dtypes = (
        cutlass.BFloat16,  # qkv
        cutlass.Float8E4M3FN,  # pool
        cutlass.Int32,  # block table
        cutlass.Int32,  # seq_lens
        cutlass.Float32,  # kv scale
        cutlass.Float16,  # partial O
        cutlass.Float32,  # partial log2-sum-exp
    )
    with torch.cuda.device(device_index):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute.compile(
            kernel,
            *[make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16) for dtype in dtypes],
            *[cutlass.Int32(1)] * 6,
            stream,
            options="--enable-tvm-ffi",
        )


#: (device index, PDL, prefetch, evict_first) builds compiled so far (``compiled_key``), and
#: ``full_attention_prep_fold``'s (the same with its fold key appended); a capture may only launch
#: one of these.
_compiled_keys: set = set()


def compiled_key(
    device_index: int,
    enable_pdl: bool,
    prefetch: bool,
    evict_first: bool,
    rescale_cols: int = HEAD_DIM,
) -> tuple:
    """The key ``_compiled_keys`` holds once that build is compiled: ``(device index, PDL,
    prefetch, evict_first)``, with ``("rescale_cols", n)`` appended for a chunked-rescale build
    (``rescale_cols`` other than 64)."""
    key = (device_index, bool(enable_pdl), bool(prefetch), bool(evict_first))
    if rescale_cols != HEAD_DIM:
        key += (("rescale_cols", int(rescale_cols)),)
    return key


class FullAttnSplitKv(OpWrapper):
    """Full-attention decode partials for one sequence and at most 8 rows.

    Reads Q (BF16, the first 16 heads of the fused operand ``[rows, (16 + 2 + 2) * 64]``) and
    the FP8 paged cache, which must already hold the step's K/V. Writes, into ``part_o``
    ``[128, num_splits, 64]`` FP16 and ``part_lse`` ``[128, num_splits]`` fp32 (row = token
    * 16 + query head; rows of tokens >= rows are not written), each split's normalized
    attention output times the V dequant scale and its log2-sum-exp in units of the score
    scaled by ``log2(e) / 8``; a split with no keys writes ``-inf`` and no O. Returns
    ``(part_o, part_lse)``. ``seq_lens[0]`` is the context length including the step's rows;
    row ``t`` attends to keys ``[0, seq_lens[0] - rows + t]``.

    ``prefetch`` (with ``enable_pdl``; off by default, it does not shorten the critical path)
    streams the pages below the step's first new position before the dependency wait;
    ``evict_first`` loads K/V with the L2 evict-first policy. Neither changes a value.

    Loud preconditions (ValueError): BF16 qkv of 1..8 rows and width 1280 with a unit last
    stride and 16-byte aligned rows; a contiguous FP8 E4M3 pool ``[blocks, 2, 32, 64]``; an
    int32 ``[2, pages]`` block table with a unit last stride and at most 256 pages per split;
    int32 ``seq_lens``; an fp32 scale; ``1 <= num_splits <= 512``; partial buffers of the
    shapes above; every tensor 16-byte aligned on one sm_103 or sm_107 device. The first call
    per device and build compiles the kernel, which must not happen inside a CUDA-graph capture.

    Not certified: more than one sequence per call, other head counts or widths, other page
    sizes, sliding windows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, ctx (= seq_lens[0], the step's rows included); optional splits (default: the
    # device's SM count), pdl, prefetch (default False).
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"{rows} rows, {ctx} keys: {why}", spec=dict(rows=rows, ctx=ctx))
        for rows, ctx, why in (
            (8, 100000, "K=7 verify at a 100k-token context"),
            (8, 455031, "K=7 verify at a 455k-token context, last page partly written"),
            (8, 100003, "the step's tokens straddle a page boundary"),
            (8, 1048581, "near the 1M sequence limit, straddling the last page boundary"),
            (3, 100003, "a partial row group (K=2 verify) across a page boundary"),
            (1, 100000, "one row (plain decode)"),
            (8, 19457, "one page more than 4 pages per split on every SM"),
            (8, 8192, "a short context: 64 splits of one tile, 88 key-less splits"),
            (8, 1027, "a short context: splits of 4 to 5 pages"),
            (8, 161, "two splits"),
            (8, 40, "one split: every other split is key-less"),
            (8, 8, "the first step: every key is new"),
            (1, 1, "the first token: one key"),
        )
    ) + (
        Cell(
            why="8 rows, 100000 keys, PDL launch with the non-default pre-wait page prefetch",
            spec=dict(rows=8, ctx=100000, pdl=True, prefetch=True),
        ),
        Cell(
            why="8 rows, 455031 keys, PDL launch (the op's build)",
            spec=dict(rows=8, ctx=455031, pdl=True),
        ),
        Cell(
            why="8 rows, 100000 keys, two CTAs' worth of splits per SM (304)",
            spec=dict(rows=8, ctx=100000, splits=304),
        ),
        Cell(
            why="8 rows, 100000 keys, a small grid (long splits)",
            spec=dict(rows=8, ctx=100000, splits=19),
        ),
    )

    note = """
    Metadata consumed: seq_lens[0] (device; read by the kernel, never on the host), the
    sequence's K and V page ids, the layer's pool view, the FP8 KV scale (1.0 in the PyTorch
    backend). Grid: num_splits CTAs of 384 threads, one CTA per SM (about 165 KB of shared
    memory and 512 TMEM columns). Call site: the target's 16 full-attention layers on decode
    steps of one sequence and at most 8 rows (fast path full_attention_split_kv), right after
    the full_attention_prep_append folded prep;
    combine_norm_gate consumes the partials.

    Gated by its own tests: the merged partials within a bound of this entry's reference
    derived from the kernel's E4M3 rounding of P; no farther from an fp64 reference than the
    built-in trtllm-gen path, in max and mean, on long contexts and on short ones; NaN bytes in
    the unwritten slots of the last page and in unreferenced pages change no output bit;
    determinism; one captured graph replayed at other contexts (8 ... 1M keys); PDL: late
    writers of the step's own pages, of Q and of older pages are observed, in configurations
    where a build whose loader does not wait (or that streams older pages before the wait) is
    caught.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_splits: int,
        enable_pdl: bool = False,
        prefetch: bool = False,
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = qkv.shape[0] if qkv.dim() == 2 else 0
        if part_o is None or part_lse is None:
            part_o, part_lse = partial_buffers(num_splits, qkv.device)
        tensors = (qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, part_o, part_lse)
        width = block_tables.shape[1] if block_tables.dim() == 2 else 0
        if (
            qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.shape[1] != QKV_WIDTH
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
            or rescale_cols not in RESCALE_COLS
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
                "full_attention_split_kv takes, on one sm_103 or sm_107 device and 16-byte aligned: "
                "BF16 qkv "
                f"[1..{MAX_ROWS}, {QKV_WIDTH}] (unit last stride, rows a multiple of 8 elements "
                f"apart), a contiguous FP8 E4M3 pool [blocks, {KV_HEADS}, {PAGE}, {HEAD_DIM}], an "
                f"int32 [2, pages] block table (unit last stride, at most {MAX_CTA_PAGES} pages "
                f"per split), int32 seq_lens, an fp32 scale, 1..{MAX_SPLITS} splits and contiguous "
                f"partials FP16 [{OUT_ROWS}, splits, {HEAD_DIM}] / fp32 [{OUT_ROWS}, splits]; got "
                f"qkv {qkv.dtype} {tuple(qkv.shape)} strides {tuple(qkv.stride())}, pool "
                f"{kv_pool.dtype} {tuple(kv_pool.shape)}, table {block_tables.dtype} "
                f"{tuple(block_tables.shape)} strides {tuple(block_tables.stride())}, seq_lens "
                f"{seq_lens.dtype}, scale {kv_scale_orig_quant.dtype}, splits {num_splits}, "
                f"partials {part_o.dtype} {tuple(part_o.shape)} / {part_lse.dtype} "
                f"{tuple(part_lse.shape)}, rescale_cols {rescale_cols} (one of {RESCALE_COLS}), "
                f"devices {[str(t.device) for t in tensors]}"
            )
        device_index = qkv.device.index
        key = compiled_key(device_index, enable_pdl, prefetch, evict_first, rescale_cols)
        if torch.cuda.is_current_stream_capturing() and key not in _compiled_keys:
            raise RuntimeError(
                "full_attention_split_kv compiles on its first call per device and build; make that call "
                "before capturing a CUDA graph"
            )
        build = {} if rescale_cols == HEAD_DIM else {"rescale_cols": int(rescale_cols)}
        compiled = _compiled_kernel(*key[:4], **build)
        _compiled_keys.add(key)
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
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_splits: int,
        enable_pdl: bool = False,
        prefetch: bool = False,
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> torch.Tensor:
        """fp32 attention over the cache with the op's Q, as one merged partial per row.

        What the op's partials merge to (``merge_partials``) up to its rounding of P, its FP16
        partial store and its accumulation order: the cache holds E4M3, and Q is the op's
        ``Q_hi + Q_lo``, formed here exactly as the kernel forms it (IEEE fp32 Veltkamp split,
        E4M3 round-to-nearest with saturation). Everything after that is fp32, with TF32 off for
        the two matmuls (``true_fp32_matmul``), and P stays fp32. Returns the fp32 attention
        output ``[rows, 16 * 64]`` (times the V dequant scale), not the partials: how the op
        cuts the keys (``num_splits``), launches (``enable_pdl``, ``prefetch``,
        ``evict_first``) and where it writes (``part_o``, ``part_lse``) change nothing it
        computes, and are ignored here.
        """
        rows = qkv.shape[0]
        length = int(seq_lens[0])
        scale = kv_scale_orig_quant.float()[0]

        def e4m3(x: torch.Tensor) -> torch.Tensor:
            return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        q = qkv[:, : Q_HEADS * HEAD_DIM].float().reshape(rows, Q_HEADS, HEAD_DIM)
        c = q * 1048577.0
        hi = c - (c - q)
        q_eff = e4m3(hi).float() + e4m3(q - hi).float()
        pool_bytes = kv_pool.view(torch.uint8)
        keys = torch.arange(length, device=qkv.device)
        dequant = 1.0 / scale

        def gather(kv: int) -> torch.Tensor:
            raw = pool_bytes[block_tables[kv, keys // PAGE].long(), :, keys % PAGE]
            return raw.view(torch.float8_e4m3fn).float() * dequant  # [keys, 2, 64]

        k, v = gather(0), gather(1)
        past = length - rows
        out = torch.empty((rows, Q_HEADS, HEAD_DIM), dtype=torch.float32, device=qkv.device)
        with true_fp32_matmul():
            for t in range(rows):
                end = past + t + 1
                for h in range(KV_HEADS):
                    qs = q_eff[t, h * Q_PER_KV : (h + 1) * Q_PER_KV]
                    s = qs @ k[:end, h].T / math.sqrt(HEAD_DIM)
                    out[t, h * Q_PER_KV : (h + 1) * Q_PER_KV] = (
                        torch.softmax(s, dim=-1) @ v[:end, h]
                    )
        return out.reshape(rows, -1)

    def is_valid(
        self,
        qkv: torch.Tensor,
        kv_pool: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_scale_orig_quant: torch.Tensor,
        num_splits: int,
        enable_pdl: bool = False,
        prefetch: bool = False,
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> None:
        """Inputs the kernel takes and answers wrongly, with no error.

        A context shorter than the step's rows puts the step's keys at negative positions; a
        block table narrower than the context reads page ids past its end; a page id outside
        the pool reads outside it.
        """
        rows = qkv.shape[0]
        length = int(seq_lens[0])
        if length < rows:
            raise ValueError(f"seq_lens[0] = {length} is shorter than the step's {rows} rows")
        pages = (length + PAGE - 1) // PAGE
        if block_tables.shape[1] < pages:
            raise ValueError(
                f"a [2, {block_tables.shape[1]}] block table cannot address {length} keys "
                f"of {PAGE}-token pages"
            )
        used = block_tables[:, :pages]
        if int(used.min()) < 0 or int(used.max()) >= kv_pool.shape[0]:
            raise ValueError(
                f"page ids [{int(used.min())}, {int(used.max())}] outside the "
                f"{kv_pool.shape[0]}-page pool"
            )


def merge_partials(part_o: torch.Tensor, part_lse: torch.Tensor, rows: int) -> torch.Tensor:
    """The merged attention output ``[rows, 16 * 64]`` fp32 of the op's partials, in torch.

    A test helper and the definition of what ``combine_norm_gate`` merges: weights
    ``2^(lse_s - max lse)`` over the splits with a finite log2-sum-exp, O averaged with them.
    """
    n = rows * Q_HEADS
    lse = part_lse[:n].double()
    o = part_o[:n].double()
    valid = torch.isfinite(lse)
    m = torch.where(valid, lse, torch.full_like(lse, -math.inf)).amax(dim=1, keepdim=True)
    w = torch.where(valid, torch.exp2(lse - m), torch.zeros_like(lse))
    o = torch.where(valid[..., None], o, torch.zeros_like(o))
    merged = (w[..., None] * o).sum(dim=1) / w.sum(dim=1, keepdim=True)
    return merged.reshape(rows, Q_HEADS * HEAD_DIM).float()


full_attention_split_kv = FullAttnSplitKv()
