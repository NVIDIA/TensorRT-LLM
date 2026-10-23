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
That is the tep8 rank's shape (16 query / 2 KV heads). A tep4 rank's 4 KV heads are two such
pairs: the grid is ``splits x 2`` CTAs with half as many splits, and every CTA computes what the
tep8 build computes on its pair's slice of the operand, the pool and the partials.

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
#: Output rows of a tep8 step: (token, query head) pairs.
OUT_ROWS = MAX_ROWS * Q_HEADS
#: KV heads of one rank per tensor-parallel size the kernel is built for (16 KV heads in the
#: model): one pair at tep8, two at tep4. Any other rank shape has no build.
RANK_KV_HEADS = {8: KV_HEADS, 4: 2 * KV_HEADS}
#: Largest grid the combine reads partials for.
MAX_SPLITS = 512
#: Page ids one CTA stages in shared memory per table row in the tep8 build (``cta_pages``).
MAX_CTA_PAGES = 256
#: Fewest pages per keyed split: a context of n pages has min(splits, ceil(n / MIN_PAGES)) keyed
#: splits, the first ones; the rest are key-less.
MIN_PAGES = 4
#: O columns the kernel's rescale holds in registers at once: 64 (the whole head) is the default
#: build, which spills part of S on every tile; 16 is the spill-free build.
RESCALE_COLS = (16, 32, HEAD_DIM)
SPILL_FREE_RESCALE_COLS = 16
#: Sequences one call of the multi-sequence entry (``full_attention_split_kv_multi``) takes.
MULTI_MAX_REQUESTS = 4
#: How many times ``cta_pages`` one CTA of the multi-sequence build stages: a sequence's share
#: of the splits follows its share of the pages, so a split may cover up to
#: ``requests * pages / (splits - requests)`` of one sequence's pages.
MULTI_CTA_PAGES_FACTOR = 4


def q_heads(kv_heads: int) -> int:
    """Query heads of a rank with ``kv_heads`` KV heads."""
    return kv_heads * Q_PER_KV


def qkv_width(kv_heads: int) -> int:
    """Width of the fused-QKV operand row of a rank with ``kv_heads`` KV heads."""
    return (q_heads(kv_heads) + 2 * kv_heads) * HEAD_DIM


def out_rows(kv_heads: int) -> int:
    """Partial rows of a step of a rank with ``kv_heads`` KV heads: (token, query head) pairs."""
    return MAX_ROWS * q_heads(kv_heads)


def cta_pages(kv_heads: int) -> int:
    """Page ids one CTA of the ``kv_heads`` build stages per table row: 256 at tep8, 512 at tep4
    (whose pairs of KV heads split the SMs, so each CTA covers twice the pages of a tep8 CTA at
    the same context)."""
    if kv_heads == RANK_KV_HEADS[8]:
        return MAX_CTA_PAGES
    elif kv_heads == RANK_KV_HEADS[4]:
        return 2 * MAX_CTA_PAGES
    raise ValueError(f"full_attention_split_kv has no build for {kv_heads} KV heads per rank")


def default_splits(kv_heads: int, sm_count: int) -> int:
    """The splits of one pair of KV heads the target launches: one CTA per SM over the grid's
    ``splits x pairs``: every SM at tep8 (one pair), half of them at tep4 (two pairs)."""
    if kv_heads == RANK_KV_HEADS[8]:
        return sm_count
    elif kv_heads == RANK_KV_HEADS[4]:
        return sm_count // 2
    raise ValueError(f"full_attention_split_kv has no build for {kv_heads} KV heads per rank")


def keyed_splits(seq_len: int, num_splits: int) -> int:
    """How many of ``num_splits`` splits hold keys at context ``seq_len`` (the kernel's
    ``eff``; splits ``>= keyed_splits`` are key-less)."""
    pages = (seq_len + PAGE - 1) // PAGE
    return min(num_splits, (pages + MIN_PAGES - 1) // MIN_PAGES)


def request_splits(lengths, num_splits: int) -> list:
    """``(first split, splits)`` of each sequence of a multi-sequence call at contexts
    ``lengths``: the kernel's ``request_of``. Sequence ``i`` of ``n_i`` pages (``N`` in all)
    starts at ``i + (num_splits - R) * (n_0 + ... + n_{i-1}) // N``; each has at least one."""
    pages = [(int(length) + PAGE - 1) // PAGE for length in lengths]
    total = sum(pages)
    reqs = len(pages)
    starts = [i + (num_splits - reqs) * sum(pages[:i]) // total for i in range(reqs)]
    ends = starts[1:] + [num_splits]
    return [(lo, hi - lo) for lo, hi in zip(starts, ends)]


def partial_buffers(
    num_splits: int, device, kv_heads: int = KV_HEADS, reqs: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uninitialized partial buffers for ``num_splits`` splits: FP16 O ``[8 hq, splits, 64]``
    and fp32 log2-sum-exp ``[8 hq, splits]`` (row = token * hq + query head; hq = 16 at tep8,
    32 at tep4)."""
    rows = reqs * out_rows(kv_heads)
    part_o = torch.empty(rows, num_splits, HEAD_DIM, dtype=torch.float16, device=device)
    part_lse = torch.empty(rows, num_splits, dtype=torch.float32, device=device)
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
    kv_heads: int = KV_HEADS,
    reqs: int = 1,
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
    two pages can sit in two CTAs) exists for that op's tests too; the ops build neither.
    ``kv_heads`` is the rank's KV heads (``RANK_KV_HEADS``: 2 at tep8, 4 at tep4); ``reqs``
    above 1 is the multi-sequence build (``full_attention_split_kv_multi``)."""
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
            kv_heads=kv_heads,
            max_cta_pages=cta_pages(kv_heads) * (MULTI_CTA_PAGES_FACTOR if reqs > 1 else 1),
            **({"reqs": reqs} if reqs > 1 else {}),
        )
    else:
        assert not prefetch and k_tile and p_split and reqs == 1, (prefetch, k_tile, p_split, reqs)
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
            kv_heads=kv_heads,
            max_cta_pages=cta_pages(kv_heads),
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
    kv_heads: int = KV_HEADS,
    reqs: int = 1,
) -> tuple:
    """The key ``_compiled_keys`` holds once that build is compiled: ``(device index, PDL,
    prefetch, evict_first)``, with ``("rescale_cols", n)`` appended for a chunked-rescale build
    (``rescale_cols`` other than 64), ``("kv_heads", n)`` for the tep4 build and ``("reqs", n)``
    for the multi-sequence build of ``n`` sequences."""
    key = (device_index, bool(enable_pdl), bool(prefetch), bool(evict_first))
    if rescale_cols != HEAD_DIM:
        key += (("rescale_cols", int(rescale_cols)),)
    if kv_heads != KV_HEADS:
        key += (("kv_heads", int(kv_heads)),)
    if reqs != 1:
        key += (("reqs", int(reqs)),)
    return key


def build_options(key: tuple) -> dict:
    """The keyword options of ``_compiled_kernel`` a key's suffix (past its first four or, for
    the fold, five entries) names."""
    return {name: value for name, value in key if name in ("rescale_cols", "kv_heads", "reqs")}


class FullAttnSplitKv(OpWrapper):
    """Full-attention decode partials for one sequence and at most 8 rows.

    The rank's shape is the pool's KV heads ``G`` (``kv_pool.shape[1]``): 2 at tep8, 4 at tep4
    (``RANK_KV_HEADS``), with ``H = 8 G`` query heads. Reads Q (BF16, the first ``H`` heads of
    the fused operand ``[rows, (H + 2 G) * 64]``) and the FP8 paged cache, which must already
    hold the step's K/V. Writes, into ``part_o`` ``[8 H, num_splits, 64]`` FP16 and
    ``part_lse`` ``[8 H, num_splits]`` fp32 (row = token * H + query head; rows of tokens >=
    rows are not written; ``num_splits`` is per pair of KV heads, the grid
    ``num_splits x G / 2``), each split's normalized
    attention output times the V dequant scale and its log2-sum-exp in units of the score
    scaled by ``log2(e) / 8``; a split with no keys writes ``-inf`` and no O. Returns
    ``(part_o, part_lse)``. ``seq_lens[0]`` is the context length including the step's rows;
    row ``t`` attends to keys ``[0, seq_lens[0] - rows + t]``.

    ``prefetch`` (with ``enable_pdl``; off by default, it does not shorten the critical path)
    streams the pages below the step's first new position before the dependency wait;
    ``evict_first`` loads K/V with the L2 evict-first policy. Neither changes a value.

    Loud preconditions (ValueError): BF16 qkv of 1..8 rows and width ``(H + 2 G) * 64`` (1280
    at tep8, 2560 at tep4) with a unit last stride and 16-byte aligned rows; a contiguous FP8
    E4M3 pool ``[blocks, G, 32, 64]`` with G 2 or 4; an int32 ``[2, pages]`` block table with
    a unit last stride and at most ``cta_pages(G)`` (256 at tep8, 512 at tep4) pages per split;
    int32 ``seq_lens``; an fp32 scale; ``1 <= num_splits <= 512``; partial buffers of the
    shapes above; every tensor 16-byte aligned on one sm_103 or sm_107 device. The first call
    per device and build compiles the kernel, which must not happen inside a CUDA-graph capture.

    Not certified: more than one sequence per call, other head counts or widths, other page
    sizes, sliding windows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The tep4 cells (kv_heads=4) are certified by the tep4 op test.

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
        kv_heads = kv_pool.shape[1] if kv_pool.dim() == 4 else 0
        if kv_heads not in RANK_KV_HEADS.values():
            raise ValueError(
                "full_attention_split_kv takes a pool [blocks, G, "
                f"{PAGE}, {HEAD_DIM}] with G KV heads in {sorted(RANK_KV_HEADS.values())}; got "
                f"{tuple(kv_pool.shape)}"
            )
        if part_o is None or part_lse is None:
            part_o, part_lse = partial_buffers(num_splits, qkv.device, kv_heads)
        tensors = (qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, part_o, part_lse)
        width = block_tables.shape[1] if block_tables.dim() == 2 else 0
        if (
            qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.shape[1] != qkv_width(kv_heads)
            or qkv.stride(-1) != 1
            or qkv.stride(0) % 8 != 0
            or not 1 <= rows <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or kv_pool.dim() != 4
            or tuple(kv_pool.shape[2:]) != (PAGE, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or block_tables.dtype != torch.int32
            or block_tables.dim() != 2
            or block_tables.shape[0] != 2
            or width < 1
            or block_tables.stride(-1) != 1
            or not 1 <= num_splits <= MAX_SPLITS
            or rescale_cols not in RESCALE_COLS
            or -(-width // num_splits) > cta_pages(kv_heads)
            or seq_lens.dtype != torch.int32
            or seq_lens.numel() < 1
            or kv_scale_orig_quant.dtype != torch.float32
            or kv_scale_orig_quant.numel() < 1
            or part_o.dtype != torch.float16
            or tuple(part_o.shape) != (out_rows(kv_heads), num_splits, HEAD_DIM)
            or not part_o.is_contiguous()
            or part_lse.dtype != torch.float32
            or tuple(part_lse.shape) != (out_rows(kv_heads), num_splits)
            or not part_lse.is_contiguous()
            or any(t.device != qkv.device or t.data_ptr() % 16 for t in tensors)
            or qkv.device.type != "cuda"
            or torch.cuda.get_device_capability(qkv.device) not in ((10, 3), (10, 7))
        ):
            raise ValueError(
                "full_attention_split_kv takes, on one sm_103 or sm_107 device and 16-byte aligned: a "
                "contiguous FP8 E4M3 pool [blocks, G, "
                f"{PAGE}, {HEAD_DIM}] with G KV heads in {sorted(RANK_KV_HEADS.values())}, BF16 "
                f"qkv [1..{MAX_ROWS}, (8 G + 2 G) * {HEAD_DIM}] (unit last stride, rows a multiple "
                "of 8 elements apart), an int32 [2, pages] block table (unit last stride, at most "
                f"{cta_pages(kv_heads)} pages per split), int32 seq_lens, an fp32 "
                f"scale, 1..{MAX_SPLITS} splits and contiguous partials FP16 [64 G, splits, "
                f"{HEAD_DIM}] / fp32 [64 G, splits]; got "
                f"qkv {qkv.dtype} {tuple(qkv.shape)} strides {tuple(qkv.stride())}, pool "
                f"{kv_pool.dtype} {tuple(kv_pool.shape)}, table {block_tables.dtype} "
                f"{tuple(block_tables.shape)} strides {tuple(block_tables.stride())}, seq_lens "
                f"{seq_lens.dtype}, scale {kv_scale_orig_quant.dtype}, splits {num_splits}, "
                f"partials {part_o.dtype} {tuple(part_o.shape)} / {part_lse.dtype} "
                f"{tuple(part_lse.shape)}, rescale_cols {rescale_cols} (one of {RESCALE_COLS}), "
                f"devices {[str(t.device) for t in tensors]}"
            )
        device_index = qkv.device.index
        key = compiled_key(device_index, enable_pdl, prefetch, evict_first, rescale_cols, kv_heads)
        if torch.cuda.is_current_stream_capturing() and key not in _compiled_keys:
            raise RuntimeError(
                "full_attention_split_kv compiles on its first call per device and build; make that call "
                "before capturing a CUDA graph"
            )
        compiled = _compiled_kernel(*key[:4], **build_options(key[4:]))
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
        output ``[rows, H * 64]`` (times the V dequant scale), not the partials: how the op
        cuts the keys (``num_splits``), launches (``enable_pdl``, ``prefetch``,
        ``evict_first``) and where it writes (``part_o``, ``part_lse``) change nothing it
        computes, and are ignored here.
        """
        rows = qkv.shape[0]
        kv_heads = kv_pool.shape[1]
        heads = q_heads(kv_heads)
        length = int(seq_lens[0])
        scale = kv_scale_orig_quant.float()[0]

        def e4m3(x: torch.Tensor) -> torch.Tensor:
            return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        q = qkv[:, : heads * HEAD_DIM].float().reshape(rows, heads, HEAD_DIM)
        c = q * 1048577.0
        hi = c - (c - q)
        q_eff = e4m3(hi).float() + e4m3(q - hi).float()
        pool_bytes = kv_pool.view(torch.uint8)
        keys = torch.arange(length, device=qkv.device)
        dequant = 1.0 / scale

        def gather(kv: int) -> torch.Tensor:
            raw = pool_bytes[block_tables[kv, keys // PAGE].long(), :, keys % PAGE]
            return raw.view(torch.float8_e4m3fn).float() * dequant  # [keys, G, 64]

        k, v = gather(0), gather(1)
        past = length - rows
        out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.float32, device=qkv.device)
        with true_fp32_matmul():
            for t in range(rows):
                end = past + t + 1
                for h in range(kv_heads):
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


class FullAttnSplitKvMulti(OpWrapper):
    """``full_attention_split_kv`` for a decode step of 2..4 sequences of equal row count, one launch.

    ``qkv`` ``[R * M, (H + 2 G) * 64]`` holds the sequences' rows one after another (``M`` <= 8
    each); ``block_tables`` ``[R, 2, pages]`` (each sequence's K and V page ids, rows of one
    sequence ``2 * table row stride`` apart, as the TRT-LLM table lays them out) and
    ``seq_lens[r]`` (its context, its rows included) are per sequence. The grid is the
    one-sequence call's ``num_splits x G / 2``; sequence ``r`` takes the splits
    ``request_splits`` gives it (at least one, the rest in proportion to its pages) and every
    CTA computes exactly what the one-sequence kernel computes for that sequence, over that
    sequence's splits. Partials: ``part_o`` ``[R * 8 H, num_splits, 64]`` and ``part_lse``
    ``[R * 8 H, num_splits]``, sequence ``r``'s rows at ``r * 8 H + token * H + query head`` and
    its splits at ``0 .. its splits - 1`` of the split axis (the rest of its split axis
    unwritten); ``combine_norm_gate`` with ``requests`` merges them.

    Numerics: per sequence bit for bit the one-sequence call's partials at a ``num_splits``
    equal to that sequence's splits; deterministic.

    Loud preconditions (ValueError): the one-sequence call's, with ``R`` in 2..4, ``R * M`` rows,
    an int32 ``[R, 2, pages]`` table (unit last stride, a sequence's K and V rows adjacent) with
    at most ``MULTI_CTA_PAGES_FACTOR * cta_pages(G)`` pages per split in the worst case
    (``R * pages / (num_splits - R)``), ``num_splits > R`` and partials of the shapes above.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: seq_lens[0..R-1] (device, read by the kernel before its dependency wait,
    like the one-sequence call's seq_lens[0]), the sequences' K and V page ids, the layer's pool
    view, the FP8 KV scale. Call site: the target's 16 full-attention layers on tep4 decode
    steps of 2..4 sequences (fast path wide_full_attention), right after the multi-sequence
    folded prep (preattn_prep_append with an [R, 2, pages] table); combine_norm_gate with
    requests consumes the partials.

    Gated by its own tests: each sequence's partials bit for bit the one-sequence kernel's at
    that sequence's split count, and the merged output against the one-sequence reference
    within that op's band; key-less splits; determinism; a captured graph replayed at other
    contexts per sequence.
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
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reqs = block_tables.shape[0] if block_tables.dim() == 3 else 0
        kv_heads = kv_pool.shape[1] if kv_pool.dim() == 4 else 0
        if kv_heads not in RANK_KV_HEADS.values() or not 2 <= reqs <= MULTI_MAX_REQUESTS:
            raise ValueError(
                "full_attention_split_kv_multi takes a pool [blocks, G, "
                f"{PAGE}, {HEAD_DIM}] with G in {sorted(RANK_KV_HEADS.values())} and an int32 "
                f"[R, 2, pages] table with R in 2..{MULTI_MAX_REQUESTS}; got pool "
                f"{tuple(kv_pool.shape)}, table {tuple(block_tables.shape)}"
            )
        if part_o is None or part_lse is None:
            part_o, part_lse = partial_buffers(num_splits, qkv.device, kv_heads, reqs)
        tensors = (qkv, kv_pool, block_tables, seq_lens, kv_scale_orig_quant, part_o, part_lse)
        width = block_tables.shape[2]
        total_rows = qkv.shape[0] if qkv.dim() == 2 else 0
        rows = total_rows // reqs
        stages = MULTI_CTA_PAGES_FACTOR * cta_pages(kv_heads)
        if (
            qkv.dtype != torch.bfloat16
            or qkv.dim() != 2
            or qkv.shape[1] != qkv_width(kv_heads)
            or qkv.stride(-1) != 1
            or qkv.stride(0) % 8 != 0
            or total_rows % reqs != 0
            or not 1 <= rows <= MAX_ROWS
            or kv_pool.dtype != torch.float8_e4m3fn
            or tuple(kv_pool.shape[2:]) != (PAGE, HEAD_DIM)
            or not kv_pool.is_contiguous()
            or block_tables.dtype != torch.int32
            or block_tables.shape[1] != 2
            or width < 1
            or block_tables.stride(-1) != 1
            or block_tables.stride(0) != 2 * block_tables.stride(1)
            or not reqs < num_splits <= MAX_SPLITS
            or rescale_cols not in RESCALE_COLS
            or -(-reqs * width // (num_splits - reqs)) > stages
            or seq_lens.dtype != torch.int32
            or seq_lens.numel() < reqs
            or kv_scale_orig_quant.dtype != torch.float32
            or kv_scale_orig_quant.numel() < 1
            or part_o.dtype != torch.float16
            or tuple(part_o.shape) != (reqs * out_rows(kv_heads), num_splits, HEAD_DIM)
            or not part_o.is_contiguous()
            or part_lse.dtype != torch.float32
            or tuple(part_lse.shape) != (reqs * out_rows(kv_heads), num_splits)
            or not part_lse.is_contiguous()
            or any(t.device != qkv.device or t.data_ptr() % 16 for t in tensors)
            or qkv.device.type != "cuda"
            or torch.cuda.get_device_capability(qkv.device) not in ((10, 3), (10, 7))
        ):
            raise ValueError(
                "full_attention_split_kv_multi takes, on one sm_103 or sm_107 device and 16-byte "
                f"aligned: the one-sequence call's pool, BF16 qkv [R x 1..{MAX_ROWS}, (8 G + 2 G) "
                f"* {HEAD_DIM}], an int32 [R, 2, pages] table (K and V rows adjacent, at most "
                f"{stages} pages per split in the worst case), int32 seq_lens of at least R, an "
                f"fp32 scale, R < splits <= {MAX_SPLITS} and partials FP16 [R 64 G, splits, "
                f"{HEAD_DIM}] / fp32 [R 64 G, splits]; got qkv {qkv.dtype} {tuple(qkv.shape)} "
                f"strides {tuple(qkv.stride())}, pool {kv_pool.dtype} {tuple(kv_pool.shape)}, "
                f"table {block_tables.dtype} {tuple(block_tables.shape)} strides "
                f"{tuple(block_tables.stride())}, seq_lens {seq_lens.dtype}"
                f"[{seq_lens.numel()}], splits {num_splits}, partials {part_o.dtype} "
                f"{tuple(part_o.shape)} / {part_lse.dtype} {tuple(part_lse.shape)}, "
                f"rescale_cols {rescale_cols}, devices {[str(t.device) for t in tensors]}"
            )
        key = compiled_key(
            qkv.device.index, enable_pdl, False, evict_first, rescale_cols, kv_heads, reqs
        )
        if torch.cuda.is_current_stream_capturing() and key not in _compiled_keys:
            raise RuntimeError(
                "full_attention_split_kv_multi compiles on its first call per device and build; "
                "make that call before capturing a CUDA graph"
            )
        compiled = _compiled_kernel(*key[:4], **build_options(key[4:]))
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
            block_tables.stride(1),
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
        evict_first: bool = True,
        part_o: Optional[torch.Tensor] = None,
        part_lse: Optional[torch.Tensor] = None,
        rescale_cols: int = HEAD_DIM,
    ) -> torch.Tensor:
        """The one-sequence reference of each sequence, rows stacked: ``[R * M, H * 64]`` fp32."""
        reqs = block_tables.shape[0]
        rows = qkv.shape[0] // reqs
        return torch.cat(
            [
                full_attention_split_kv.reference(
                    qkv[r * rows : (r + 1) * rows],
                    kv_pool,
                    block_tables[r],
                    seq_lens[r : r + 1],
                    kv_scale_orig_quant,
                    num_splits,
                )
                for r in range(reqs)
            ]
        )

    def is_valid(
        self,
        qkv: torch.Tensor,
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
        """Each sequence as the one-sequence call's ``is_valid`` checks it."""
        reqs = block_tables.shape[0]
        rows = qkv.shape[0] // reqs
        for r in range(reqs):
            full_attention_split_kv.is_valid(
                qkv[r * rows : (r + 1) * rows],
                kv_pool,
                block_tables[r],
                seq_lens[r : r + 1],
                kv_scale_orig_quant,
                num_splits,
            )


def merge_partials(part_o: torch.Tensor, part_lse: torch.Tensor, rows: int) -> torch.Tensor:
    """The merged attention output ``[rows, H * 64]`` fp32 of the op's partials, in torch (H
    query heads: ``part_o.shape[0] // 8``, 16 at tep8, 32 at tep4).

    A test helper and the definition of what ``combine_norm_gate`` merges: weights
    ``2^(lse_s - max lse)`` over the splits with a finite log2-sum-exp, O averaged with them.
    """
    heads = part_o.shape[0] // MAX_ROWS
    n = rows * heads
    lse = part_lse[:n].double()
    o = part_o[:n].double()
    valid = torch.isfinite(lse)
    m = torch.where(valid, lse, torch.full_like(lse, -math.inf)).amax(dim=1, keepdim=True)
    w = torch.where(valid, torch.exp2(lse - m), torch.zeros_like(lse))
    o = torch.where(valid[..., None], o, torch.zeros_like(o))
    merged = (w[..., None] * o).sum(dim=1) / w.sum(dim=1, keepdim=True)
    return merged.reshape(rows, heads * HEAD_DIM).float()


full_attention_split_kv = FullAttnSplitKv()
full_attention_split_kv_multi = FullAttnSplitKvMulti()
