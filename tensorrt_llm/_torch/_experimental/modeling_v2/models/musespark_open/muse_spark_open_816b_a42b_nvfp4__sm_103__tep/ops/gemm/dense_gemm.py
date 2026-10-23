# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense BF16 projections of a decode step, with the kernel named by the caller.

At tep8 every decoder layer multiplies the step's rows by four BF16 weights per
rank: the fused QKV projection ``[2304, 8192]``, the attention output projection
``[8192, 1024]`` and, with sharded projections, the rank's slices of the MoE
pooler ``[512, 8192]`` and unpooler ``[1024, 4096]``. The built-in low-M
dispatcher races cuBLAS against flashinfer's kernels on each rank separately and
keeps each rank's winner, so the ranks of one engine can run different kernels
for the same layer, and the slowest rank's choice sets every collective's pace.
This entry runs the kernel its caller names instead, so a target that names one
kernel per shape runs it on every rank:

* ``"splitk:<mma_m>,<mma_n>,<split_k>,<stages>"`` -- flashinfer's CuTe-DSL
  cluster split-K kernel. It swaps A and B (the weight's rows are the MMA's M),
  each CTA of a cluster accumulates one K slice in FP32 in TMEM, the peers send
  their partials to the cluster's first CTA through DSMEM, which adds them in
  CTA order and rounds once to BF16. Launched with programmatic dependent launch,
  it issues the weight's TMA loads before ``griddepcontrol.wait`` and the
  activation's after it, so it may be launched early by a predecessor that
  triggers its dependents early.
* ``"direct:<block>,<outputs>"`` -- flashinfer's CuTe-DSL register-prefetch
  (SIMT) kernel: one CTA of ``block`` threads per ``outputs`` weight rows; each
  thread loads its strided share of those rows' K values into registers and
  multiplies them by every activation row (all M rows in one CTA), summing in
  FP32; a warp shuffle and then a sum over the CTA's warps in warp order give
  the dot product, rounded once to BF16. It waits on ``griddepcontrol.wait``
  before any load, so an early trigger hides only its launch latency.
* ``"cublas"`` -- ``torch.mm`` into a fresh output, the call the built-in
  dispatcher's cuBLAS runner makes.

The direct kernel holds every row in one CTA and serves at most ``MAX_ROWS``; the split-K
kernel and cuBLAS also serve the 9..``WIDE_MAX_ROWS``-row steps of a tep4 rank
(``wide_dense_gemms``, ``wide_projection_gathers``).

All accumulate in FP32 and round once; they differ from each other, and the
split-K kernel's tactics with different ``split_k`` differ among themselves, only
in the FP32 summation order. Split-K tactics with the same ``split_k`` sum the
same K slices in the same order, whatever their tile widths, and a row's sum never
involves another row: a split-K call of 9..32 rows gives each row the bits a split-K
call of the same ``split_k`` gives it in a call of at most 8 rows.

``l2_prefetch`` asks the L2 to fetch a tensor without waiting for it: one
``cp.async.bulk.prefetch.L2`` per 32 KiB chunk, from a one-warp-per-32-chunks
grid that exits as soon as the requests are issued. It reads nothing back,
writes nothing, and orders nothing; a later kernel's ordinary loads find the
lines in L2 if they are still there.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    OpWrapper,
    true_fp32_matmul,
)

#: Rows this entry is certified for: one request's last accepted token and up to 7 drafts.
MAX_ROWS = 8
#: Rows the split-K and cuBLAS kernels are also certified for (flashinfer's split-K takes
#: M <= 32): decode steps of up to four requests at K = 7, on the TP4 per-rank shapes.
WIDE_MAX_ROWS = 32
#: The four per-rank projection shapes ``(K, N)`` of this checkpoint at tep8.
QKV_SHAPE = (8192, 2304)
O_PROJ_SHAPE = (1024, 8192)
POOL_SHAPE = (8192, 512)
UNPOOL_SHAPE = (4096, 1024)

CUBLAS = "cublas"
SPLITK = "splitk"
DIRECT = "direct"
_FIELDS = {SPLITK: 4, DIRECT: 2}

#: Bytes one bulk L2 prefetch request covers, and requests per program (one warp).
PREFETCH_CHUNK = 32768
PREFETCH_LANES = 32


@functools.lru_cache(maxsize=None)
def parse_kernel(kernel: str) -> Tuple[str, Tuple[int, ...]]:
    """``(family, fields)``: ``("cublas", ())``, ``("splitk", (mma_m, mma_n, split_k,
    stages))`` or ``("direct", (block, outputs))``.

    Raises:
        ValueError: for any other string.
    """
    if kernel == CUBLAS:
        return CUBLAS, ()
    family, _, rest = kernel.partition(":")
    fields = rest.split(",")
    if family in _FIELDS and len(fields) == _FIELDS[family]:
        if all(f.strip().isdigit() for f in fields):
            return family, tuple(int(f) for f in fields)
    raise ValueError(
        f"unknown dense GEMM kernel {kernel!r}; expected 'cublas', "
        "'splitk:<mma_m>,<mma_n>,<split_k>,<stages>' or 'direct:<block>,<outputs>'"
    )


@functools.lru_cache(maxsize=None)
def _tactic(family: str, fields: Tuple[int, ...], rows: int):
    """flashinfer's tactic object; a direct tactic holds all ``rows`` in one CTA."""
    if family == SPLITK:
        from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as splitk

        return splitk.SplitKTactic(*fields)
    from flashinfer.gemm.kernels import dense_bf16_gemm_direct as direct

    return direct.DirectTactic(*fields, rows)


def kernel_error(kernel: str, rows: int, k: int, n: int) -> Optional[str]:
    """Why ``kernel`` cannot serve ``rows x K -> N``, or ``None``.

    cuBLAS serves every shape; a flashinfer tactic is checked with flashinfer's
    own validation (shared memory, K divisibility, tile and block limits).
    """
    family, fields = parse_kernel(kernel)
    if family == CUBLAS:
        return None
    if family == SPLITK:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import validate_tactic
    else:
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import validate_tactic
    try:
        validate_tactic(_tactic(family, fields, rows), rows, n, k)
    except ValueError as error:
        return str(error)
    return None


def _run(x, weight, out, enable_pdl, family, fields):
    if family == SPLITK:
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import run_splitk_dense

        tactic = _tactic(family, fields, 0)
        return run_splitk_dense(x, weight.t(), None, out, enable_pdl, tactic)
    from flashinfer.gemm.kernels.dense_bf16_gemm_direct import run_direct_dense

    tactic = _tactic(family, fields, int(x.shape[0]))
    return run_direct_dense(x, weight.t(), out, enable_pdl, tactic)


class DenseGemm(OpWrapper):
    """``x @ weight.T`` for ``x [M, K]`` and ``weight [N, K]``, BF16, by the named kernel: M <= 8
    for every kernel, M <= 32 for split-K and cuBLAS."""

    ARCHS = frozenset({Arch.SM_103})

    note = """
    The flashinfer kernels validate their operands themselves (2-D, row- or
    column-major, 32-byte aligned, K a multiple of the tactic's K tile, shared
    memory of the tactic) and raise on anything else, so `is_valid` adds only
    what they accept and answer wrongly. The direct kernel is compiled once per
    row count, since its CTA holds all rows. `enable_pdl` changes when the kernel may start, not what it
    computes; the GPU test drives both. The output is a fresh tensor, so inside a
    CUDA graph capture it comes from the graph's pool like any other activation,
    unless the caller hands one in (`out`, ``unpool_direct``: a transport slot).
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        kernel: str,
        enable_pdl: bool = True,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """... into ``out`` (``[M, N]``, contiguous) when given, else into a fresh tensor.

        The flashinfer kernels store each output element once, from their epilogue
        and after their dependency wait, so ``out`` may be a slot of a multicast
        transport (``unpool_direct``); cuBLAS (``torch.mm``) makes no such promise."""
        if out is None:
            out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
        family, fields = parse_kernel(kernel)
        if family == CUBLAS:
            torch.mm(x, weight.t(), out=out)
        else:
            _run(x, weight, out, enable_pdl, family, fields)
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        kernel: str,
        enable_pdl: bool = True,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The product in float64, rounded once to BF16 (through FP32).

        ``kernel`` and ``enable_pdl`` are ignored: every kernel computes this
        product. Float64 makes the reference strictly the more accurate side; the
        FP32 step before BF16 can differ from a direct float64 -> BF16 rounding
        only when the float64 value lies within 2^-24 relative of a BF16 midpoint.
        Given ``out``, the product is also written into it, as the op does.
        """
        with true_fp32_matmul():
            exact = x.double() @ weight.double().t()
        product = exact.float().to(x.dtype)
        if out is not None:
            out.copy_(product)
        return product

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        kernel: str,
        enable_pdl: bool = True,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        # torch.mm accepts other dtypes and returns their product; the cells,
        # the error budget and the plan's shapes are BF16 only.
        assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, "BF16 operands only"
        assert x.dim() == 2 and weight.dim() == 2, "x [M, K] and weight [N, K]"
        top = MAX_ROWS if parse_kernel(kernel)[0] == DIRECT else WIDE_MAX_ROWS
        assert 1 <= x.shape[0] <= top, f"1 <= M <= {top} rows for {kernel}"
        assert x.shape[1] == weight.shape[1], "weight is [N, K], not its transpose"
        if out is not None:
            # torch.mm would write a strided out in place; the flashinfer kernels
            # address their output as a dense row-major [M, N] matrix.
            assert out.dtype == x.dtype and tuple(out.shape) == (
                x.shape[0],
                weight.shape[0],
            ), "out is [M, N] in the operands' dtype"
            assert out.is_contiguous() and out.data_ptr() % 32 == 0, (
                "a contiguous, 32-byte aligned out"
            )


dense_gemm = DenseGemm()


@triton.jit
def _bulk_prefetch_l2(address, nbytes):
    """One ``cp.async.bulk.prefetch.L2`` per element; ``nbytes <= 0`` issues nothing."""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; .reg .u64 ga; setp.gt.s32 p, $2, 0; cvta.to.global.u64 ga, $1; "
        "@p cp.async.bulk.prefetch.L2.global [ga], $2; mov.u32 $0, 0; }",
        "=r,l,r",
        [address, nbytes],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _l2_prefetch_kernel(base, nbytes, sink_ptr, CHUNK: tl.constexpr, LANES: tl.constexpr):
    lane = (tl.program_id(0) * LANES + tl.arange(0, LANES)).to(tl.int64)
    start = lane * CHUNK
    size = tl.minimum(nbytes - start, CHUNK)
    size = tl.where(size > 0, size, 0).to(tl.int32)
    issued = _bulk_prefetch_l2(base + start, size)
    # The asm always yields 0, so this store never happens; it keeps the requests live.
    tl.store(sink_ptr + lane, issued, mask=issued != 0)


def prefetch_programs(nbytes: int) -> int:
    """Programs ``l2_prefetch`` launches for ``nbytes``."""
    return triton.cdiv(triton.cdiv(nbytes, PREFETCH_CHUNK), PREFETCH_LANES)


class L2Prefetch(OpWrapper):
    """Ask the L2 to hold ``tensor``'s bytes soon; return without waiting for them."""

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Nothing observable is computed, so `reference` returns None and the test
    proves that memory is unchanged, that the compiled kernel issues the bulk
    prefetches, and that a GEMM after it computes the same bits with and without
    it. A prefetch is a hint: the L2 may drop requests or evict lines before use.
    """

    def __call__(self, tensor: torch.Tensor, sink: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        _l2_prefetch_kernel[(prefetch_programs(nbytes),)](
            tensor.data_ptr(),
            nbytes,
            sink,
            CHUNK=PREFETCH_CHUNK,
            LANES=PREFETCH_LANES,
            num_warps=1,
        )

    def reference(self, tensor: torch.Tensor, sink: torch.Tensor) -> None:
        """Nothing: a prefetch computes no value. ``tensor`` and ``sink`` stay unchanged."""
        return None

    def is_valid(self, tensor: torch.Tensor, sink: torch.Tensor) -> None:
        # The bulk prefetch needs a 16-byte aligned address and a size that is a
        # multiple of 16; a misaligned request faults or fetches other lines.
        assert tensor.is_cuda and tensor.is_contiguous(), "a contiguous CUDA tensor"
        assert tensor.data_ptr() % 16 == 0, "16-byte aligned start"
        assert (tensor.numel() * tensor.element_size()) % 16 == 0, "size a multiple of 16 bytes"
        assert sink.is_cuda and sink.dtype == torch.int32, "an int32 CUDA sink"


l2_prefetch = L2Prefetch()


#: K columns per k-tile of flashinfer's split-K kernel (its ``_CTA_K``): one weight
#: TMA load per CTA and k-tile covers ``mma_m`` rows x this many columns.
SPLITK_CTA_K = 128


def post_wait_columns(kernel: str, k: int) -> Optional[Tuple[int, int, int, int]]:
    """The weight columns a split-K ``kernel`` loads only after its dependency wait, per row.

    flashinfer's split-K kernel gives each CTA of a cluster one of ``split_k``
    equal K slices; its weight warp issues the slice's k-tiles in order without
    waiting while the ``stages``-deep ring has room, and the k-tiles after the
    first ``stages`` only as the MMA frees stages, which it does only after the
    activation arrives, i.e. after ``griddepcontrol.wait``. Returns
    ``(offset, length, stride, count)`` in columns: in every weight row the
    ranges ``[offset + j * stride, offset + j * stride + length)`` for ``j <
    count``. ``None`` when nothing is loaded after the wait (a slice of at most
    ``stages`` k-tiles) or for a kernel that is not split-K (the direct kernel
    waits before any load; cuBLAS has no early start).
    """
    family, fields = parse_kernel(kernel)
    if family != SPLITK:
        return None
    _, _, split_k, stages = fields
    slice_columns = k // split_k
    tiles = slice_columns // SPLITK_CTA_K
    if tiles <= stages:
        return None
    offset = stages * SPLITK_CTA_K
    return offset, slice_columns - offset, slice_columns, split_k
