# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/642c18d64c4cb6db3ded8fbe1786718cd4005fcb/vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/common/prefetch.py
"""L2 prefetch of a persistent device table of weight byte ranges (SM100/SM103/SM107).

This is the weight prefetch kernel of vLLM's Kimi-K3 specialized decode (``L2Prefetch``,
``specialized_decode/cutedsl/common/prefetch.py``), written with the inline-asm primitives of
:mod:`k3_l2_prefetch`. Each one-warp CTA walks a device table of ``(address, bytes)`` rows in
table order and issues ``cp.async.bulk.prefetch.L2`` for :data:`CHUNK_BYTES` chunks. The chunks of
a row go to the grid's warps round robin, and each row starts at the warp after the one that took
the previous row's last chunk. Many small rows (block scales, small projections) are therefore
spread over the whole grid instead of all landing on the first warps.

Compared with :func:`k3_l2_prefetch.run_k3_l2_prefetch`, which passes at most eight ranges by
value and restarts every range at warp 0:

* one launch covers any number of ranges, so a whole layer's weights and their scales (a byte cap
  applied in consumption order, see :func:`capped_ranges`) go out in one launch;
* the ranges live in a table owned by :class:`L2PrefetchTable`. A captured CUDA graph reads the
  same table on every replay, and the launch passes only the table address and the row count;
* the launch can take the max-shared carveout. Without a preference its CTAs still fit on SMs
  whose shared memory is split for a GEMM running there, but an idle SM a CTA lands on gets a
  small split, and a large-shared-memory kernel launched next cannot place a CTA on that SM until
  the prefetch exits. With carveout 100 the SM keeps the maximum split.

The prefetches are hints: nothing waits for them and no result depends on them. Launch the kernel
on a side stream behind an event and without the PDL attribute, so its incoming dependencies stay
full ones. It triggers its programmatic dependents at entry and waits for its own programmatic
predecessor at exit (a no-op without the PDL attribute). A kernel launched after it with the PDL
attribute on the same stream therefore cannot run ahead of this kernel's predecessor.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .k3_l2_prefetch import _bulk_prefetch_l2, _griddepcontrol_launch_dependents

#: Bytes per ``cp.async.bulk.prefetch.L2`` instruction.
CHUNK_BYTES = 16 << 10
#: Default CTAs (one warp each) of a launch.
DEFAULT_CTAS = 32
_THREADS = 32
#: Bytes of one table row: ``int64`` address, then ``int64`` byte count.
_ROW_BYTES = 16


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _ld_s64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.s64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_s64(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.s64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


class K3L2PrefetchTableKernel:
    """``ctas`` one-warp CTAs prefetch the ``count`` rows of an ``int64 [count, 2]`` table of
    ``(address, bytes)`` (16-byte aligned addresses, sizes multiples of 16) in
    :data:`CHUNK_BYTES` chunks.

    Chunk ``i`` of the whole table (rows in order, chunks of a row in order) goes to warp
    ``i % (ctas * 32)``. With ``record``, the warp that issues chunk ``i`` also stores its
    ``(address, bytes)`` at row ``i`` of an ``int64 [chunks, 2]`` trace; tests use the trace to
    check which bytes a table covers.
    """

    def __init__(self, ctas: int, carveout: Optional[int] = None, record: bool = False):
        if ctas < 1:
            raise ValueError(f"ctas={ctas} must be positive")
        if carveout is not None and not 0 <= carveout <= 100:
            raise ValueError(f"carveout={carveout} must be a percentage")
        self.ctas = ctas
        self.carveout = carveout
        self.record = record

    @cute.jit
    def __call__(self, table: Int64, count: Int32, trace: Int64, stream: cuda.CUstream):
        self.kernel(table, count, trace).launch(
            grid=(self.ctas, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
            preferred_smem_carveout=self.carveout,
        )

    @cute.kernel
    def kernel(self, table: Int64, count: Int32, trace: Int64):
        lane, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        # Nothing reads what this kernel does: a programmatic dependent need not wait for it.
        _griddepcontrol_launch_dependents()
        workers = self.ctas * _THREADS
        worker = Int32(cta * _THREADS + lane)
        # Chunks of the rows before the current one.
        phase = Int32(0)
        row = Int32(0)
        while row < count:
            entry = table + Int64(row) * _ROW_BYTES
            base = _ld_s64(entry)
            nbytes = _ld_s64(entry + 8)
            chunks = Int32((nbytes + (CHUNK_BYTES - 1)) // CHUNK_BYTES)
            # The first chunk of this row that falls to this warp.
            index = (worker + workers - phase % workers) % workers
            while index < chunks:
                offset = Int64(index) * CHUNK_BYTES
                size = nbytes - offset
                if size > CHUNK_BYTES:
                    size = Int64(CHUNK_BYTES)
                _bulk_prefetch_l2(base + offset, Int32(size))
                if cutlass.const_expr(self.record):
                    slot = trace + Int64(phase + index) * _ROW_BYTES
                    _st_s64(slot, base + offset)
                    _st_s64(slot + 8, size)
                index += workers
            phase += chunks
            row += 1
        # Without the PDL attribute this returns at once; with it, a later PDL launch on the same
        # stream could otherwise start before this kernel's own predecessor completes.
        _griddepcontrol_wait()


@functools.cache
def _compiled(ctas: int, carveout: Optional[int], record: bool):
    zero = Int64(0)
    return cute.compile(
        K3L2PrefetchTableKernel(ctas, carveout, record),
        zero,
        Int32(0),
        zero,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def capped_ranges(ranges: Sequence[tuple[int, int]], byte_cap: int = 0) -> list[tuple[int, int]]:
    """The first ``byte_cap`` bytes of ``ranges`` in order (the last range cut down to a multiple
    of 16 bytes; ``byte_cap <= 0``: all of them). Empty ranges are dropped."""
    out = []
    remaining = byte_cap
    for address, nbytes in ranges:
        if nbytes <= 0:
            continue
        if byte_cap > 0:
            nbytes = min(nbytes, remaining) // 16 * 16
            remaining -= nbytes
        if nbytes > 0:
            out.append((address, nbytes))
        if byte_cap > 0 and remaining < 16:
            break
    return out


def table_chunks(ranges: Sequence[tuple[int, int]]) -> int:
    """Number of :data:`CHUNK_BYTES` chunks a table of ``ranges`` issues."""
    return sum((n + CHUNK_BYTES - 1) // CHUNK_BYTES for _, n in ranges)


class L2PrefetchTable:
    """A device table of ``(address, bytes)`` ranges and the compiled kernel that prefetches it.

    The table lives as long as this object; keep the object (and the tensors the ranges point
    into) alive for as long as a CUDA graph that launched it may replay. Build it outside graph
    capture: the constructor copies the table to the device and may compile the kernel.

    Args:
        ranges: ``(address, bytes)`` pairs, addresses and sizes multiples of 16, sizes positive.
        device: CUDA device of the table.
        ctas: one-warp CTAs of each launch.
        carveout: preferred shared-memory carveout of the launch in percent (``None``: the
            driver's default).
        record: compile the tracing variant (:class:`K3L2PrefetchTableKernel`).
    """

    def __init__(
        self,
        ranges: Sequence[tuple[int, int]],
        device: torch.device,
        *,
        ctas: int = DEFAULT_CTAS,
        carveout: Optional[int] = None,
        record: bool = False,
    ):
        ranges = tuple((int(a), int(n)) for a, n in ranges)
        if not ranges:
            raise ValueError("an L2 prefetch table needs at least one range")
        for address, nbytes in ranges:
            if address <= 0 or address % 16 or nbytes <= 0 or nbytes % 16:
                raise ValueError(f"range ({address:#x}, {nbytes}) is not 16-byte aligned")
        self.ranges = ranges
        self.ctas = ctas
        self.carveout = carveout
        self.record = record
        self.nbytes = sum(n for _, n in ranges)
        self.chunks = table_chunks(ranges)
        self.table = torch.tensor(ranges, dtype=torch.int64).to(device)
        self._kernel = _compiled(ctas, carveout, record)

    def launch(self, trace: Optional[torch.Tensor] = None) -> None:
        """Prefetch the table's ranges on the current stream. ``trace``: the ``int64 [chunks, 2]``
        output of a ``record`` table."""
        if self.record != (trace is not None):
            raise ValueError("a trace goes with a record table, and only with one")
        if trace is not None and (
            trace.dtype != torch.int64 or trace.numel() < 2 * self.chunks or not trace.is_cuda
        ):
            raise ValueError(f"the trace must be a CUDA int64 [{self.chunks}, 2] tensor")
        self._kernel(
            Int64(self.table.data_ptr()),
            Int32(len(self.ranges)),
            Int64(trace.data_ptr() if trace is not None else 0),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


__all__ = [
    "CHUNK_BYTES",
    "DEFAULT_CTAS",
    "K3L2PrefetchTableKernel",
    "L2PrefetchTable",
    "capped_ranges",
    "table_chunks",
]
