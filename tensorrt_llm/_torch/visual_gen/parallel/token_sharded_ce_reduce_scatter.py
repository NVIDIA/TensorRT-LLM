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
"""Copy-engine reduce-scatter for token-sharded TP's FP8 block-scale row boundaries.

At a token-sharded row boundary (``to_out``, ``down_proj``) every TP rank holds the row GEMM's
input for all ``tp * m`` padded rows and needs the sum over ranks of its own ``m`` rows. The
NCCL path runs the GEMM on all rows, then ``reduce_scatter`` (SM kernels that contend with the
GEMM; a bf16 ring at ``tp >= 3``). Here the GEMM runs per destination block -- block ``d`` of
the padded stream is exactly destination ``d``'s rows (``TokenShardPlan.row_start = d * m``) --
and each remote block is *pushed* straight into the destination's region of a symmetric-memory
slot with one copy-engine ``cudaMemcpyAsync`` on a side stream (no SMs) as soon as its GEMM is
enqueued; the own block is computed last, directly into the own region, while the pushes are
in flight. After the ``tp - 1`` ready flags one kernel reduces the chunk's landed partials in
fixed source-rank order (fp32 adds, one bf16 rounding). The whole boundary is one custom
op (``trtllm::token_sharded_fp8_ce_gemm_reduce_scatter``) so a compiled block stays one graph.

Destination blocks are processed in rounds of at most :data:`CHUNK_ROWS` rows. A round
finishes its reduce and joins every push before the next round reuses the slot and staging
buffer; a tail round uses its actual row count. At fixed TP and output width, both buffers
are bounded independently of token count. The returned output still has ``m`` rows.

Protocol (rank ``r``, one boundary on slot ``s``)
--------------------------------------------------
Producer side, :meth:`CeReduceScatter.push`, per destination ``d`` in
:attr:`CeReduceScatter.dest_order` (``r - 1, r - 2, ...``: at every step the ranks target
distinct destinations), on ``d``'s side stream after an event recorded behind block ``d``'s
GEMM on the current stream::

    wait(d, consumed(s))        # skipped on the slot's first use: d finished reducing slot s
    d's slot s, region r: copy_(block d)      # one CE copy of the contiguous bf16 [m, N] block
    signal(d, ready(s))         # same stream: ordered after the copy

Consumer side, on the current stream: the own block's GEMM writes region ``r`` of MY slot
(:meth:`CeReduceScatter.own_view`); :meth:`CeReduceScatter.wait_all` waits ``ready(s)`` from
every source in :attr:`CeReduceScatter.wait_order` (``r + 1, r + 2, ...``, the arrival order);
:func:`fixed_order_reduce_bf16` reduces :meth:`CeReduceScatter.partials`;
:meth:`CeReduceScatter.done` joins every side stream into the current stream and signals
``consumed(s)`` to every peer. Channels: ``ready(s) = s``, ``consumed(s) = slots + s``. One slot
by default: the producer waits for ``consumed`` on its side stream, and the destination signals
``consumed`` right after its reduce. The next chunk's GEMM can run while a remote destination
finishes reducing the previous chunk; its push waits before overwriting that destination's slot.

Every wait is satisfiable: ``ready(s)`` of boundary ``b`` is put by every producer inside its
``push(b)``, enqueued on side streams that only wait for ``consumed(s)`` of ``b - slots``, which
the destination put in ``done(b - slots)`` after a reduce that depended on ``ready(s)`` of
``b - slots``. One put per wait per ``(channel, src -> dst)``. A rank skipping ``done`` or
ranks pushing different slot sequences end in the signal timeout (a device-side trap), never
in silent corruption.

Slot state machine and the balance invariant
--------------------------------------------
Per slot ``s`` every rank keeps ``uses[s]``, the number of boundaries COMPLETED on it -- by
:meth:`CeReduceScatter.done` or by :meth:`CeReduceScatter.reset` alike; rank-symmetric because
every rank runs the same boundary sequence. The flags are one bit per ``(channel, p -> d)``
(``put_signal`` blocks on a set flag, ``wait_signal`` on a clear one), and a slot is in one of
three states:

* FRESH (``uses[s] = 0``, nothing in flight): no flag of ``s`` set. Entered by a (re)allocating
  :meth:`CeReduceScatter.reserve`, which zeroes the pads.
* IN FLIGHT (``begin`` ran; rank ``r`` has pushed to ``dest_order[:k]``): for each ``d`` in
  ``dest_order[:k]`` the push waited ``consumed(s)`` ``d -> r`` (unless FRESH) and put
  ``ready(s)`` ``r -> d``; under rank symmetry ``r`` has received ``ready(s)`` from exactly
  ``wait_order[:k]`` (source ``p`` reaches ``r`` at its step ``p - r - 1 mod tp``).
* IDLE (``uses[s] > 0``, nothing in flight): every ``ready(s)`` flag clear, every
  ``consumed(s)`` flag SET (put by the last completion, to be waited by the destination's next
  push on ``s``).

``done`` and ``reset`` both take a slot from IN FLIGHT to IDLE through the same four steps
(``CeReduceScatter._release`` around the base's release): (i) wait ``consumed(s)`` from every
destination NOT pushed to in this boundary, unless FRESH -- the wait the un-issued push would
have done (a no-op from ``done``, which requires every push); (ii) wait ``ready(s)`` from every
source that pushed (``wait_order[:k]``) and that ``wait_all`` did not wait; (iii) join the
forked side streams; (iv) put ``consumed(s)`` to every destination. Invariant (asserted by the
CPU test on a fake pool):
after every completed boundary, for every slot ``s`` and ordered pair ``p -> d``: the
``ready(s)`` flag is clear and ``puts(ready) = waits(ready)`` = the number of boundaries on
``s`` in which ``p`` issued its push to ``d`` (``uses[s]`` unless a boundary was cut short
before that push); the ``consumed(s)`` flag is set iff ``uses[s] > 0``, with
``puts(consumed) = uses[s]`` and ``waits(consumed) = uses[s] - 1`` (0 when FRESH) -- so the
next boundary's producer side (one ``consumed`` wait per push) and consumer side (one
``ready`` wait per source) each find exactly one unconsumed flag per wait, whatever completed
the previous boundary. A recovery never re-enters FRESH: ``uses[s]`` only grows, so a captured
graph (whose pushes wait ``consumed``) stays replayable after one.

The boundary bookkeeping is :class:`~tensorrt_llm._torch.distributed.ce_protocol.CeTransport`'s
(``open`` / ``mark_pushed`` / ``wait_ready`` / ``done``, the copy streams, the view cache, the
reserve rules, ``reset`` and the resync after it); this module holds the layout, the channels,
the schedule and the ``consumed`` credit, which is the reduce-scatter's own as it lives across
boundaries (steps (i) and (iv) above, around the base's release).

Streams and CUDA graphs
-----------------------
By default one side stream per destination (``tp - 1`` streams; measured 750 vs 731 GB/s and
the better pipelined boundary against a single side stream). With the gather's side stream and
the compute stream this fits the default 8 CUDA connections up to ``tp = 7``; at ``tp = 8`` two
streams share a hardware queue (performance only, no correctness impact).
``per_peer_streams=False`` (``CeReduceScatterState(per_peer_streams=False)``) selects one side
stream for every destination; it is the only knob, there is no environment variable.
Each push forks from the current stream through an event recorded behind its block's GEMM and
every side stream is joined back in ``done`` of the same boundary, so after ``done`` nothing is
pending on any side stream: no event recorded in one forward is waited on in a later one (a
CUDA-graph capture boundary is safe), ``drain`` is a plain check and no ``Tensor.record_stream``
is needed (the staging tensor and the op's inputs are alive for the op call and the join
orders every later reuse of their memory on the current stream behind the copies). Per-boundary
``torch.cuda.Event`` objects become graph nodes under capture. The pool is sized eagerly
(:meth:`CeReduceScatterState.prepare`, from ``TokenShardedTP.begin`` in the pipeline's eager
warm-up forwards; the op grows it on demand in eager mode as a backstop); reserving under
capture raises. Two rules are enforced rather than assumed: a graph bakes in the pool's
addresses and flags, so once any boundary was captured the pool may not grow
(:meth:`CeReduceScatter.reserve` raises), and a slot's first use (which skips the ``consumed``
wait) may not be captured (:meth:`CeReduceScatter.begin` raises; run at least ``slots`` eager
boundaries after reserving, as the pipeline's warm-up forwards do). The Triton reduce kernel is
JIT-compiled per ``(tp, numel)``: warm every served shape eagerly before capturing.

Numerics
--------
The per-block GEMM is bitwise the whole-M GEMM (DeepGEMM computes each output row from its own
activation row in one block-ordered K loop; the scales of a block are re-strided with one small
copy, :func:`token_sharded_ce_gather.mn_major_scales`). That holds where the row Linear's own
GEMM is DeepGEMM, which is where the adapters take the fused op: the rows main's FP8 block-scale
low-M GEMM serves (``1 <= M <= fp8_low_m_gemm.max_m``, decode-sized; a kernel that is not bitwise
with DeepGEMM) stay on the module's GEMM and the NCCL reduce-scatter
(``TokenShardedTP.uses_ce_reduce_scatter``). The reduce computes, for every element,
``out = bf16(((p_0' + p_1) + p_2) + ... + p_{tp-1})`` with every partial upcast to fp32, the
adds in source-rank order ``0 .. tp - 1`` whatever arrived first, and ``p_0' = bf16(p_0 + bias)``
when the row Linear's bias is given (passed on every rank -- each destination adds it once to source 0's
partial exactly where the NCCL path's rank-0 Linear adds it), else ``p_0``. No atomics, no
accumulate-as-they-land: :func:`reference_fixed_order_reduce` (the plain torch chain) reproduces
it bitwise, which is what the tests assert. It is the precision class of NCCL's NVLS
reduce-scatter (fp32 accumulation, one rounding) and strictly tighter than the bf16 ring
(``tp - 1`` roundings): bitwise with NCCL at ``tp = 2`` (one add), not at ``tp >= 3``.

Failure triage
--------------
A rank that raises between ``push`` and ``done`` leaves its peers spinning in ``wait_signal``
until the timeout traps their CUDA context: a "device-side trap" on three of four ranks means
another rank raised; read that rank's traceback. Shape errors in the op body are checked
against the plan ints and the weight shape only, which are identical on every rank, so they
raise on every rank together; a failed op call resets the transport so the next ``begin`` does
not raise, and the first eager boundary after it agrees the ranks' boundary counts once
(:class:`~tensorrt_llm._torch.distributed.ce_protocol.CeResyncError` when they differ, which the
op body turns into ``CeReduceScatterState.disable``: the group continues on NCCL).
:meth:`CeReduceScatter.reset` completes the failed boundary's protocol exactly as
``done`` would have (the state machine above): a rank-symmetric raise after ``k`` of the
``tp - 1`` GEMM + push steps (a DeepGEMM failure, say) means every rank pushed to
``dest_order[:k]`` and received exactly the pushes of ``wait_order[:k]``; the reset waits those
ready flags only (waiting for the others would spin to the deadline), waits the ``consumed``
flags of the ``tp - 1 - k`` destinations it never pushed to (still set from the previous
boundary: a ``consumed`` put onto such a destination without that wait is a second put on a
set flag, which blocks until the deadline), then puts ``consumed`` to every destination, so the
repaired protocol continues without a device trap.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

from ...distributed.ce_protocol import CeResyncError, CeState, CeTransport, _is_capturing
from ...distributed.symm_mem_pool import DEFAULT_ALIGN, DEFAULT_TIMEOUT_MS, SymmMemPool
from .token_sharded_ce_gather import (
    fp8_block_gemm_out,
    mn_major_scales,
    validate_fp8_block_consumer,
)
from .token_sharded_tp import FP8_BLOCK_SIZE, fp8_scale_cols

if TYPE_CHECKING:
    from triton import JITFunction

    from .token_sharded_tp import TokenShardPlan

__all__ = [
    "CHUNK_ROWS",
    "DEFAULT_ALIGN",
    "DEFAULT_TIMEOUT_MS",
    "REDUCE_BLOCK",
    "REDUCE_NUM_WARPS",
    "SLOTS",
    "Bf16RowsRegionLayout",
    "CeReduceScatter",
    "CeReduceScatterState",
    "ce_gemm_reduce_scatter_impl",
    "fixed_order_reduce_bf16",
    "get_rs_state",
    "reference_fixed_order_reduce",
    "register_rs_state",
    "release_rs_state",
    "validate_fp8_block_row_consumer",
]

SLOTS = 1
"""int: Slots the reduce-scatter transport alternates over."""

CHUNK_ROWS = 8192
"""Maximum rows per destination in a transport round. At TP4, N=5120 the slot is
320 MiB and staging is 240 MiB, independent of the token count. Output and quantized
input storage still scale with tokens. Each round joins its pushes before staging is reused.
"""

REDUCE_BLOCK = 2048
"""int: Elements per program of the fixed-order reduce kernel (fixed launch config, no
autotune: 2048 x 4 warps measured 6.5 TB/s at [4, 37800, 5120], ahead of 4096 x 8 warps)."""

REDUCE_NUM_WARPS = 4
"""int: Warps per program of the fixed-order reduce kernel."""


# =============================================================================
# Region layout
# =============================================================================


@dataclass(frozen=True)
class Bf16RowsRegionLayout:
    """One source's region of a reduce-scatter slot: a bf16 ``[m, N]`` partial, payload only.

    Every rank holds the same ``m`` (``TokenShardPlan.local_rows``), so one layout describes
    every region of a slot; a slot is ``tp_size`` regions in source-rank order. The regions sit
    :attr:`region_nbytes` apart when the slot is carved tightly for this layout; inside a pool
    they sit the POOL's region stride apart, which is larger once the pool grew for a bigger
    layout (grow-only), so :meth:`partials_view` takes that stride. Python ints only (capture
    safe). Same slot-carving surface (``region_nbytes``, ``slot_nbytes``) as the gather's
    ``RegionLayout``.

    Attributes:
        rows: ``m``, rows per destination (and per source region).
        cols: ``N``, the row Linear's ``out_features``.
        align: Byte alignment of a region (a power of two >= 16; the pool's ``DEFAULT_ALIGN``,
            so a slot of ``tp_size`` regions is carved exactly).
    """

    rows: int
    cols: int
    align: int = DEFAULT_ALIGN

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError(
                f"Bf16RowsRegionLayout needs rows >= 1 and cols >= 1 (got rows={self.rows}, "
                f"cols={self.cols})."
            )
        if self.align < 16 or self.align & (self.align - 1):
            raise ValueError(
                f"Bf16RowsRegionLayout align must be a power of two >= 16 (got {self.align})."
            )

    @property
    def payload_nbytes(self) -> int:
        """Bytes of the bf16 ``[m, N]`` partial."""
        return 2 * self.rows * self.cols

    @property
    def region_nbytes(self) -> int:
        return pad_up(self.payload_nbytes, self.align)

    @property
    def src_stride(self) -> int:
        """bf16 elements between consecutive sources' regions in a slot carved tightly for
        this layout (``region_nbytes / 2``): the default stride of :meth:`partials_view`. A
        pool that grew past this layout places the regions further apart (its
        ``region_nbytes``); the transport passes that stride explicitly."""
        return self.region_nbytes // 2

    def slot_nbytes(self, tp_size: int) -> int:
        """Bytes one slot needs: ``tp_size`` regions."""
        return tp_size * self.region_nbytes

    def consumer_view(self, region: torch.Tensor) -> torch.Tensor:
        """The bf16 ``[m, N]`` partial over a region's uint8 bytes (what a GEMM writes into
        the own region and what the reduce reads per source)."""
        self._check_region(region)
        return region[: self.payload_nbytes].view(torch.bfloat16).view(self.rows, self.cols)

    def producer_view(self, region: torch.Tensor) -> torch.Tensor:
        """The uint8 ``[2 * m * N]`` payload span of a region: what one push copies into."""
        self._check_region(region)
        return region[: self.payload_nbytes]

    def partials_view(
        self, slot: torch.Tensor, tp_size: int, region_stride_nbytes: int | None = None
    ) -> torch.Tensor:
        """The bf16 ``[tp_size, m, N]`` view over a slot whose regions sit
        ``region_stride_nbytes`` apart (strides ``(region_stride_nbytes / 2, N, 1)``).

        ``slot`` is a 1-D uint8 view of the slot's first region or of the whole slot; the view
        reaches ``tp_size`` regions from its start, which must lie inside its storage.
        ``region_stride_nbytes`` is the pool's region stride (``SymmMemPool.region_nbytes``,
        >= :attr:`region_nbytes` once the pool grew for a larger layout); None means a slot
        carved tightly for this layout (:attr:`src_stride`).
        """
        self._check_region(slot)
        stride = self.region_nbytes if region_stride_nbytes is None else region_stride_nbytes
        if stride < self.region_nbytes or stride % self.align:
            raise ValueError(
                f"Bf16RowsRegionLayout.partials_view: region_stride_nbytes must be a multiple of "
                f"{self.align} >= {self.region_nbytes}; got {stride}."
            )
        need = slot.storage_offset() + (tp_size - 1) * stride + self.payload_nbytes
        if slot.untyped_storage().nbytes() < need:
            raise ValueError(
                f"Bf16RowsRegionLayout.partials_view: the storage holds "
                f"{slot.untyped_storage().nbytes()} bytes, {tp_size} regions {stride} bytes "
                f"apart from byte {slot.storage_offset()} need {need}."
            )
        return slot.view(torch.bfloat16).as_strided(
            (tp_size, self.rows, self.cols), (stride // 2, self.cols, 1)
        )

    def _check_region(self, region: torch.Tensor) -> None:
        if region.dtype != torch.uint8 or region.dim() != 1 or region.numel() < self.region_nbytes:
            raise ValueError(
                f"Bf16RowsRegionLayout: need a 1-D uint8 region of >= {self.region_nbytes} "
                f"bytes; got {region.dtype} {tuple(region.shape)}."
            )
        if region.data_ptr() % 16:
            raise RuntimeError(
                f"Bf16RowsRegionLayout: region base {region.data_ptr():#x} is not 16-byte "
                "aligned (the GEMM's TMA store and the reduce's vector loads need it); the "
                "pool's region stride is misaligned."
            )


# =============================================================================
# The deterministic reduce
# =============================================================================


def reference_fixed_order_reduce(
    partials: Sequence[torch.Tensor] | torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """The reduce's contract as the plain torch chain: ``bf16((((p_0' + p_1) + p_2) + ...))``.

    Every partial is upcast to fp32 and added in source order; ``p_0' = bf16(p_0 + bias)`` when
    ``bias`` is given (source 0 carries the bias, as the NCCL path's rank-0 partial does; every
    destination passes its copy of the bias), else
    ``p_0``; one bf16 rounding at the end. IEEE fp32 adds and RNE casts, so
    :func:`fixed_order_reduce_bf16` reproduces it bitwise. CPU-testable; several launches and
    fp32 intermediates, so it is the reference and the fallback, not the hot path.

    Args:
        partials: ``tp`` bf16 ``[m, N]`` tensors in source-rank order (a list, or a
            ``[tp, m, N]`` tensor, which iterates over its first dimension).
        bias: ``[N]`` bf16 or None.
    """
    parts = list(partials)
    if not parts:
        raise ValueError("reference_fixed_order_reduce needs at least one partial.")
    acc = parts[0].float()
    if bias is not None:
        acc = (acc + bias.float()).to(torch.bfloat16).float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


@functools.lru_cache(maxsize=1)
def _reduce_kernel() -> "JITFunction":
    """The Triton kernel, built on first use (so this module imports without Triton)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _fixed_order_reduce_bf16_kernel(
        base_ptr,
        out_ptr,
        bias_ptr,
        src_stride,
        numel,
        n_cols,
        TP: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # out[i] = bf16(((p0'[i] + p1[i]) + p2[i]) + ...), every p upcast to fp32; p_r lives at
        # base_ptr + r * src_stride, flat over [m, N]; p0' = bf16(p0 + bias[i % N]) when HAS_BIAS.
        # Source order is fixed by the unrolled loop whatever arrived first; one RNE rounding at
        # the end. Triton vectorizes the loads (16 B) from the pointer and stride alignment it
        # specializes on: the region base is 256-byte aligned and src_stride a multiple of 128.
        # All index arithmetic in int64: Triton types a Python int argument i32 when it fits,
        # and r * src_stride would wrap for regions over 2^31 / (TP - 1) elements.
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        stride = src_stride.to(tl.int64)
        mask = offs < numel
        acc = tl.load(base_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(bias_ptr + offs % n_cols, mask=mask, other=0.0).to(tl.float32)
            acc = (acc + b).to(tl.bfloat16).to(tl.float32)
        for r in tl.static_range(1, TP):
            acc += tl.load(base_ptr + r * stride + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)

    return _fixed_order_reduce_bf16_kernel


def fixed_order_reduce_bf16(
    partials: torch.Tensor, bias: torch.Tensor | None, out: torch.Tensor
) -> torch.Tensor:
    """``out = bf16(fixed-order fp32 sum over the sources of partials, bias on source 0)``.

    One Triton launch with a fixed configuration (:data:`REDUCE_BLOCK` x :data:`REDUCE_NUM_WARPS`,
    JIT-compiled per ``(tp, numel)``), bitwise :func:`reference_fixed_order_reduce`. On CPU
    tensors (tests) the torch chain runs instead. The kernel indexes in int64, so the only size
    limits are the tensors' own (``numel`` and ``S`` below 2^63).

    Args:
        partials: bf16 ``[tp, m, N]`` with strides ``(S, N, 1)`` (the slot's
            :meth:`Bf16RowsRegionLayout.partials_view` with the pool's region stride; any
            source stride ``S``).
        bias: ``[N]`` bf16 or None.
        out: Contiguous bf16 ``[m, N]``.

    Returns:
        ``out``.
    """
    if partials.dim() != 3 or partials.dtype != torch.bfloat16:
        raise ValueError(
            f"fixed_order_reduce_bf16: partials must be bf16 [tp, m, N]; got {partials.dtype} "
            f"{tuple(partials.shape)}."
        )
    tp, m, n = partials.shape
    if partials.stride(2) != 1 or partials.stride(1) != n:
        raise ValueError(
            f"fixed_order_reduce_bf16: partials must have strides (S, N, 1); got "
            f"{tuple(partials.stride())} for N={n}."
        )
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n) or not out.is_contiguous():
        raise ValueError(
            f"fixed_order_reduce_bf16: out must be a contiguous bf16 [{m}, {n}]; got {out.dtype} "
            f"{tuple(out.shape)}."
        )
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
        raise ValueError(
            f"fixed_order_reduce_bf16: bias must be bf16 [{n}] or None; got {bias.dtype} "
            f"{tuple(bias.shape)}."
        )
    if partials.device.type != "cuda":
        out.copy_(reference_fixed_order_reduce(partials, bias))
        return out
    numel = m * n
    grid = ((numel + REDUCE_BLOCK - 1) // REDUCE_BLOCK,)
    _reduce_kernel()[grid](
        partials,
        out,
        out if bias is None else bias,
        partials.stride(0),
        numel,
        n,
        TP=tp,
        HAS_BIAS=bias is not None,
        BLOCK=REDUCE_BLOCK,
        num_warps=REDUCE_NUM_WARPS,
    )
    return out


# =============================================================================
# The consumer check
# =============================================================================


def validate_fp8_block_row_consumer(linear: object) -> None:
    """Check once that a row ``Linear``'s operands fit the fused GEMM + reduce-scatter.

    The same operand contract as the gather's ``validate_fp8_block_consumer`` (contiguous
    float8_e4m3fn ``[N, K_local]`` weight with ``K_local % 128 == 0`` and the packed int32
    MN-major ``weight_scale`` that ``FP8BlockScalesLinearMethod.transform_weights`` produces),
    plus the bias, when present, a bf16 ``[N]`` (added inside the reduce on source 0). Called
    once after the weights are loaded (``TokenShardedTP`` does so at the first effective
    ``begin()``); the op body re-checks only dtypes and shapes per call, rank-symmetrically.

    Raises:
        ValueError: ``linear`` is not an FP8 block-scale ``Linear`` or an operand does not fit.
    """
    try:
        validate_fp8_block_consumer(linear)
    except ValueError as e:
        raise ValueError(
            f"copy-engine reduce-scatter: the row consumer's operands do not fit the fused "
            f"GEMM: {e}"
        ) from None
    bias = getattr(linear, "bias", None)
    n = linear.weight.shape[0]
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
        raise ValueError(
            f"copy-engine reduce-scatter: the row consumer's bias must be bf16 [{n}]; got "
            f"{bias.dtype} {tuple(bias.shape)}."
        )


# =============================================================================
# Transport
# =============================================================================


class CeReduceScatter(CeTransport):
    """Per-destination copy-engine reduce-scatter over a ``SymmMemPool`` (module docstring).

    The boundary bookkeeping, the copy streams, the view cache, the reserve rules and the
    recovery path are :class:`CeTransport`'s; this class holds the reduce-scatter's layout,
    channels and schedule. Owns the pool for its lifetime (data channels ``0 .. 2 * slots - 1``).
    Per boundary: :meth:`begin`, a GEMM + :meth:`push` per destination in :attr:`dest_order`,
    the own GEMM into :meth:`own_view`, :meth:`wait_all`, the reduce over :meth:`partials`,
    :meth:`done`. :meth:`reserve` sizes the pool eagerly and collectively.

    Args:
        pool: The TP group's reduce-scatter pool (its ``signal`` / ``wait`` carry the timeout).
        tp_size: Ranks in the TP group.
        rank: This process's rank in the group.
        device: The CUDA device the copy streams run on.
        slots: Slots to alternate over.
        per_peer_streams: One copy stream per destination (default) or one for all.
    """

    def __init__(
        self,
        pool: SymmMemPool,
        tp_size: int,
        rank: int,
        device: torch.device,
        *,
        slots: int = SLOTS,
        per_peer_streams: bool = True,
    ) -> None:
        self.dest_order: tuple[int, ...] = tuple((rank - i) % tp_size for i in range(1, tp_size))
        """Destinations in push order ``me - 1, me - 2, ...``: at step ``i`` every rank targets
        a distinct destination, whose ``wait_order`` (``me + 1, me + 2, ...``) is the arrival
        order."""
        self.wait_order: tuple[int, ...] = tuple((rank + i) % tp_size for i in range(1, tp_size))
        """Remote sources in the order their ready flags are waited, ``me + 1, me + 2, ...``,
        which is the order their pushes reach this rank (the base's ``arrival_order``)."""
        super().__init__(
            pool,
            tp_size,
            rank,
            device,
            slots=slots,
            arrival_order=self.wait_order,
            per_peer_streams=per_peer_streams,
        )
        self._layout: list[Bf16RowsRegionLayout | None] = [None] * slots
        self._uses = [0] * slots

    # --- channels / capacity ----------------------------------------------------------------

    @property
    def tp_size(self) -> int:
        """``world_size``, in the TP vocabulary of the op body."""
        return self.world_size

    def _ready_channel(self, slot: int) -> int:
        return slot

    def _consumed_channel(self, slot: int) -> int:
        return self.slots + slot

    @property
    def min_channels(self) -> int:
        """Data channels the pool must offer this transport."""
        return 2 * self.slots

    @property
    def slot_nbytes(self) -> int:
        """Bytes per slot of the current reservation (0 before :meth:`reserve`)."""
        return self.pool.slot_nbytes

    def covers(self, layout: Bf16RowsRegionLayout) -> bool:
        """Whether the current reservation holds ``layout`` on every slot."""
        return self._covers(layout.slot_nbytes(self.world_size), self.slots)

    def reserve(self, layout: Bf16RowsRegionLayout) -> bool:
        """Size the pool for ``layout`` on :attr:`slots` slots (collective, grow-only).

        Call eagerly, every rank in the same order, never under capture (the pool raises) and
        never for a larger layout once a boundary was captured into a CUDA graph (raises: the
        graph holds the current allocation's addresses and flags). Returns True when the pool
        (re)allocated: the signal pad is then zeroed, so the protocol restarts from "first use"
        and the cached views are dropped.
        """
        realloc = self._reserve(layout.slot_nbytes(self.world_size), self.slots, self.min_channels)
        if realloc:
            self._uses = [0] * self.slots
        if (
            self.slot_nbytes // self.world_size < layout.region_nbytes
            or self.pool.slots < self.slots
        ):
            raise RuntimeError(
                f"CeReduceScatter.reserve: the pool holds {self.pool.slots} x {self.slot_nbytes} "
                f"bytes after reserve(); need {self.slots} x {layout.slot_nbytes(self.world_size)}."
            )
        return realloc

    def _bind_layout(self, slot: int, layout: Bf16RowsRegionLayout) -> None:
        # Region views are valid for the layout a slot was begun with: drop that slot's cache
        # when its layout changes; another slot (possibly in flight) keeps its own.
        if layout != self._layout[slot]:
            self._layout[slot] = layout
            self._drop_views(slot)

    def _peer_view(self, slot: int, peer: int) -> torch.Tensor:
        """MY region's payload span in ``peer``'s slot: what my push writes (cached)."""
        layout = self._layout[slot]
        return self.view(
            (slot, "peer", peer),
            lambda: layout.producer_view(
                self.pool.peer_region(peer, slot, 0, layout.region_nbytes)
            ),
        )

    # --- the boundary -----------------------------------------------------------------------

    def begin(self, slot: int, layout: Bf16RowsRegionLayout) -> None:
        """Open a boundary on slot ``slot`` for ``layout`` (the plan's ``m`` x the consumer's
        ``N``): bind the layout, open the slot with the ready flags it owes. Enqueues nothing;
        the host checks come first, so a refusal leaves the slot idle."""
        self._check_slot(slot)
        if self._in_flight[slot]:
            raise RuntimeError(
                f"CeReduceScatter.begin(slot={slot}) before done({slot}) of its previous use: the "
                "peers would wait for the consumed flag forever."
            )
        if not self.covers(layout):
            raise RuntimeError(
                f"CeReduceScatter.begin: the pool holds {self.slot_nbytes} bytes per slot, a "
                f"[{layout.rows}, {layout.cols}] boundary needs "
                f"{layout.slot_nbytes(self.world_size)}; reserve() eagerly for the largest boundary "
                "first."
            )
        if self._uses[slot] == 0 and _is_capturing():
            raise RuntimeError(
                f"CeReduceScatter.begin: first use of slot {slot} under CUDA-graph capture (its "
                f"consumed wait would be missing from every replay); run at least {self.slots} "
                "eager boundaries after reserving before capturing."
            )
        self._bind_layout(slot, layout)
        ready = self._ready_channel(slot)
        self.open(slot, [(src, ready) for src in self.wait_order])

    def push(self, dst: int, slot: int, block: torch.Tensor) -> None:
        """Push ``block`` (this rank's K-partial of destination ``dst``'s rows, a contiguous
        bf16 ``[m, N]``) into ``dst``'s slot ``slot`` and raise its ready flag.

        Forks ``dst``'s copy stream from the current stream behind the GEMM that produced
        ``block``; enqueues only. Each destination exactly once per boundary, in any order.
        """
        self._require_in_flight(slot, "push")
        self._check_peer(dst, "push")
        ready = self._ready_channel(slot)
        if (dst, ready) in self._pushed_to[slot]:
            raise RuntimeError(
                f"CeReduceScatter.push(dst={dst}, slot={slot}) twice in one boundary: the peer "
                "would see a second ready flag."
            )
        layout = self._layout[slot]
        if (
            block.dtype != torch.bfloat16
            or tuple(block.shape) != (layout.rows, layout.cols)
            or not block.is_contiguous()
        ):
            raise ValueError(
                f"CeReduceScatter.push: block must be a contiguous bf16 [{layout.rows}, "
                f"{layout.cols}]; got {block.dtype} {tuple(block.shape)}."
            )
        # The destination's view is the last host-side raise point (a cache miss carves it
        # from the pool): resolve it before enqueuing anything, so on every exit of this method
        # `(dst, ready) in _pushed_to` <=> 'this push's consumed wait is on dst's copy stream'
        # -- the equivalence _release step (i) relies on to wait that flag exactly once.
        view = self._peer_view(slot, dst)
        payload = block.view(torch.uint8).view(-1)
        fork = torch.cuda.Event()
        fork.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.streams.fork(dst, fork)):
            if self._uses[slot] > 0:
                self.pool.wait(dst, self._consumed_channel(slot))
            view.copy_(payload)
            self.mark_pushed(slot, dst, ready)

    def own_view(self, slot: int) -> torch.Tensor:
        """Region ``rank`` of MY slot as the bf16 ``[m, N]`` the own block's GEMM writes (only
        this rank ever writes it, so no peer races it)."""
        self._require_in_flight(slot, "own_view")
        layout = self._layout[slot]
        return self.view(
            (slot, "own"),
            lambda: layout.consumer_view(
                self.pool.region(slot, self.rank, 0, layout.region_nbytes)
            ),
        )

    def wait_all(self, slot: int) -> None:
        """Make the current stream wait for every remote source's ready flag (in
        :attr:`wait_order`); after it the slot holds all ``tp`` partials."""
        self._require_in_flight(slot, "wait_all")
        ready = self._ready_channel(slot)
        for src in self.wait_order:
            self.wait_ready(slot, src, ready)

    def partials(self, slot: int) -> torch.Tensor:
        """The slot's ``[tp, m, N]`` bf16 view in source-rank order (strides
        ``(pool.region_nbytes / 2, N, 1)``): the reduce's input. Valid after :meth:`wait_all`.

        The sources' regions sit the POOL's region stride apart (where the pushes and the own
        GEMM land: ``pool.peer_region`` / ``pool.region``), not the layout's packed
        ``src_stride``: the pool is grow-only, so after a larger reservation (a bigger token
        shape in the warm-up, a consumer with a larger ``N``) its regions are further apart
        than this boundary's layout alone would carve them.
        """
        self._require_in_flight(slot, "partials")
        layout = self._layout[slot]
        return self.view(
            (slot, "partials"),
            lambda: layout.partials_view(
                self.pool.region(slot, 0, 0, layout.region_nbytes),
                self.world_size,
                region_stride_nbytes=self.pool.region_nbytes,
            ),
        )

    def done(self, slot: int) -> None:
        """Complete the boundary on ``slot``: join the forked copy streams into the current
        stream, release the slot to the peers and note a captured boundary. Call after the
        reduce was issued on the current stream."""
        self._require_in_flight(slot, "done")
        ready = self._ready_channel(slot)
        missing = set(self.dest_order) - {dst for dst, ch in self._pushed_to[slot] if ch == ready}
        if missing:
            raise RuntimeError(
                f"CeReduceScatter.done(slot={slot}) before pushing to destination(s) "
                f"{sorted(missing)}: they would wait for the ready flag forever."
            )
        super().done(slot)

    def _release(self, slot: int) -> None:
        # IN FLIGHT -> IDLE with exactly one wait per put on every (channel, p -> d) pair, from
        # done() and reset() alike (module docstring, "Slot state machine"). The base waits the
        # owed ready flags (after a raise inside the push loop, reset() first trims them to the
        # pushes that reached this rank) and joins the forked copy streams; around it, the
        # consumed credits, which are the reduce-scatter's own as they live across boundaries:
        #  (i)  BEFORE the base: consumed(s) from every destination this boundary did NOT push
        #       to -- the wait its un-issued push would have done, skipped on the slot's first
        #       use as push skips it. A no-op from done(), which requires every push. Without
        #       it step (iv) would put a second consumed flag onto one the peer never cleared,
        #       and put_signal blocks on a set flag until the deadline (a device trap on every
        #       rank). Relies on push() enqueuing its consumed wait only after its last
        #       host-side raise point: `(dst, ready) in _pushed_to` <=> 'its consumed wait is
        #       enqueued'.
        #  (iv) AFTER the base: consumed(s) to every destination: its next push on this slot
        #       waits it.
        # Invariant afterwards, per pair: the ready flag clear with puts(ready) == waits(ready)
        # (this boundary's put, when its push was issued, waited by wait_all() or by the base's
        # release); the consumed flag set with puts(consumed) == uses[slot] and
        # waits(consumed) == uses[slot] - 1 (uses[slot] counting this completion).
        ready, consumed = self._ready_channel(slot), self._consumed_channel(slot)
        if self._uses[slot] > 0:
            for dst in self.dest_order:
                if (dst, ready) not in self._pushed_to[slot]:
                    self.pool.wait(dst, consumed)
        super()._release(slot)
        for dst in self.dest_order:
            self.pool.signal(dst, consumed)
        self._uses[slot] += 1


# =============================================================================
# Per-group state and registry
# =============================================================================


class CeReduceScatterState(CeState):
    """The copy-engine reduce-scatter state of one TP group: its own pool (separate from the
    gather's: different region size, no slot aliasing hazard), transport, slot counter, probe
    verdict.

    Created and registered by ``TokenShardedTP`` (two instances over one group, e.g. the two
    Wan2.2 experts, share it), released by ``TokenShardedTP.close()``; a closed state refuses
    :meth:`prepare`. :meth:`prepare` runs eagerly from ``begin()``; the first call probes the
    group's eligibility, rank-agreed, and the verdict decides :attr:`effective` for the state's
    lifetime (:class:`CeState`: probe once, warn once, one registry per protocol).

    Args:
        group: The TP process group.
        group_name: ``group.group_name`` (the registry key and the op's handle).
        timeout_ms: Device-side deadline of every signal wait (:data:`DEFAULT_TIMEOUT_MS`).
        per_peer_streams: One push copy stream per destination (default) or one for all.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        group_name: str,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        per_peer_streams: bool = True,
    ) -> None:
        super().__init__(group, group_name, timeout_ms=timeout_ms)
        self.per_peer_streams = per_peer_streams
        self.consumer_ks: set[int] = set()
        """set[int]: The noted row consumers' ``K_local`` (``in_features``)."""
        self.consumer_ns: set[int] = set()
        """set[int]: The noted row consumers' ``N`` (``out_features``), which size the pool."""
        self._counter = 0

    @property
    def tp_size(self) -> int:
        """``world_size``, in the TP vocabulary of the adapters."""
        return self.world_size

    # --- conversion time ------------------------------------------------------------------

    def note_row_consumer(self, k_local: int, out_features: int | None = None) -> None:
        """Record a row consumer: ``k_local`` (``in_features``, a positive multiple of 128, the
        fused GEMM's K) and, when given, ``out_features`` (``N``), which lets :meth:`prepare`
        size the pool eagerly; without it the op sizes the pool at its first eager call."""
        if k_local < 1 or k_local % FP8_BLOCK_SIZE:
            raise ValueError(
                f"note_row_consumer: K_local must be a positive multiple of {FP8_BLOCK_SIZE}; got "
                f"{k_local}."
            )
        if out_features is not None and out_features < 1:
            raise ValueError(f"note_row_consumer: N must be >= 1; got {out_features}.")
        self.consumer_ks.add(int(k_local))
        if out_features is not None:
            self.consumer_ns.add(int(out_features))

    # --- begin() ----------------------------------------------------------------------------

    def prepare(self, plan: "TokenShardPlan") -> None:
        """Probe once, start the forward's slot sequence and size the pool for ``plan``.

        Eager, every rank in the same order (``TokenShardedTP.begin``). Checks that the
        previous boundary chain is complete and restarts the slots at 0 (:meth:`begin_forward`,
        so the caller need not call it), then sizes the pool for
        ``min(plan.local_rows, CHUNK_ROWS)`` x the
        largest noted consumer ``N``, grow-only. Under CUDA-graph capture only the slot counter
        is reset; a probe or an allocation under capture raises, as does a closed state.
        """
        what = f"Token-sharded TP ({self.group_name}): copy-engine reduce-scatter"
        if not self.probe_once(min_channels=2 * SLOTS, what=what):
            return
        self.begin_forward()
        if not self.consumer_ns:
            return
        layout = Bf16RowsRegionLayout(min(plan.local_rows, CHUNK_ROWS), max(self.consumer_ns))
        if self.transport.covers(layout):
            return
        if _is_capturing():
            raise RuntimeError(
                f"copy-engine reduce-scatter: TokenShardedTP.begin() under CUDA-graph capture "
                f"needs a pool for {layout.slot_nbytes(self.world_size)} bytes per slot, the pool "
                f"holds {self.transport.slot_nbytes}; run an eager forward of this shape first."
            )
        self.transport.reserve(layout)  # logs the size, the deadline and the triage hint
        logger.info_once(
            f"Token-sharded TP ({self.group_name}): copy-engine reduce-scatter on.",
            key=("token_sharded_ce_reduce_scatter", self.group_name),
        )

    def begin_forward(self) -> None:
        """Start a forward: the previous boundary chain must be complete; slots restart at 0."""
        if self.transport is not None:
            self.transport.drain()
        self._counter = 0

    def next_slot(self) -> int:
        """The slot of the next boundary (alternating over :data:`SLOTS`; reset by
        :meth:`begin_forward`)."""
        slot = self._counter % SLOTS
        self._counter += 1
        return slot

    # --- probe --------------------------------------------------------------------------------

    def _make_transport(self) -> None:
        self.pool = SymmMemPool(self.group, self._device, self.timeout_ms, align=DEFAULT_ALIGN)
        self.transport = CeReduceScatter(
            self.pool,
            self.world_size,
            self.rank,
            self._device,
            slots=SLOTS,
            per_peer_streams=self.per_peer_streams,
        )


# The registry of this protocol, as TokenShardedTP and the op body call it.
get_rs_state = CeReduceScatterState.get
register_rs_state = CeReduceScatterState.register
release_rs_state = CeReduceScatterState.release

# =============================================================================
# The op body
# =============================================================================


def _check_op_operands(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    w_fp8: torch.Tensor,
    w_sf: torch.Tensor,
    bias: torch.Tensor | None,
    padded_rows: int,
) -> None:
    # Against the plan ints and the weight shape only: identical on every rank, so a raise here
    # is rank-symmetric and leaves no peer waiting.
    if w_fp8.dim() != 2 or w_fp8.dtype != torch.float8_e4m3fn or w_sf.dtype != torch.int32:
        raise ValueError(
            "token_sharded_fp8_ce_gemm_reduce_scatter: the weight must be a float8_e4m3fn "
            f"[N, K_local] with the packed int32 weight_scale (post_load_weights); got "
            f"{w_fp8.dtype} {tuple(w_fp8.shape)} / {w_sf.dtype}."
        )
    n, k = w_fp8.shape
    if x_fp8.dtype != torch.float8_e4m3fn or tuple(x_fp8.shape) != (padded_rows, k):
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: expected the padded float8_e4m3fn "
            f"[{padded_rows}, {k}] row-path input; got {x_fp8.dtype} {tuple(x_fp8.shape)}."
        )
    if x_sf.dtype != torch.int32 or tuple(x_sf.shape) != (padded_rows, fp8_scale_cols(k)):
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: expected int32 packed scales "
            f"[{padded_rows}, {fp8_scale_cols(k)}]; got {x_sf.dtype} {tuple(x_sf.shape)}."
        )
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,)):
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: bias must be bf16 [{n}] or None; got "
            f"{bias.dtype} {tuple(bias.shape)}."
        )


def ce_gemm_reduce_scatter_impl(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    w_fp8: torch.Tensor,
    w_sf: torch.Tensor,
    bias: torch.Tensor | None,
    group_name: str,
    tp_rank: int,
    tp_size: int,
    batch_size: int,
    seq_len: int,
    padded_seq_len: int,
) -> torch.Tensor:
    """Body of ``trtllm::token_sharded_fp8_ce_gemm_reduce_scatter``: per-destination GEMM +
    push, own GEMM, wait, fixed-order reduce, done.

    Returns this rank's reduced ``[m, N]`` bf16 rows of the padded stream (``m = batch_size *
    padded_seq_len / tp_size``; the caller takes its ``local_view``). Order on the current
    stream: for each destination in ``transport.dest_order`` the GEMM of its row block into a
    transient staging tensor, then its push (side stream); the own block's GEMM straight into
    the own region; the ``tp - 1`` ready waits; the reduce with ``bias`` on source 0; the join
    and the consumed signals. Repeat over at most CHUNK_ROWS rows per destination, reusing
    staging only after done() joins every push; write reduced chunks into the full output.

    Args:
        x_fp8: The padded ``[batch_size * padded_seq_len, K_local]`` float8_e4m3fn row-path
            input of all tokens (``quantize_fp8_block`` of ``pad_row_input(act)``).
        x_sf: Its ``(batch_size * padded_seq_len, P)`` int32 packed UE8M0 scales, any stride.
        w_fp8: The row Linear's ``[N, K_local]`` float8_e4m3fn weight.
        w_sf: Its packed int32 weight scale.
        bias: The replicated row Linear's ``[N]`` bf16 bias on every rank, or None when the
            Linear has none. Each destination uses bf16(p0 + bias) as source 0's partial.
        group_name: The TP group's c10d name (the registry key).
        tp_rank: This rank in the group.
        tp_size: Ranks in the group.
        batch_size: ``B``.
        seq_len: ``S`` (not used by the body: the output keeps the pad rows, as the NCCL
            reduce-scatter's does; part of the plan ints the graph specializes on). A pad row of
            the output holds the bias (it enters source 0's block for every row, as rank 0's
            Linear adds it on the NCCL row path); pad rows are dropped at the next gather by
            contract.
        padded_seq_len: ``S_pad``.
    """
    del seq_len
    try:
        state = get_rs_state(group_name)
    except KeyError:
        raise RuntimeError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: no CeReduceScatterState is registered "
            f"for TP group {group_name!r}; TokenShardedTP(reduce_scatter_mode='copy_engine') "
            "registers one at its first begin()."
        ) from None
    transport = state.transport
    if transport is None:
        raise RuntimeError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: the copy-engine reduce-scatter is not "
            f"effective for TP group {group_name!r} ({state.reason}); the adapters must take the "
            "NCCL path."
        )
    if tp_size != transport.tp_size or tp_rank != transport.rank:
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: plan rank {tp_rank}/{tp_size} does not "
            f"match the group's {transport.rank}/{transport.tp_size}."
        )
    padded_rows = batch_size * padded_seq_len
    if padded_rows % tp_size:
        raise ValueError(
            f"token_sharded_fp8_ce_gemm_reduce_scatter: B * S_pad = {padded_rows} is not a "
            f"multiple of tp_size={tp_size}."
        )
    m = padded_rows // tp_size
    _check_op_operands(x_fp8, x_sf, w_fp8, w_sf, bias, padded_rows)
    n = w_fp8.shape[0]
    chunk_rows = min(m, CHUNK_ROWS)
    layout = Bf16RowsRegionLayout(chunk_rows, n)
    if not transport.covers(layout):
        # Eager backstop (rank-symmetric: every rank calls the op with the same plan ints and
        # N): a consumer whose N was not noted before begin(). Under capture the pool raises.
        logger.info(
            f"Token-sharded TP ({group_name}): sizing the reduce-scatter pool for "
            f"[{chunk_rows}, {n}] at "
            "the op's first eager call; note the consumer's out_features before begin() to "
            "reserve eagerly."
        )
        transport.reserve(layout)
    x_fp8 = x_fp8.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x_fp8.device)
    staging = torch.empty(
        ((tp_size - 1) * chunk_rows, n), dtype=torch.bfloat16, device=x_fp8.device
    )
    try:
        for start in range(0, m, chunk_rows):
            count = min(chunk_rows, m - start)
            slot = state.next_slot()
            transport.begin(slot, Bf16RowsRegionLayout(count, n))
            for i, dst in enumerate(transport.dest_order):
                rows = slice(dst * m + start, dst * m + start + count)
                block = staging[i * chunk_rows : i * chunk_rows + count]
                fp8_block_gemm_out(x_fp8[rows], mn_major_scales(x_sf[rows]), w_fp8, w_sf, block)
                transport.push(dst, slot, block)
            rows = slice(tp_rank * m + start, tp_rank * m + start + count)
            fp8_block_gemm_out(
                x_fp8[rows], mn_major_scales(x_sf[rows]), w_fp8, w_sf, transport.own_view(slot)
            )
            transport.wait_all(slot)
            fixed_order_reduce_bf16(transport.partials(slot), bias, out[start : start + count])
            transport.done(slot)
    except CeResyncError as exc:
        # The resync after a failed boundary found the ranks disagreeing on their boundary count
        # (raised by ``begin`` before anything was enqueued): take the group off the copy-engine
        # reduce-scatter for the state's lifetime (one warning) and propagate; the adapters' next
        # call reads ``effective`` and reduce-scatters with NCCL.
        state.disable(str(exc))
        raise
    except Exception:
        transport.reset()
        raise
    return out
