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
"""Copy-engine token-sharded TP using the shared :class:`CeTransport`.

The caller owns payload layouts, FP8 GEMM scheduling and deterministic reduction.
Each boundary declares its incoming ready flags, submits peer copies through
``CeTransport.push``, and waits before reading remote regions. Boundary exit joins the
copy streams and fences every rank's final read before scratch can be reused. Slots
select payload storage; one boundary is active at a time on an ordered compute stream.

Reserve the largest workspace eagerly before CUDA graph capture. Captured allocations
cannot grow until all graphs using them are retired. Interrupted eager boundaries use
collective transport recovery on the next use; mixed completion or unmatched device
waits require restarting the process group at its deadline.
"""

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

from .ce_transport import CeTransport
from .symm_mem_pool import DEFAULT_ALIGN, DEFAULT_TIMEOUT_MS
from .token_sharded_ce_gather import (
    fp8_block_gemm_out,
    mn_major_scales,
    validate_fp8_block_consumer,
)
from .token_sharded_ce_workspace import CeState, CeUnavailableError, CeWorkspace, _is_capturing
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


class CeReduceScatter(CeWorkspace):
    """Per-destination copy-engine reduce-scatter over a ``SymmMemPool`` (module docstring).

    Payload views and graph-safe capacity belong to :class:`CeWorkspace`; stream and
    signal ordering belong to the shared :class:`CeTransport`. The final pool channel
    fences completion. This class chooses the payload layout and GEMM schedule.

    Args:
        ce: The TP group's shared transport and reduce-scatter pool (its ``signal`` / ``wait`` carry the timeout).
        tp_size: Ranks in the TP group.
        rank: This process's rank in the group.
        device: The CUDA device the copy streams run on.
        slots: Slots to alternate over.
    """

    def __init__(
        self,
        ce: CeTransport,
        tp_size: int,
        rank: int,
        device: torch.device,
        *,
        slots: int = SLOTS,
    ) -> None:
        self.dest_order: tuple[int, ...] = tuple((rank - i) % tp_size for i in range(1, tp_size))
        """Destinations in push order ``me - 1, me - 2, ...``: at step ``i`` every rank targets
        a distinct destination, whose ``wait_order`` (``me + 1, me + 2, ...``) is the arrival
        order."""
        self.wait_order: tuple[int, ...] = tuple((rank + i) % tp_size for i in range(1, tp_size))
        """Remote sources in the order their ready flags are waited, ``me + 1, me + 2, ...``,
        which is the order their pushes reach this rank."""
        super().__init__(
            ce,
            tp_size,
            rank,
            device,
            slots=slots,
        )
        self._pushed: list[set[int]] = [set() for _ in range(slots)]
        self._layout: list[Bf16RowsRegionLayout | None] = [None] * slots

    # --- channels / capacity ----------------------------------------------------------------

    @property
    def tp_size(self) -> int:
        """``world_size``, in the TP vocabulary of the op body."""
        return self.world_size

    def _ready_channel(self, slot: int) -> int:
        return slot

    @property
    def min_channels(self) -> int:
        """Required channels, including the transport completion channel."""
        return self.slots + 1

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
        (re)allocated, dropping cached views into the previous allocation.
        """
        realloc = self._reserve(layout.slot_nbytes(self.world_size), self.slots, self.min_channels)
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
                "complete the active boundary before reusing its slot."
            )
        if not self.covers(layout):
            raise RuntimeError(
                f"CeReduceScatter.begin: the pool holds {self.slot_nbytes} bytes per slot, a "
                f"[{layout.rows}, {layout.cols}] boundary needs "
                f"{layout.slot_nbytes(self.world_size)}; reserve() eagerly for the largest boundary "
                "first."
            )
        self._bind_layout(slot, layout)
        ready = self._ready_channel(slot)
        self.open(slot, [(src, ready) for src in self.wait_order])
        self._pushed[slot].clear()

    def push(self, dst: int, slot: int, block: torch.Tensor) -> None:
        """Push ``block`` (this rank's K-partial of destination ``dst``'s rows, a contiguous
        bf16 ``[m, N]``) into ``dst``'s slot ``slot`` and raise its ready flag.

        Forks ``dst``'s copy stream from the current stream behind the GEMM that produced
        ``block``; enqueues only. Each destination exactly once per boundary, in any order.
        """
        self._require_in_flight(slot, "push")
        self._check_peer(dst, "push")
        ready = self._ready_channel(slot)
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
        view = self._peer_view(slot, dst)
        payload = block.view(torch.uint8).view(-1)

        def copy_to(peer: int) -> None:
            view.copy_(payload)
            block.record_stream(torch.cuda.current_stream())

        self.ce.push(ready, copy_to, destinations=(dst,))
        self._pushed[slot].add(dst)

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
        self.ce.wait_many((src, ready) for src in self.wait_order)

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
        missing = set(self.dest_order) - self._pushed[slot]
        if missing:
            raise RuntimeError(
                f"CeReduceScatter.done(slot={slot}) before pushing to destination(s) "
                f"{sorted(missing)}: they would wait for the ready flag forever."
            )
        super().done(slot)


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
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        group_name: str,
        *,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        super().__init__(group, group_name, timeout_ms=timeout_ms)
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
        if not self.probe_once(min_channels=SLOTS + 1, what=what):
            return
        self.begin_forward()
        assert self.transport is not None
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
        assert self.ce is not None
        self.transport = CeReduceScatter(
            self.ce,
            self.world_size,
            self.rank,
            self._device,
            slots=SLOTS,
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
    and the transport completion fence. Repeat over at most CHUNK_ROWS rows per destination, reusing
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
    except CeUnavailableError as exc:
        state.disable(str(exc))
        raise
    except Exception:
        transport.reset()
        raise
    return out
