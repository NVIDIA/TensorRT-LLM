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

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tensorrt_llm import deep_gemm
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

from ...modules.linear import FP8BlockScalesLinearMethod, Linear
from .ce_transport import CeTransport
from .symm_mem_pool import DEFAULT_ALIGN, DEFAULT_TIMEOUT_MS
from .token_sharded_ce_workspace import CeState, CeUnavailableError, CeWorkspace, _is_capturing
from .token_sharded_tp import FP8_BLOCK_SIZE, drop_padding, fp8_scale_cols

if TYPE_CHECKING:
    from .token_sharded_tp import TokenShardPlan

__all__ = [
    "DEFAULT_ALIGN",
    "DEFAULT_TIMEOUT_MS",
    "SLOTS",
    "CeAllGather",
    "CeGatherState",
    "RegionLayout",
    "ce_gather_gemm_impl",
    "ce_gather_gemms_impl",
    "drop_padding",
    "fp8_block_gemm_out",
    "get_state",
    "mn_major_scales",
    "packed_weight_scale_cols",
    "register_state",
    "release_state",
    "scale_span_elems",
    "validate_fp8_block_consumer",
]

SLOTS = 2
"""int: Payload slots alternated between completed boundaries."""

_PACKED_SCALES_PER_INT32 = 4


# =============================================================================
# Shape helpers
# =============================================================================


def _pad4(x: int) -> int:
    return pad_up(x, 4)


def scale_span_elems(rows: int, scale_cols: int) -> int:
    """int32 elements of an MN-major packed-scale storage that hold data.

    The last element of an ``(rows, P)`` view with strides ``(1, pad4(rows))`` sits at
    ``(P - 1) * pad4(rows) + rows - 1``; the quantizer sizes its storage to exactly that view,
    so a copy may move this many elements, not ``P * pad4(rows)``.
    """
    return (scale_cols - 1) * _pad4(rows) + rows


def packed_weight_scale_cols(k: int) -> int:
    """int32 columns of DeepGEMM's packed weight scale for ``K = k``: ``ceil(ceil(K/128) / 4)``."""
    return math.ceil(math.ceil(k / FP8_BLOCK_SIZE) / _PACKED_SCALES_PER_INT32)


def mn_major_scales(sf: torch.Tensor) -> torch.Tensor:
    """``sf`` in the layout DeepGEMM reads: ``(m, P)`` int32 with strides ``(1, pad4(m))``.

    Returned as is when already in that layout (the quantizer's output); otherwise one small
    copy, as ``Fp8PrequantizedSwapABGemmRunner`` makes (Inductor materializes a scale with
    strides ``(1, m)``).
    """
    lead = _pad4(sf.shape[0])
    if sf.stride() == (1, lead):
        return sf
    fixed = torch.empty_strided(sf.shape, (1, lead), dtype=sf.dtype, device=sf.device)
    fixed.copy_(sf)
    return fixed


def _scale_span(sf: torch.Tensor) -> torch.Tensor:
    """The contiguous 1-D int32 storage span behind MN-major ``sf`` (one memcpy moves it)."""
    m, p = sf.shape
    return sf.as_strided((scale_span_elems(m, p),), (1,), sf.storage_offset())


# =============================================================================
# Region layout
# =============================================================================


@dataclass(frozen=True)
class RegionLayout:
    """One source's region of a slot: the fp8 payload ``[m, K]``, then the packed scales.

    The scales are stored as DeepGEMM reads them, int32 ``(m, P)`` with strides
    ``(1, pad4(m))`` over a ``[P, pad4(m)]`` storage, ``P = fp8_scale_cols(K)``. Payload and
    scale storage start ``align``-byte aligned. Every rank holds the same ``m``
    (``TokenShardPlan.local_rows``), so one layout describes every region of a slot; a slot is
    ``tp_size`` regions in source-rank order. Python ints only (capture safe).

    Attributes:
        rows: ``m``, rows per source rank.
        cols: ``K``, a positive multiple of 128.
        align: Byte alignment of the region and of the scale storage (a power of two >= 16;
            the pool's ``DEFAULT_ALIGN``, so a slot of ``tp_size`` regions is carved exactly).
    """

    rows: int
    cols: int
    align: int = DEFAULT_ALIGN

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1 or self.cols % FP8_BLOCK_SIZE:
            raise ValueError(
                f"RegionLayout needs rows >= 1 and cols a positive multiple of "
                f"{FP8_BLOCK_SIZE} (got rows={self.rows}, cols={self.cols})."
            )
        if self.align < 16 or self.align & (self.align - 1):
            raise ValueError(f"RegionLayout align must be a power of two >= 16 (got {self.align}).")

    @property
    def scale_cols(self) -> int:
        """``P``, packed int32 scale columns."""
        return fp8_scale_cols(self.cols)

    @property
    def scale_lead(self) -> int:
        """``pad4(m)``, the leading dimension of the scale storage."""
        return _pad4(self.rows)

    @property
    def payload_nbytes(self) -> int:
        return self.rows * self.cols

    @property
    def scale_offset(self) -> int:
        """Byte offset of the scale storage inside the region."""
        return pad_up(self.payload_nbytes, self.align)

    @property
    def scale_nbytes(self) -> int:
        return self.scale_cols * self.scale_lead * 4

    @property
    def span_elems(self) -> int:
        """int32 elements of the scale storage a push writes (:func:`scale_span_elems`)."""
        return scale_span_elems(self.rows, self.scale_cols)

    @property
    def region_nbytes(self) -> int:
        return pad_up(self.scale_offset + self.scale_nbytes, self.align)

    def slot_nbytes(self, tp_size: int) -> int:
        """Bytes one slot needs: ``tp_size`` regions."""
        return tp_size * self.region_nbytes

    def consumer_views(self, region: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(fp8 [m, K], scales (m, P) strided (1, pad4(m)))`` over a region's uint8 bytes."""
        self._check_region(region)
        fp8 = region[: self.payload_nbytes].view(torch.float8_e4m3fn).view(self.rows, self.cols)
        store = region[self.scale_offset : self.scale_offset + self.scale_nbytes].view(torch.int32)
        return fp8, store.view(self.scale_cols, self.scale_lead).t()[: self.rows]

    def producer_views(self, region: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(payload uint8 [m * K], scale span int32)``: what a push writes into a region."""
        self._check_region(region)
        payload = region[: self.payload_nbytes]
        store = region[self.scale_offset : self.scale_offset + self.scale_nbytes].view(torch.int32)
        return payload, store[: self.span_elems]

    def _check_region(self, region: torch.Tensor) -> None:
        if region.dtype != torch.uint8 or region.dim() != 1 or region.numel() < self.region_nbytes:
            raise ValueError(
                f"RegionLayout: need a 1-D uint8 region of >= {self.region_nbytes} bytes; got "
                f"{region.dtype} {tuple(region.shape)}."
            )
        if region.data_ptr() % 16:
            raise RuntimeError(
                f"RegionLayout: region base {region.data_ptr():#x} is not 16-byte aligned "
                "(TMA operands need it); the pool's region stride is misaligned."
            )


# =============================================================================
# Per-chunk GEMM and the consumer check
# =============================================================================


def fp8_block_gemm_out(
    a_fp8: torch.Tensor,
    a_sf: torch.Tensor,
    w_fp8: torch.Tensor,
    w_sf: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out[:] = a @ w.T`` with the kernel ``trtllm::fp8_prequantized_swap_ab_gemm`` launches.

    No per-call validation: the consumer's operands are validated once at conversion
    (:func:`validate_fp8_block_consumer`) and the activation views come from the transport.

    Args:
        a_fp8: ``[m, K]`` float8_e4m3fn, contiguous.
        a_sf: ``(m, P)`` int32 packed UE8M0 scales with strides ``(1, pad4(m))``.
        w_fp8: ``[N, K]`` float8_e4m3fn (the Linear's ``weight``).
        w_sf: ``(N, ceil(ceil(K/128)/4))`` int32 with strides ``(1, pad4(N))`` (the Linear's
            ``weight_scale`` after ``FP8BlockScalesLinearMethod.transform_weights``).
        out: ``[m, N]`` bfloat16, contiguous (a row block of a larger output qualifies).
    """
    deep_gemm.fp8_gemm_nt((a_fp8, a_sf), (w_fp8, w_sf), out, disable_ue8m0_cast=True)


def validate_fp8_block_consumer(linear: object) -> None:
    """Check once that ``linear``'s operands fit :func:`fp8_block_gemm_out`.

    Called once after the weights are loaded (``TokenShardedTP`` does so at the first
    effective ``begin()``): ``weight`` must be a contiguous float8_e4m3fn ``[N, K]`` with
    ``K % 128 == 0`` and ``weight_scale`` the packed int32 ``(N, ceil(ceil(K/128)/4))`` with
    strides ``(1, pad4(N))`` that ``FP8BlockScalesLinearMethod.transform_weights`` produces and
    the kernel reads. The op body re-checks only the dtypes per call, as a rank-symmetric
    backstop.

    Raises:
        ValueError: ``linear`` is not an FP8 block-scale ``Linear`` or an operand does not fit.
    """
    if (
        not isinstance(linear, Linear)
        or type(linear.quant_method) is not FP8BlockScalesLinearMethod
    ):
        raise ValueError(
            "copy-engine gather: the consumer must be a Linear with the "
            f"FP8BlockScalesLinearMethod; got {type(linear).__name__}."
        )
    w = linear.weight
    if w.dtype != torch.float8_e4m3fn or w.dim() != 2 or not w.is_contiguous():
        raise ValueError(
            f"copy-engine gather: weight must be a contiguous float8_e4m3fn [N, K]; got "
            f"{w.dtype} {tuple(w.shape)}."
        )
    n, k = w.shape
    if k % FP8_BLOCK_SIZE:
        raise ValueError(f"copy-engine gather: K={k} is not a multiple of {FP8_BLOCK_SIZE}.")
    ws = linear.weight_scale
    packed = (n, packed_weight_scale_cols(k))
    if ws.dtype != torch.int32 or tuple(ws.shape) != packed or ws.stride() != (1, _pad4(n)):
        raise ValueError(
            f"copy-engine gather: weight_scale must be the packed int32 {packed} with strides "
            f"(1, {_pad4(n)}) that transform_weights produces after loading; got {ws.dtype} "
            f"{tuple(ws.shape)} strides {tuple(ws.stride())}."
        )


# =============================================================================
# Transport
# =============================================================================


class CeAllGather(CeWorkspace):
    """Per-source copy-engine all-gather over a ``SymmMemPool`` (see the module docstring).

    Payload views and graph-safe capacity belong to :class:`CeWorkspace`; stream and
    signal ordering belong to the shared :class:`CeTransport`. The final pool channel
    fences completion. This class chooses the payload layout and GEMM schedule.

    Args:
        ce: The TP group's shared transport and pool (its ``signal`` / ``wait`` carry the timeout).
        tp_size: Ranks in the TP group.
        rank: This process's rank in the group.
        device: The CUDA device the side stream runs on.
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
        self.push_order: tuple[int, ...] = tuple((rank - i) % tp_size for i in range(1, tp_size))
        """Peers in push order ``me - 1, me - 2, ...``: each consumer's first remote chunk is
        the first one pushed to it (its ``consume_order`` starts at ``me + 1``)."""
        self.consume_order: tuple[int, ...] = tuple((rank + i) % tp_size for i in range(1, tp_size))
        """Remote sources in the order the GEMMs consume them, ``me + 1, me + 2, ...``, which is
        the order their pushes reach this rank."""
        super().__init__(
            ce,
            tp_size,
            rank,
            device,
            slots=slots,
        )
        self._layout: list[RegionLayout | None] = [None] * slots
        self._local: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * slots

    # --- channels / capacity --------------------------------------------------------------

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

    def covers(self, layout: RegionLayout) -> bool:
        """Whether the current reservation holds ``layout`` on every slot."""
        return self._covers(layout.slot_nbytes(self.world_size), self.slots)

    def reserve(self, layout: RegionLayout) -> bool:
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
                f"CeAllGather.reserve: the pool holds {self.pool.slots} x {self.slot_nbytes} "
                f"bytes after reserve(); need {self.slots} x {layout.slot_nbytes(self.world_size)}."
            )
        return realloc

    def _bind_layout(self, slot: int, layout: RegionLayout) -> None:
        # Region views are valid for the layout a slot was issued with: drop that slot's cache
        # when its layout changes; the other slot (possibly in flight) keeps its own.
        if layout != self._layout[slot]:
            self._layout[slot] = layout
            self._drop_views(slot)

    def _local_region(self, slot: int, src: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Source ``src``'s region in MY slot: what my GEMM reads (cached views)."""
        layout = self._layout[slot]
        return self.view(
            (slot, "local", src),
            lambda: layout.consumer_views(self.pool.region(slot, src, 0, layout.region_nbytes)),
        )

    def _peer_region(self, slot: int, peer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """MY region in ``peer``'s slot: what my push writes (cached views)."""
        layout = self._layout[slot]
        return self.view(
            (slot, "peer", peer),
            lambda: layout.producer_views(
                self.pool.peer_region(peer, slot, 0, layout.region_nbytes)
            ),
        )

    # --- the boundary -----------------------------------------------------------------------

    def issue(self, x_fp8: torch.Tensor, x_sf: torch.Tensor, slot: int) -> None:
        """Push this rank's rows into every peer's slot ``slot`` and raise the ready flags.

        ``x_fp8`` is ``[m, K]`` float8_e4m3fn, ``x_sf`` its ``(m, P)`` int32 packed scales in any
        stride layout (re-strided with one small copy when not MN-major). Enqueues only. The
        host checks come first, so a refusal leaves the slot idle; the boundary is open (the
        ready flags owed, a capture noted) before the first copy is enqueued.
        """
        self._check_slot(slot)
        if self._in_flight[slot]:
            raise RuntimeError(
                f"CeAllGather.issue(slot={slot}) before done({slot}) of its previous use: the "
                "complete the active boundary before reusing its slot."
            )
        m, k = x_fp8.shape
        layout = RegionLayout(m, k)
        if not self.covers(layout):
            raise RuntimeError(
                f"CeAllGather.issue: the pool holds {self.slot_nbytes} bytes per slot, a "
                f"[{m}, {k}] boundary needs {layout.slot_nbytes(self.world_size)}; reserve() "
                "eagerly for the largest boundary first."
            )
        x_fp8 = x_fp8.contiguous()
        x_sf = mn_major_scales(x_sf)
        span = _scale_span(x_sf)
        payload_src = x_fp8.view(torch.uint8).view(-1)
        self._bind_layout(slot, layout)
        ready = self._ready_channel(slot)
        self.open(slot, [(src, ready) for src in self.consume_order])
        self._local[slot] = (x_fp8, x_sf)

        def copy_to(peer: int) -> None:
            payload_dst, span_dst = self._peer_region(slot, peer)
            payload_dst.copy_(payload_src)
            span_dst.copy_(span)
            # Contiguous conversions can own temporary storage on the producer stream.
            x_fp8.record_stream(torch.cuda.current_stream())
            x_sf.record_stream(torch.cuda.current_stream())

        self.ce.push(ready, copy_to, destinations=self.push_order)

    def chunk(self, src: int, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Source ``src``'s ``(fp8 [m, K], scales (m, P) strided (1, pad4(m)))`` for the GEMM.

        The own rank returns the local tensors without a wait; a remote source makes the current
        stream wait for its ready flag first (once per boundary).
        """
        self._check_slot(slot)
        if not self._in_flight[slot]:
            raise RuntimeError(f"CeAllGather.chunk(slot={slot}) needs an issue({slot}) in flight.")
        if not 0 <= src < self.world_size:
            raise ValueError(f"source rank {src} out of range for tp_size={self.world_size}.")
        if src == self.rank:
            return self._local[slot]
        self.ce.wait_ready(src, self._ready_channel(slot))
        return self._local_region(slot, src)

    def done(self, slot: int) -> None:
        """Complete the boundary on ``slot``: wait the ready flags no chunk consumed, join the
        side stream into the current stream, release the slot to the peers and note a captured
        boundary. Call after the last GEMM of the boundary was issued on the current stream.
        """
        self._check_slot(slot)
        if not self._in_flight[slot]:
            raise RuntimeError(f"CeAllGather.done(slot={slot}) without an issue({slot}) in flight.")
        super().done(slot)
        self._local[slot] = None


# =============================================================================
# Per-group state and registry
# =============================================================================


class CeGatherState(CeState):
    """The copy-engine gather state of one TP group: pool, transport, slot counter, probe verdict.

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
        self, group: dist.ProcessGroup, group_name: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS
    ) -> None:
        super().__init__(group, group_name, timeout_ms=timeout_ms)
        self.consumer_ks: set[int] = set()
        self._counter = 0

    @property
    def tp_size(self) -> int:
        """``world_size``, in the TP vocabulary of the adapters."""
        return self.world_size

    # --- conversion time ------------------------------------------------------------------

    def note_consumer(self, k: int) -> None:
        """Record a consumer's ``K`` (``in_features``) so :meth:`prepare` can size the pool."""
        if k < 1 or k % FP8_BLOCK_SIZE:
            raise ValueError(
                f"note_consumer: K must be a positive multiple of {FP8_BLOCK_SIZE}; got {k}."
            )
        self.consumer_ks.add(int(k))

    # --- begin() ----------------------------------------------------------------------------

    def prepare(self, plan: "TokenShardPlan") -> None:
        """Probe once, start the forward's slot sequence and size the pool for ``plan``.

        Eager, every rank in the same order (``TokenShardedTP.begin``). Checks that the
        previous boundary chain is complete and restarts the slots at 0 (:meth:`begin_forward`,
        so the caller need not call it), then sizes the pool for ``plan.local_rows`` x the
        largest noted consumer ``K``, grow-only. Under CUDA-graph capture only the slot counter
        is reset; a probe or an allocation under capture raises, as does a closed state.
        """
        what = f"Token-sharded TP ({self.group_name}): copy-engine all-gather"
        if not self.probe_once(min_channels=SLOTS + 1, what=what):
            return
        self.begin_forward()
        assert self.transport is not None
        if not self.consumer_ks:
            return
        layout = RegionLayout(plan.local_rows, max(self.consumer_ks))
        if self.transport.covers(layout):
            return
        if _is_capturing():
            raise RuntimeError(
                f"copy-engine gather: TokenShardedTP.begin() under CUDA-graph capture needs a "
                f"pool for {layout.slot_nbytes(self.world_size)} bytes per slot, the pool holds "
                f"{self.transport.slot_nbytes}; run an eager forward of this shape first."
            )
        self.transport.reserve(layout)  # logs the size, the deadline and the triage hint
        logger.info_once(
            f"Token-sharded TP ({self.group_name}): copy-engine all-gather on.",
            key=("token_sharded_ce_gather", self.group_name),
        )

    def begin_forward(self) -> None:
        """Start a forward: the previous boundary chain must be complete; slots restart at 0."""
        if self.transport is not None:
            self.transport.drain()
        self._counter = 0

    def next_slot(self) -> int:
        """The slot of the next boundary (alternating; reset by :meth:`begin_forward`)."""
        slot = self._counter % SLOTS
        self._counter += 1
        return slot

    # --- probe --------------------------------------------------------------------------------

    def _make_transport(self) -> None:
        assert self.ce is not None
        self.transport = CeAllGather(self.ce, self.world_size, self.rank, self._device)


# The registry of this protocol, as TokenShardedTP and the op body call it.
get_state = CeGatherState.get
register_state = CeGatherState.register
release_state = CeGatherState.release


# =============================================================================
# The op body
# =============================================================================


def _check_op_operands(
    x_fp8: torch.Tensor, x_sf: torch.Tensor, w_fp8: torch.Tensor, w_sf: torch.Tensor, m: int
) -> None:
    # Against the plan ints and the weight shape only: identical on every rank, so a raise here
    # is rank-symmetric and leaves no peer waiting.
    n, k = w_fp8.shape
    if x_fp8.dtype != torch.float8_e4m3fn or tuple(x_fp8.shape) != (m, k):
        raise ValueError(
            f"token_sharded_fp8_ce_gather_gemm: expected this rank's float8_e4m3fn [{m}, {k}] "
            f"rows; got {x_fp8.dtype} {tuple(x_fp8.shape)}."
        )
    if x_sf.dtype != torch.int32 or tuple(x_sf.shape) != (m, fp8_scale_cols(k)):
        raise ValueError(
            f"token_sharded_fp8_ce_gather_gemm: expected int32 packed scales "
            f"[{m}, {fp8_scale_cols(k)}]; got {x_sf.dtype} {tuple(x_sf.shape)}."
        )
    if w_fp8.dtype != torch.float8_e4m3fn or w_sf.dtype != torch.int32:
        raise ValueError(
            "token_sharded_fp8_ce_gather_gemm: the weight must be float8_e4m3fn with the packed "
            f"int32 weight_scale (post_load_weights); got {w_fp8.dtype} / {w_sf.dtype} for "
            f"N={n}, K={k}."
        )


def ce_gather_gemm_impl(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    w_fp8: torch.Tensor,
    w_sf: torch.Tensor,
    group_name: str,
    tp_rank: int,
    tp_size: int,
    batch_size: int,
    seq_len: int,
    padded_seq_len: int,
) -> torch.Tensor:
    """One projection using the shared copy-engine gather protocol."""
    return ce_gather_gemms_impl(
        x_fp8,
        x_sf,
        [w_fp8],
        [w_sf],
        group_name,
        tp_rank,
        tp_size,
        batch_size,
        seq_len,
        padded_seq_len,
    )[0]


def ce_gather_gemms_impl(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    weights: list[torch.Tensor],
    weight_scales: list[torch.Tensor],
    group_name: str,
    tp_rank: int,
    tp_size: int,
    batch_size: int,
    seq_len: int,
    padded_seq_len: int,
) -> list[torch.Tensor]:
    """One exchange, all consumer GEMMs per source, then release the shared slot.

    Returns the ``[batch_size * seq_len, N]`` bf16 GEMM output of every rank's real rows (pad
    rows dropped), without bias, for each weight. Order on the current stream: own chunk, then each
    remote source in ``transport.consume_order`` after a stream wait for its arrival, each into
    its row block ``out[src * m:(src + 1) * m]`` of the padded ``[tp_size * m, N]`` output.

    Args:
        x_fp8: This rank's ``[m, K]`` float8_e4m3fn rows, ``m = batch_size * padded_seq_len /
            tp_size``.
        x_sf: Their ``(m, P)`` int32 packed UE8M0 scales.
        weights: Each consumer's ``[N_local, K]`` float8_e4m3fn weight, in consumer order.
        weight_scales: Corresponding packed int32 weight scales.
        group_name: The TP group's c10d name (the registry key).
        tp_rank: This rank in the group.
        tp_size: Ranks in the group.
        batch_size: ``B``.
        seq_len: ``S``.
        padded_seq_len: ``S_pad``.
    """
    if not weights or len(weights) != len(weight_scales):
        raise ValueError("shared gather needs a nonempty weight list and one scale per weight")
    try:
        state = get_state(group_name)
    except KeyError:
        raise RuntimeError(
            f"token_sharded_fp8_ce_gather_gemm: no CeGatherState is registered for TP group "
            f"{group_name!r}; TokenShardedTP(gather_mode='copy_engine') registers one at its "
            "first begin()."
        ) from None
    transport = state.transport
    if transport is None:
        raise RuntimeError(
            f"token_sharded_fp8_ce_gather_gemm: the copy-engine gather is not effective for TP "
            f"group {group_name!r} ({state.reason}); the adapters must take the NCCL path."
        )
    if tp_size != transport.tp_size or tp_rank != transport.rank:
        raise ValueError(
            f"token_sharded_fp8_ce_gather_gemm: plan rank {tp_rank}/{tp_size} does not match "
            f"the group's {transport.rank}/{transport.tp_size}."
        )
    padded_rows = batch_size * padded_seq_len
    if padded_rows % tp_size:
        raise ValueError(
            f"token_sharded_fp8_ce_gather_gemm: B * S_pad = {padded_rows} is not a multiple of "
            f"tp_size={tp_size}."
        )
    m = padded_rows // tp_size
    for weight, scale in zip(weights, weight_scales):
        _check_op_operands(x_fp8, x_sf, weight, scale, m)
    outputs = [
        torch.empty((tp_size * m, w.shape[0]), dtype=torch.bfloat16, device=x_fp8.device)
        for w in weights
    ]
    slot = state.next_slot()
    try:
        transport.issue(x_fp8, x_sf, slot)
        for src in (tp_rank, *transport.consume_order):
            a_fp8, a_sf = transport.chunk(src, slot)
            for weight, scale, out in zip(weights, weight_scales, outputs):
                fp8_block_gemm_out(a_fp8, a_sf, weight, scale, out[src * m : (src + 1) * m])
        transport.done(slot)
    except CeUnavailableError as exc:
        state.disable(str(exc))
        raise
    except Exception:
        transport.reset()
        raise
    outputs = [drop_padding(out, batch_size, seq_len, padded_seq_len) for out in outputs]
    return [out if out.is_contiguous() else out.contiguous() for out in outputs]
