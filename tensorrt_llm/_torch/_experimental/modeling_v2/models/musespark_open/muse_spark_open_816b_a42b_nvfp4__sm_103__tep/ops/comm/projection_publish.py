# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transports the column-sharded pool and unpool GEMMs publish into on 33..512-row decode steps (TP4).

From 33 rows the built-in MoE multiplies the whole replicated pool (8192 -> 4096) and
unpool (4096 -> 8192) weights on every rank: 67 MB per projection per rank. Here each
rank multiplies only its column slice (16.8 MB: pool ``[R, 8192] x [1024, 8192]^T``,
unpool ``[R, 4096] x [2048, 4096]^T``) and the GEMM's output tensor *is* this rank's slot
of a multicast transport, so the GEMM's own epilogue stores publish the slice to every
rank. The consumers -- the pre-expert RMSNorm + NVFP4 quantization
(``pool_publish_fp4quant``) and the post-FFN join (``unpool_publish_join`` below) --
poll their rows of their own copy, reset them and compute the built-in kernels'
arithmetic on the gathered row.

Transport (Lamport, payload as flag). Every rank holds ``BUFFERS x TP source ranks x
max_rows x columns`` BF16 -- rank-major, so rank ``r``'s slot of buffer ``b`` is one
contiguous ``[rows, columns]`` block, a GEMM output as it is -- filled with the sentinel
word ``0xFFFEFFFE``: two BF16 NaNs with payload ``0x7E``. Float-to-BF16 rounding
(``cvt.rn.bf16{x2}.f32``, the GEMM epilogues') turns every NaN into the canonical
``0x7FFF``, so no GEMM output holds the half ``0xFFFE``; a 16-bit half that still holds
it has not arrived. The test is per 16-bit half, so a store that reached the poller in
pieces reads as not arrived until every piece did. Own slot included: the producer's
multicast stores reach the local copy asynchronously, like the peers'.

Ownership and reuse (cuFlow):

* One writer per element per call: rank ``r``'s slice GEMM (``publish_projection``,
  cuBLAS through ``torch.mm`` with the slot as its output) stores each element of slot
  ``r`` once, from its epilogue, with its final value. cuBLAS picks a beta-zero kernel
  (no read-back of the output); where it splits K it reduces the partial products from
  its workspace in a separate reduction kernel, whose single store is the publication.
  The census test drives this with a concurrent poller at every served row count;
  the census holds for the cuBLAS it ran on, so the path publishes only with that
  cuBLAS, torch and device (``producer_refusal``).
* One reader and resetter per element per call: the consumer thread of each rank that
  owns that element of its row, on its own copy.
* Reuse: no two consecutive calls on a transport use the same buffer, the last call of a
  step and the first of the next included (``buffer_sequence``: the buffers alternate
  call by call and the last call of an odd cycle takes the third). Rank ``A``'s producer
  of call ``m`` stores only after ``A``'s call ``m - 1`` completed (stream order), which
  needed rank ``B``'s call-``(m - 1)`` slot, which ``B`` stored only after every earlier
  call of ``B`` -- among them ``B``'s last call on that buffer, resets included --
  completed.
* Progress: producers wait on nothing remote; a consumer thread waits only for its own
  elements of the call. A consumer is launched in stream order after its producer, so no
  poller of a call is ever in flight on a rank while that rank still has to launch a
  producer the poll depends on (no co-residency assumption).
* Poll guard: a poll that has seen no arrival for its guard sets a guard word in pinned
  host memory and keeps polling; it never consumes a half that has not arrived. A late
  peer still completes the call exactly; a slot that never arrives leaves the rank
  waiting (the host watch logs the rank, the executor's hang detector is the backstop).
  ``guard_break``, which stops the poll, exists for tests of a broken call sequence only.

Calls on one transport must be issued in the same order on every rank, with the same
row counts and buffers. The transports are this path's alone: the 8-row and 32-row
inboxes and every other collective use their own memory.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from typing import Optional, Sequence, Tuple

import torch
from cuda.bindings import runtime as _cudart

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .unpool_direct_boundary import buffer_sequence, check_buffer_sequence

__all__ = [
    "BUFFERS",
    "MAX_ROWS",
    "MIN_ROWS",
    "POOL_HIDDEN",
    "PublishTransport",
    "UNPOOL_HIDDEN",
    "WORLD",
    "buffer_sequence",
    "check_buffer_sequence",
    "local_columns",
    "producer_refusal",
    "publish_projection",
    "unpool_publish_join",
]

#: The TP size the transports connect: one GB300 tray.
WORLD = 4
#: The decode rows the transports serve: five to 64 requests at K = 7 (and the counts between).
MIN_ROWS, MAX_ROWS = 33, 512
#: Width of the pooled (expert) activation and of the residual stream.
POOL_HIDDEN, UNPOOL_HIDDEN = 4096, 8192
#: Buffers of a transport (``buffer_sequence``).
BUFFERS = 3
#: The empty-slot word: two BF16 NaNs with the payload no float-to-BF16 rounding produces.
SENTINEL = 0xFFFEFFFE
SENTINEL_HALF = 0xFFFE
_SENTINEL_I16 = SENTINEL_HALF - (1 << 16)
#: Per-rank control words in pinned host memory: ``GUARD_WORD`` is set by a poll that
#: outlived its guard.
CONTROL_WORDS = 4
GUARD_WORD = 0
#: Default poll guard of the CuTe consumer in ns, far above any wait the protocol produces.
GUARD_NS = 1 << 33
#: Default poll guard of the Triton consumer in polling rounds (each round reloads the row;
#: far above any wait the protocol produces).
GUARD_ROUNDS = 1 << 24
#: Seconds between two reads of the guard word by the watch thread.
GUARD_WATCH_S = 1.0
#: Seconds between two warnings of the watch thread while polls keep outliving their guard.
GUARD_RELOG_S = 60.0
#: The producers the census certified: at every served row count of both sites, cuBLAS
#: (``torch.mm`` into the slot) picks beta-zero kernels that store each element once, and
#: splits K only through a separate reduction kernel. Keyed by (torch version, cuBLASLt
#: version, SM count): cuBLAS's kernel choice depends on all three.
CERTIFIED_PRODUCERS = frozenset({("2.14.0a0+b2c75dd062.nv26.09", 130800, 152)})
#: Environment variables that resize torch's cuBLAS workspaces, which changes cuBLAS's choice
#: of split-K kernels; the census ran with none set.
CUBLAS_WORKSPACE_ENV = ("CUBLAS_WORKSPACE_CONFIG", "CUBLASLT_WORKSPACE_SIZE")


def local_columns(width: int, world: int = WORLD) -> int:
    """Columns of a ``width``-wide projection output one of ``world`` ranks computes."""
    return width // world


class PublishTransport:
    """One site's multicast transport on every rank of a TP4 group, for up to ``max_rows`` rows.

    ``slot(buffer, rows)`` is this rank's ``[rows, columns]`` BF16 slot of ``buffer`` on the
    multicast address: a GEMM given it as its output publishes into every rank's copy;
    nothing may read it. ``receive()`` is this rank's copy (``int16``,
    ``[BUFFERS, WORLD, max_rows, columns]``), which the consumers poll and reset.
    Construction is collective over the group (fabric allocation, sentinel fill) and
    succeeds on every rank or raises on every rank.

    The guard word lives in pinned host memory, so the host reads it while the device still
    polls (``guard_fired(sync=False)``, ``start_guard_watch``).
    """

    def __init__(self, mapping: Mapping, columns: int, max_rows: int = MAX_ROWS, comm=None):
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_device_index,
            _mnnvl_workspace_all_succeeded,
        )

        if mapping.tp_size != WORLD or mapping.enable_attention_dp or mapping.cp_size != 1:
            raise ValueError(
                "the projection publish transport needs TP4 without attention DP or CP"
            )
        if columns not in (local_columns(POOL_HIDDEN), local_columns(UNPOOL_HIDDEN)):
            raise ValueError(f"no projection site has {columns} columns per rank at TP{WORLD}")
        if not MIN_ROWS <= max_rows <= MAX_ROWS:
            raise ValueError(f"max_rows must be in {MIN_ROWS}..{MAX_ROWS}, got {max_rows}")
        self.rank = mapping.tp_rank
        self.world = mapping.tp_size
        self.columns = int(columns)
        self.max_rows = int(max_rows)
        self.nbytes = BUFFERS * self.world * self.max_rows * self.columns * 2
        self.device = torch.device("cuda", _mnnvl_device_index(mapping))
        self._watch: Optional[threading.Thread] = None
        #: Warnings the watch thread logged (tests).
        self.guard_warnings = 0
        #: Device memory the transport took on this rank (the multicast allocation rounds up).
        self.allocated = 0
        self._comm = comm if comm is not None else _get_mnnvl_workspace_comm(mapping)
        refusal: Optional[str] = None
        if torch.cuda.get_device_capability(self.device) != (10, 3):
            refusal = "the projection publish transport runs on SM103 only"
        elif not prefer_pinned():
            refusal = "the guard words need pinned host memory, which this platform does not pin"
        if not _mnnvl_workspace_all_succeeded(self._comm, refusal is None):
            raise RuntimeError(
                "projection publish transport refused on at least one rank"
                f"{f' (this rank: {refusal})' if refusal is not None else ''}"
            )
        error: Optional[BaseException] = None
        try:
            torch.cuda.synchronize(self.device)
            free_before, _ = torch.cuda.mem_get_info(self.device)
            self._handle = _make_mnnvl_mcast_buffer(
                self._comm, self.nbytes, mapping, mapping.is_multi_node()
            )
            shape = (BUFFERS, self.world, self.max_rows, self.columns)
            self._receive = self._handle.get_uc_buffer(self.rank, shape, torch.int16, 0)
            self._slots = self._handle.get_mc_buffer(shape, torch.bfloat16, 0)
            with torch.cuda.device(self.device):
                self._receive.fill_(_SENTINEL_I16)
                # Pinned host memory is mapped into the device's address space at the same
                # address (unified addressing): a guard store reaches it at system scope and
                # the host reads it without a synchronization.
                self.control = torch.zeros(
                    CONTROL_WORDS, dtype=torch.int32, pin_memory=prefer_pinned()
                )
                status, pointer = _cudart.cudaHostGetDevicePointer(self.control.data_ptr(), 0)
                if status != _cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaHostGetDevicePointer of the guard words: {status}")
                self._control_ptr = int(pointer)
                torch.cuda.synchronize(self.device)
            self.allocated = max(0, free_before - torch.cuda.mem_get_info(self.device)[0])
        except Exception as exc:  # noqa: BLE001 - every failure must reach the agreement
            error = exc
        if not _mnnvl_workspace_all_succeeded(self._comm, error is None):
            raise RuntimeError(
                "projection publish transport failed on at least one rank"
                f"{f' (this rank: {error!r})' if error is not None else ''}"
            ) from error

    def slot(self, buffer: int, rows: int) -> torch.Tensor:
        """This rank's ``[rows, columns]`` BF16 slot of ``buffer``, on the multicast address.

        Write-only: a store through it reaches every rank's copy; nothing may read it.
        """
        if buffer not in range(BUFFERS) or not 1 <= rows <= self.max_rows:
            raise ValueError(f"no slot for buffer {buffer}, {rows} rows")
        return self._slots[buffer, self.rank, :rows]

    def is_slot(self, tensor: torch.Tensor, buffer: int, rows: int) -> bool:
        """Whether ``tensor`` is this rank's slot of ``buffer`` at ``rows`` rows (address and shape)."""
        return (
            buffer in range(BUFFERS)
            and 1 <= rows <= self.max_rows
            and tensor.dim() == 2
            and tuple(tensor.shape) == (rows, self.columns)
            and tensor.data_ptr() == self._slots[buffer, self.rank].data_ptr()
        )

    def receive(self) -> torch.Tensor:
        """This rank's copy of the transport, ``int16 [BUFFERS, WORLD, max_rows, columns]``."""
        return self._receive

    def control_ptr(self) -> int:
        """The device address of the guard words (pinned host memory)."""
        return self._control_ptr

    def is_clean(self) -> bool:
        """Whether every half of this rank's copy holds the sentinel (synchronizes; tests)."""
        torch.cuda.synchronize(self.device)
        return bool((self._receive == _SENTINEL_I16).all().item())

    def reset(self) -> None:
        """Every half of this rank's copy back to the sentinel (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self._receive.fill_(_SENTINEL_I16)
        torch.cuda.synchronize(self.device)

    def guard_fired(self, sync: bool = True) -> bool:
        """Whether a poll of this transport ever outlasted its guard.

        ``sync``: synchronize the device first (tests). Without it the word is read as it
        stands in host memory, while kernels may still be polling.
        """
        if sync:
            torch.cuda.synchronize(self.device)
        return int(self.control[GUARD_WORD]) != 0

    def clear_guard(self) -> None:
        """Zero the guard words (tests; nothing may be in flight)."""
        torch.cuda.synchronize(self.device)
        self.control.zero_()

    def start_guard_watch(
        self,
        name: str,
        interval_s: float = GUARD_WATCH_S,
        abort: bool = False,
        relog_s: float = GUARD_RELOG_S,
    ) -> bool:
        """Read the guard word from a host thread every ``interval_s`` seconds (0: no thread).

        An expired guard means a poll has waited longer than its guard for its slot. The
        consumer keeps polling and consumes nothing before the slot arrives: a peer that is
        only late (a host stall) still completes the call exactly, while a slot that never
        arrives (the ranks' call sequences differ, or a peer stopped) leaves this rank
        waiting, which the executor's hang detector reports. The thread logs a warning
        naming the rank, clears the word and keeps watching (a poll still waiting may set it
        again; at most one warning per ``relog_s``). With ``abort`` it aborts the process at
        the first expiry instead. Idempotent; returns whether a thread runs.
        """
        if self._watch is not None:
            return True
        if interval_s <= 0:
            return False
        stop = threading.Event()

        def watch():
            logged = None
            while not stop.wait(interval_s):
                if not self.guard_fired(sync=False):
                    continue
                if abort:
                    logger.error(
                        f"[musespark_open] projection_publish: rank {self.rank} polled the {name} "
                        "transport past its guard without its slot arriving; aborting"
                    )
                    os.abort()
                self.control[GUARD_WORD] = 0
                now = time.monotonic()
                if logged is not None and now - logged < relog_s:
                    continue
                logged = now
                self.guard_warnings += 1
                logger.warning(
                    f"[musespark_open] projection_publish: rank {self.rank} has polled the {name} "
                    "transport for longer than its guard without its slot arriving. The consumer "
                    "keeps polling and consumes nothing before the slot arrives: a late peer "
                    "completes the call normally; if the slot never arrives (the ranks' call "
                    "sequences differ, or a peer stopped), this rank waits and the executor's hang "
                    "detector reports it"
                )

        self._watch_stop = stop
        self._watch = threading.Thread(
            target=watch, name=f"musespark_open.projection_publish_{name}_watch", daemon=True
        )
        self._watch.start()
        return True

    def stop_guard_watch(self) -> None:
        """Stop the watch thread, if one runs (tests)."""
        if self._watch is not None:
            self._watch_stop.set()
            self._watch.join()
            self._watch = None


def cublaslt_version() -> Optional[int]:
    """``cublasLtGetVersion()`` of the cuBLASLt this process loaded (torch's), or None."""
    try:
        with open("/proc/self/maps") as maps:
            paths = sorted({line.split()[-1] for line in maps if "libcublasLt.so" in line})
    except OSError:
        return None
    for path in paths:
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        library.cublasLtGetVersion.restype = ctypes.c_size_t
        return int(library.cublasLtGetVersion())
    return None


def producer_refusal(device: torch.device) -> Optional[str]:
    """Why ``publish_projection`` may not publish on ``device`` in this process, or None.

    The single-store property of the cuBLAS producer is a property of the kernels cuBLAS
    picks, which the census certified for one torch, cuBLASLt and SM count
    (``CERTIFIED_PRODUCERS``) with torch's default cuBLAS workspaces. Anything else may
    pick a kernel that writes partial sums into the slot or reads it back.
    """
    torch.cuda.init()
    key = (
        str(torch.__version__),
        cublaslt_version(),
        torch.cuda.get_device_properties(device).multi_processor_count,
    )
    if key not in CERTIFIED_PRODUCERS:
        return (
            f"the cuBLAS producer census does not cover (torch, cuBLASLt, SMs) = {key} "
            f"(certified: {sorted(CERTIFIED_PRODUCERS)})"
        )
    overrides = [name for name in CUBLAS_WORKSPACE_ENV if os.environ.get(name)]
    if overrides:
        return f"{', '.join(overrides)} resize the cuBLAS workspaces the producer census ran with"
    return None


def publish_projection(hidden: torch.Tensor, weight: torch.Tensor, slot: torch.Tensor) -> None:
    """``hidden @ weight^T`` (cuBLAS through ``torch.mm``) stored straight into ``slot``.

    ``slot`` is a transport slot (``PublishTransport.slot``): the GEMM's epilogue stores
    are the publication. ``weight`` is the rank's ``[columns, K]`` row slice of the
    projection's weight (contiguous rows of the loaded weight; no copy). Only with a
    certified producer (``producer_refusal``).
    """
    torch.mm(hidden, weight.t(), out=slot)


class _GuardWords:
    """The guard words handed to the Triton consumer as a pointer argument.

    Triton reads ``data_ptr()`` and ``dtype`` of a pointer argument; the pinned guard
    words' device address is their host address (unified addressing).
    """

    dtype = torch.int32

    def __init__(self, pointer: int) -> None:
        self._pointer = int(pointer)

    def data_ptr(self) -> int:
        return self._pointer


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the image ships Triton
    triton = None

if triton is not None:

    @triton.jit(do_not_specialize=["buffer"])
    def _unpool_publish_join_kernel(
        out_ptr,  # [T, HIDDEN] fp32, the new residual stream
        residual_ptr,  # [T, HIDDEN] fp32, the old one
        receive_ptr,  # int16 [BUFFERS, WORLD, CAPACITY, COLUMNS], this rank's copy
        gain_ptr,  # [HIDDEN] (1 + w) when HAS_GAIN, else unread
        alpha_ptr,  # [HIDDEN] fp32
        beta_ptr,  # [HIDDEN] fp32
        eps,
        next_out_ptr,
        next_weight_ptr,
        next_eps,
        guard_ptr,  # int32, pinned host memory
        buffer,
        WORLD_SIZE: tl.constexpr,
        CAPACITY: tl.constexpr,
        COLUMNS: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_GAIN: tl.constexpr,
        NORM_FP32_RESIDUAL: tl.constexpr,
        SENTINEL_I16: tl.constexpr,
        GUARD: tl.constexpr,
        GUARD_BREAK: tl.constexpr,
    ):
        """The built-in ``_gated_residual_after_rmsnorm_kernel`` (FUSE_NEXT_NORM) on a published branch.

        The branch row is read from the transport: element ``c`` of row ``token`` is element
        ``c % COLUMNS`` of source rank ``c // COLUMNS``'s slot of ``buffer``. The row is
        polled until no half holds the sentinel, reset to the sentinel, and then the
        built-in kernel's statements run unchanged on it (the same ``[BLOCK_H]`` tensors and
        ``num_warps``, so the same reductions).
        """
        token = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < HIDDEN
        base = token * HIDDEN + offs

        source = offs // COLUMNS
        box = (
            receive_ptr
            + ((buffer * WORLD_SIZE + source) * CAPACITY + token) * COLUMNS
            + (offs - source * COLUMNS)
        )
        raw = tl.load(box, mask=mask, other=0, volatile=True)
        pending = tl.max((raw == SENTINEL_I16).to(tl.int32), axis=0)
        rounds = 0
        while pending != 0:
            raw = tl.load(box, mask=mask, other=0, volatile=True)
            pending = tl.max((raw == SENTINEL_I16).to(tl.int32), axis=0)
            if GUARD > 0:
                rounds += 1
                if rounds == GUARD:
                    tl.atomic_xchg(guard_ptr, 1, sem="relaxed", scope="sys")
                    if GUARD_BREAK:
                        pending = pending * 0
        # Retire the consumed halves before this grid can complete.
        tl.store(box, tl.full([BLOCK_H], SENTINEL_I16, tl.int16), mask=mask)
        branch = raw.to(tl.bfloat16, bitcast=True)

        scaled = branch.to(tl.float32)
        if HAS_GAIN:
            gain = tl.load(gain_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            scaled = (scaled * gain).to(branch.dtype).to(tl.float32)
        variance = tl.sum(scaled * scaled, axis=0) / HIDDEN
        normed = (scaled * tl.rsqrt(variance + eps)).to(branch.dtype).to(tl.float32)

        residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
        alpha = tl.load(alpha_ptr + offs, mask=mask, other=0.0)
        beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
        updated = alpha * residual + beta * normed
        tl.store(out_ptr + base, updated, mask=mask)
        if NORM_FP32_RESIDUAL:
            next_input = updated
        else:
            next_input = updated.to(branch.dtype).to(tl.float32)
        next_variance = tl.sum(next_input * next_input, axis=0) / HIDDEN
        next_weight = tl.load(next_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        next_output = next_input * tl.rsqrt(next_variance + next_eps)
        next_output = next_output * (next_weight + 1.0)
        tl.store(next_out_ptr + base, next_output, mask=mask)


def _residual_num_warps() -> int:
    """The built-in join's warps per program (its reductions follow from them)."""
    from tensorrt_llm._torch.models.modeling_musespark_open import _RESIDUAL_NUM_WARPS

    return _RESIDUAL_NUM_WARPS


class UnpoolPublishJoin(OpWrapper):
    """The built-in post-FFN join (residual join + next pre-attention norm) on a published unpool slot.

    ``transport`` is the unpool site's ``PublishTransport``; ``buffer`` the call's buffer,
    whose slots every rank's unpool GEMM wrote for this call (``publish_projection``).
    ``rows`` rows; ``residual`` FP32 ``[rows, 8192]``; ``gain``, ``alpha``, ``beta``,
    ``weight``, the epsilons and ``norm_fp32_residual`` are the built-in
    ``MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm``'s. Returns its
    ``(updated, next_input)`` for the gathered branch.

    A collective over the TP4 group: every rank calls it with the same row count and
    buffer, in the same order relative to its other calls on ``transport``, and no two
    consecutive calls on one buffer (``buffer_sequence``). ``guard`` (polling rounds, 0:
    none) and ``guard_break`` select the poll guard (``guard_break`` for tests only).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="eight requests at K = 7 (64 rows)",
            spec=dict(rows=64, with_gain=True, norm_fp32_residual=False),
        ),
        Cell(
            why="five requests: the first served row count of the band (40 rows)",
            spec=dict(rows=40, with_gain=True, norm_fp32_residual=False, buffer=1),
        ),
        Cell(
            why="16 requests at K = 7 (128 rows) on the third buffer",
            spec=dict(rows=128, with_gain=True, norm_fp32_residual=False, buffer=2),
        ),
        Cell(
            why="24 requests at K = 7 (192 rows)",
            spec=dict(rows=192, with_gain=True, norm_fp32_residual=False),
        ),
        Cell(
            why="64 requests: the transport full (512 rows)",
            spec=dict(rows=512, with_gain=True, norm_fp32_residual=False, buffer=1),
        ),
        Cell(
            why="the FP32-residual pre-QKV norm ordering switch",
            spec=dict(rows=128, with_gain=True, norm_fp32_residual=True),
        ),
        Cell(
            why="a weightless first norm (the join's other arm)",
            spec=dict(rows=64, with_gain=False, norm_fp32_residual=False),
        ),
        Cell(
            why="33 rows: the first row count of the band, an odd one",
            spec=dict(rows=33, with_gain=True, norm_fp32_residual=False),
        ),
    )

    note = """
    Certified bit for bit against the built-in Triton join run on the gathered branch
    (the four ranks' slots concatenated), through CUDA-graph replays with changing inputs,
    back-to-back 61-layer chains on the model's buffer sequence with a delayed rank, and
    against `reference` within its band. The slots are published by the production
    producer (cuBLAS into the multicast slot) and by an elementwise copy.
    """

    def __call__(
        self,
        transport: PublishTransport,
        buffer: int,
        rows: int,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        guard: int = GUARD_ROUNDS,
        guard_break: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        updated = torch.empty_like(residual)
        next_input = torch.empty(
            (rows, UNPOOL_HIDDEN), dtype=torch.bfloat16, device=residual.device
        )
        _unpool_publish_join_kernel[(rows,)](
            updated,
            residual,
            transport.receive(),
            gain if gain is not None else alpha,
            alpha,
            beta,
            post_eps,
            next_input,
            weight,
            pre_eps,
            _GuardWords(transport.control_ptr()),
            int(buffer),
            WORLD_SIZE=transport.world,
            CAPACITY=transport.max_rows,
            COLUMNS=transport.columns,
            HIDDEN=UNPOOL_HIDDEN,
            BLOCK_H=UNPOOL_HIDDEN,
            HAS_GAIN=gain is not None,
            NORM_FP32_RESIDUAL=bool(norm_fp32_residual),
            SENTINEL_I16=_SENTINEL_I16,
            GUARD=int(guard),
            GUARD_BREAK=bool(guard_break),
            num_warps=_residual_num_warps(),
        )
        return updated, next_input

    def reference(
        self,
        per_rank_slices: Sequence[torch.Tensor],
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The join on the concatenated slices, float64 between the built-in's BF16 roundings.

        The collective exception to mirroring: ``per_rank_slices`` is every rank's published
        slice in ascending rank order.
        """
        branch = torch.cat(list(per_rank_slices), dim=1)
        scaled = branch.double()
        if gain is not None:
            scaled = (scaled * gain.double()).to(torch.bfloat16).double()
        normed = scaled * torch.rsqrt(scaled.square().mean(-1, keepdim=True) + post_eps)
        normed = normed.to(torch.bfloat16).double()
        updated = alpha.double() * residual.double() + beta.double() * normed
        nxt = updated if norm_fp32_residual else updated.to(torch.bfloat16).double()
        out = nxt * torch.rsqrt(nxt.square().mean(-1, keepdim=True) + pre_eps)
        out = out * (weight.double() + 1.0)
        return updated.float(), out.to(torch.bfloat16)

    def is_valid(
        self,
        transport: PublishTransport,
        buffer: int,
        rows: int,
        residual: torch.Tensor,
        gain: Optional[torch.Tensor],
        alpha: torch.Tensor,
        beta: torch.Tensor,
        weight: torch.Tensor,
        post_eps: float,
        pre_eps: float,
        norm_fp32_residual: bool = False,
        guard: int = GUARD_ROUNDS,
        guard_break: bool = False,
        **unused_kwargs,
    ) -> None:
        """Rows beyond the transport's capacity address outside it; a strided or misaligned
        operand is read as if packed; a non-unpool transport has another row layout."""
        if transport.columns != local_columns(UNPOOL_HIDDEN, transport.world):
            raise ValueError(
                f"transport carries {transport.columns}-column slots, not the unpool's"
            )
        if buffer not in range(BUFFERS) or not 1 <= rows <= transport.max_rows:
            raise ValueError(f"buffer {buffer} / rows {rows} outside the transport")
        expected = (
            (residual, (rows, UNPOOL_HIDDEN), torch.float32),
            (alpha, (UNPOOL_HIDDEN,), torch.float32),
            (beta, (UNPOOL_HIDDEN,), torch.float32),
            (weight, (UNPOOL_HIDDEN,), torch.bfloat16),
        ) + (((gain, (UNPOOL_HIDDEN,), torch.bfloat16),) if gain is not None else ())
        for tensor, shape, dtype in expected:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
                or tensor.device != transport.device
            ):
                raise ValueError(
                    f"expected a contiguous 16-byte aligned {dtype} {shape} tensor on "
                    f"{transport.device}, got {tensor.dtype} {tuple(tensor.shape)}"
                )


unpool_publish_join = UnpoolPublishJoin()


def available() -> bool:
    """Whether this image can build the unpool consumer (Triton)."""
    return triton is not None
