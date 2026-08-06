# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FMHA TS context/prefill kernels.

This file is the unified context-kernel implementation for both query-paired
FMHA and head-paired GQA/sliding-window prefill. The query-paired path maps one
CTA work tile to two consecutive 128-token Q tiles for the same batch and Q
head. With a short left sliding window, those consecutive sequence ranges can
touch three K/V tile ranges. The head-paired path maps one CTA work tile to two
same-sequence-index Q tiles from different Q heads sharing one K/V head, which
keeps each tile to two K/V ranges and enables the left-window case.

Resources are programming-model constructs associated with physical buffers. Each
resource owns the pipeline state that guards concurrent producer and consumer
uses, and defines the programmer work for writing to or reading from that
resource. Tasks describe the schedule order of those work calls for each warp
role. The kernel builder connects the resulting resource graph so the TS
verifier can check producer and consumer ordering.

Schedule phase terms follow TS schedule-builder naming. HEAD is the one-time
schedule before the repeated K/V tile loop, LOOP is the repeated K/V tile body,
and TAIL is the one-time cleanup and drain after LOOP exits.

Feature Support Matrix:

  | Feature          | Support                                                                                       |
  |------------------|-----------------------------------------------------------------------------------------------|
  | Workload         | Context (prefill)                                                                             |
  | Head-paired mode | Optional under GQA; auto-enabled for GQA with any positive left window                         |
  | Paged KV         | Page size 16/32/64/128; fp16/bf16/e4m3; scheduler mode is selected internally                   |
  | Variable seqlen  | Supported through flattened tensors + `cum_seqlen_*`                                          |
  | dtype            | fp16/bf16/e4m3 Q/K/V/O, fp32 QK and PV accumulation                                           |
  | Masking          | non-causal and causal; causal left sliding window uses head-paired mode                       |
  | head dimension   | D=128 and D=256                                                                                |
  | `S_q` / `S_kv`   | Query-paired causal requires `S_q <= S_kv`; arbitrary positive tails are supported             |
  | GQA              | Must satisfy `h_q % h_kv == 0`; causal GQA can use head-paired scheduling                    |
  | Sliding window   | `mask_type="causal", window_left=N`; left window only                                        |
  | Scheduler modes  | Query-paired: static CTAs, persistent, CLC; head-paired: persistent, CLC                      |

ASCII Flow Chart:

  Legend
    +----------+  TS resource
    --Task-->    task-owned action
    0/1          two peer resource instances

                    +----------+
                    | GmemQKV  |
                    | Q/K/V    |
                    +-+--+-----+
                      |  |
         +------------+  |
         |               |  LoadTask: TMA Q and TMA K/V
         v               v
   +-----+----+      +--------+      +-----------------+
   | SmemQ    |      | SmemKV |      | TmemStatsDone0/1|
   +-----+----+      +---+----+      +-----+-----------+
         |               |                 |
         +---------------+-----------------+
                         |
                         +---------------- KV loop -----------------+
                         |                                          |
                         | MmaTask BMM1: SmemQ + SmemKV.K -> S,     |
                         | waiting TmemStatsDone0/1 before S overwrites   |
                         | aliased stats columns.                   |
                         v                                          |
                   +-----+------+                                   |
                   | TmemSP0/1  | ---Softmax0/1Task: S -> P, stats--+--+
                   | S then P   |                                      |
                   +-----+------+                                      v
                         |                                       +--------------+
                         | MmaTask BMM2: wait P from the         | TmemStats0/1 |
                         | same TmemSP0/1 + SmemKV.V -> O        +--------------+
                         v                                             |
                   +-----+------+ <--- CorrectionTask: stats + O ------+
                   | TmemO      |      rescale O in-place           ^
                   +-----+------+                                   |
                         |                                          |
                         | CorrectionTask release frees TmemO       |
                         | before next BMM1 writes S in TmemSP.     |
                         +------------------------------------------+
                         |
                         | CorrectionTask tail
                         v
                   +-----+----+
                   | SmemO0/1 |
                   +-----+----+
                         |
                         | EpilogueTask: TMA store
                         v
                   +-----+----+
                   | GmemO0/1 |
                   +----------+

In the illustrated D128 schedule, P readiness is the TmemSP0/1 pipeline state:
Softmax0/1Task stores P and releases the S/P slot, then MmaTask re-acquires the
same slot for BMM2. Staged D256 instead has a separate TmemPResource handoff so
next-tile QK can overlap previous-tile PV. CorrectionTask releases each TmemO
stage after rescaling it.

TmemStats0/1 in the diagram are TmemStatsResource instances. They store the
correction statistics old_row_max, row_max, row_sum, and pad.

The optional WorkQueue is scheduler state, not Q/K/V/O dataflow. Persistent
tasks consume it to select work tiles; static hardware scheduling omits it.

Default D128 tasks and warp ownership (D256 uses the staged 12-warp layout
described by ``FmhaConfig``):
  Softmax0Task   warps 0-3    softmax on S0 -> P0, produce stats0
  Softmax1Task   warps 4-7    softmax on S1 -> P1, produce stats1
  CorrectionTask warps 8-11   rescale O using correction stats
  MmaTask        warp 12      Q*K and P*V tcgen05 MMA
  LoadTask       warp 13      TMA loads Q/K/V
  EpilogueTask   warp 14      TMA stores O
  PaddingTask    warp 15      warp-group register participation

Entry points:
  - build_fmha_task_manager() -- constructs and validates the runtime task graph
  - fmha_ts_kernel()         -- @cute.kernel, the GPU kernel
  - FmhaTs                   -- class with @cute.jit __call__ and kernel
"""

import warnings
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Tuple

import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cuda.bindings import driver as cuda_drv
from cutlass import Int32
from ..tensor_map import create_tensor_map_ragged_from_tensor

from cutlass.experimental.task_scheduling.memory import (
    SmemAllocation,
    SmemAllocator,
    TmemAllocator,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    TileSchedulerConfig,
    WorkQueue,
)
from cutlass.experimental.task_scheduling.task import Task
from cutlass.experimental.task_scheduling.task_manager import TaskManager

from .fmha_resources import (
    _SUPPORTED_CONTEXT_PAGE_SIZES,
    FmhaConfig,
    GmemOResource,
    GmemQKVResource,
    S0S1SequenceResource,
    SmemKVResource,
    SmemOResource,
    SmemPageOffsetsKvResource,
    SmemQResource,
    TmemOResource,
    TmemPResource,
    TmemSPResource,
    TmemStatsResource,
    TmemStatsDoneResource,
)
from .fmha_tasks import (
    PackedContextWorkQueue,
    create_correction_task,
    create_epilogue_task,
    create_load_task,
    create_mma_task,
    create_padding_task,
    create_page_offsets_task,
    create_scheduler_task,
    create_softmax_task,
)


from .helpers import (
    bottom_right_window_max_tiles,
    bottom_right_window_tile_start,
)
from cutlass.experimental import primitives as prims


def _as_i32(x: int | Int32) -> Int32:
    return Int32(x) if isinstance(x, int) else x


def _domain_min(a: int | Int32, b: int | Int32) -> int | Int32:
    """``min`` that stays a Python ``int`` for static inputs and defers to
    ``cute.math.min`` otherwise."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return cute.math.min(_as_i32(a), _as_i32(b))


def _domain_max(a: int | Int32, b: int | Int32) -> int | Int32:
    """``max`` counterpart of :func:`_domain_min`."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return cute.math.max(_as_i32(a), _as_i32(b))


def _init_task_with_domain(
    task: Task,
    kwargs: dict[str, Any],
    domain: int | None,
    task_init: Callable[..., None] = Task.__init__,
) -> None:
    """Initialize a captured Task with its static validation loop domain."""
    schedule = kwargs.get("schedule")
    if schedule is None:
        raise ValueError("Causal domain tasks require a captured schedule.")
    if domain is not None:
        schedule.loop_end = domain
    task_init(task, **kwargs)


def _init_causal_domain_state(
    task: Task,
    *,
    num_kv_tiles: int | Int32,
    cta_m: int | Int32,
    kv_n: int | Int32,
    q_offset: int | Int32,
    seq_idx: int,
    batch_idx: int | None,
    cum_seqlen_q: cute.Tensor | None,
    cum_seqlen_k: cute.Tensor | None,
    runtime_kv_tile_multiple: int,
    reverse_seq_tiles: int | Int32 | None,
    offset: int,
    window_size_left: int,
    packed_window: bool,
    kwargs: dict[str, Any],
    task_init: Callable[..., None] = Task.__init__,
) -> None:
    """Initialize causal-domain fields and the static validation domain."""
    static_domain = None
    if isinstance(num_kv_tiles, int):
        assert isinstance(cta_m, int) and isinstance(kv_n, int)
        if packed_window:
            static_domain = min(
                num_kv_tiles,
                bottom_right_window_max_tiles(
                    q_tile_m=cta_m,
                    kv_tile_n=kv_n,
                    window_size_left=window_size_left,
                ),
            )
        else:
            seq_coord = 0
            max_q_row = q_offset + seq_coord * cta_m + cta_m - 1
            causal_n = max_q_row // kv_n + 1
            static_domain = max(cta_m // kv_n, min(num_kv_tiles, causal_n))
        if window_size_left > 0 and not packed_window:
            static_domain -= bottom_right_window_tile_start(
                seq_coord=0,
                q_tile_m=cta_m,
                kv_tile_n=kv_n,
                q_offset=q_offset,
                window_size_left=window_size_left,
            )
        if offset:
            static_domain -= offset
        static_domain = max(static_domain, 0)
    _init_task_with_domain(task, kwargs, static_domain, task_init=task_init)
    task._num_kv_tiles = num_kv_tiles
    # M dimension size for Q*K.
    task._cta_m = cta_m
    # N dimension size for Q*K.
    task._kv_n = kv_n
    task._q_offset = q_offset
    # Index of the sequence coordinate in tile_coord.
    task._seq_idx = seq_idx
    # Mixed packed causal launches derive a request-local safe K-loop extent
    # from live metadata. Fixed and uniform-packed launches leave these fields
    # unset and retain the static plan-time domain calculation above.
    task._batch_idx = batch_idx
    task._cum_seqlen_q = cum_seqlen_q
    task._cum_seqlen_k = cum_seqlen_k
    task._runtime_kv_tile_multiple = runtime_kv_tile_multiple
    task._reverse_seq_tiles = reverse_seq_tiles
    # Offset adjusts the domain count.
    task._offset = offset
    task._window_size_left = window_size_left
    task._packed_window = packed_window


class CausalDomainTask(Task):
    """Task with causal and optional left-window masking domain.

    For causal FMHA, each Q tile only attends to K tiles where k <= q.
    The domain (K-loop iteration count) varies per tile based on the
    tile's sequence position.  Head-paired sliding-window schedules reuse
    the same domain calculation after subtracting skipped left-window KV
    tiles.
    """

    def __init__(
        self,
        num_kv_tiles: int | Int32,
        cta_m: int | Int32,
        kv_n: int | Int32,
        q_offset: int | Int32 = 0,
        seq_idx: int = 0,
        batch_idx: int | None = None,
        cum_seqlen_q: cute.Tensor | None = None,
        cum_seqlen_k: cute.Tensor | None = None,
        runtime_kv_tile_multiple: int = 1,
        reverse_seq_tiles: int | Int32 | None = None,
        offset: int = 0,
        window_size_left: int = 0,
        packed_window: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize causal-domain parameters.

        Args:
            num_kv_tiles: Total number of K/V tiles in the sequence.
            cta_m: Number of Q rows covered by one CTA tile.
            kv_n: Number of K/V rows covered by one K-loop iteration.
            q_offset: Causal row-index shift for S_q < S_kv.
            seq_idx: Index of the sequence coordinate in ``tile_coord``.
            batch_idx: Index of the request coordinate in ``tile_coord`` for
                packed causal launches.
            cum_seqlen_q: Live packed-Q cumulative offsets. Together with
                ``cum_seqlen_k``, selects a request-local causal domain.
            cum_seqlen_k: Live packed-K/V cumulative offsets. Paged context
                passes its plan-time logical-K snapshot through this tensor.
            runtime_kv_tile_multiple: Round a request-local K/V tile count up
                to this multiple. Query-paired zero-offset causal scheduling
                uses two to retain its synthetic peer-0 tail slot.
            reverse_seq_tiles: Number of Q sequence work tiles when causal
                balancing reverses launch order. ``None`` keeps natural order.
            offset: Loop-count decrement after the causal/window tile count is
                computed. Use 0 for N, 1 for N-1, and 2 for N-2 domains.
            window_size_left: Left sliding-window size in tokens. Zero disables
                left-window domain trimming.
            packed_window: Use an offset-independent maximum window span for a
                packed-ragged batch; runtime loads still use each request's offset.
            **kwargs: Remaining ``Task`` constructor arguments, including an
                optional captured ``schedule``.
        """
        _init_causal_domain_state(
            self,
            num_kv_tiles=num_kv_tiles,
            cta_m=cta_m,
            kv_n=kv_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            reverse_seq_tiles=reverse_seq_tiles,
            offset=offset,
            window_size_left=window_size_left,
            packed_window=packed_window,
            kwargs=kwargs,
        )

    def get_domain(self, tile_coord: cute.Coord) -> int | Int32:
        """Return the per-work-tile K-loop iteration count.

        Folds to a Python ``int`` when the tile coordinate and shape fields are
        static, and defers to ``cute.math`` for runtime coordinates.
        """
        seq_coord = tile_coord[self._seq_idx]
        if cutlass.const_expr(self._reverse_seq_tiles is not None):
            seq_coord = self._reverse_seq_tiles - seq_coord - 1
        q_offset = self._q_offset
        num_kv_tiles = self._num_kv_tiles
        runtime_q_tile_active = None
        if cutlass.const_expr(self._cum_seqlen_q is not None):
            assert self._cum_seqlen_k is not None
            assert self._batch_idx is not None
            batch_coord = Int32(tile_coord[self._batch_idx])
            q_begin = Int32(self._cum_seqlen_q[batch_coord])
            k_begin = Int32(self._cum_seqlen_k[batch_coord])
            seqlen_q = Int32(self._cum_seqlen_q[batch_coord + Int32(1)]) - q_begin
            seqlen_k = Int32(self._cum_seqlen_k[batch_coord + Int32(1)]) - k_begin
            runtime_q_tile_active = Int32(seq_coord * self._cta_m < seqlen_q)
            q_offset = seqlen_k - seqlen_q
            num_kv_tiles = cute.ceil_div(seqlen_k, self._kv_n)
            if cutlass.const_expr(self._runtime_kv_tile_multiple > 1):
                num_kv_tiles = (
                    cute.ceil_div(num_kv_tiles, self._runtime_kv_tile_multiple)
                    * self._runtime_kv_tile_multiple
                )
        if self._packed_window:
            max_window_tiles = bottom_right_window_max_tiles(
                q_tile_m=self._cta_m,
                kv_tile_n=self._kv_n,
                window_size_left=self._window_size_left,
            )
            result = _domain_min(num_kv_tiles, max_window_tiles)
        else:
            max_q_row = q_offset + seq_coord * self._cta_m + self._cta_m - 1
            causal_n = max_q_row // self._kv_n + 1
            # Softmax assumes there is at least one Q-tile-width of K work.
            result = _domain_max(
                self._cta_m // self._kv_n,
                _domain_min(num_kv_tiles, causal_n),
            )
        if self._window_size_left > 0 and not self._packed_window:
            result -= bottom_right_window_tile_start(
                seq_coord=seq_coord,
                q_tile_m=self._cta_m,
                kv_tile_n=self._kv_n,
                q_offset=q_offset,
                window_size_left=self._window_size_left,
            )
        if self._offset:
            result = result - self._offset
        if cutlass.const_expr(runtime_q_tile_active is not None):
            # The outer grid is sized by the planned maximum Q length, so a
            # mixed packed request can have trailing work-tile slots with no Q
            # rows. Retain the smallest legal N/N-1/N-2 pipeline domains for
            # those slots instead of traversing K/V according to a large
            # bottom-right offset. Resource-level ragged extents suppress the
            # dummy Q/O traffic; the minimum domains preserve task handoffs.
            minimum_domain = max(self._cta_m // self._kv_n - self._offset, 0)
            result = (
                runtime_q_tile_active * result
                + (Int32(1) - runtime_q_tile_active) * minimum_domain
            )
        return result


class CausalSoftmaxDomainTask(CausalDomainTask):
    """Softmax task with causal domain selection."""

    def __init__(
        self,
        num_kv_tiles: int | Int32,
        cta_m: int | Int32,
        kv_n: int | Int32,
        q_offset: int | Int32 = 0,
        seq_idx: int = 0,
        batch_idx: int | None = None,
        cum_seqlen_q: cute.Tensor | None = None,
        cum_seqlen_k: cute.Tensor | None = None,
        runtime_kv_tile_multiple: int = 1,
        reverse_seq_tiles: int | Int32 | None = None,
        offset: int = 0,
        window_size_left: int = 0,
        packed_window: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize softmax causal-domain parameters.

        Parameters
        ----------
        num_kv_tiles : int or Int32
            Total number of K/V tiles in the sequence.
        cta_m : int or Int32
            Number of Q rows covered by one CTA tile.
        kv_n : int or Int32
            Number of K/V rows covered by one K-loop iteration.
        q_offset : int or Int32
            Causal row-index shift for S_q < S_kv.
        seq_idx : int
            Index of the sequence coordinate in ``tile_coord``.
        batch_idx : int or None
            Index of the request coordinate for packed causal launches.
        cum_seqlen_q, cum_seqlen_k : cute.Tensor or None
            Live cumulative offsets used to derive the request-local causal
            shift and K/V tile count.
        runtime_kv_tile_multiple : int
            Request-local K/V tile-count alignment for paired-tail scheduling.
        reverse_seq_tiles : int or Int32 or None
            Number of Q sequence work tiles for reversed causal-balanced order.
        offset : int
            Loop-count decrement after the causal/window tile count is
            computed.
        window_size_left : int
            Left sliding-window size in tokens. Zero disables left-window
            domain trimming.
        packed_window : bool
            Whether to use an offset-independent packed-ragged window span.
        **kwargs : Any
            Remaining ``Task`` constructor arguments.
        """
        _init_causal_domain_state(
            self,
            num_kv_tiles=num_kv_tiles,
            cta_m=cta_m,
            kv_n=kv_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            reverse_seq_tiles=reverse_seq_tiles,
            offset=offset,
            window_size_left=window_size_left,
            packed_window=packed_window,
            kwargs=kwargs,
        )


DomainPolicyValue = int | bool | Int32 | type[Task] | cute.Tensor
DomainKwargs = dict[str, DomainPolicyValue]


def resolve_head_paired_mode(
    *,
    head_paired: bool,
    is_causal: bool,
    window_size_left: int,
) -> bool:
    """Return the effective head-paired mode for a mask/window configuration."""
    if window_size_left < 0:
        raise ValueError("window_size_left must be non-negative")
    if window_size_left > 0 and not is_causal:
        raise ValueError("window_size_left requires is_causal=True")
    if window_size_left > 0 and not head_paired:
        warnings.warn(
            f"window_size_left={window_size_left} requires head_paired=True; "
            "got head_paired=False, enabling head_paired",
            UserWarning,
            stacklevel=2,
        )
    return head_paired or window_size_left > 0


def validate_head_paired_head_ratio(*, head_paired: bool, h_r: int) -> None:
    """Validate the Q/KV head ratio required by head-paired scheduling."""
    if not head_paired:
        return
    if h_r <= 1:
        raise ValueError(
            f"head_paired requires grouped-query attention with h_q > h_kv, got {h_r=}"
        )
    if h_r % 2 != 0:
        raise ValueError(
            f"head_paired requires an even Q/KV head ratio, got h_q / h_kv = {h_r}"
        )


@dataclass(frozen=True)
class FmhaDomainPolicy:
    """Domain and mask policy selected for one FMHA launch flavor."""

    domain_n_kwargs: DomainKwargs
    domain_n_minus_1_kwargs: DomainKwargs
    softmax0_domain_kwargs: DomainKwargs
    softmax1_domain_kwargs: DomainKwargs


def build_context_task_manager(
    *,
    cfg: FmhaConfig,
    tile_sched_params: (
        utils.PersistentTileSchedulerParams
        | utils.ClcDynamicPersistentTileSchedulerParams
        | None
    ),
    tma_q_desc: cutlass.Pointer | None,
    tma_k_desc: cutlass.Pointer | None,
    tma_v_desc: cutlass.Pointer | None,
    tma_o_desc: cutlass.Pointer | None,
    cum_seqlen_q: cute.Tensor | None,
    cum_seqlen_k: cute.Tensor | None,
    scale_softmax_log2: cute.Tensor | None = None,
    output_scale: cute.Tensor | None = None,
    g_page_idx_kv: cute.Pointer | None = None,
    g_seq_lens_kv: cute.Pointer | None = None,
    max_seq_len_kv: int | Int32 | None = None,
    num_kv_tiles: int | Int32,
    q_offset: int | Int32,
    domain_n_kwargs: DomainKwargs,
    domain_n_minus_1_kwargs: DomainKwargs,
    softmax0_domain_kwargs: DomainKwargs,
    softmax1_domain_kwargs: DomainKwargs,
    tmem_o_extra_kwargs: DomainKwargs | None = None,
    is_persistent: bool = True,
    is_clc_dynamic: bool = False,
    clc_response_ptr: cute.Pointer | None = None,
    exhaustive_deadlock_race_check: bool = True,
    debug_print: bool = False,
) -> Tuple[
    TaskManager,
    list[MemoryResource],
    SmemAllocation,
    SmemAllocation,
    WorkQueue | None,
    SmemAllocation | None,
]:
    """Build the FMHA TaskManager with all resources, tasks, and dependency graph.

    ``scale_softmax_log2`` and ``output_scale`` are optional one-element
    float32 device tensors. Resource setup loads element 0 once before the K/V
    loop and reuses the value through task-local dataflow. ``scale_softmax_log2``
    is the base-2 softmax multiplier, normally
    ``softmax_scale * log2(e)``; FP8 callers can fold Q/K dequant scales into
    it. ``output_scale`` multiplies the final O store; FP8 output can fold the
    V dequant scale and output quant scale into it.

    SMEM buffers are declared via ``SmemAllocation`` and bound from
    ``ResourceContext`` by auxiliary resource init work. The ``SmemAllocator``
    is passed to ``TaskManager`` for unified allocation. Infrastructure slots
    (``tmem_ptr_i32``, ``tmem_dealloc_mbar``) are included in the unified SMEM
    block and returned as allocation descriptors.

    Parameters
    ----------
    cfg : FmhaConfig
        Kernel-wide configuration.
    tile_sched_params : PersistentTileSchedulerParams or ClcDynamicPersistentTileSchedulerParams, optional
        Persistent tile scheduler params. ``None`` for static or validation-only builds.
    tma_q_desc, tma_k_desc, tma_v_desc, tma_o_desc : cutlass.Pointer, optional
        TMA descriptor pointers for Q, K, V, O.
    cum_seqlen_q, cum_seqlen_k : cute.Tensor, optional
        Cumulative sequence-length metadata for varlen launches.
    num_kv_tiles : int or Int32
        Number of KV tiles in the loop domain.
    q_offset : int or Int32
        Default right shift of Q rows for causal S_q < S_kv masking. Mixed
        packed launches derive the request-local shift from live metadata.
    domain_n_kwargs : dict
        Domain policy for tasks that process the full KV loop.
    domain_n_minus_1_kwargs : dict
        Domain policy for tasks whose HEAD handles the first KV tile and whose
        LOOP handles the remaining tiles.
    softmax0_domain_kwargs, softmax1_domain_kwargs : dict
        Domain policies for the two softmax peers.
    tmem_o_extra_kwargs : dict, optional
        Extra domain/resource policy knobs for TMEM O.
    is_persistent : bool
        Whether tasks use WorkQueue-backed persistent scheduling.
    is_clc_dynamic : bool
        Whether to use CLC dynamic persistent scheduling.
    clc_response_ptr : cute.Pointer, optional
        SMEM pointer for the CLC response buffer, required when
        ``is_clc_dynamic`` is true.
    exhaustive_deadlock_race_check : bool
        Whether TaskManager runs the exhaustive schedule checker during
        construction.
    debug_print : bool
        Whether to enable debug prints in generated tasks.

    Returns
    -------
    tuple
        ``(TaskManager, tmem_resources, tmem_ptr_alloc, dealloc_mbar_alloc,
        work_queue, clc_response_alloc)``.
    """
    if cfg.use_paged_kv:
        if cfg.num_tokens_per_page not in _SUPPORTED_CONTEXT_PAGE_SIZES:
            raise ValueError(
                "paged context requires num_tokens_per_page in "
                f"{_SUPPORTED_CONTEXT_PAGE_SIZES}; got "
                f"{cfg.num_tokens_per_page}"
            )
    if cfg.causal_single_kv_tile and (cfg.use_paged_kv or cfg.has_varlen):
        raise ValueError("causal_single_kv_tile requires fixed contiguous K/V storage")
    # ---------------------------------------------------------------------------
    # Cluster / CTA layout
    # ---------------------------------------------------------------------------
    # CTA layout in CuTe VMNK order. V is the CTA-group dimension used for
    # 2-CTA cooperative MMA; MNK are the logical cluster axes. FMHA uses a
    # single-CTA cluster here, so (V, M, N, K) = (1, 1, 1, 1).
    cluster_shape_vmnk = (1, 1, 1, 1)

    # ---------------------------------------------------------------------------
    # Warp counts
    # ---------------------------------------------------------------------------
    # Warp ownership: 4 softmax, 4 correction, 1 MMA, 1 load, 1 epilogue, and
    # one padding/scheduler warp-group participant.
    num_mma_warps = 1
    num_softmax_warps = 4
    num_correction_warps = 4
    num_epilogue_warps = 1

    # ---------------------------------------------------------------------------
    # Cooperative groups
    # ---------------------------------------------------------------------------
    warp_size = cute.arch.WARP_SIZE
    Agent = pipeline.Agent

    # For TMA pipelines: elect_one arrival, producer_group = 1 thread.
    tma_producer_group = pipeline.CooperativeGroup(Agent.Thread)

    # ---------------------------------------------------------------------------
    # Pipeline configs
    # ---------------------------------------------------------------------------

    # SmemQ: TmaUmma, topology-derived stages, Load -> MMA. The paired D128
    # schedule uses two Q instances; staged D256 uses one.
    # advance_on_wait=True advances consumer_state on ConsumerWait.
    smem_q_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.q_stage,
        num_bytes=cfg.tma_copy_q_bytes,
        producer_group=tma_producer_group,
        consumer_group=pipeline.CooperativeGroup(Agent.Thread),
        cta_layout_vmnk=cluster_shape_vmnk,
        advance_on_wait=True,
    )
    # SmemKV: capacity-derived stages, Load -> MMA.
    # advance_on_wait=True advances consumer_state at ConsumerWait rather than
    # ConsumerRelease. This keeps the previous V tile live while MMA starts the
    # next QK tile, giving QK0 -> PV1(previous V) -> QK1 ordering without
    # releasing the previous V tile first.
    smem_kv_pipeline_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.kv_stage,
        num_bytes=cfg.tma_copy_kv_bytes,
        producer_group=tma_producer_group,
        consumer_group=pipeline.CooperativeGroup(Agent.Thread),
        cta_layout_vmnk=cluster_shape_vmnk,
        advance_on_wait=True,
    )
    # Page-offsets prefetch (paged-KV only): the auxiliary warp produces and
    # the load warp consumes.
    # Async-async because both are CUDA-thread groups (no TMA arrival barrier).
    smem_page_offsets_pipeline_cfg = None
    smem_page_offsets_v_pipeline_cfg = None
    split_page_offset_pipelines = False
    if cfg.stages_page_offsets_in_smem:
        # A staged single-instance schedule keeps independent K and V page-ID
        # rings because its K-ahead/V-delayed lifetimes overlap.  Split the
        # ordinary page-offset stage budget across those two rings, just as
        # FMHA decode does for its split-head-dimension page-offset flow.  V
        # needs one fewer credit because the boundary tile's IDs are retained
        # in registers before its SMEM window is released.  The shared K/V
        # ring used by every other topology retains the full depth.
        page_offset_stage_counts = cfg.page_offset_pipeline_stage_counts
        page_offsets_k_stages = page_offset_stage_counts[0]
        page_offsets_v_stages = None
        if len(page_offset_stage_counts) == 2:
            split_page_offset_pipelines = True
            page_offsets_v_stages = page_offset_stage_counts[1]

        # The load warp (1 warp = 32 threads) is the only consumer; signal-All
        # requires the consumer group to size match num_warps × warp_size.
        def make_page_offsets_pipeline_cfg(num_stages: int) -> PipelineConfig:
            return PipelineConfig.create_async_async_pipeline_cfg(
                num_stages=num_stages,
                producer_group=pipeline.CooperativeGroup(
                    Agent.Thread,
                    cfg.page_offsets_num_warps * warp_size,
                ),
                consumer_group=pipeline.CooperativeGroup(
                    Agent.Thread,
                    warp_size,
                ),
                cta_layout_vmnk=cluster_shape_vmnk,
                producer_op=pipeline.PipelineOp.AsyncLoad,
            )

        smem_page_offsets_pipeline_cfg = make_page_offsets_pipeline_cfg(
            page_offsets_k_stages
        )
        if page_offsets_v_stages is not None:
            smem_page_offsets_v_pipeline_cfg = make_page_offsets_pipeline_cfg(
                page_offsets_v_stages
            )

    softmax_group = pipeline.CooperativeGroup(
        Agent.Thread,
        num_softmax_warps * warp_size,
    )
    correction_group = pipeline.CooperativeGroup(
        Agent.Thread,
        num_correction_warps * warp_size,
    )
    epilogue_group = pipeline.CooperativeGroup(
        Agent.Thread,
        num_epilogue_warps * warp_size,
    )
    umma_hw_group = pipeline.CooperativeGroup(Agent.Thread)
    mma_group = pipeline.CooperativeGroup(Agent.Thread, num_mma_warps * warp_size)

    tmem_sp0_pipeline_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.mma_softmax_stage,
        producer_group=umma_hw_group,
        consumer_group=softmax_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    tmem_sp1_pipeline_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.mma_softmax_stage,
        producer_group=umma_hw_group,
        consumer_group=softmax_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    tmem_p0_pipeline_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
        num_stages=cfg.mma_softmax_stage,
        producer_group=softmax_group,
        consumer_group=umma_hw_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    tmem_o_pipeline_cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=cfg.mma_corr_stage,
        producer_group=umma_hw_group,
        consumer_group=correction_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )

    tmem_vec0_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=cfg.softmax_corr_stage,
        producer_group=softmax_group,
        consumer_group=correction_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    tmem_vec1_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=cfg.softmax_corr_stage,
        producer_group=softmax_group,
        consumer_group=correction_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    smem_o_0_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=1,
        producer_group=correction_group,
        consumer_group=(
            correction_group if cfg.fuse_epilogue_into_correction else epilogue_group
        ),
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    smem_o_1_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=1,
        producer_group=correction_group,
        consumer_group=epilogue_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    # S0-S1 sequence barrier: PipelineAsync, 1 stage.
    # Ensures Softmax0 finishes P store before Softmax1 starts P compute.
    # Both groups have 4 warps (128 threads), but they are different warps.
    s0s1_seq_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=1,
        producer_group=softmax_group,
        consumer_group=softmax_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    # TmemStatsDone barriers: MMA acquires before QK->S that aliases stats TMEM
    # columns, and Correction releases after reading stats from TMEM. The
    # pipeline starts empty, so MMA's first ProducerAcquire succeeds without
    # priming. Separate barriers let MMA start QK->S0 as soon as TmemStats0 is
    # read, without waiting for TmemStats1.
    tmem_stats_done_0_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=1,
        producer_group=mma_group,
        consumer_group=correction_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )
    tmem_stats_done_1_pipeline_cfg = PipelineConfig.create_async_async_pipeline_cfg(
        num_stages=1,
        producer_group=mma_group,
        consumer_group=correction_group,
        cta_layout_vmnk=cluster_shape_vmnk,
    )

    # ---------------------------------------------------------------------------
    # Create resource instances
    # ---------------------------------------------------------------------------
    gmem_qkv = GmemQKVResource(
        tma_q_desc=tma_q_desc,
        tma_k_desc=tma_k_desc,
        tma_v_desc=tma_v_desc,
        cum_seqlen_q=cum_seqlen_q,
        cum_seqlen_k=cum_seqlen_k,
        q_offset=q_offset,
        cfg=cfg,
        seqlens_kv=g_seq_lens_kv,
        max_seq_len_kv=max_seq_len_kv,
        name="gmem_qkv",
    )
    smem_q = SmemQResource(
        tma_q_desc=tma_q_desc,
        pipeline_config=smem_q_pipeline_cfg,
        cfg=cfg,
        name="smem_q",
    )
    smem_page_offsets_kv: SmemPageOffsetsKvResource | None = None
    smem_page_offsets_v: SmemPageOffsetsKvResource | None = None
    if cfg.stages_page_offsets_in_smem:
        smem_page_offsets_kv = SmemPageOffsetsKvResource(
            page_idx_kv=g_page_idx_kv,
            pipeline_config=smem_page_offsets_pipeline_cfg,
            cfg=cfg,
            name="smem_page_offsets_kv",
        )
        if split_page_offset_pipelines:
            # The D256 schedule keeps independent K/V page-window
            # stages because its K-ahead/V-delayed pipeline crosses window
            # boundaries.  D128 consumes K and V together and FlashInfer's
            # public paged API supplies one shared page-ID row, so it retains
            # one stage for both sides.
            smem_page_offsets_v = SmemPageOffsetsKvResource(
                page_idx_kv=g_page_idx_kv,
                pipeline_config=smem_page_offsets_v_pipeline_cfg,
                cfg=cfg,
                page_table_is_v=True,
                name="smem_page_offsets_v",
            )
    smem_kv = SmemKVResource(
        tma_k_desc=tma_k_desc,
        tma_v_desc=tma_v_desc,
        pipeline_config=smem_kv_pipeline_cfg,
        cfg=cfg,
        page_offsets_kv=smem_page_offsets_kv,
        page_offsets_v=smem_page_offsets_v,
        page_idx_kv=g_page_idx_kv,
        name="smem_kv",
    )

    # WorkQueue: persistent tile scheduler state (static or CLC dynamic), not
    # Q/K/V/O dataflow. Non-persistent launches omit it so each CTA executes
    # one tile directly from its hardware tile coordinate.
    work_queue: WorkQueue | None = None
    if not is_persistent and tile_sched_params is not None:
        raise ValueError(
            "non-persistent FMHA task managers must not receive tile_sched_params"
        )
    if is_clc_dynamic:
        if not is_persistent:
            raise ValueError("CLC dynamic scheduling requires persistent mode")
        # CLC dynamic: pipeline + CLC tile scheduler config. FMHA uses
        # single-CTA clusters.
        cluster_size = 1
        num_consumer_threads = cfg.block_warps * warp_size * cluster_size
        work_queue_pipeline_cfg = PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=1,
            num_bytes=16,
            producer_group=pipeline.CooperativeGroup(Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                Agent.Thread,
                num_consumer_threads,
            ),
            cta_layout_vmnk=cluster_shape_vmnk,
        )
        tile_scheduler_config = (
            TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=clc_response_ptr,
            )
        )
        work_queue_kwargs = {
            "tile_scheduler_config": tile_scheduler_config,
            "pipeline_config": work_queue_pipeline_cfg,
            "name": "work_queue",
        }
        if (
            cfg.is_causal
            and cfg.has_varlen
            and not cfg.has_uniform_varlen
            and cum_seqlen_q is not None
        ):
            work_queue = PackedContextWorkQueue(
                cfg=cfg,
                cum_seqlen_q=cum_seqlen_q,
                **work_queue_kwargs,
            )
        else:
            work_queue = WorkQueue(**work_queue_kwargs)
    elif is_persistent:
        tile_scheduler_config = (
            TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
            )
        )
        work_queue_kwargs = {
            "tile_scheduler_config": tile_scheduler_config,
            "name": "work_queue",
        }
        if (
            cfg.is_causal
            and cfg.has_varlen
            and not cfg.has_uniform_varlen
            and cum_seqlen_q is not None
        ):
            work_queue = PackedContextWorkQueue(
                cfg=cfg,
                cum_seqlen_q=cum_seqlen_q,
                **work_queue_kwargs,
            )
        else:
            work_queue = WorkQueue(**work_queue_kwargs)

    tmem_sp0 = TmemSPResource(
        pipeline_config=tmem_sp0_pipeline_cfg,
        cfg=cfg,
        tmem_s_offset=cfg.tmem_s0_offset,
        tmem_p_offset=cfg.tmem_p0_offset,
        q_half=0,
        q_offset=q_offset,
        cum_seqlen_q=cum_seqlen_q,
        cum_seqlen_k=cum_seqlen_k,
        scale_softmax_log2=scale_softmax_log2,
        name="tmem_sp0",
    )
    tmem_p0: TmemPResource | None = None
    if cfg.has_tmem_p_pipeline:
        tmem_p0 = TmemPResource(
            pipeline_config=tmem_p0_pipeline_cfg,
            cfg=cfg,
            tmem_p_offset=cfg.tmem_p0_offset,
            name="tmem_p0",
        )
    tmem_vec0 = TmemStatsResource(
        pipeline_config=tmem_vec0_pipeline_cfg,
        cfg=cfg,
        tmem_vec_offset=cfg.tmem_vec0_offset,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        name="tmem_vec0",
    )
    smem_o_0 = SmemOResource(
        pipeline_config=smem_o_0_pipeline_cfg,
        cfg=cfg,
        stage_idx=0,
        tmem_vec_resource=tmem_vec0,
        name="smem_o_0",
    )
    gmem_o_0 = GmemOResource(
        tma_o_desc=tma_o_desc,
        cum_seqlen_q=cum_seqlen_q,
        cfg=cfg,
        stage_idx=0,
        name="gmem_o_0",
    )
    # TmemStatsDone barrier shared between MMA
    # (producer) and Correction (consumer). Prevents cross-tile aliasing races.
    tmem_stats_done_0 = TmemStatsDoneResource(
        pipeline_config=tmem_stats_done_0_pipeline_cfg,
        name="tmem_stats_done_0",
    )

    single_qkv_instance = cfg.single_qkv_instance
    tmem_sp1: TmemSPResource | None = None
    tmem_vec1: TmemStatsResource | None = None
    smem_o_1: SmemOResource | None = None
    gmem_o_1: GmemOResource | None = None
    s0s1_seq: S0S1SequenceResource | None = None
    tmem_stats_done_1: TmemStatsDoneResource | None = None

    if not single_qkv_instance:
        tmem_sp1 = TmemSPResource(
            pipeline_config=tmem_sp1_pipeline_cfg,
            cfg=cfg,
            tmem_s_offset=cfg.tmem_s1_offset,
            tmem_p_offset=cfg.tmem_p1_offset,
            q_half=1,
            q_offset=q_offset,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            scale_softmax_log2=scale_softmax_log2,
            name="tmem_sp1",
        )
        tmem_vec1 = TmemStatsResource(
            pipeline_config=tmem_vec1_pipeline_cfg,
            cfg=cfg,
            tmem_vec_offset=cfg.tmem_vec1_offset,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            name="tmem_vec1",
        )
        smem_o_1 = SmemOResource(
            pipeline_config=smem_o_1_pipeline_cfg,
            cfg=cfg,
            stage_idx=1,
            tmem_vec_resource=tmem_vec1,
            name="smem_o_1",
        )
        gmem_o_1 = GmemOResource(
            tma_o_desc=tma_o_desc,
            cum_seqlen_q=cum_seqlen_q,
            cfg=cfg,
            stage_idx=1,
            name="gmem_o_1",
        )
        # S0-S1 sequence barrier: one sequencing resource shared between Softmax0
        # (producer) and Softmax1 (consumer), like other shared resources.
        s0s1_seq = S0S1SequenceResource(
            pipeline_config=s0s1_seq_pipeline_cfg,
            name="s0s1_seq",
        )
        tmem_stats_done_1 = TmemStatsDoneResource(
            pipeline_config=tmem_stats_done_1_pipeline_cfg,
            name="tmem_stats_done_1",
        )

    tmem_o_kwargs = tmem_o_extra_kwargs or {}
    tmem_o = TmemOResource(
        pipeline_config=tmem_o_pipeline_cfg,
        cfg=cfg,
        tmem_o0_offset=cfg.tmem_o0_offset,
        tmem_o1_offset=cfg.tmem_o1_offset,
        tmem_vec0_resource=tmem_vec0,
        tmem_vec1_resource=tmem_vec1,
        name="tmem_o",
        **tmem_o_kwargs,
    )

    # ---------------------------------------------------------------------------
    # Create tasks & dependency graph
    # ---------------------------------------------------------------------------
    # Per-task domains match the handwritten FMHA schedule's iteration counts:
    # - Softmax0/1: N iterations (one per KV tile)
    # - MMA/Load/Correction: N-1 iterations
    #   (HEAD handles first tile, LOOP handles remaining N-1)
    # - Epilogue/Padding: TAIL-only (domain does not matter, use N)
    #
    # With causal masking, the domain depends on the Q tile position: only
    # process K tiles where at least one Q row can attend (k <= q). The
    # CausalDomainTask subclass overrides get_domain(); seq_idx follows the
    # tile-coordinate order selected by the host launch.
    def scheduler_deps(*resources: MemoryResource) -> list[MemoryResource]:
        """Append the scheduler resource only for WorkQueue-backed launches."""
        deps = list(resources)
        if work_queue is not None:
            deps.append(work_queue)
        return deps

    mma_domain_kwargs = domain_n_minus_1_kwargs
    if single_qkv_instance and not cfg.has_tmem_p_pipeline:
        # The non-split single-instance schedule performs every QK/PV pair
        # inside its loop; it has no separate HEAD QK or TAIL PV iteration.
        mma_domain_kwargs = domain_n_kwargs
    load_task = create_load_task(
        gmem_qkv,
        smem_q,
        smem_kv,
        work_queue,
        smem_page_offsets_kv=smem_page_offsets_kv,
        smem_page_offsets_v=smem_page_offsets_v,
        debug_print=debug_print,
        **domain_n_kwargs,
    )
    mma_task = create_mma_task(
        smem_q,
        smem_kv,
        tmem_sp0,
        tmem_sp1,
        tmem_p0,
        tmem_o,
        tmem_stats_done_0,
        tmem_stats_done_1,
        work_queue,
        debug_print=debug_print,
        **mma_domain_kwargs,
    )

    # Causal query-paired moves masked/invalid K iterations from LOOP to TAIL
    # with no runtime branch. SP resources derive mask selection from cfg.
    softmax0_task = create_softmax_task(
        0,
        tmem_sp0,
        tmem_vec0,
        tmem_p0,
        s0s1_seq,
        work_queue,
        debug_print=debug_print,
        **softmax0_domain_kwargs,
    )
    softmax1_task: Task | None = None
    if not single_qkv_instance:
        if tmem_sp1 is None or tmem_vec1 is None:
            raise ValueError("paired softmax scheduling requires peer-1 resources")
        softmax1_task = create_softmax_task(
            1,
            tmem_sp1,
            tmem_vec1,
            None,
            s0s1_seq,
            work_queue,
            debug_print=debug_print,
            **softmax1_domain_kwargs,
        )
    correction_task = create_correction_task(
        tmem_vec0,
        tmem_vec1,
        tmem_o,
        smem_o_0,
        smem_o_1,
        gmem_o_0,
        gmem_o_1,
        tmem_stats_done_0,
        tmem_stats_done_1,
        work_queue,
        debug_print=debug_print,
        **domain_n_minus_1_kwargs,
    )
    epilogue_task: Task | None = None
    freed_epilogue_task: Task | None = None
    if cfg.fuse_epilogue_into_correction:
        if not is_clc_dynamic:
            # Warp 10 remains part of the producer/correction warpgroup and
            # must participate in setmaxnreg before it becomes a scheduler.
            freed_epilogue_task = create_padding_task(
                work_queue,
                warp_idx=cfg.epilogue_warp_id,
                num_registers=cfg.num_regs_other,
                name="EpiloguePaddingTask",
                **domain_n_kwargs,
            )
    else:
        epilogue_task = create_epilogue_task(
            smem_o_0,
            smem_o_1,
            gmem_o_0,
            gmem_o_1,
            work_queue,
            debug_print=debug_print,
            **domain_n_kwargs,
        )
    scheduler_task: Task | None = None
    if is_clc_dynamic:
        scheduler_task = create_scheduler_task(
            work_queue,
            warp_idx=(
                cfg.epilogue_warp_id
                if cfg.fuse_epilogue_into_correction
                else cfg.empty_warp_id
            ),
            num_registers=cfg.num_regs_other,
            domain=0,
        )
    if smem_page_offsets_kv is not None:
        # Paged-KV: the auxiliary warp prefetches page-table entries instead
        # of padding and still participates in setmaxnreg.sync.
        auxiliary_task = create_page_offsets_task(
            gmem_qkv,
            smem_page_offsets_kv,
            work_queue,
            num_registers=cfg.num_regs_other,
            smem_page_offsets_v=smem_page_offsets_v,
            **domain_n_kwargs,
        )
    elif scheduler_task is not None and not cfg.fuse_epilogue_into_correction:
        # D128 retains the original topology where the scheduler itself is the
        # final warp-group participant.
        auxiliary_task = scheduler_task
        scheduler_task = None
    else:
        # setmaxnreg.sync requires every warp in the final warp group to
        # participate, so the empty warp needs a padding/scheduler task.
        auxiliary_task = create_padding_task(
            work_queue,
            warp_idx=cfg.empty_warp_id,
            num_registers=cfg.num_regs_other,
            **domain_n_kwargs,
        )

    task_list = [softmax0_task]
    if softmax1_task is not None:
        task_list.append(softmax1_task)
    task_list.extend([correction_task, mma_task, load_task])
    if epilogue_task is not None:
        task_list.append(epilogue_task)
    if freed_epilogue_task is not None:
        task_list.append(freed_epilogue_task)
    if scheduler_task is not None:
        task_list.append(scheduler_task)
    task_list.append(auxiliary_task)

    if single_qkv_instance:
        tmem_o_source = tmem_p0 if tmem_p0 is not None else tmem_sp0
        tmem_o_dependencies = scheduler_deps(tmem_o_source)
    else:
        if (
            tmem_sp1 is None
            or tmem_vec1 is None
            or smem_o_1 is None
            or gmem_o_1 is None
            or s0s1_seq is None
            or tmem_stats_done_1 is None
        ):
            raise ValueError("paired resource graph requires peer-1 resources")
        tmem_o_dependencies = scheduler_deps(tmem_sp0, tmem_sp1)

    smem_kv_deps: list[MemoryResource] = [gmem_qkv]
    if smem_page_offsets_kv is not None:
        smem_kv_deps.append(smem_page_offsets_kv)
    if smem_page_offsets_v is not None:
        smem_kv_deps.append(smem_page_offsets_v)
    stats_done_0_deps = [] if cfg.stats_via_smem else [tmem_stats_done_0]
    resource_dependency_graph: dict[MemoryResource, list[MemoryResource]] = {
        smem_q: scheduler_deps(gmem_qkv),
        smem_kv: scheduler_deps(*smem_kv_deps),
        tmem_sp0: scheduler_deps(tmem_sp0, smem_q, smem_kv, *stats_done_0_deps),
        tmem_vec0: scheduler_deps(tmem_sp0),
        tmem_o: tmem_o_dependencies,
        smem_o_0: scheduler_deps(tmem_vec0, tmem_o),
        gmem_o_0: scheduler_deps(smem_o_0),
    }
    if not cfg.stats_via_smem:
        resource_dependency_graph[tmem_stats_done_0] = [tmem_vec0]
    if tmem_p0 is not None:
        resource_dependency_graph[tmem_p0] = scheduler_deps(tmem_sp0)
    if not single_qkv_instance:
        resource_dependency_graph.update(
            {
                tmem_sp1: scheduler_deps(
                    tmem_sp1,
                    smem_q,
                    smem_kv,
                    s0s1_seq,
                    *([] if cfg.stats_via_smem else [tmem_stats_done_1]),
                ),
                tmem_vec1: scheduler_deps(tmem_sp1),
                smem_o_1: scheduler_deps(tmem_vec1, tmem_o),
                gmem_o_1: scheduler_deps(smem_o_1),
                s0s1_seq: [tmem_sp0],
            }
        )
        if not cfg.stats_via_smem:
            resource_dependency_graph[tmem_stats_done_1] = [tmem_vec1]
    if work_queue is not None:
        resource_dependency_graph[work_queue] = [work_queue] if is_clc_dynamic else []
    if smem_page_offsets_kv is not None:
        resource_dependency_graph[smem_page_offsets_kv] = scheduler_deps(gmem_qkv)
    if smem_page_offsets_v is not None:
        resource_dependency_graph[smem_page_offsets_v] = scheduler_deps(gmem_qkv)

    smem_allocator = SmemAllocator()
    registered_smem_resource_ids: set[int] = set()

    def add_smem_resource(resource: MemoryResource | None) -> None:
        """Register one resource's data and barriers exactly once."""
        if resource is None or id(resource) in registered_smem_resource_ids:
            return
        smem_allocator.add_resource(resource)
        registered_smem_resource_ids.add(id(resource))

    add_smem_resource(smem_q)
    add_smem_resource(smem_kv)
    if smem_page_offsets_kv is not None:
        add_smem_resource(smem_page_offsets_kv)
    if smem_page_offsets_v is not None:
        add_smem_resource(smem_page_offsets_v)
    # Register every pipeline resource, including pipeline-only TMEM handoffs,
    # with the unified allocator.  Otherwise CUTLASS materializes those
    # barriers as separate dynamic-SMEM arrays that the capacity selector
    # cannot see.
    add_smem_resource(tmem_sp0)
    if tmem_p0 is not None:
        add_smem_resource(tmem_p0)
    add_smem_resource(tmem_vec0)
    add_smem_resource(tmem_o)
    if not cfg.stats_via_smem:
        add_smem_resource(tmem_stats_done_0)
    if tmem_sp1 is not None:
        add_smem_resource(tmem_sp1)
    if tmem_vec1 is not None:
        add_smem_resource(tmem_vec1)
    if s0s1_seq is not None:
        add_smem_resource(s0s1_seq)
    if tmem_stats_done_1 is not None and not cfg.stats_via_smem:
        add_smem_resource(tmem_stats_done_1)
    if work_queue is not None and work_queue.pipeline_config is not None:
        add_smem_resource(work_queue)
    if single_qkv_instance:
        add_smem_resource(smem_o_0)
        add_smem_resource(gmem_o_0)
        smem_allocator.add_alias_group(
            [
                [smem_o_0._alloc],
                [gmem_o_0._alloc],
            ]
        )
    else:
        add_smem_resource(smem_o_0)
        add_smem_resource(smem_o_1)
        add_smem_resource(gmem_o_0)
        add_smem_resource(gmem_o_1)
        smem_allocator.add_alias_group(
            [
                [smem_o_0._alloc],
                [gmem_o_0._alloc],
            ]
        )
        smem_allocator.add_alias_group(
            [
                [smem_o_1._alloc],
                [gmem_o_1._alloc],
            ]
        )
    tmem_ptr_alloc = smem_allocator.add_tmem_ptr(
        SmemAllocation("tmem_ptr_i32", dtype=cutlass.Int32, count=2, alignment=4)
    )
    dealloc_mbar_alloc = smem_allocator.add(
        SmemAllocation("tmem_dealloc_mbar", dtype=cutlass.Int64, alignment=8)
    )
    clc_response_alloc: SmemAllocation | None = None
    if is_clc_dynamic and clc_response_ptr is None:
        # Keep the CLC response inside the unified TS allocation. The kernel
        # derives its pointer from this descriptor after allocate(), so no
        # assumption about its physical offset is required.
        assert work_queue is not None
        assert work_queue.pipeline_config is not None
        clc_response_alloc = smem_allocator.add(
            SmemAllocation(
                "clc_response",
                dtype=cutlass.Int128,
                count=work_queue.pipeline_config.num_stages,
                alignment=16,
            )
        )
    smem_allocator.compute_layout()
    expected_barrier_bytes = (
        sum(
            _context_pipeline_stage_counts(
                cfg,
                kv_stages=cfg.kv_stage,
                is_clc_dynamic=is_clc_dynamic,
            ).values()
        )
        * _PIPELINE_BARRIER_BYTES_PER_STAGE
    )
    if smem_allocator.barrier_smem_bytes != expected_barrier_bytes:
        raise AssertionError(
            "context pipeline barrier accounting drifted: allocator has "
            f"{smem_allocator.barrier_smem_bytes} bytes, topology requires "
            f"{expected_barrier_bytes} bytes"
        )

    tmem_allocator = TmemAllocator()
    if single_qkv_instance:
        tmem_allocator.add_resource(tmem_o)
        tmem_allocator.add_resource(tmem_sp0)
        tmem_allocator.add_resource(tmem_vec0)
        if cfg.stage_scoped_tmem_stats and not cfg.stats_via_smem:
            tmem_allocator.add_alias_group(
                [
                    [tmem_sp0._alloc],
                    [tmem_vec0._alloc],
                ]
            )
    else:
        tmem_allocator.add_resource(tmem_sp0)
        tmem_allocator.add_resource(tmem_sp1)
        tmem_allocator.add_resource(tmem_vec0)
        tmem_allocator.add_resource(tmem_vec1)
        tmem_allocator.add_resource(tmem_o)
        tmem_allocator.add_alias_group(
            [
                [tmem_sp0._alloc],
                [tmem_vec0._alloc],
            ]
        )
        tmem_allocator.add_alias_group(
            [
                [tmem_sp1._alloc],
                [tmem_vec1._alloc],
            ]
        )
    tmem_allocator.compute_layout()

    skip = not isinstance(num_kv_tiles, int)
    task_manager = TaskManager(
        tasks=task_list,
        resource_dependency_graph=resource_dependency_graph,
        skip_validation=skip,
        verbose=not skip,
        smem_allocator=smem_allocator,
        tmem_allocator=tmem_allocator,
        exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
    )

    if single_qkv_instance:
        tmem_resources = [tmem_sp0, tmem_vec0, tmem_o, smem_o_0]
    else:
        tmem_resources = [
            tmem_sp0,
            tmem_sp1,
            tmem_vec0,
            tmem_vec1,
            tmem_o,
            smem_o_0,
            smem_o_1,
        ]
    return (
        task_manager,
        tmem_resources,
        tmem_ptr_alloc,
        dealloc_mbar_alloc,
        work_queue,
        clc_response_alloc,
    )


def _should_use_tmem_p_pipeline(cfg: FmhaConfig) -> bool:
    """Return whether P readiness needs its own TMEM pipeline resource.

    The TMEM P pipeline belongs to the staged, single-QKV-instance topology.
    That path stages K/V by 128-wide head-dimension slices and uses a separate
    P-ready handoff so the MMA task can overlap next-tile QK with previous-tile
    PV. The paired D128 path keeps P on TmemSPResource.
    """
    return cfg.single_qkv_instance and cfg.stage_kv_by_head_dim


_Q_ROW_SMEM_ALIGNMENT_BYTES = 128
_PIPELINE_BARRIER_BYTES_PER_STAGE = 2 * cutlass.Int64.width // 8


def _context_pipeline_stage_counts(
    cfg: FmhaConfig,
    *,
    kv_stages: int,
    is_clc_dynamic: bool,
) -> dict[str, int]:
    """Return every physical pipeline's mbarrier stage count."""
    counts = {
        "smem_q": cfg.q_stage,
        "smem_kv": kv_stages,
        "smem_page_offsets": (
            sum(cfg.page_offset_pipeline_stage_counts)
            if cfg.stages_page_offsets_in_smem
            else 0
        ),
        "tmem_sp": cfg.mma_softmax_stage * cfg.num_qkv_instances,
        "tmem_p": cfg.mma_softmax_stage if cfg.has_tmem_p_pipeline else 0,
        "tmem_vec": cfg.softmax_corr_stage * cfg.num_qkv_instances,
        "tmem_o": cfg.mma_corr_stage,
        "smem_o": cfg.num_qkv_instances,
        "s0s1_seq": 0 if cfg.single_qkv_instance else 1,
        "tmem_stats_done": 0 if cfg.stats_via_smem else cfg.num_qkv_instances,
        "work_queue": 1 if is_clc_dynamic else 0,
    }
    return {name: stages for name, stages in counts.items() if stages}


def _infer_single_instance_kv_stages(
    cfg: FmhaConfig,
    *,
    is_clc_dynamic: bool,
    page_table_window_entries: int | None = None,
    require_cadence: bool = True,
) -> int:
    """Return the deepest K/V ring that fits the exact TS SMEM footprint.

    All terms come from resource topology or public CUTLASS hardware metadata:
    Q, O, correction statistics, page-ID rings, fixed control records, and one
    16-byte pipeline barrier per physical stage. The task manager remains the
    authoritative check and uses the same stage-count policy below.
    """
    q_row_bytes = (cfg.q_dtype.width * cfg.qk_mma_tiler[2] + 7) // 8
    q_row_bytes = (
        (q_row_bytes + _Q_ROW_SMEM_ALIGNMENT_BYTES - 1)
        // _Q_ROW_SMEM_ALIGNMENT_BYTES
        * _Q_ROW_SMEM_ALIGNMENT_BYTES
    )
    q_tile_bytes = q_row_bytes * cfg.qk_mma_tiler[0]

    o_head_dim = (
        cfg.head_dim_per_stage_kv if cfg.stage_o_by_head_dim else cfg.epi_tile[1]
    )
    o_stage_bytes = (cfg.epi_tile[0] * o_head_dim * cfg.o_dtype.width + 7) // 8

    stats_bytes = 0
    if cfg.stats_via_smem:
        stats_rows = len(cfg.softmax0_warp_ids) * cute.arch.WARP_SIZE
        stats_values_per_row = 2
        stats_bytes = (
            cfg.softmax_corr_stage
            * stats_rows
            * stats_values_per_row
            * cutlass.Float32.width
            // 8
        )

    page_offset_stage_counts = cfg.page_offset_pipeline_stage_counts
    page_offset_stages = (
        sum(page_offset_stage_counts) if cfg.stages_page_offsets_in_smem else 0
    )
    if page_table_window_entries is None:
        page_table_window_entries = cfg.page_table_window_entries
    page_offsets_bytes = (
        page_offset_stages * page_table_window_entries * cutlass.Int32.width // 8
    )

    control_bytes = (2 * cutlass.Int32.width + cutlass.Int64.width) // 8
    if is_clc_dynamic:
        control_bytes += cutlass.Int128.width // 8
    fixed_barrier_stages = sum(
        _context_pipeline_stage_counts(
            cfg,
            kv_stages=0,
            is_clc_dynamic=is_clc_dynamic,
        ).values()
    )
    fixed_smem_bytes = (
        q_tile_bytes * cfg.q_stage
        + o_stage_bytes
        + stats_bytes
        + page_offsets_bytes
        + control_bytes
        + fixed_barrier_stages * _PIPELINE_BARRIER_BYTES_PER_STAGE
    )
    kv_dtype_width = max(cfg.k_dtype.width, cfg.v_dtype.width)
    kv_stage_bytes = (
        cfg.qk_mma_tiler[1] * cfg.head_dim_per_stage_kv * kv_dtype_width // 8
    )
    kv_stage_footprint_bytes = kv_stage_bytes + _PIPELINE_BARRIER_BYTES_PER_STAGE
    kv_budget_bytes = utils.get_smem_capacity_in_bytes("sm_100") - fixed_smem_bytes
    memory_fit_stages = kv_budget_bytes // kv_stage_footprint_bytes
    cadence_stages = cfg.num_head_dim_stages_k + cfg.num_head_dim_stages_v
    if require_cadence and memory_fit_stages < cadence_stages:
        raise ValueError(
            "single-instance context staging requires at least "
            f"{cadence_stages} K/V stages, but the shared-memory budget fits "
            f"only {memory_fit_stages}"
        )
    return memory_fit_stages


def _configure_pipeline_stages(cfg: FmhaConfig, *, is_clc_dynamic: bool) -> None:
    """Set topology- and capacity-derived context pipeline stage counts."""
    cfg.q_stage = cfg.num_qkv_instances
    cfg.kv_stage = 3
    cfg.has_tmem_p_pipeline = _should_use_tmem_p_pipeline(cfg)
    cfg.stage_scoped_tmem_stats = cfg.has_tmem_p_pipeline
    cfg.mma_softmax_stage = 2 if cfg.has_tmem_p_pipeline else 1
    cfg.softmax_corr_stage = 2 if cfg.stage_scoped_tmem_stats else 1
    # SMEM-backed D256 removes the independent StatsDone credit and always
    # writes the same physical O0 accumulator. Its MMA->Correction handoff must
    # therefore be single-stage so PV(i+1) cannot overwrite O0 before
    # Correction consumes PV(i). TMEM-stats schedules retain their established
    # two-stage O + StatsDone ordering.
    cfg.mma_corr_stage = 1 if cfg.single_qkv_instance and cfg.stats_via_smem else 2
    if cfg.single_qkv_instance:
        natural_page_window_entries = cute.arch.WARP_SIZE
        cfg.page_table_window_entries = natural_page_window_entries
        candidate_page_window_entries = cfg.page_table_window_candidate_entries
        if candidate_page_window_entries > natural_page_window_entries:
            candidate_kv_stages = _infer_single_instance_kv_stages(
                cfg,
                is_clc_dynamic=is_clc_dynamic,
                page_table_window_entries=candidate_page_window_entries,
                require_cadence=False,
            )
            cadence_stages = cfg.num_head_dim_stages_k + cfg.num_head_dim_stages_v
            if candidate_kv_stages >= cadence_stages:
                cfg.page_table_window_entries = candidate_page_window_entries
        cfg.kv_stage = _infer_single_instance_kv_stages(
            cfg,
            is_clc_dynamic=is_clc_dynamic,
        )


# Dense work traverses the full K domain for every Q tile, so its persistent
# mainloop keeps more registers on the load/MMA/epilogue/scheduler warpgroup.
# Causal work has a triangular, request-local K domain and retains the
# softmax/correction-heavy allocation. Both policies consume the same complete
# CTA register budget across the 8/4/4 participating warps; neither depends on
# batch size, sequence length, head count, layout, or a measured crossover.
_EARLY_TILE_SUM_DENSE_REGISTER_BUDGET = (176, 80, 80)
_EARLY_TILE_SUM_CAUSAL_REGISTER_BUDGET = (184, 88, 56)


def _configure_early_tile_sum_policy(
    cfg: FmhaConfig,
    *,
    is_persistent: bool,
) -> None:
    """Couple the concrete early-sum algorithm to its register budget."""
    # The current implementation has the Q2/KV1 paired topology and the D256
    # single-instance topology.  Q2/KV1 benefits across scheduler and head
    # mappings; keep the persistence condition explicit for parity with the
    # upstream policy if another paired geometry is added later.
    # D256 staged FP8 retires each probability through conversion and row-sum
    # reduction eight values after EXP2. It therefore returns a scalar tile
    # sum through the same task-local path as the paired early-sum policy while
    # retaining the D256 200/192/112 register split below.
    # D128 uses the same early-sum dataflow for every scheduler and storage
    # layout; scheduler selection must not silently change its math pipeline.
    cfg.enable_early_tile_sum = cfg.uses_d256_fp8_softmax_cadence or (
        cfg.uses_early_tile_sum and (not cfg.single_qkv_instance or is_persistent)
    )
    if cfg.single_qkv_instance:
        return
    if not cfg.enable_early_tile_sum:
        # Preserve the established register split for unsupported paired paths.
        # The legacy 192/96/32 fallback serves Q1/KV2, which is not implemented.
        return
    (
        cfg.num_regs_softmax,
        cfg.num_regs_correction,
        cfg.num_regs_other,
    ) = (
        _EARLY_TILE_SUM_CAUSAL_REGISTER_BUDGET
        if cfg.is_causal
        else _EARLY_TILE_SUM_DENSE_REGISTER_BUDGET
    )


def _configure_smem_shapes(cfg: FmhaConfig) -> None:
    """Derive per-stage SMEM element counts from the configured tile shapes."""
    kv_head_dim = (
        cfg.head_dim_per_stage_kv if cfg.stage_kv_by_head_dim else cfg.qk_mma_tiler[2]
    )
    o_head_dim = (
        cfg.head_dim_per_stage_kv if cfg.stage_o_by_head_dim else cfg.epi_tile[1]
    )
    cfg.sQ_shape = (
        cfg.q_stage,
        cfg.qk_mma_tiler[0] * cfg.qk_mma_tiler[2],
    )
    cfg.sK_shape = (
        cfg.kv_stage,
        cfg.qk_mma_tiler[1] * kv_head_dim,
    )
    cfg.sO_stage_elements = cfg.epi_tile[0] * o_head_dim


def _validate_tmem_columns(cfg: FmhaConfig) -> None:
    """Reject tile shapes that exceed the selected context TMEM layout."""
    sp_tmem_cols = cfg.num_qkv_instances * cfg.qk_mma_tiler[1] * cfg.mma_softmax_stage
    o_tmem_cols = cfg.epi_tile[1]
    stats_tmem_cols = 0
    if cfg.single_qkv_instance and not cfg.stage_scoped_tmem_stats:
        stats_tmem_cols = cfg.tmem_stats_cols
    required_tmem_cols = (
        sp_tmem_cols + cfg.num_qkv_instances * o_tmem_cols + stats_tmem_cols
    )
    if required_tmem_cols > cfg.tmem_alloc_cols:
        raise ValueError(
            f"head dimension {o_tmem_cols} requires {required_tmem_cols} "
            f"TMEM columns for the current FMHA context schedule, "
            f"but only {cfg.tmem_alloc_cols} are available"
        )


def _configure_single_instance_tmem_layout(cfg: FmhaConfig) -> None:
    """Select the one-S/P, one-O TMEM layout used for d>128 context FMHA."""
    cfg.tmem_o0_offset = 0
    cfg.tmem_s0_offset = cfg.epi_tile[1]
    cfg.tmem_p0_offset = cfg.tmem_s0_offset + cfg.tmem_x_load_s
    if cfg.stage_scoped_tmem_stats:
        cfg.tmem_vec0_offset = cfg.tmem_s0_offset
    else:
        cfg.tmem_vec0_offset = cfg.tmem_s0_offset + cfg.qk_mma_tiler[1]


def _configure_single_instance_warp_layout(cfg: FmhaConfig) -> None:
    """Use one softmax warpgroup and a compact producer/consumer warpgroup."""
    cfg.softmax0_warp_ids = (0, 1, 2, 3)
    cfg.softmax1_warp_ids = ()
    cfg.correction_warp_ids = (4, 5, 6, 7)
    cfg.mma_warp_id = 8
    cfg.load_warp_id = 9
    cfg.epilogue_warp_id = 10
    cfg.empty_warp_id = 11
    cfg.block_warps = 12
    # Match TRT-LLM-GEN's generated setmaxnreg split for Config_1_d256:
    # 4 softmax warps, 4 correction warps, and 4 producer/scheduler warps.
    cfg.num_regs_softmax = 200
    cfg.num_regs_correction = 192
    cfg.num_regs_other = 112


def _configure_head_dim_staging(cfg: FmhaConfig) -> None:
    """Split d>128 single-instance K/V/O staging into 128-wide slices."""
    cfg.head_dim_per_stage_kv = 0
    cfg.num_head_dim_stages_k = 1
    cfg.num_head_dim_stages_v = 1
    cfg.num_o_head_dim_stages = 1
    cfg.stage_kv_by_head_dim = False
    cfg.stage_o_by_head_dim = False
    if cfg.num_qkv_instances != 1:
        return
    cfg.head_dim_per_stage_kv = 128
    cfg.num_head_dim_stages_k = cfg.qk_mma_tiler[2] // cfg.head_dim_per_stage_kv
    cfg.num_head_dim_stages_v = cfg.pv_mma_tiler[1] // cfg.head_dim_per_stage_kv
    cfg.num_o_head_dim_stages = cfg.epi_tile[1] // cfg.head_dim_per_stage_kv
    cfg.stage_kv_by_head_dim = True
    cfg.stage_o_by_head_dim = True
    # Match TRT-LLM-GEN's d>128 policy: K, V, and O are staged as 128-wide
    # head-dimension slices so the K/V pipeline can run deeper without
    # exceeding Blackwell's SMEM budget.


def _configure_head_paired_tilers(
    cfg: FmhaConfig,
    *,
    mma_tiler_mn: tuple[int, int],
    d: int,
) -> None:
    """Set CTA/MMA/epilogue tile shapes for head-paired FMHA."""
    # Head-paired maps the two peer tiles onto Q heads instead of sequence
    # rows, so it keeps a one-tile CTA shape while reusing the same FmhaConfig
    # and task-manager entry point as query-paired FMHA.
    mma_tiler = (*mma_tiler_mn, d)
    cfg.qk_mma_tiler = mma_tiler
    cfg.pv_mma_tiler = (mma_tiler[0], mma_tiler[2], mma_tiler[1])
    cfg.epi_tile = cfg.pv_mma_tiler[:2]


def _configure_head_paired_tma_copy_metadata(
    cfg: FmhaConfig,
    *,
    q_dtype: type,
    k_dtype: type,
    o_dtype: type,
) -> None:
    """Derive TMA copy granularities for head-paired Q/K/V/O tensors."""
    inner_dim_size = cfg.qk_mma_tiler[2] * q_dtype.width // 8
    cfg.tma_copy_qkv_iters = 1
    if inner_dim_size % 128 == 0:
        cfg.tma_copy_qkv_iters = inner_dim_size // 128
    elif inner_dim_size != 64 and inner_dim_size != 32:
        raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")
    tma_copy_qkv_granu_inner = cfg.qk_mma_tiler[2] // cfg.tma_copy_qkv_iters

    cfg.q_tile_m = cfg.qk_mma_tiler[0]
    cfg.tma_copy_q_elements = cfg.sQ_shape[1]
    cfg.tma_copy_q_granu_inner = tma_copy_qkv_granu_inner
    cfg.tma_copy_q_granu_elems = cfg.tma_copy_q_elements // cfg.tma_copy_qkv_iters
    cfg.tma_copy_q_bytes = cfg.tma_copy_q_elements * q_dtype.width // 8

    cfg.seq_tile_n = cfg.qk_mma_tiler[1]
    cfg.kv_tile_n = cfg.qk_mma_tiler[1]
    kv_head_dim = (
        cfg.head_dim_per_stage_kv if cfg.stage_kv_by_head_dim else cfg.qk_mma_tiler[2]
    )
    cfg.tma_copy_kv_elements = cfg.sK_shape[1]
    cfg.tma_copy_kv_granu_inner = tma_copy_qkv_granu_inner
    cfg.tma_copy_kv_stage_iters = kv_head_dim // tma_copy_qkv_granu_inner
    cfg.tma_copy_kv_granu_elems = (
        cfg.tma_copy_kv_elements // cfg.tma_copy_kv_stage_iters
    )
    cfg.tma_copy_kv_bytes = cfg.tma_copy_kv_elements * k_dtype.width // 8

    output_inner_dim_size = cfg.epi_tile[1] * o_dtype.width // 8
    cfg.tma_copy_o_iters = 1
    if output_inner_dim_size % 128 == 0:
        cfg.tma_copy_o_iters = output_inner_dim_size // 128
    elif output_inner_dim_size != 64 and output_inner_dim_size != 32:
        raise RuntimeError(
            f"Unsupported output inner dimension size: {output_inner_dim_size}"
        )
    cfg.tma_copy_o_granu_inner = cfg.epi_tile[1] // cfg.tma_copy_o_iters
    o_head_dim = (
        cfg.head_dim_per_stage_kv if cfg.stage_o_by_head_dim else cfg.epi_tile[1]
    )
    cfg.tma_copy_o_stage_iters = o_head_dim // cfg.tma_copy_o_granu_inner
    cfg.tma_copy_o_elements = cfg.epi_tile[0] * o_head_dim
    cfg.tma_copy_o_granu_elems = cfg.tma_copy_o_elements // cfg.tma_copy_o_stage_iters


def _configure_common_launch_flags(
    cfg: FmhaConfig,
    *,
    d: int,
    h_r: int,
    is_causal: bool,
    balance_causal_workload: bool,
    window_size_left: int,
) -> None:
    """Fill launch flags shared by query-paired and head-paired modes."""
    cfg.h_r = h_r
    cfg.is_causal = is_causal
    cfg.balance_causal_workload = balance_causal_workload
    cfg.window_size_left = window_size_left


def _causal_domain_kwargs(
    *,
    num_kv_tiles: int | Int32,
    cta_m: int,
    kv_n: int,
    q_offset: int | Int32,
    seq_idx: int,
    batch_idx: int | None = None,
    cum_seqlen_q: cute.Tensor | None = None,
    cum_seqlen_k: cute.Tensor | None = None,
    runtime_kv_tile_multiple: int = 1,
    offset: int,
    reverse_seq_tiles: int | Int32 | None = None,
    window_size_left: int | None = None,
    packed_window: bool = False,
) -> DomainKwargs:
    """Build kwargs for a causal/window-aware task domain.

    ``offset`` is the loop-domain decrement applied after the causal/window
    tile count is computed. The schedule uses it to derive N, N-1, and N-2
    loop domains from the same causal formula without changing the Q/K tile
    coordinates or the S_q < S_kv ``q_offset`` mask shift.
    """
    result: DomainKwargs = {
        "task_class": CausalDomainTask,
        "num_kv_tiles": num_kv_tiles,
        "cta_m": cta_m,
        "kv_n": kv_n,
        "q_offset": q_offset,
        "seq_idx": seq_idx,
        "offset": offset,
    }
    if reverse_seq_tiles is not None:
        result["reverse_seq_tiles"] = reverse_seq_tiles
    if cum_seqlen_q is not None:
        if batch_idx is None or cum_seqlen_k is None:
            raise ValueError(
                "runtime causal domains require batch_idx and both cumulative "
                "sequence-length tensors"
            )
        result["batch_idx"] = batch_idx
        result["cum_seqlen_q"] = cum_seqlen_q
        result["cum_seqlen_k"] = cum_seqlen_k
        if runtime_kv_tile_multiple > 1:
            result["runtime_kv_tile_multiple"] = runtime_kv_tile_multiple
    if window_size_left is not None:
        result["window_size_left"] = window_size_left
    if packed_window:
        result["packed_window"] = True
    return result


def _select_fmha_domain_policy(
    cfg: FmhaConfig,
    *,
    num_kv_tiles: int | Int32,
    q_offset: int | Int32,
    cum_seqlen_q: cute.Tensor | None,
    cum_seqlen_k: cute.Tensor | None,
) -> FmhaDomainPolicy:
    """Select loop domains and softmax masks for the configured FMHA mode."""
    seq_idx = cfg.work_tile_coord_indices[0]
    batch_idx = cfg.work_tile_coord_indices[2]
    reverse_seq_tiles = (
        cfg.num_seq_tiles
        if cfg.uses_causal_reversed_head_batch_seq_tile_order
        else None
    )
    # Head-paired causal/window: both peers use the same Q sequence tile from
    # adjacent Q heads, so both softmax tasks share the same causal tail domain.
    if cfg.head_paired and cfg.is_causal:
        causal_n = _causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            cta_m=cfg.q_tile_m,
            kv_n=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            offset=0,
            reverse_seq_tiles=reverse_seq_tiles,
            window_size_left=cfg.window_size_left,
            packed_window=cfg.has_varlen,
        )
        causal_n_minus_1 = _causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            cta_m=cfg.q_tile_m,
            kv_n=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            offset=1,
            reverse_seq_tiles=reverse_seq_tiles,
            window_size_left=cfg.window_size_left,
            packed_window=cfg.has_varlen,
        )
        return FmhaDomainPolicy(
            domain_n_kwargs=causal_n,
            domain_n_minus_1_kwargs=causal_n_minus_1,
            softmax0_domain_kwargs={
                **causal_n_minus_1,
                "task_class": CausalSoftmaxDomainTask,
            },
            softmax1_domain_kwargs={
                **causal_n_minus_1,
                "task_class": CausalSoftmaxDomainTask,
            },
        )
    # Head-paired dense: no causal/window trimming is needed, so both peers use
    # the full static K/V domain and the same non-tail softmax mask settings.
    if cfg.head_paired:
        domain_n_kwargs: DomainKwargs = {"domain": num_kv_tiles}
        domain_n_minus_1_kwargs: DomainKwargs = {"domain": num_kv_tiles - 1}
        return FmhaDomainPolicy(
            domain_n_kwargs=domain_n_kwargs,
            domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
            softmax0_domain_kwargs=domain_n_kwargs,
            softmax1_domain_kwargs=domain_n_kwargs,
        )
    # Query-paired causal: peer0 and peer1 cover consecutive Q sequence tiles,
    # so peer0 may need one fewer K/V loop tile than peer1.
    if cfg.is_causal:
        if cfg.causal_single_kv_tile:
            # The whole fixed K/V extent is one tile.  Static N=1/N-1=0
            # domains balance the existing head/tail protocol without the
            # generic synthetic peer0 tail iteration.
            domain_n_kwargs: DomainKwargs = {"domain": 1}
            domain_n_minus_1_kwargs: DomainKwargs = {"domain": 0}
            return FmhaDomainPolicy(
                domain_n_kwargs=domain_n_kwargs,
                domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
                softmax0_domain_kwargs=domain_n_minus_1_kwargs,
                softmax1_domain_kwargs=domain_n_minus_1_kwargs,
            )
        runtime_kv_tile_multiple = (
            (cfg.cta_tiler[0] + cfg.kv_tile_n - 1) // cfg.kv_tile_n
            if cfg.skip_causal_invalid_peer0
            else 1
        )
        causal_n = _causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            cta_m=cfg.cta_tiler[0],
            kv_n=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            offset=0,
            reverse_seq_tiles=reverse_seq_tiles,
        )
        causal_n_minus_1 = _causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            cta_m=cfg.cta_tiler[0],
            kv_n=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            offset=1,
            reverse_seq_tiles=reverse_seq_tiles,
        )
        causal_n_minus_2 = _causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            cta_m=cfg.cta_tiler[0],
            kv_n=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            offset=2,
            reverse_seq_tiles=reverse_seq_tiles,
        )
        # Query-paired causal maps peer0 one Q tile before peer1.
        # Whether peer0 has an extra invalid SP tile depends on the QK MMA
        # tile geometry: if peer0 spans fewer K/V tiles than the paired CTA,
        # Softmax0 uses the N-2 domain and consumes the remaining extra tile in
        # TAIL.
        # Otherwise it uses the same N-1 domain as peer1. S_q < S_kv keeps N-1
        # and lets the shifted causal mask handle the right edge.
        mma0_has_invalid_tail = cfg.skip_causal_invalid_peer0
        mma0_domain = causal_n_minus_2 if mma0_has_invalid_tail else causal_n_minus_1
        mma1_domain = causal_n_minus_1
        softmax0_domain = {
            **mma0_domain,
            "task_class": CausalSoftmaxDomainTask,
        }
        softmax1_domain = {
            **mma1_domain,
            "task_class": CausalSoftmaxDomainTask,
        }
        return FmhaDomainPolicy(
            domain_n_kwargs=causal_n,
            domain_n_minus_1_kwargs=causal_n_minus_1,
            softmax0_domain_kwargs=softmax0_domain,
            softmax1_domain_kwargs=softmax1_domain,
        )
    # Dense query-paired fallback: no head-pairing or causal mask, so both
    # peers traverse the full static K/V domain.
    domain_n_kwargs = {"domain": num_kv_tiles}
    domain_n_minus_1_kwargs = {"domain": num_kv_tiles - 1}
    return FmhaDomainPolicy(
        domain_n_kwargs=domain_n_kwargs,
        domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
        softmax0_domain_kwargs=domain_n_kwargs,
        softmax1_domain_kwargs=domain_n_kwargs,
    )


def build_fmha_task_manager(
    cfg: FmhaConfig,
    tile_sched_params: (
        utils.PersistentTileSchedulerParams
        | utils.ClcDynamicPersistentTileSchedulerParams
        | None
    ),
    tma_q_desc: cutlass.Pointer | None,
    tma_k_desc: cutlass.Pointer | None,
    tma_v_desc: cutlass.Pointer | None,
    tma_o_desc: cutlass.Pointer | None,
    cum_seqlen_q: cute.Tensor | None,
    cum_seqlen_k: cute.Tensor | None,
    num_kv_tiles: int | Int32,
    scale_softmax_log2: cute.Tensor | None = None,
    output_scale: cute.Tensor | None = None,
    q_offset: int | Int32 = 0,
    g_page_idx_kv: cute.Pointer | None = None,
    g_seq_lens_kv: cute.Pointer | None = None,
    max_seq_len_kv: int | Int32 | None = None,
    is_persistent: bool = True,
    is_clc_dynamic: bool = False,
    clc_response_ptr: cute.Pointer | None = None,
    exhaustive_deadlock_race_check: bool = True,
) -> Tuple[
    TaskManager,
    list[MemoryResource],
    SmemAllocation,
    SmemAllocation,
    WorkQueue | None,
    SmemAllocation | None,
]:
    """Build the FMHA TaskManager using the shared context TS graph.

    Runtime scale arguments follow the same contract as
    :func:`build_context_task_manager`: one-element float32 device tensors,
    indexed at element 0 and cached by resource auxiliary work.
    """
    if (
        cfg.causal_single_kv_tile
        and isinstance(num_kv_tiles, int)
        and num_kv_tiles != 1
    ):
        raise ValueError(
            "causal_single_kv_tile requires exactly one compile-time K/V tile"
        )
    if cfg.causal_single_kv_tile and (cfg.use_paged_kv or cfg.has_varlen):
        raise ValueError("causal_single_kv_tile requires fixed contiguous K/V storage")
    domain_num_kv_tiles = num_kv_tiles
    if cfg.skip_causal_invalid_peer0:
        # Query-paired causal skips peer0 work with a constexpr last-loop test.
        # Partial final CTAs need the task domain padded so the extra peer0 slot
        # remains statically invalid; aligned static domains need no padding.
        paired_kv_tiles = (cfg.cta_tiler[0] + cfg.kv_tile_n - 1) // cfg.kv_tile_n
        if not isinstance(num_kv_tiles, int) or num_kv_tiles % paired_kv_tiles != 0:
            domain_num_kv_tiles = (
                cute.ceil_div(num_kv_tiles, paired_kv_tiles) * paired_kv_tiles
            )
    if cfg.reuses_page_table_windows:
        # The paged plan's maximum page count is compile-time semantic geometry.
        # Its tile count equals ceil(max_seq_len_kv / kv_tile_n), so exposing it
        # here gives stock TaskManager a structural stride-window loop without a
        # shape-tuned threshold or a custom control-flow feature.
        pages_per_kv_tile = cfg.kv_tile_n // cfg.num_tokens_per_page
        static_paged_num_kv_tiles = (
            cfg.max_num_pages_per_seq_kv + pages_per_kv_tile - 1
        ) // pages_per_kv_tile
        domain_num_kv_tiles = static_paged_num_kv_tiles
    # Keep equal-length fixed launches constexpr-zero while preserving the
    # runtime per-request offset path for packed/bottom-right attention.
    effective_q_offset = q_offset if cfg.has_q_offset else 0
    domain_policy = _select_fmha_domain_policy(
        cfg,
        num_kv_tiles=domain_num_kv_tiles,
        q_offset=effective_q_offset,
        # A uniform packed plan remains uniform under its replay contract, so
        # keep that specialization free of redundant GMEM indptr loads. Mixed
        # packed and paged plans derive their causal domain from live Q and
        # logical-K cumulative offsets instead.
        cum_seqlen_q=(
            cum_seqlen_q if cfg.has_varlen and not cfg.has_uniform_varlen else None
        ),
        cum_seqlen_k=(
            cum_seqlen_k if cfg.has_varlen and not cfg.has_uniform_varlen else None
        ),
    )

    return build_context_task_manager(
        cfg=cfg,
        tile_sched_params=tile_sched_params,
        tma_q_desc=tma_q_desc,
        tma_k_desc=tma_k_desc,
        tma_v_desc=tma_v_desc,
        tma_o_desc=tma_o_desc,
        cum_seqlen_q=cum_seqlen_q,
        cum_seqlen_k=cum_seqlen_k,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        g_page_idx_kv=g_page_idx_kv,
        g_seq_lens_kv=g_seq_lens_kv,
        max_seq_len_kv=max_seq_len_kv,
        num_kv_tiles=domain_num_kv_tiles,
        q_offset=effective_q_offset,
        domain_n_kwargs=domain_policy.domain_n_kwargs,
        domain_n_minus_1_kwargs=domain_policy.domain_n_minus_1_kwargs,
        softmax0_domain_kwargs=domain_policy.softmax0_domain_kwargs,
        softmax1_domain_kwargs=domain_policy.softmax1_domain_kwargs,
        is_persistent=is_persistent,
        is_clc_dynamic=is_clc_dynamic,
        clc_response_ptr=clc_response_ptr,
        exhaustive_deadlock_race_check=exhaustive_deadlock_race_check,
    )


# ---------------------------------------------------------------------------
# GPU Kernel
# ---------------------------------------------------------------------------


class FmhaTs:
    """Warp-specialised persistent FMHA kernel using the TS framework.

    Usage::

        fmha = FmhaTs(qk_acc_dtype=Float32, pv_acc_dtype=Float32,
                        mma_tiler_mn=(128, 128))
        fmha(q_cute, k_cute, v_cute, o_cute, stream)

    Parameters
    ----------
    qk_acc_dtype : type, optional
        Accumulator dtype for QK GEMM (default: Float32).
    pv_acc_dtype : type, optional
        Accumulator dtype for PV GEMM (default: Float32).
    in_dtype : type, optional
        Input tensor dtype for Q, K, V (default: Float16).
    out_dtype : type, optional
        Output tensor dtype for O (default: Float16).
    mma_tiler_mn : Tuple[int, int], optional
        MMA tile shape (M, N) (default: (128, 128)).
    d : int, optional
        Head dimension (default: 128).
    is_persistent : bool, optional
        Use persistent scheduling (default: True).
    is_causal : bool, optional
        Enable causal masking (default: False).
    balance_causal_workload : bool, optional
        Use TRT-style causal workload balancing: head_batch_seq logical tile
        order with reversed Q sequence tiles. Paired causal CLC schedules
        enable this automatically; setting the flag also requests it for other
        causal scheduler topologies.
    is_clc_dynamic : bool, optional
        Use CLC dynamic persistent scheduling (default: False).
        Requires ``is_persistent=True``.
    h_r : int, optional
        Number of GQA head repeats (default: 1). Head-paired mode requires
        grouped-query attention with an even repeat count.
    enable_skip_correction : bool, optional
        Enable skip-correction for softmax rescaling (default: True).
    causal_single_kv_tile : bool, optional
        Use the fixed causal one-K/V-tile task domains. The context runner
        enables this only for query-paired, fixed-length inputs whose K/V
        extent fits one 128-token tile (default: False).
    """

    def __init__(
        self,
        qk_acc_dtype: type = None,
        pv_acc_dtype: type = None,
        in_dtype: type = None,
        out_dtype: type = None,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        d: int = 128,
        is_persistent: bool = True,
        is_causal: bool = False,
        balance_causal_workload: bool = False,
        is_clc_dynamic: bool = False,
        head_paired: bool = False,
        window_size_left: int = 0,
        h_r: int = 1,
        enable_skip_correction: bool = True,
        use_paged_kv: bool = False,
        num_tokens_per_page: int = 32,
        max_num_pages_per_seq_kv: int = 1,
        causal_single_kv_tile: bool = False,
        exhaustive_deadlock_race_check: bool = True,
    ) -> None:
        """Initialize mode-specific tiling, dtype, and schedule configuration."""
        head_paired = resolve_head_paired_mode(
            head_paired=head_paired,
            is_causal=is_causal,
            window_size_left=window_size_left,
        )
        if causal_single_kv_tile and (not is_causal or head_paired):
            raise ValueError(
                "causal_single_kv_tile requires query-paired causal attention"
            )
        if causal_single_kv_tile and use_paged_kv:
            raise ValueError(
                "causal_single_kv_tile requires fixed contiguous K/V storage"
            )
        if is_clc_dynamic and not is_persistent:
            raise ValueError("CLC dynamic scheduling requires persistent mode")
        if head_paired and not is_persistent:
            raise ValueError("Head-paired scheduling requires persistent mode")
        validate_head_paired_head_ratio(head_paired=head_paired, h_r=h_r)
        if use_paged_kv:
            if num_tokens_per_page not in _SUPPORTED_CONTEXT_PAGE_SIZES:
                raise ValueError(
                    "paged context requires num_tokens_per_page in "
                    f"{_SUPPORTED_CONTEXT_PAGE_SIZES}; got "
                    f"{num_tokens_per_page}"
                )
            if max_num_pages_per_seq_kv < 1:
                raise ValueError(
                    f"max_num_pages_per_seq_kv must be >= 1, got "
                    f"{max_num_pages_per_seq_kv}"
                )
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_clc_dynamic = is_clc_dynamic
        self.exhaustive_deadlock_race_check = exhaustive_deadlock_race_check

        q_dtype = in_dtype or cutlass.Float16
        k_dtype = in_dtype or cutlass.Float16
        v_dtype = in_dtype or cutlass.Float16
        o_dtype = out_dtype or cutlass.Float16

        cfg = FmhaConfig()
        self.cfg = cfg
        if d > 128:
            cfg.num_qkv_instances = 1
        cfg.use_paged_kv = use_paged_kv
        single_instance_persistent = (
            is_persistent and cfg.single_qkv_instance and not head_paired
        )
        # Paired Q2 and persistent D256 schedules route correction statistics
        # through a compact SMEM ring and omit the per-K StatsDone
        # serialization. This keeps the stats payload disjoint from the S/P
        # TMEM columns while producer latency varies across storage layouts.
        cfg.stats_via_smem = single_instance_persistent or (not cfg.single_qkv_instance)
        cfg.fuse_epilogue_into_correction = cfg.single_qkv_instance
        cfg.num_tokens_per_page = num_tokens_per_page
        cfg.max_num_pages_per_seq_kv = max_num_pages_per_seq_kv
        cfg.causal_single_kv_tile = causal_single_kv_tile
        # FP16/BF16 causal attention retains the default 192/96/32
        # softmax/correction/auxiliary split. Other topologies start from
        # 184/88/56; the paired D128 early-sum policy is rebalanced below.
        # Every selected split totals 2048 registers across the 16 warps.
        if not (is_causal and q_dtype.width == 16):
            cfg.num_regs_softmax = 184
            cfg.num_regs_correction = 88
            cfg.num_regs_other = 56
        cfg.enable_skip_correction = enable_skip_correction
        cfg.qk_acc_dtype = qk_acc_dtype or cutlass.Float32
        cfg.pv_acc_dtype = pv_acc_dtype or cutlass.Float32

        # Store dtypes as compile-time constants
        cfg.q_dtype = q_dtype
        cfg.k_dtype = k_dtype
        cfg.v_dtype = v_dtype
        cfg.o_dtype = o_dtype
        cfg.head_paired = head_paired
        cfg.is_causal = is_causal
        balance_causal_workload = balance_causal_workload or (
            is_causal and is_clc_dynamic and not cfg.single_qkv_instance
        )

        if head_paired:
            _configure_head_paired_tilers(cfg, mma_tiler_mn=mma_tiler_mn, d=d)
            _configure_head_dim_staging(cfg)
            _configure_pipeline_stages(cfg, is_clc_dynamic=is_clc_dynamic)
            if cfg.single_qkv_instance:
                _configure_single_instance_tmem_layout(cfg)
                _configure_single_instance_warp_layout(cfg)
            _configure_smem_shapes(cfg)
            _validate_tmem_columns(cfg)
            _configure_head_paired_tma_copy_metadata(
                cfg,
                q_dtype=q_dtype,
                k_dtype=k_dtype,
                o_dtype=o_dtype,
            )
            _configure_common_launch_flags(
                cfg,
                d=d,
                h_r=h_r,
                is_causal=is_causal,
                balance_causal_workload=balance_causal_workload,
                window_size_left=window_size_left,
            )
            _configure_early_tile_sum_policy(cfg, is_persistent=is_persistent)
            return

        # MMA tiler: (M, N, K) = (128, 128, 128)
        mma_tiler = (*mma_tiler_mn, d)
        cfg.qk_mma_tiler = mma_tiler
        cfg.pv_mma_tiler = (mma_tiler[0], mma_tiler[2], mma_tiler[1])
        cfg.epi_tile = cfg.pv_mma_tiler[:2]
        _configure_head_dim_staging(cfg)
        _configure_pipeline_stages(cfg, is_clc_dynamic=is_clc_dynamic)
        if cfg.single_qkv_instance:
            _configure_single_instance_tmem_layout(cfg)
            _configure_single_instance_warp_layout(cfg)

        # Pipeline stages and SMEM shapes (stages, elements_per_stage)
        _configure_smem_shapes(cfg)
        _validate_tmem_columns(cfg)

        # TMA copy granularity for Q
        qkv_tma_bits = cfg.qk_mma_tiler[2] * q_dtype.width
        if qkv_tma_bits % (128 * 8) != 0:
            raise ValueError(
                "FMHA TS requires a 128-byte aligned Q/K/V inner dimension, "
                f"got {qkv_tma_bits // 8} bytes"
            )
        cfg.tma_copy_qkv_iters = qkv_tma_bits // (128 * 8)
        cfg.q_tile_m = cfg.qk_mma_tiler[0]
        cfg.tma_copy_q_granu_inner = cfg.qk_mma_tiler[2] // cfg.tma_copy_qkv_iters
        cfg.tma_copy_q_elements = cfg.sQ_shape[1]
        cfg.tma_copy_q_granu_elems = cfg.tma_copy_q_elements // cfg.tma_copy_qkv_iters
        cfg.tma_copy_q_bytes = cfg.tma_copy_q_elements * q_dtype.width // 8

        # TMA copy granularity for KV
        cfg.kv_tile_n = cfg.qk_mma_tiler[1]
        kv_head_dim = (
            cfg.head_dim_per_stage_kv
            if cfg.stage_kv_by_head_dim
            else cfg.qk_mma_tiler[2]
        )
        cfg.tma_copy_kv_granu_inner = cfg.qk_mma_tiler[2] // cfg.tma_copy_qkv_iters
        cfg.tma_copy_kv_elements = cfg.sK_shape[1]
        cfg.tma_copy_kv_stage_iters = kv_head_dim // cfg.tma_copy_kv_granu_inner
        cfg.tma_copy_kv_granu_elems = (
            cfg.tma_copy_kv_elements // cfg.tma_copy_kv_stage_iters
        )
        cfg.tma_copy_kv_bytes = cfg.tma_copy_kv_elements * k_dtype.width // 8

        # TMA copy granularity for O
        cfg.tma_copy_o_iters = (cfg.epi_tile[1] * o_dtype.width) // 1024
        cfg.tma_copy_o_granu_inner = cfg.epi_tile[1] // cfg.tma_copy_o_iters
        o_head_dim = (
            cfg.head_dim_per_stage_kv if cfg.stage_o_by_head_dim else cfg.epi_tile[1]
        )
        cfg.tma_copy_o_stage_iters = o_head_dim // cfg.tma_copy_o_granu_inner
        cfg.tma_copy_o_elements = cfg.epi_tile[0] * o_head_dim
        cfg.tma_copy_o_granu_elems = (
            cfg.tma_copy_o_elements // cfg.tma_copy_o_stage_iters
        )

        _configure_common_launch_flags(
            cfg,
            d=d,
            h_r=h_r,
            is_causal=is_causal,
            balance_causal_workload=balance_causal_workload,
            window_size_left=window_size_left,
        )
        _configure_early_tile_sum_policy(cfg, is_persistent=is_persistent)

    # ---------------------------------------------------------------------------
    # Host entry point
    # ---------------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        q_cute: cute.Tensor,
        k_cute: cute.Tensor,
        v_cute: cute.Tensor,
        o_cute: cute.Tensor,
        scale_softmax_log2: cute.Tensor,
        output_scale: cute.Tensor,
        max_active_clusters: int,
        stream: cuda_drv.CUstream,
        cum_seqlen_q: cute.Tensor | None = None,
        cum_seqlen_k: cute.Tensor | None = None,
        max_seqlen_q: Int32 | None = None,
        max_seqlen_k: Int32 | None = None,
        page_idx_kv: cute.Tensor | None = None,
        seq_lens_kv: cute.Tensor | None = None,
    ) -> None:
        """Set up TMA descriptors, compute grid, and launch the kernel.

        ``scale_softmax_log2`` and ``output_scale`` must be one-element float32
        device tensors. Example host values are
        ``[math.log2(math.e) / math.sqrt(d)]`` for the softmax scale and
        ``[1.0]`` for FP16 output. FP8 callers may fold Q/K/V and output
        quantization scales into these runtime tensors.
        """
        cfg = self.cfg

        # Create TMA descriptors. Both query-paired and head-paired modes use
        # the same logical Q/K/V/O tensor-map boxes after FmhaConfig lowers the
        # mode-specific CTA shape. Fixed-shape launches include the batch rank;
        # varlen launches drop it because the flattened tensors are indexed via
        # cum_seqlen metadata.
        tma_qkv_swizzle = cuda.TensorMapSwizzle.s128b
        # Paged K/V issues one 128-byte tensor-map fragment per physical page.
        # Promote that exact fragment width so the Q-head tiles sharing each
        # logical K/V tile reuse the same cache line, matching the reference
        # tensor-map descriptor policy.
        tma_kv_l2_promotion = (
            cuda.TensorMapL2Promotion.l2_128b
            if cutlass.const_expr(cfg.use_paged_kv)
            else cuda.TensorMapL2Promotion.none
        )
        if cutlass.const_expr(cfg.head_paired):
            inner_dim_size = cfg.qk_mma_tiler[2] * cfg.q_dtype.width // 8
            tma_qkv_swizzle = cuda.TensorMapSwizzle.none
            if cutlass.const_expr(inner_dim_size % 128 == 0):
                tma_qkv_swizzle = cuda.TensorMapSwizzle.s128b
                tma_kv_l2_promotion = cuda.TensorMapL2Promotion.l2_128b
            elif cutlass.const_expr(inner_dim_size == 64):
                tma_qkv_swizzle = cuda.TensorMapSwizzle.s64b
                tma_kv_l2_promotion = cuda.TensorMapL2Promotion.l2_64b
            elif cutlass.const_expr(inner_dim_size == 32):
                tma_qkv_swizzle = cuda.TensorMapSwizzle.s32b

        output_inner_dim_size = cfg.tma_copy_o_granu_inner * cfg.o_dtype.width // 8
        tma_o_swizzle = cuda.TensorMapSwizzle.none
        if cutlass.const_expr(output_inner_dim_size % 128 == 0):
            tma_o_swizzle = cuda.TensorMapSwizzle.s128b
        elif cutlass.const_expr(output_inner_dim_size == 64):
            tma_o_swizzle = cuda.TensorMapSwizzle.s64b
        elif cutlass.const_expr(output_inner_dim_size == 32):
            tma_o_swizzle = cuda.TensorMapSwizzle.s32b

        q_box_dims = (1, cfg.qk_mma_tiler[0], 1, cfg.tma_copy_q_granu_inner)
        kv_box_dims = (1, cfg.qk_mma_tiler[1], 1, cfg.tma_copy_kv_granu_inner)
        v_box_dims = (
            1,
            cfg.pv_mma_tiler[2],
            1,
            cfg.pv_mma_tiler[1] // cfg.tma_copy_qkv_iters,
        )
        o_box_dims = (1, cfg.epi_tile[0], 1, cfg.tma_copy_o_granu_inner)
        stride_order = (3, 2, 1, 0)
        # K/V descriptors track the K/V tensor rank, which paged-KV pins to 4
        # (page pool) regardless of var-len. Q/O follow the var-len decision.
        kv_stride_order = stride_order
        if cutlass.const_expr(cum_seqlen_q is not None):
            q_box_dims = (cfg.qk_mma_tiler[0], 1, cfg.tma_copy_q_granu_inner)
            o_box_dims = (cfg.epi_tile[0], 1, cfg.tma_copy_o_granu_inner)
            stride_order = (2, 1, 0)
            if cutlass.const_expr(not cfg.use_paged_kv):
                kv_box_dims = (
                    cfg.qk_mma_tiler[1],
                    1,
                    cfg.tma_copy_kv_granu_inner,
                )
                v_box_dims = (
                    cfg.pv_mma_tiler[2],
                    1,
                    cfg.pv_mma_tiler[1] // cfg.tma_copy_qkv_iters,
                )
                kv_stride_order = stride_order
        if cutlass.const_expr(cum_seqlen_q is not None):
            # Q/O use the packed sequence axis as a ragged TMA dimension.
            # Dense TMA bounds only see the full sum_seqlen tensor and cannot
            # prevent a partial final tile from crossing into the next packed
            # sequence.  K/V can keep dense descriptors because invalid K/V
            # lanes are masked out before softmax/PV consumes them.
            tma_q_desc = create_tensor_map_ragged_from_tensor(
                q_cute,
                box_dims=q_box_dims,
                ragged_dim=0,
                stride_order=stride_order,
                swizzle=tma_qkv_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.none,
            )
        else:
            tma_q_desc = cuda.create_tensor_map_tiled_from_view(
                q_cute,
                box_dims=q_box_dims,
                stride_order=stride_order,
                swizzle=tma_qkv_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.none,
            )

        if cutlass.const_expr(cfg.use_paged_kv):
            # Paged-KV path: K/V are pool tensors with shape
            # (total_pages, h_kv, num_tokens_per_page, d). The TMA box covers
            # one page × one d-fragment; the loader stitches pages and d-halves
            # together via per-fragment coords. Inner d-tile must equal
            # tma_copy_kv_granu_inner (64 fp16 = 128 B) to match the s128b
            # swizzle the contiguous path uses.
            paged_kv_box_dims = (
                1,
                1,
                cfg.num_tokens_per_page,
                cfg.tma_copy_kv_granu_inner,
            )
            tma_k_desc = cuda.create_tensor_map_tiled_from_view(
                k_cute,
                box_dims=paged_kv_box_dims,
                stride_order=kv_stride_order,
                swizzle=tma_qkv_swizzle,
                l2_promotion=tma_kv_l2_promotion,
            )
            tma_v_desc = cuda.create_tensor_map_tiled_from_view(
                v_cute,
                box_dims=paged_kv_box_dims,
                stride_order=kv_stride_order,
                swizzle=tma_qkv_swizzle,
                l2_promotion=tma_kv_l2_promotion,
            )
        else:
            tma_k_desc = cuda.create_tensor_map_tiled_from_view(
                k_cute,
                box_dims=kv_box_dims,
                stride_order=kv_stride_order,
                swizzle=tma_qkv_swizzle,
                l2_promotion=tma_kv_l2_promotion,
            )
            tma_v_desc = cuda.create_tensor_map_tiled_from_view(
                v_cute,
                box_dims=v_box_dims,
                stride_order=kv_stride_order,
                swizzle=tma_qkv_swizzle,
                l2_promotion=tma_kv_l2_promotion,
            )

        if cutlass.const_expr(cum_seqlen_q is not None):
            tma_o_desc = create_tensor_map_ragged_from_tensor(
                o_cute,
                box_dims=o_box_dims,
                ragged_dim=0,
                stride_order=stride_order,
                swizzle=tma_o_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.none,
            )
        else:
            tma_o_desc = cuda.create_tensor_map_tiled_from_view(
                o_cute,
                box_dims=o_box_dims,
                stride_order=stride_order,
                swizzle=tma_o_swizzle,
                l2_promotion=cuda.TensorMapL2Promotion.none,
            )

        # Compute tile scheduler and grid
        if cutlass.const_expr(cum_seqlen_q is None):
            b = o_cute.shape[0]
            s_q = o_cute.shape[1]
            h_q = o_cute.shape[2]
            if cutlass.const_expr(cfg.use_paged_kv):
                # k_cute is the page pool — its sequence axis is per-page,
                # not the logical max_seq_len_kv. Caller must pass it in.
                s_k = max_seqlen_k
            else:
                s_k = k_cute.shape[1]
        else:
            b = cute.size(cum_seqlen_q) - 1
            s_q = max_seqlen_q
            h_q = o_cute.shape[1]
            s_k = max_seqlen_k

        num_seq_tiles = cute.ceil_div(s_q, cfg.cta_tiler[0])
        num_kv_tiles = cute.ceil_div(s_k, cfg.kv_tile_n)
        num_head_tiles = h_q // cfg.work_tile_q_heads
        q_offset = Int32(0) if cutlass.const_expr(cfg.head_paired) else 0
        if cutlass.const_expr(self.is_causal):
            q_offset = s_k - s_q

        # Tile scheduling order:
        #
        # Causal defaults to seq_head_batch to keep adjacent Q sequence tiles
        # close for K/V locality. Paired FP8 uses head_batch_seq, and paired
        # causal CLC additionally reverses its sequence axis so the dynamic
        # queue retires the heaviest causal tiles first. An explicit
        # balance_causal_workload request uses that same reversed order.
        #
        # Dense GQA uses head_batch_seq so Q-head groups that share one K/V
        # head stay adjacent. Dense MHA retains seq_head_batch because it has
        # no cross-head K/V reuse.
        if cutlass.const_expr(cfg.uses_head_batch_seq_tile_order):
            problem_shape = (num_head_tiles, b, num_seq_tiles)
        else:
            problem_shape = (num_seq_tiles, num_head_tiles, b)

        if cutlass.const_expr(self.is_clc_dynamic):
            tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                problem_shape,
                cfg.cluster_shape_mn + (1,),
            )
            grid = tile_sched_params.get_grid_shape()
        elif cutlass.const_expr(self.is_persistent):
            tile_sched_params = utils.PersistentTileSchedulerParams(
                problem_shape,
                cfg.cluster_shape_mn + (1,),
            )
            grid = utils.StaticPersistentTileScheduler.get_grid_shape(
                tile_sched_params,
                max_active_clusters,
            )
        else:
            # Non-persistent launch: one CTA per tile. No tile scheduler params
            # are passed into the task manager, so tasks use the hardware tile
            # coordinate directly instead of a WorkQueue.
            tile_sched_params = None
            grid = problem_shape

        block_size = cfg.block_warps * 32

        # Paged-KV side-channel data: page table iterator + per-batch K/V
        # length lookup. None in the contiguous path; the kernel branches on
        # ``cfg.use_paged_kv`` so passing None is safe.
        page_idx_kv_iter = (
            page_idx_kv.iterator
            if cutlass.const_expr(page_idx_kv is not None)
            else None
        )
        seq_lens_kv_iter = (
            seq_lens_kv.iterator
            if cutlass.const_expr(seq_lens_kv is not None)
            else None
        )

        self.kernel(
            tma_q_desc,
            tma_k_desc,
            tma_v_desc,
            tma_o_desc,
            tile_sched_params,
            scale_softmax_log2,
            output_scale,
            num_kv_tiles,
            num_seq_tiles,
            q_offset,
            cum_seqlen_q,
            cum_seqlen_k,
            page_idx_kv_iter,
            seq_lens_kv_iter,
            Int32(s_k),
            self.is_persistent,
            self.is_clc_dynamic,
        ).launch(
            grid=grid,
            block=[block_size, 1, 1],
            cluster=cfg.cluster_shape_mn + (1,),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ---------------------------------------------------------------------------
    # Device kernel
    # ---------------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tma_q_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_k_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_v_desc: cutlass.GridConstant[cuda.TensorMap],
        tma_o_desc: cutlass.GridConstant[cuda.TensorMap],
        tile_sched_params: (
            utils.PersistentTileSchedulerParams
            | utils.ClcDynamicPersistentTileSchedulerParams
            | None
        ),
        scale_softmax_log2: cute.Tensor,
        output_scale: cute.Tensor,
        num_kv_tiles: Int32,
        num_seq_tiles: Int32,
        q_offset: Int32,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        page_idx_kv: cute.Pointer | None,
        seq_lens_kv: cute.Pointer | None,
        max_seq_len_kv: Int32 | None,
        is_persistent: cutlass.Constexpr[bool] = True,
        is_clc_dynamic: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Warp-specialised persistent FMHA TS kernel.

        ``scale_softmax_log2`` and ``output_scale`` are one-element float32
        device tensors passed through to the TS resources. The resources load
        element 0 in auxiliary setup before the K/V loop, so the hot loop uses
        the cached task-local values rather than reloading global memory.

        Structure:
        1. Prefetch TMA descriptors
        2. Build TaskManager (resources + tasks + dependency graph)
        3. setup_resources_and_tasks() — unified SMEM alloc + pipeline barriers
        4. Derive infrastructure ptrs (tmem_ptr, dealloc mbar) from SMEM block
        5. Init tmem dealloc barrier + fence + cluster sync
        6. TMEM allocation + immediate permit relinquish (MMA warp) + named barrier sync
        7. task_manager.run() — persistent execution
        8. TMEM deallocation
        """
        cfg = self.cfg
        cfg.num_seq_tiles = num_seq_tiles
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # 1. Prefetch TMA descriptors on load warp
        if warp_idx == cfg.load_warp_id:
            prims.prefetch_tensormap(tma_q_desc.get_ptr())
            prims.prefetch_tensormap(tma_k_desc.get_ptr())
            prims.prefetch_tensormap(tma_v_desc.get_ptr())
            prims.prefetch_tensormap(tma_o_desc.get_ptr())

        # 2. CLC dynamic: the response buffer is declared by the builder and
        # bound from the unified task-manager SMEM allocation below.
        clc_response_ptr = None

        # 3. Build TaskManager (infrastructure slots via SmemAllocator)
        (
            task_manager,
            tmem_resources,
            tmem_ptr_alloc,
            dealloc_mbar_alloc,
            work_queue,
            clc_response_alloc,
        ) = build_fmha_task_manager(
            cfg=cfg,
            tile_sched_params=tile_sched_params,
            tma_q_desc=tma_q_desc.get_ptr(),
            tma_k_desc=tma_k_desc.get_ptr(),
            tma_v_desc=tma_v_desc.get_ptr(),
            tma_o_desc=tma_o_desc.get_ptr(),
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            g_page_idx_kv=page_idx_kv,
            g_seq_lens_kv=seq_lens_kv,
            max_seq_len_kv=max_seq_len_kv,
            num_kv_tiles=num_kv_tiles,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            q_offset=q_offset,
            is_persistent=is_persistent,
            is_clc_dynamic=is_clc_dynamic,
            clc_response_ptr=clc_response_ptr,
            exhaustive_deadlock_race_check=self.exhaustive_deadlock_race_check,
        )

        # Bind the CLC scheduler to its exact suballocation before WorkQueue
        # create() materializes the concrete scheduler during normal setup.
        if cutlass.const_expr(is_clc_dynamic):
            assert work_queue is not None
            assert clc_response_alloc is not None
            smem_allocator = task_manager.smem_allocator
            assert smem_allocator is not None
            smem_allocator.allocate()
            clc_response = smem_allocator.get(clc_response_alloc)
            clc_response_ptr = cute.make_ptr(
                cutlass.Int128,
                clc_response.data_ptr(),
                mem_space=cutlass.AddressSpace.smem,
                assumed_align=16,
            )
            work_queue.tile_scheduler_config = (
                TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                    tile_scheduler_params=tile_sched_params,
                    response_ptr=clc_response_ptr,
                )
            )

        # 4. Initialize all pipeline barriers + allocate unified SMEM
        task_manager.setup_resources_and_tasks()

        # Derive infrastructure pointers from the unified SMEM block.
        # tmem_ptr_i32 for ResourceContext is auto-populated by TaskManager
        # in setup_resources_and_tasks() via SmemAllocator.tmem_ptr_alloc.
        smem_allocator = task_manager.smem_allocator
        tmem_ptr_i32 = smem_allocator.get(tmem_ptr_alloc)
        tmem_dealloc_mbar = smem_allocator.get(dealloc_mbar_alloc)

        # 5. Initialize tmem dealloc barrier
        num_tmem_consumer_threads = cute.arch.WARP_SIZE * (
            len(cfg.softmax0_warp_ids)
            + len(cfg.softmax1_warp_ids)
            + len(cfg.correction_warp_ids)
        )
        if warp_idx == cfg.empty_warp_id:
            if prims.elect_sync():
                prims.mbarrier_init(tmem_dealloc_mbar, num_tmem_consumer_threads)

        # Fence barrier inits from setup_resources_and_tasks() — pipeline
        # barriers need fencing before first use.
        prims.fence_mbarrier_init()

        # Cluster sync before TMEM allocation / task execution
        prims.barrier_cluster_arrive_relaxed()
        prims.barrier_cluster_wait()

        # 6. TMEM allocation (MMA warp)
        if warp_idx == cfg.mma_warp_id:
            tmem_alloc_cols = Int32(cfg.tmem_alloc_cols)
            prims.tcgen05_alloc(tmem_ptr_i32, tmem_alloc_cols)
            prims.tcgen05_relinquish_alloc_permit()

        # All-warp barrier to ensure TMEM allocation is visible.
        # Previously only correction+MMA warps synced; expanded to all
        # warps so we can cache tmem_ptr_i32 once and avoid repeated
        # ld.shared in resource work methods (SMEM is volatile to LLVM).
        prims.barrier_cta_sync(
            cfg.tmem_bar_id,
            thread_count=cfg.block_warps * cute.arch.WARP_SIZE,
        )

        # Cache TMEM address: one ld.shared per warp.  This replaces
        # ~6 ld.shared calls spread across resource work methods (each
        # cascading into shr/and/shl address arithmetic that the compiler
        # cannot hoist because SMEM is volatile).
        tmem_addr_cached = tmem_ptr_i32.load()
        for r in tmem_resources:
            r.tmem_addr_cached = tmem_addr_cached

        # 7. Execute all tasks via the TS scheduler
        task_manager.run()

        # 8. TMEM deallocation
        is_softmax_warp = (
            warp_idx >= cfg.softmax0_warp_ids[0]
            and warp_idx <= cfg.softmax0_warp_ids[-1]
        )
        if cutlass.const_expr(len(cfg.softmax1_warp_ids) > 0):
            is_softmax_warp = is_softmax_warp or (
                warp_idx >= cfg.softmax1_warp_ids[0]
                and warp_idx <= cfg.softmax1_warp_ids[-1]
            )
        is_correction_warp = (
            warp_idx >= cfg.correction_warp_ids[0]
            and warp_idx <= cfg.correction_warp_ids[-1]
        )
        if is_softmax_warp or is_correction_warp:
            prims.mbarrier_arrive(tmem_dealloc_mbar)

        if warp_idx == cfg.mma_warp_id:
            while not prims.mbarrier_try_wait_parity(tmem_dealloc_mbar, 0):
                pass
            tmem_alloc_cols = Int32(cfg.tmem_alloc_cols)
            tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)
            prims.tcgen05_dealloc(tmem_ptr, tmem_alloc_cols)
