# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused fc1 + fc2 MegaMoE scheduler."""

from enum import IntEnum
from typing import List, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (Boolean, Int32, Integer, const_expr,
                                 dsl_user_op, extract_mlir_values,
                                 new_from_mlir_values)

try:
    from cutlass.cute import iket  # type: ignore
except (ImportError, NotImplementedError):  # pragma: no cover
    from .iket_compat import iket

from .moe_persistent_scheduler import (_DEFAULT_SCHED_EXT, MoESchedulerBase,
                                       MoESchedulerParamsBase, MoEWorkTileInfo,
                                       WorkTileState)
from .moe_utils import (compute_expert_token_count_from_sizes,
                        compute_expert_token_range,
                        mbarrier_arrive_expect_tx_on_peer,
                        store_i32_to_peer_cluster_smem_async)

# =============================================================================
# Block phase
# =============================================================================


class BlockPhase(IntEnum):
    """Fused fc1+fc2 work-tile phase. ``None_`` reserved as sentinel for sched
    invalid tiles (alongside ``WorkTileState.DONE``)."""

    None_ = 0
    Linear1 = 1
    Linear2 = 2


# =============================================================================
# Persistent state objects
# =============================================================================


class _FusedFc12SchedState:
    """Sched warp register-resident state for the (group, phase, expert) state
    machine.  Field set kept flat so MLIR serialization is one extend per
    field; nested sub-states would not save anything in this footprint."""

    def __init__(
        self,
        current_group_idx: Int32,
        current_group_first_expert: Int32,
        current_group_last_expert_exclusive: Int32,
        current_phase: Int32,
        current_expert_idx: Int32,
        current_expert_tile_start: Int32,
        current_expert_tile_end: Int32,
        current_group_fc1_subphase_end: Int32,
        current_group_end: Int32,
        cumulative_fc1_tiles_at_group_end: Int32,
        cumulative_fc2_tiles_at_group_end: Int32,
        # Physical-row / token-block running cumulatives.  Each invariant is
        # "(...)_cumul reflects the
        # current expert's *start* offset under that padding granularity":
        #   current_data_cumul        = sum_{e' < current_expert_idx}
        #                                   round_up(valid_e', params.token_padding_block)
        #   current_sf_cumul          = sum_{e' < current_expert_idx}
        #                                   round_up(valid_e', params.sf_padding_block)
        #   current_token_block_cumul = sum_{e' < current_expert_idx}
        #                                   ceil_div(valid_e', cluster_tile_m)
        # Each ``advance_expert_within_phase`` pushes the *previous* expert's
        # occupation into these before bumping ``current_expert_idx``.
        current_data_cumul: Int32,
        current_sf_cumul: Int32,
        current_token_block_cumul: Int32,
        # Group-start checkpoints used by ``switch_to_fc2`` to rewind cumul
        # state from group-end (fc1 phase's last expert) back to group-start
        # (fc2 phase will re-walk the same experts).  Captured at
        # ``advance_group`` time *after* pushing the previous group's tail.
        group_start_data_cumul: Int32,
        group_start_sf_cumul: Int32,
        group_start_token_block_cumul: Int32,
        current_token_block_count: Int32,
        current_token_offset: Int32,
        current_this_expert_token_cnt: Int32,
        current_work_linear_tile_idx: Int32,
    ):
        self.current_group_idx = current_group_idx
        self.current_group_first_expert = current_group_first_expert
        self.current_group_last_expert_exclusive = current_group_last_expert_exclusive
        self.current_phase = current_phase
        self.current_expert_idx = current_expert_idx
        self.current_expert_tile_start = current_expert_tile_start
        self.current_expert_tile_end = current_expert_tile_end
        self.current_group_fc1_subphase_end = current_group_fc1_subphase_end
        self.current_group_end = current_group_end
        self.cumulative_fc1_tiles_at_group_end = cumulative_fc1_tiles_at_group_end
        self.cumulative_fc2_tiles_at_group_end = cumulative_fc2_tiles_at_group_end
        self.current_data_cumul = current_data_cumul
        self.current_sf_cumul = current_sf_cumul
        self.current_token_block_cumul = current_token_block_cumul
        self.group_start_data_cumul = group_start_data_cumul
        self.group_start_sf_cumul = group_start_sf_cumul
        self.group_start_token_block_cumul = group_start_token_block_cumul
        self.current_token_block_count = current_token_block_count
        self.current_token_offset = current_token_offset
        self.current_this_expert_token_cnt = current_this_expert_token_cnt
        self.current_work_linear_tile_idx = current_work_linear_tile_idx

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.current_group_idx))
        values.extend(extract_mlir_values(self.current_group_first_expert))
        values.extend(
            extract_mlir_values(self.current_group_last_expert_exclusive))
        values.extend(extract_mlir_values(self.current_phase))
        values.extend(extract_mlir_values(self.current_expert_idx))
        values.extend(extract_mlir_values(self.current_expert_tile_start))
        values.extend(extract_mlir_values(self.current_expert_tile_end))
        values.extend(extract_mlir_values(self.current_group_fc1_subphase_end))
        values.extend(extract_mlir_values(self.current_group_end))
        values.extend(
            extract_mlir_values(self.cumulative_fc1_tiles_at_group_end))
        values.extend(
            extract_mlir_values(self.cumulative_fc2_tiles_at_group_end))
        values.extend(extract_mlir_values(self.current_data_cumul))
        values.extend(extract_mlir_values(self.current_sf_cumul))
        values.extend(extract_mlir_values(self.current_token_block_cumul))
        values.extend(extract_mlir_values(self.group_start_data_cumul))
        values.extend(extract_mlir_values(self.group_start_sf_cumul))
        values.extend(extract_mlir_values(self.group_start_token_block_cumul))
        values.extend(extract_mlir_values(self.current_token_block_count))
        values.extend(extract_mlir_values(self.current_token_offset))
        values.extend(extract_mlir_values(self.current_this_expert_token_cnt))
        values.extend(extract_mlir_values(self.current_work_linear_tile_idx))
        return values

    def __new_from_mlir_values__(
            self, values: List[ir.Value]) -> "_FusedFc12SchedState":
        idx = 0

        def _take(obj):
            nonlocal idx
            n = len(extract_mlir_values(obj))
            result = new_from_mlir_values(obj, values[idx:idx + n])
            idx += n
            return result

        return _FusedFc12SchedState(
            current_group_idx=_take(self.current_group_idx),
            current_group_first_expert=_take(self.current_group_first_expert),
            current_group_last_expert_exclusive=_take(
                self.current_group_last_expert_exclusive),
            current_phase=_take(self.current_phase),
            current_expert_idx=_take(self.current_expert_idx),
            current_expert_tile_start=_take(self.current_expert_tile_start),
            current_expert_tile_end=_take(self.current_expert_tile_end),
            current_group_fc1_subphase_end=_take(
                self.current_group_fc1_subphase_end),
            current_group_end=_take(self.current_group_end),
            cumulative_fc1_tiles_at_group_end=_take(
                self.cumulative_fc1_tiles_at_group_end),
            cumulative_fc2_tiles_at_group_end=_take(
                self.cumulative_fc2_tiles_at_group_end),
            current_data_cumul=_take(self.current_data_cumul),
            current_sf_cumul=_take(self.current_sf_cumul),
            current_token_block_cumul=_take(self.current_token_block_cumul),
            group_start_data_cumul=_take(self.group_start_data_cumul),
            group_start_sf_cumul=_take(self.group_start_sf_cumul),
            group_start_token_block_cumul=_take(
                self.group_start_token_block_cumul),
            current_token_block_count=_take(self.current_token_block_count),
            current_token_offset=_take(self.current_token_offset),
            current_this_expert_token_cnt=_take(
                self.current_this_expert_token_cnt),
            current_work_linear_tile_idx=_take(
                self.current_work_linear_tile_idx),
        )


class _DynamicLoadBalanceState:
    """Atomic-counter load-balance state.  Set on the scheduler only when
    ``params.load_balance_mode == 'atomic_counter'``.  ``atomic_res`` caches
    the first pre-init claim so the first advance site does not issue another
    atom.add.
    """

    def __init__(
        self,
        counter_ptr,
        broadcast_ptr,
        is_leader_cta: Boolean,
        producer_state,
        consumer_state,
        atomic_res: Int32,
    ):
        self.counter_ptr = counter_ptr
        self.broadcast_ptr = broadcast_ptr
        self.is_leader_cta = is_leader_cta
        self.producer_state = producer_state
        self.consumer_state = consumer_state
        self.atomic_res = atomic_res

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.counter_ptr))
        values.extend(extract_mlir_values(self.broadcast_ptr))
        values.extend(extract_mlir_values(self.is_leader_cta))
        values.extend(extract_mlir_values(self.producer_state))
        values.extend(extract_mlir_values(self.consumer_state))
        values.extend(extract_mlir_values(self.atomic_res))
        return values

    def __new_from_mlir_values__(
            self, values: List[ir.Value]) -> "_DynamicLoadBalanceState":
        idx = 0

        def _take(obj):
            nonlocal idx
            n = len(extract_mlir_values(obj))
            result = new_from_mlir_values(obj, values[idx:idx + n])
            idx += n
            return result

        return _DynamicLoadBalanceState(
            counter_ptr=_take(self.counter_ptr),
            broadcast_ptr=_take(self.broadcast_ptr),
            is_leader_cta=_take(self.is_leader_cta),
            producer_state=_take(self.producer_state),
            consumer_state=_take(self.consumer_state),
            atomic_res=_take(self.atomic_res),
        )


# =============================================================================
# Scheduler Parameters
# =============================================================================


class MoEFusedFc12SchedulerParams(MoESchedulerParamsBase):
    """Codegen-time + runtime parameters for the fused fc1+fc2 mega scheduler.

    Inherits ``expert_shape``, ``cta_tile_shape_mnk``, ``cluster_shape_mn``,
    ``scenario``, ``is_swap_ab``, ``num_sched_stages`` handling from
    ``MoESchedulerParamsBase``.  ``cta_tile_shape_mnk`` is shared by fc1 / fc2
    (v1 simplification).

    This params type currently backs the inference fc12 path.  In that path
    ``expert_shape[1]`` is ``intermediate_gateup`` (gate + up concatenated).
    Future training/non-swap MegaMoE variants should add their own params
    contract instead of overloading this one.
    """

    def __init__(
        self,
        scenario: Literal["2Dx3D"],
        expert_shape: Tuple[int | Int32, int | Int32, int | Int32],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: Literal["static", "atomic_counter",
                                   "clc"] = "static",
        load_balance_counter_ptr=None,
        override_num_stages: Optional[int] = None,
        is_swap_ab: bool = True,
        # Exactly one of the next two must be non-None (sizes preferred when
        # the host can expose a direct view onto ``expert_recv_count_sum``;
        # prefix_sum required when only a host-precomputed cumsum is
        # available).  The scheduler picks the data source at codegen time
        # via ``cutlass.const_expr(self.expert_token_sizes is not None)``;
        # serialization is type-discriminated below.
        expert_token_sizes: Optional[cute.Tensor] = None,
        expert_token_prefix_sum: Optional[cute.Tensor] = None,
    ):
        """Create fused fc12 scheduler params."""
        if scenario != "2Dx3D":
            raise ValueError(
                f"fused fc1+fc2 only supports 2Dx3D, got {scenario!r}")
        if load_balance_mode not in ("static", "atomic_counter", "clc"):
            raise ValueError(
                f"load_balance_mode must be one of 'static' / 'atomic_counter' / 'clc', "
                f"got {load_balance_mode!r}")
        if load_balance_mode == "atomic_counter" and load_balance_counter_ptr is None:
            raise ValueError(
                "load_balance_counter_ptr must be provided when load_balance_mode == "
                "'atomic_counter' (GMEM int32 ptr, host-allocated and zero-init per launch)"
            )
        if group_hint <= 0:
            raise ValueError(f"group_hint must be positive, got {group_hint}")
        if token_padding_block <= 0:
            raise ValueError(
                f"token_padding_block must be positive, got {token_padding_block}"
            )
        if sf_padding_block <= 0:
            raise ValueError(
                f"sf_padding_block must be positive, got {sf_padding_block}")
        if (expert_token_sizes is None) == (expert_token_prefix_sum is None):
            raise ValueError(
                "Exactly one of expert_token_sizes / expert_token_prefix_sum "
                "must be provided (got "
                f"sizes={'set' if expert_token_sizes is not None else 'None'}, "
                f"prefix_sum={'set' if expert_token_prefix_sum is not None else 'None'})."
            )

        super().__init__(
            scenario=scenario,
            expert_shape=expert_shape,
            cta_tile_shape_mnk=cta_tile_shape_mnk,
            cluster_shape_mn=cluster_shape_mn,
            override_num_stages=override_num_stages,
            is_swap_ab=is_swap_ab,
        )
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode
        self.load_balance_counter_ptr = load_balance_counter_ptr
        self.expert_token_sizes = expert_token_sizes
        self.expert_token_prefix_sum = expert_token_prefix_sum

    def get_scheduler_type(self) -> type:
        return MoEFusedFc12PersistentTileScheduler

    def get_grid_shape(
        self,
        max_active_clusters: int,
    ) -> Tuple[int, int, int]:
        if self.is_swap_ab:
            return (
                self.cluster_shape_mn[1],
                self.cluster_shape_mn[0],
                max_active_clusters,
            )
        return (
            self.cluster_shape_mn[0],
            self.cluster_shape_mn[1],
            max_active_clusters,
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        """Type-discriminated serialization (see ``MoEStaticSchedulerParams``).

        Python int fields supplied via ``static_expert_shape`` skip the
        MLIR carry and remain inlined codegen-time literals; Int32
        fields (the dynamic ``fc1_weight.shape`` path) flow through as
        SSA values as usual.

        Exactly one of ``expert_token_sizes`` / ``expert_token_prefix_sum``
        is non-None (enforced in ``__init__``); whichever it is gets
        extended.  ``__new_from_mlir_values__`` reads the same prototype
        ``self`` to decide which side to consume.
        """
        values = []
        if isinstance(self.expert_cnt, Int32):
            values.extend(extract_mlir_values(self.expert_cnt))
        if isinstance(self.intermediate, Int32):
            values.extend(extract_mlir_values(self.intermediate))
        if isinstance(self.hidden, Int32):
            values.extend(extract_mlir_values(self.hidden))
        if self.load_balance_mode == "atomic_counter":
            values.extend(extract_mlir_values(self.load_balance_counter_ptr))
        if self.expert_token_sizes is not None:
            values.extend(extract_mlir_values(self.expert_token_sizes))
        else:
            values.extend(extract_mlir_values(self.expert_token_prefix_sum))
        return values

    def __new_from_mlir_values__(
            self, values: List[ir.Value]) -> "MoEFusedFc12SchedulerParams":
        # Bypass __init__: stored cta_tile_shape_mnk / cluster_shape_mn are
        # already in post-swap form, going through __init__ would double-swap.
        # Mirrors MoEStaticSchedulerParams.__new_from_mlir_values__.
        result = MoEFusedFc12SchedulerParams.__new__(
            MoEFusedFc12SchedulerParams)
        result.scenario = self.scenario
        result.is_swap_ab = self.is_swap_ab
        result.cta_tile_shape_mnk = self.cta_tile_shape_mnk
        result.cluster_shape_mn = self.cluster_shape_mn
        result.num_sched_stages = self.num_sched_stages
        result.group_hint = self.group_hint
        result.token_padding_block = self.token_padding_block
        result.sf_padding_block = self.sf_padding_block
        result.load_balance_mode = self.load_balance_mode

        # Type-discriminated rebind: Python int fields copy from
        # prototype (``self``), Int32 fields consume from ``values``.
        idx = 0
        if isinstance(self.expert_cnt, Int32):
            result.expert_cnt = new_from_mlir_values(self.expert_cnt,
                                                     [values[idx]])
            idx += 1
        else:
            result.expert_cnt = self.expert_cnt
        if isinstance(self.intermediate, Int32):
            result.intermediate = new_from_mlir_values(self.intermediate,
                                                       [values[idx]])
            idx += 1
        else:
            result.intermediate = self.intermediate
        if isinstance(self.hidden, Int32):
            result.hidden = new_from_mlir_values(self.hidden, [values[idx]])
            idx += 1
        else:
            result.hidden = self.hidden
        if self.load_balance_mode == "atomic_counter":
            ptr_len = len(extract_mlir_values(self.load_balance_counter_ptr))
            result.load_balance_counter_ptr = new_from_mlir_values(
                self.load_balance_counter_ptr, values[idx:idx + ptr_len])
            idx += ptr_len
        else:
            result.load_balance_counter_ptr = None
        # Sizes / prefix_sum: prototype tells us which side carries the
        # actual tensor; the other side stays None on the result.
        if self.expert_token_sizes is not None:
            t_len = len(extract_mlir_values(self.expert_token_sizes))
            result.expert_token_sizes = new_from_mlir_values(
                self.expert_token_sizes, values[idx:idx + t_len])
            idx += t_len
            result.expert_token_prefix_sum = None
        else:
            t_len = len(extract_mlir_values(self.expert_token_prefix_sum))
            result.expert_token_prefix_sum = new_from_mlir_values(
                self.expert_token_prefix_sum, values[idx:idx + t_len])
            idx += t_len
            result.expert_token_sizes = None
        assert idx == len(values), (
            f"Fused fc12 sched params type-discrim mismatch: idx={idx} "
            f"len(values)={len(values)}")
        return result


# =============================================================================
# Scheduler — Fused fc1 + fc2 Persistent (Device-side)
# =============================================================================


class MoEFusedFc12PersistentTileScheduler(MoESchedulerBase):
    """Mega scheduler for fused fc1+fc2 grouped GEMM under swap-AB.

    Tile space: ``(group, phase, expert, token_block, intermediate_or_hidden_block)``.
    Within a group: full fc1 sub-segment (all experts in expert order, each expert
    expanded as ``token_block`` slow / ``intermediate_block`` fast) → full fc2
    sub-segment (same expert order, each expert expanded short-side-first).
    """

    def __init__(
        self,
        params: MoEFusedFc12SchedulerParams,
        num_persistent_clusters: Int32,
        cta_id_in_cluster: cute.Coord,
        current_work: MoEWorkTileInfo,
        fused_state: _FusedFc12SchedState,
        dynamic_state: Optional[_DynamicLoadBalanceState],
        # Cached scheduler-wide derived constants (computed once in create()
        # from params.intermediate / params.hidden / params.cluster_tile_n;
        # avoid recomputing in the hot path of advance / decode).
        num_fc1_intermediate_blocks: Int32,
        num_fc2_hidden_blocks: Int32,
        ext,
        sched_pipeline,
        smem_buf_tensor,
        num_sched_stages: int,
        cluster_pipeline,
        producer_state,
        sched_storage=None,
    ):
        # Per-expert token range data source lives on ``params``: either
        # ``params.expert_token_sizes`` (sizes-mode, e.g. zero-copy view of
        # ``expert_recv_count_sum``) or ``params.expert_token_prefix_sum``
        # (cumulative-end, host-precomputed).  See
        # ``compute_expert_token_range`` / ``compute_expert_token_count_from_sizes``
        # in ``moe_utils.py`` for the per-mode helpers.
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self.cta_id_in_cluster = cta_id_in_cluster
        self.current_work = current_work
        self._fused_state = fused_state
        self._dynamic_state = dynamic_state
        self._num_fc1_intermediate_blocks = num_fc1_intermediate_blocks
        self._num_fc2_hidden_blocks = num_fc2_hidden_blocks
        self._ext = ext
        self._pipeline = sched_pipeline
        self._smem_buf_tensor = smem_buf_tensor
        self._num_sched_stages = num_sched_stages
        self._cluster_pipeline = cluster_pipeline
        self._producer_state = producer_state
        # Python-only reference to the SMEM scheduler storage struct.  Held
        # only so that ``internal_init`` can reach
        # ``sched_storage.cluster_pipeline_mbar`` /
        # ``sched_storage.cluster_broadcast_slot`` to build the
        # cluster_pipeline (atomic_counter mode); does NOT serialize.
        self._sched_storage = sched_storage
        # Codegen-time Python attribute (NOT MLIR-serialized).  Set to True
        # by ``internal_init`` to mark "scheduler state has been greedily
        # advanced one step (atomic_add cached for atomic_counter mode /
        # current_work decoded for static mode)".  ``gen_next_work``'s
        # ``cutlass.const_expr(self._first_advance_pending)`` branch reads
        # this at trace time to elide the corresponding work for the first
        # call site, then sets it back to False so the second trace site
        # (while-body) compiles the vanilla path.
        self._first_advance_pending: bool = False

    @staticmethod
    def make_storage_struct(
        params: MoEFusedFc12SchedulerParams,
        ext=_DEFAULT_SCHED_EXT,
        **kwargs,
    ) -> type:
        if params.load_balance_mode == "clc":
            raise NotImplementedError(
                "load_balance_mode='clc' is reserved; CLC path is in"
                " MoEDynamicPersistentTileScheduler, not the mega scheduler")

        num_tile_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields

        @cute.struct
        class StaticSchedulerStorage:
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_tile_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32,
                                     fields_per_stage * num_tile_stages],
                16,
            ]

        @cute.struct
        class AtomicCounterSchedulerStorage:
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_tile_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32,
                                     fields_per_stage * num_tile_stages],
                16,
            ]
            cluster_pipeline_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            cluster_broadcast_slot: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 1],
                16,
            ]

        if params.load_balance_mode == "atomic_counter":
            return AtomicCounterSchedulerStorage
        return StaticSchedulerStorage

    @staticmethod
    @dsl_user_op
    def create(
        params: MoEFusedFc12SchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        sched_storage,
        num_consumer_threads: int,
        ext=_DEFAULT_SCHED_EXT,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "MoEFusedFc12PersistentTileScheduler":
        if num_consumer_threads <= 0:
            raise ValueError(
                f"num_consumer_threads must be positive, got {num_consumer_threads}"
            )
        if params.load_balance_mode == "clc":
            raise NotImplementedError(
                "load_balance_mode='clc' is reserved; CLC path is in"
                " MoEDynamicPersistentTileScheduler, not the mega scheduler")

        num_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields

        num_persistent_clusters = cute.size(
            grid_dim, loc=loc, ip=ip) // cute.size(
                params.cluster_shape_mn, loc=loc, ip=ip)

        bidx, bidy, bidz = block_idx

        # ``params.cluster_shape_mn`` is scheduler-internal.  Under swap-AB,
        # launch axes map to the opposite internal M/N slots.
        if const_expr(params.is_swap_ab):
            cta_id_in_cluster = (
                Int32(bidy % params.cluster_shape_mn[0]),
                Int32(bidx % params.cluster_shape_mn[1]),
                Int32(0),
            )
        else:
            cta_id_in_cluster = (
                Int32(bidx % params.cluster_shape_mn[0]),
                Int32(bidy % params.cluster_shape_mn[1]),
                Int32(0),
            )

        # State machine sentinel init.  The 0 values for current_group_end /
        # current_expert_tile_end force gen_next_work's first call to enter
        # advance_group() and advance_expert_within_phase(), which then fill
        # the rest of the state from offs.  current_group_idx / current_expert_idx
        # = -1 so that advance_* increments cleanly to 0 on first call.
        #
        # All cumul fields (current_*_cumul / group_start_*_cumul) start at 0;
        # current_this_expert_token_cnt and current_token_block_count also start
        # at 0 so that the first advance_expert call inside the first
        # advance_group call pushes a no-op (round_up(0, ...) = 0) into the
        # cumul state before reading expert 0's valid count.
        fused_state = _FusedFc12SchedState(
            current_group_idx=Int32(-1),
            current_group_first_expert=Int32(0),
            current_group_last_expert_exclusive=Int32(0),
            current_phase=Int32(BlockPhase.Linear1),
            current_expert_idx=Int32(-1),
            current_expert_tile_start=Int32(0),
            current_expert_tile_end=Int32(0),
            current_group_fc1_subphase_end=Int32(0),
            current_group_end=Int32(0),
            cumulative_fc1_tiles_at_group_end=Int32(0),
            cumulative_fc2_tiles_at_group_end=Int32(0),
            current_data_cumul=Int32(0),
            current_sf_cumul=Int32(0),
            current_token_block_cumul=Int32(0),
            group_start_data_cumul=Int32(0),
            group_start_sf_cumul=Int32(0),
            group_start_token_block_cumul=Int32(0),
            current_token_block_count=Int32(0),
            current_token_offset=Int32(0),
            current_this_expert_token_cnt=Int32(0),
            current_work_linear_tile_idx=Int32(bidz),
        )

        # Cached derived constants (scheduler-wide, computed once).
        #
        # ``params.intermediate`` carries ``intermediate_gateup`` semantics
        # (= ``mat_b.shape[2]``, the full gate+up-concat fc1 weight dim;
        # see ``MoESchedulerParamsBase.__init__`` docstring).  fc1 GEMM-M
        # under swap-AB IS that gateup axis, so the number of cluster work
        # tiles along intermediate per ``(expert, token_block)`` is just
        # ``ceil_div(intermediate_gateup, cluster_tile_n_post_swap)`` --
        # the same formula ``MoEStaticPersistentTileScheduler._get_cluster_
        # tile_counts`` uses for its swap-AB 2Dx3D N-axis count.  A prior
        # ``2 *`` multiplier here was a bug that doubled the per-tile-block
        # work tile count (assumed ``params.intermediate`` was the half-dim
        # ``intermediate_downproj``); removed to match the base scheduler.
        #
        # ``params.hidden`` is single-semantic (fc2 GEMM-M / fc2 output cols).
        intermediate_gateup = params.intermediate
        hidden = params.hidden
        num_fc1_intermediate_blocks = (intermediate_gateup +
                                       params.cluster_tile_n -
                                       1) // params.cluster_tile_n
        num_fc2_hidden_blocks = (hidden + params.cluster_tile_n -
                                 1) // params.cluster_tile_n

        # current_work init must use ext.WorkTileInfo (8 fields) to match the
        # shape that gen_next_work writes; otherwise MLIR serialization slot
        # Scheduler emits the final 8-field work tile directly.
        current_work = ext.WorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            cumulative_data_physical_row=Int32(0),
            cumulative_sf_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_tile=Int32(0),
            phase_and_peek=Int32(BlockPhase.None_),
        )

        sched_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 32)
        sched_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_consumer_threads)
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=num_stages,
            producer_group=sched_producer_group,
            consumer_group=sched_consumer_group,
            barrier_storage=sched_storage.sched_mbar.data_ptr(),
            defer_sync=True,
        )
        smem_buf_tensor = cute.make_tensor(
            sched_storage.sched_buf.data_ptr(),
            cute.make_layout(
                (fields_per_stage, num_stages),
                stride=(1, fields_per_stage),
            ),
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_stages)

        # Atomic-counter dynamic load-balance state.  cluster_pipeline is
        # left as None here and constructed lazily inside ``internal_init``
        # (it must be created BEFORE ``pipeline_init_arrive`` so all warps
        # participate in the cluster mbarrier init; ``create`` runs from
        # the kernel prologue, which already satisfies that).  ``atomic_res``
        # starts at 0; ``internal_init`` overwrites it with the first claimed
        # cluster-linear tile id.
        dynamic_state: Optional[_DynamicLoadBalanceState] = None
        if const_expr(params.load_balance_mode == "atomic_counter"):
            is_leader_cta = (cta_id_in_cluster[0] + cta_id_in_cluster[1] +
                             cta_id_in_cluster[2]) == Int32(0)
            cluster_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            cluster_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1)
            dynamic_state = _DynamicLoadBalanceState(
                counter_ptr=params.load_balance_counter_ptr,
                broadcast_ptr=sched_storage.cluster_broadcast_slot.data_ptr(),
                is_leader_cta=is_leader_cta,
                producer_state=cluster_producer_state,
                consumer_state=cluster_consumer_state,
                atomic_res=Int32(0),
            )

        return MoEFusedFc12PersistentTileScheduler(
            params=params,
            num_persistent_clusters=num_persistent_clusters,
            cta_id_in_cluster=cta_id_in_cluster,
            current_work=current_work,
            fused_state=fused_state,
            dynamic_state=dynamic_state,
            num_fc1_intermediate_blocks=num_fc1_intermediate_blocks,
            num_fc2_hidden_blocks=num_fc2_hidden_blocks,
            ext=ext,
            sched_pipeline=sched_pipeline,
            smem_buf_tensor=smem_buf_tensor,
            num_sched_stages=num_stages,
            cluster_pipeline=None,
            producer_state=producer_state,
            sched_storage=sched_storage,
        )

    # -------------------------------------------------------------------------
    # internal_init: first-tile pre-init before pipeline_init_arrive
    # -------------------------------------------------------------------------

    @dsl_user_op
    @cute.jit
    def internal_init(
        self,
        warp_idx,
        sched_warp_id: int,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Claim/decode the first work tile during kernel prologue."""
        if const_expr(self.params.load_balance_mode == "atomic_counter"):
            cluster_size = (self.params.cluster_shape_mn[0] *
                            self.params.cluster_shape_mn[1])

            # Cluster-wide broadcast pipeline for the leader CTA's atom.add.
            self._cluster_pipeline = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * cluster_size),
                barrier_storage=self._sched_storage.cluster_pipeline_mbar.
                data_ptr(),
                defer_sync=True,
            )

            # Pre-claim and broadcast first dynamic tile id.
            if warp_idx == sched_warp_id:
                tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
                atomic_res = Int32(0)
                if (self._dynamic_state.is_leader_cta
                        and tidx % 32 == Int32(0)):
                    atomic_res = cute.arch.atomic_add(
                        self._dynamic_state.counter_ptr,
                        Int32(1),
                        loc=loc,
                        ip=ip,
                    )
                atomic_res = cute.arch.shuffle_sync(
                    atomic_res,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
                self._dynamic_state.atomic_res = atomic_res
                self._dynamic_state = self._dynamic_state  # DSL carry
            else:
                self._dynamic_state = self._dynamic_state  # balance scf.if yield
        elif const_expr(self.params.load_balance_mode == "static"):
            # Static mode eagerly decodes the first tile.
            if warp_idx == sched_warp_id:
                cluster_linear_tile_idx = (
                    self._advance_work_linear_tile_idx_static(loc=loc, ip=ip))
                self._gen_work_from_cluster_idx(cluster_linear_tile_idx,
                                                loc=loc,
                                                ip=ip)
                self._fused_state = self._fused_state  # DSL carry
                self.current_work = self.current_work
            else:
                self._fused_state = self._fused_state  # balance scf.if yield
                self.current_work = self.current_work
        else:
            raise NotImplementedError(
                "load_balance_mode='clc' is reserved; CLC scheduler is "
                "MoEDynamicPersistentTileScheduler, not the mega scheduler")

        # Codegen-time signal: gen_next_work's first trace site sees
        # this True and emits the first-tile-finalize path; second trace
        # site (while-body) sees False and emits the vanilla path.
        self._first_advance_pending = True

    # -------------------------------------------------------------------------
    # State-machine advance helpers (group → phase → expert)
    # -------------------------------------------------------------------------

    @dsl_user_op
    @cute.jit
    def _advance_work_linear_tile_idx_static(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """Static stride mode: read current_work_linear_tile_idx, advance by
        num_persistent_clusters for next iteration, return the read value."""
        state = self._fused_state
        cluster_linear_tile_idx = state.current_work_linear_tile_idx
        state.current_work_linear_tile_idx = (cluster_linear_tile_idx +
                                              self.num_persistent_clusters)
        return cluster_linear_tile_idx

    @dsl_user_op
    @cute.jit
    def _advance_work_linear_tile_idx_dynamic(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """Atomic-counter mode: returns the cluster-linear tile idx broadcast
        to all CTAs in the cluster.

        Internally const_expr-forks on ``self._first_advance_pending``
        (codegen-time Python bool):

          - True (first trace site, post-``internal_init``): use the cached
            ``self._dynamic_state.atomic_res`` ``atom.add`` result; skip
            issuing a fresh atomic.  The cached value was computed and
            shuffled across the sched warp by ``internal_init`` BEFORE
            ``pipeline_init_arrive`` so its memory round trip overlaps with
            cluster init wait.
          - False (vanilla, second-and-onward trace sites): leader CTA
            lane 0 issues a fresh ``atom.global.add.s32`` and shuffles the
            result across the sched warp.

        Cluster-internal protocol (DSMEM broadcast + cluster_pipeline mbar
        wait) is identical between the two paths.  Mirrors
        ``cute_dsl_kernel_library/dsl_kernels/moe/moe_persistent_scheduler.py``
        ``_fetch_next_cluster_idx`` (lines 838-894).
        """
        ds = self._dynamic_state
        cluster_pipeline = self._cluster_pipeline
        broadcast_tensor = cute.make_tensor(ds.broadcast_ptr,
                                            cute.make_layout((1, )))
        cluster_size = (self.params.cluster_shape_mn[0] *
                        self.params.cluster_shape_mn[1])

        # --- Producer side (leader CTA only) ---
        if ds.is_leader_cta:
            cluster_pipeline.producer_acquire(ds.producer_state)
            full_barrier_ptr = cluster_pipeline.sync_object_full.get_barrier(
                ds.producer_state.index, loc=loc, ip=ip)
            tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
            lane_idx = tidx % Int32(32)

            if cutlass.const_expr(self._first_advance_pending):
                # First-tile path: consume the cached atomic_res that
                # internal_init shuffled across the sched warp.
                atomic_idx = ds.atomic_res
            else:
                # Vanilla path: lane 0 atom.add, shuffle to all lanes.
                atomic_idx = Int32(0)
                if lane_idx == Int32(0):
                    atomic_idx = cute.arch.atomic_add(
                        ds.counter_ptr,
                        Int32(1),
                        loc=loc,
                        ip=ip,
                    )
                atomic_idx = cute.arch.shuffle_sync(
                    atomic_idx,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )

            # DSMEM fan-out: lanes [0, cluster_size) each write to one peer
            # CTA.  Each lane targets a distinct peer (lane_idx == peer rank).
            if lane_idx < Int32(cluster_size):
                store_i32_to_peer_cluster_smem_async(
                    ds.broadcast_ptr,
                    atomic_idx,
                    full_barrier_ptr,
                    lane_idx,
                    loc=loc,
                    ip=ip,
                )
                # Set expect_tx on the peer mbarrier to match the 4-byte
                # store above; pairs with the consumer_wait below.
                mbarrier_arrive_expect_tx_on_peer(
                    full_barrier_ptr,
                    Int32(4),
                    lane_idx,
                    loc=loc,
                    ip=ip,
                )
        ds.producer_state.advance()

        # --- Consumer side (all CTAs sched warp threads) ---
        cluster_pipeline.consumer_wait(ds.consumer_state)
        cluster_idx = broadcast_tensor[0]
        cute.arch.fence_acq_rel_cta()
        cluster_pipeline.sync_object_empty.arrive(ds.consumer_state.index,
                                                  Int32(0))
        ds.consumer_state.advance()

        return cluster_idx

    @dsl_user_op
    @cute.jit
    def _advance_expert_within_phase(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Advance to the next expert in the current phase."""
        state = self._fused_state
        params = self.params
        cluster_tile_m = params.cluster_tile_m

        # Push previous expert into cumul state before bumping expert_idx.
        token_padding = params.token_padding_block
        sf_padding = params.sf_padding_block
        prev_valid = state.current_this_expert_token_cnt
        state.current_data_cumul = state.current_data_cumul + (
            (prev_valid + Int32(token_padding - 1)) //
            Int32(token_padding)) * Int32(token_padding)
        state.current_sf_cumul = state.current_sf_cumul + (
            (prev_valid + Int32(sf_padding - 1)) //
            Int32(sf_padding)) * Int32(sf_padding)
        state.current_token_block_cumul = (state.current_token_block_cumul +
                                           state.current_token_block_count)

        # Refresh current expert token range.
        #
        # Two data-source modes (selected at codegen time by which side of
        # the params Optional pair is non-None):
        #   - prefix-sum mode: random-access ``offs[i] - offs[i-1]`` gives
        #     both offset and count in O(1).
        #   - sizes mode: ``sizes[i]`` gives only the count; the cumulative
        #     offset is maintained as a running cumul on
        #     ``state.current_token_offset`` (push prev_valid before bumping
        #     expert_idx).  This works because ``_advance_expert_within_phase``
        #     is always called in monotonically-increasing expert order
        #     (every group advance walks through residual experts of the
        #     finishing group first, so no random jumps).
        state.current_expert_idx = state.current_expert_idx + Int32(1)
        if cutlass.const_expr(self.params.expert_token_sizes is not None):
            state.current_token_offset = (state.current_token_offset +
                                          prev_valid)
            this_expert_token_cnt = compute_expert_token_count_from_sizes(
                self.params.expert_token_sizes,
                state.current_expert_idx,
                loc=loc,
                ip=ip,
            )
        else:
            token_offset, this_expert_token_cnt = compute_expert_token_range(
                self.params.expert_token_prefix_sum,
                state.current_expert_idx,
                loc=loc,
                ip=ip,
            )
            state.current_token_offset = token_offset
        state.current_this_expert_token_cnt = this_expert_token_cnt
        state.current_token_block_count = (this_expert_token_cnt +
                                           Int32(cluster_tile_m) -
                                           1) // Int32(cluster_tile_m)

        # --- Step 3: slide expert_tile_start / expert_tile_end.
        state.current_expert_tile_start = state.current_expert_tile_end
        # Prebind due to DSL AST.
        tiles_in_expert = Int32(0)
        if state.current_phase == Int32(BlockPhase.Linear1):
            tiles_in_expert = (state.current_token_block_count *
                               self._num_fc1_intermediate_blocks)
        else:
            tiles_in_expert = (state.current_token_block_count *
                               self._num_fc2_hidden_blocks)
        state.current_expert_tile_end = (state.current_expert_tile_start +
                                         tiles_in_expert)

    @dsl_user_op
    @cute.jit
    def _switch_to_fc2(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Switch current group from Linear1 to Linear2."""
        state = self._fused_state
        state.current_phase = Int32(BlockPhase.Linear2)
        state.current_expert_idx = state.current_group_first_expert - Int32(1)
        state.current_expert_tile_end = state.current_group_fc1_subphase_end
        # Zero previous-expert cache before rewinding to group start.
        state.current_this_expert_token_cnt = Int32(0)
        state.current_token_block_count = Int32(0)
        state.current_data_cumul = state.group_start_data_cumul
        state.current_sf_cumul = state.group_start_sf_cumul
        state.current_token_block_cumul = state.group_start_token_block_cumul
        self._advance_expert_within_phase(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def _advance_group(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Open the next group and prime its first Linear1 expert."""
        state = self._fused_state
        params = self.params
        cluster_tile_m = params.cluster_tile_m

        # Push residual experts from the just-finished group into cumul state.
        residual_expert_idx = state.current_expert_idx
        residual_group_last_expert_exclusive = state.current_group_last_expert_exclusive
        while residual_expert_idx + Int32(
                1) < residual_group_last_expert_exclusive:
            self._advance_expert_within_phase(loc=loc, ip=ip)
            self._fused_state = self._fused_state
            state = self._fused_state
            residual_expert_idx = state.current_expert_idx
            residual_group_last_expert_exclusive = (
                state.current_group_last_expert_exclusive)
        state = self._fused_state

        # Final push; cumul now reflects the next group's first expert start.
        token_padding = params.token_padding_block
        sf_padding = params.sf_padding_block
        prev_valid = state.current_this_expert_token_cnt
        state.current_data_cumul = state.current_data_cumul + (
            (prev_valid + Int32(token_padding - 1)) //
            Int32(token_padding)) * Int32(token_padding)
        state.current_sf_cumul = state.current_sf_cumul + (
            (prev_valid + Int32(sf_padding - 1)) //
            Int32(sf_padding)) * Int32(sf_padding)
        state.current_token_block_cumul = (state.current_token_block_cumul +
                                           state.current_token_block_count)

        # --- Step 3: snapshot new group_start cumul checkpoint.
        state.group_start_data_cumul = state.current_data_cumul
        state.group_start_sf_cumul = state.current_sf_cumul
        state.group_start_token_block_cumul = state.current_token_block_cumul

        # --- Step 4: roll group state forward.
        base_fc1 = state.cumulative_fc1_tiles_at_group_end
        base_fc2 = state.cumulative_fc2_tiles_at_group_end

        state.current_group_idx = state.current_group_idx + Int32(1)
        state.current_group_first_expert = state.current_group_last_expert_exclusive

        # Greedy walk: accumulate per-expert fc1+fc2 tile counts until fc1
        # cumulative crosses (base + group_hint), or experts exhausted.
        threshold = base_fc1 + Int32(params.group_hint)
        cumulative_fc1 = base_fc1
        cumulative_fc2 = base_fc2
        expert_cursor = state.current_group_first_expert

        while expert_cursor < self.expert_cnt and cumulative_fc1 < threshold:
            # Only the per-expert token count drives the group greedy walk
            # (the offset is not consumed here), so the sizes-mode branch
            # is the simpler one.
            if cutlass.const_expr(self.params.expert_token_sizes is not None):
                token_count_e = compute_expert_token_count_from_sizes(
                    self.params.expert_token_sizes,
                    expert_cursor,
                    loc=loc,
                    ip=ip,
                )
            else:
                _, token_count_e = compute_expert_token_range(
                    self.params.expert_token_prefix_sum,
                    expert_cursor,
                    loc=loc,
                    ip=ip,
                )
            token_block_count_e = (token_count_e + Int32(cluster_tile_m) -
                                   1) // Int32(cluster_tile_m)
            cumulative_fc1 = (
                cumulative_fc1 +
                token_block_count_e * self._num_fc1_intermediate_blocks)
            cumulative_fc2 = (cumulative_fc2 +
                              token_block_count_e * self._num_fc2_hidden_blocks)
            expert_cursor = expert_cursor + Int32(1)

        state.current_group_last_expert_exclusive = expert_cursor
        state.cumulative_fc1_tiles_at_group_end = cumulative_fc1
        state.cumulative_fc2_tiles_at_group_end = cumulative_fc2

        group_total_fc1_tiles = cumulative_fc1 - base_fc1
        group_total_fc2_tiles = cumulative_fc2 - base_fc2

        # Previous group's end = this group's start in tile space.
        group_start_tile = state.current_group_end
        state.current_group_fc1_subphase_end = (group_start_tile +
                                                group_total_fc1_tiles)
        state.current_group_end = (state.current_group_fc1_subphase_end +
                                   group_total_fc2_tiles)

        # No-op push barrier before priming the group's first expert.
        state.current_phase = Int32(BlockPhase.Linear1)
        state.current_expert_idx = state.current_group_first_expert - Int32(1)
        state.current_expert_tile_end = group_start_tile
        state.current_this_expert_token_cnt = Int32(0)
        state.current_token_block_count = Int32(0)
        self._advance_expert_within_phase(loc=loc, ip=ip)

    # -------------------------------------------------------------------------
    # Fast-path decode
    # -------------------------------------------------------------------------

    @dsl_user_op
    @cute.jit
    def _decode_inside_expert(
        self,
        cluster_linear_tile_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> MoEWorkTileInfo:
        """Decode one cluster-linear tile using the current scheduler state."""
        state = self._fused_state
        params = self.params
        cta_tile_m = params.cta_tile_shape_mnk[0]

        local_id = cluster_linear_tile_idx - state.current_expert_tile_start

        # Prebind due to DSL AST.
        cluster_token_block_idx = Int32(0)
        cluster_intermediate_or_hidden_block_idx = Int32(0)

        is_fc1 = state.current_phase == Int32(BlockPhase.Linear1)
        if is_fc1:
            num_fc1_intermediate_blocks = self._num_fc1_intermediate_blocks
            cluster_token_block_idx = local_id // num_fc1_intermediate_blocks
            cluster_intermediate_or_hidden_block_idx = (
                local_id -
                cluster_token_block_idx * num_fc1_intermediate_blocks)
        else:
            num_fc2_hidden_blocks = self._num_fc2_hidden_blocks
            # Keep token-block as the slow axis in both phases.
            cluster_token_block_idx = local_id // num_fc2_hidden_blocks
            cluster_intermediate_or_hidden_block_idx = (
                local_id - cluster_token_block_idx * num_fc2_hidden_blocks)

        # Cluster → CTA granularity (mirrors MoESchedulerBase._get_work_tile_for_linear_idx)
        cta_token_block_idx = (
            cluster_token_block_idx * params.cluster_shape_mn[0] +
            self.cta_id_in_cluster[0])
        cta_intermediate_or_hidden_block_idx = (
            cluster_intermediate_or_hidden_block_idx *
            params.cluster_shape_mn[1] + self.cta_id_in_cluster[1])

        # valid_tokens_in_tile: clip cta_tile_m tokens at the current expert
        # right boundary.
        token_idx_start_in_expert = cta_token_block_idx * Int32(cta_tile_m)
        remaining_in_expert = (state.current_this_expert_token_cnt -
                               token_idx_start_in_expert)
        remaining_in_expert = cutlass.max(remaining_in_expert, Int32(0))
        valid_tokens_in_tile = cutlass.min(remaining_in_expert,
                                           Int32(cta_tile_m))

        # Swap scheduler-internal M/N back to GEMM-domain M/N on output.
        if const_expr(params.is_swap_ab):
            tile_m_idx = cta_intermediate_or_hidden_block_idx
            tile_n_idx = cta_token_block_idx
        else:
            tile_m_idx = cta_token_block_idx
            tile_n_idx = cta_intermediate_or_hidden_block_idx

        # ext.enrich_work_tile_info may OR the peek bit into phase_and_peek.
        return self._ext.WorkTileInfo(
            expert_idx=state.current_expert_idx,
            tile_m_idx=tile_m_idx,
            tile_n_idx=tile_n_idx,
            cumulative_data_physical_row=state.current_data_cumul,
            cumulative_sf_physical_row=state.current_sf_cumul,
            cumulative_token_block_count=state.current_token_block_cumul,
            valid_tokens_in_tile=valid_tokens_in_tile,
            phase_and_peek=state.current_phase,
        )

    @dsl_user_op
    @cute.jit
    def _gen_work_from_cluster_idx(
        self,
        cluster_linear_tile_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Decode and publish current work from one cluster-linear tile id."""
        state = self._fused_state

        # Sentinel-by-default work tile; conditionally overwritten by decode.
        base_work = self._ext.WorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            cumulative_data_physical_row=Int32(0),
            cumulative_sf_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_tile=Int32(0),
            phase_and_peek=Int32(BlockPhase.None_),
        )

        # DSL carry for mutated self and while-condition fields.
        outer_group_end = state.current_group_end
        outer_group_last_expert_exclusive = state.current_group_last_expert_exclusive
        while (cluster_linear_tile_idx >= outer_group_end
               and outer_group_last_expert_exclusive < self.expert_cnt):
            self._advance_group(loc=loc, ip=ip)
            self._fused_state = self._fused_state  # DSL carry
            state = (
                self._fused_state
            )  # re-bind alias inside body so refresh below sees new SSA
            outer_group_end = state.current_group_end
            outer_group_last_expert_exclusive = (
                state.current_group_last_expert_exclusive)
        state = self._fused_state  # re-bind alias to the post-while yielded SSA

        is_valid = cluster_linear_tile_idx < state.current_group_end
        if is_valid:
            # fc1 → fc2 phase transition inside current group, if crossed.
            if (state.current_phase == Int32(BlockPhase.Linear1)
                    and cluster_linear_tile_idx
                    >= state.current_group_fc1_subphase_end):
                self._switch_to_fc2(loc=loc, ip=ip)
                self._fused_state = (self._fused_state)  # DSL carry
            else:
                self._fused_state = self._fused_state  # balanced else-side rebind
            state = self._fused_state  # re-bind alias

            # non-PyIR: carry loop-condition fields as locals.
            inner_expert_tile_end = state.current_expert_tile_end
            while cluster_linear_tile_idx >= inner_expert_tile_end:
                self._advance_expert_within_phase(loc=loc, ip=ip)
                self._fused_state = self._fused_state  # DSL carry
                state = self._fused_state  # re-bind alias inside body
                inner_expert_tile_end = state.current_expert_tile_end
            state = self._fused_state  # re-bind alias

            base_work = self._decode_inside_expert(cluster_linear_tile_idx,
                                                   loc=loc,
                                                   ip=ip)
        else:
            # Balance scf.if yield for self.
            self._fused_state = self._fused_state

        self.current_work = self._ext.enrich_work_tile_info(base_work)

    @dsl_user_op
    @cute.jit
    def gen_next_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Produce the next work tile for this cluster's persistent loop.

        Codegen-time fork on ``self._first_advance_pending`` (Python bool,
        set True by ``internal_init``):

          - First trace site (kernel main loop's first
            ``scheduler.gen_next_work()``): static mode is a noop
            (``self.current_work`` was decoded by ``internal_init``);
            atomic mode runs the full ``_advance_work_linear_tile_idx_dynamic
            + _gen_work_from_cluster_idx`` pipeline, with
            ``_advance_work_linear_tile_idx_dynamic`` itself reading
            ``_first_advance_pending`` to consume the cached
            ``ds.atomic_res`` (set by ``internal_init`` BEFORE
            ``pipeline_init_arrive``) instead of issuing a fresh
            ``atom.add``.  At the tail, ``_first_advance_pending`` flips
            to False so subsequent trace sites pick the vanilla path.

          - Second trace site (inside-while-body call): vanilla
            ``_advance_work_linear_tile_idx_*`` + ``_gen_work_from_cluster_idx``
            for both modes.

        First trace site consumes pre-init work; later trace sites advance normally.
        """
        iket.range_push("produce_tile_id")
        # static mode first call short-circuits: internal_init already wrote
        # the first work tile to self.current_work, so just leave it alone.
        # All other (mode, call-site) combinations run the full advance +
        # decode pipeline.  ``_advance_work_linear_tile_idx_dynamic`` itself
        # const_expr-forks on _first_advance_pending to consume the cached
        # atomic_res on its own first trace site.
        if cutlass.const_expr(self._first_advance_pending
                              and self.params.load_balance_mode == "static"):
            pass
        else:
            if const_expr(self.params.load_balance_mode == "atomic_counter"):
                cluster_linear_tile_idx = (
                    self._advance_work_linear_tile_idx_dynamic(loc=loc, ip=ip))
            elif const_expr(self.params.load_balance_mode == "static"):
                cluster_linear_tile_idx = (
                    self._advance_work_linear_tile_idx_static(loc=loc, ip=ip))
            else:  # "clc"
                raise NotImplementedError(
                    "load_balance_mode='clc' is reserved; CLC scheduler is "
                    "MoEDynamicPersistentTileScheduler, not the mega scheduler")
            self._gen_work_from_cluster_idx(cluster_linear_tile_idx,
                                            loc=loc,
                                            ip=ip)

        # Codegen-time flip after the first trace site so subsequent traces
        # (the while-body call) pick the vanilla path.  This Python
        # attribute write is observed at trace time by the next jit
        # invocation of gen_next_work / _advance_work_linear_tile_idx_dynamic.
        if cutlass.const_expr(self._first_advance_pending):
            self._first_advance_pending = False
        iket.range_pop()

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self.params))
        values.extend(extract_mlir_values(self.num_persistent_clusters))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self.current_work))
        values.extend(extract_mlir_values(self._fused_state))
        values.extend(extract_mlir_values(self._num_fc1_intermediate_blocks))
        values.extend(extract_mlir_values(self._num_fc2_hidden_blocks))
        if self.params.load_balance_mode == "atomic_counter":
            values.extend(extract_mlir_values(self._dynamic_state))
        values.extend(extract_mlir_values(self._producer_state))
        return values

    def __new_from_mlir_values__(
            self,
            values: List[ir.Value]) -> "MoEFusedFc12PersistentTileScheduler":
        idx = 0

        def _take(obj):
            nonlocal idx
            n = len(extract_mlir_values(obj))
            result = new_from_mlir_values(obj, values[idx:idx + n])
            idx += n
            return result

        new_params = _take(self.params)
        new_num_persistent_clusters = _take(self.num_persistent_clusters)
        new_cta_id_in_cluster = _take(self.cta_id_in_cluster)
        new_current_work = _take(self.current_work)
        new_fused_state = _take(self._fused_state)
        new_num_fc1_intermediate_blocks = _take(
            self._num_fc1_intermediate_blocks)
        new_num_fc2_hidden_blocks = _take(self._num_fc2_hidden_blocks)
        new_dynamic_state = (_take(self._dynamic_state)
                             if self.params.load_balance_mode
                             == "atomic_counter" else None)
        new_producer_state = _take(self._producer_state)

        result = MoEFusedFc12PersistentTileScheduler.__new__(
            MoEFusedFc12PersistentTileScheduler)
        result.params = new_params
        result.num_persistent_clusters = new_num_persistent_clusters
        result.cta_id_in_cluster = new_cta_id_in_cluster
        result.current_work = new_current_work
        result._fused_state = new_fused_state
        result._num_fc1_intermediate_blocks = new_num_fc1_intermediate_blocks
        result._num_fc2_hidden_blocks = new_num_fc2_hidden_blocks
        result._dynamic_state = new_dynamic_state
        result._ext = self._ext
        result._pipeline = self._pipeline
        result._smem_buf_tensor = self._smem_buf_tensor
        result._num_sched_stages = self._num_sched_stages
        result._cluster_pipeline = self._cluster_pipeline
        result._producer_state = new_producer_state
        # Python-only attrs: copy from prototype, not MLIR values.
        result._sched_storage = self._sched_storage
        result._first_advance_pending = self._first_advance_pending
        return result
