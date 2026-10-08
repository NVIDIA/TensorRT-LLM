# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable grouped and phase-interleaved FC12 schedulers."""

import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, Integer, extract_mlir_values, new_from_mlir_values

from ...api import ImplDesc, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.dsl_helpers import spin_peek, spin_wait
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.utils import ceil_div
from .base import SchedulerBase, SchedulerWorkTileBase, WorkIdAcquisitionMode
from .fc12_mapping import (
    BlockPhase,
    NonSwapAbFc12WorkTileInfo,
    SwapAbFc12WorkTileInfo,
    create_fc12_task_mapping_state,
    create_phase_interleaved_fc12_mapping_state,
    make_fc12_done_tile,
    map_fc12_linear_work_id,
    map_phase_interleaved_fc12_work_id,
    resolve_phase_interleaved_fc1_claim_target,
)
from .non_clc_mixed_cga import NonClcMixedCgaConfig, NonClcMixedCgaSchedulerWorker


def _make_non_clc_mixed_cga_config(impl_desc: ImplDesc) -> NonClcMixedCgaConfig:
    return NonClcMixedCgaConfig(
        preferred_cluster_shape=impl_desc["cluster_shape_mn"],
        fallback_cluster_shape=impl_desc.get("fallback_cluster_shape_mn"),
        launch_cluster_count=impl_desc.get("launch_cluster_count"),
        preferred_cluster_count=impl_desc.get("preferred_cluster_count"),
        fallback_cluster_count=impl_desc.get("fallback_cluster_count"),
    )


def _mixed_cga_impl_requirements() -> dict:
    return {
        "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
        "launch_cluster_count": OptionalRequirement(Optional[int]),
        "preferred_cluster_count": OptionalRequirement(Optional[int]),
        "fallback_cluster_count": OptionalRequirement(Optional[int]),
    }


def minimum_phase_interleave_fc1_claims(
    *, blocks_fc1: int, blocks_fc2: int, launch_cluster_cnt_merge_as_preferred: int
) -> int:
    """Return the global FC1 claim watermark covering one canonical FC2 wave."""
    effective_wave_width = launch_cluster_cnt_merge_as_preferred
    max_dependent_token_blocks = ceil_div(effective_wave_width + blocks_fc2 - 1, blocks_fc2)
    return max_dependent_token_blocks * blocks_fc1


def minimum_phase_interleave_hint(
    *, blocks_fc1: int, blocks_fc2: int, launch_cluster_cnt_merge_as_preferred: int
) -> int:
    """Return the per-cluster FC1 prologue covering one canonical FC2 claim wave."""
    required_fc1_claims = minimum_phase_interleave_fc1_claims(
        blocks_fc1=blocks_fc1,
        blocks_fc2=blocks_fc2,
        launch_cluster_cnt_merge_as_preferred=launch_cluster_cnt_merge_as_preferred,
    )
    return max(1, ceil_div(required_fc1_claims, launch_cluster_cnt_merge_as_preferred))


def _to_fc12_mapping_cta_coord(
    cta_coord_in_preferred_cluster: cute.Coord, is_swap_ab: bool
) -> cute.Coord:
    if cutlass.const_expr(not is_swap_ab):
        return cta_coord_in_preferred_cluster
    return (
        cta_coord_in_preferred_cluster[1],
        cta_coord_in_preferred_cluster[0],
        cta_coord_in_preferred_cluster[2],
    )


class BlackwellFusedFc12Scheduler(SchedulerBase):
    """Compose work-ID claim, FC12 mapping, and SMEM tile transport."""

    pipeline_mbarriers_region = "blackwell.fc12.scheduler.pipeline_mbarriers"
    work_tiles_region = "blackwell.fc12.scheduler.work_tiles"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "hint": Optional[int],
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "is_swap_ab": bool,
            **_mixed_cga_impl_requirements(),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.hint = impl_desc["hint"]
        self.group_hint = self.hint
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.is_swap_ab = impl_desc["is_swap_ab"]
        self.non_clc_mixed_cga_config = _make_non_clc_mixed_cga_config(impl_desc)
        self.launch_cluster_cnt_merge_as_preferred = (
            self.non_clc_mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        )
        self.work_tile_type = (
            SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo
        )
        if self.hint is None:
            self.hint = self.launch_cluster_cnt_merge_as_preferred
            self.group_hint = self.hint

        self._validate_configuration()
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        launch_cta_tile_shape_mnk = (
            self.mma_tiler_mnk[0] // mma_cta_count,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )
        if self.is_swap_ab:
            self.mapping_cta_tile_shape_mnk = (
                launch_cta_tile_shape_mnk[1],
                launch_cta_tile_shape_mnk[0],
                launch_cta_tile_shape_mnk[2],
            )
            self.mapping_cluster_shape_mn = (self.cluster_shape_mn[1], self.cluster_shape_mn[0])
        else:
            self.mapping_cta_tile_shape_mnk = launch_cta_tile_shape_mnk
            self.mapping_cluster_shape_mn = self.cluster_shape_mn

        if self.work_id_mode == "cluster_launch_control":
            raise NotImplementedError("cluster_launch_control is not implemented for FC12.")
        self._work_id_worker = NonClcMixedCgaSchedulerWorker(
            config=self.non_clc_mixed_cga_config, work_id_mode=self.work_id_mode, stream_count=1
        )

    def _validate_configuration(self) -> None:
        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.mma_tiler_mnk[0] % mma_cta_count != 0:
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if self.group_hint is not None and self.group_hint <= 0:
            raise ValueError("group_hint must be positive.")
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.sf_padding_block <= 0:
            raise ValueError("sf_padding_block must be positive.")
        if self.launch_cluster_cnt_merge_as_preferred <= 0:
            raise ValueError("launch_cluster_cnt_merge_as_preferred must be positive.")
        if self.work_id_mode not in ("grid_stride", "atomic_counter", "cluster_launch_control"):
            raise ValueError(
                "work_id_mode must be 'grid_stride', 'atomic_counter', or "
                f"'cluster_launch_control', got {self.work_id_mode!r}."
            )
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        if self.work_id_mode == "atomic_counter" and cluster_size > 32:
            raise ValueError("The atomic broadcast protocol supports at most 32 CTAs per cluster.")
        fallback_cluster_shape = self.non_clc_mixed_cga_config.fallback_cluster_shape
        if fallback_cluster_shape is not None:
            fallback_cluster_size = fallback_cluster_shape[0] * fallback_cluster_shape[1]
            if self.work_id_mode == "atomic_counter" and fallback_cluster_size > 32:
                raise ValueError(
                    "The atomic broadcast protocol supports at most 32 CTAs per fallback cluster."
                )
        for field_name in ("expert_count", "intermediate_gateup_size", "hidden_size"):
            value = getattr(self, field_name)
            if isinstance(value, int) and value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        static_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_dimensions) and not all(static_dimensions):
            raise ValueError("FC12 expert dimensions must be either all static or all runtime.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register scheduler-owned SMEM transport and claim regions."""
        super().register_smem_regions(smem_workspace)
        self._work_id_worker.register_smem_regions(smem_workspace)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register work-ID counters and optional fixed-group fallback state."""
        self._work_id_worker.register_device_workspace(device_workspace)

    @cute.jit
    def initialize_fallback_group(self) -> None:
        """Register this physical fallback cluster with its fixed logical group."""
        self._work_id_worker.initialize_fallback_group()

    def get_grid_shape(
        self, *, max_active_clusters: Optional[int] = None, problem_desc=None
    ) -> Tuple[int, int, int]:
        """Return the persistent launch grid in GEMM-domain orientation."""
        if (
            not self.non_clc_mixed_cga_config.is_mixed
            and max_active_clusters is not None
            and max_active_clusters < self.launch_cluster_cnt_merge_as_preferred
        ):
            raise ValueError(
                f"max_active_clusters ({max_active_clusters}) must be at least "
                "launch_cluster_cnt_merge_as_preferred "
                f"({self.launch_cluster_cnt_merge_as_preferred})."
            )
        return (
            self.cluster_shape_mn[0],
            self.cluster_shape_mn[1],
            self.launch_cluster_cnt_merge_as_preferred,
        )

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        actual_expert_shape: Optional[Tuple],
        block_idx: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
        is_fallback_cluster: Optional[Boolean] = None,
    ) -> None:
        """Initialize all FC12 scheduler state rooted for one CTA lifetime."""
        if cutlass.const_expr((expert_token_sizes is None) == (expert_token_prefix_sum is None)):
            raise ValueError(
                "Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided."
            )
        needs_actual_shape = not all(
            isinstance(dimension, int)
            for dimension in (self.expert_count, self.intermediate_gateup_size, self.hidden_size)
        )
        if cutlass.const_expr(needs_actual_shape and actual_expert_shape is None):
            raise ValueError("actual_expert_shape is required for runtime dimensions.")
        if cutlass.const_expr(isinstance(self.expert_count, int)):
            expert_count = self.expert_count
        else:
            expert_count = actual_expert_shape[0]
        if cutlass.const_expr(isinstance(self.intermediate_gateup_size, int)):
            intermediate_gateup_size = self.intermediate_gateup_size
        else:
            intermediate_gateup_size = actual_expert_shape[1]
        if cutlass.const_expr(isinstance(self.hidden_size, int)):
            hidden_size = self.hidden_size
        else:
            hidden_size = actual_expert_shape[2]

        self.create_scheduler_pipelines(smem_workspace, smem_base)
        self._work_id_worker.assign_device_members(
            is_fallback_cluster=is_fallback_cluster,
            block_idx=block_idx,
            smem_workspace=smem_workspace,
            smem_base=smem_base,
            device_workspace=device_workspace,
        )

        task_mapping_state = create_fc12_task_mapping_state(
            expert_count=expert_count,
            intermediate_gateup_size=intermediate_gateup_size,
            hidden_size=hidden_size,
            mapping_cta_tile_shape_mnk=(self.mapping_cta_tile_shape_mnk),
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=self.is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
        )

        self._task_mapping_state = task_mapping_state

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim and map one work tile without first-tile prefetch."""
        work_id = self._work_id_worker.claim_next_work()
        cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
            self._work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
        )
        work_tile, self._task_mapping_state = map_fc12_linear_work_id(
            work_id, cta_id_in_mapping_cluster, self._task_mapping_state
        )
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        values.extend(extract_mlir_values(self._work_id_worker))
        values.extend(extract_mlir_values(self._task_mapping_state))
        return values

    def __new_from_mlir_values__(self, values: list) -> "BlackwellFusedFc12Scheduler":
        base_value_count = len(super().__extract_mlir_values__())
        if len(values) < base_value_count:
            raise ValueError(
                "BlackwellFusedFc12Scheduler MLIR value count is smaller than "
                f"its base state: expected at least {base_value_count}, got {len(values)}."
            )
        result = super().__new_from_mlir_values__(values[:base_value_count])
        value_index = base_value_count

        def rebuild(state):
            nonlocal value_index
            state_value_count = len(extract_mlir_values(state))
            rebuilt_state = new_from_mlir_values(
                state, values[value_index : value_index + state_value_count]
            )
            value_index += state_value_count
            return rebuilt_state

        result._work_id_worker = rebuild(self._work_id_worker)
        result._task_mapping_state = rebuild(self._task_mapping_state)
        if value_index != len(values):
            raise ValueError(
                f"BlackwellFusedFc12Scheduler MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )

        for field_name in (
            "expert_count",
            "intermediate_gateup_size",
            "hidden_size",
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "use_2cta_instrs",
            "hint",
            "group_hint",
            "token_padding_block",
            "sf_padding_block",
            "work_id_mode",
            "is_swap_ab",
            "launch_cluster_cnt_merge_as_preferred",
            "non_clc_mixed_cga_config",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
        ):
            setattr(result, field_name, getattr(self, field_name))
        return result


class _PhaseInterleaveControlState:
    """Per-cluster phase cadence and stream exhaustion state."""

    def __init__(
        self,
        prologue_remaining: Int32,
        cycle_position: Int32,
        fc1_exhausted: Boolean,
        fc2_exhausted: Boolean,
        prologue_claims_synchronized: Boolean,
    ) -> None:
        self.prologue_remaining = prologue_remaining
        self.cycle_position = cycle_position
        self.fc1_exhausted = fc1_exhausted
        self.fc2_exhausted = fc2_exhausted
        self.prologue_claims_synchronized = prologue_claims_synchronized

    def _fields(self) -> Tuple:
        return (
            self.prologue_remaining,
            self.cycle_position,
            self.fc1_exhausted,
            self.fc2_exhausted,
            self.prologue_claims_synchronized,
        )

    def __extract_mlir_values__(self) -> list:
        values = []
        for field in self._fields():
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: list) -> "_PhaseInterleaveControlState":
        value_index = 0
        rebuilt_fields = []
        for field in self._fields():
            field_value_count = len(extract_mlir_values(field))
            rebuilt_fields.append(
                new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            )
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(
                f"_PhaseInterleaveControlState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(*rebuilt_fields)


class PhaseInterleavedFc12Scheduler(SchedulerBase):
    """Schedule independent FC1 and FC2 streams with per-phase atomic counters."""

    pipeline_mbarriers_region = "fc12.phase_interleaved.scheduler.pipeline_mbarriers"
    work_tiles_region = "fc12.phase_interleaved.scheduler.work_tiles"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {"expert_count": int, "intermediate_gateup_size": int, "hidden_size": int}

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "hint": int,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "is_swap_ab": bool,
            **_mixed_cga_impl_requirements(),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.hint = impl_desc["hint"]
        self.fc1_prologue_tiles = self.hint
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.is_swap_ab = impl_desc["is_swap_ab"]
        self.non_clc_mixed_cga_config = _make_non_clc_mixed_cga_config(impl_desc)
        self.launch_cluster_cnt_merge_as_preferred = (
            self.non_clc_mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        )
        self.work_tile_type = (
            SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo
        )

        self._validate_configuration()
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        launch_cta_tile_shape_mnk = (
            self.mma_tiler_mnk[0] // mma_cta_count,
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )
        if self.is_swap_ab:
            self.mapping_cta_tile_shape_mnk = (
                launch_cta_tile_shape_mnk[1],
                launch_cta_tile_shape_mnk[0],
                launch_cta_tile_shape_mnk[2],
            )
            self.mapping_cluster_shape_mn = (self.cluster_shape_mn[1], self.cluster_shape_mn[0])
        else:
            self.mapping_cta_tile_shape_mnk = launch_cta_tile_shape_mnk
            self.mapping_cluster_shape_mn = self.cluster_shape_mn
        self._work_id_worker = NonClcMixedCgaSchedulerWorker(
            config=self.non_clc_mixed_cga_config, work_id_mode=self.work_id_mode, stream_count=2
        )

        mapping_cluster_tile_n = (
            self.mapping_cta_tile_shape_mnk[1] * self.mapping_cluster_shape_mn[1]
        )
        self.blocks_fc1 = (
            self.intermediate_gateup_size + mapping_cluster_tile_n - 1
        ) // mapping_cluster_tile_n
        self.blocks_fc2 = (self.hidden_size + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
        interleave_gcd = math.gcd(self.blocks_fc1, self.blocks_fc2)
        self.interleave_fc2_slots = self.blocks_fc2 // interleave_gcd
        self.interleave_cycle_length = (self.blocks_fc1 + self.blocks_fc2) // interleave_gcd
        self.minimum_global_fc1_claims = minimum_phase_interleave_fc1_claims(
            blocks_fc1=self.blocks_fc1,
            blocks_fc2=self.blocks_fc2,
            launch_cluster_cnt_merge_as_preferred=self.launch_cluster_cnt_merge_as_preferred,
        )
        minimum_hint = minimum_phase_interleave_hint(
            blocks_fc1=self.blocks_fc1,
            blocks_fc2=self.blocks_fc2,
            launch_cluster_cnt_merge_as_preferred=self.launch_cluster_cnt_merge_as_preferred,
        )
        if self.fc1_prologue_tiles < minimum_hint:
            raise ValueError(
                f"phase_interleave hint {self.fc1_prologue_tiles} cannot cover a "
                f"{self.launch_cluster_cnt_merge_as_preferred}-cluster FC2 claim wave; "
                f"raise hint to at least {minimum_hint}."
            )

    def _validate_configuration(self) -> None:
        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool)
            for dimension in self.mma_tiler_mnk
        ):
            raise TypeError("mma_tiler_mnk dimensions must be Python ints.")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool)
            for dimension in self.cluster_shape_mn
        ):
            raise TypeError("cluster_shape_mn dimensions must be Python ints.")
        if any(dimension <= 0 for dimension in self.mma_tiler_mnk):
            raise ValueError("mma_tiler_mnk dimensions must be positive.")
        if any(dimension <= 0 for dimension in self.cluster_shape_mn):
            raise ValueError("cluster_shape_mn dimensions must be positive.")
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        if self.mma_tiler_mnk[0] % mma_cta_count != 0:
            raise ValueError("mma_tiler M must be divisible by the MMA CTA count.")
        if (
            isinstance(self.fc1_prologue_tiles, bool)
            or not isinstance(self.fc1_prologue_tiles, int)
            or self.fc1_prologue_tiles <= 0
        ):
            raise ValueError(
                "fc1_prologue_tiles must be a positive Python int resolved by the kernel frontend."
            )
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.sf_padding_block <= 0:
            raise ValueError("sf_padding_block must be positive.")
        if self.launch_cluster_cnt_merge_as_preferred <= 0:
            raise ValueError("launch_cluster_cnt_merge_as_preferred must be positive.")
        if self.work_id_mode != "atomic_counter":
            raise ValueError(
                "Phase-interleaved FC12 scheduling currently requires work_id_mode='atomic_counter'."
            )
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        if cluster_size > 32:
            raise ValueError("The atomic broadcast protocol supports at most 32 CTAs per cluster.")
        fallback_cluster_shape = self.non_clc_mixed_cga_config.fallback_cluster_shape
        if fallback_cluster_shape is not None:
            fallback_cluster_size = fallback_cluster_shape[0] * fallback_cluster_shape[1]
            if fallback_cluster_size > 32:
                raise ValueError(
                    "The atomic broadcast protocol supports at most 32 CTAs per fallback cluster."
                )
        maximum_int32 = (1 << 31) - 1
        for field_name in ("expert_count", "intermediate_gateup_size", "hidden_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive Python int.")
            if value > maximum_int32:
                raise ValueError(f"{field_name} must fit in a signed Int32, got {value}.")

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Register work transport and the phase-counter broadcast channel."""
        super().register_smem_regions(smem_workspace)
        self._work_id_worker.register_smem_regions(smem_workspace)

    def register_device_workspace(self, device_workspace: DeviceWorkspace) -> None:
        """Register independently reset FC1 and FC2 work-ID streams."""
        self._work_id_worker.register_device_workspace(device_workspace)

    @cute.jit
    def initialize_fallback_group(self) -> None:
        """Register this physical fallback cluster with its fixed logical group."""
        self._work_id_worker.initialize_fallback_group()

    def get_grid_shape(
        self, *, max_active_clusters: Optional[int] = None, problem_desc=None
    ) -> Tuple[int, int, int]:
        """Return the statically configured persistent launch grid."""
        if (
            not self.non_clc_mixed_cga_config.is_mixed
            and max_active_clusters is not None
            and max_active_clusters < self.launch_cluster_cnt_merge_as_preferred
        ):
            raise ValueError(
                f"max_active_clusters ({max_active_clusters}) must be at least "
                "launch_cluster_cnt_merge_as_preferred "
                f"({self.launch_cluster_cnt_merge_as_preferred})."
            )
        return (
            self.cluster_shape_mn[0],
            self.cluster_shape_mn[1],
            self.launch_cluster_cnt_merge_as_preferred,
        )

    @cute.jit
    def assign_device_members(
        self,
        *,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        actual_expert_shape: Optional[Tuple],
        block_idx: Tuple[Integer, Integer, Integer],
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        device_workspace: DeviceWorkspace,
        is_fallback_cluster: Optional[Boolean] = None,
    ) -> None:
        """Initialize phase-local mapping, cadence, and counter state."""
        if cutlass.const_expr((expert_token_sizes is None) == (expert_token_prefix_sum is None)):
            raise ValueError(
                "Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided."
            )
        self.create_scheduler_pipelines(smem_workspace, smem_base)
        self._work_id_worker.assign_device_members(
            is_fallback_cluster=is_fallback_cluster,
            block_idx=block_idx,
            smem_workspace=smem_workspace,
            smem_base=smem_base,
            device_workspace=device_workspace,
        )
        self._task_mapping_state = create_phase_interleaved_fc12_mapping_state(
            expert_count=self.expert_count,
            intermediate_gateup_size=self.intermediate_gateup_size,
            hidden_size=self.hidden_size,
            mapping_cta_tile_shape_mnk=self.mapping_cta_tile_shape_mnk,
            mapping_cluster_shape_mn=self.mapping_cluster_shape_mn,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            is_swap_ab=self.is_swap_ab,
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
        )
        self._control_state = _PhaseInterleaveControlState(
            prologue_remaining=Int32(self.fc1_prologue_tiles),
            cycle_position=Int32(0),
            fc1_exhausted=Boolean(False),
            fc2_exhausted=Boolean(False),
            prologue_claims_synchronized=Boolean(False),
        )

    @cute.jit
    def _wait_for_fc1_prologue_claims(
        self,
        fc1_exhausted: Boolean,
        work_id_worker: NonClcMixedCgaSchedulerWorker,
        task_mapping_state,
    ) -> Boolean:
        """Wait until the static FC1 watermark or the smaller runtime stream end is claimed."""
        if not fc1_exhausted:
            fc1_counter_pointer = work_id_worker.get_atomic_counter_pointer(Int32(0))
            minimum_claim_count = Int32(self.minimum_global_fc1_claims)
            if not spin_peek(fc1_counter_pointer, lambda value: value >= minimum_claim_count):
                claim_target, stream_ends_before_target = (
                    resolve_phase_interleaved_fc1_claim_target(
                        minimum_claim_count, task_mapping_state
                    )
                )
                spin_wait(fc1_counter_pointer, lambda value: value >= claim_target)
                if stream_ends_before_target:
                    fc1_exhausted = Boolean(True)
        return fc1_exhausted

    @cute.jit
    def gen_next_work(self) -> SchedulerWorkTileBase:
        """Claim until one stream yields valid work or both streams terminate."""
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        prologue_remaining = control_state.prologue_remaining
        cycle_position = control_state.cycle_position
        fc1_exhausted = control_state.fc1_exhausted
        fc2_exhausted = control_state.fc2_exhausted
        prologue_claims_synchronized = control_state.prologue_claims_synchronized
        resolved = Boolean(False)

        while not resolved:
            if fc1_exhausted and fc2_exhausted:
                work_tile = make_fc12_done_tile(self.is_swap_ab)
                resolved = Boolean(True)
            else:
                want_fc1 = Boolean(True)
                if prologue_remaining <= Int32(0):
                    is_fc2_slot = (cycle_position * Int32(self.interleave_fc2_slots)) % Int32(
                        self.interleave_cycle_length
                    ) < Int32(self.interleave_fc2_slots)
                    want_fc1 = not is_fc2_slot
                if want_fc1 and fc1_exhausted:
                    want_fc1 = Boolean(False)
                if (not want_fc1) and fc2_exhausted:
                    want_fc1 = Boolean(True)

                atomic_counter_index = Int32(1)
                if want_fc1:
                    atomic_counter_index = Int32(0)
                linear_work_id = work_id_worker.claim_next_work(atomic_counter_index)
                want_fc1 = work_id_worker.claimed_stream_index == Int32(0)
                phase = Int32(BlockPhase.Linear2)
                if want_fc1:
                    phase = Int32(BlockPhase.Linear1)
                cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                    work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
                )
                work_tile, stream_has_work, task_mapping_state = map_phase_interleaved_fc12_work_id(
                    linear_work_id, phase, cta_id_in_mapping_cluster, task_mapping_state
                )
                if stream_has_work:
                    if (not want_fc1) and (not prologue_claims_synchronized):
                        fc1_exhausted = self._wait_for_fc1_prologue_claims(
                            fc1_exhausted, work_id_worker, task_mapping_state
                        )
                        prologue_claims_synchronized = Boolean(True)
                    if prologue_remaining > Int32(0):
                        prologue_remaining = prologue_remaining - Int32(1)
                    else:
                        cycle_position = (cycle_position + Int32(1)) % Int32(
                            self.interleave_cycle_length
                        )
                    resolved = Boolean(True)
                else:
                    if want_fc1:
                        fc1_exhausted = Boolean(True)
                    else:
                        fc2_exhausted = Boolean(True)

        control_state.prologue_remaining = prologue_remaining
        control_state.cycle_position = cycle_position
        control_state.fc1_exhausted = fc1_exhausted
        control_state.fc2_exhausted = fc2_exhausted
        control_state.prologue_claims_synchronized = prologue_claims_synchronized
        self._work_id_worker = work_id_worker
        self._task_mapping_state = task_mapping_state
        self._control_state = control_state
        return work_tile

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        for state in (self._work_id_worker, self._task_mapping_state, self._control_state):
            values.extend(extract_mlir_values(state))
        return values

    def __new_from_mlir_values__(self, values: list) -> "PhaseInterleavedFc12Scheduler":
        base_value_count = len(super().__extract_mlir_values__())
        if len(values) < base_value_count:
            raise ValueError(
                "PhaseInterleavedFc12Scheduler MLIR value count is smaller than "
                f"its base state: expected at least {base_value_count}, got {len(values)}."
            )
        result = super().__new_from_mlir_values__(values[:base_value_count])
        value_index = base_value_count

        def rebuild(state):
            nonlocal value_index
            state_value_count = len(extract_mlir_values(state))
            rebuilt_state = new_from_mlir_values(
                state, values[value_index : value_index + state_value_count]
            )
            value_index += state_value_count
            return rebuilt_state

        result._work_id_worker = rebuild(self._work_id_worker)
        result._task_mapping_state = rebuild(self._task_mapping_state)
        result._control_state = rebuild(self._control_state)
        if value_index != len(values):
            raise ValueError(
                f"PhaseInterleavedFc12Scheduler MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )

        for field_name in (
            "expert_count",
            "intermediate_gateup_size",
            "hidden_size",
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "use_2cta_instrs",
            "hint",
            "fc1_prologue_tiles",
            "token_padding_block",
            "sf_padding_block",
            "work_id_mode",
            "is_swap_ab",
            "launch_cluster_cnt_merge_as_preferred",
            "non_clc_mixed_cga_config",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
            "blocks_fc1",
            "blocks_fc2",
            "interleave_fc2_slots",
            "interleave_cycle_length",
            "minimum_global_fc1_claims",
        ):
            setattr(result, field_name, getattr(self, field_name))
        return result


__all__ = [
    "BlackwellFusedFc12Scheduler",
    "PhaseInterleavedFc12Scheduler",
    "minimum_phase_interleave_hint",
]
