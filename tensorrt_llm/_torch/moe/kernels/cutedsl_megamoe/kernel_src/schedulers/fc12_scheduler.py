"""Composable grouped and phase-interleaved FC12 schedulers."""

import math
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, Integer, extract_mlir_values, new_from_mlir_values

from ...api import ImplDesc, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.dsl_helpers import ReadyFlag, spin_wait, wait_ready
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.utils import ceil_div
from .base import SchedulerBase, SchedulerWorkTileBase, WorkIdAcquisitionMode
from .clc_mixed_cga import ClcMixedCgaSchedulerWorker
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
    resolve_phase_interleaved_stream_extent,
    rewind_fc12_mapping_cursor,
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


def minimum_phase_interleave_ring_token_blocks(
    *, blocks_fc1: int, blocks_fc2: int, launch_cluster_cnt_merge_as_preferred: int, hint: int
) -> int:
    """Return the smallest ring workspace, in token blocks, a kernel accepts for the phase-interleaved claim order.

    With a ring workspace of R token blocks, block b takes over the slot of block b - R, so an FC1 tile of
    block b cannot run before every FC2 tile of block b - R has: until then token_in does not write the tokens
    of b (in the local kernel: their scale rows). A cluster runs its claims in order, so those FC2 tiles are
    certain to run only if they were claimed before the FC1 tile, not behind it in some cluster's queue. Each
    cluster claims ``hint`` FC1 tiles and then keeps the blocks_fc1 : blocks_fc2 cadence, in which the FC1 and
    the FC2 count of a cluster each stay within one claim of their share. In token blocks the FC1 stream
    therefore leads the FC2 stream by at most

        clusters * hint / blocks_fc1 + clusters * (1 / blocks_fc1 + 1 / blocks_fc2)

    and the bound is that many blocks, rounded up, plus two (the block an id falls into, and one block of
    margin). It is conservative, and tests/test_ring_workspace_contract.py runs the claims of the clusters
    against it in sampled orders; it is not a proof.
    """
    if blocks_fc1 <= 0 or blocks_fc2 <= 0 or launch_cluster_cnt_merge_as_preferred <= 0 or hint < 0:
        raise ValueError("The ring bound needs positive block and cluster counts and a non-negative hint.")
    clusters = launch_cluster_cnt_merge_as_preferred
    # Integer ceiling of clusters * (hint + 1) / blocks_fc1 + clusters / blocks_fc2.
    numerator = clusters * (hint + 1) * blocks_fc2 + clusters * blocks_fc1
    return ceil_div(numerator, blocks_fc1 * blocks_fc2) + 2


def _to_fc12_mapping_cta_coord(cta_coord_in_preferred_cluster: cute.Coord, is_swap_ab: bool) -> cute.Coord:
    if cutlass.const_expr(not is_swap_ab):
        return cta_coord_in_preferred_cluster
    return (cta_coord_in_preferred_cluster[1], cta_coord_in_preferred_cluster[0], cta_coord_in_preferred_cluster[2])


def shared_expert_problem_requirements() -> dict:
    return {
        "shared_expert": OptionalRequirement(bool),
        "shared_intermediate_gateup_size": OptionalRequirement(Optional[int]),
        "shared_output_size": OptionalRequirement(Optional[int]),
        "max_tokens_per_rank": OptionalRequirement(StaticOrRuntimeIntegerType),
    }


def shared_expert_impl_requirements() -> dict:
    return {
        "shared_expert_stream": OptionalRequirement(bool),
        "shared_l2_position": OptionalRequirement(Optional[str]),
        "shared_work_id_mode": OptionalRequirement(Optional[str]),
        "shared_fc1_swizzle": OptionalRequirement(Optional[int]),
        "expert_sizes_ready_region": OptionalRequirement(str),
        "expert_sizes_ready_target": OptionalRequirement(int),
        "expert_sizes_ready_scope": OptionalRequirement(str),
    }


# Shared expert stages of a cluster: claiming FC1 tiles, claiming FC2 tiles, no shared work left.
_SHARED_FC1, _SHARED_FC2, _SHARED_DONE = 0, 1, 2


class SharedExpertStreamMixin:
    """Places the shared FC1/FC2 tiles around the routed work (FC2 first when ``early``, last when ``tail``)."""

    def _bind_shared_expert_stream(
        self,
        problem_desc: ProblemDesc,
        impl_desc: ImplDesc,
        *,
        cta_tile_m: int,
        cta_tile_n: int,
        routed_stream_count: int,
    ) -> None:
        self._expert_sizes_ready_region = impl_desc.get("expert_sizes_ready_region")
        self._expert_sizes_ready_target = impl_desc.get("expert_sizes_ready_target")
        # Peers publish the sizes in the prefill kernel (system scope); the gen-phase helper CTA is local (GPU scope).
        self._expert_sizes_ready_scope = impl_desc.get("expert_sizes_ready_scope", "gpu")
        if self._expert_sizes_ready_region is not None and self._expert_sizes_ready_target is None:
            raise ValueError("expert_sizes_ready_region needs expert_sizes_ready_target.")
        self._expert_sizes_ready_pointer = None
        # A Python True is never loop-carried; a Boolean replaces it once the ready region is bound on device.
        self._expert_sizes_ready = True
        self._shared_stream_index = routed_stream_count
        self._shared_stage = None
        self._shared_next = None
        self._routed_done = None
        self.shared_fc2_position = None
        self._shared_fc1_swizzle = 0
        if not (impl_desc.get("shared_expert_stream", False) and problem_desc.get("shared_expert", False)):
            return
        if not self.is_swap_ab:
            raise ValueError("shared_expert tiles need the swap-AB work tile layout.")
        if self.work_id_mode == "cluster_launch_control":
            raise ValueError("shared_expert is not supported with work_id_mode='cluster_launch_control'.")
        if self.cluster_shape_mn[1] != 1:
            raise ValueError("shared_expert tiles decode one token block per cluster; cluster N must be 1.")
        sizes = []
        for key in ("max_tokens_per_rank", "shared_intermediate_gateup_size", "shared_output_size"):
            value = problem_desc.get(key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"shared_expert tiles need {key} as a positive static int, got {value!r}.")
            sizes.append(value)
        tokens, gateup, output = sizes
        cluster_m = self.cluster_shape_mn[0]
        feature_tile = cta_tile_m * cluster_m
        self._shared_tile_n = cta_tile_n
        self._shared_cluster_m = cluster_m
        self._shared_token_count = tokens
        self.shared_token_blocks = ceil_div(tokens, cta_tile_n)
        self._shared_feature_blocks = (ceil_div(gateup, feature_tile), ceil_div(output, feature_tile))
        self._shared_phase_tiles = tuple(blocks * self.shared_token_blocks for blocks in self._shared_feature_blocks)
        position = impl_desc.get("shared_l2_position")
        if position is None or position == "auto":
            # DeepGEMM's rule: FC2 goes first when one wave of clusters covers its tiles.
            position = "early" if self._shared_phase_tiles[1] <= self.compute_cluster_count else "tail"
        if position not in ("early", "tail"):
            raise ValueError(f"shared_l2_position must be 'auto', 'early' or 'tail', got {position!r}.")
        self.shared_fc2_position = position
        swizzle = int(impl_desc.get("shared_fc1_swizzle") or 0)
        if swizzle < 0:
            raise ValueError(f"shared_fc1_swizzle must be a non-negative feature-block count, got {swizzle}.")
        # A group as wide as the FC1 feature blocks is the unswizzled walk.
        self._shared_fc1_swizzle = swizzle if 0 < swizzle < self._shared_feature_blocks[0] else 0
        mode = impl_desc.get("shared_work_id_mode")
        if mode not in (None, "grid_stride", "atomic_counter"):
            raise ValueError(f"shared_work_id_mode must be 'atomic_counter' or 'grid_stride', got {mode!r}.")
        if mode == "atomic_counter" and self.work_id_mode != "atomic_counter":
            raise ValueError("shared_work_id_mode='atomic_counter' needs work_id_mode='atomic_counter'.")
        self.shared_ids_atomic = self.work_id_mode == "atomic_counter" and mode != "grid_stride"
        mixed_atomic = self.non_clc_mixed_cga_config.is_mixed and self.work_id_mode == "atomic_counter"
        if mixed_atomic and not self.shared_ids_atomic:
            # Fallback CTAs take their M slot from the registered group, which a block_idx-strided shared id ignores.
            raise ValueError("shared_work_id_mode='grid_stride' is unsupported on a mixed-cluster atomic launch.")

    @property
    def shared_stream_count(self) -> int:
        """The extra atomic counters the work-id worker carries for shared ids: one for FC1, one for FC2."""
        return 2 if self.shared_fc2_position is not None and self.shared_ids_atomic else 0

    @cute.jit
    def _assign_shared_expert_stream(self, block_idx: Tuple, device_workspace: DeviceWorkspace) -> None:
        if cutlass.const_expr(self._expert_sizes_ready_region is not None):
            self._expert_sizes_ready_pointer = device_workspace.ptr(self._expert_sizes_ready_region)
            self._expert_sizes_ready = Boolean(False)
        if cutlass.const_expr(self.shared_fc2_position is not None):
            self._shared_stage = Int32(_SHARED_FC1)
            # Grid-stride only: this cluster's next shared id in the current stage.
            self._shared_next = Int32(block_idx[2])
            self._routed_done = Boolean(False)

    @cute.jit
    def _wait_expert_sizes_ready(self, sizes_ready: Boolean) -> Boolean:
        """Block once per CTA until the communication kernel has published the expert sizes."""
        if not sizes_ready:
            wait_ready(
                ReadyFlag(
                    self._expert_sizes_ready_pointer, self._expert_sizes_ready_target, self._expert_sizes_ready_scope
                )
            )
            sizes_ready = Boolean(True)
        return sizes_ready

    @cute.jit
    def _decode_shared_tile(self, local_id: Int32, phase_offset: int, segment=None) -> SwapAbFc12WorkTileInfo:
        """Map a shared FC1 (``phase_offset`` 0) or FC2 (1) cluster work id to this CTA's tile, feature block fast.

        With ``segment`` = (first feature block, feature block count) the id counts that part of the feature blocks.
        """
        feature_blocks = Int32(self._shared_feature_blocks[phase_offset])
        if cutlass.const_expr(segment is not None):
            feature_blocks = segment[1]
        token_block = local_id // feature_blocks
        feature_block = local_id - token_block * feature_blocks
        if cutlass.const_expr(phase_offset == 0 and self._shared_fc1_swizzle > 0):
            # L2 swizzle: every token block of one group of feature blocks before the next group.
            group_width = Int32(self._shared_fc1_swizzle)
            group_tiles = group_width * Int32(self.shared_token_blocks)
            group = local_id // group_tiles
            in_group = local_id - group * group_tiles
            width = cutlass.min(group_width, feature_blocks - group * group_width)
            token_block = in_group // width
            feature_block = group * group_width + (in_group - token_block * width)
        if cutlass.const_expr(segment is not None):
            feature_block = feature_block + segment[0]
        cta_m = Int32(cute.arch.block_idx_in_cluster() % self._shared_cluster_m)
        if cutlass.const_expr(self.non_clc_mixed_cga_config.is_mixed):
            # A fallback cluster's CTAs take their place in the logical preferred cluster their group claims for.
            cta_m = Int32(self._work_id_worker.cta_coord_in_preferred_cluster[0]) % Int32(self._shared_cluster_m)
        valid_tokens = cutlass.min(
            Int32(self._shared_token_count) - token_block * Int32(self._shared_tile_n), Int32(self._shared_tile_n)
        )
        return SwapAbFc12WorkTileInfo(
            expert_idx=Int32(0),
            tile_m_idx=feature_block * Int32(self._shared_cluster_m) + cta_m,
            tile_n_idx=token_block,
            cumulative_data_physical_row=Int32(0),
            cumulative_sf_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_cta_tile=valid_tokens,
            phase_and_flags=Int32(int((BlockPhase.SharedLinear1, BlockPhase.SharedLinear2)[phase_offset])),
        )

    @cute.jit
    def _claim_shared_tile(self, work_id_worker, stage: Int32, next_id: Int32, fc2: int):
        """Claim this cluster's next shared FC1 or FC2 tile; an exhausted claim advances ``stage`` and returns DONE."""
        if cutlass.const_expr(self.shared_ids_atomic):
            # As in DeepGEMM, FC1 and FC2 ids come from separate counters.
            local_id = work_id_worker.claim_next_work(self._shared_stream_index + fc2)
        else:
            local_id = next_id
            next_id = next_id + Int32(self.compute_cluster_count)
        phase_tiles = Int32(self._shared_phase_tiles[fc2])
        work_tile = make_fc12_done_tile(True)
        if local_id < phase_tiles:
            work_tile = self._decode_shared_tile(local_id, fc2)
        else:
            stage = stage + Int32(1)
            # A grid-stride id past FC1 carries on as this cluster's first FC2 id.
            next_id = local_id - phase_tiles
        return work_tile, stage, next_id, work_id_worker

    @cute.jit
    def gen_next_work(self, work_id_offset: Int32 = Int32(0)) -> SchedulerWorkTileBase:
        """Shared FC1, early shared FC2, routed work past the expert-sizes gate, tail shared FC2, then DONE."""
        if cutlass.const_expr(self.shared_fc2_position is None):
            if cutlass.const_expr(self._expert_sizes_ready_pointer is not None):
                self._expert_sizes_ready = self._wait_expert_sizes_ready(self._expert_sizes_ready)
            return self._gen_next_routed_work(work_id_offset)

        work_id_worker = self._work_id_worker
        stage = self._shared_stage
        next_id = self._shared_next
        routed_done = self._routed_done
        work_tile = make_fc12_done_tile(True)
        if stage == Int32(_SHARED_FC1):
            work_tile, stage, next_id, work_id_worker = self._claim_shared_tile(work_id_worker, stage, next_id, 0)
        if cutlass.const_expr(self.shared_fc2_position == "early"):
            if stage == Int32(_SHARED_FC2):
                work_tile, stage, next_id, work_id_worker = self._claim_shared_tile(work_id_worker, stage, next_id, 1)
        if not work_tile.is_valid_tile:
            if not routed_done:
                # Stores on ``self`` make the DSL carry the state touched by the routed call out of this branch.
                if cutlass.const_expr(self._expert_sizes_ready_pointer is not None):
                    self._expert_sizes_ready = self._wait_expert_sizes_ready(self._expert_sizes_ready)
                self._work_id_worker = work_id_worker
                work_tile = self._gen_next_routed_work(work_id_offset)
                work_id_worker = self._work_id_worker
                if not work_tile.is_valid_tile:
                    routed_done = Boolean(True)
            if cutlass.const_expr(self.shared_fc2_position == "tail"):
                if routed_done & (stage == Int32(_SHARED_FC2)):
                    work_tile, stage, next_id, work_id_worker = self._claim_shared_tile(
                        work_id_worker, stage, next_id, 1
                    )
        self._shared_stage = stage
        self._shared_next = next_id
        self._routed_done = routed_done
        self._work_id_worker = work_id_worker
        return work_tile

    def _shared_carried_names(self) -> tuple:
        return ("_shared_stage", "_shared_next", "_routed_done") if self.shared_fc2_position is not None else ()

    def _extract_shared_stream_values(self) -> list:
        values = list(extract_mlir_values(self._expert_sizes_ready))
        for name in self._shared_carried_names():
            values.extend(extract_mlir_values(getattr(self, name)))
        return values

    def _rebuild_shared_stream(self, result, rebuild) -> None:
        """Copy the static stream fields onto ``result`` and rebuild the loop-carried ones through ``rebuild``."""
        for name, value in vars(self).items():
            if name.startswith("_shared_") or name.startswith("shared_") or name.startswith("_expert_sizes_ready_"):
                setattr(result, name, value)
        result._expert_sizes_ready = rebuild(self._expert_sizes_ready)
        result._routed_done = None
        for name in self._shared_carried_names():
            setattr(result, name, rebuild(getattr(self, name)))


class BlackwellFusedFc12Scheduler(SharedExpertStreamMixin, SchedulerBase):
    """Compose work-ID claim, FC12 mapping, and SMEM tile transport."""

    pipeline_mbarriers_region = "blackwell.fc12.scheduler.pipeline_mbarriers"
    work_tiles_region = "blackwell.fc12.scheduler.work_tiles"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            **shared_expert_problem_requirements(),
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
            "compute_cluster_count": OptionalRequirement(int),
            **shared_expert_impl_requirements(),
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
        self.launch_cluster_cnt_merge_as_preferred = self.non_clc_mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        self.compute_cluster_count = impl_desc.get("compute_cluster_count", self.launch_cluster_cnt_merge_as_preferred)
        self.work_tile_type = SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo
        if self.hint is None:
            self.hint = self.compute_cluster_count
            self.group_hint = self.hint

        self._validate_configuration()
        self.expert_split_factor = 1
        self.expert_split_chunk_tokens = 0
        self.raster_group_n = 0
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

        self._bind_shared_expert_stream(
            problem_desc,
            impl_desc,
            cta_tile_m=launch_cta_tile_shape_mnk[0],
            cta_tile_n=launch_cta_tile_shape_mnk[1],
            routed_stream_count=1,
        )
        if self.work_id_mode == "cluster_launch_control":
            self._work_id_worker = ClcMixedCgaSchedulerWorker(config=self.non_clc_mixed_cga_config)
        else:
            self._work_id_worker = NonClcMixedCgaSchedulerWorker(
                config=self.non_clc_mixed_cga_config,
                work_id_mode=self.work_id_mode,
                stream_count=1 + self.shared_stream_count,
                work_id_stride=self.compute_cluster_count,
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
        if (
            isinstance(self.compute_cluster_count, bool)
            or not isinstance(self.compute_cluster_count, int)
            or not 0 < self.compute_cluster_count <= self.launch_cluster_cnt_merge_as_preferred
        ):
            raise ValueError(
                f"compute_cluster_count must be a Python int in (0, {self.launch_cluster_cnt_merge_as_preferred}], got "
                f"{self.compute_cluster_count}."
            )
        if self.work_id_mode not in ("grid_stride", "atomic_counter", "cluster_launch_control"):
            raise ValueError(
                "work_id_mode must be 'grid_stride', 'atomic_counter', or "
                f"'cluster_launch_control', got {self.work_id_mode!r}."
            )
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        uses_peer_lane_protocol = self.work_id_mode in ("atomic_counter", "cluster_launch_control")
        if uses_peer_lane_protocol and cluster_size > 32:
            raise ValueError("The per-lane peer barrier protocol supports at most 32 CTAs per cluster.")
        fallback_cluster_shape = self.non_clc_mixed_cga_config.fallback_cluster_shape
        if fallback_cluster_shape is not None:
            fallback_cluster_size = fallback_cluster_shape[0] * fallback_cluster_shape[1]
            if uses_peer_lane_protocol and fallback_cluster_size > 32:
                raise ValueError("The per-lane peer barrier protocol supports at most 32 CTAs per fallback cluster.")
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

    # CUDA caps gridDim.z at 65535; CLC tickets ride on blockIdx.z.
    _clc_grid_z_limit = 65535

    def clc_ticket_count(self, total_token_rows):
        """Return the CLC ticket count, an upper bound on FC12 tasks (surplus tickets decode to DONE)."""
        if not isinstance(self.expert_count, int):
            raise ValueError("cluster_launch_control needs a static expert_count to size the ticket grid.")
        if not isinstance(self.intermediate_gateup_size, int) or not isinstance(self.hidden_size, int):
            raise ValueError("cluster_launch_control needs static intermediate/hidden sizes to size the ticket grid.")
        if isinstance(total_token_rows, bool) or not isinstance(total_token_rows, int) or total_token_rows < 0:
            raise TypeError(
                "cluster_launch_control needs a host-known non-negative int total_token_rows; the grid "
                "must be validated against the gridDim.z limit before launch."
            )
        mapping_cluster_tile_m = self.mapping_cta_tile_shape_mnk[0] * self.mapping_cluster_shape_mn[0]
        mapping_cluster_tile_n = self.mapping_cta_tile_shape_mnk[1] * self.mapping_cluster_shape_mn[1]
        num_fc1_blocks = (self.intermediate_gateup_size + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
        num_fc2_blocks = (self.hidden_size + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
        token_blocks = (total_token_rows + mapping_cluster_tile_m - 1) // mapping_cluster_tile_m + (
            self.expert_count - 1
        )
        # Zero rows still needs one cluster to bootstrap, observe DONE and exit (gridDim.z >= 1).
        return max(1, token_blocks * (num_fc1_blocks + num_fc2_blocks))

    def get_grid_shape(
        self,
        *,
        max_active_clusters: Optional[int] = None,
        problem_desc=None,
        total_token_rows=None,
    ) -> Tuple[int, int, int]:
        """Return the launch grid in GEMM-domain orientation (full ticket grid under cluster launch control)."""
        if self.work_id_mode == "cluster_launch_control":
            if total_token_rows is None:
                raise ValueError("cluster_launch_control requires total_token_rows to size the ticket grid.")
            ticket_count = self.clc_ticket_count(total_token_rows)
            if ticket_count > self._clc_grid_z_limit:
                raise ValueError(
                    f"cluster_launch_control ticket count {ticket_count} exceeds gridDim.z limit "
                    f"{self._clc_grid_z_limit}."
                )
            return (self.cluster_shape_mn[0], self.cluster_shape_mn[1], ticket_count)
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
        return (self.cluster_shape_mn[0], self.cluster_shape_mn[1], self.launch_cluster_cnt_merge_as_preferred)

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
            raise ValueError("Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided.")
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

        if cutlass.const_expr(self.expert_split_factor > 1):
            expert_count = expert_count * self.expert_split_factor
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
            expert_split_factor=self.expert_split_factor,
            expert_split_chunk_tokens=self.expert_split_chunk_tokens,
            raster_group_n=self.raster_group_n,
        )

        self._task_mapping_state = task_mapping_state
        self._assign_shared_expert_stream(block_idx, device_workspace)

    @cute.jit
    def _gen_next_routed_work(self, work_id_offset: Int32 = Int32(0)) -> SchedulerWorkTileBase:
        """Claim and map one work tile without first-tile prefetch."""
        work_id = self._work_id_worker.claim_next_work() + work_id_offset
        cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
            self._work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
        )
        if cutlass.const_expr(self.work_id_mode == "cluster_launch_control"):
            # CLC tickets on a mixed-cluster launch are not monotonic; the mapper is forward-only, so
            # rewind when the ticket falls before the cursor's current segment.
            if self._work_id_worker.claimed_is_valid:
                if work_id < self._task_mapping_state.cursor_state.current_expert_tile_start:
                    self._task_mapping_state = rewind_fc12_mapping_cursor(self._task_mapping_state)
        work_tile, self._task_mapping_state = map_fc12_linear_work_id(
            work_id, cta_id_in_mapping_cluster, self._task_mapping_state
        )
        if cutlass.const_expr(self.work_id_mode == "cluster_launch_control"):
            # A failed try_cancel means no cluster is left to steal: terminate this CTA's stream.
            mapped_tile = work_tile
            work_tile = make_fc12_done_tile(self.is_swap_ab)
            if self._work_id_worker.claimed_is_valid:
                work_tile = mapped_tile
        return work_tile

    @cute.jit
    def produce_tail(self) -> None:
        """Drain the work transport, then the work-ID leader waits for the cluster's final release."""
        super().produce_tail()
        self._work_id_worker.produce_tail()

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        values.extend(extract_mlir_values(self._work_id_worker))
        values.extend(extract_mlir_values(self._task_mapping_state))
        values.extend(self._extract_shared_stream_values())
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
            rebuilt_state = new_from_mlir_values(state, values[value_index : value_index + state_value_count])
            value_index += state_value_count
            return rebuilt_state

        result._work_id_worker = rebuild(self._work_id_worker)
        result._task_mapping_state = rebuild(self._task_mapping_state)
        self._rebuild_shared_stream(result, rebuild)
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
            "compute_cluster_count",
            "non_clc_mixed_cga_config",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
            "expert_split_factor",
            "expert_split_chunk_tokens",
            "raster_group_n",
        ):
            setattr(result, field_name, getattr(self, field_name))
        return result


class BlackwellFusedExtFc12Scheduler(BlackwellFusedFc12Scheduler):
    """BlackwellFusedFc12Scheduler plus streaming early-exit for CTA-specialized stages.

    Early-exit clusters (``blockIdx.z < early_exit_cluster_count``) stop claiming new
    work once their mapping cursor reaches ``early_exit_expert_idx`` and leave the
    scheduler loop with a done tile, so the CTA can stream into the next stage while
    the remaining clusters finish the tail experts.  Only ``atomic_counter`` claims
    keep unclaimed tiles reachable, so ``configure_early_exit`` rejects every other
    work-id mode.

    ``early_exit_cluster_start`` / ``early_exit_work_id_limit`` is the work-id cutoff
    of ``ContiguousFc1Scheduler``: clusters with ``blockIdx.z >= early_exit_cluster_start``
    stop claiming at work id ``early_exit_work_id_limit`` (atomic claims only).
    """

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "early_exit_cluster_count": OptionalRequirement(int),
            "early_exit_expert_idx": OptionalRequirement(Optional[int]),
            "early_exit_cluster_start": OptionalRequirement(Optional[int]),
            "early_exit_work_id_limit": OptionalRequirement(Optional[int]),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        self.early_exit_cluster_count = int(impl_desc.get("early_exit_cluster_count", 0) or 0)
        self.early_exit_expert_idx = impl_desc.get("early_exit_expert_idx", None)
        self.early_exit_cluster_start = impl_desc.get("early_exit_cluster_start", None)
        self.early_exit_work_id_limit = impl_desc.get("early_exit_work_id_limit", None)

    def configure_early_exit(self, cluster_count: int, expert_idx: int) -> None:
        """Select the early-exit clusters and the expert cutoff from host code."""
        if self.work_id_mode != "atomic_counter":
            raise ValueError(
                "Routed early-exit requires scheduler work_id_mode='atomic_counter'. "
                "grid_stride and cluster_launch_control drop the tiles an early-exit "
                "cluster stops claiming."
            )
        self.early_exit_cluster_count = int(cluster_count)
        self.early_exit_expert_idx = int(expert_idx)

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
        """Initialize base FC12 state, then the early-exit cluster predicate."""
        if cutlass.const_expr(
            self.early_exit_cluster_start is not None
            and self.early_exit_work_id_limit is not None
        ):
            if cutlass.const_expr(self.work_id_mode != "atomic_counter"):
                raise ValueError("The work-id early-exit cutoff requires scheduler work_id_mode='atomic_counter'.")
            self._work_id_worker.early_exit_cluster_start = self.early_exit_cluster_start
            self._work_id_worker.early_exit_work_id_limit = self.early_exit_work_id_limit
        super().assign_device_members(
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            actual_expert_shape=actual_expert_shape,
            block_idx=block_idx,
            smem_workspace=smem_workspace,
            smem_base=smem_base,
            device_workspace=device_workspace,
            is_fallback_cluster=is_fallback_cluster,
        )
        if cutlass.const_expr(
            self.early_exit_cluster_count > 0
            and self.early_exit_expert_idx is not None
        ):
            self._is_early_exit_cluster = Int32(block_idx[2]) < Int32(self.early_exit_cluster_count)

    @cute.jit
    def gen_next_work(self, work_id_offset: Int32 = Int32(0)) -> SchedulerWorkTileBase:
        """Claim and map one work tile; early-exit clusters stop claiming past the cutoff."""
        early_exit_enabled = (
            self.early_exit_cluster_count > 0
            and self.early_exit_expert_idx is not None
        )
        if cutlass.const_expr(early_exit_enabled and self.work_id_mode != "atomic_counter"):
            raise ValueError("Routed early-exit requires scheduler work_id_mode='atomic_counter'.")
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        if cutlass.const_expr(early_exit_enabled):
            work_id_worker = self._work_id_worker
            task_mapping_state = self._task_mapping_state
            should_claim = Boolean(True)
            if (
                self._is_early_exit_cluster
                and task_mapping_state.cursor_state.current_expert_idx
                >= Int32(self.early_exit_expert_idx * self.expert_split_factor)
            ):
                should_claim = Boolean(False)
            if should_claim:
                work_id = work_id_worker.claim_next_work() + work_id_offset
                cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                    work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
                )
                work_tile, task_mapping_state = map_fc12_linear_work_id(
                    work_id, cta_id_in_mapping_cluster, task_mapping_state
                )
            self._work_id_worker = work_id_worker
            self._task_mapping_state = task_mapping_state
        else:
            work_tile = super().gen_next_work(work_id_offset)
        return work_tile

    @cute.jit
    def produce_tail(self) -> None:
        """Drain the work transport, then the CLC leader's final release.

        Spelled out without ``super().produce_tail()``: a zero-arg ``super``
        inside a jit method inherited by a subclass re-enters the inherited
        method in this CuTeDSL build (the MRO scan restarts at ``type(self)``),
        which turns the inherited tail into unbounded recursion.  The two steps
        mirror ``SchedulerBase.produce_tail`` and the CLC tail of
        ``BlackwellFusedFc12Scheduler.produce_tail``.
        """
        self._pipeline.producer_tail(self._producer_state)
        if cutlass.const_expr(self.work_id_mode == "cluster_launch_control"):
            self._work_id_worker.produce_tail()

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        if (
            self.early_exit_cluster_count > 0
            and self.early_exit_expert_idx is not None
        ):
            values.extend(extract_mlir_values(self._is_early_exit_cluster))
        return values

    def __new_from_mlir_values__(self, values: list) -> "BlackwellFusedExtFc12Scheduler":
        early_exit_enabled = (
            self.early_exit_cluster_count > 0
            and self.early_exit_expert_idx is not None
        )
        base_value_count = len(super().__extract_mlir_values__())
        expected_value_count = base_value_count + (1 if early_exit_enabled else 0)
        if len(values) != expected_value_count:
            raise ValueError(
                "BlackwellFusedExtFc12Scheduler MLIR value count mismatch: "
                f"expected {expected_value_count}, got {len(values)}."
            )
        result = super().__new_from_mlir_values__(values[:base_value_count])
        result.early_exit_cluster_count = self.early_exit_cluster_count
        result.early_exit_expert_idx = self.early_exit_expert_idx
        result.early_exit_cluster_start = self.early_exit_cluster_start
        result.early_exit_work_id_limit = self.early_exit_work_id_limit
        if early_exit_enabled:
            result._is_early_exit_cluster = new_from_mlir_values(
                self._is_early_exit_cluster, values[base_value_count:]
            )
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
        shared_num_tokens: Int32,
    ) -> None:
        self.prologue_remaining = prologue_remaining
        self.cycle_position = cycle_position
        self.fc1_exhausted = fc1_exhausted
        self.fc2_exhausted = fc2_exhausted
        self.prologue_claims_synchronized = prologue_claims_synchronized
        self.shared_num_tokens = shared_num_tokens

    def _fields(self) -> Tuple:
        return (
            self.prologue_remaining,
            self.cycle_position,
            self.fc1_exhausted,
            self.fc2_exhausted,
            self.prologue_claims_synchronized,
            self.shared_num_tokens,
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
            rebuilt_fields.append(new_from_mlir_values(field, values[value_index : value_index + field_value_count]))
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(
                f"_PhaseInterleaveControlState MLIR value count mismatch: consumed {value_index}, got {len(values)}."
            )
        return type(self)(*rebuilt_fields)


class PhaseInterleavedFc12Scheduler(SharedExpertStreamMixin, SchedulerBase):
    """Schedule independent FC1 and FC2 streams with per-phase atomic counters."""

    pipeline_mbarriers_region = "fc12.phase_interleaved.scheduler.pipeline_mbarriers"
    work_tiles_region = "fc12.phase_interleaved.scheduler.work_tiles"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "intermediate_gateup_size": int,
            "hidden_size": int,
            **shared_expert_problem_requirements(),
        }

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
            "dfc1_m_group": OptionalRequirement(int),
            "defer_consumer_until_full_ready": OptionalRequirement(bool),
            "fuse_ready_probe_and_linear1_claim": OptionalRequirement(bool),
            **_mixed_cga_impl_requirements(),
            "compute_cluster_count": OptionalRequirement(int),
            **shared_expert_impl_requirements(),
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
        self.dfc1_m_group = impl_desc.get("dfc1_m_group", 1)
        self.defer_consumer_until_full_ready = impl_desc.get(
            "defer_consumer_until_full_ready", False
        )
        self.fuse_ready_probe_and_linear1_claim = impl_desc.get(
            "fuse_ready_probe_and_linear1_claim", False
        )
        self.non_clc_mixed_cga_config = _make_non_clc_mixed_cga_config(impl_desc)
        self.launch_cluster_cnt_merge_as_preferred = self.non_clc_mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        self.compute_cluster_count = impl_desc.get("compute_cluster_count", self.launch_cluster_cnt_merge_as_preferred)
        self.work_tile_type = SwapAbFc12WorkTileInfo if self.is_swap_ab else NonSwapAbFc12WorkTileInfo
        # (output blocks, tokens per tile, cluster M) once claim_shared_fc2_tiles is called.
        self.shared_fc2_claim_shape = None

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
        self._bind_shared_expert_stream(
            problem_desc,
            impl_desc,
            cta_tile_m=launch_cta_tile_shape_mnk[0],
            cta_tile_n=launch_cta_tile_shape_mnk[1],
            routed_stream_count=2,
        )
        self._work_id_worker = NonClcMixedCgaSchedulerWorker(
            config=self.non_clc_mixed_cga_config,
            work_id_mode=self.work_id_mode,
            stream_count=2 + self.shared_stream_count,
        )

        mapping_cluster_tile_n = self.mapping_cta_tile_shape_mnk[1] * self.mapping_cluster_shape_mn[1]
        self.blocks_fc1 = (self.intermediate_gateup_size + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
        self.blocks_fc2 = (self.hidden_size + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
        self.fc2_claims_per_token_block = self.blocks_fc2
        interleave_gcd = math.gcd(
            self.blocks_fc1, self.fc2_claims_per_token_block
        )
        self.interleave_fc2_slots = (
            self.fc2_claims_per_token_block // interleave_gcd
        )
        self.interleave_cycle_length = (
            self.blocks_fc1 + self.fc2_claims_per_token_block
        ) // interleave_gcd
        self.minimum_global_fc1_claims = minimum_phase_interleave_fc1_claims(
            blocks_fc1=self.blocks_fc1,
            blocks_fc2=self.blocks_fc2,
            launch_cluster_cnt_merge_as_preferred=self.compute_cluster_count,
        )
        minimum_hint = minimum_phase_interleave_hint(
            blocks_fc1=self.blocks_fc1,
            blocks_fc2=self.blocks_fc2,
            launch_cluster_cnt_merge_as_preferred=self.compute_cluster_count,
        )
        if (
            self.fc1_prologue_tiles < minimum_hint
            and not (
                self.fc1_prologue_tiles == 0
                and self.defer_consumer_until_full_ready
            )
        ):
            raise ValueError(
                f"phase_interleave hint {self.fc1_prologue_tiles} cannot cover a "
                f"{self.compute_cluster_count}-cluster FC2 claim wave; "
                f"raise hint to at least {minimum_hint}."
            )

    def _validate_configuration(self) -> None:
        if len(self.mma_tiler_mnk) != 3:
            raise ValueError("mma_tiler_mnk must contain three dimensions.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must contain two dimensions.")
        if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in self.mma_tiler_mnk):
            raise TypeError("mma_tiler_mnk dimensions must be Python ints.")
        if not all(
            isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in self.cluster_shape_mn
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
            or self.fc1_prologue_tiles < 0
        ):
            raise ValueError(
                "fc1_prologue_tiles must be a non-negative Python int "
                "resolved by the kernel frontend."
            )
        if type(self.dfc1_m_group) is not int or self.dfc1_m_group not in (1, 16):
            raise ValueError("dfc1_m_group must be 1 or 16")
        if self.dfc1_m_group > 1 and (
            self.is_swap_ab
            or not self.defer_consumer_until_full_ready
        ):
            raise ValueError(
                "Grouped dFC1 M traversal requires non-swap singleton consumers, "
                "ordinary producer order and full-ready admission"
            )
        if not isinstance(self.defer_consumer_until_full_ready, bool):
            raise TypeError("defer_consumer_until_full_ready must be a bool.")
        if not isinstance(self.fuse_ready_probe_and_linear1_claim, bool):
            raise TypeError(
                "fuse_ready_probe_and_linear1_claim must be a bool."
            )
        if (
            self.defer_consumer_until_full_ready
            and self.non_clc_mixed_cga_config.is_mixed
        ):
            raise NotImplementedError(
                "defer_consumer_until_full_ready does not yet support fixed "
                "preferred/fallback cluster groups."
            )
        if self.fuse_ready_probe_and_linear1_claim:
            if not self.defer_consumer_until_full_ready:
                raise ValueError(
                    "fuse_ready_probe_and_linear1_claim requires "
                    "defer_consumer_until_full_ready=True."
                )
        if self.fc1_prologue_tiles == 0 and not self.defer_consumer_until_full_ready:
            raise ValueError("fc1_prologue_tiles=0 requires deferred full-readiness.")
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.sf_padding_block <= 0:
            raise ValueError("sf_padding_block must be positive.")
        if self.launch_cluster_cnt_merge_as_preferred <= 0:
            raise ValueError("launch_cluster_cnt_merge_as_preferred must be positive.")
        if (
            isinstance(self.compute_cluster_count, bool)
            or not isinstance(self.compute_cluster_count, int)
            or not 0 < self.compute_cluster_count <= self.launch_cluster_cnt_merge_as_preferred
        ):
            raise ValueError(
                f"compute_cluster_count must be a Python int in (0, {self.launch_cluster_cnt_merge_as_preferred}], got "
                f"{self.compute_cluster_count}."
            )
        if self.work_id_mode != "atomic_counter":
            raise ValueError("Phase-interleaved FC12 scheduling currently requires work_id_mode='atomic_counter'.")
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        if cluster_size > 32:
            raise ValueError("The atomic broadcast protocol supports at most 32 CTAs per cluster.")
        fallback_cluster_shape = self.non_clc_mixed_cga_config.fallback_cluster_shape
        if fallback_cluster_shape is not None:
            fallback_cluster_size = fallback_cluster_shape[0] * fallback_cluster_shape[1]
            if fallback_cluster_size > 32:
                raise ValueError("The atomic broadcast protocol supports at most 32 CTAs per fallback cluster.")
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
        self,
        *,
        max_active_clusters: Optional[int] = None,
        problem_desc=None,
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
        return (self.cluster_shape_mn[0], self.cluster_shape_mn[1], self.launch_cluster_cnt_merge_as_preferred)

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
            raise ValueError("Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided.")
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
            dfc1_m_group=self.dfc1_m_group,
        )
        self._control_state = _PhaseInterleaveControlState(
            prologue_remaining=Int32(self.fc1_prologue_tiles),
            # Stagger the steady-state cadence by physical cluster.  Starting
            # every persistent cluster at slot zero turns an intended rolling
            # mix into chip-wide FC1/FC2 bursts; the phase offset keeps both
            # streams represented in every claim wave after the prologue.
            cycle_position=(
                Int32(block_idx[2]) % Int32(self.interleave_cycle_length)
            ),
            fc1_exhausted=Boolean(False),
            fc2_exhausted=Boolean(False),
            prologue_claims_synchronized=Boolean(
                self.defer_consumer_until_full_ready
            ),
            shared_num_tokens=Int32(0),
        )
        self._assign_shared_expert_stream(block_idx, device_workspace)

    def claim_shared_fc2_tiles(
        self, *, output_blocks: int, tokens_per_tile: int, cluster_m: int
    ) -> None:
        """Continue the FC2 claim stream into the fused shared expert's down-projection tiles, so the clusters
        that finish their routed work first take more of them."""
        if not self.is_swap_ab:
            raise ValueError("Shared down-projection claims need the swap-AB mapping.")
        self.shared_fc2_claim_shape = (output_blocks, tokens_per_tile, cluster_m)

    def set_shared_fc2_tokens(self, num_tokens: Int32) -> None:
        """The shared expert's (local) token count, which sizes its claimed down-projection tiles."""
        self._control_state.shared_num_tokens = num_tokens

    def _map_shared_fc2_work_id(
        self, shared_work_id: Int32, cta_id_in_mapping_cluster: cute.Coord, num_tokens: Int32
    ) -> Tuple[SchedulerWorkTileBase, Boolean]:
        """Map an FC2 ID's offset past the routed tiles to a shared down-projection tile."""
        output_blocks, tokens_per_tile, cluster_m = self.shared_fc2_claim_shape
        token_blocks = (num_tokens + Int32(tokens_per_tile - 1)) // Int32(tokens_per_tile)
        is_valid = shared_work_id < token_blocks * Int32(output_blocks)
        token_block = shared_work_id // Int32(output_blocks)
        output_block = shared_work_id - token_block * Int32(output_blocks)
        valid_tokens = cutlass.min(
            num_tokens - token_block * Int32(tokens_per_tile), Int32(tokens_per_tile)
        )
        work_tile = SwapAbFc12WorkTileInfo(
            expert_idx=Int32(0),
            tile_m_idx=output_block * Int32(cluster_m) + cta_id_in_mapping_cluster[1],
            tile_n_idx=token_block,
            cumulative_data_physical_row=Int32(0),
            cumulative_sf_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_cta_tile=valid_tokens,
            phase_and_flags=Int32(BlockPhase.SharedLinear2),
        )
        return work_tile, is_valid

    @cute.jit
    def cluster_uniform_counter_ready(
        self,
        counter_pointer: cute.Pointer,
        ready_threshold: Int32,
        ready_is_mask: bool = False,
    ) -> Boolean:
        """Observe one counter on the cluster leader and broadcast readiness."""
        if cutlass.const_expr(not self.defer_consumer_until_full_ready):
            raise ValueError(
                "cluster readiness polling requires "
                "defer_consumer_until_full_ready=True."
            )
        work_id_worker = self._work_id_worker
        ready = work_id_worker.cluster_uniform_counter_ready(
            counter_pointer, ready_threshold, ready_is_mask
        )
        self._work_id_worker = work_id_worker
        return ready

    @cute.jit
    def gen_next_linear1_work_unless_counter_ready(
        self,
        counter_pointer: cute.Pointer,
        ready_threshold: Int32,
        ready_is_mask: bool = False,
    ) -> Tuple[Boolean, SchedulerWorkTileBase]:
        """Probe full readiness and otherwise claim one Linear1 tile.

        When Linear1 still has unclaimed work, the readiness load and atomic
        claim share one cluster broadcast transaction.  Once the stream is
        exhausted this falls back to a probe-only transaction, avoiding
        repeated tail over-claims while the held consumer drains.
        """
        if cutlass.const_expr(not self.fuse_ready_probe_and_linear1_claim):
            raise ValueError(
                "fused readiness/Linear1 claims require "
                "fuse_ready_probe_and_linear1_claim=True."
            )

        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        fc1_exhausted = control_state.fc1_exhausted
        ready = Boolean(False)
        if task_mapping_state.stream_token_blocks < Int32(0):
            task_mapping_state = resolve_phase_interleaved_stream_extent(task_mapping_state)

        if fc1_exhausted:
            ready = work_id_worker.cluster_uniform_counter_ready(
                counter_pointer, ready_threshold, ready_is_mask
            )
        else:
            ready, linear_work_id = (
                work_id_worker.cluster_uniform_counter_ready_or_claim(
                    counter_pointer,
                    ready_threshold,
                    claim_stream_index=Int32(0),
                    ready_is_mask=ready_is_mask,
                )
            )
            if not ready:
                cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                    work_id_worker.cta_coord_in_preferred_cluster,
                    self.is_swap_ab,
                )
                work_tile, stream_has_work, task_mapping_state = (
                    map_phase_interleaved_fc12_work_id(
                        linear_work_id,
                        Int32(BlockPhase.Linear1),
                        cta_id_in_mapping_cluster,
                        task_mapping_state,
                    )
                )
                if not stream_has_work:
                    fc1_exhausted = Boolean(True)

        control_state.fc1_exhausted = fc1_exhausted
        self._work_id_worker = work_id_worker
        self._task_mapping_state = task_mapping_state
        self._control_state = control_state
        return ready, work_tile


    @cute.jit
    def gen_next_linear1_work(self) -> SchedulerWorkTileBase:
        """Force one dependency-free Linear1 claim while a Linear2 tile is held.

        This corrective claim is outside the normal rolling cadence: the
        deferred Linear2 claim already consumed its cadence slot.  The caller
        must publish a returned Linear1 tile before rechecking or publishing
        the held consumer.  Consequently no producer descriptor can become
        trapped behind an unready consumer.
        """
        if cutlass.const_expr(not self.defer_consumer_until_full_ready):
            raise ValueError(
                "forced Linear1 claims require "
                "defer_consumer_until_full_ready=True."
            )

        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        fc1_exhausted = control_state.fc1_exhausted
        if task_mapping_state.stream_token_blocks < Int32(0):
            task_mapping_state = resolve_phase_interleaved_stream_extent(task_mapping_state)

        if not fc1_exhausted:
            linear_work_id = work_id_worker.claim_next_work(
                Int32(0)
            )
            cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                work_id_worker.cta_coord_in_preferred_cluster,
                self.is_swap_ab,
            )
            work_tile, stream_has_work, task_mapping_state = (
                map_phase_interleaved_fc12_work_id(
                    linear_work_id,
                    Int32(BlockPhase.Linear1),
                    cta_id_in_mapping_cluster,
                    task_mapping_state,
                )
            )
            if not stream_has_work:
                fc1_exhausted = Boolean(True)

        control_state.fc1_exhausted = fc1_exhausted
        self._work_id_worker = work_id_worker
        self._task_mapping_state = task_mapping_state
        self._control_state = control_state
        return work_tile

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
            claim_target, stream_ends_before_target = resolve_phase_interleaved_fc1_claim_target(
                Int32(self.minimum_global_fc1_claims), task_mapping_state
            )
            spin_wait(fc1_counter_pointer, lambda value: value >= claim_target)
            # Exhaustion comes from the mapping, not the live counter, so every CTA of a cluster agrees on it.
            if stream_ends_before_target:
                fc1_exhausted = Boolean(True)
        return fc1_exhausted

    @cute.jit
    def _gen_next_routed_work(self, work_id_offset: Int32 = Int32(0)) -> SchedulerWorkTileBase:
        """Claim until one stream yields valid work or both streams terminate."""
        _ = work_id_offset
        work_tile = make_fc12_done_tile(self.is_swap_ab)
        work_id_worker = self._work_id_worker
        task_mapping_state = self._task_mapping_state
        control_state = self._control_state
        prologue_remaining = control_state.prologue_remaining
        cycle_position = control_state.cycle_position
        fc1_exhausted = control_state.fc1_exhausted
        fc2_exhausted = control_state.fc2_exhausted
        stream_has_work = Boolean(False)
        prologue_claims_synchronized = control_state.prologue_claims_synchronized
        resolved = Boolean(False)
        # The first call runs after the expert sizes are ready.
        if task_mapping_state.stream_token_blocks < Int32(0):
            task_mapping_state = resolve_phase_interleaved_stream_extent(task_mapping_state)

        while not resolved:
            if fc1_exhausted and fc2_exhausted:
                work_tile = make_fc12_done_tile(self.is_swap_ab)
                resolved = Boolean(True)

            if not resolved:
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
                cta_id_in_mapping_cluster = _to_fc12_mapping_cta_coord(
                    work_id_worker.cta_coord_in_preferred_cluster, self.is_swap_ab
                )
                phase = Int32(BlockPhase.Linear2)
                if want_fc1:
                    phase = Int32(BlockPhase.Linear1)
                work_tile, stream_has_work, task_mapping_state = (
                    map_phase_interleaved_fc12_work_id(
                        linear_work_id,
                        phase,
                        cta_id_in_mapping_cluster,
                        task_mapping_state,
                    )
                )
                if cutlass.const_expr(self.shared_fc2_claim_shape is not None):
                    if (not want_fc1) and (not stream_has_work):
                        work_tile, stream_has_work = self._map_shared_fc2_work_id(
                            linear_work_id
                            - task_mapping_state.stream_tiles(
                                task_mapping_state.fc2_claims_per_token_block
                            ),
                            cta_id_in_mapping_cluster,
                            control_state.shared_num_tokens,
                        )
                if stream_has_work:
                    # Deferred admission starts synchronized, so the wait is not emitted for it (DSL 4.10's
                    # remove-dead-values pass poisons the never-taken spin and LLVM then drops the claim path).
                    if cutlass.const_expr(not self.defer_consumer_until_full_ready):
                        if (not want_fc1) and (not prologue_claims_synchronized):
                            fc1_exhausted = self._wait_for_fc1_prologue_claims(
                                fc1_exhausted, work_id_worker, task_mapping_state
                            )
                            prologue_claims_synchronized = Boolean(True)
                    if prologue_remaining > Int32(0):
                        prologue_remaining = prologue_remaining - Int32(1)
                    else:
                        cycle_position = (cycle_position + Int32(1)) % Int32(self.interleave_cycle_length)
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

    @cute.jit
    def produce_tail(self) -> None:
        """Drain the work transport, then the work-ID leader waits for the cluster's final release."""
        super().produce_tail()
        self._work_id_worker.produce_tail()

    def __extract_mlir_values__(self) -> list:
        values = super().__extract_mlir_values__()
        for state in (self._work_id_worker, self._task_mapping_state, self._control_state):
            values.extend(extract_mlir_values(state))
        values.extend(self._extract_shared_stream_values())
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
            rebuilt_state = new_from_mlir_values(state, values[value_index : value_index + state_value_count])
            value_index += state_value_count
            return rebuilt_state

        result._work_id_worker = rebuild(self._work_id_worker)
        result._task_mapping_state = rebuild(self._task_mapping_state)
        result._control_state = rebuild(self._control_state)
        self._rebuild_shared_stream(result, rebuild)
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
            "compute_cluster_count",
            "non_clc_mixed_cga_config",
            "mapping_cta_tile_shape_mnk",
            "mapping_cluster_shape_mn",
            "blocks_fc1",
            "blocks_fc2",
            "dfc1_m_group",
            "defer_consumer_until_full_ready",
            "fuse_ready_probe_and_linear1_claim",
            "fc2_claims_per_token_block",
            "interleave_fc2_slots",
            "interleave_cycle_length",
            "minimum_global_fc1_claims",
            "shared_fc2_claim_shape",
        ):
            setattr(result, field_name, getattr(self, field_name))
        return result


__all__ = [
    "BlackwellFusedFc12Scheduler",
    "BlackwellFusedExtFc12Scheduler",
    "PhaseInterleavedFc12Scheduler",
    "minimum_phase_interleave_hint",
    "minimum_phase_interleave_ring_token_blocks",
]
