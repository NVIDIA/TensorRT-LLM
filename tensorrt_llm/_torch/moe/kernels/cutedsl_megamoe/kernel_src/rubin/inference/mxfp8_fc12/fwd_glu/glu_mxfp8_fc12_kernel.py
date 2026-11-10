# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Fused fc1+fc2 GLU MXFP8 MegaMoE kernel for SM107.
"""

import dataclasses
from typing import Any, Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import CollectorOp

from ......api import ImplDesc, ProblemDesc
from ......communication.nvlink_domain.token_comm import TokenCommArgs
from ......helpers.device_workspace import DeviceWorkspace
from ......helpers.dsl_helpers import spin_wait, tma_multicast_mask
from ......helpers.iket_compat import iket
from ......helpers.smem_workspace import SmemWorkspace
from .....schedulers.fc12_mapping import BlockPhase
from .....schedulers.fc12_scheduler import (
    BlackwellFusedFc12Scheduler,
    PhaseInterleavedFc12Scheduler,
    minimum_phase_interleave_hint,
)
from .....schedulers.non_clc_mixed_cga import NonClcMixedCgaConfig, NonClcMixedCgaSchedulerWorker
from .....blackwell.custom_mix_cga_helpers import (
    PipelineTmaUmmaMixedCga,
    TmaAtomOrPair,
    bind_executable_tma_load_fields,
    make_executable_tma_atom_or_pair,
    pipeline_init_arrive_mixed_cga,
    pipeline_init_wait_mixed_cga,
    select_tma_atom_for_cluster,
)
from .glu_mxfp8_fc12_epilogue import EpiWarpCount, Fc1CTMAStages, GluMxfp8Epilogue
from .glu_mxfp8_fc12_epilogue_breuse import GluMxfp8BreuseEpilogue
from .glu_mxfp8_fc12_extension import (
    GluMxFp8Fc12SchedExtension,
    OneShotSchedConsumer,
    WeightStorageMode,
    discrete_weight_descriptor_workspace_alignment,
    discrete_weight_descriptor_workspace_size,
    raw_discrete_weight_descriptor_pointer,
)


@dataclasses.dataclass(frozen=True)
class _EpilogueCommView:
    """The exact fields the GLU epilogue reads for cross-rank FC2 routing.

    Built inside the device kernel (so it carries no MLIR-marshaled scalars -- its fields
    are already-traced device views) and passed to ``epilogue.run(token_comm_args=...)`` on
    the MegaMoE path. Bridges next's Router-push component (``token_src_metadata`` /
    ``fc2_done_counter`` come from ``TokenComm`` accessors; ``combine_output`` is the symmetric
    ``pre_reduced_activation``) to the epilogue's ``Fc2OutputDest`` peer-store expectations.
    """

    token_src_metadata: Any
    combine_output: Any
    peer_rank_ptr_mapper: Any
    fc2_output_sf: Any = None
    fc2_done_counter: Any = None
    fc2_output_workspace: Any = None


@cute.jit
def _make_executable_tma_atom_for_cluster(
    atom_or_pair: TmaAtomOrPair, is_fallback_cluster: cutlass.Boolean
) -> cute.CopyAtom:
    """Materialize the executable TMA atom of the active runtime cluster."""
    return select_tma_atom_for_cluster(
        make_executable_tma_atom_or_pair(atom_or_pair), is_fallback_cluster
    )


@cute.jit
def _int64_pointer_array_tensor(pointer_array: cute.Pointer, element_count: int):
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Int64,
            pointer_array.toint(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        ),
        cute.make_layout((element_count,)),
    )


# =============================================================================
# Sm107Mxfp8GluFc12Kernel
# =============================================================================


class Sm107Mxfp8GluFc12Kernel:
    # SMEM budget for all "non-problem-tensor" buffers (mbarriers, sched
    # work-tile buffer, TMEM allocator state).  Reserved at host side in
    # ``_compute_stages``.  Bump if ``SharedStorage`` over-allocates SMEM.
    _SmemMiscBudget = 1024

    # Supported (ab_dtype, sf_vec_size) pairings.
    # MXFP8 → Float8E4M3FN / Float8E5M2 + sf_vec_size=32  (FP8-E8M0 scales, MmaMXF8Op)
    VALID_AB_DTYPE_SF_SIZE: dict = {
        32: (
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ),
    }

    # Interleave granularity for gate and up in SwiGLU / GeGlu
    GateUpInterleave: int = 32

    # setmaxnreg split of the two-warpgroup (384-thread) layout; 168 regs/thread at entry.
    _EpiWarpRegs = 232
    _OtherWarpRegs = 40

    # Self-resetting fc1_done slot under PDL: the low 16 bits count fc1 tile
    # arrivals, the high 16 bits count fc2 consumer passes in this unit.
    FC1_DONE_FC2_PASS_UNIT = 1 << 16

    def __init__(
        self,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        schedule_mode: Literal["grouped", "phase_interleave"] = "grouped",
        load_balance_mode: Literal["static", "atomic_counter", "cluster_launch_control"] = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
        fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None,
        preferred_cluster_count: Optional[int] = None,
        fallback_cluster_count: Optional[int] = None,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Tuple[int, int] = (1, 1),
        gate_up_clamp: Optional[float] = None,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        apply_topk_in_fc1: bool = False,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
        act_func: str = "swiglu",
        combine_format: Optional[Any] = None,
        fc2_use_bulk: bool = False,
        fc2_sf_blocked: bool = False,
        one_shot_sched: bool = False,
        fc2_tma_stages: Optional[int] = None,
        use_pdl: bool = False,
        weight_storage_mode: WeightStorageMode = "contiguous",
    ) -> None:
        if not force_static_sched:
            raise NotImplementedError(
                "v1 only implements force_static_sched=True (8-warp layout: 4 epilogue + "
                "mma + tma_a + tma_b + sched; 12 warps with the two-warpgroup epilogue). "
                "Hardware work-id acquisition is load_balance_mode='cluster_launch_control'."
            )

        # Validate (ab_dtype, sf_vec_size) pairing.
        if sf_vec_size in self.VALID_AB_DTYPE_SF_SIZE:
            valid_ab = self.VALID_AB_DTYPE_SF_SIZE[sf_vec_size]
            if ab_dtype not in valid_ab:
                raise ValueError(
                    f"ab_dtype={ab_dtype.__name__} is not valid for "
                    f"sf_vec_size={sf_vec_size}. "
                    f"Expected one of: {[t.__name__ for t in valid_ab]}."
                )
        else:
            valid_sf_vec_sizes = tuple(self.VALID_AB_DTYPE_SF_SIZE)
            raise NotImplementedError(
                f"sf_vec_size must be one of {valid_sf_vec_sizes} (MXFP8); got {sf_vec_size}."
            )

        if load_balance_mode not in ("static", "atomic_counter", "cluster_launch_control"):
            raise ValueError(
                "load_balance_mode must be 'static', 'atomic_counter', or "
                f"'cluster_launch_control'; got {load_balance_mode!r}."
            )
        if one_shot_sched and load_balance_mode != "static":
            raise ValueError(
                "one_shot_sched derives every tile from blockIdx and requires "
                f"load_balance_mode='static'; got {load_balance_mode!r}."
            )
        if one_shot_sched and (static_expert_shape is None or static_expert_shape[0] != 1):
            raise ValueError(
                "one_shot_sched needs static_expert_shape=(1, intermediate_gateup, hidden) "
                f"to size its grid at trace time; got {static_expert_shape!r}."
            )
        if one_shot_sched and tuple(cluster_shape_mnk[:2]) != (1, 1):
            raise ValueError(
                "one_shot_sched maps one CTA per blockIdx.z tile and requires "
                f"cluster_shape (1,1); got {cluster_shape_mnk}."
            )
        if schedule_mode not in ("grouped", "phase_interleave"):
            raise ValueError(
                f"schedule_mode must be 'grouped' or 'phase_interleave'; got {schedule_mode!r}."
            )
        if schedule_mode == "phase_interleave" and load_balance_mode != "atomic_counter":
            raise ValueError(
                "phase_interleave requires load_balance_mode='atomic_counter'."
                f"got {schedule_mode!r}."
            )
        if weight_storage_mode not in ("contiguous", "discrete"):
            raise ValueError(
                "weight_storage_mode must be 'contiguous' or 'discrete'; "
                f"got {weight_storage_mode!r}."
            )
        if weight_storage_mode == "discrete" and static_expert_shape is None:
            raise ValueError(
                "Discrete weights require static_expert_shape so descriptor "
                "shapes can be constructed without contiguous weight tensors."
            )
        if act_func not in ("swiglu", "geglu", "situ"):
            raise ValueError(f"act_func must be 'swiglu', 'geglu', or 'situ'; got {act_func!r}.")
        if act_func == "situ":
            if situ_beta is None or situ_linear_beta is None:
                raise ValueError("act_func='situ' requires both situ_beta and situ_linear_beta.")
            if situ_beta <= 0 or situ_linear_beta <= 0:
                raise ValueError(
                    "SiTU beta parameters must be positive; got "
                    f"{situ_beta} and {situ_linear_beta}."
                )
            if gate_up_clamp is not None:
                raise ValueError("SiTU does not support gate_up_clamp.")
        elif situ_beta is not None or situ_linear_beta is not None:
            raise ValueError("situ_beta and situ_linear_beta are only valid with act_func='situ'.")
        if act_func == "geglu":
            raise NotImplementedError(
                f"act_func={act_func!r} is not yet implemented; only "
                "'swiglu' and 'situ' are currently supported."
            )

        # Supported (M,N) / 2-CTA geometries; the (128,64) tile relies on the SFB N round-up path.
        m, n, _k = mma_tiler_mnk
        _geom = ((m, n), bool(use_2cta_instrs))
        if _geom not in (
            ((256, 256), True),
            ((512, 256), True),
            ((128, 128), False),
            ((128, 64), False),
        ):
            raise ValueError(
                "Sm107Mxfp8GluFc12Kernel supports mma_tiler (256,256)/(512,256) "
                "with use_2cta_instrs=True or (128,128)/(128,64) with "
                f"use_2cta_instrs=False; got {mma_tiler_mnk}, "
                f"use_2cta_instrs={use_2cta_instrs}."
            )
        if m == 128:
            if tuple(cluster_shape_mnk[:2]) != (1, 1):
                raise ValueError(
                    f"mma_tiler (128,n) requires cluster_shape (1,1); got {cluster_shape_mnk}."
                )
            if mma_tiler_mnk[2] != 128:
                raise ValueError(f"mma_tiler (128,n) requires K tile 128; got {mma_tiler_mnk[2]}.")

        # Store ab_dtype so workspace-size helpers can use it without tensors.
        self.ab_dtype = ab_dtype
        self.c_dtype = cutlass.BFloat16
        self.act_func = act_func
        self.combine_format = combine_format

        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.epi_flag_batch = epi_flag_batch
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.generate_c = generate_c
        self.use_stg_fc1 = use_stg_fc1
        self.fc2_sf_blocked = fc2_sf_blocked
        self.one_shot_sched = one_shot_sched
        # Programmatic dependent launch: only warps that read the preceding grid's writes execute
        # griddepcontrol.wait, and the fc1_done slot self-resets (see FC1_DONE_FC2_PASS_UNIT).
        self.use_pdl = use_pdl
        self.fc2_use_bulk = fc2_use_bulk
        self.weight_storage_mode = weight_storage_mode
        self._fc2_tma_stages_arg = fc2_tma_stages
        self.fc2_tma_stages = fc2_tma_stages if fc2_tma_stages is not None else 0
        self.gate_up_clamp = abs(gate_up_clamp) if gate_up_clamp is not None else None
        self.situ_beta = float(situ_beta) if situ_beta is not None else None
        self.situ_linear_beta = float(situ_linear_beta) if situ_linear_beta is not None else None

        self.acc_dtype = acc_dtype
        self.mma_tiler_mnk = mma_tiler_mnk
        self.cluster_shape_mn = (cluster_shape_mnk[0], cluster_shape_mnk[1])
        self.fallback_cluster_shape_mn = (
            tuple(fallback_cluster_shape_mn) if fallback_cluster_shape_mn is not None else None
        )
        self.preferred_cluster_count = preferred_cluster_count
        self.fallback_cluster_count = fallback_cluster_count
        # Mirror NonClcMixedCgaConfig's degeneration rule
        self.is_mixed_cga = (
            self.fallback_cluster_shape_mn is not None
            and self.fallback_cluster_shape_mn != self.cluster_shape_mn
            and (fallback_cluster_count is None or fallback_cluster_count != 0)
        )
        self.resolved_fallback_cluster_shape_mn = (
            self.fallback_cluster_shape_mn if self.is_mixed_cga else self.cluster_shape_mn
        )
        if self.is_mixed_cga:
            fallback_m, fallback_n = self.resolved_fallback_cluster_shape_mn
            cluster_m, cluster_n = self.cluster_shape_mn
            if one_shot_sched:
                raise ValueError("mixed CGA is incompatible with one_shot_sched.")
            if cluster_n != 1 or fallback_n != 1:
                raise ValueError(
                    "mixed CGA requires cluster N == 1 in both shapes; got "
                    f"preferred {self.cluster_shape_mn}, fallback "
                    f"{self.resolved_fallback_cluster_shape_mn}."
                )
            min_fallback_m = 2 if use_2cta_instrs else 1
            if (
                cluster_m % fallback_m != 0
                or fallback_m < min_fallback_m
                or fallback_m & (fallback_m - 1)
            ):
                raise ValueError(
                    f"fallback cluster M ({fallback_m}) must be a power of two, "
                    f">= {min_fallback_m}, and divide preferred M ({cluster_m})."
                )
            if preferred_cluster_count is None or fallback_cluster_count is None:
                raise ValueError(
                    "mixed CGA requires preferred_cluster_count and fallback_cluster_count."
                )
        self.use_2cta_instrs = use_2cta_instrs
        self.force_static_sched = force_static_sched
        # static_expert_shape / clc_bundle_size / num_sched_stages
        self.static_expert_shape = static_expert_shape
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages

        # Fused fc12 sched-side knobs
        self.group_hint = group_hint
        self.schedule_mode = schedule_mode
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode

        self.sf_vec_size = sf_vec_size
        self.arch = "sm_107"

        # B-reuse: a per-CTA M of 256 runs two 128-row MMAs per k-block on one B tile (2-CTA
        # instruction only); instruction M is then decoupled from tiler M.
        _per_cta_m = mma_tiler_mnk[0] // (2 if use_2cta_instrs else 1)
        self.enable_breuse = _per_cta_m == 256
        self.mma_inst_shape_m = 256 if self.enable_breuse else mma_tiler_mnk[0]

        self._validate_mma_tiler_and_cluster_shape()
        self.mma_tiler = mma_tiler_mnk

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        # Two-warpgroup epilogue (B-reuse only): warps 0-7 drain both accumulator halves; the
        # setmaxnreg split below is the largest that does not stall at setmaxnreg.inc.
        self.breuse_two_warpgroups = self.enable_breuse
        self.num_regs_epilogue_warps = self._EpiWarpRegs if self.breuse_two_warpgroups else 0
        self.num_regs_other_warps = self._OtherWarpRegs if self.breuse_two_warpgroups else 0

        # Warp specialization (lean 8-warp / 256 thread; 12-warp / 384 thread with breuse_two_warpgroups)
        self.occupancy = 1
        if self.breuse_two_warpgroups:
            self.epilogue_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
            self.mma_warp_id = 8
            self.tma_a_warp_id = 9
            self.tma_b_warp_id = 10
            self.sched_warp_id = 11
        else:
            self.epilogue_warp_id = (0, 1, 2, 3)
            self.mma_warp_id = 4
            self.tma_a_warp_id = 5
            self.tma_b_warp_id = 6
            self.sched_warp_id = 7
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_id,
            )
        )
        if self.breuse_two_warpgroups:
            # setmaxnreg can only redistribute the per-thread register pool ptxas assigns at entry.
            _epi_threads = 32 * len(self.epilogue_warp_id)
            _other_threads = self.threads_per_cta - _epi_threads
            _regs_per_thread_at_entry = (65536 // self.threads_per_cta) // 8 * 8
            _budget = self.threads_per_cta * _regs_per_thread_at_entry
            _need = (
                _other_threads * self.num_regs_other_warps
                + _epi_threads * self.num_regs_epilogue_warps
            )
            if _need > _budget:
                raise ValueError(
                    f"setmaxnreg budget exceeded: {_other_threads}*{self.num_regs_other_warps} + "
                    f"{_epi_threads}*{self.num_regs_epilogue_warps} = {_need} > {_budget}"
                )

        # NamedBarrier IDs.
        #
        # Per-subtile rotated-leader scheme lives inside the epilogue; this
        # kernel only owns the four reserved IDs and forwards
        # them via ``self.epilog_sync_bar_id`` to the epilogue ctor.
        # IDs 8 and 9 are reserved for MXFP8 warp-pair absmax exchange.
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.epi_subtile_bar_ids = (4, 5, 6, 7)

        self.enable_token_comm: bool = False
        # Set per launch from the ``token_comm_args`` argument so that ``_setup_attributes`` applies
        # the same token_comm guards as for the MegaMoE wrapper.
        self._legacy_token_comm_args: bool = False
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_standalone: bool = False

        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

    def _validate_mma_tiler_and_cluster_shape(self) -> None:
        """Validate user-provided geometry against v1 fused-fc12 constraints.

        The geometry whitelist is in ``__init__``; this checks the derived constraints
        (B-reuse scope, SF K granularity, cluster shape, PDL slot packing). Short-N is
        handled by the swap-AB scheduler via subtile-level early-exit.
        """
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn

        if self.enable_breuse:
            assert self.use_2cta_instrs and n == 256, (
                "B-reuse tiling requires the 2-CTA instruction and mma_tiler N 256"
            )
            # B-reuse tiler scope: standalone fc12 with a static single-expert shape and K 128;
            # everything else is rejected as unsupported.
            if k != 128:
                raise NotImplementedError(
                    "B-reuse tiler (mma_tiler M 512) is only validated with "
                    f"mma_tiler K 128; got K={k}."
                )
            if self.static_expert_shape is None or self.static_expert_shape[0] != 1:
                raise NotImplementedError(
                    "B-reuse tiler (mma_tiler M 512) is supported only on the standalone "
                    "fc12 path and requires static_expert_shape=(1, intermediate_gateup, hidden); "
                    f"got {self.static_expert_shape!r}."
                )
            # Standalone fc12 lean stg_fc1 path only.
            _unsupported = {
                "one_shot_sched": self.one_shot_sched,
                "fc2_use_bulk": self.fc2_use_bulk,
                "fc2_in_kernel_topk_reduce": self.fc2_in_kernel_topk_reduce,
                "apply_topk_in_fc1": self.apply_topk_in_fc1,
                "use_stg_fc1=False": not self.use_stg_fc1,
            }
            _bad = [name for name, flag in _unsupported.items() if flag]
            if _bad:
                raise NotImplementedError(
                    "B-reuse tiling (mma_tiler M 512) supports only the lean "
                    f"stg_fc1 path; unsupported with: {', '.join(_bad)}."
                )

        sf_k_granularity = self.sf_vec_size * 4
        if k % sf_k_granularity != 0:
            raise ValueError(
                f"mma_tiler K ({k}) must be a multiple of sf_vec_size * 4 = {sf_k_granularity}"
            )

        if cm % (2 if self.use_2cta_instrs else 1) != 0:
            raise ValueError(f"cluster_shape M ({cm}) must be even when use_2cta_instrs=True")

        is_pow2 = lambda x: x > 0 and (x & (x - 1)) == 0  # noqa: E731
        if cm * cn > 16 or not is_pow2(cm) or not is_pow2(cn) or cm > 4 or cn > 4:
            raise ValueError(
                f"Invalid cluster_shape ({cm}, {cn}): each dim must be "
                f"a power of 2 and <= 4, product must be <= 16"
            )

        if cn > 2:
            raise NotImplementedError(
                f"cluster_n={cn} is not supported yet (deadlocks in the "
                f"mainloop, likely the 4-way scale-factor multicast).  "
                f"cluster_n in {{1, 2}} is supported and verified."
            )

        if cm > 4:
            raise NotImplementedError(
                f"cluster_m={cm} is not supported (cluster_m in {{1, 2, 4}})."
            )

        if cn > 1 and self.static_expert_shape is not None:
            cta_tile_n = n  # N is not split across the 2-CTA (M) pair
            cluster_tile_n = cta_tile_n * cn
            _experts, intermediate_gateup, hidden = self.static_expert_shape
            if intermediate_gateup % cluster_tile_n != 0:
                raise ValueError(
                    f"cluster_n={cn}: fc1 intermediate_gateup "
                    f"({intermediate_gateup}) must be a multiple of "
                    f"cta_tile_n * cluster_n (= {cta_tile_n} * {cn} = "
                    f"{cluster_tile_n}) to avoid ragged N-peers.  Ragged-N "
                    f"(per-peer N-store predication) is not yet supported."
                )
            if hidden % cluster_tile_n != 0:
                raise ValueError(
                    f"cluster_n={cn}: fc2 hidden ({hidden}) must be a multiple "
                    f"of cta_tile_n * cluster_n (= {cta_tile_n} * {cn} = "
                    f"{cluster_tile_n}) to avoid ragged N-peers.  Ragged-N "
                    f"(per-peer N-store predication) is not yet supported."
                )

        # Neither packed half of the fc1_done slot (see FC1_DONE_FC2_PASS_UNIT) may overflow within a token band.
        if self.use_pdl and self.static_expert_shape is not None:
            _experts, intermediate_gateup, hidden = self.static_expert_shape
            cluster_tile_n = n * cn
            ctas_per_cluster_tile = cn * (2 if self.use_2cta_instrs else 1)
            fc2_spin_threshold = (
                (intermediate_gateup + cluster_tile_n - 1) // cluster_tile_n
            ) * ctas_per_cluster_tile
            fc2_consumer_count = (
                (hidden + cluster_tile_n - 1) // cluster_tile_n
            ) * ctas_per_cluster_tile
            if fc2_spin_threshold >= self.FC1_DONE_FC2_PASS_UNIT:
                raise ValueError(
                    f"use_pdl: fc1 tile arrivals per token band ({fc2_spin_threshold}) "
                    f"must be < {self.FC1_DONE_FC2_PASS_UNIT}: they occupy the low 16 "
                    "bits of the fc1_done Int32 slot and must not carry into the fc2 "
                    "pass count in the high 16 bits."
                )
            if fc2_consumer_count > 0x7FFF:
                raise ValueError(
                    f"use_pdl: fc2 consumer passes per token band ({fc2_consumer_count}) "
                    "must be <= 32767: they occupy the high 16 bits of the signed "
                    "Int32 fc1_done slot and must not flip its sign."
                )

    def _create_tiled_mmas(self) -> Tuple[cute.TiledMma, cute.TiledMma]:
        common = (
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
        )
        # Rubin: the SM107 blockscaled FP8 MMA op instruction
        # B-reuse: permute the M mode so each CTA of the pair owns a contiguous 256-row slice
        # whose two accumulators are its 128-row halves.
        permutation_mnk = None
        if self.enable_breuse:
            permutation_mnk = (
                cute.make_layout((128, 2, 2), stride=(1, 256, 128)),
                self.mma_tiler[1],
                64,
            )
        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            *common,
            self.cta_group,
            (*self.mma_inst_shape_mn, 64),
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            permutation_mnk=permutation_mnk,
        )
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            *common,
            tcgen05.CtaGroup.ONE,
            (*self.mma_inst_shape_mn_sfb, 64),
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        return tiled_mma, tiled_mma_sfb

    def _build_scheduler(
        self, *, expert_cnt, intermediate_gateup, hidden_dim, launch_cluster_count
    ) -> None:
        """Construct FC12 scheduler and its SMEM/device workspaces."""
        if self.schedule_mode == "phase_interleave" and not self.enable_token_comm:
            raise NotImplementedError(
                "phase_interleave is currently supported only by the GLU "
                "MegaKernel, which owns the scheduler atomic workspace."
            )
        work_id_mode = {
            "static": "grid_stride",
            "atomic_counter": "atomic_counter",
            "cluster_launch_control": "cluster_launch_control",
        }[self.load_balance_mode]
        # Consumer group = every warp that calls ``consume_work`` (tma_a, tma_b, mma, epilogue).
        num_scheduler_consumer_threads = 32 * (len(self.epilogue_warp_id) + 3)
        if self.static_expert_shape is not None:
            expert_cnt, intermediate_gateup, hidden_dim = self.static_expert_shape
        problem_desc = ProblemDesc(
            {
                "expert_count": expert_cnt,
                "intermediate_gateup_size": intermediate_gateup,
                "hidden_size": hidden_dim,
            }
        )
        resolved_hint = self.group_hint
        if self.schedule_mode == "phase_interleave":
            mapping_cluster_tile_n = self.mma_tiler[1] * self.cluster_shape_mn[1]
            blocks_fc1 = (
                intermediate_gateup + mapping_cluster_tile_n - 1
            ) // mapping_cluster_tile_n
            blocks_fc2 = (hidden_dim + mapping_cluster_tile_n - 1) // mapping_cluster_tile_n
            raw_minimum_hint = minimum_phase_interleave_hint(
                blocks_fc1=blocks_fc1,
                blocks_fc2=blocks_fc2,
                launch_cluster_cnt_merge_as_preferred=launch_cluster_count,
            )
            minimum_safe_hint = (
                (raw_minimum_hint + self.epi_flag_batch[0] - 1)
                // self.epi_flag_batch[0]
                * self.epi_flag_batch[0]
            )
            if resolved_hint < minimum_safe_hint:
                raise ValueError(
                    f"phase_interleave hint {resolved_hint} must be at least "
                    f"{minimum_safe_hint} for this training geometry."
                )

        impl_desc = ImplDesc(
            {
                "num_scheduler_consumer_threads": num_scheduler_consumer_threads,
                "mma_tiler_mnk": self.mma_tiler,
                "cluster_shape_mn": self.cluster_shape_mn,
                "use_2cta_instrs": self.use_2cta_instrs,
                "hint": resolved_hint,
                "token_padding_block": self.token_padding_block,
                "sf_padding_block": self.sf_padding_block,
                "work_id_mode": work_id_mode,
                "is_swap_ab": False,
                "launch_cluster_count": launch_cluster_count,
                **(
                    {
                        "fallback_cluster_shape_mn": self.fallback_cluster_shape_mn,
                        "preferred_cluster_count": self.preferred_cluster_count,
                        "fallback_cluster_count": self.fallback_cluster_count,
                    }
                    if self.fallback_cluster_shape_mn is not None
                    else {}
                ),
            }
        )
        scheduler_type = (
            BlackwellFusedFc12Scheduler
            if self.schedule_mode == "grouped"
            else PhaseInterleavedFc12Scheduler
        )
        self.scheduler = scheduler_type(problem_desc, impl_desc)

        sched_smem_ws = SmemWorkspace()
        self.scheduler.register_smem_regions(sched_smem_ws)
        sched_smem_ws.finalize(max_bytes=self.smem_capacity)
        self.sched_smem_ws = sched_smem_ws

        sched_device_ws = DeviceWorkspace()
        self.scheduler.register_device_workspace(sched_device_ws)
        sched_device_ws.finalize()
        self.sched_device_ws = sched_device_ws

    def _setup_attributes(self) -> None:
        """Set up MMA / cluster / tile shapes, SMEM layouts, stage counts.

        The fc12 path shares ``mma_tiler_mnk`` and SMEM layouts across phases.
        """
        if self.enable_token_comm:
            if self.dispatch_warp_id is None:
                dispatch_warp_base = self.sched_warp_id + 1
                self.dispatch_warp_id = tuple(range(dispatch_warp_base, dispatch_warp_base + 4))
            num_token_back_warps = len(self.token_back_warp_id) if self.token_back_standalone else 0
            self.threads_per_cta = 32 * (
                len(self.epilogue_warp_id)
                + 1  # mma
                + 1  # tma_a
                + 1  # tma_b
                + 1  # sched
                + len(self.dispatch_warp_id)
                + num_token_back_warps
            )

        _any_token_comm = self.enable_token_comm or self._legacy_token_comm_args
        if _any_token_comm and self.enable_breuse:
            raise NotImplementedError(
                "B-reuse tiling (mma_tiler M 512) and the two-warpgroup / drained "
                "epilogues it implies are supported only on the standalone fc12 "
                "path; not with token_comm (the MegaMoE wrapper or the legacy "
                "token_comm_args argument)."
            )
        self.mma_inst_shape_mn = (self.mma_inst_shape_m, self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        assert self.mma_tiler[2] % mma_inst_shape_k == 0, (
            f"mma_tiler K ({self.mma_tiler[2]}) must be a multiple of "
            f"MMA instruction K ({mma_inst_shape_k})"
        )

        # SFB-specific tiler: rounded-up MN; same K as main tiler.
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )
        # Fallback-cluster layouts (== preferred when not mixed).
        self.fallback_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.resolved_fallback_cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.fallback_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.resolved_fallback_cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Multicast CTA counts
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # fc1_done slot granularity, in CTA tiles folded per slot. The standalone path owns its
        # counter and indexes it per MMA band (see fc1_done_counter_slots); under MegaMoE the
        # counter is the token_comm ready buffer, which has one slot per cluster token tile
        # (tokens_per_fc1_ready_slot = cta_tile_m * cluster_m), so M folds by cluster_m there.
        self.fc1_done_slot_m = (
            self.cluster_shape_mn[0]
            if self.enable_token_comm
            else (2 if self.use_2cta_instrs else 1)
        )

        # Epilogue owns all epi-side decisions (acc stages, subtile dispatch,
        # TMA commit/drain, and piggyback red.add).
        _epi_common = dict(
            fc1_done_slot_m=self.fc1_done_slot_m,
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            sf_vec_size=self.sf_vec_size,
            fc1_output_dtype=self.fc1_output_dtype,
            fc1_output_layout=self.fc1_output_layout,
            acc_dtype=self.acc_dtype,
            epilog_sync_bar_id=self.epilog_sync_bar_id,
            epilogue_warp_ids=self.epilogue_warp_id,
            static_expert_shape=self.static_expert_shape,
            fc2_in_kernel_topk_reduce=self.fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=self.token_back_by_dispatch,
            epi_flag_batch=self.epi_flag_batch,
            glu_clamp=self.gate_up_clamp,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            generate_c=self.generate_c,
            use_stg_fc1=self.use_stg_fc1,
            combine_format=getattr(self, "combine_format", None),
            act_func=self.act_func,
            fc2_use_bulk=self.fc2_use_bulk,
            fc2_tma_stages=self._fc2_tma_stages_arg,
            # One-shot: exactly one work tile per CTA, so a single accumulator stage suffices.
            one_task_per_cta=self.one_shot_sched,
        )
        # The tiler picks the epilogue class (B-reuse tile -> two-warpgroup epilogue).
        _epi_cls = GluMxfp8BreuseEpilogue if self.enable_breuse else GluMxfp8Epilogue
        self.epilogue = _epi_cls(**_epi_common)

        if self.num_sched_stages is None:
            self.num_sched_stages = 2

        # sD stages are indexed by epilogue warp (within its warpgroup), not by subtile, so
        # EpiWarpCount stages suffice.
        self.num_d_stage = min(len(self.epilogue_warp_id), EpiWarpCount)
        # fc1 output (fp8 quantised) SMEM — always present.
        d_bytes_total = self.epilogue.bytes_per_stage * self.num_d_stage  # noqa: F841
        # Raw gate+up SMEM (BF16, ping-pong) — only when generate_c=True.
        c_bytes_total = 0  # noqa: F841
        if self.generate_c:
            self.num_c_raw_stage = Fc1CTMAStages
        else:
            self.num_c_raw_stage = 0

        (
            self.num_a_stage,
            self.num_b_stage,
            self.num_sched_stages,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.epilogue.scratch_bytes,
            self.smem_capacity,
            self.occupancy,
            self.num_sched_stages,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_a_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_b_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_a_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_b_stage,
        )
        self.d_smem_layout_staged = self.epilogue.staged_smem_layout(
            self.num_d_stage,
        )
        # Raw gate+up SMEM layout (only meaningful when generate_c=True; pass as
        # dummy to kernel when False).
        if self.generate_c:
            self.c_smem_layout_staged = self.epilogue.staged_c_smem_layout(self.num_c_raw_stage)
        else:
            self.c_smem_layout_staged = None

        # Read epilogue's accumulator and scale-factor sizing decisions.
        self.num_acc_pipeline_stages = self.epilogue.num_acc_pipeline_stages
        self.num_acc_stage = self.epilogue.num_acc_stage
        self.num_sfa_tmem_cols = self.epilogue.num_sfa_tmem_cols
        self.num_sfb_tmem_cols = self.epilogue.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.epilogue.num_accumulator_tmem_cols
        # SFA / SFB live in TMEM columns past the accumulator inside the single
        # tmem.allocate(num_tmem_alloc_cols); nothing else checks the sum.
        _tmem_cols = (
            self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        )
        if _tmem_cols > self.num_tmem_alloc_cols:
            raise ValueError(
                f"TMEM footprint {self.num_accumulator_tmem_cols} acc + "
                f"{self.num_sfa_tmem_cols} SFA + {self.num_sfb_tmem_cols} SFB = "
                f"{_tmem_cols} columns exceeds the {self.num_tmem_alloc_cols}-column allocation"
            )

        # TMA load bytes per stage (A + B + SFA + SFB).
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.atom_thr_size = atom_thr_size  # store as Python int for use in @cute.kernel
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_a_bytes = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.num_tma_load_b_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

    def _smem_misc_budget_bytes(self) -> int:
        """Per-CTA SMEM reserved outside the ABC-stage pipeline."""
        return self._SmemMiscBudget

    def _compute_stages(
        self,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_bytes_total: int,
        smem_capacity: int,
        occupancy: int,
        num_sched_stages: int,
    ) -> Tuple[int, int, int]:
        """Compute the A ring, B ring and scheduler stage counts."""
        a_smem_layout_staged_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_staged_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        b_bytes_per_stage = cute.size_in_bytes(
            b_dtype, b_smem_layout_staged_one
        ) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)

        fixed_overhead = self._smem_misc_budget_bytes() + c_bytes_total

        num_ab_stage = (smem_capacity // occupancy - fixed_overhead) // ab_bytes_per_stage
        num_a_stage = num_ab_stage
        num_b_stage = num_ab_stage

        smem_per_cta = smem_capacity // occupancy
        unused_smem = smem_per_cta - fixed_overhead - num_ab_stage * ab_bytes_per_stage
        if unused_smem > b_bytes_per_stage:
            num_b_stage = num_b_stage + 1

        return num_a_stage, num_b_stage, num_sched_stages

    def fc1_done_counter_slots(self, data_total_rows: int, experts: int) -> int:
        """Upper bound on the Int32 slots of the fc1_done counter (one per MMA band)."""
        atom_thr_size = 2 if self.use_2cta_instrs else 1
        bands_per_cluster_tile = self.cluster_shape_mn[0] // atom_thr_size
        cluster_tile_rows = self.mma_tiler_mnk[0] * bands_per_cluster_tile
        cluster_tiles_upper = (
            data_total_rows + cluster_tile_rows - 1
        ) // cluster_tile_rows + experts
        return cluster_tiles_upper * bands_per_cluster_tile

    def load_balance_counter_bytes(self) -> int:
        """Bytes of the ``atomic_counter`` scheduler's device workspace (0 in other modes)."""
        if self.load_balance_mode != "atomic_counter":
            return 0
        config = NonClcMixedCgaConfig(
            preferred_cluster_shape=self.cluster_shape_mn,
            fallback_cluster_shape=self.resolved_fallback_cluster_shape_mn
            if self.is_mixed_cga
            else None,
            # The sizes do not depend on the launch cluster count; the config only
            # requires it to be positive without a fallback shape.
            launch_cluster_count=None if self.is_mixed_cga else 1,
            preferred_cluster_count=self.preferred_cluster_count if self.is_mixed_cga else None,
            fallback_cluster_count=self.fallback_cluster_count if self.is_mixed_cga else None,
        )
        worker = NonClcMixedCgaSchedulerWorker(
            config=config, work_id_mode="atomic_counter", stream_count=1
        )
        workspace = DeviceWorkspace()
        worker.register_device_workspace(workspace)
        workspace.finalize()
        return workspace.total_bytes("local")

    def load_balance_counter_is_undersized(self, shape, element_bytes: int) -> bool:
        """True if a static ``load_balance_counter`` buffer is undersized (False if dynamic)."""
        dims = []

        def _flatten(s):
            if isinstance(s, (tuple, list)):
                for x in s:
                    _flatten(x)
            else:
                dims.append(s)

        _flatten(shape)
        if not all(isinstance(d, int) and not isinstance(d, bool) for d in dims):
            return False
        words = 1
        for d in dims:
            words *= d
        return words * element_bytes < self.load_balance_counter_bytes()

    def one_shot_tile_count(self) -> int:
        """CTAs a one-shot launch needs: one per FC1 and one per FC2 N-tile of the single expert."""
        _, gateup, hidden = self.static_expert_shape
        n = self.mma_tiler[1]  # == cta_tile_shape_mnk[1]; usable before _setup_attributes
        return (gateup + n - 1) // n + (hidden + n - 1) // n

    def get_workspace_size_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Compute opaque workspace size, including discrete TensorMaps."""
        _, total_bytes = self._get_workspace_layout_in_bytes(
            fc1_activation_tensor, fc1_weight_tensor
        )
        return total_bytes

    def get_weight_descriptor_workspace_size_in_bytes(self) -> int:
        """Return zero for contiguous mode or the discrete TensorMap region size."""
        if self.weight_storage_mode == "contiguous":
            return 0
        return discrete_weight_descriptor_workspace_size(
            self.static_expert_shape[0],
            self.is_mixed_cga,
        )

    def get_weight_descriptor_workspace_offset_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Return the descriptor region offset in the opaque lean workspace."""
        if self.weight_storage_mode != "discrete":
            raise ValueError("Contiguous weights do not allocate a descriptor workspace.")
        descriptor_offset, _ = self._get_workspace_layout_in_bytes(
            fc1_activation_tensor, fc1_weight_tensor
        )
        return descriptor_offset

    @cute.jit
    def make_weight_descriptor_workspace_pointer(
        self,
        workspace_ptr: cute.Pointer,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> cute.Pointer:
        """Derive the discrete TensorMap pointer from the opaque workspace."""
        byte_offset = self.get_weight_descriptor_workspace_offset_in_bytes(
            fc1_activation_tensor, fc1_weight_tensor
        )
        return cute.make_ptr(
            cutlass.Uint8,
            workspace_ptr.toint() + cutlass.Int64(byte_offset),
            cute.AddressSpace.gmem,
            assumed_align=discrete_weight_descriptor_workspace_alignment(),
        )

    def _get_workspace_layout_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> Tuple[Optional[int], int]:
        sf_padding_block = self.sf_padding_block

        data_total_rows, _hidden = fc1_activation_tensor.shape
        if self.weight_storage_mode == "contiguous":
            experts, _hidden_w, intermediate_gateup = fc1_weight_tensor.shape
        else:
            experts, intermediate_gateup, _hidden_w = self.static_expert_shape
        intermediate_downproj = intermediate_gateup // 2

        # Conservative upper bound for sf_total_rows.
        sf_total_rows_upper = data_total_rows + experts * sf_padding_block

        fc1_output_bytes = data_total_rows * intermediate_downproj * self.ab_dtype.width // 8

        # fc1_output_sf sf_vec_size matches the kernel's sf_vec_size.
        fc1_out_sf_vec_size = self.sf_vec_size
        sf_block_cols = ((intermediate_downproj // fc1_out_sf_vec_size) + 3) // 4 * 4
        fc1_output_sf_bytes = sf_total_rows_upper * sf_block_cols

        # fc1_done_counter: one Int32 per MMA band (see fc1_done_counter_slots).
        fc1_done_counter_bytes = self.fc1_done_counter_slots(data_total_rows, experts) * 4

        scratch_bytes = fc1_output_bytes + fc1_output_sf_bytes + fc1_done_counter_bytes
        # load_balance_counter: the atomic_counter scheduler's device workspace, 16-byte aligned.
        load_balance_counter_bytes = self.load_balance_counter_bytes()
        if load_balance_counter_bytes:
            scratch_bytes = (scratch_bytes + 15) // 16 * 16 + load_balance_counter_bytes

        # Keep the legacy scratch layout unchanged, then append aligned TensorMaps.
        alignment = discrete_weight_descriptor_workspace_alignment()
        scratch_bytes = ((scratch_bytes + alignment - 1) // alignment) * alignment
        if self.weight_storage_mode == "contiguous":
            return None, scratch_bytes

        descriptor_offset = scratch_bytes
        total_bytes = descriptor_offset + discrete_weight_descriptor_workspace_size(
            experts, self.is_mixed_cga
        )
        return descriptor_offset, total_bytes

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """SMEM → TMEM tiled copy + partition for SFA / SFB."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def token_comm_extra_smem_storage_class(self) -> type:
        """Return a ``@cute.struct`` for dispatch-warp SMEM, or None."""
        return None

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Return dispatch->fc1 release counter pointer, or None (lean: disabled)."""
        return None

    def sched_ext_fc1_peek_threshold(self) -> int:
        """Return the fc1 ready-counter peek threshold for GluMxFp8Fc12SchedExtension.

        Must match the spin threshold in ``token_comm_hook_fc1_tma_b_predispatch_spin``
        so that an early peek hit does not skip the spin and expose stale pool rows.
        Default 0 → use ``valid_tokens_in_tile`` (no cluster, base class behaviour).
        MegaMoE overrides to return ``cluster_tile_tokens`` to match the cluster spin.
        """
        return 0

    def sched_ext_fc1_counter_cumul_scale(self) -> int:
        """Return the scale factor for the fc1 ready-counter slot formula.

        Slot = scale * (cumul + fc1_counter_index) + tile_m_idx % scale.
        Default 1 = cluster-level granularity (slot = cumul + cluster_token_block_idx).
        A MegaMoE subclass can override to cluster_m for per-CTA granularity.
        """
        return 1

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """Sched warp: wait for dispatch barrier before reading sizes.  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        """TMA warp: spin until dispatch-pulled tokens are resident.  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Body for dispatch warps 8-11 (MegaMoE-only).  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Body for standalone token-back warps 12-15 (MegaMoE-only).  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_kernel_tail(self, token_comm_args, *, warp_idx, lane_idx, tidx):
        """All-warp kernel tail (NVLink release, etc.).  No-op base."""
        pass

    @cute.kernel
    def _initialize_discrete_weight_descriptors(
        self,
        fc1_weight_ptrs: cute.Pointer,
        fc1_weight_sf_ptrs: cute.Pointer,
        fc2_weight_ptrs: cute.Pointer,
        fc2_weight_sf_ptrs: cute.Pointer,
        fc1_n: cutlass.Int32,
        fc1_k: cutlass.Int32,
        fc1_stride_n: cutlass.Int64,
        fc2_n: cutlass.Int32,
        fc2_k: cutlass.Int32,
        fc2_stride_n: cutlass.Int64,
        descriptor_workspace: cute.Pointer,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        b_smem_layout,
        sfb_smem_layout,
        cluster_layout_vmnk_shape: cutlass.Constexpr,
        cluster_layout_sfb_vmnk_shape: cutlass.Constexpr,
        fallback_cluster_layout_vmnk_shape: cutlass.Constexpr,
        fallback_cluster_layout_sfb_vmnk_shape: cutlass.Constexpr,
    ) -> None:
        """Build preferred (and mixCGA fallback) weight TensorMaps per expert."""
        expert_idx = cute.arch.block_idx()[0]
        expert_count = self.static_expert_shape[0]
        singleton = cutlass.Int32(1)
        zero = cutlass.Int64(0)

        fc1_weight_pointer_values = _int64_pointer_array_tensor(fc1_weight_ptrs, expert_count)
        fc1_weight_sf_pointer_values = _int64_pointer_array_tensor(fc1_weight_sf_ptrs, expert_count)
        fc2_weight_pointer_values = _int64_pointer_array_tensor(fc2_weight_ptrs, expert_count)
        fc2_weight_sf_pointer_values = _int64_pointer_array_tensor(fc2_weight_sf_ptrs, expert_count)

        b_operation = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_operation = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )

        fc1_weight_tensor = cute.make_tensor(
            cute.make_ptr(
                self.ab_dtype,
                fc1_weight_pointer_values[expert_idx],
                cute.AddressSpace.gmem,
            ),
            cute.make_layout(
                (fc1_n, fc1_k, singleton),
                stride=(fc1_stride_n, cutlass.Int64(1), zero),
            ),
        )
        fc1_weight_sf_tensor = cute.make_tensor(
            cute.make_ptr(
                self.sf_dtype,
                fc1_weight_sf_pointer_values[expert_idx],
                cute.AddressSpace.gmem,
            ),
            blockscaled_utils.tile_atom_to_shape_SF((fc1_n, fc1_k, singleton), self.sf_vec_size),
        )
        fc2_weight_tensor = cute.make_tensor(
            cute.make_ptr(
                self.ab_dtype,
                fc2_weight_pointer_values[expert_idx],
                cute.AddressSpace.gmem,
            ),
            cute.make_layout(
                (fc2_n, fc2_k, singleton),
                stride=(fc2_stride_n, cutlass.Int64(1), zero),
            ),
        )
        fc2_weight_sf_tensor = cute.make_tensor(
            cute.make_ptr(
                self.sf_dtype,
                fc2_weight_sf_pointer_values[expert_idx],
                cute.AddressSpace.gmem,
            ),
            blockscaled_utils.tile_atom_to_shape_SF((fc2_n, fc2_k, singleton), self.sf_vec_size),
        )

        for cluster_variant in cutlass.range_constexpr(1 + int(self.is_mixed_cga)):
            vmnk_shape = cluster_layout_vmnk_shape
            sfb_vmnk_shape = cluster_layout_sfb_vmnk_shape
            if cutlass.const_expr(cluster_variant == 1):
                vmnk_shape = fallback_cluster_layout_vmnk_shape
                sfb_vmnk_shape = fallback_cluster_layout_sfb_vmnk_shape
            cluster_variant_i32 = cutlass.Int32(cluster_variant)

            fc1_weight_atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_operation,
                fc1_weight_tensor,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                vmnk_shape,
            )
            cpasync.copy_tensormap(
                fc1_weight_atom,
                raw_discrete_weight_descriptor_pointer(
                    descriptor_workspace,
                    "fc1_weight",
                    expert_idx,
                    cluster_variant_i32,
                    self.is_mixed_cga,
                ),
            )

            fc1_weight_sf_atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_operation,
                fc1_weight_sf_tensor,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                sfb_vmnk_shape,
                internal_type=cutlass.Uint64,
            )
            cpasync.copy_tensormap(
                fc1_weight_sf_atom,
                raw_discrete_weight_descriptor_pointer(
                    descriptor_workspace,
                    "fc1_weight_sf",
                    expert_idx,
                    cluster_variant_i32,
                    self.is_mixed_cga,
                ),
            )

            fc2_weight_atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_operation,
                fc2_weight_tensor,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                vmnk_shape,
            )
            cpasync.copy_tensormap(
                fc2_weight_atom,
                raw_discrete_weight_descriptor_pointer(
                    descriptor_workspace,
                    "fc2_weight",
                    expert_idx,
                    cluster_variant_i32,
                    self.is_mixed_cga,
                ),
            )

            fc2_weight_sf_atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_operation,
                fc2_weight_sf_tensor,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                sfb_vmnk_shape,
                internal_type=cutlass.Uint64,
            )
            cpasync.copy_tensormap(
                fc2_weight_sf_atom,
                raw_discrete_weight_descriptor_pointer(
                    descriptor_workspace,
                    "fc2_weight_sf",
                    expert_idx,
                    cluster_variant_i32,
                    self.is_mixed_cga,
                ),
            )

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,  # (token_sum_padded, hidden)
        fc1_weight,  # Contiguous tensor or device Int64 pointer array
        activation_sf: cute.Tensor,  # (token_sum_padded_sf, hidden / sf_vec_size)
        fc1_weight_sf,  # Contiguous tensor or device Int64 pointer array
        fc1_output: cute.Tensor,  # (token_sum_padded, intermediate_downproj)
        fc1_output_sf: cute.Tensor,  # (token_sum_padded_sf, intermediate_downproj / sf_vec_size)
        fc2_weight,  # Contiguous tensor or device Int64 pointer array
        fc2_weight_sf,  # Contiguous tensor or device Int64 pointer array
        fc2_output: cute.Tensor,  # (token_sum_padded, hidden) BFloat16, hidden stride-1
        topk_scores: cute.Tensor,  # (token_sum_padded,) Float32
        fc1_done_counter: cute.Tensor,  # (>= fc1_done_counter_slots(rows, experts),) Int32, one per MMA band
        offs: Optional[cute.Tensor] = None,  # (experts,) Int32 cumulative end offsets
        max_active_clusters: cutlass.Constexpr = None,
        stream: cuda.CUstream = None,
        # Host-known token row count; sizes the CLC ticket grid and must equal the per-CTA tile M
        # under one_shot_sched.
        host_token_rows: cutlass.Constexpr = None,
        norm_const_tensor: Optional[cute.Tensor] = None,
        global_activation_sf: Optional[cute.Tensor] = None,
        global_fc1_weight_sf: Optional[cute.Tensor] = None,
        load_balance_counter: Optional[cute.Tensor] = None,
        expert_token_sizes: Optional[cute.Tensor] = None,
        token_comm_args=None,
        fc1_c: Optional[cute.Tensor] = None,
        # ── Per-rank FC12 overflow output ────────────────────────────────
        overflow_flag: cute.Tensor = None,
        fc2_output_sf: Optional[cute.Tensor] = None,
        mega_peer_rank_ptr_mapper=None,
        mega_local_rank: Optional[cutlass.Int32] = None,
        mega_local_workspace: Optional[cute.Pointer] = None,
        mega_shared_workspace: Optional[cute.Pointer] = None,
        mega_activation: Optional[cute.Tensor] = None,
        mega_activation_sf: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation_sf: Optional[cute.Tensor] = None,
        # Discrete-only region derived from the opaque lean workspace via
        # make_weight_descriptor_workspace_pointer(); MegaMoE supplies its
        # registered DeviceWorkspace region directly.
        weight_descriptor_workspace: Optional[cute.Pointer] = None,
    ) -> None:
        """Launch the fused fc1+fc2 GLU MXFP8 kernel."""

        if cutlass.const_expr(
            self.weight_storage_mode == "discrete" and weight_descriptor_workspace is None
        ):
            raise ValueError("Discrete weights require weight_descriptor_workspace.")

        if cutlass.const_expr(self.static_expert_shape is not None):
            (
                experts_static,
                intermediate_gateup_static,
                hidden_static,
            ) = self.static_expert_shape
            intermediate_downproj_static = intermediate_gateup_static // 2

            if cutlass.const_expr(self.weight_storage_mode == "contiguous"):
                fc1_weight = cute.make_tensor(
                    fc1_weight.iterator,
                    cute.make_layout(
                        (experts_static, hidden_static, intermediate_gateup_static),
                        stride=fc1_weight.stride,
                    ),
                )
                fc2_weight = cute.make_tensor(
                    fc2_weight.iterator,
                    cute.make_layout(
                        (experts_static, intermediate_downproj_static, hidden_static),
                        stride=fc2_weight.stride,
                    ),
                )
            activation = cute.make_tensor(
                activation.iterator,
                cute.make_layout(
                    (activation.shape[0], hidden_static),
                    stride=activation.stride,
                ),
            )
            fc1_output = cute.make_tensor(
                fc1_output.iterator,
                cute.make_layout(
                    (fc1_output.shape[0], intermediate_downproj_static),
                    stride=fc1_output.stride,
                ),
            )
            # fc2_output is 2D (tokens, hidden) on the lean path and 3D
            # (max_tokens, topk, hidden) on the MegaMoE path.
            if cutlass.const_expr(len(fc2_output.shape) == 3):
                fc2_output = cute.make_tensor(
                    fc2_output.iterator,
                    cute.make_layout(
                        (fc2_output.shape[0], fc2_output.shape[1], hidden_static),
                        stride=fc2_output.stride,
                    ),
                )
            else:
                fc2_output = cute.make_tensor(
                    fc2_output.iterator,
                    cute.make_layout(
                        (fc2_output.shape[0], hidden_static),
                        stride=fc2_output.stride,
                    ),
                )

        # GEMM-domain transform for fc1 phase
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # A_gemm (fc1 activations): (tokens_sum, hidden) -> (M=tokens, K=hidden, L=1).
        tokens_sum, hidden = activation.shape
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens_sum, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )

        # B_gemm (fc1 weights): (experts, hidden, intermediate_gateup) with hidden stride-1 (K-major)
        # -> (N=intermediate_gateup, K=hidden, L=experts).
        if cutlass.const_expr(self.weight_storage_mode == "contiguous"):
            experts, hidden_b, intermediate_gateup = fc1_weight.shape
            fc1_weight_gemm = cute.make_tensor(
                fc1_weight.iterator,
                cute.make_layout(
                    (intermediate_gateup, hidden_b, experts),
                    stride=(
                        fc1_weight.stride[2],
                        fc1_weight.stride[1],
                        fc1_weight.stride[0],
                    ),
                ),
            )
        else:
            experts, intermediate_gateup, hidden_b = self.static_expert_shape
            fc1_weight_n = cutlass.Int32(intermediate_gateup)
            fc1_weight_k = cutlass.Int32(hidden_b)
            fc1_weight_gemm = cute.make_tensor(
                cute.make_ptr(
                    self.ab_dtype,
                    fc1_weight.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(
                    (fc1_weight_n, fc1_weight_k, cutlass.Int32(1)),
                    stride=(
                        cutlass.Int64(hidden_b),
                        cutlass.Int64(1),
                        cutlass.Int64(0),
                    ),
                ),
            )

        # D_gemm is a user-view output tensor; epilogue owns its store path.
        intermediate_downproj = fc1_output.shape[1]
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (tokens_sum, intermediate_downproj, 1),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )

        # SFA / SFB scale tensors (atom-tiled) — fc1 phase.
        #   SFA (mma M-side) = activation_sf (activation scales, A-side)
        #   SFB (mma N-side) = fc1_weight_sf (weight scales, B-side)
        tokens_sum_padded = activation_sf.shape[0]
        hidden_padded = activation_sf.shape[1] * self.sf_vec_size
        activation_sf_gemm = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, hidden_padded, 1), self.sf_vec_size
            ),
        )
        if cutlass.const_expr(self.weight_storage_mode == "contiguous"):
            intermediate_gateup_padded_mul_hidden_padded = fc1_weight_sf.shape[1]
            intermediate_gateup_padded = (
                intermediate_gateup_padded_mul_hidden_padded * self.sf_vec_size
            ) // hidden_padded
            fc1_weight_sf_iterator = fc1_weight_sf.iterator
            fc1_weight_sf_shape = (
                intermediate_gateup_padded,
                hidden_padded,
                experts,
            )
        else:
            fc1_weight_sf_iterator = cute.make_ptr(
                activation_sf.element_type,
                fc1_weight_sf.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            fc1_weight_sf_shape = (
                fc1_weight_n,
                fc1_weight_k,
                cutlass.Int32(1),
            )
        fc1_weight_sf_gemm = cute.make_tensor(
            fc1_weight_sf_iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                fc1_weight_sf_shape,
                self.sf_vec_size,
            ),
        )

        # ── GEMM-domain transform for fc2 phase ──
        #
        # fc2 roles: M=hidden, N=tokens_sum, K=intermediate_downproj.

        # A_gemm (fc2 weights): (experts, intermediate_downproj, hidden)
        # -> (M=hidden, K=intermediate_downproj, L=experts).
        if cutlass.const_expr(self.weight_storage_mode == "contiguous"):
            experts2, intermediate_downproj_b2, hidden_b2 = fc2_weight.shape
            fc2_weight_gemm = cute.make_tensor(
                fc2_weight.iterator,
                cute.make_layout(
                    (hidden_b2, intermediate_downproj_b2, experts2),
                    stride=(
                        fc2_weight.stride[2],
                        fc2_weight.stride[1],
                        fc2_weight.stride[0],
                    ),
                ),
            )
        else:
            experts2 = experts
            intermediate_downproj_b2 = intermediate_gateup // 2
            hidden_b2 = hidden_b
            fc2_weight_n = cutlass.Int32(hidden_b2)
            fc2_weight_k = cutlass.Int32(intermediate_downproj_b2)
            fc2_weight_gemm = cute.make_tensor(
                cute.make_ptr(
                    self.ab_dtype,
                    fc2_weight.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(
                    (fc2_weight_n, fc2_weight_k, cutlass.Int32(1)),
                    stride=(
                        cutlass.Int64(intermediate_downproj_b2),
                        cutlass.Int64(1),
                        cutlass.Int64(0),
                    ),
                ),
            )

        if cutlass.const_expr(len(fc2_output.shape) == 3):
            fc2_hidden_out = fc2_output.shape[2]
            fc2_output_gemm = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], fc2_hidden_out, c1),
                    stride=(fc2_output.stride[0], fc2_output.stride[2], c0),
                ),
            )
        else:
            fc2_hidden_out = fc2_output.shape[1]
            fc2_output_gemm = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (tokens_sum, fc2_hidden_out, c1),
                    stride=(fc2_output.stride[0], fc2_output.stride[1], c0),
                ),
            )

        # SFA / SFB for fc2:
        #   SFA (mma M-side) = fc2_weight_sf (fc2 weight scales, sf_vec_size)
        #   SFB (mma N-side) = fc1_output_sf (fc1 epilogue SFs, uses sf_vec_size)
        # Both paths produce SFs with self.sf_vec_size:
        fc1_out_sf_vec_size = self.sf_vec_size
        tokens_sum_padded_sf = fc1_output_sf.shape[0]
        intermediate_downproj_padded = fc1_output_sf.shape[1] * fc1_out_sf_vec_size
        fc1_output_sf_gemm_for_fc2_load = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded_sf, intermediate_downproj_padded, 1),
                fc1_out_sf_vec_size,
            ),
        )

        if cutlass.const_expr(self.weight_storage_mode == "contiguous"):
            hidden_padded_fc2_mul_intermediate_downproj_padded = fc2_weight_sf.shape[1]
            hidden_padded_fc2 = (
                hidden_padded_fc2_mul_intermediate_downproj_padded * self.sf_vec_size
            ) // intermediate_downproj_padded
            fc2_weight_sf_iterator = fc2_weight_sf.iterator
            fc2_weight_sf_shape = (
                hidden_padded_fc2,
                intermediate_downproj_padded,
                experts2,
            )
        else:
            fc2_weight_sf_iterator = cute.make_ptr(
                activation_sf.element_type,
                fc2_weight_sf.toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            fc2_weight_sf_shape = (
                fc2_weight_n,
                fc2_weight_k,
                cutlass.Int32(1),
            )
        fc2_weight_sf_gemm = cute.make_tensor(
            fc2_weight_sf_iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                fc2_weight_sf_shape,
                self.sf_vec_size,
            ),
        )

        expert_cnt = experts
        hidden_dim = hidden

        # ── Infer dtypes and major modes ──
        self.a_dtype: Type[cutlass.Numeric] = activation_gemm.element_type
        self.b_dtype: Type[cutlass.Numeric] = fc1_weight_gemm.element_type
        self.fc1_output_dtype: Type[cutlass.Numeric] = fc1_output_gemm.element_type
        self.sf_dtype: Type[cutlass.Numeric] = activation_sf_gemm.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(activation_gemm).mma_major_mode()
        if cutlass.const_expr(self.weight_storage_mode == "contiguous"):
            self.b_major_mode = utils.LayoutEnum.from_tensor(fc1_weight_gemm).mma_major_mode()
        else:
            self.b_major_mode = OperandMajorMode.K
        self.fc1_output_layout = utils.LayoutEnum.from_tensor(fc1_output_gemm)

        self._legacy_token_comm_args = token_comm_args is not None
        self._setup_attributes()
        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        def make_tma_b_maybe_pair(
            operation,
            gmem_tensor,
            smem_layout,
            mma_tiler,
            tiled_mma_arg,
            preferred_layout,
            fallback_layout,
            *,
            internal_type=None,
        ):
            preferred_atom, preferred_tensor = cute.nvgpu.make_tiled_tma_atom_B(
                operation,
                gmem_tensor,
                smem_layout,
                mma_tiler,
                tiled_mma_arg,
                preferred_layout.shape,
                internal_type=internal_type,
            )
            atom_or_pair = preferred_atom
            if cutlass.const_expr(self.is_mixed_cga):
                fallback_atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
                    operation,
                    gmem_tensor,
                    smem_layout,
                    mma_tiler,
                    tiled_mma_arg,
                    fallback_layout.shape,
                    internal_type=internal_type,
                )
                atom_or_pair = (preferred_atom, fallback_atom)
            return atom_or_pair, preferred_tensor

        # ── fc1 TMA atoms ──

        # TMA load A1
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_activation, tma_tensor_fc1_activation = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            activation_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load B1
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_weight, tma_tensor_fc1_weight = make_tma_b_maybe_pair(
            b_op,
            fc1_weight_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
        )

        # TMA load SFA1
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_activation_sf, tma_tensor_fc1_activation_sf = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            activation_sf_gemm,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )

        # TMA load SFB1
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_weight_sf, tma_tensor_fc1_weight_sf = make_tma_b_maybe_pair(
            sfb_op,
            fc1_weight_sf_gemm,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_sfb_vmnk,
            internal_type=cutlass.Uint64,
        )

        # TMA store for fc1 MXFP8 output.
        fc1_output_tma_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_fc1_output, tma_tensor_fc1_output = cpasync.make_tiled_tma_atom(
            fc1_output_tma_op,
            fc1_output_gemm,
            self.epilogue.smem_layout_one_stage,
            self.epilogue.epi_tile,
        )

        # TMA store for raw fc1 accumulator
        if cutlass.const_expr(self.generate_c):
            c_gemm = cute.make_tensor(
                fc1_c.iterator,
                cute.make_layout(
                    (tokens_sum, intermediate_gateup, 1),
                    stride=(fc1_c.stride[0], fc1_c.stride[1], 0),
                ),
            )
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_gemm,
                self.epilogue.c_smem_layout_one_stage,
                self.epilogue.epi_tile_c,
            )
        else:
            tma_atom_c = tma_atom_fc1_output
            tma_tensor_c = tma_tensor_fc1_output

        # fc1 SFC GMEM tensor (= fc1_output_sf user view).
        fc1_output_sf_gemm = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, intermediate_downproj, 1),
                self.sf_vec_size,
            ),
        )

        # ── fc2 TMA atoms: fc1_output → A-side (M=tokens), fc2_weight → B-side (N=hidden) ──
        tma_atom_fc2_activation, tma_tensor_fc2_activation = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            fc1_output_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_fc2_weight, tma_tensor_fc2_weight = make_tma_b_maybe_pair(
            b_op,
            fc2_weight_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
        )
        tma_atom_fc2_activation_sf, tma_tensor_fc2_activation_sf = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            fc1_output_sf_gemm_for_fc2_load,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint64,
        )
        tma_atom_fc2_weight_sf, tma_tensor_fc2_weight_sf = make_tma_b_maybe_pair(
            sfb_op,
            fc2_weight_sf_gemm,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_sfb_vmnk,
            internal_type=cutlass.Uint64,
        )

        # Mixed CGA: fallback-shaped B/SFB atoms (the gmem coordinate tensors are shape-independent
        # and reused; A/SFA need no fallback variant since cluster N == 1 in both shapes).
        if cutlass.const_expr(self.is_mixed_cga):
            # Fallback atoms reuse the preferred multicast ops with only the cluster layout changed;
            # a plain op would stall the full barrier on demoted clusters.
            tma_atom_fc1_weight_fallback, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                fc1_weight_gemm,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.fallback_cluster_layout_vmnk.shape,
            )
            tma_atom_fc1_weight_sf_fallback, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                fc1_weight_sf_gemm,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.fallback_cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
            tma_atom_fc2_weight_fallback, _ = cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                fc2_weight_gemm,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.fallback_cluster_layout_vmnk.shape,
            )
            tma_atom_fc2_weight_sf_fallback, _ = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                fc2_weight_sf_gemm,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.fallback_cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        else:
            tma_atom_fc1_weight_fallback = tma_atom_fc1_weight
            tma_atom_fc1_weight_sf_fallback = tma_atom_fc1_weight_sf
            tma_atom_fc2_weight_fallback = tma_atom_fc2_weight
            tma_atom_fc2_weight_sf_fallback = tma_atom_fc2_weight_sf

        if cutlass.const_expr(self.weight_storage_mode == "discrete"):
            self._initialize_discrete_weight_descriptors(
                fc1_weight,
                fc1_weight_sf,
                fc2_weight,
                fc2_weight_sf,
                fc1_weight_n,
                fc1_weight_k,
                cutlass.Int64(hidden_b),
                fc2_weight_n,
                fc2_weight_k,
                cutlass.Int64(intermediate_downproj_b2),
                weight_descriptor_workspace,
                tiled_mma,
                tiled_mma_sfb,
                b_smem_layout,
                sfb_smem_layout,
                self.cluster_layout_vmnk.shape,
                self.cluster_layout_sfb_vmnk.shape,
                self.fallback_cluster_layout_vmnk.shape,
                self.fallback_cluster_layout_sfb_vmnk.shape,
            ).launch(
                grid=(expert_cnt, 1, 1),
                block=(1, 1, 1),
                stream=stream,
            )

        # ── Scheduler params + grid + launch ──
        if cutlass.const_expr(
            self.load_balance_mode == "atomic_counter" and not self.enable_token_comm
        ):
            if cutlass.const_expr(load_balance_counter is None):
                raise ValueError(
                    "load_balance_counter must be provided when "
                    "load_balance_mode == 'atomic_counter'"
                )
        # Host-side size check of the scheduler workspace buffer (static shapes only;
        # a dynamic leading dimension cannot be checked at compile time).
        _load_balance_counter_undersized = False
        if cutlass.const_expr(
            self.load_balance_mode == "atomic_counter" and load_balance_counter is not None
        ):
            _load_balance_counter_undersized = self.load_balance_counter_is_undersized(
                load_balance_counter.shape, load_balance_counter.element_type.width // 8
            )
        if cutlass.const_expr(_load_balance_counter_undersized):
            raise ValueError(
                "load_balance_counter is smaller than the atomic_counter scheduler workspace: "
                f"needs {self.load_balance_counter_bytes()} bytes (kernel.load_balance_counter_bytes())."
            )

        # On the MegaMoE path the per-expert sizes come from the Router (device-side), so the
        # caller supplies neither offs nor expert_token_sizes.
        if cutlass.const_expr(not self.enable_token_comm):
            if cutlass.const_expr((offs is None) == (expert_token_sizes is None)):
                raise ValueError(
                    "Exactly one of `offs` / `expert_token_sizes` must be "
                    "provided; got "
                    f"offs={'set' if offs is not None else 'None'}, "
                    f"expert_token_sizes="
                    f"{'set' if expert_token_sizes is not None else 'None'}."
                )

        self._build_scheduler(
            expert_cnt=expert_cnt,
            intermediate_gateup=intermediate_gateup,
            hidden_dim=hidden_dim,
            launch_cluster_count=max_active_clusters,
        )
        # The host-side load_balance_counter_bytes() must agree with what the scheduler
        # just registered.
        if cutlass.const_expr(
            self.load_balance_mode == "atomic_counter"
            and self.sched_device_ws.total_bytes("local") != self.load_balance_counter_bytes()
        ):
            raise RuntimeError(
                "load_balance_counter_bytes() disagrees with the scheduler's device workspace "
                f"({self.load_balance_counter_bytes()} vs {self.sched_device_ws.total_bytes('local')} bytes)."
            )
        # CLC launches the full ticket grid sized from the host-known token count; persistent
        # modes launch one cluster per resident slot.
        if cutlass.const_expr(self.load_balance_mode == "cluster_launch_control"):
            if cutlass.const_expr(host_token_rows is None):
                raise ValueError(
                    "load_balance_mode='cluster_launch_control' requires host_token_rows "
                    "(host-known token row count) to size the ticket grid."
                )
            if cutlass.const_expr(self.enable_token_comm):
                raise NotImplementedError(
                    "cluster_launch_control is not supported on the MegaMoE token_comm path: "
                    "its grid sync and linear CTA index assume the persistent launch grid."
                )
            grid = self.scheduler.get_grid_shape(
                max_active_clusters=max_active_clusters,
                total_token_rows=host_token_rows,
            )
        else:
            grid = self.scheduler.get_grid_shape(
                max_active_clusters=max_active_clusters,
            )
        # One-shot is defined for exactly one M band, so the token count must equal the
        # per-CTA tile M.
        if cutlass.const_expr(self.one_shot_sched and host_token_rows is None):
            raise ValueError(
                "one_shot_sched requires host_token_rows (host-known token row count)."
            )
        if cutlass.const_expr(
            self.one_shot_sched and host_token_rows != self.cta_tile_shape_mnk[0]
        ):
            raise ValueError(
                "one_shot_sched is defined for exactly one M band: host_token_rows must "
                f"equal the per-CTA tile M ({self.cta_tile_shape_mnk[0]}); got {host_token_rows}."
            )
        # One-shot derives every tile from blockIdx.z; a grid shorter than one_shot_tile_count()
        # would silently skip tiles.
        if cutlass.const_expr(self.one_shot_sched and grid[2] < self.one_shot_tile_count()):
            raise ValueError(
                f"one_shot_sched needs grid z >= fc1 N-tiles + fc2 N-tiles = {self.one_shot_tile_count()} "
                f"(one CTA per tile); got {grid[2]}. Pass max_active_clusters=one_shot_tile_count()."
            )

        # FC2 TMASTG (fc2_use_bulk) store atom (host-side)
        if cutlass.const_expr(self.enable_token_comm and self.epilogue.fc2_use_tma):
            self._mega_device_workspace.assign_device_members(
                mega_local_workspace, mega_shared_workspace
            )
            _fc2_pool3d = self.token_comm.fc2_activation_tensor(self._mega_device_workspace)
            _fc2_pool2d = cute.make_tensor(
                _fc2_pool3d.iterator,
                cute.make_layout(
                    (_fc2_pool3d.shape[0], _fc2_pool3d.shape[2]),
                    stride=(_fc2_pool3d.stride[0], _fc2_pool3d.stride[2]),
                ),
            )
            tma_atom_fc2_output, fc2_tma_output = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                _fc2_pool2d,
                self.epilogue.fc2_tma_smem_layout_one_stage,
                self.epilogue.fc2_tma_tile,
            )
        else:
            tma_atom_fc2_output = None
            fc2_tma_output = None

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            # fc1 TMA atoms / tensors (A=activations, B=weights)
            tma_atom_fc1_activation,
            tma_tensor_fc1_activation,
            tma_atom_fc1_weight,
            tma_tensor_fc1_weight,
            tma_atom_fc1_activation_sf,
            tma_tensor_fc1_activation_sf,
            tma_atom_fc1_weight_sf,
            tma_tensor_fc1_weight_sf,
            tma_atom_fc1_output,
            tma_tensor_fc1_output,
            # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
            tma_atom_fc2_activation,
            tma_tensor_fc2_activation,
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            tma_atom_fc2_activation_sf,
            tma_tensor_fc2_activation_sf,
            tma_atom_fc2_weight_sf,
            tma_tensor_fc2_weight_sf,
            # GEMM-domain tensors (fc1)
            activation_gemm,
            fc1_weight_gemm,
            fc1_output_gemm,
            activation_sf_gemm,
            fc1_weight_sf_gemm,
            fc1_output_sf_gemm,
            # GEMM-domain tensors (fc2)
            fc2_weight_gemm,
            fc2_output_gemm,
            fc2_weight_sf_gemm,
            fc1_output_sf_gemm_for_fc2_load,
            # topk + cross-phase sync workspace
            topk_scores,
            fc1_done_counter,
            load_balance_counter,
            # Scheduling
            offs,
            expert_token_sizes,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_vmnk,
            self.fallback_cluster_layout_sfb_vmnk,
            # SMEM layouts
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.d_smem_layout_staged,
            self.c_smem_layout_staged,
            tma_atom_c,
            tma_tensor_c,
            overflow_flag,
            token_comm_args,
            fc2_output_sf,
            mega_peer_rank_ptr_mapper,
            mega_local_rank,
            mega_local_workspace,
            mega_shared_workspace,
            mega_activation,
            mega_activation_sf,
            mega_pre_reduced_activation,
            mega_pre_reduced_activation_sf,
            tma_atom_fc2_output,
            fc2_tma_output,
            # Mixed CGA (== preferred objects when not mixed)
            tma_atom_fc1_weight_fallback,
            tma_atom_fc1_weight_sf_fallback,
            tma_atom_fc2_weight_fallback,
            tma_atom_fc2_weight_sf_fallback,
            weight_descriptor_workspace,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
            use_pdl=self.use_pdl,
            **(
                dict(
                    fallback_cluster=(
                        *self.resolved_fallback_cluster_shape_mn,
                        1,
                    )
                )
                if self.is_mixed_cga
                else {}
            ),
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        # fc1 TMA atoms / tensors
        tma_atom_fc1_activation_1: cute.CopyAtom,
        tma_tensor_fc1_activation_1: cute.Tensor,
        tma_atom_weight: TmaAtomOrPair,
        tma_tensor_weight: cute.Tensor,
        tma_atom_fc1_activation_1_sf: cute.CopyAtom,
        tma_tensor_fc1_activation_1_sf: cute.Tensor,
        tma_atom_fc1_weight_sf: TmaAtomOrPair,
        tma_tensor_fc1_weight_sf: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        tma_tensor_fc1_output: cute.Tensor,
        # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
        tma_atom_fc2_activation: cute.CopyAtom,
        tma_tensor_fc2_activation: cute.Tensor,
        tma_atom_fc2_weight: TmaAtomOrPair,
        tma_tensor_fc2_weight: cute.Tensor,
        tma_atom_fc2_activation_sf: cute.CopyAtom,
        tma_tensor_fc2_activation_sf: cute.Tensor,
        tma_atom_fc2_weight_sf: TmaAtomOrPair,
        tma_tensor_fc2_weight_sf: cute.Tensor,
        # GEMM-domain tensors (fc1)
        activation_gemm: cute.Tensor,
        fc1_weight_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        activation_sf_gemm: cute.Tensor,
        fc1_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2)
        fc2_weight_gemm: cute.Tensor,
        fc2_output_gemm: cute.Tensor,
        fc2_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm_for_fc2_load: cute.Tensor,
        # topk + cross-phase sync workspace
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        load_balance_counter: Optional[cute.Tensor],
        # Scheduling
        offs: Optional[cute.Tensor],
        expert_token_sizes: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        fallback_cluster_layout_vmnk: cute.Layout,
        fallback_cluster_layout_sfb_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        c_smem_layout_staged: Optional[Union[cute.Layout, cute.ComposedLayout]] = None,
        tma_atom_c: Optional[cute.CopyAtom] = None,
        tma_tensor_c: Optional[cute.Tensor] = None,
        overflow_flag: cute.Tensor = None,
        token_comm_args=None,
        fc2_output_sf: Optional[cute.Tensor] = None,
        # MegaMoE push-model token-comm inputs
        mega_peer_rank_ptr_mapper=None,
        mega_local_rank: Optional[cutlass.Int32] = None,
        mega_local_workspace: Optional[cute.Pointer] = None,
        mega_shared_workspace: Optional[cute.Pointer] = None,
        mega_activation: Optional[cute.Tensor] = None,
        mega_activation_sf: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation: Optional[cute.Tensor] = None,
        mega_pre_reduced_activation_sf: Optional[cute.Tensor] = None,
        # FC2 TMASTG (fc2_use_bulk): store atom + token-major pool view (None off-path)
        tma_atom_fc2_output: Optional[cute.CopyAtom] = None,
        fc2_tma_output: Optional[cute.Tensor] = None,
        # Mixed CGA: fallback-shaped B/SFB atoms + fallback cluster layouts
        # (all equal to the preferred objects when not mixed).
        tma_atom_weight_fallback: Optional[cute.CopyAtom] = None,
        tma_atom_fc1_weight_sf_fallback: Optional[cute.CopyAtom] = None,
        tma_atom_fc2_weight_fallback: Optional[cute.CopyAtom] = None,
        tma_atom_fc2_weight_sf_fallback: Optional[cute.CopyAtom] = None,
        weight_descriptor_workspace: Optional[cute.Pointer] = None,
    ):
        """Device kernel for fused fc1+fc2 swap-AB GLU MXFP8 grouped GEMM."""
        # MegaMoE (push model)
        if cutlass.const_expr(self.enable_token_comm):
            self._mega_device_workspace.assign_device_members(
                mega_local_workspace, mega_shared_workspace
            )

        # fc2 waits for all fc1 intermediate N-tiles in the same token block. Every CTA that
        # shares the slot (fc1_done_slot_m of them in M) increments it once per N-tile.
        ext_fc2_spin_threshold = (
            (fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[1] - 1)
            // self.cta_tile_shape_mnk[1]
            * self.fc1_done_slot_m
        )

        ext = GluMxFp8Fc12SchedExtension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_pointer=self.token_comm_hook_fc1_ready_counter_ptr(token_comm_args),
            cluster_m=self.cluster_shape_mn[0],
            done_slot_m=self.fc1_done_slot_m,
            weight_storage_mode=self.weight_storage_mode,
            weight_descriptor_workspace=weight_descriptor_workspace,
            is_mixed_cga=self.is_mixed_cga,
        )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        active_cluster_m = cutlass.Int32(self.cluster_shape_mn[0])
        is_fallback_cluster = cutlass.Boolean(False)
        if cutlass.const_expr(self.is_mixed_cga):
            active_cluster_m, _, _ = cute.arch.block_in_cluster_dim()
            is_fallback_cluster = active_cluster_m == cutlass.Int32(
                self.resolved_fallback_cluster_shape_mn[0]
            )
        is_preferred_cluster = is_fallback_cluster == cutlass.Boolean(False)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Mixed CGA: detect the actual hardware cluster this CTA landed in (M-only compare;
        # cluster N == 1 in both shapes).
        is_fallback_cluster = cutlass.Boolean(False)
        if cutlass.const_expr(self.is_mixed_cga):
            _active_cluster_m, _, _ = cute.arch.block_in_cluster_dim()
            is_fallback_cluster = _active_cluster_m == cutlass.Int32(
                self.resolved_fallback_cluster_shape_mn[0]
            )
        tidx, _, _ = cute.arch.thread_idx()

        # PDL: let the dependent grid start launching once every CTA of this grid is resident.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # MegaMoE (push model): bind token-comm device members (transfer-warp state + the
        # NVLink barrier's peer mapper) before any token_in / token_back / size-wait runs.
        if cutlass.const_expr(self.enable_token_comm):
            _mega_token_comm_args = TokenCommArgs(
                mega_activation,
                mega_activation_sf,
                mega_pre_reduced_activation,
                mega_pre_reduced_activation_sf,
                mega_peer_rank_ptr_mapper,
            )
            _mega_linear_cta_idx = (
                bidx
                + bidy * cutlass.Int32(self.cluster_shape_mn[0])
                + bidz * cutlass.Int32(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])
            )
            self.token_comm.assign_device_members(
                device_workspace=self._mega_device_workspace,
                token_comm_args=_mega_token_comm_args,
                local_rank=mega_local_rank,
                linear_cta_idx=_mega_linear_cta_idx,
            )

        # SharedStorage (mainloop + epilogue SMEM). next's scheduler owns its own
        # SMEM workspace, allocated separately below.
        #
        # The six mbarrier fields must stay adjacent, in order and unpadded for the folded init below.
        @cute.struct
        class SharedStorage:
            a_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_pipeline_stages * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            epi_scratch: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, self.epilogue.scratch_bytes],
                128,
            ]

        _mbar_field_offsets = SharedStorage._offsets
        _mbar_bytes = cutlass.Int64.width // 8
        assert _mbar_field_offsets["b_full_mbar_ptr"] == (
            _mbar_field_offsets["a_full_mbar_ptr"] + self.num_a_stage * 2 * _mbar_bytes
        ), "SharedStorage: b_full_mbar_ptr must directly follow a_full_mbar_ptr"
        assert _mbar_field_offsets["acc_full_mbar_ptr"] == (
            _mbar_field_offsets["b_full_mbar_ptr"] + self.num_b_stage * 2 * _mbar_bytes
        ), "SharedStorage: acc_full_mbar_ptr must directly follow b_full_mbar_ptr"

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # next scheduler SMEM: a self-contained workspace carved from the same
        # allocator; its transport regions resolve against ``sched_smem_base``.
        sched_storage = smem.allocate(self.sched_smem_ws.storage_class())
        sched_smem_base = sched_storage.buffer.data_ptr()

        # MegaMoE-only dispatch-warp SMEM (pull_buffer, mbarriers, etc.).
        # Kept out of ``SharedStorage`` so the lean path never allocates it.
        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls)
        else:
            token_comm_storage = None

        # ── Pipelines: separate producer/consumer groups for A and B. ──

        a_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        a_pipeline = PipelineTmaUmmaMixedCga.create(
            barrier_storage=storage.a_full_mbar_ptr.data_ptr(),
            num_stages=self.num_a_stage,
            producer_group=a_pipeline_producer_group,
            num_mma_consumer_warps=1,
            tx_count=self.num_tma_load_a_bytes,
            preferred_cta_layout_vmnk=cluster_layout_vmnk,
            fallback_cta_layout_vmnk=fallback_cluster_layout_vmnk,
            cta_coord_vmnk=block_in_cluster_coord_vmnk,
            is_fallback_cluster=is_fallback_cluster,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        b_pipeline = PipelineTmaUmmaMixedCga.create(
            barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=b_pipeline_producer_group,
            num_mma_consumer_warps=1,
            tx_count=self.num_tma_load_b_bytes,
            preferred_cta_layout_vmnk=cluster_layout_vmnk,
            fallback_cta_layout_vmnk=fallback_cluster_layout_vmnk,
            cta_coord_vmnk=block_in_cluster_coord_vmnk,
            is_fallback_cluster=is_fallback_cluster,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * 32 * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # TMEM allocator
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        scheduler = self.scheduler
        if cutlass.const_expr(self.enable_token_comm):
            _sched_expert_sizes = self.token_comm.local_expert_sizes(
                self._mega_device_workspace, mega_local_rank
            )
            _sched_prefix_sum = None
            self.sched_device_ws.assign_device_members(
                cute.make_ptr(
                    cutlass.Uint8,
                    mega_local_workspace.toint()
                    + self._mega_device_workspace.offset(self.sched_work_id_region),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                mega_shared_workspace,
            )
        else:
            _sched_expert_sizes = expert_token_sizes
            _sched_prefix_sum = offs
        # Lean atomic mode: point the scheduler's work-id counter region at the caller's
        # load_balance_counter buffer.
        if cutlass.const_expr(
            self.load_balance_mode == "atomic_counter" and load_balance_counter is not None
        ):
            self.sched_device_ws.assign_device_members(load_balance_counter.iterator, None)
        _sched_is_fallback = None
        if cutlass.const_expr(self.is_mixed_cga):
            _sched_is_fallback = is_fallback_cluster
        # One-shot: every CTA derives its tile from blockIdx.z, so the scheduler warp and its
        # state init are compiled out.
        if cutlass.const_expr(self.one_shot_sched):
            _, _one_shot_gateup, _one_shot_hidden = self.static_expert_shape
            _one_shot_tile_n = self.cta_tile_shape_mnk[1]
            sched_consumer = OneShotSchedConsumer(
                (_one_shot_gateup + _one_shot_tile_n - 1) // _one_shot_tile_n,
                (_one_shot_hidden + _one_shot_tile_n - 1) // _one_shot_tile_n,
                (self.cta_tile_shape_mnk[0] << 16) | self.cta_tile_shape_mnk[0],
            )
            # The uninitialized scheduler must not be captured by dynamic-if regions below.
            scheduler = None
        else:
            scheduler.assign_device_members(
                expert_token_sizes=_sched_expert_sizes,
                expert_token_prefix_sum=_sched_prefix_sum,
                actual_expert_shape=None,
                block_idx=cute.arch.block_idx(),
                smem_workspace=self.sched_smem_ws,
                smem_base=sched_smem_base,
                device_workspace=self.sched_device_ws,
                is_fallback_cluster=_sched_is_fallback,
            )
            sched_consumer = scheduler.make_consumer()

        pipeline_init_arrive_mixed_cga(
            self.cluster_shape_mn,
            self.resolved_fallback_cluster_shape_mn,
            is_fallback_cluster,
            is_relaxed=True,
        )

        # ── SMEM tensors A / B / SFA / SFB (shared by fc1 / fc2) ──
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # acc_fake layout: (MMA, MMA_M, MMA_N, STAGE).
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before TMEM alloc.
        pipeline_init_wait_mixed_cga(
            self.cluster_shape_mn,
            self.resolved_fallback_cluster_shape_mn,
            is_fallback_cluster,
        )

        mma_tiler_k = self.mma_tiler[2]
        k_tile_cnt_fc1 = (fc1_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        k_tile_cnt_fc2 = (fc2_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k

        cluster_n = self.cluster_shape_mn[1]
        _fc1_n_cluster_tile = self.cta_tile_shape_mnk[1] * cluster_n
        fc2_spin_threshold = (
            ((fc1_weight_gemm.shape[0] + _fc1_n_cluster_tile - 1) // _fc1_n_cluster_tile)
            * cluster_n
            * self.fc1_done_slot_m
        )
        # fc2 consumer passes per token band: the high-half target of the self-resetting fc1_done
        # slot under PDL (bounds checked on the host).
        fc2_consumer_count = (
            ((fc2_weight_gemm.shape[0] + _fc1_n_cluster_tile - 1) // _fc1_n_cluster_tile)
            * cluster_n
            * self.epilogue._atom_thr_size
        )

        # Warpgroup-scoped register redistribution: warps 8..11 form one warpgroup and must all
        # execute the .dec inside their own region.
        if cutlass.const_expr(self.breuse_two_warpgroups):
            if warp_idx >= self.mma_warp_id:
                cute.arch.setmaxregister_decrease(self.num_regs_other_warps)

        # ════════════════════════════════════════════════════════════════════
        # Scheduler warp — lean path
        # ════════════════════════════════════════════════════════════════════
        if warp_idx == self.sched_warp_id:
            # One-shot mode compiles the whole producer loop out (its publish
            # pipeline would otherwise block with no consumers).
            if cutlass.const_expr(not self.one_shot_sched):
                # PDL: expert sizes / offsets and the work-id counters are written
                # by the preceding grid.
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                # Mixed CGA (atomic mode): each physical fallback cluster registers
                # into a fixed logical preferred-cluster group BEFORE any claim.
                if cutlass.const_expr(
                    self.is_mixed_cga and self.load_balance_mode == "atomic_counter"
                ):
                    scheduler.initialize_fallback_group()
                # MegaMoE: block until the Router has published this rank's per-expert sizes
                # (cross-rank), so the lazy size walk in ``gen_next_work`` reads valid counts.
                # No-op on the lean path.
                self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
                work_tile = scheduler.gen_next_work()
                while work_tile.is_valid_tile:
                    scheduler.publish_work(ext.prepare_work_tile(work_tile))
                    work_tile = scheduler.gen_next_work()
                # Sentinel publish (the tile is already invalid here).
                scheduler.publish_work(work_tile)
                scheduler.produce_tail()

        # ════════════════════════════════════════════════════════════════════
        # TMA load warps (tma_a / tma_b)
        # ════════════════════════════════════════════════════════════════════
        #
        # TMA-A loads activations/SFA into the A pipeline.
        # TMA-B loads weights/SFB into the B pipeline and waits for
        # fc1 workspace readiness in the fc2 phase.

        # ── TMA-A warp ─────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            _iket_active = tidx == cutlass.Int32(32 * self.tma_a_warp_id)
            a_full_mcast_mask = None
            sfa_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            sfa_cta_layout = a_cta_layout

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_a_stage
            )

            # PDL: activations, SFA and the fc1_done slots belong to the
            # preceding grid's writes; nothing above this line reads them.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                if is_phase_linear1:
                    # fc1 phase A-side
                    if _iket_active:
                        iket.range_push("tma_token_fc1")
                    ext.wait_for_input(work_tile_info)
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args,
                        work_tile_info,
                    )

                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc1_activation",
                        tma_tensor_fc1_activation_1,
                        work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "fc1_activation_sf",
                        tma_tensor_fc1_activation_1_sf,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Producer-side backpressure: TMA-A blocked waiting for the
                        # MMA to free an A SMEM slot. First k-tile only (later tiles
                        # overlap via peek-ahead). Complements mma_ab_operand_wait.
                        if _iket_active:
                            iket.range_push("tma_a_buf_acquire_wait")
                        a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)
                        if _iket_active:
                            iket.range_pop()  # tma_a_buf_acquire_wait
                        tma_barrier = a_pipeline.producer_get_barrier(a_producer_state)
                        stage_index = a_producer_state.index
                        a_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if k_tile + 1 < k_tile_cnt:
                            peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)
                        cute.copy(
                            tma_atom_fc1_activation_1,
                            tAgA_slice[(None, k_tile)],
                            tAsA[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc1_activation_1_sf,
                            tAgSFA_slice[(None, k_tile)],
                            tAsSFA[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                else:
                    # fc2 phase A-side: load fc1_output (M=tokens) + wait for fc1 done
                    if _iket_active:
                        iket.range_push("tma_token_fc2")

                    # Must mirror the epilogue's producer slot exactly.
                    counter_slot = work_tile_info.cumulative_token_block_count * cutlass.Int32(
                        self.cluster_shape_mn[0] // self.fc1_done_slot_m
                    ) + work_tile_info.tile_m_idx // cutlass.Int32(self.fc1_done_slot_m)
                    counter_ptr = fc1_done_counter.iterator + counter_slot
                    if _iket_active:
                        iket.range_push("tma_token_fc2_a_wait")
                    if cutlass.const_expr(self.use_pdl):
                        # Self-resetting slot: the high half is only non-zero after the low half
                        # completed, so the masked test is exact.
                        spin_wait(
                            counter_ptr,
                            lambda v: v % cutlass.Int32(self.FC1_DONE_FC2_PASS_UNIT)
                            >= fc2_spin_threshold,
                            sleep_cycles=20,
                        )
                    else:
                        spin_wait(
                            counter_ptr,
                            lambda v: v >= fc2_spin_threshold,
                            sleep_cycles=20,
                        )
                    if _iket_active:
                        iket.range_pop()

                    cute.arch.load(counter_ptr, counter_ptr.dtype, sem="acquire", scope="gpu")
                    cute.arch.fence_proxy("async")
                    cute.arch.fence_proxy("async.global")

                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc2_activation",
                        tma_tensor_fc2_activation,
                        work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "fc2_activation_sf",
                        tma_tensor_fc2_activation_sf,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_activation,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc2_activation_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    # fc2 A-side = fc1_output (M=tokens).
                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(tiled_mma.thr_id.shape)
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Producer-side backpressure: TMA-A blocked waiting for the
                        # MMA to free an A SMEM slot. First k-tile only (later tiles
                        # overlap via peek-ahead). Complements mma_ab_operand_wait.
                        if _iket_active:
                            iket.range_push("tma_a_buf_acquire_wait")
                        a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)
                        if _iket_active:
                            iket.range_pop()  # tma_a_buf_acquire_wait
                        tma_barrier = a_pipeline.producer_get_barrier(a_producer_state)
                        stage_index = a_producer_state.index
                        a_producer_state.advance()
                        peek_a_empty_status = cutlass.Boolean(1)
                        if k_tile + 1 < k_tile_cnt:
                            peek_a_empty_status = a_pipeline.producer_try_acquire(a_producer_state)
                        cute.copy(
                            tma_atom_fc2_activation,
                            tAgA_slice[(None, k_tile)],
                            tAsA[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_activation_sf,
                            tAgSFA_slice[(None, k_tile)],
                            tAsSFA[(None, stage_index)],
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )

                    if cutlass.const_expr(self.use_pdl):
                        # Self-resetting slot: count this pass in the high half; the last consumer
                        # of the band resets the slot for the next launch (relaxed ordering suffices).
                        if cute.arch.lane_idx() == 0:
                            _fc2_pass_old = cute.arch.atomic_add(
                                counter_ptr,
                                cutlass.Int32(self.FC1_DONE_FC2_PASS_UNIT),
                                sem="relaxed",
                                scope="gpu",
                            )
                            if (
                                _fc2_pass_old // cutlass.Int32(self.FC1_DONE_FC2_PASS_UNIT)
                                == fc2_consumer_count - 1
                            ):
                                cute.arch.store(
                                    counter_ptr,
                                    cutlass.Int32(0),
                                    sem="relaxed",
                                    scope="gpu",
                                )

                if _iket_active:
                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            a_pipeline.producer_tail(a_producer_state)

        # ── TMA-B warp ─────────────────────────────────────────────
        if warp_idx == self.tma_b_warp_id:
            _iket_active = tidx == cutlass.Int32(32 * self.tma_b_warp_id)
            fc1_weight_partition_atom = (
                tma_atom_weight[0] if isinstance(tma_atom_weight, tuple) else tma_atom_weight
            )
            fc1_weight_sf_partition_atom = (
                tma_atom_fc1_weight_sf[0]
                if isinstance(tma_atom_fc1_weight_sf, tuple)
                else tma_atom_fc1_weight_sf
            )
            fc2_weight_partition_atom = (
                tma_atom_fc2_weight[0]
                if isinstance(tma_atom_fc2_weight, tuple)
                else tma_atom_fc2_weight
            )
            fc2_weight_sf_partition_atom = (
                tma_atom_fc2_weight_sf[0]
                if isinstance(tma_atom_fc2_weight_sf, tuple)
                else tma_atom_fc2_weight_sf
            )
            fc1_weight_exec_atom = _make_executable_tma_atom_for_cluster(
                tma_atom_weight, is_fallback_cluster
            )
            fc1_weight_sf_exec_atom = _make_executable_tma_atom_for_cluster(
                tma_atom_fc1_weight_sf, is_fallback_cluster
            )
            fc2_weight_exec_atom = _make_executable_tma_atom_for_cluster(
                tma_atom_fc2_weight, is_fallback_cluster
            )
            fc2_weight_sf_exec_atom = _make_executable_tma_atom_for_cluster(
                tma_atom_fc2_weight_sf, is_fallback_cluster
            )
            b_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                # Clusters are M-major, and mixed-CGA only narrows M, so the preferred
                # cluster M splits a physical rank into its (M, N) coordinate.
                mcast_cta_coord = (
                    cta_rank_in_cluster % cutlass.Int32(self.cluster_shape_mn[0]),
                    cta_rank_in_cluster // cutlass.Int32(self.cluster_shape_mn[0]),
                    cutlass.Int32(0),
                )
                b_full_mcast_mask = tma_multicast_mask(
                    self.cluster_shape_mn,
                    self.resolved_fallback_cluster_shape_mn if self.is_mixed_cga else None,
                    mcast_cta_coord,
                    is_preferred_cluster if self.is_mixed_cga else None,
                    self.use_2cta_instrs,
                    "b",
                )
                sfb_full_mcast_mask = tma_multicast_mask(
                    self.cluster_shape_mn,
                    self.resolved_fallback_cluster_shape_mn if self.is_mixed_cga else None,
                    mcast_cta_coord,
                    is_preferred_cluster if self.is_mixed_cga else None,
                    self.use_2cta_instrs,
                    "sfb",
                )

            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )

            # Mixed CGA: fallback-cluster coords / masks / partition layouts, only meaningful
            # when is_fallback_cluster.
            if cutlass.const_expr(self.is_mixed_cga):
                block_in_cluster_coord_vmnk_fallback = fallback_cluster_layout_vmnk.get_flat_coord(
                    cta_rank_in_cluster
                )
                block_in_cluster_coord_sfb_vmnk_fallback = (
                    fallback_cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
                )
                b_cta_layout_fallback = cute.make_layout(
                    cute.slice_(fallback_cluster_layout_vmnk, (0, None, 0, 0)).shape
                )
                sfb_cta_layout_fallback = cute.make_layout(
                    cute.slice_(fallback_cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
                )
                _b_mask_fallback = cpasync.create_tma_multicast_mask(
                    fallback_cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk_fallback,
                    mcast_mode=1,
                )
                _sfb_mask_fallback = cpasync.create_tma_multicast_mask(
                    fallback_cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk_fallback,
                    mcast_mode=1,
                )
                if is_fallback_cluster:
                    b_full_mcast_mask = _b_mask_fallback
                    sfb_full_mcast_mask = _sfb_mask_fallback

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
            b_partition_coord = block_in_cluster_coord_vmnk[1]
            sfb_partition_coord = block_in_cluster_coord_sfb_vmnk[1]
            if cutlass.const_expr(self.is_mixed_cga):
                if is_fallback_cluster:
                    split_factor_m = (
                        self.cluster_shape_mn[0] // self.resolved_fallback_cluster_shape_mn[0]
                    )
                    b_partition_coord = b_partition_coord * cutlass.Int32(split_factor_m)
                    sfb_partition_coord = sfb_partition_coord * cutlass.Int32(split_factor_m)

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_b_stage
            )

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)

                if is_phase_linear1:
                    # fc1 phase B-side (fc1_weight, N-side GEMM-B)
                    if _iket_active:
                        iket.range_push("tma_weight_fc1")

                    k_tile_cnt = k_tile_cnt_fc1
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc1_weight",
                        tma_tensor_weight,
                        work_tile_info,
                        is_fallback_cluster,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "fc1_weight_sf",
                        tma_tensor_fc1_weight_sf,
                        work_tile_info,
                        is_fallback_cluster,
                    )

                    # N-K tiling for N-side weight (N=intermediate, K=hidden).
                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )

                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        fc1_weight_partition_atom,
                        b_partition_coord,
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        fc1_weight_sf_partition_atom,
                        sfb_partition_coord,
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    # Use tile_n_idx for N-side weight: invariant across token blocks.
                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                    # N64: one SFB block covers two N tiles, so the gmem slice index halves;
                    # the MMA side selects the odd half in TMEM.
                    fc1_sfb_slice_n = work_tile_info.tile_n_idx
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        fc1_sfb_slice_n = work_tile_info.tile_n_idx // 2
                    tBgSFB_slice = tBgSFB[(None, fc1_sfb_slice_n, None, 0)]

                    if cutlass.const_expr(self.is_mixed_cga):
                        tBsB_fallback, tBgB_fallback = cpasync.tma_partition(
                            tma_atom_weight_fallback,
                            block_in_cluster_coord_vmnk_fallback[1],
                            b_cta_layout_fallback,
                            cute.group_modes(sB, 0, 3),
                            cute.group_modes(tCgB, 0, 3),
                        )
                        tBsSFB_fallback, tBgSFB_fallback = cpasync.tma_partition(
                            tma_atom_fc1_weight_sf_fallback,
                            block_in_cluster_coord_sfb_vmnk_fallback[1],
                            sfb_cta_layout_fallback,
                            cute.group_modes(sSFB, 0, 3),
                            cute.group_modes(tCgSFB, 0, 3),
                        )
                        tBsSFB_fallback = cute.filter_zeros(tBsSFB_fallback)
                        tBgSFB_fallback = cute.filter_zeros(tBgSFB_fallback)
                        tBgB_fallback_slice = tBgB_fallback[
                            (None, work_tile_info.tile_n_idx, None, 0)
                        ]
                        tBgSFB_fallback_slice = tBgSFB_fallback[(None, fc1_sfb_slice_n, None, 0)]

                    peek_b_empty_status = b_pipeline.producer_try_acquire(b_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Producer-side backpressure: TMA-B blocked waiting for the
                        # MMA to free a B SMEM slot. First k-tile only (later tiles
                        # overlap via peek-ahead). Complements mma_ab_operand_wait.
                        if _iket_active:
                            iket.range_push("tma_b_buf_acquire_wait")
                        b_pipeline.producer_acquire(b_producer_state, peek_b_empty_status)
                        if _iket_active:
                            iket.range_pop()  # tma_b_buf_acquire_wait
                        tma_barrier = b_pipeline.producer_get_barrier(b_producer_state)
                        stage_index = b_producer_state.index
                        b_producer_state.advance()
                        peek_b_empty_status = cutlass.Boolean(1)
                        if k_tile + 1 < k_tile_cnt:
                            peek_b_empty_status = b_pipeline.producer_try_acquire(b_producer_state)
                        b_issue_atom = bind_executable_tma_load_fields(
                            fc1_weight_exec_atom,
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            b_issue_atom, tBgB_slice[(None, k_tile)], tBsB[(None, stage_index)]
                        )
                        sfb_issue_atom = bind_executable_tma_load_fields(
                            fc1_weight_sf_exec_atom,
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                        cute.copy(
                            sfb_issue_atom,
                            tBgSFB_slice[(None, k_tile)],
                            tBsSFB[(None, stage_index)],
                        )
                else:
                    # fc2 phase B-side: load fc2_weight (N=hidden)
                    if _iket_active:
                        iket.range_push("tma_weight_fc2")

                    k_tile_cnt = k_tile_cnt_fc2
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc2_weight",
                        tma_tensor_fc2_weight,
                        work_tile_info,
                        is_fallback_cluster,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "fc2_weight_sf",
                        tma_tensor_fc2_weight_sf,
                        work_tile_info,
                        is_fallback_cluster,
                    )

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        fc2_weight_partition_atom,
                        b_partition_coord,
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        fc2_weight_sf_partition_atom,
                        sfb_partition_coord,
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    # fc2 B-side = fc2_weight (N=hidden)
                    fc2_b_hidden_tile = work_tile_info.tile_n_idx
                    tBgB_slice = tBgB[(None, fc2_b_hidden_tile, None, 0)]
                    fc2_sfb_slice_n = fc2_b_hidden_tile
                    if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                        fc2_sfb_slice_n = fc2_b_hidden_tile // 2
                    tBgSFB_slice = tBgSFB[(None, fc2_sfb_slice_n, None, 0)]

                    if cutlass.const_expr(self.is_mixed_cga):
                        tBsB_fallback, tBgB_fallback = cpasync.tma_partition(
                            tma_atom_fc2_weight_fallback,
                            block_in_cluster_coord_vmnk_fallback[1],
                            b_cta_layout_fallback,
                            cute.group_modes(sB, 0, 3),
                            cute.group_modes(tCgB, 0, 3),
                        )
                        tBsSFB_fallback, tBgSFB_fallback = cpasync.tma_partition(
                            tma_atom_fc2_weight_sf_fallback,
                            block_in_cluster_coord_sfb_vmnk_fallback[1],
                            sfb_cta_layout_fallback,
                            cute.group_modes(sSFB, 0, 3),
                            cute.group_modes(tCgSFB, 0, 3),
                        )
                        tBsSFB_fallback = cute.filter_zeros(tBsSFB_fallback)
                        tBgSFB_fallback = cute.filter_zeros(tBgSFB_fallback)
                        tBgB_fallback_slice = tBgB_fallback[(None, fc2_b_hidden_tile, None, 0)]  # noqa: F841
                        tBgSFB_fallback_slice = tBgSFB_fallback[(None, fc2_sfb_slice_n, None, 0)]  # noqa: F841

                    peek_b_empty_status = b_pipeline.producer_try_acquire(b_producer_state)

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Producer-side backpressure: TMA-B blocked waiting for the
                        # MMA to free a B SMEM slot. First k-tile only (later tiles
                        # overlap via peek-ahead). Complements mma_ab_operand_wait.
                        if _iket_active:
                            if k_tile == 0:
                                iket.range_push("tma_b_buf_acquire_wait")
                        b_pipeline.producer_acquire(b_producer_state, peek_b_empty_status)
                        if _iket_active:
                            if k_tile == 0:
                                iket.range_pop()  # tma_b_buf_acquire_wait
                        tma_barrier = b_pipeline.producer_get_barrier(b_producer_state)
                        stage_index = b_producer_state.index
                        b_producer_state.advance()
                        peek_b_empty_status = cutlass.Boolean(1)
                        if k_tile + 1 < k_tile_cnt:
                            peek_b_empty_status = b_pipeline.producer_try_acquire(b_producer_state)
                        b_issue_atom = bind_executable_tma_load_fields(
                            fc2_weight_exec_atom,
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            b_issue_atom, tBgB_slice[(None, k_tile)], tBsB[(None, stage_index)]
                        )
                        sfb_issue_atom = bind_executable_tma_load_fields(
                            fc2_weight_sf_exec_atom,
                            tma_bar_ptr=tma_barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                        cute.copy(
                            sfb_issue_atom,
                            tBgSFB_slice[(None, k_tile)],
                            tBsSFB[(None, stage_index)],
                        )
                if _iket_active:
                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            b_pipeline.producer_tail(b_producer_state)

        # ════════════════════════════════════════════════════════════════════
        # MMA warp
        # ════════════════════════════════════════════════════════════════════
        #
        # Both phases share tiled_mma and TMEM; only K-tile count differs.
        if warp_idx == self.mma_warp_id:
            _iket_active = tidx == cutlass.Int32(32 * self.mma_warp_id)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_base = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            # SFA TMEM tensor (placed after the acc cols).
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # SFB TMEM tensor (after acc + SFA cols).
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages
            )
            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_a_stage
            )
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_b_stage
            )

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                # Prebind k_tile_cnt due to DSL AST.
                k_tile_cnt = cutlass.Int32(0)
                if is_phase_linear1:
                    k_tile_cnt = k_tile_cnt_fc1
                    if _iket_active:
                        iket.range_push("mma_fc1")
                else:
                    k_tile_cnt = k_tile_cnt_fc2
                    if _iket_active:
                        iket.range_push("mma_fc2")

                acc_stage_index = acc_producer_state.index

                # N64: the SFB TMEM block holds two N tiles' scales; odd tiles shift the MMA
                # SFB pointer by 2 words.
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    _sfb_shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + cutlass.Int32((work_tile_info.tile_n_idx % 2) * 2),
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(_sfb_shifted_ptr, tCtSFB_layout)

                if is_leader_cta:
                    tCtAcc = acc_base[(None, None, None, acc_stage_index)]

                    if _iket_active:
                        iket.range_push("mma_acc_acquire")
                    peek_a_full_status = cutlass.Boolean(1)
                    peek_b_full_status = cutlass.Boolean(1)
                    next_a_consumer_state = a_consumer_state.clone()
                    next_b_consumer_state = b_consumer_state.clone()
                    if k_tile_cnt > 0:
                        next_a_consumer_state.advance()
                        next_b_consumer_state.advance()
                        peek_a_full_status = a_pipeline.consumer_try_wait(a_consumer_state)
                        peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)
                        acc_pipeline.producer_acquire(acc_producer_state)
                    if _iket_active:
                        iket.range_pop()

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Isolate the real AB-operand stall: the initial k-tile wait
                        # for A/B to arrive from the TMA warps. Later k-tiles overlap
                        # with MMA compute via the peek-ahead pipeline, so only the
                        # first tile's wait reflects true operand-arrival latency.
                        if _iket_active:
                            iket.range_push("mma_ab_operand_wait")
                        a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)
                        b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)
                        if _iket_active:
                            iket.range_pop()  # mma_ab_operand_wait
                        peek_a_full_status = cutlass.Boolean(1)
                        peek_b_full_status = cutlass.Boolean(1)
                        if k_tile + 1 < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(next_a_consumer_state)
                            peek_b_full_status = b_pipeline.consumer_try_wait(next_b_consumer_state)

                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[(None, None, None, None, a_consumer_state.index)],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[(None, None, None, None, b_consumer_state.index)],
                            tCtSFB_compact_s2t,
                        )

                        if cutlass.const_expr(self.enable_breuse):
                            # B-reuse: one B/SFB tile feeds two MMAs over the two A M-halves
                            # (FILL then LASTUSE in the collector).
                            tCtAcc_keep = tCtAcc[(None, 0, 0)]
                            tCtAcc_reuse = tCtAcc[(None, 1, 0)]
                            num_kblocks = cute.size(tCrA, mode=[2])
                            for k_block in cutlass.range(num_kblocks, unroll_full=True):
                                cute.gemm(
                                    tiled_mma.with_bkeep(accumulate=k_tile != 0 or k_block != 0),
                                    tCtAcc_keep,
                                    [
                                        tCrA[(None, 0, k_block, a_consumer_state.index)],
                                        tCtSFA[(None, 0, k_block)],
                                    ],
                                    [
                                        tCrB[(None, 0, k_block, b_consumer_state.index)],
                                        tCtSFB_mma[(None, 0, k_block)],
                                    ],
                                    tCtAcc_keep,
                                )
                                cute.gemm(
                                    tiled_mma.with_breuse(accumulate=k_tile != 0 or k_block != 0),
                                    tCtAcc_reuse,
                                    [
                                        tCrA[(None, 1, k_block, a_consumer_state.index)],
                                        tCtSFA[(None, 1, k_block)],
                                    ],
                                    [
                                        tCrB[(None, 0, k_block, b_consumer_state.index)],
                                        tCtSFB_mma[(None, 0, k_block)],
                                    ],
                                    tCtAcc_reuse,
                                )
                        else:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                [tCrA[(None, None, None, a_consumer_state.index)], tCtSFA],
                                [tCrB[(None, None, None, b_consumer_state.index)], tCtSFB_mma],
                                tCtAcc,
                            )
                        a_pipeline.consumer_release(a_consumer_state)
                        b_pipeline.consumer_release(b_consumer_state)
                        a_consumer_state.advance()
                        b_consumer_state.advance()
                        next_a_consumer_state.advance()
                        next_b_consumer_state.advance()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)

                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                if _iket_active:
                    iket.range_pop()

                work_tile_info = sched_consumer.consume_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # FC1 (sD + optional sC) and FC2 (sFC2 or sRED) scratch have
        # mutually-exclusive task-tile lifetimes.  Reinterpret one aligned raw
        # slab with each path's own dtype/layout after the tile-boundary drain.
        epi_scratch_ptr = storage.epi_scratch.data_ptr()
        sD = cute.make_tensor(
            cute.recast_ptr(
                epi_scratch_ptr,
                d_smem_layout_staged.inner,
                dtype=self.fc1_output_dtype,
            ),
            d_smem_layout_staged.outer,
        )

        if cutlass.const_expr(self.generate_c):
            sC = cute.make_tensor(
                cute.recast_ptr(
                    epi_scratch_ptr + self.epilogue.fc1_c_scratch_offset,
                    c_smem_layout_staged.inner,
                    dtype=self.epilogue._c_dtype,
                ),
                c_smem_layout_staged.outer,
            )

        if cutlass.const_expr(self.epilogue.fc2_needs_staging):
            sFC2 = cute.make_tensor(
                cute.recast_ptr(
                    epi_scratch_ptr,
                    dtype=self.epilogue._fc2_wire_dtype,
                ),
                self.epilogue.fc2_tma_staged_smem_layout(self.epilogue.fc2_tma_stages),
            )

        if cutlass.const_expr(self.epilogue.fc2_reduce_coalesce):
            sRED = cute.make_tensor(
                cute.recast_ptr(
                    epi_scratch_ptr,
                    dtype=cutlass.BFloat16,
                ),
                self.epilogue.fc2_reduce_smem_layout(),
            )

        # ════════════════════════════════════════════════════════════════════
        # Epilogue warps
        # ════════════════════════════════════════════════════════════════════
        #
        # Fully delegated to ``self.epilogue.run(...)`` -- the epilogue owns
        # the entire 2-phase task-tile loop.
        if warp_idx < self.mma_warp_id:
            epi_warp_idx = warp_idx
            if cutlass.const_expr(self.breuse_two_warpgroups):
                cute.arch.setmaxregister_increase(self.num_regs_epilogue_warps)

            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_tensor = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            # acc_tensor = cute.make_tensor(
            #    acc_tmem_ptr,
            #    cute.make_layout(
            #        (((128, self.cta_tile_shape_mnk[1]), 1),),
            #        stride=(((1 << 16, 1), 0),),
            #    ),
            # )

            # PDL: topk scores are read, and the outputs / fc1_done slots are
            # written, from here on; the TMEM allocation above is local.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # Build common kwargs shared by both epilogue flavours.
            if cutlass.const_expr(self.generate_c):
                _smem_c_raw_arg = sC
                _tma_atom_c_arg = tma_atom_c
                _gmem_c_arg = tma_tensor_c
            else:
                _smem_c_raw_arg = None
                _tma_atom_c_arg = None
                _gmem_c_arg = None
            if cutlass.const_expr(self.use_stg_fc1):
                _gmem_fc1_output_arg = fc1_output_gemm
            else:
                _gmem_fc1_output_arg = tma_tensor_fc1_output
            _run_kwargs = dict(
                tmem_acc_tensor=acc_tensor,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                smem_fc1_output_buffer=sD,
                tma_atom_fc1_output=tma_atom_fc1_output,
                gmem_fc1_output=_gmem_fc1_output_arg,
                gmem_fc1_output_sf=fc1_output_sf_gemm,
                gmem_topk_scores=topk_scores,
                gmem_fc2_output=fc2_output_gemm,
                gmem_fc1_done_counter=fc1_done_counter,
                smem_c_buffer=_smem_c_raw_arg,
                tma_atom_c=_tma_atom_c_arg,
                gmem_c=_gmem_c_arg,
                warp_idx=epi_warp_idx,
                tidx=tidx,
                alpha=cutlass.Float32(1.0),
                norm_const=cutlass.Float32(1.0),
            )

            _run_extra = {}
            if cutlass.const_expr(self.fc2_sf_blocked and fc2_output_sf is not None):
                # Blocked fc2 scale-factor plane (tile_atom_to_shape_SF layout), the
                # same encoding as fc1_output_sf.
                _run_extra["gmem_fc2_output_sf_blocked"] = cute.make_tensor(
                    fc2_output_sf.iterator,
                    blockscaled_utils.tile_atom_to_shape_SF(
                        (fc2_output_gemm.shape[0], fc2_output_gemm.shape[1], 1),
                        self.sf_vec_size,
                    ),
                )
            if cutlass.const_expr(fc2_output_sf is not None):
                _run_extra["gmem_fc2_output_sf"] = fc2_output_sf
            if cutlass.const_expr(self.epilogue.fc2_needs_staging):
                # Both bulk paths stage into sFC2; only the dispatch TMASTG path
                # also needs the TMA atom + local-pool tensor (UBLK peer-writes).
                _run_extra["smem_fc2_tma_buffer"] = sFC2
            if cutlass.const_expr(self.epilogue.fc2_use_tma):
                _run_extra["tma_atom_fc2_output"] = tma_atom_fc2_output
                _run_extra["gmem_fc2_tma_output"] = fc2_tma_output
            if cutlass.const_expr(self.epilogue.fc2_reduce_coalesce):
                _run_extra["smem_fc2_reduce_buffer"] = sRED
            if cutlass.const_expr(self.enable_token_comm):
                # MegaMoE (push model): bridge next's TokenComm accessors + peer mapper into
                # the epilogue's Fc2OutputDest peer-store expectations.
                _epi_comm = _EpilogueCommView(
                    token_src_metadata=self.token_comm.token_src_metadata_tensor(
                        self._mega_device_workspace
                    ),
                    combine_output=mega_pre_reduced_activation,
                    peer_rank_ptr_mapper=mega_peer_rank_ptr_mapper,
                    fc2_output_sf=self.token_comm.fc2_activation_sf_tensor(
                        self._mega_device_workspace
                    ),
                    fc2_done_counter=self.token_comm.fc2_done_counter_tensor(
                        self._mega_device_workspace
                    ),
                    fc2_output_workspace=self.token_comm.fc2_activation_tensor(
                        self._mega_device_workspace
                    ),
                )
                self.epilogue.run(**_run_kwargs, **_run_extra, token_comm_args=_epi_comm)
            elif cutlass.const_expr(token_comm_args is not None):
                self.epilogue.run(**_run_kwargs, **_run_extra, token_comm_args=token_comm_args)
            else:
                self.epilogue.run(**_run_kwargs, **_run_extra)

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.fence_acq_rel_sys()

        # ════════════════════════════════════════════════════════════════════
        # Dispatch warps hook (warps 8-11; MegaMoE-only)
        # ════════════════════════════════════════════════════════════════════
        #
        # ``enable_token_comm=False`` → warps 8-11 don't exist (threads_per_cta
        # = 256), so the guard is const_expr-eliminated in the lean path.
        if cutlass.const_expr(self.enable_token_comm):
            if warp_idx >= self.dispatch_warp_id[0]:
                lane_idx_for_dispatch = cute.arch.lane_idx()
                if cutlass.const_expr(self.token_back_standalone):
                    if warp_idx < self.token_back_warp_id[0]:
                        self.token_comm_hook_dispatch_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                    else:
                        self.token_comm_hook_token_back_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                else:
                    self.token_comm_hook_dispatch_warp_body(
                        token_comm_args,
                        token_comm_storage,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx_for_dispatch,
                        tidx=tidx,
                    )

            # ════════════════════════════════════════════════════════════════════
            # Kernel tail hook (MegaMoE-only; lean base = no-op)
            # ════════════════════════════════════════════════════════════════════
            lane_idx = cute.arch.lane_idx()
            self.token_comm_hook_kernel_tail(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )
