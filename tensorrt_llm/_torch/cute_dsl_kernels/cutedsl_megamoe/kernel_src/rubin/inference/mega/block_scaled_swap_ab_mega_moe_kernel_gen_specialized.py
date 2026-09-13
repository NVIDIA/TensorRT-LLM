# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Generation-phase MegaMoE kernel composition for Rubin.

Specialised for few tokens per rank and large expert parallelism. The narrowing
buys a short communication path: ``GenphaseTokenComm`` is a separate launch that
fires ``griddepcontrol_launch_dependents`` at entry, so this kernel becomes
eligible immediately and streams weights while the payload is still on the wire.

Three consequences of that split shape the kernel.

Readiness is whole-rank. The communication component publishes four counters for
its entire grid rather than one per token tile, so each consumer warp waits once
before its work loop instead of per tile, and the FC12 extension runs with no FC1
ready counter at all.

There are no transfer warps. Nothing here pulls payloads into an expert pool and
nothing pushes results back: FC1 gathers straight from the dense received plane,
and the FC2 epilogue warps send results to their owning rank themselves through
the bulk reduce-add. Twelve warps cover every role.

The kernel owns the workspace tail reset. Because it outlives the communication
kernel, it is the only participant that can safely zero the counters, which it
does between two NVLink barriers so that no rank observes a half-cleared
workspace or exits before the clearing is visible.
"""

import math
from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .....api import ImplDesc, KernelClass, ProblemDesc, StaticOrRuntimeIntegerType
from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferDevice
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    make_tcgen05_tmem_plan,
    tcgen05_block_scaled_acc_dtype,
)
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dsl_helpers import spin_wait
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.software_sync import NvlinkBarrier
from .....helpers.utils import ceil_div, round_up
from .....quant_def import CombineFormat, QuantKind
from ....blackwell.inference.mega.topk_reduce import TopkReduce
from ....schedulers.base import WorkIdAcquisitionMode
from ....schedulers.fc12_scheduler import BlackwellFusedFc12Scheduler, PhaseInterleavedFc12Scheduler
from .block_scaled_swap_ab_fc12_epilogue import GatedActEpilogueArgs, SwapABGatedActEpilogue
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .block_scaled_swap_ab_fc12_mainloop_gen_specialized import (
    BlockScaledSwapAbFc12MainloopGenSpecialized,
)
from .token_comm_gen_specialized import GenphaseTokenComm, GenphaseTokenCommArgs

_aot_symbol_prefix = "rubin_genphase_mega_moe_aot"


class BlockScaledSwapAbGenphaseMoeKernel(KernelClass):
    """Compose generation-phase dispatch, persistent FC12, and the return path."""

    fc1_output_region: ClassVar[str] = "rubin.genphase_moe.fc1_output"
    fc1_output_sf_region: ClassVar[str] = "rubin.genphase_moe.fc1_output_sf"
    fc1_done_counter_region: ClassVar[str] = "rubin.genphase_moe.fc1_done_counter"
    pre_reduced_activation_region: ClassVar[str] = "rubin.genphase_moe.pre_reduced_activation"

    epilogue_warp_ids: ClassVar[Tuple[int, int, int, int]] = (0, 1, 2, 3)
    mma_warp_id: ClassVar[int] = 4
    tma_a_warp_id: ClassVar[int] = 5
    tma_b_warp_id: ClassVar[int] = 6
    scheduler_warp_id: ClassVar[int] = 7
    gather_b_warp_ids: ClassVar[Tuple[int, int, int, int]] = (8, 9, 10, 11)
    scheduler_consumer_thread_count: ClassVar[int] = 11 * 32
    epilogue_register_count: ClassVar[int] = 256
    other_warp_register_count: ClassVar[int] = 80
    tail_barrier_id: ClassVar[int] = 8

    # Tuned for Rubin, not portable. VR200 carries 212 SMs and this kernel's 46
    # four-CTA clusters occupy 184, leaving 28 for the communication grid; go over that
    # and the main kernel loses its last cluster, which is not optional because the
    # tail barrier needs every CTA resident at once. The communication exchange CTAs
    # exit as soon as their count rows are published, so the resident set is one helper
    # plus these pushers. Porting to Blackwell means deriving this number again from
    # that part's SM count and cluster capacity.
    pusher_cta_count: ClassVar[int] = 27
    refine_participant_groups: ClassVar[int] = 2
    refine_output_stages: ClassVar[int] = 4

    cluster_shape_mn: ClassVar[Tuple[int, int]] = (4, 1)
    mma_k_mode: ClassVar[str] = "2x"
    a_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K
    b_major_mode: ClassVar[OperandMajorMode] = OperandMajorMode.K

    @classmethod
    def problem_desc_require(cls) -> dict[str, object]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            "quant_kind": str,
            "combine_format": CombineFormat,
            "gate_up_clamp": Optional[float],
            "world_size": int,
            "topk": int,
            "topk_index_dtype": type,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "use_2cta_instrs": bool,
            "schedule_policy": tuple,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "fc2_use_bulk": bool,
            "epi_flag_batches": tuple,
            "launch_cluster_count": int,
            "reduce_topk_in_kernel": bool,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        for field_name, fixed_value, supplied_value in (
            ("cluster_shape_mn", self.cluster_shape_mn, impl_desc.get("cluster_shape_mn")),
            ("mma_k_mode", self.mma_k_mode, impl_desc.get("mma_k_mode")),
            ("a_major_mode", self.a_major_mode, problem_desc.get("a_major_mode")),
            ("b_major_mode", self.b_major_mode, problem_desc.get("b_major_mode")),
        ):
            if supplied_value is not None and supplied_value != fixed_value:
                raise ValueError(
                    f"The generation-phase kernel fixes {field_name} at {fixed_value}, got {supplied_value}."
                )

        fallback_cluster_shape_mn = impl_desc.get("fallback_cluster_shape_mn")
        if (
            fallback_cluster_shape_mn is not None
            and fallback_cluster_shape_mn != self.cluster_shape_mn
        ):
            raise ValueError(
                f"The generation-phase kernel launches a single {self.cluster_shape_mn} cluster shape, so "
                f"fallback_cluster_shape_mn must be unset or equal to it, got {fallback_cluster_shape_mn}."
            )

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.combine_format = problem_desc["combine_format"]
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.world_size = problem_desc["world_size"]
        self.topk = problem_desc["topk"]
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.schedule_policy = impl_desc["schedule_policy"]
        if len(self.schedule_policy) != 2:
            raise ValueError("schedule_policy must contain a mode and hint.")
        self.schedule_mode, self.hint = self.schedule_policy
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.fc2_use_bulk = impl_desc["fc2_use_bulk"]
        self.epi_flag_batches = impl_desc["epi_flag_batches"]
        self.launch_cluster_count = impl_desc["launch_cluster_count"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]

        self.occupancy = 1
        self.architecture = "sm_107"
        self.threads_per_cta = 12 * 32
        self.local_expert_count = self.expert_count // self.world_size
        self.cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        self.promised_launchable_sm_count = self.launch_cluster_count * self.cluster_size

        self._validate_geometry()
        self._resolve_schedule_hint()
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.mma_instruction_mnk,
            participates=mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        tmem_plan = make_tcgen05_tmem_plan(mma_instruction, self.architecture, self.mma_tiler_mnk)
        self.cta_tile_m = self.mma_tiler_mnk[0] // mma_cta_count
        self.cta_tile_n = self.mma_tiler_mnk[1]

        token_comm_impl_desc = ImplDesc(
            {
                **impl_desc,
                "pusher_cta_count": self.pusher_cta_count,
                "refine_participant_groups": self.refine_participant_groups,
                "refine_output_stages": self.refine_output_stages,
            }
        )
        self.token_comm = GenphaseTokenComm(problem_desc, token_comm_impl_desc)
        self.max_routed_rows = self.token_comm.worst_case_data_rows
        self._validate_communication_geometry()
        self.tail_barrier = NvlinkBarrier(
            world_size=self.world_size, barrier_id=self.tail_barrier_id
        )

        fc12_problem_desc = ProblemDesc({**problem_desc, "expert_count": self.local_expert_count})
        resolved_impl_desc = ImplDesc(
            {
                **impl_desc,
                # Fixed here, but the scheduler and the epilogue read them from the
                # descriptor, so they have to be forwarded rather than assumed.
                "cluster_shape_mn": self.cluster_shape_mn,
                "mma_k_mode": self.mma_k_mode,
                "hint": self.hint,
                "tmem_plan": tmem_plan,
                "is_swap_ab": True,
                "max_tokens": self.max_routed_rows,
                "num_scheduler_consumer_threads": self.scheduler_consumer_thread_count,
                "num_accumulator_consumer_warps_per_cta": len(self.epilogue_warp_ids),
                "communication_enabled": True,
                # No token-back path exists in this component, so results never
                # take the local route and the FC2 store is always the peer-facing
                # bulk reduce-add.
                "token_back_push_data": False,
                "fc1_epi_flag_batch": self.epi_flag_batches[0],
                "fc2_epi_flag_batch": self.epi_flag_batches[1],
            }
        )
        scheduler_type = (
            BlackwellFusedFc12Scheduler
            if self.schedule_mode == "grouped"
            else PhaseInterleavedFc12Scheduler
        )
        self.scheduler = scheduler_type(fc12_problem_desc, resolved_impl_desc)
        self.epilogue = SwapABGatedActEpilogue(fc12_problem_desc, resolved_impl_desc)
        if self.epilogue.fc1_output_dtype is not self.b_dtype:
            raise ValueError("Epilogue FC1 output dtype must match the Mainloop B dtype.")
        if self.epilogue.fc1_output_sf_dtype is not self.sf_dtype:
            raise ValueError("Epilogue FC1 output scale dtype must match the Mainloop scale dtype.")
        if self.epilogue.sf_vec_size != self.sf_vec_size:
            raise ValueError("Epilogue and Mainloop scale vector sizes must match.")
        if not self.epilogue.fc2_use_ublk:
            raise ValueError(
                "The generation-phase FC2 store must be the bulk path: set fc2_use_bulk=True so the epilogue "
                "selects the peer-facing UBLK store."
            )
        if self.epilogue.token_back_enabled:
            raise ValueError(
                "The generation-phase communication component provides no token-back path."
            )
        self._device_workspace = self._build_device_workspace()
        self._mainloop, self._smem_workspace = self._build_mainloop_and_smem(
            fc12_problem_desc, resolved_impl_desc
        )
        self._topk_reduce = (
            None
            if self.reduce_topk_in_kernel
            else TopkReduce(self.hidden_size, self.topk, self.combine_format)
        )

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def _validate_geometry(self) -> None:
        static_expert_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_expert_dimensions) and not all(static_expert_dimensions):
            raise ValueError(
                "Genphase MegaMoE expert dimensions must be either all static or all runtime."
            )
        if not all(static_expert_dimensions):
            raise NotImplementedError(
                "The Genphase MegaMoE kernel currently requires static expert dimensions."
            )
        if self.expert_count <= 0 or self.intermediate_gateup_size <= 0 or self.hidden_size <= 0:
            raise ValueError("Genphase MegaMoE expert dimensions must be positive.")
        if self.world_size <= 0 or self.expert_count % self.world_size != 0:
            raise ValueError("expert_count must be divisible by a positive world_size.")
        if self.topk <= 0 or self.topk > self.expert_count:
            raise ValueError("topk must be positive and no greater than expert_count.")
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(
                f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}."
            )
        # The FC2 epilogue carries no per-row top-k weight, so folding the weight
        # in at FC1 is the only way an in-kernel reduction can be correct. The
        # alternative costs a (token, topk, hidden) symmetric plane and a second
        # launch -- see the pre-reduced region in _build_device_workspace.
        if self.reduce_topk_in_kernel and not self.apply_topk_at_fc1:
            raise ValueError(
                "In-kernel top-k reduction requires apply_topk_at_fc1=True. Applying the weight at FC2 instead "
                "means reduce_topk_in_kernel=False, which allocates the (token, topk, hidden) staging plane and "
                "adds a standalone TopkReduce launch."
            )
        # A quantized combine format would derive token_back_push_sf, and with it a
        # token-back path this component does not have: no FC2 done counter, no
        # FC2 output scale plane, no transfer warps.
        if self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError(
                "The generation-phase kernel requires a BF16 combine format: a quantized payload would need the "
                "token-back path, which the generation-phase communication component does not provide."
            )
        if self.intermediate_gateup_size % 2 != 0:
            raise ValueError("The SwiGLU intermediate dimension must be even.")
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive.")
        if self.token_padding_block <= 0 or self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a positive multiple of 64.")
        if self.launch_cluster_count <= 0:
            raise ValueError("launch_cluster_count must be positive.")
        if self.schedule_mode not in ("grouped", "phase_interleave"):
            raise ValueError(
                f"schedule_policy mode must be 'grouped' or 'phase_interleave', got {self.schedule_mode!r}."
            )
        if self.hint is not None and (
            isinstance(self.hint, bool) or not isinstance(self.hint, int) or self.hint <= 0
        ):
            raise ValueError(
                f"schedule_policy hint must be a positive Python int, got {self.hint!r}."
            )
        if self.schedule_mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError("phase_interleave currently requires work_id_mode='atomic_counter'.")
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches must contain FC1 and FC2 batch sizes.")
        if any(batch < 1 or batch > len(self.epilogue_warp_ids) for batch in self.epi_flag_batches):
            raise ValueError(
                f"epi_flag_batches values must be in [1, {len(self.epilogue_warp_ids)}]."
            )
        if self.sf_vec_size <= 0:
            raise ValueError("sf_vec_size must be positive.")

        for field_name, dimensions, expected_rank in (
            ("mma_instruction_mnk", self.mma_instruction_mnk, 3),
            ("mma_tiler_mnk", self.mma_tiler_mnk, 3),
        ):
            if len(dimensions) != expected_rank:
                raise ValueError(f"{field_name} must contain {expected_rank} dimensions.")
            if not all(
                isinstance(dimension, int) and not isinstance(dimension, bool)
                for dimension in dimensions
            ):
                raise TypeError(f"{field_name} dimensions must be Python integers.")
            if any(dimension <= 0 for dimension in dimensions):
                raise ValueError(f"{field_name} dimensions must be positive.")

        mma_m, mma_n, mma_k = self.mma_tiler_mnk
        instruction_m, instruction_n, instruction_k = self.mma_instruction_mnk
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        expected_instruction_m = 256 if self.use_2cta_instrs else 128
        if instruction_m != expected_instruction_m:
            raise ValueError(
                f"{'2CTA' if self.use_2cta_instrs else '1CTA'} Rubin MMA requires "
                f"instruction M={expected_instruction_m}, got {instruction_m}."
            )
        expected_instruction_k = self.quant_kind.instruction_k(self.mma_k_mode)
        if instruction_k != expected_instruction_k:
            raise ValueError(
                f"{self.quant_kind} requires Rubin 2x instruction K={expected_instruction_k}, got {instruction_k}."
            )
        if (instruction_m, instruction_n) != (mma_m, mma_n):
            raise NotImplementedError(
                "Rubin Genphase MegaMoE does not implement M/N instruction repetition."
            )
        # Tile N 64 is excluded because it turns on the halved SFB tile index and
        # the SFB TMEM shift, and 512 is excluded because the FC1 gather needs one
        # B row per 128-byte swizzle atom (see the mainloop's gather geometry).
        if mma_n not in (128, 256):
            raise ValueError(f"mma_tiler N must be 128 or 256, got {mma_n}.")
        if mma_k % instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if mma_k % (self.sf_vec_size * 4) != 0:
            raise ValueError("mma_tiler K must be divisible by four scale-factor vectors.")
        if self.intermediate_gateup_size % (self.sf_vec_size * 4) != 0:
            raise ValueError(
                "The intermediate dimension must be divisible by four scale-factor vectors."
            )
        if mma_m // mma_cta_count != 128:
            raise ValueError(
                f"The epilogue fixes the per-CTA output tile at 128, got {mma_m // mma_cta_count}."
            )
        # The epilogue picks the TMA owner warp with `warp_idx == subtile_idx` and
        # does not check the bound itself, so a wider token tile would drop stores.
        subtile_count = mma_n // 64
        if subtile_count > len(self.epilogue_warp_ids):
            raise ValueError(
                f"mma_tiler N {mma_n} needs {subtile_count} epilogue subtiles, more than the "
                f"{len(self.epilogue_warp_ids)} epilogue warps that own them."
            )

    def _validate_communication_geometry(self) -> None:
        """Check where the FC12 tiling has to agree with the communication pools.

        A gather row index is read from GMEM before it reaches the descriptor, so
        unlike a coordinate it is not bounds-checked: the read itself has to stay
        inside the index array. A gather window starts at a tile base inside the
        padded data pool and spans a whole token tile, so its last row can sit
        ``cta_tile_n - 1`` past the final expert, and the index array carries
        exactly that much readable slack for exactly this reason.
        """
        if self.cta_tile_n > self.token_comm.gather_index_tail_slack:
            raise ValueError(
                f"A token tile of {self.cta_tile_n} rows can read up to {self.cta_tile_n - 1} rows past the last "
                f"expert, exceeding the {self.token_comm.gather_index_tail_slack} rows of readable slack the gather "
                f"index array carries."
            )

    def _resolve_schedule_hint(self) -> None:
        if self.schedule_mode != "phase_interleave":
            return

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        cluster_feature_tile = self.mma_tiler_mnk[0] // mma_cta_count * self.cluster_shape_mn[0]
        blocks_fc1 = ceil_div(self.intermediate_gateup_size, cluster_feature_tile)
        blocks_fc2 = ceil_div(self.hidden_size, cluster_feature_tile)
        # Cover the worst token-block alignment of one full persistent-cluster FC2 claim wave.
        max_dependent_token_blocks = ceil_div(
            self.launch_cluster_count + blocks_fc2 - 1, blocks_fc2
        )
        required_fc1_work = max_dependent_token_blocks * blocks_fc1
        raw_minimum_hint = max(1, ceil_div(required_fc1_work, self.launch_cluster_count))
        minimum_safe_hint = round_up(raw_minimum_hint, self.epi_flag_batches[0])
        if self.hint is None:
            self.hint = minimum_safe_hint
        elif self.hint < minimum_safe_hint:
            raise ValueError(
                f"phase_interleave hint {self.hint} is unsafe with "
                f"fc1_epi_flag_batch={self.epi_flag_batches[0]}; minimum legal hint is {minimum_safe_hint}."
            )

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def _build_device_workspace(self) -> DeviceWorkspace:
        intermediate_downproj = self.intermediate_gateup_size // 2
        sf_column_count = round_up(ceil_div(intermediate_downproj, self.sf_vec_size), 4)
        max_sf_rows = self.token_comm.worst_case_sf_rows
        # One slot per token block across every local expert, which is the space
        # `cumulative_token_block_count + tile_n_idx` indexes.
        counter_slot_count = (
            self.token_comm.worst_case_padded_rows(self.cta_tile_n) // self.cta_tile_n
        )

        device_workspace = DeviceWorkspace()
        device_workspace.register(
            self.fc1_output_region,
            self.epilogue.fc1_output_dtype,
            (self.max_routed_rows, intermediate_downproj),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_output_sf_region,
            self.epilogue.fc1_output_sf_dtype,
            (max_sf_rows, sf_column_count),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_done_counter_region,
            cutlass.Int32,
            (counter_slot_count,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )
        if not self.reduce_topk_in_kernel:
            # Peers write their contributions straight into this plane, one row per
            # (token, top-k slot), so it has to be symmetric. The standalone
            # TopkReduce launch folds it into the caller's output afterwards.
            device_workspace.register(
                self.pre_reduced_activation_region,
                self.combine_format.act_dtype,
                (self.max_tokens_per_rank, self.topk, self.hidden_size),
                buffer_space="shared",
                mem_order=(2, 1, 0),
                byte_alignment=128,
            )
        self.scheduler.register_device_workspace(device_workspace)
        self.tail_barrier.register_device_workspace(device_workspace)
        self.token_comm.register_device_workspace(device_workspace)
        device_workspace.finalize()
        return device_workspace

    def _build_mainloop_and_smem(
        self, problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc
    ) -> Tuple[BlockScaledSwapAbFc12MainloopGenSpecialized, SmemWorkspace]:
        # The communication component builds its own SMEM workspace for its own
        # launch, so nothing of it is charged against this kernel's budget.
        smem_limit = cutlass.memory.get_smem_capacity_in_bytes(self.architecture) // self.occupancy
        smem_workspace = SmemWorkspace()
        self.scheduler.register_smem_regions(smem_workspace)
        self.epilogue.register_smem_regions(smem_workspace)
        fixed_component_bytes = smem_workspace.estimate_total_bytes()
        alignment_shift_bytes = max(region.byte_alignment for region in smem_workspace.regions())
        mainloop_smem_budget_bytes = smem_limit - fixed_component_bytes - alignment_shift_bytes
        if mainloop_smem_budget_bytes <= 0:
            raise ValueError("Fixed kernel components leave no SMEM budget for the Mainloop.")

        mainloop_impl_desc = ImplDesc(
            {**resolved_impl_desc, "mainloop_smem_budget_bytes": mainloop_smem_budget_bytes}
        )
        mainloop = BlockScaledSwapAbFc12MainloopGenSpecialized(problem_desc, mainloop_impl_desc)

        mainloop.register_smem_regions(smem_workspace)
        smem_workspace.finalize(max_bytes=smem_limit)
        return mainloop, smem_workspace

    def get_workspace_sizes(self) -> Tuple[int, int]:
        return self._device_workspace.local_and_shared_bytes

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        return self._device_workspace.require_zero_workspace_leading_bytes

    def name(self) -> str:
        """Canonical encoding of every constexpr: the compiled-kernel cache key.

        Two candidates that differ in any construction knob must produce different names, otherwise a tuning sweep
        silently collapses to whichever one compiled first.
        """

        def dtype_name(dtype: type) -> str:
            return getattr(dtype, "__name__", str(dtype)).lower()

        instruction = "x".join(str(dimension) for dimension in self.mma_instruction_mnk)
        tile = "x".join(str(dimension) for dimension in self.mma_tiler_mnk)
        cluster = "x".join(str(dimension) for dimension in self.cluster_shape_mn)
        epi_flags = "x".join(str(batch) for batch in self.epi_flag_batches)
        return (
            f"sm107_block_scaled_swap_ab_gen_phase_specialized_moe_{self.quant_kind}_"
            f"{dtype_name(self.a_dtype)}_{dtype_name(self.b_dtype)}_{dtype_name(self.acc_dtype)}_"
            f"sfvec{self.sf_vec_size}_a{self.a_major_mode.name.lower()}_b{self.b_major_mode.name.lower()}_"
            f"w{self.world_size}_e{self.expert_count}_topk{self.topk}_"
            f"topkidx{dtype_name(self.topk_index_dtype)}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_maxtoken{self.max_tokens_per_rank}_"
            f"inst{instruction}_{self.mma_k_mode}_tile{tile}_"
            f"cluster{cluster}_{'2cta' if self.use_2cta_instrs else '1cta'}_"
            f"sched{self.schedule_mode}_hint{self.hint}_pad{self.token_padding_block}x{self.sf_padding_block}_"
            f"work{self.work_id_mode}_"
            f"fc2store{'ublk' if self.epilogue.fc2_use_ublk else 'stg'}_"
            f"fc2tmastages{self.epilogue.fc2_tma_stages}_"
            f"epiflag{epi_flags}_clusters{self.launch_cluster_count}_"
            f"pusher{self.pusher_cta_count}_refine{self.refine_participant_groups}x{self.refine_output_stages}_"
            f"combine{self.combine_format.name}_clamp{self.gate_up_clamp}_"
            f"{'apply_topk_fc1' if self.apply_topk_at_fc1 else 'apply_topk_fc2'}_"
            f"{'inkernel_reduce' if self.reduce_topk_in_kernel else 'separate_reduce'}_"
            f"abstages{self._mainloop.num_weight_ab_stages}x{self._mainloop.num_token_ab_stages}"
        )

    def aot_compile(self, out_path: Optional[str] = None, **_compile_kwargs):
        """Compile against fake (metadata-only) inputs; ``out_path=None`` returns the in-memory callable.

        The fake tensors define the runtime ABI, so each mirrors what the caller stages: the local expert tensors,
        the padded payload strides the communication component advertises, and ``to_cute``'s
        ``mark_layout_dynamic``. Occupancy is a construction-time knob (``launch_cluster_count``), so this entry
        accepts no occupancy argument and tolerates leftover caller kwargs.
        """
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
        from cutlass.cute.typing import AddressSpace, sym_int64

        from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferHost

        def fake_tensor(dtype, shape, stride_order, dynamic_axes, alignment):
            extents = tuple(
                sym_int64(divisibility=math.gcd(int(extent), 128))
                if axis in dynamic_axes
                else int(extent)
                for axis, extent in enumerate(shape)
            )
            return make_fake_compact_tensor(
                dtype, extents, stride_order=stride_order, assumed_align=alignment
            )

        tokens = self.max_tokens_per_rank
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        experts = self.local_expert_count
        sf_vec_size = self.sf_vec_size
        # Weight SF is atom-swizzled and opaque, so its extent is whatever the caller staged.
        fc1_weight_sf_columns = round_up(intermediate_gateup, 128) * round_up(
            hidden // sf_vec_size, 4
        )
        fc2_weight_sf_columns = round_up(hidden, 128) * round_up(
            intermediate_downproj // sf_vec_size, 4
        )

        fake_arguments = dict(
            activation=fake_tensor(
                self.token_comm.activation_dtype,
                (tokens, self.token_comm.plain_activation_row_elements),
                (1, 0),
                {},
                16,
            ),
            activation_sf=fake_tensor(
                self.token_comm.activation_sf_dtype,
                (tokens, self.token_comm.plain_activation_sf_row_elements),
                (1, 0),
                {},
                16,
            ),
            topk_indices=fake_tensor(self.topk_index_dtype, (tokens, self.topk), (1, 0), {}, 16),
            topk_scores=fake_tensor(cutlass.Float32, (tokens, self.topk), (1, 0), {}, 4),
            fc1_weight=fake_tensor(
                self.a_dtype, (experts, hidden, intermediate_gateup), (2, 0, 1), {1, 2}, 16
            ),
            fc1_weight_sf=fake_tensor(
                self.sf_dtype, (experts, fc1_weight_sf_columns), (1, 0), {1}, 16
            ),
            fc2_weight=fake_tensor(
                self.a_dtype, (experts, intermediate_downproj, hidden), (2, 0, 1), {1, 2}, 16
            ),
            fc2_weight_sf=fake_tensor(
                self.sf_dtype, (experts, fc2_weight_sf_columns), (1, 0), {1}, 16
            ),
            output_activation=fake_tensor(cutlass.BFloat16, (tokens, hidden), (1, 0), {}, 16),
            local_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            shared_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            # Placeholder field values: they marshal as runtime scalars, so their widths
            # are part of the ABI and bare Python ints would narrow the offsets to i32.
            # Only max_ranks is constexpr.
            peer_rank_ptr_mapper_host=SymmetricBufferHost(
                base_address=Int64(0),
                offsets=tuple(Int64(0) for _ in range(self.world_size)),
                rank=Int32(0),
                max_ranks=self.world_size,
            ),
            stream=make_fake_stream(),
        )
        if self.quant_kind.uses_global_scale:
            fake_arguments.update(
                fc1_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc2_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc1_norm_const=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
            )

        compiled = cute.compile[cute.EnableTVMFFI(True)](self, **fake_arguments)
        if out_path is None:
            return compiled
        compiled.export_to_c(
            out_path, function_name=_aot_symbol_prefix, export_only_tvm_ffi_symbols=True
        )
        return out_path

    @staticmethod
    def load_compiled(path: str):
        from cutlass.cute.runtime import load_module

        return load_module(path, enable_tvm_ffi=True)[_aot_symbol_prefix]

    # ------------------------------------------------------------------
    # Host entry
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,  # (max_tokens_per_rank, padded hidden), padded row stride
        activation_sf: cute.Tensor,  # (max_tokens_per_rank, padded hidden / sf_vec_size)
        topk_indices: cute.Tensor,  # (max_tokens_per_rank, topk)
        topk_scores: cute.Tensor,  # (max_tokens_per_rank, topk)
        fc1_weight: cute.Tensor,  # (local_experts, hidden, intermediate_gateup)
        fc1_weight_sf: cute.Tensor,  # (local_experts, packed_fc1_scale_factors)
        fc2_weight: cute.Tensor,  # (local_experts, intermediate_downproj, hidden)
        fc2_weight_sf: cute.Tensor,  # (local_experts, packed_fc2_scale_factors)
        output_activation: cute.Tensor,  # (max_tokens_per_rank, hidden)
        local_workspace: cute.Pointer,  # local GMEM byte workspace
        shared_workspace: cute.Pointer,  # symmetric GMEM byte workspace
        peer_rank_ptr_mapper_host,
        stream: cuda.CUstream,
        fc1_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc2_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc1_norm_const: Optional[cute.Tensor] = None,  # (local_experts,)
    ) -> None:
        """Launch generation-phase dispatch, fused FC12, and the optional top-k reduce."""

        def rewrite_tensor_shape(tensor: cute.Tensor, shape: Tuple) -> cute.Tensor:
            return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=tensor.stride))

        if cutlass.const_expr(topk_indices.element_type is not self.topk_index_dtype):
            raise TypeError(
                f"topk_indices must be the declared {self.topk_index_dtype}, got {topk_indices.dtype}."
            )
        for operand_name, operand, expected_dtype in (
            ("activation", activation, self.b_dtype),
            ("activation_sf", activation_sf, self.sf_dtype),
            ("fc1_weight", fc1_weight, self.a_dtype),
            ("fc1_weight_sf", fc1_weight_sf, self.sf_dtype),
            ("fc2_weight", fc2_weight, self.a_dtype),
            ("fc2_weight_sf", fc2_weight_sf, self.sf_dtype),
        ):
            if cutlass.const_expr(operand.element_type is not expected_dtype):
                raise TypeError(
                    f"{self.quant_kind} requires {operand_name} to be {expected_dtype}, got {operand.element_type}."
                )
        if cutlass.const_expr(self.quant_kind.uses_global_scale):
            for scalar_name, scalar in (
                ("fc1_alpha", fc1_alpha),
                ("fc2_alpha", fc2_alpha),
                ("fc1_norm_const", fc1_norm_const),
            ):
                if cutlass.const_expr(scalar is None):
                    raise ValueError(f"{self.quant_kind} requires {scalar_name}.")

        local_rank = peer_rank_ptr_mapper_host.rank
        router_topk_scores = topk_scores if cutlass.const_expr(self.apply_topk_at_fc1) else None
        self.token_comm.launch(
            GenphaseTokenCommArgs(
                topk_indices=topk_indices,
                topk_scores=router_topk_scores,
                activation=activation,
                activation_sf=activation_sf,
            ),
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host,
            self._device_workspace,
            stream,
        )

        self._device_workspace.assign_device_members(local_workspace, shared_workspace)
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_object()
        dense_activation = self.token_comm.dense_activation_tensor(self._device_workspace)
        canonical_activation_sf = self.token_comm.canonical_activation_sf_tensor(
            self._device_workspace
        )
        scheduler_expert_sizes = self.token_comm.scheduler_expert_sizes_tensor(
            self._device_workspace
        )
        fc1_output = self._device_workspace.tensor(self.fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.fc1_output_sf_region)

        experts = self.local_expert_count
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        fc1_weight = rewrite_tensor_shape(fc1_weight, (experts, hidden, intermediate_gateup))
        fc2_weight = rewrite_tensor_shape(fc2_weight, (experts, intermediate_downproj, hidden))

        singleton = cutlass.Int32(1)
        sf_vec_size = self.sf_vec_size

        fc1_a = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (cutlass.Int32(intermediate_gateup), cutlass.Int32(hidden), experts),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
            ),
        )
        # FC1's B is the dense (source rank, source token) plane the communication
        # kernel filled, indexed by `gather_index` rather than by a pool row, so it
        # carries no per-expert offset.
        fc1_b = cute.make_tensor(
            dense_activation.iterator,
            cute.make_layout(
                (dense_activation.shape[0], cutlass.Int32(hidden), singleton),
                stride=(dense_activation.stride[0], dense_activation.stride[1], 0),
            ),
        )
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (self.max_routed_rows, cutlass.Int32(intermediate_downproj), singleton),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )
        # The canonical scale-factor pool already carries the MMA's atom layout, so
        # FC1's SFB leg is an ordinary tiled TMA over it.
        fc1_sfb = cute.make_tensor(
            canonical_activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (self.token_comm.worst_case_sf_rows, cutlass.Int32(hidden), singleton), sf_vec_size
            ),
        )
        padded_intermediate_gateup = cute.round_up(intermediate_gateup, sf_vec_size * 4)
        expected_fc1_weight_sf_columns = padded_intermediate_gateup * hidden // sf_vec_size
        if cutlass.const_expr(
            isinstance(fc1_weight_sf.shape[1], int)
            and isinstance(expected_fc1_weight_sf_columns, int)
        ):
            if cutlass.const_expr(fc1_weight_sf.shape[1] != expected_fc1_weight_sf_columns):
                raise ValueError("fc1_weight_sf has an incompatible column count.")
        fc1_sfa = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    cutlass.Int32(padded_intermediate_gateup),
                    cutlass.Int32(hidden),
                    cutlass.Int32(experts),
                ),
                sf_vec_size,
            ),
        )

        experts_fc2, intermediate_fc2, hidden_fc2 = fc2_weight.shape
        fc2_a = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (cutlass.Int32(hidden_fc2), cutlass.Int32(intermediate_fc2), experts_fc2),
                stride=(fc2_weight.stride[2], fc2_weight.stride[1], fc2_weight.stride[0]),
            ),
        )
        fc2_b = fc1_output_gemm
        padded_fc2_hidden = cute.round_up(hidden_fc2, 128)
        padded_fc2_intermediate = cute.round_up(intermediate_fc2, sf_vec_size * 4)
        expected_fc2_weight_sf_columns = padded_fc2_hidden * padded_fc2_intermediate // sf_vec_size
        if cutlass.const_expr(
            isinstance(fc2_weight_sf.shape[1], int)
            and isinstance(expected_fc2_weight_sf_columns, int)
        ):
            if cutlass.const_expr(fc2_weight_sf.shape[1] != expected_fc2_weight_sf_columns):
                raise ValueError("fc2_weight_sf has an incompatible column count.")
        fc2_sfa = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    cutlass.Int32(padded_fc2_hidden),
                    cutlass.Int32(padded_fc2_intermediate),
                    cutlass.Int32(experts_fc2),
                ),
                sf_vec_size,
            ),
        )
        fc2_sfb = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    fc1_output_sf.shape[0],
                    cutlass.Int32(fc1_output_sf.shape[1] * sf_vec_size),
                    singleton,
                ),
                sf_vec_size,
            ),
        )

        if cutlass.const_expr(self.reduce_topk_in_kernel):
            # Every top-k contribution reduces into the topk=0 plane, so the caller's
            # output doubles as the (token, topk, hidden) destination. It has to live
            # in the symmetric heap: peers reach it through the per-rank offset.
            pre_reduced_activation = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (self.max_tokens_per_rank, 1, hidden),
                    stride=(
                        output_activation.stride[0],
                        output_activation.stride[0],
                        output_activation.stride[1],
                    ),
                ),
            )
        else:
            pre_reduced_activation = self._device_workspace.tensor(
                self.pre_reduced_activation_region
            )
        fc2_output = rewrite_tensor_shape(
            pre_reduced_activation,
            (pre_reduced_activation.shape[0], pre_reduced_activation.shape[1], hidden),
        )

        self._mainloop.materialize_codegen_members()
        (
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_atom,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom,
            fc2_tma_b_tensor,
            fc2_tma_b_atom,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom,
        ) = self._mainloop.prepare_tma_load_params(
            fc1_a=fc1_a,
            fc1_b=fc1_b,
            fc1_sfa=fc1_sfa,
            fc1_sfb=fc1_sfb,
            fc2_a=fc2_a,
            fc2_b=fc2_b,
            fc2_sfa=fc2_sfa,
            fc2_sfb=fc2_sfb,
        )

        # FC2 stores through the peer-facing bulk path, so only the FC1 store atom
        # comes back populated.
        (fc1_output_tma_atom, fc1_output_tma_tensor, _, _) = self.epilogue.prepare_tma_store_params(
            fc1_output_gemm, fc2_output
        )

        grid = self.scheduler.get_grid_shape(max_active_clusters=self.launch_cluster_count)
        self._device_workspace.remove_device_members()
        self._kernel(
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_atom,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom,
            fc2_tma_b_tensor,
            fc2_tma_b_atom,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom,
            fc1_output_tma_atom,
            fc1_output_tma_tensor,
            fc2_output,
            (experts, intermediate_gateup, hidden),
            scheduler_expert_sizes,
            peer_rank_ptr_mapper,
            local_workspace,
            shared_workspace,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
            use_pdl=True,
        )
        if cutlass.const_expr(not self.reduce_topk_in_kernel):
            reduce_scores = None if cutlass.const_expr(self.apply_topk_at_fc1) else topk_scores
            self._topk_reduce(
                pre_reduced_activation, None, output_activation, reduce_scores, stream
            )

    # ------------------------------------------------------------------
    # Device entry
    # ------------------------------------------------------------------

    @cute.kernel
    def _kernel(
        self,
        fc1_tma_a_tensor: cute.Tensor,
        fc1_tma_a_atom: cute.CopyAtom,
        fc1_tma_sfa_tensor: cute.Tensor,
        fc1_tma_sfa_atom: cute.CopyAtom,
        fc2_tma_a_tensor: cute.Tensor,
        fc2_tma_a_atom: cute.CopyAtom,
        fc2_tma_sfa_tensor: cute.Tensor,
        fc2_tma_sfa_atom: cute.CopyAtom,
        fc1_tma_b_atom: cute.CopyAtom,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom: cute.CopyAtom,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom: cute.CopyAtom,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom: cute.CopyAtom,
        fc1_output_tma_atom: cute.CopyAtom,
        fc1_output_tma_tensor: cute.Tensor,
        fc2_output: cute.Tensor,
        actual_expert_shape: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        # Per-expert sizes in the order the scheduler walks them, which is the
        # slot order rather than the expert order.
        scheduler_expert_sizes: cute.Tensor,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
    ):
        """Compose scheduler, mainloop, epilogue, and the workspace tail reset."""
        self._mainloop.materialize_codegen_members()
        storage_type = self._smem_workspace.storage_class()
        smem_allocator = utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()
        self._device_workspace.assign_device_members(local_workspace, shared_workspace)

        fc1_done_counter = self._device_workspace.tensor(self.fc1_done_counter_region)
        fc1_output_sf_storage = self._device_workspace.tensor(self.fc1_output_sf_region)
        if cutlass.const_expr(isinstance(self.hidden_size, int)):
            hidden = self.hidden_size
        else:
            hidden = actual_expert_shape[2]
        if cutlass.const_expr(isinstance(self.intermediate_gateup_size, int)):
            intermediate_gateup = self.intermediate_gateup_size
        else:
            intermediate_gateup = actual_expert_shape[1]
        fc1_output_sf = cute.make_tensor(
            fc1_output_sf_storage.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (fc1_output_sf_storage.shape[0], intermediate_gateup // 2, cutlass.Int32(1)),
                self.epilogue.sf_vec_size,
            ),
        )

        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // 32)
        block_idx = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_coord_in_cluster = (
            cta_rank_in_cluster % self.cluster_shape_mn[0],
            cta_rank_in_cluster // self.cluster_shape_mn[0],
            cutlass.Int32(0),
        )
        linear_cta_idx = cta_rank_in_cluster + block_idx[2] * Int32(self.cluster_size)

        token_src_metadata = self.token_comm.token_src_metadata_tensor(self._device_workspace)
        gather_index = self.token_comm.gather_index_tensor(self._device_workspace)
        pool_topk_scores = self.token_comm.fc1_topk_scores_tensor(self._device_workspace)
        self.tail_barrier.assign_device_members(self._device_workspace, peer_rank_ptr_mapper)

        # Under the split-depth plan the weights and the tokens ride separate
        # pipelines of different depth; under the shared plan `token_pipeline` is
        # the same object, so every role below reads the same as it did before.
        if cutlass.const_expr(self._mainloop.uses_asymmetric_ab_stages):
            ab_pipeline = self._mainloop.create_weight_pipeline(self._smem_workspace, smem_base)
            token_pipeline = self._mainloop.create_token_pipeline(self._smem_workspace, smem_base)
        else:
            ab_pipeline = self._mainloop.create_ab_pipeline(self._smem_workspace, smem_base)
            token_pipeline = ab_pipeline
        tma_a_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_weight_ab_stages
        )
        tma_b_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_token_ab_stages
        )
        gather_b_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_token_ab_stages
        )
        mma_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._mainloop.num_weight_ab_stages
        )
        mma_token_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._mainloop.num_token_ab_stages
        )
        acc_pipeline = self._mainloop.create_acc_pipeline(self._smem_workspace, smem_base)
        tmem_allocator = self._mainloop.create_tmem_allocator(
            self._smem_workspace, smem_base, allocator_warp_id=self.epilogue_warp_ids[0]
        )
        self.scheduler.assign_device_members(
            expert_token_sizes=scheduler_expert_sizes,
            expert_token_prefix_sum=None,
            actual_expert_shape=actual_expert_shape,
            block_idx=block_idx,
            smem_workspace=self._smem_workspace,
            smem_base=smem_base,
            device_workspace=self._device_workspace,
        )
        scheduler = self.scheduler

        fc2_spin_threshold = (
            intermediate_gateup + self._mainloop.cta_tile_m - 1
        ) // self._mainloop.cta_tile_m
        # The scheduler enumerates slots, so the tile it produces has to be
        # rewritten into an expert index before any consumer sees it: the weights,
        # their scale factors, the per-expert alphas and the per-expert
        # scale-factor gate all address an expert. The row offsets it also carries
        # are already resolved and stay as they are.
        expert_remap = self.token_comm.slot_to_expert_tensor(self._device_workspace)
        # No FC1 ready counter: readiness is published per rank, not per tile, and
        # the waits below cover it once per warp.
        kernel_extension = BlockScaledSwapAbFc12Extension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=None,
            expert_remap=expert_remap,
        )
        optional_epilogue_args = GatedActEpilogueArgs(
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            topk_scores=pool_topk_scores,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        self._mainloop.assign_device_members(
            self._smem_workspace, smem_base, cta_coord_in_cluster, hidden, intermediate_gateup
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.scheduler_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.scheduler")
            iket.range_push("scheduler.wait_sizes_ready")
            # One helper CTA folds the per-rank counts into `expert_sizes` and
            # publishes once, so the target is one rather than a derived count.
            spin_wait(
                self._device_workspace.ptr(self.token_comm.sizes_ready_region),
                lambda value: value == Int32(1),
                scope="gpu",
            )
            iket.range_pop()
            iket.range_push("scheduler.gen_work")
            work_tile = scheduler.gen_next_work()
            iket.range_pop()
            while work_tile.is_valid_tile:
                iket.range_push("scheduler.publish_work")
                scheduler.publish_work(kernel_extension.prepare_work_tile(work_tile))
                iket.range_pop()
                iket.range_push("scheduler.gen_work")
                work_tile = scheduler.gen_next_work()
                iket.range_pop()
            iket.range_push("scheduler.publish_tail")
            scheduler.publish_work(work_tile)
            scheduler.produce_tail()
            iket.range_pop()
            iket.range_pop()

        if warp_idx == self.tma_a_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.tma_a")
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_a(
                fc1_tma_a_tensor=fc1_tma_a_tensor,
                fc1_tma_a_atom=fc1_tma_a_atom,
                fc1_tma_sfa_tensor=fc1_tma_sfa_tensor,
                fc1_tma_sfa_atom=fc1_tma_sfa_atom,
                fc2_tma_a_tensor=fc2_tma_a_tensor,
                fc2_tma_a_atom=fc2_tma_a_atom,
                fc2_tma_sfa_tensor=fc2_tma_sfa_tensor,
                fc2_tma_sfa_atom=fc2_tma_sfa_atom,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=tma_a_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
            )
            iket.range_pop()

        if warp_idx == self.tma_b_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.tma_b")
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_b(
                fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
                fc1_tma_sfb_atom=fc1_tma_sfb_atom,
                fc2_tma_b_tensor=fc2_tma_b_tensor,
                fc2_tma_b_atom=fc2_tma_b_atom,
                fc2_tma_sfb_tensor=fc2_tma_sfb_tensor,
                fc2_tma_sfb_atom=fc2_tma_sfb_atom,
                ab_pipeline=token_pipeline,
                ab_pipeline_state=tma_b_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                fc1_done_counter_pointer=fc1_done_counter.iterator,
                fc2_spin_threshold=fc2_spin_threshold,
                fc1_sf_ready_pointer=self._device_workspace.ptr(
                    self.token_comm.fc1_sf_ready_region
                ),
                expert_sizes=self._device_workspace.tensor(self.token_comm.expert_sizes_region),
            )
            iket.range_pop()

        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.mma")
            sched_consumer = scheduler.make_consumer()
            if cutlass.const_expr(self._mainloop.uses_asymmetric_ab_stages):
                self._mainloop.run_mma_asymmetric(
                    tmem_allocator=tmem_allocator,
                    weight_pipeline=ab_pipeline,
                    weight_pipeline_state=mma_pipeline_state,
                    token_pipeline=token_pipeline,
                    token_pipeline_state=mma_token_pipeline_state,
                    acc_pipeline=acc_pipeline,
                    sched_consumer=sched_consumer,
                )
            else:
                self._mainloop.run_mma(
                    tmem_allocator=tmem_allocator,
                    ab_pipeline=ab_pipeline,
                    ab_pipeline_state=mma_pipeline_state,
                    acc_pipeline=acc_pipeline,
                    sched_consumer=sched_consumer,
                )
            iket.range_pop()

        if warp_idx < len(self.epilogue_warp_ids):
            cute.arch.warpgroup_reg_alloc(self.epilogue_register_count)
            iket.range_push("genphase.epilogue")
            iket.range_push("epilogue.wait_metadata_ready")
            # FC1 reads the pool-aligned top-k scores and FC2 reads the routing
            # records; both land with the same counter.
            spin_wait(
                self._device_workspace.ptr(self.token_comm.metadata_ready_region),
                lambda value: value == Int32(self.token_comm.metadata_ready_target),
                scope="gpu",
            )
            iket.range_pop()
            sched_consumer = scheduler.make_consumer()
            tmem_allocator.allocate(self._mainloop.num_tmem_alloc_cols)
            tmem_allocator.wait_for_alloc()
            tmem_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
            self.epilogue.run(
                self._smem_workspace,
                smem_base,
                tmem_pointer,
                acc_pipeline,
                sched_consumer,
                kernel_extension,
                fc1_output_tma_atom,
                fc1_output_tma_tensor,
                fc1_output_sf,
                None,
                None,
                fc2_output,
                fc1_done_counter,
                thread_idx,
                token_src_metadata,
                None,
                None,
                peer_rank_ptr_mapper,
                optional_epilogue_args,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(
                tmem_allocator.retrieve_ptr(self.acc_dtype), self._mainloop.num_tmem_alloc_cols
            )

            epilogue_thread_count = 32 * len(self.epilogue_warp_ids)

            iket.range_push("tail.grid_sync_before_reset")
            self.tail_barrier.sync(
                epilogue_thread_count,
                Int32(self.promised_launchable_sm_count),
                linear_cta_idx,
                thread_idx,
            )
            iket.range_pop()

            iket.range_push("tail.reset_workspace")
            total_reset_threads = self.promised_launchable_sm_count * epilogue_thread_count
            global_reset_thread = linear_cta_idx * Int32(epilogue_thread_count) + thread_idx
            self._device_workspace.reset_tail_space(
                "shared", global_reset_thread, total_reset_threads
            )
            self._device_workspace.reset_tail_space(
                "local", global_reset_thread, total_reset_threads
            )
            iket.range_pop()

            iket.range_push("tail.nvlink_publish")
            self.tail_barrier.arrive_and_wait(
                epilogue_thread_count,
                Int32(self.promised_launchable_sm_count),
                linear_cta_idx,
                thread_idx,
                prologue_grid_sync=True,
                epilogue_grid_sync=False,
            )
            iket.range_pop()
            iket.range_pop()

        if (warp_idx >= Int32(self.gather_b_warp_ids[0])) & (
            warp_idx <= Int32(self.gather_b_warp_ids[-1])
        ):
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.gather_b1")
            iket.range_push("gather_b1.wait_input_ready")
            # `gather_index` is local, published by this rank's sort workers; the
            # payload flag is written by peers, hence the wider scope.
            spin_wait(
                self._device_workspace.ptr(self.token_comm.token_data_ready_region),
                lambda value: value == Int32(self.token_comm.payload_ready_target),
                scope="sys",
            )
            spin_wait(
                self._device_workspace.ptr(self.token_comm.metadata_ready_region),
                lambda value: value == Int32(self.token_comm.metadata_ready_target),
                scope="gpu",
            )
            iket.range_pop()
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_gather_b1(
                fc1_tma_b_atom=fc1_tma_b_atom,
                gather_index=gather_index,
                ab_pipeline=token_pipeline,
                ab_pipeline_state=gather_b_pipeline_state,
                sched_consumer=sched_consumer,
                gather_warp_idx=warp_idx - Int32(self.gather_b_warp_ids[0]),
            )
            iket.range_pop()

        self.tail_barrier.remove_device_members()
        self._device_workspace.remove_device_members()


__all__ = ["BlockScaledSwapAbGenphaseMoeKernel"]
