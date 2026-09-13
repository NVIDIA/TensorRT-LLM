# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused pull-dispatch MegaMoE kernel composition for Rubin."""

from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Boolean, Int32

from .....api import (
    ImplDesc,
    KernelClass,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from .....communication.nvlink_domain.token_comm import TokenCommArgs, TokenCommNonDeterministic
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    make_tcgen05_tmem_plan,
    tcgen05_block_scaled_acc_dtype,
)
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import ceil_div, round_up
from .....quant_def import CombineFormat, QuantKind
from ....blackwell.custom_mix_cga_helpers import (
    TmaAtomOrPair,
    pipeline_init_arrive_mixed_cga,
    pipeline_init_wait_mixed_cga,
)
from ....blackwell.inference.mega.topk_reduce import TopkReduce
from ....schedulers.base import WorkIdAcquisitionMode
from ....schedulers.fc12_scheduler import (
    BlackwellFusedFc12Scheduler,
    PhaseInterleavedFc12Scheduler,
    minimum_phase_interleave_hint,
)
from ....schedulers.non_clc_mixed_cga import NonClcMixedCgaConfig
from .block_scaled_swap_ab_fc12_epilogue import GatedActEpilogueArgs, SwapABGatedActEpilogue
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .block_scaled_swap_ab_fc12_mainloop import BlockScaledSwapAbFc12Mainloop

_aot_symbol_prefix = "rubin_mega_moe_aot"


class BlockScaledSwapAbMegaMoeKernel(KernelClass):
    """Compose pull dispatch, persistent FC12, token-back, and reduction."""

    fc1_output_region: ClassVar[str] = "rubin.mega_moe.fc1_output"
    fc1_output_sf_region: ClassVar[str] = "rubin.mega_moe.fc1_output_sf"
    fc1_done_counter_region: ClassVar[str] = "rubin.mega_moe.fc1_done_counter"
    epilogue_warp_ids: ClassVar[Tuple[int, int, int, int]] = (0, 1, 2, 3)
    mma_warp_id: ClassVar[int] = 4
    tma_a_warp_id: ClassVar[int] = 5
    tma_b_warp_id: ClassVar[int] = 6
    scheduler_warp_id: ClassVar[int] = 7
    transfer_warp_idx_start: ClassVar[int] = 8
    token_in_end_warp_idx: ClassVar[int] = 12
    scheduler_consumer_thread_count: ClassVar[int] = 7 * 32
    epilogue_register_count: ClassVar[int] = 256
    scheduler_register_count: ClassVar[int] = 72

    @classmethod
    def problem_desc_require(cls) -> dict[str, object]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            "quant_kind": str,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
            "combine_format": CombineFormat,
            "gate_up_clamp": Optional[float],
            "situ_beta": OptionalRequirement(Optional[float]),
            "situ_linear_beta": OptionalRequirement(Optional[float]),
            "world_size": int,
            "topk": int,
            "topk_index_dtype": type,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "mma_k_mode": str,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "schedule_policy": tuple,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "fc2_use_bulk": bool,
            "epi_flag_batches": tuple,
            "launch_cluster_count": int,
            "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
            "preferred_cluster_count": OptionalRequirement(Optional[int]),
            "fallback_cluster_count": OptionalRequirement(Optional[int]),
            "token_in_flag_batch": int,
            "token_back_mode": str,
            "reduce_topk_in_kernel": bool,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.a_major_mode = problem_desc["a_major_mode"]
        self.b_major_mode = problem_desc["b_major_mode"]
        self.combine_format = problem_desc["combine_format"]
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.situ_beta = problem_desc.get("situ_beta")
        self.situ_linear_beta = problem_desc.get("situ_linear_beta")
        self.world_size = problem_desc["world_size"]
        self.topk = problem_desc["topk"]
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.mma_k_mode = impl_desc["mma_k_mode"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
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
        self.token_in_flag_batch = impl_desc["token_in_flag_batch"]
        self.token_back_mode = impl_desc["token_back_mode"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]

        self.occupancy = 1
        self.architecture = "sm_107"
        self.local_expert_count = self.expert_count // self.world_size
        self.threads_per_cta = 16 * 32 if self.token_back_mode == "standalone_warps" else 12 * 32
        self.other_warp_register_count = 64 if self.token_back_mode == "standalone_warps" else 72

        self.mixed_cga_config = NonClcMixedCgaConfig(
            preferred_cluster_shape=self.cluster_shape_mn,
            fallback_cluster_shape=impl_desc.get("fallback_cluster_shape_mn"),
            launch_cluster_count=self.launch_cluster_count,
            preferred_cluster_count=impl_desc.get("preferred_cluster_count"),
            fallback_cluster_count=impl_desc.get("fallback_cluster_count"),
        )
        self.fallback_cluster_shape_mn = self.mixed_cga_config.fallback_cluster_shape
        self.preferred_cluster_count = self.mixed_cga_config.preferred_cluster_count
        self.fallback_cluster_count = self.mixed_cga_config.fallback_cluster_count
        self.resolved_fallback_cluster_shape_mn = (
            self.cluster_shape_mn
            if self.fallback_cluster_shape_mn is None
            else self.fallback_cluster_shape_mn
        )
        if self.launch_cluster_count != self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred:
            raise ValueError(
                "launch_cluster_count must equal the canonical preferred-cluster slot count resolved from "
                "preferred_cluster_count and fallback_cluster_count."
            )

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
        tokens_per_fc1_ready_slot = self.mma_tiler_mnk[1] * self.cluster_shape_mn[1]
        instruction_cta_m = self.mma_instruction_mnk[0] // mma_cta_count
        instruction_m_repetitions = self.mma_tiler_mnk[0] // self.mma_instruction_mnk[0]
        hidden_per_fc2_cluster_tile = (
            instruction_cta_m * instruction_m_repetitions * self.cluster_shape_mn[0]
        )
        fc2_done_signals_per_token_tile = (
            ceil_div(self.hidden_size, hidden_per_fc2_cluster_tile)
            * self.cluster_shape_mn[0]
            * self.cluster_shape_mn[1]
        )
        token_comm_impl_desc = ImplDesc(
            {
                **impl_desc,
                "fallback_cluster_shape_mn": self.fallback_cluster_shape_mn,
                "preferred_cluster_count": self.preferred_cluster_count,
                "fallback_cluster_count": self.fallback_cluster_count,
                "tokens_per_fc1_ready_slot": tokens_per_fc1_ready_slot,
                "fc2_done_signals_per_token_tile": fc2_done_signals_per_token_tile,
                "promised_launchable_sm_count": self.mixed_cga_config.total_cta_cnt,
                "token_back_schedule_mode": (
                    "atomic_counter" if self.work_id_mode == "atomic_counter" else "static"
                ),
                "router_smem_limit_bytes": cutlass.memory.get_smem_capacity_in_bytes(
                    self.architecture
                ),
            }
        )
        self.token_comm = TokenCommNonDeterministic(problem_desc, token_comm_impl_desc)
        self.max_tokens = self.token_comm.worst_case_token_count

        fc12_problem_desc = ProblemDesc({**problem_desc, "expert_count": self.local_expert_count})
        resolved_impl_desc = ImplDesc(
            {
                **impl_desc,
                "fallback_cluster_shape_mn": self.fallback_cluster_shape_mn,
                "preferred_cluster_count": self.preferred_cluster_count,
                "fallback_cluster_count": self.fallback_cluster_count,
                "hint": self.hint,
                "tmem_plan": tmem_plan,
                "is_swap_ab": True,
                "max_tokens": self.max_tokens,
                "num_scheduler_consumer_threads": self.scheduler_consumer_thread_count,
                "num_accumulator_consumer_warps_per_cta": len(self.epilogue_warp_ids),
                "communication_enabled": True,
                "token_back_push_data": self.token_comm.token_back_push_data,
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
        self._device_workspace = self._build_device_workspace()
        self._mainloop, self._smem_workspace = self._build_mainloop_and_smem(
            fc12_problem_desc, resolved_impl_desc
        )
        self._topk_reduce = (
            None
            if self.reduce_topk_in_kernel
            else TopkReduce(self.hidden_size, self.topk, self.combine_format)
        )

    def _validate_geometry(self) -> None:
        static_expert_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_expert_dimensions) and not all(static_expert_dimensions):
            raise ValueError("MegaMoE expert dimensions must be either all static or all runtime.")
        if not all(static_expert_dimensions):
            raise NotImplementedError(
                "The MegaMoE Kernel currently requires static expert dimensions."
            )
        if self.expert_count <= 0 or self.intermediate_gateup_size <= 0 or self.hidden_size <= 0:
            raise ValueError("MegaMoE expert dimensions must be positive.")
        if self.world_size <= 0 or self.expert_count % self.world_size != 0:
            raise ValueError("expert_count must be divisible by a positive world_size.")
        if self.topk <= 0 or self.topk > self.expert_count:
            raise ValueError("topk must be positive and no greater than expert_count.")
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(
                f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}."
            )
        if self.reduce_topk_in_kernel and not self.apply_topk_at_fc1:
            raise ValueError("In-kernel top-k reduction requires apply_topk_at_fc1=True.")
        if self.intermediate_gateup_size % 2 != 0:
            raise ValueError("The SwiGLU intermediate dimension must be even.")
        intermediate_downproj = self.intermediate_gateup_size // 2
        if self.max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive.")
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a multiple of 64.")
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
                f"Rubin asynchronous epi_flag_batches values must be in [1, {len(self.epilogue_warp_ids)}]."
            )
        if self.sf_vec_size <= 0:
            raise ValueError("sf_vec_size must be positive.")
        for field_name, dimensions, expected_rank in (
            ("mma_instruction_mnk", self.mma_instruction_mnk, 3),
            ("mma_tiler_mnk", self.mma_tiler_mnk, 3),
            ("cluster_shape_mn", self.cluster_shape_mn, 2),
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
        if self.quant_kind == QuantKind.mxfp4_mxfp8:
            # Rubin consumes the mixed FP4 weight from native packed SMEM. Its K-major source only
            # needs ordinary 16-byte TMA alignment and complete SFVec32 blocks.
            for name, extent in (
                ("hidden_size", self.hidden_size),
                ("intermediate_downproj", intermediate_downproj),
            ):
                if extent * int(self.a_dtype.width) % 128 != 0 or extent % self.sf_vec_size != 0:
                    raise ValueError(
                        f"Rubin {self.quant_kind} requires {name} to satisfy ordinary TMA and "
                        f"SFVec{self.sf_vec_size} alignment, got {extent}."
                    )
        if self.mma_k_mode != "2x":
            raise ValueError(f"Rubin MegaMoE requires mma_k_mode='2x', got {self.mma_k_mode!r}.")

        mma_m, mma_n, mma_k = self.mma_tiler_mnk
        instruction_m, instruction_n, instruction_k = self.mma_instruction_mnk
        cluster_m, cluster_n = self.cluster_shape_mn
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
                "Rubin MegaMoE does not implement M/N instruction repetition or B-reuse."
            )
        if mma_n not in (64, 128, 256):
            raise ValueError(f"mma_tiler N must be 64, 128, or 256, got {mma_n}.")
        if mma_k % instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if mma_k % (self.sf_vec_size * 4) != 0:
            raise ValueError("mma_tiler K must be divisible by four scale-factor vectors.")
        if self.quant_kind == QuantKind.nvfp4 and (mma_n, mma_k) == (256, 512):
            raise ValueError(
                "Rubin MegaMoE excludes NVFP4 N256 K512 because its multi-window path has no measured gain."
            )
        if self.quant_kind == QuantKind.mxfp4_mxfp8:
            sf_atom_size = self.sf_vec_size * 4
            if intermediate_downproj % sf_atom_size != 0:
                raise ValueError(
                    f"The mixed-precision down-projection dimension must be divisible by one "
                    f"four-vector SF atom ({sf_atom_size} elements)."
                )
        elif self.intermediate_gateup_size % (self.sf_vec_size * 4) != 0:
            raise ValueError(
                "The intermediate dimension must be divisible by four scale-factor vectors."
            )
        if cluster_n != 1:
            raise ValueError(f"The swap-AB FC12 path requires cluster N=1, got {cluster_n}.")
        if cluster_m <= 0 or cluster_m > 16 or cluster_m & (cluster_m - 1):
            raise ValueError("cluster M must be a power of two no greater than 16.")
        if self.use_2cta_instrs and cluster_m % 2 != 0:
            raise ValueError("Two-CTA MMA requires an even cluster M.")
        if self.fallback_cluster_shape_mn is not None:
            fallback_cluster_m, fallback_cluster_n = self.fallback_cluster_shape_mn
            if fallback_cluster_n != 1:
                raise ValueError(
                    f"The swap-AB FC12 fallback path requires cluster N=1, got {fallback_cluster_n}."
                )
            if (
                fallback_cluster_m <= 0
                or fallback_cluster_m > 16
                or fallback_cluster_m & (fallback_cluster_m - 1)
            ):
                raise ValueError("fallback cluster M must be a power of two no greater than 16.")
            if self.use_2cta_instrs and fallback_cluster_m % 2 != 0:
                raise ValueError("Two-CTA MMA requires an even fallback cluster M.")

    def _resolve_schedule_hint(self) -> None:
        if self.schedule_mode != "phase_interleave":
            return

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        cluster_feature_tile = self.mma_tiler_mnk[0] // mma_cta_count * self.cluster_shape_mn[0]
        blocks_fc1 = ceil_div(self.intermediate_gateup_size, cluster_feature_tile)
        blocks_fc2 = ceil_div(self.hidden_size, cluster_feature_tile)
        raw_minimum_hint = minimum_phase_interleave_hint(
            blocks_fc1=blocks_fc1,
            blocks_fc2=blocks_fc2,
            launch_cluster_cnt_merge_as_preferred=self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred,
        )
        minimum_safe_hint = round_up(raw_minimum_hint, self.epi_flag_batches[0])
        if self.hint is None:
            self.hint = minimum_safe_hint
        elif self.hint < minimum_safe_hint:
            raise ValueError(
                f"phase_interleave hint {self.hint} is unsafe with "
                f"fc1_epi_flag_batch={self.epi_flag_batches[0]}; "
                f"minimum legal hint is {minimum_safe_hint}."
            )

    def _build_device_workspace(self) -> DeviceWorkspace:
        intermediate_downproj = self.intermediate_gateup_size // 2
        sf_column_count = round_up(ceil_div(intermediate_downproj, self.sf_vec_size), 4)
        max_sf_rows = self.token_comm.worst_case_sf_token_count
        counter_slot_count = self.token_comm.max_fc1_ready_slot_count

        device_workspace = DeviceWorkspace()
        device_workspace.register(
            self.fc1_output_region,
            self.epilogue.fc1_output_dtype,
            (self.max_tokens, intermediate_downproj),
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
        self.scheduler.register_device_workspace(device_workspace)
        self.token_comm.register_device_workspace(device_workspace)
        device_workspace.finalize()
        return device_workspace

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return required local and shared workspace bytes."""
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
        fallback_cluster = (
            "none"
            if self.fallback_cluster_shape_mn is None
            else "x".join(str(dimension) for dimension in self.fallback_cluster_shape_mn)
        )
        epi_flags = "x".join(str(batch) for batch in self.epi_flag_batches)
        return (
            f"sm107_block_scaled_swap_ab_mega_moe_{self.quant_kind}_"
            f"{dtype_name(self.a_dtype)}_{dtype_name(self.b_dtype)}_{dtype_name(self.acc_dtype)}_"
            f"sfvec{self.sf_vec_size}_a{self.a_major_mode.name.lower()}_b{self.b_major_mode.name.lower()}_"
            f"e{self.expert_count}_ep{self.world_size}_topk{self.topk}_"
            f"topkidx{dtype_name(self.topk_index_dtype)}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_maxtoken{self.max_tokens_per_rank}_"
            f"inst{instruction}_{self.mma_k_mode}_tile{tile}_"
            f"cluster{cluster}_{'2cta' if self.use_2cta_instrs else '1cta'}_"
            f"fallback{fallback_cluster}_prefclusters{self.preferred_cluster_count}_"
            f"fallbackclusters{self.fallback_cluster_count}_"
            f"sched{self.schedule_mode}_hint{self.hint}_pad{self.token_padding_block}x{self.sf_padding_block}_"
            f"work{self.work_id_mode}_"
            f"fc2store{'tma' if self.epilogue.fc2_use_tma else 'ublk' if self.epilogue.fc2_use_ublk else 'stg'}_"
            f"fc2tmastages{self.epilogue.fc2_tma_stages}_"
            f"epiflag{epi_flags}_clusters{self.launch_cluster_count}_ctas{self.mixed_cga_config.total_cta_cnt}_"
            f"tokeninflag{self.token_in_flag_batch}_tokenback{self.token_back_mode}_"
            f"combine{self.combine_format.name}_clamp{self.gate_up_clamp}_"
            # The SiTU betas are codegen-time constants, so they must key the compiled kernel.
            f"situ{self.situ_beta}x{self.situ_linear_beta}_"
            f"{'apply_topk_fc1' if self.apply_topk_at_fc1 else 'apply_topk_fc2'}_"
            f"{'inkernel_redg' if self.reduce_topk_in_kernel else 'separate_reduce'}"
        )

    def aot_compile(self, out_path: Optional[str] = None, **_compile_kwargs):
        """Compile against fake (metadata-only) inputs; ``out_path=None`` returns the in-memory callable.

        The fake tensors define the runtime ABI, so each mirrors what the caller stages: the per-rank expert slice,
        the padded SF extents, and ``to_cute``'s ``mark_layout_dynamic`` (the stride-1 axis stays static, every
        other axis becomes a runtime SymInt). ``stride_order`` follows the input generator's ``memory_order``, where
        0 is the innermost axis. Occupancy is a construction-time knob (``launch_cluster_count``), so unlike the
        pre-``next`` kernel this entry accepts no occupancy argument and tolerates leftover caller kwargs.
        """
        import math

        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
        from cutlass.cute.typing import AddressSpace, sym_int64
        from cutlass.cutlass_dsl import Int64

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
        # Weight SF is atom-swizzled and opaque, so its extent is whatever the caller staged, not a kernel choice.
        fc1_weight_sf_columns = round_up(intermediate_gateup, 128) * round_up(
            hidden // sf_vec_size, 4
        )
        fc2_weight_sf_columns = round_up(hidden, 128) * round_up(
            intermediate_downproj // sf_vec_size, 4
        )
        # TopkReduce and the in-kernel REDG both take their output dtype from this tensor.
        output_dtype = cutlass.BFloat16

        fake_arguments = dict(
            # Swap-AB puts the weights on the MMA A operand, so the token wire dtype is TokenComm's, not a_dtype.
            activation=fake_tensor(
                self.token_comm.activation_dtype, (tokens, hidden), (1, 0), {0}, 16
            ),
            activation_sf=fake_tensor(
                self.token_comm.activation_sf_dtype,
                (tokens, self.token_comm.activation_sf_hidden_padded),
                (1, 0),
                {0},
                16,
            ),
            topk_indices=fake_tensor(self.topk_index_dtype, (tokens, self.topk), (1, 0), {0}, 16),
            topk_scores=fake_tensor(cutlass.Float32, (tokens, self.topk), (1, 0), {0}, 4),
            fc1_weight=fake_tensor(
                self.a_dtype, (experts, hidden, intermediate_gateup), (2, 0, 1), {0, 2}, 16
            ),
            fc1_weight_sf=fake_tensor(
                self.sf_dtype, (experts, fc1_weight_sf_columns), (1, 0), {0}, 16
            ),
            fc2_weight=fake_tensor(
                self.a_dtype, (experts, intermediate_downproj, hidden), (2, 0, 1), {0, 2}, 16
            ),
            fc2_weight_sf=fake_tensor(
                self.sf_dtype, (experts, fc2_weight_sf_columns), (1, 0), {0}, 16
            ),
            output_activation=fake_tensor(output_dtype, (tokens, hidden), (1, 0), {0}, 16),
            # Opaque byte workspaces: a bare base pointer, like to_cute_ptr. The 128-byte promise matches the
            # alignment the registered regions assume, so the caller must hand over 128-byte-aligned bases.
            local_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            shared_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            # Placeholder field values: they marshal as runtime scalars. Only max_ranks is constexpr.
            peer_rank_ptr_mapper_host=SymmetricBufferHost(
                base_address=Int64(0),
                offsets=tuple(Int64(0) for _ in range(self.world_size)),
                rank=Int32(0),
                max_ranks=self.world_size,
            ),
            stream=make_fake_stream(),
        )
        if self.quant_kind.uses_global_scale:
            # nvfp4 carries per-expert dequant scalars. Under an e8m0 scale they do not exist at
            # all, and declaring them here would put three tensors in the ABI that the caller has
            # no value for.
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

    def _build_mainloop_and_smem(
        self, problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc
    ) -> Tuple[BlockScaledSwapAbFc12Mainloop, SmemWorkspace]:
        smem_limit = cutlass.memory.get_smem_capacity_in_bytes(self.architecture) // self.occupancy
        smem_workspace = SmemWorkspace()
        self.token_comm.register_smem_regions(smem_workspace)
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
        mainloop = BlockScaledSwapAbFc12Mainloop(problem_desc, mainloop_impl_desc)

        mainloop.register_smem_regions(smem_workspace)
        smem_workspace.finalize(max_bytes=smem_limit)
        return mainloop, smem_workspace

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,  # (tokens, hidden)
        activation_sf: cute.Tensor,  # (tokens, hidden // activation_sf_vector_size)
        topk_indices: cute.Tensor,  # (tokens, topk)
        topk_scores: cute.Tensor,  # (tokens, topk)
        fc1_weight: cute.Tensor,  # (local_experts, hidden, intermediate_gateup)
        fc1_weight_sf: cute.Tensor,  # (local_experts, packed_fc1_scale_factors)
        fc2_weight: cute.Tensor,  # (local_experts, intermediate_downproj, hidden)
        fc2_weight_sf: cute.Tensor,  # (local_experts, packed_fc2_scale_factors)
        output_activation: cute.Tensor,  # (tokens, hidden)
        local_workspace: cute.Pointer,  # local GMEM byte workspace
        shared_workspace: cute.Pointer,  # symmetric GMEM byte workspace
        peer_rank_ptr_mapper_host,
        stream: cuda.CUstream,
        fc1_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc2_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc1_norm_const: Optional[cute.Tensor] = None,  # (local_experts,)
    ) -> None:
        """Launch router, fused MegaMoE compute, and optional TopK reduce."""

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
        if cutlass.const_expr(not self.quant_kind.uses_global_scale):
            for scalar_name, scalar in (
                ("fc1_alpha", fc1_alpha),
                ("fc2_alpha", fc2_alpha),
                ("fc1_norm_const", fc1_norm_const),
            ):
                if cutlass.const_expr(scalar is not None):
                    raise ValueError(
                        f"{self.quant_kind} folds its rescale into the e8m0 scale factors, "
                        f"so {scalar_name} must be None."
                    )

        router_topk_scores = topk_scores if cutlass.const_expr(self.apply_topk_at_fc1) else None
        local_rank = peer_rank_ptr_mapper_host.rank
        self.token_comm.launch_router(
            topk_indices,
            router_topk_scores,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host,
            self._device_workspace,
            stream,
        )

        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_object()
        self._device_workspace.assign_device_members(local_workspace, shared_workspace)
        activation_pool = self.token_comm.fc1_activation_tensor(self._device_workspace)
        activation_sf_pool = self.token_comm.fc1_activation_sf_tensor(self._device_workspace)
        fc1_output = self._device_workspace.tensor(self.fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.fc1_output_sf_region)

        experts = self.local_expert_count
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        fc1_weight = rewrite_tensor_shape(fc1_weight, (experts, hidden, intermediate_gateup))
        fc2_weight = rewrite_tensor_shape(fc2_weight, (experts, intermediate_downproj, hidden))

        singleton = cutlass.Int32(1)
        token_rows = self.max_tokens

        fc1_a = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (cutlass.Int32(intermediate_gateup), cutlass.Int32(hidden), experts),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
            ),
        )
        fc1_b = cute.make_tensor(
            activation_pool.iterator,
            cute.make_layout(
                (token_rows, cutlass.Int32(hidden), singleton),
                stride=(activation_pool.stride[0], activation_pool.stride[1], 0),
            ),
        )
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (token_rows, cutlass.Int32(intermediate_downproj), singleton),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )

        sf_vec_size = self.sf_vec_size
        padded_token_rows = self.token_comm.worst_case_sf_token_count
        padded_hidden = hidden
        fc1_sfb = cute.make_tensor(
            activation_sf_pool.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (padded_token_rows, cutlass.Int32(padded_hidden), singleton), sf_vec_size
            ),
        )
        padded_intermediate_gateup = cute.round_up(intermediate_gateup, sf_vec_size * 4)
        expected_fc1_weight_sf_columns = padded_intermediate_gateup * padded_hidden // sf_vec_size
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
                    cutlass.Int32(padded_hidden),
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
            pre_reduced_activation_sf = None
        else:
            pre_reduced_activation = self.token_comm.pre_reduced_activation_tensor(
                self._device_workspace
            )
            pre_reduced_activation_sf = self.token_comm.pre_reduced_activation_sf_tensor(
                self._device_workspace
            )

        if cutlass.const_expr(self.token_comm.token_back_push_data):
            fc2_output = self.token_comm.fc2_activation_tensor(self._device_workspace)
        else:
            fc2_output = rewrite_tensor_shape(
                pre_reduced_activation,
                (pre_reduced_activation.shape[0], pre_reduced_activation.shape[1], hidden),
            )
        fc2_output_sf = self.token_comm.fc2_activation_sf_tensor(self._device_workspace)

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
            fc1_tma_b_tensor,
            fc1_tma_b_atom_or_pair,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom_or_pair,
            fc2_tma_b_tensor,
            fc2_tma_b_atom_or_pair,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom_or_pair,
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

        (fc1_output_tma_atom, fc1_output_tma_tensor, fc2_output_tma_atom, fc2_output_tma_tensor) = (
            self.epilogue.prepare_tma_store_params(fc1_output_gemm, fc2_output)
        )

        grid = self.scheduler.get_grid_shape(max_active_clusters=self.launch_cluster_count)
        self._device_workspace.remove_device_members()
        kernel = self._kernel(
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_tensor,
            fc1_tma_b_atom_or_pair,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom_or_pair,
            fc2_tma_b_tensor,
            fc2_tma_b_atom_or_pair,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom_or_pair,
            fc1_output_tma_atom,
            fc1_output_tma_tensor,
            fc2_output_tma_atom,
            fc2_output_tma_tensor,
            fc2_output,
            fc2_output_sf,
            (experts, intermediate_gateup, hidden),
            activation,
            activation_sf,
            pre_reduced_activation,
            pre_reduced_activation_sf,
            peer_rank_ptr_mapper,
            local_rank,
            local_workspace,
            shared_workspace,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
        )
        if cutlass.const_expr(self.mixed_cga_config.is_mixed):
            kernel.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                fallback_cluster=(*self.resolved_fallback_cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=self.occupancy,
                use_pdl=True,
            )
        else:
            kernel.launch(
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
                pre_reduced_activation,
                pre_reduced_activation_sf,
                output_activation,
                reduce_scores,
                stream,
            )

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
        fc1_tma_b_tensor: cute.Tensor,
        fc1_tma_b_atom_or_pair: TmaAtomOrPair,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom_or_pair: TmaAtomOrPair,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom_or_pair: TmaAtomOrPair,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom_or_pair: TmaAtomOrPair,
        fc1_output_tma_atom: cute.CopyAtom,
        fc1_output_tma_tensor: cute.Tensor,
        fc2_output_tma_atom: Optional[cute.CopyAtom],
        fc2_output_tma_tensor: Optional[cute.Tensor],
        fc2_output: cute.Tensor,
        fc2_output_sf: Optional[cute.Tensor],
        actual_expert_shape: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        pre_reduced_activation: cute.Tensor,
        pre_reduced_activation_sf: Optional[cute.Tensor],
        peer_rank_ptr_mapper,
        local_rank: cutlass.Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
    ):
        """Compose TokenComm, Scheduler, Mainloop, and Epilogue."""
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
        warp_idx = thread_idx // 32
        block_idx = cute.arch.block_idx()
        active_cluster_m = Int32(self.cluster_shape_mn[0])
        is_fallback_cluster = Boolean(False)
        if cutlass.const_expr(self.mixed_cga_config.is_mixed):
            active_cluster_m, _, _ = cute.arch.block_in_cluster_dim()
            is_fallback_cluster = active_cluster_m == Int32(
                self.resolved_fallback_cluster_shape_mn[0]
            )
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_coord_in_cluster = (cta_rank_in_cluster, Int32(0), Int32(0))
        linear_cta_idx = block_idx[0] + block_idx[2] * Int32(self.cluster_shape_mn[0])
        token_comm_args = TokenCommArgs(
            activation=activation,
            activation_sf=activation_sf,
            pre_reduced_activation=pre_reduced_activation,
            pre_reduced_activation_sf=pre_reduced_activation_sf,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
        )
        self.token_comm.assign_device_members(
            device_workspace=self._device_workspace,
            token_comm_args=token_comm_args,
            local_rank=local_rank,
            linear_cta_idx=linear_cta_idx,
        )
        expert_token_sizes = self.token_comm.local_expert_sizes(self._device_workspace, local_rank)
        token_src_metadata = self.token_comm.token_src_metadata_tensor(self._device_workspace)
        fc2_done_counter = self.token_comm.fc2_done_counter_tensor(self._device_workspace)
        pool_topk_scores = self.token_comm.fc1_topk_scores_tensor(self._device_workspace)

        self._mainloop.assign_device_members(
            self._smem_workspace,
            smem_base,
            cta_coord_in_cluster,
            is_fallback_cluster,
            hidden,
            intermediate_gateup,
        )
        ab_pipeline = self._mainloop.create_ab_pipeline(self._smem_workspace, smem_base)
        tma_a_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_ab_pipeline_stages
        )
        tma_b_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_ab_pipeline_stages
        )
        mma_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._mainloop.num_ab_pipeline_stages
        )
        acc_pipeline = self._mainloop.create_acc_pipeline(self._smem_workspace, smem_base)
        tmem_allocator = self._mainloop.create_tmem_allocator(
            self._smem_workspace, smem_base, allocator_warp_id=self.epilogue_warp_ids[0]
        )
        self.scheduler.assign_device_members(
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=None,
            actual_expert_shape=actual_expert_shape,
            block_idx=block_idx,
            smem_workspace=self._smem_workspace,
            smem_base=smem_base,
            device_workspace=self._device_workspace,
            is_fallback_cluster=is_fallback_cluster,
        )
        scheduler = self.scheduler

        fc2_spin_threshold = (
            intermediate_gateup + self._mainloop.cta_tile_m - 1
        ) // self._mainloop.cta_tile_m
        kernel_extension = BlockScaledSwapAbFc12Extension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=self.token_comm.fc1_ready_counter_pointer(
                self._device_workspace
            ),
        )
        optional_epilogue_args = GatedActEpilogueArgs(
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            topk_scores=pool_topk_scores,
        )

        pipeline_init_arrive_mixed_cga(
            self.cluster_shape_mn,
            self.resolved_fallback_cluster_shape_mn,
            is_fallback_cluster,
            is_relaxed=True,
        )
        pipeline_init_wait_mixed_cga(
            self.cluster_shape_mn, self.resolved_fallback_cluster_shape_mn, is_fallback_cluster
        )
        finalize_barrier_thread_count = (
            12 * 32 if self.token_back_mode == "standalone_warps" else 8 * 32
        )
        finalize_barrier = pipeline.NamedBarrier(
            barrier_id=13, num_threads=finalize_barrier_thread_count
        )

        if warp_idx == self.scheduler_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.scheduler_register_count)
            iket.range_push("mega.scheduler")
            if cutlass.const_expr(hasattr(scheduler, "initialize_fallback_group")):
                iket.range_push("scheduler.register_group")
                scheduler.initialize_fallback_group()
                iket.range_pop()
            iket.range_push("scheduler.wait_sizes_ready")
            self.token_comm.wait_for_sizes_ready(self._device_workspace)
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
            iket.range_push("mega.tma_a")
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
            iket.range_push("mega.tma_b")
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_b(
                fc1_tma_b_tensor=fc1_tma_b_tensor,
                fc1_tma_b_atom_or_pair=fc1_tma_b_atom_or_pair,
                fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
                fc1_tma_sfb_atom_or_pair=fc1_tma_sfb_atom_or_pair,
                fc2_tma_b_tensor=fc2_tma_b_tensor,
                fc2_tma_b_atom_or_pair=fc2_tma_b_atom_or_pair,
                fc2_tma_sfb_tensor=fc2_tma_sfb_tensor,
                fc2_tma_sfb_atom_or_pair=fc2_tma_sfb_atom_or_pair,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=tma_b_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                fc1_done_counter_pointer=fc1_done_counter.iterator,
                fc2_spin_threshold=fc2_spin_threshold,
            )
            iket.range_pop()

        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("mega.mma")
            sched_consumer = scheduler.make_consumer()
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
            iket.range_push("mega.epilogue")
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
                fc2_output_tma_atom,
                fc2_output_tma_tensor,
                fc2_output,
                fc1_done_counter,
                thread_idx,
                token_src_metadata,
                fc2_done_counter,
                fc2_output_sf,
                peer_rank_ptr_mapper,
                optional_epilogue_args,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(
                tmem_allocator.retrieve_ptr(self.acc_dtype), self._mainloop.num_tmem_alloc_cols
            )
            iket.range_pop()
            finalize_barrier.arrive_unaligned()

        if warp_idx >= Int32(self.transfer_warp_idx_start):
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            if warp_idx < Int32(self.token_in_end_warp_idx):
                iket.range_push("mega.token_in")
                self.token_comm.token_in(self._smem_workspace, smem_base)
                iket.range_pop()
                if cutlass.const_expr(
                    self.token_comm.token_back_enabled
                    and self.token_back_mode != "standalone_warps"
                ):
                    iket.range_push("mega.token_back")
                    self.token_comm.token_back(self._smem_workspace, smem_base)
                    iket.range_pop()
                finalize_barrier.arrive_and_wait()
                iket.range_push("mega.tail_reset")
                self.token_comm.reset_tail()
                iket.range_pop()
            elif cutlass.const_expr(self.token_back_mode == "standalone_warps"):
                iket.range_push("mega.token_back_standalone")
                self.token_comm.token_back(self._smem_workspace, smem_base)
                iket.range_pop()
                finalize_barrier.arrive_unaligned()
        self.token_comm.remove_device_members()
        self._device_workspace.remove_device_members()
