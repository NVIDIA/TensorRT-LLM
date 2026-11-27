"""Local MegaMoE kernel composition for Rubin."""

import hashlib

from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .....api import ImplDesc, KernelClass, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from .....helpers.cute_py_helpers import Tcgen05MmaInstruction, make_tcgen05_tmem_plan, tcgen05_block_scaled_acc_dtype
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dlb_expert_count import resolve_dlb_expert_counts
from .....helpers.iket_compat import iket
from .....helpers.megamoe_aot import make_megamoe_aot_callable
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.topk_reduce import TopkReduce
from .....helpers.utils import ceil_div, ring_token_blocks_in_effect, round_up
from .....helpers.helper_weight_ready_gate import (
    derive_helper_weight_ready_topology,
    make_helper_weight_ready_generation_gate,
)
from .....quant_def import CombineFormat, QuantKind
from ....schedulers.base import WorkIdAcquisitionMode
from ....schedulers.fc12_scheduler import (
    BlackwellFusedFc12Scheduler,
    PhaseInterleavedFc12Scheduler,
    minimum_phase_interleave_hint,
    minimum_phase_interleave_ring_token_blocks,
)
from .block_scaled_swap_ab_fc12_epilogue import GatedActEpilogueArgs, SwapABGatedActEpilogue
from ..mega.block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .block_scaled_swap_ab_fc12_mainloop import BlockScaledSwapAbFc12Mainloop
from .local_routing import LocalRouting, LocalRoutingArgs


_aot_symbol_prefix = "rubin_local_mega_moe_aot"


class BlockScaledSwapAbLocalMegaMoeKernel(KernelClass):
    """Compose local routing, persistent FC12, and top-k reduction."""

    fc1_output_region: ClassVar[str] = "rubin.local_mega.fc1_output"
    fc1_output_sf_region: ClassVar[str] = "rubin.local_mega.fc1_output_sf"
    fc1_done_counter_region: ClassVar[str] = "rubin.local_mega.fc1_done_counter"
    # Ring workspace only: the retire counter, one per pool token block. The FC2 epilogue adds one for every FC2
    # tile of the block that no longer reads the block's FC1 output; token_in reads it (the slot lease).
    ring_retire_region: ClassVar[str] = "rubin.local_mega.ring_retire_count"
    epilogue_warp_ids: ClassVar[Tuple[int, int, int, int]] = (0, 1, 2, 3)
    mma_warp_id: ClassVar[int] = 4
    tma_a_warp_id: ClassVar[int] = 5
    tma_b_warp_id: ClassVar[int] = 6
    scheduler_warp_id: ClassVar[int] = 7
    transfer_warp_idx_start: ClassVar[int] = 8
    token_in_end_warp_idx: ClassVar[int] = 12
    gather_b_warp_ids: ClassVar[Tuple[int, int, int, int]] = (12, 13, 14, 15)
    scheduler_consumer_thread_count: ClassVar[int] = 11 * 32
    epilogue_register_count: ClassVar[int] = 256
    scheduler_register_count: ClassVar[int] = 72

    @classmethod
    def problem_desc_require(cls) -> dict[str, object]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
            "gate_up_clamp": Optional[float],
            "swiglu_alpha": OptionalRequirement(Optional[float]),
            "swiglu_beta": OptionalRequirement(Optional[float]),
            "situ_beta": OptionalRequirement(Optional[float]),
            "situ_linear_beta": OptionalRequirement(Optional[float]),
            "topk": int,
            "topk_index_dtype": type,
            "max_tokens": int,
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
            "fc1_epi_flag_batch": int,
            "launch_cluster_count": int,
            "routing_flag_batch": int,
            "reduce_topk_in_kernel": bool,
            "helper_expert_count": OptionalRequirement(int),
            "total_helper_slots": OptionalRequirement(int),
            "weight_ready_source_count": OptionalRequirement(int),
            "ring_token_blocks": OptionalRequirement(Optional[int]),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)
        problem_desc, impl_desc = resolve_dlb_expert_counts(
            problem_desc, impl_desc, world_size=1
        )

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind.nvfp4
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.a_major_mode = problem_desc["a_major_mode"]
        self.b_major_mode = problem_desc["b_major_mode"]
        self.combine_format = CombineFormat.parse("bf16")
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.swiglu_alpha = problem_desc.get("swiglu_alpha")
        self.swiglu_beta = problem_desc.get("swiglu_beta")
        self.situ_beta = problem_desc.get("situ_beta")
        self.situ_linear_beta = problem_desc.get("situ_linear_beta")
        self.topk = problem_desc["topk"]
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.max_tokens = problem_desc["max_tokens"]
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
        self.fc1_epi_flag_batch = impl_desc["fc1_epi_flag_batch"]
        self.launch_cluster_count = impl_desc["launch_cluster_count"]
        self.routing_flag_batch = impl_desc["routing_flag_batch"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]
        # Ring workspace: the ring_token_blocks the caller asked for (None or 0: none). The ring in effect is
        # self.ring_token_blocks, resolved below once the geometry is validated.
        self.requested_ring_token_blocks = impl_desc.get("ring_token_blocks")
        self.helper_expert_count = impl_desc.get("helper_expert_count", 0)
        self.weight_ready_source_count = impl_desc.get("weight_ready_source_count", 1)
        self._validate_helper_configuration()

        self.occupancy = 1
        self.architecture = "sm_107"
        self.threads_per_cta = 16 * 32
        self.other_warp_register_count = 80

        self._validate_geometry()
        # Ring workspace (0: off). The FC1 scale pool and the FC1 output are sized for ring_token_blocks token
        # blocks instead of the worst-case routed tokens: pool token block b lives in slot b % ring_token_blocks
        # of both and keeps it until the block is retired, that is until its last reader is done (the slot
        # lease). token_in alone waits for the ring: it writes a block once the block that held its slot before
        # is retired. The FC1 activation is not pooled here (it is gathered from the caller's tensor), and the
        # routing metadata stays indexed by absolute pool row. A requested ring that would not be smaller than
        # the worst case is not in effect: the kernel is then the kernel without the key, in everything it
        # sizes, generates and names.
        self.ring_token_blocks = ring_token_blocks_in_effect(
            self.requested_ring_token_blocks,
            tokens_per_block=self.tokens_per_token_block,
            source_tokens=self.max_tokens,
            local_experts=self.expert_count,
            topk=self.topk,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
        )
        # A requested ring is checked as a ring, in effect or not: what is supported does not depend on the
        # problem size.
        self._validate_ring_workspace()
        # The key never makes an invalid configuration valid: the caller's hint is validated against the
        # caller's FC1 flag batch, as without the key, before the ring takes that batch over.
        self._resolve_schedule_hint()
        if self.ring_token_blocks:
            # The ring in effect takes this tuning value over, and name() shows what is in effect: a held FC1
            # completion flag could be the very one a ring slot waits for, so each tile publishes at once. The
            # prologue of a caller who left the hint to the kernel is then the smallest safe one, not that one
            # rounded up to the caller's batch; a hint the caller gave stays the caller's.
            self.fc1_epi_flag_batch = 1
            self.hint = self._ring_schedule()["hint"]
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
        tokens_per_fc1_ready_slot = self.tokens_per_token_block
        local_routing_impl_desc = ImplDesc(
            {
                **impl_desc,
                "token_in_flag_batch": self.routing_flag_batch,
                "tokens_per_fc1_ready_slot": tokens_per_fc1_ready_slot,
                "promised_launchable_sm_count": (
                    self.launch_cluster_count * self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
                ),
                "router_smem_limit_bytes": utils.get_smem_capacity_in_bytes(self.architecture),
                # Ring workspace, the slot lease: a block is retired once every in-bound FC2 tile of it has its
                # accumulator. The FC2 MMAs were the last reads of the block's FC1 output, and its FC1 tiles, the
                # readers of its scale rows, are done before them. The FC2 epilogue counts those tiles, one per
                # hidden CTA tile.
                "ring_token_blocks": self.ring_token_blocks,
                "ring_retire_region": self.ring_retire_region if self.ring_token_blocks else None,
                "ring_retire_threshold": (
                    ceil_div(self.hidden_size, self.mma_tiler_mnk[0] // mma_cta_count)
                    if self.ring_token_blocks
                    else None
                ),
            }
        )
        resolved_problem_desc = ProblemDesc(
            {
                **problem_desc,
                "quant_kind": self.quant_kind.value,
                "combine_format": self.combine_format,
            }
        )
        self.local_routing = LocalRouting(resolved_problem_desc, local_routing_impl_desc)
        self.max_routed_tokens = self.local_routing.worst_case_token_count

        fc12_problem_desc = ProblemDesc({**resolved_problem_desc, "expert_count": self.expert_count})
        resolved_impl_desc = ImplDesc(
            {
                **impl_desc,
                "hint": self.hint,
                "tmem_plan": tmem_plan,
                "is_swap_ab": True,
                "max_tokens": self.max_routed_tokens,
                "num_scheduler_consumer_threads": self.scheduler_consumer_thread_count,
                "num_accumulator_consumer_warps_per_cta": len(self.epilogue_warp_ids),
                "fc1_epi_flag_batch": self.fc1_epi_flag_batch,
            }
        )
        scheduler_type = (
            BlackwellFusedFc12Scheduler if self.schedule_mode == "grouped" else PhaseInterleavedFc12Scheduler
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
        self._mainloop, self._smem_workspace = self._build_mainloop_and_smem(fc12_problem_desc, resolved_impl_desc)
        self._topk_reduce = (
            None if self.reduce_topk_in_kernel else TopkReduce(self.hidden_size, self.topk, self.combine_format)
        )

    def _validate_helper_configuration(self) -> None:
        if type(self.helper_expert_count) is not int or self.helper_expert_count < 0:
            raise ValueError("helper_expert_count must be a non-negative exact int")
        if type(self.weight_ready_source_count) is not int or self.weight_ready_source_count <= 0:
            raise ValueError("weight_ready_source_count must be a positive exact int")
        if self.helper_expert_count > 0:
            derive_helper_weight_ready_topology(self.expert_count, self.helper_expert_count)

    def _validate_geometry(self) -> None:
        static_expert_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_expert_dimensions) and not all(static_expert_dimensions):
            raise ValueError("Local MegaMoE expert dimensions must be either all static or all runtime.")
        if not all(static_expert_dimensions):
            raise NotImplementedError("The Local MegaMoE kernel currently requires static expert dimensions.")
        if self.expert_count <= 0 or self.intermediate_gateup_size <= 0 or self.hidden_size <= 0:
            raise ValueError("Local MegaMoE expert dimensions must be positive.")
        if self.topk <= 0 or self.topk > self.expert_count:
            raise ValueError("topk must be positive and no greater than expert_count.")
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}.")
        if self.reduce_topk_in_kernel and not self.apply_topk_at_fc1:
            raise ValueError("In-kernel top-k reduction requires apply_topk_at_fc1=True.")
        if self.intermediate_gateup_size % 2 != 0:
            raise ValueError("The SwiGLU intermediate dimension must be even.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
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
        if self.hint is not None and (isinstance(self.hint, bool) or not isinstance(self.hint, int) or self.hint <= 0):
            raise ValueError(f"schedule_policy hint must be a positive Python int, got {self.hint!r}.")
        if self.schedule_mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError("phase_interleave currently requires work_id_mode='atomic_counter'.")
        if not 1 <= self.fc1_epi_flag_batch <= len(self.epilogue_warp_ids):
            raise ValueError(
                f"Rubin asynchronous fc1_epi_flag_batch must be in [1, {len(self.epilogue_warp_ids)}]."
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
            if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in dimensions):
                raise TypeError(f"{field_name} dimensions must be Python integers.")
            if any(dimension <= 0 for dimension in dimensions):
                raise ValueError(f"{field_name} dimensions must be positive.")
        if self.mma_k_mode != "2x":
            raise ValueError(f"Rubin Local MegaMoE requires mma_k_mode='2x', got {self.mma_k_mode!r}.")

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
            raise NotImplementedError("Rubin Local MegaMoE does not implement M/N instruction repetition or B-reuse.")
        if mma_n not in (64, 128, 256):
            raise ValueError(f"mma_tiler N must be 64, 128, or 256, got {mma_n}.")
        if mma_k % instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if mma_k % (self.sf_vec_size * 4) != 0:
            raise ValueError("mma_tiler K must be divisible by four scale-factor vectors.")
        if (mma_n, mma_k) == (256, 512):
            raise ValueError(
                "Rubin Local MegaMoE excludes NVFP4 N256 K512 because its multi-window path has no measured gain."
            )
        if self.intermediate_gateup_size % (self.sf_vec_size * 4) != 0:
            raise ValueError("The intermediate dimension must be divisible by four scale-factor vectors.")
        if cluster_n != 1:
            raise ValueError(f"The swap-AB FC12 path requires cluster N=1, got {cluster_n}.")
        if cluster_m <= 0 or cluster_m > 4 or cluster_m & (cluster_m - 1):
            raise ValueError("cluster M must be a power of two no greater than four.")
        if self.use_2cta_instrs and cluster_m % 2 != 0:
            raise ValueError("Two-CTA MMA requires an even cluster M.")

    def _resolve_schedule_hint(self) -> None:
        if self.schedule_mode != "phase_interleave":
            return

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        cluster_feature_tile = self.mma_tiler_mnk[0] // mma_cta_count * self.cluster_shape_mn[0]
        blocks_fc1 = ceil_div(self.intermediate_gateup_size, cluster_feature_tile)
        blocks_fc2 = ceil_div(self.hidden_size, cluster_feature_tile)
        # Cover the worst token-block alignment of one full persistent-cluster FC2 claim wave.
        max_dependent_token_blocks = ceil_div(self.launch_cluster_count + blocks_fc2 - 1, blocks_fc2)
        required_fc1_work = max_dependent_token_blocks * blocks_fc1
        raw_minimum_hint = max(1, ceil_div(required_fc1_work, self.launch_cluster_count))
        minimum_safe_hint = round_up(raw_minimum_hint, self.fc1_epi_flag_batch)
        if self.hint is None:
            self.hint = minimum_safe_hint
        elif self.hint < minimum_safe_hint:
            raise ValueError(
                f"phase_interleave hint {self.hint} is unsafe with "
                f"fc1_epi_flag_batch={self.fc1_epi_flag_batch}; "
                f"minimum legal hint is {minimum_safe_hint}."
            )

    def minimum_ring_token_blocks(self) -> int:
        """Smallest ring_token_blocks a kernel of this configuration accepts (phase_interleave only).

        It is the minimum a ring kernel enforces, also when this kernel was built without a ring.
        """
        if self.schedule_mode != "phase_interleave":
            raise ValueError("The ring workspace requires schedule_policy phase_interleave.")
        return minimum_phase_interleave_ring_token_blocks(**self._ring_schedule())

    def _ring_schedule(self) -> dict:
        """The phase_interleave schedule of a ring kernel of this configuration: the FC1 and FC2 cluster tiles of
        a token block, the clusters that claim them and the FC1 prologue (hint).

        A ring publishes every FC1 completion at once, so where the caller leaves the hint to the kernel the
        prologue of a ring kernel is the smallest safe one, not that one rounded up to the caller's FC1 flag
        batch. It is the same whether a ring is in effect or not.
        """
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        cluster_feature_tile = self.mma_tiler_mnk[0] // mma_cta_count * self.cluster_shape_mn[0]
        blocks_fc1 = ceil_div(self.intermediate_gateup_size, cluster_feature_tile)
        blocks_fc2 = ceil_div(self.hidden_size, cluster_feature_tile)
        # Every launched cluster claims work here, as in _resolve_schedule_hint.
        clusters = self.launch_cluster_count
        schedule = dict(blocks_fc1=blocks_fc1, blocks_fc2=blocks_fc2, launch_cluster_cnt_merge_as_preferred=clusters)
        hint = self.schedule_policy[1]
        if hint is None:
            # The raw minimum of _resolve_schedule_hint: the prologue covering one FC2 claim wave.
            hint = minimum_phase_interleave_hint(**schedule)
        return dict(schedule, hint=hint)

    def _validate_ring_workspace(self) -> None:
        """A requested ring is checked as a ring, whatever the problem size.

        Everything the ring cannot work with is refused for the ring the caller asked for, also where that ring
        would save nothing and the kernel runs without it (ring_token_blocks is 0 then): whether a configuration
        is supported does not depend on max_tokens_per_rank. The value itself was checked when the ring in effect
        was resolved. What the ring takes over instead of refusing is in the constructor.
        """
        ring_token_blocks = self.requested_ring_token_blocks
        if not ring_token_blocks:
            return
        if self.schedule_mode != "phase_interleave":
            # grouped runs the FC1 tiles of a whole expert group before its FC2 tiles: every block of the
            # group is live at once, and a single expert can hold every routed token.
            raise ValueError("The ring workspace requires schedule_policy phase_interleave.")
        token_tile = self.tokens_per_token_block
        if round_up(token_tile, 128) % token_tile != 0:
            # A ring slot is one token tile of data rows and that tile rounded up to whole 128-row scale atoms;
            # the kernel extension derives the data row of a slot from its scale row (get_gmem_tensor).
            raise ValueError(
                f"The ring workspace needs a token tile that divides the scale rows of its slot, got {token_tile}."
            )
        # DLB helper experts need no check: a helper slot is a physical expert, numbered in the pool like every
        # other, and the scheduler warp's wait for its weights holds a claimed tile back without changing the
        # claim order.
        minimum = self.minimum_ring_token_blocks()
        if ring_token_blocks < minimum:
            raise ValueError(
                f"ring_token_blocks {ring_token_blocks} cannot hold the FC1 lead of this schedule "
                f"(hint {self._ring_schedule()['hint']}, {self.launch_cluster_count} clusters); "
                f"the minimum is {minimum}."
            )

    @property
    def tokens_per_token_block(self) -> int:
        """Routed tokens of one token block: one token tile of one expert, the ready slot of the routing."""
        return self.mma_tiler_mnk[1] * self.cluster_shape_mn[1]

    @property
    def separate_tmp_buffer(self) -> bool:
        """True: the per-launch data lives in the caller-owned tmp buffer (third pointer operand)."""
        return True

    def _build_device_workspace(self) -> DeviceWorkspace:
        intermediate_downproj = self.intermediate_gateup_size // 2
        sf_column_count = round_up(ceil_div(intermediate_downproj, self.sf_vec_size), 4)
        # The FC1 output has the rows of the routing's FC1 pool: the worst case, or the ring.
        max_sf_rows = self.local_routing.fc1_pool_sf_token_count
        counter_slot_count = self.local_routing.max_fc1_ready_slot_count

        device_workspace = DeviceWorkspace(separate_tmp=self.separate_tmp_buffer)
        device_workspace.register(
            self.fc1_output_region,
            self.epilogue.fc1_output_dtype,
            (self.local_routing.fc1_pool_token_count, intermediate_downproj),
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
        if self.ring_token_blocks:
            # One counter per pool token block, like the FC1 done counter: the ring reuses rows, not counters.
            device_workspace.register(
                self.ring_retire_region,
                cutlass.Int32,
                (counter_slot_count,),
                buffer_space="local",
                byte_alignment=16,
                reset="tail_reset",
            )
        self.scheduler.register_device_workspace(device_workspace)
        self.local_routing.register_device_workspace(device_workspace)
        device_workspace.finalize()
        return device_workspace

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return required local and shared workspace bytes. The tmp buffer is sized by get_tmp_buffer_bytes()."""
        return self._device_workspace.local_and_shared_bytes

    def get_tmp_buffer_bytes(self) -> int:
        """Return the size of the caller-owned tmp buffer (0: the per-launch data stays in the local workspace)."""
        return self._device_workspace.tmp_bytes

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
        return (
            f"sm107_block_scaled_swap_ab_local_mega_moe_{self.quant_kind}_"
            f"{dtype_name(self.a_dtype)}_{dtype_name(self.b_dtype)}_{dtype_name(self.acc_dtype)}_"
            f"sfvec{self.sf_vec_size}_a{self.a_major_mode.name.lower()}_b{self.b_major_mode.name.lower()}_"
            f"e{self.expert_count}_topk{self.topk}_"
            f"helpers{self.helper_expert_count}_weightsources{self.weight_ready_source_count}_"
            f"topkidx{dtype_name(self.topk_index_dtype)}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_maxtoken{self.max_tokens}_"
            f"inst{instruction}_{self.mma_k_mode}_tile{tile}_"
            f"cluster{cluster}_{'2cta' if self.use_2cta_instrs else '1cta'}_"
            f"sched{self.schedule_mode}_hint{self.hint}_pad{self.token_padding_block}x{self.sf_padding_block}_"
            f"work{self.work_id_mode}_"
            f"fc2store{'tma' if self.epilogue.fc2_use_tma else 'ublk' if self.epilogue.fc2_use_ublk else 'stg'}_"
            f"fc2tmastages{self.epilogue.fc2_tma_stages}_"
            f"fc1epiflag{self.fc1_epi_flag_batch}_clusters{self.launch_cluster_count}_"
            f"routingflag{self.routing_flag_batch}_"
            f"combine{self.combine_format.name}_clamp{self.gate_up_clamp}_"
            f"swiglua{self.swiglu_alpha}_swiglub{self.swiglu_beta}_"
            # The SiTU betas are codegen-time constants, so they must key the compiled kernel.
            f"situ{self.situ_beta}x{self.situ_linear_beta}_"
            f"{'apply_topk_fc1' if self.apply_topk_at_fc1 else 'apply_topk_fc2'}_"
            f"{'inkernel_redg' if self.reduce_topk_in_kernel else 'separate_reduce'}"
            + ("_dlb_v1" if self.helper_expert_count > 0 else "")
            # Only the non-default value changes the name, so existing compiled kernels keep their keys.
            + ("_tanhsigmoid" if self.epilogue.use_tanh_sigmoid else "")
            # The tmp buffer is a third pointer operand, an ABI change: it keys the compiled kernel and its AOT symbol.
            + ("_tmpbuf_v1" if self.separate_tmp_buffer else "")
            # The ring size is baked into the pool layouts and the slot arithmetic.
            + (f"_ring{self.ring_token_blocks}_v1" if self.ring_token_blocks else "")
        )

    def aot_compile(self, out_path: Optional[str] = None, **_compile_kwargs):
        """Compile against fake (metadata-only) inputs; ``out_path=None`` returns the in-memory callable.

        The fake tensors define the runtime ABI, so each mirrors what the caller stages: the local expert tensors,
        the padded SF extents, and ``to_cute``'s ``mark_layout_dynamic`` (the stride-1 axis stays static, every
        other axis becomes a runtime SymInt). ``stride_order`` follows the input generator's ``memory_order``, where
        0 is the innermost axis. Occupancy is a construction-time knob (``launch_cluster_count``), so unlike the
        pre-``next`` kernel this entry accepts no occupancy argument and tolerates leftover caller kwargs.
        """
        import math

        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
        from cutlass.cute.typing import AddressSpace, sym_int64

        def fake_tensor(dtype, shape, stride_order, dynamic_axes, alignment):
            extents = tuple(
                sym_int64(divisibility=math.gcd(int(extent), 128)) if axis in dynamic_axes else int(extent)
                for axis, extent in enumerate(shape)
            )
            return make_fake_compact_tensor(dtype, extents, stride_order=stride_order, assumed_align=alignment)

        tokens = self.max_tokens
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        experts = self.expert_count
        sf_vec_size = self.sf_vec_size
        # Weight SF is atom-swizzled and opaque, so its extent is whatever the caller staged, not a kernel choice.
        fc1_weight_sf_columns = round_up(intermediate_gateup, 128) * round_up(hidden // sf_vec_size, 4)
        fc2_weight_sf_columns = round_up(hidden, 128) * round_up(intermediate_downproj // sf_vec_size, 4)
        # TopkReduce and the in-kernel REDG both take their output dtype from this tensor.
        output_dtype = cutlass.BFloat16

        fake_arguments = dict(
            # Swap-AB puts the weights on the MMA A operand, so activation uses the routing input dtype.
            activation=fake_tensor(self.local_routing.activation_dtype, (tokens, hidden), (1, 0), {0}, 16),
            activation_sf=fake_tensor(
                self.local_routing.activation_sf_dtype,
                (tokens, self.local_routing.activation_sf_hidden_padded),
                (1, 0),
                {0},
                16,
            ),
            topk_indices=fake_tensor(self.topk_index_dtype, (tokens, self.topk), (1, 0), {0}, 16),
            topk_scores=fake_tensor(cutlass.Float32, (tokens, self.topk), (1, 0), {0}, 4),
            fc1_weight=fake_tensor(self.a_dtype, (experts, hidden, intermediate_gateup), (2, 0, 1), {0, 2}, 16),
            fc1_weight_sf=fake_tensor(self.sf_dtype, (experts, fc1_weight_sf_columns), (1, 0), {0}, 16),
            fc2_weight=fake_tensor(self.a_dtype, (experts, intermediate_downproj, hidden), (2, 0, 1), {0, 2}, 16),
            fc2_weight_sf=fake_tensor(self.sf_dtype, (experts, fc2_weight_sf_columns), (1, 0), {0}, 16),
            output_activation=fake_tensor(output_dtype, (tokens, hidden), (1, 0), {0}, 16),
            # Opaque byte workspaces: a bare base pointer, like to_cute_ptr. The 128-byte promise matches the
            # alignment the registered regions assume, so the caller must hand over 128-byte-aligned bases.
            local_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            shared_workspace=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            stream=make_fake_stream(),
        )
        if self.separate_tmp_buffer:
            # Caller-owned tmp buffer: a bare base pointer with the same 128-byte promise as the workspaces.
            fake_arguments.update(tmp_buffer=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128))
        fake_arguments.update(
            fc1_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
            fc2_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
            fc1_norm_const=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
        )

        if self.helper_expert_count > 0:
            fake_arguments.update(
                hot_expert_weight_ready_flags=make_ptr(
                    cutlass.Uint64, 0, AddressSpace.gmem, assumed_align=8
                ),
                hot_expert_weight_ready_generation=cutlass.Uint64(1),
            )

        compiled = cute.compile[cute.EnableTVMFFI(True)](self, **fake_arguments)
        if out_path is None:
            return compiled
        compiled.export_to_c(out_path, function_name=self._aot_symbol_name(), export_only_tvm_ffi_symbols=True)
        return out_path

    def _aot_symbol_name(self) -> str:
        identity = hashlib.sha256(self.name().encode()).hexdigest()[:16]
        return f"{_aot_symbol_prefix}_{identity}"

    def load_compiled(self, path: str):
        from cutlass.cute.runtime import load_module

        function = getattr(load_module(path, enable_tvm_ffi=True), self._aot_symbol_name())
        return make_megamoe_aot_callable(function, self)

    def _build_mainloop_and_smem(
        self, problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc
    ) -> Tuple[BlockScaledSwapAbFc12Mainloop, SmemWorkspace]:
        smem_limit = utils.get_smem_capacity_in_bytes(self.architecture) // self.occupancy
        smem_workspace = SmemWorkspace()
        self.local_routing.register_smem_regions(smem_workspace)
        self.scheduler.register_smem_regions(smem_workspace)
        self.epilogue.register_smem_regions(smem_workspace)
        fixed_component_bytes = smem_workspace.estimate_total_bytes()
        alignment_shift_bytes = max(region.byte_alignment for region in smem_workspace.regions())
        mainloop_smem_budget_bytes = smem_limit - fixed_component_bytes - alignment_shift_bytes
        if mainloop_smem_budget_bytes <= 0:
            raise ValueError("Fixed kernel components leave no SMEM budget for the Mainloop.")

        mainloop_impl_desc = ImplDesc({**resolved_impl_desc, "mainloop_smem_budget_bytes": mainloop_smem_budget_bytes})
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
        local_workspace: cute.Pointer,  # local GMEM byte workspace: kernel state, persistent
        shared_workspace: cute.Pointer,  # shared GMEM byte workspace
        # Caller-owned per-launch buffer of get_tmp_buffer_bytes() bytes, 128-byte aligned: any content
        # at launch, nothing in it survives the launch, may be a different buffer on every launch.
        tmp_buffer: Optional[cute.Pointer],
        stream: cuda.CUstream,
        fc1_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc2_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc1_norm_const: Optional[cute.Tensor] = None,  # (local_experts,)
        hot_expert_weight_ready_flags: Optional[cute.Pointer] = None,  # uint64[weight_ready_source_count]
        hot_expert_weight_ready_generation: Optional[cutlass.Uint64] = None,
    ) -> None:
        """Launch router, fused MegaMoE compute, and optional TopK reduce."""

        helper_ready_bundle = (
            hot_expert_weight_ready_flags,
            hot_expert_weight_ready_generation,
        )
        if cutlass.const_expr(
            (self.helper_expert_count > 0 and not all(value is not None for value in helper_ready_bundle))
            or (self.helper_expert_count == 0 and any(value is not None for value in helper_ready_bundle))
        ):
            raise ValueError(
                "helper READY flags/generation must both be present iff helper_expert_count > 0"
            )
        if cutlass.const_expr(self.separate_tmp_buffer != (tmp_buffer is not None)):
            raise ValueError(
                "tmp_buffer must be a pointer iff the kernel keeps its per-launch data in the tmp buffer "
                f"(separate_tmp_buffer={self.separate_tmp_buffer})."
            )

        def rewrite_tensor_shape(tensor: cute.Tensor, shape: Tuple) -> cute.Tensor:
            return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=tensor.stride))

        if cutlass.const_expr(topk_indices.element_type is not self.topk_index_dtype):
            raise TypeError(f"topk_indices must be the declared {self.topk_index_dtype}, got {topk_indices.dtype}.")
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
        router_topk_scores = topk_scores if cutlass.const_expr(self.apply_topk_at_fc1) else None
        self.local_routing.launch_router(
            topk_indices,
            router_topk_scores,
            local_workspace,
            shared_workspace,
            self._device_workspace,
            stream,
            tmp_buffer=tmp_buffer,
        )

        self._device_workspace.assign_device_members(local_workspace, shared_workspace, tmp_buffer)
        activation_sf_pool = self.local_routing.fc1_activation_sf_tensor(self._device_workspace)
        expert_token_sizes = self.local_routing.expert_sizes(self._device_workspace)
        fc1_output = self._device_workspace.tensor(self.fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.fc1_output_sf_region)

        experts = self.expert_count
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        fc1_weight = rewrite_tensor_shape(fc1_weight, (experts, hidden, intermediate_gateup))
        fc2_weight = rewrite_tensor_shape(fc2_weight, (experts, intermediate_downproj, hidden))

        singleton = cutlass.Int32(1)
        token_rows = self.local_routing.fc1_pool_token_count

        fc1_a = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (cutlass.Int32(intermediate_gateup), cutlass.Int32(hidden), experts),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
            ),
        )
        fc1_b = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (activation.shape[0], cutlass.Int32(hidden), singleton),
                stride=(activation.stride[0], activation.stride[1], 0),
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
        padded_token_rows = self.local_routing.fc1_pool_sf_token_count
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
            isinstance(fc1_weight_sf.shape[1], int) and isinstance(expected_fc1_weight_sf_columns, int)
        ):
            if cutlass.const_expr(fc1_weight_sf.shape[1] != expected_fc1_weight_sf_columns):
                raise ValueError("fc1_weight_sf has an incompatible column count.")
        fc1_sfa = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (cutlass.Int32(padded_intermediate_gateup), cutlass.Int32(padded_hidden), cutlass.Int32(experts)),
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
            isinstance(fc2_weight_sf.shape[1], int) and isinstance(expected_fc2_weight_sf_columns, int)
        ):
            if cutlass.const_expr(fc2_weight_sf.shape[1] != expected_fc2_weight_sf_columns):
                raise ValueError("fc2_weight_sf has an incompatible column count.")
        fc2_sfa = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (cutlass.Int32(padded_fc2_hidden), cutlass.Int32(padded_fc2_intermediate), cutlass.Int32(experts_fc2)),
                sf_vec_size,
            ),
        )
        fc2_sfb = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (fc1_output_sf.shape[0], cutlass.Int32(fc1_output_sf.shape[1] * sf_vec_size), singleton), sf_vec_size
            ),
        )
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            pre_reduced_activation = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (self.max_tokens, 1, hidden),
                    stride=(output_activation.stride[0], output_activation.stride[0], output_activation.stride[1]),
                ),
            )
        else:
            pre_reduced_activation = self.local_routing.pre_reduced_activation_tensor(self._device_workspace)

        fc2_output = rewrite_tensor_shape(
            pre_reduced_activation, (pre_reduced_activation.shape[0], pre_reduced_activation.shape[1], hidden)
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
            fc1_tma_b_tensor,
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

        fc1_output_tma_atom, fc1_output_tma_tensor = self.epilogue.prepare_tma_store_params(fc1_output_gemm)

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
            fc1_tma_b_tensor,
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
            activation_sf,
            pre_reduced_activation,
            expert_token_sizes,
            local_workspace,
            shared_workspace,
            tmp_buffer,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            hot_expert_weight_ready_flags,
            hot_expert_weight_ready_generation,
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
            self._topk_reduce(pre_reduced_activation, None, output_activation, reduce_scores, stream)

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
        activation_sf: cute.Tensor,
        pre_reduced_activation: cute.Tensor,
        expert_token_sizes: cute.Tensor,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        tmp_buffer: Optional[cute.Pointer],
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        hot_expert_weight_ready_flags: Optional[cute.Pointer],
        hot_expert_weight_ready_generation: Optional[cutlass.Uint64],
    ):
        """Compose local routing, scheduler, mainloop, and epilogue."""
        self._mainloop.materialize_codegen_members()
        storage_type = self._smem_workspace.storage_class()
        smem_allocator = utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()
        self._device_workspace.assign_device_members(local_workspace, shared_workspace, tmp_buffer)
        fc1_done_counter = self._device_workspace.tensor(self.fc1_done_counter_region)
        ring_retire_counter = None
        if cutlass.const_expr(self.ring_token_blocks > 0):
            ring_retire_counter = self._device_workspace.tensor(self.ring_retire_region)
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
                (fc1_output_sf_storage.shape[0], intermediate_gateup // 2, cutlass.Int32(1)), self.epilogue.sf_vec_size
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
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        linear_cta_idx = cta_rank_in_cluster + block_idx[2] * Int32(cluster_size)
        local_routing_args = LocalRoutingArgs(
            activation_sf=activation_sf,
        )
        self.local_routing.assign_device_members(
            device_workspace=self._device_workspace,
            local_routing_args=local_routing_args,
            linear_cta_idx=linear_cta_idx,
        )
        token_src_metadata = self.local_routing.token_src_metadata_tensor(self._device_workspace)
        pool_topk_scores = self.local_routing.fc1_topk_scores_tensor(self._device_workspace)

        ab_pipeline = self._mainloop.create_ab_pipeline(self._smem_workspace, smem_base)
        tma_a_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_ab_pipeline_stages
        )
        tma_b_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_ab_pipeline_stages
        )
        gather_b_pipeline_state = pipeline.make_pipeline_state(
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
        )
        scheduler = self.scheduler

        fc2_spin_threshold = (intermediate_gateup + self._mainloop.cta_tile_m - 1) // self._mainloop.cta_tile_m
        helper_weight_ready_gate = None
        if cutlass.const_expr(self.helper_expert_count > 0):
            helper_weight_ready_gate = make_helper_weight_ready_generation_gate(
                memory_slot_count=self.expert_count,
                helper_count=self.helper_expert_count,
                source_count=self.weight_ready_source_count,
                terminal_flags=hot_expert_weight_ready_flags,
                expected_generation=hot_expert_weight_ready_generation,
            )
        kernel_extension = BlockScaledSwapAbFc12Extension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=self.local_routing.fc1_ready_counter_pointer(self._device_workspace),
            # Ring workspace: tiles are published in block form. The gather-B warps and the FC2 epilogue read
            # token_src_metadata at the tile's data row, which stays the absolute pool row.
            ring_token_blocks=self.ring_token_blocks,
            ring_data_rows_per_block=self.local_routing.tokens_per_fc1_ready_slot if self.ring_token_blocks else 0,
            ring_sf_rows_per_block=self.local_routing.ring_sf_rows_per_block if self.ring_token_blocks else 0,
        )
        optional_epilogue_args = GatedActEpilogueArgs(
            fc1_alpha=fc1_alpha, fc2_alpha=fc2_alpha, fc1_norm_const=fc1_norm_const, topk_scores=pool_topk_scores
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        self._mainloop.assign_device_members(
            self._smem_workspace, smem_base, cta_coord_in_cluster, hidden, intermediate_gateup
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.scheduler_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.scheduler_register_count)
            iket.range_push("local_mega.scheduler")
            iket.range_push("scheduler.wait_sizes_ready")
            self.local_routing.wait_for_sizes_ready(self._device_workspace)
            iket.range_pop()
            iket.range_push("scheduler.gen_work")
            work_tile = scheduler.gen_next_work()
            iket.range_pop()
            helper_ready_latched = Int32(0)
            while work_tile.is_valid_tile:
                iket.range_push("scheduler.publish_work")
                prepared_tile = kernel_extension.prepare_work_tile(work_tile)
                if cutlass.const_expr(helper_weight_ready_gate is not None):
                    helper_ready_latched = helper_weight_ready_gate.wait_before_first_helper_publish(
                        prepared_tile.expert_idx, helper_ready_latched
                    )
                scheduler.publish_work(prepared_tile)
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
            iket.range_push("local_mega.tma_a")
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
            iket.range_push("local_mega.tma_b")
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_b(
                fc1_tma_b_tensor=fc1_tma_b_tensor,
                fc1_tma_b_atom=fc1_tma_b_atom,
                fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
                fc1_tma_sfb_atom=fc1_tma_sfb_atom,
                fc2_tma_b_tensor=fc2_tma_b_tensor,
                fc2_tma_b_atom=fc2_tma_b_atom,
                fc2_tma_sfb_tensor=fc2_tma_sfb_tensor,
                fc2_tma_sfb_atom=fc2_tma_sfb_atom,
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
            iket.range_push("local_mega.mma")
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
            iket.range_push("local_mega.epilogue")
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
                fc2_output,
                fc1_done_counter,
                thread_idx,
                token_src_metadata,
                optional_epilogue_args,
                ring_retire_counter=ring_retire_counter,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(tmem_allocator.retrieve_ptr(self.acc_dtype), self._mainloop.num_tmem_alloc_cols)
            iket.range_pop()

        if warp_idx >= Int32(self.transfer_warp_idx_start):
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            if warp_idx < Int32(self.token_in_end_warp_idx):
                iket.range_push("local_mega.token_in")
                self.local_routing.token_in(self._smem_workspace, smem_base)
                iket.range_pop()

        if (warp_idx >= Int32(self.gather_b_warp_ids[0])) & (
            warp_idx <= Int32(self.gather_b_warp_ids[-1])
        ):
            iket.range_push("local_mega.gather_b1")
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_gather_b1(
                fc1_tma_b_atom=fc1_tma_b_atom,
                token_src_metadata=token_src_metadata,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=gather_b_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                gather_warp_idx=warp_idx - Int32(self.gather_b_warp_ids[0]),
            )
            iket.range_pop()

        iket.range_push("local_mega.kernel_tail_wait")
        cute.arch.sync_threads()
        iket.range_pop()
        if (warp_idx >= Int32(self.transfer_warp_idx_start)) & (warp_idx < Int32(self.token_in_end_warp_idx)):
            iket.range_push("local_mega.tail_reset")
            self.local_routing.reset_tail()
            iket.range_pop()
        self.local_routing.remove_device_members()
        self._device_workspace.remove_device_members()
