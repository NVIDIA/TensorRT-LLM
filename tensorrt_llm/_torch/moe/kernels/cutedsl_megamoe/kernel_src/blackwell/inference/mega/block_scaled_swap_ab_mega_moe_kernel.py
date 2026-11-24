"""Fused pull-dispatch MegaMoE kernel composition for Blackwell."""

import hashlib
import os

from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Boolean, Int32, Int64

from .....api import ImplDesc, KernelClass, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from .....communication.nvlink_domain.one_round_router import _OneRoundRouter
from .....communication.nvlink_domain.token_comm import TokenCommArgs, TokenCommNonDeterministic
from .....communication.nvlink_domain.token_comm_dedup import TokenCommDedup
from .....communication.token_protocol import TokenSrcMetadata
from .....helpers.cute_py_helpers import Tcgen05MmaInstruction, make_tcgen05_tmem_plan, tcgen05_block_scaled_acc_dtype
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dlb_expert_count import resolve_dlb_expert_counts
from .....helpers.fc2_ready_geometry import derive_fc2_ready_geometry
from .....helpers.iket_compat import iket
from .....helpers.megamoe_aot import make_megamoe_aot_callable
from .....helpers.route_reset import ClearInvalidRoutes
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.topk_reduce import REDUCE_OPERAND_ALIGN_BYTES, TopkReduce, check_reduce_operand_alignment
from .....helpers.topk_reduce_dedup import TopkReduceDedup
from .....helpers.utils import ceil_div, ring_token_blocks_in_effect, round_up
from .....helpers.helper_weight_ready_gate import (
    derive_helper_weight_ready_topology,
    make_helper_weight_ready_generation_gate,
)
from .....quant_def import CombineFormat, QuantKind
from ....schedulers.base import WorkIdAcquisitionMode
from ....schedulers.fc12_mapping import BlockPhase
from ....schedulers.fc12_scheduler import (
    BlackwellFusedFc12Scheduler,
    PhaseInterleavedFc12Scheduler,
    minimum_phase_interleave_hint,
    minimum_phase_interleave_ring_token_blocks,
)
from ....schedulers.non_clc_mixed_cga import NonClcMixedCgaConfig
from ....schedulers.shared_fc12_scheduler import SharedExpertTileSchedule
from ...custom_mix_cga_helpers import TmaAtomOrPair, pipeline_init_arrive_mixed_cga, pipeline_init_wait_mixed_cga
from .block_scaled_swap_ab_fc12_epilogue import GatedActEpilogueArgs, SwapABGatedActEpilogue
from .block_scaled_swap_ab_fc12_epilogue_dedup import GatedActEpilogueArgsDedup, SwapABGatedActEpilogueDedup
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .block_scaled_swap_ab_fc12_mainloop import BlockScaledSwapAbFc12Mainloop

# ImplDesc keys of the internode component; a kernel with internode=False (the default) rejects them.
_INTERNODE_IMPL_KEYS = ("enable_nccl_gin", "relay_warp_count", "fc2_done_release_scope", "router_pack_cta_count", "phase_stamps")


def _internode_component():
    """The internode (GIN rail) component, imported only by a kernel that opts in: the NVLink-domain
    kernel never loads communication/internode or the NCCL device bindings it reaches."""
    from .....communication.internode.token_comm_internode import TokenCommInternodeGin

    return TokenCommInternodeGin


# One-round push dispatch: EP width from which it pays (EP8 measured slower) and the largest bucket.
_ONE_ROUND_MIN_WORLD_SIZE = 16
_ONE_ROUND_MAX_TOKENS_PER_RANK = 256

_aot_symbol_prefix = "blackwell_mega_moe_aot"


class BlockScaledSwapAbMegaMoeKernel(KernelClass):
    """Compose pull dispatch, persistent FC12, token-back, and reduction."""

    fc1_output_region: ClassVar[str] = "blackwell.mega_moe.fc1_output"
    fc1_output_sf_region: ClassVar[str] = "blackwell.mega_moe.fc1_output_sf"
    fc1_done_counter_region: ClassVar[str] = "blackwell.mega_moe.fc1_done_counter"
    # Ring workspace only: the region of the retire counter, one element per pool token block. The last reader of
    # a block adds to it: the MMA warp for every FC2 MMA tile, or the combine warps for every row they pushed out
    # of the local FC2 output.
    ring_retire_region: ClassVar[str] = "blackwell.mega_moe.ring_retire_count"
    topk_reduce_counter_region: ClassVar[str] = "blackwell.mega_moe.topk_reduce_counters"
    shared_fc1_output_region: ClassVar[str] = "blackwell.mega_moe.shared_fc1_output"
    shared_fc1_output_sf_region: ClassVar[str] = "blackwell.mega_moe.shared_fc1_output_sf"
    shared_output_region: ClassVar[str] = "blackwell.mega_moe.shared_output"
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
    # PDL lets the kernel start while its dispatch router still runs; the op clears it for eager launches.
    use_pdl = True

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
            # "swiglu" (default) or "quick_geglu"; validated by the epilogue, keyed in name().
            "gate_activation": OptionalRequirement(str),
            "swiglu_alpha": OptionalRequirement(Optional[float]),
            "swiglu_beta": OptionalRequirement(Optional[float]),
            # sigmoid on tanh.approx for the SwiGLU and SiTU cores; validated by the epilogue, keyed in name().
            "use_tanh_sigmoid": OptionalRequirement(bool),
            "world_size": int,
            "topk": int,
            "topk_index_dtype": type,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
            # The FP8 shared expert fused as extra tiles: its intermediate size and SwiGLU clamp.
            "shared_intermediate_size": OptionalRequirement(Optional[int]),
            "shared_gate_up_clamp": OptionalRequirement(Optional[float]),
            # The FC1 activation was quantized from x / s_t with a per-token fp32 s_t; the kernel
            # multiplies s_t back before the gated activation. Optional so existing descriptors
            # stay valid; it keys the compiled kernel (workspace region + epilogue path).
            "fc1_activation_per_token_scale": OptionalRequirement(bool),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "schedule_policy": tuple,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "fc2_use_bulk": bool,
            "epi_async_release": OptionalRequirement(bool),
            "epi_flag_batches": tuple,
            "launch_cluster_count": int,
            "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
            "preferred_cluster_count": OptionalRequirement(Optional[int]),
            "fallback_cluster_count": OptionalRequirement(Optional[int]),
            "token_in_flag_batch": int,
            "token_back_mode": str,
            "reduce_topk_in_kernel": bool,
            "skip_topk_reduce": OptionalRequirement(bool),
            "helper_expert_count": OptionalRequirement(int),
            "total_helper_slots": OptionalRequirement(int),
            # Master switch of the internode code paths (False: main's NVLink-domain kernel), the
            # component selection it implies ("nvlink" / "internode_gin") and the internode
            # component's own keys; see communication/internode/token_comm_internode.py.
            "internode": OptionalRequirement(bool),
            "token_comm_kind": OptionalRequirement(str),
            "ranks_per_node": OptionalRequirement(int),
            "enable_nccl_gin": OptionalRequirement(bool),
            "relay_warp_count": OptionalRequirement(int),
            "fc2_done_release_scope": OptionalRequirement(str),
            "topk_reduce_persistent": OptionalRequirement(bool),
            # Combine-side token dedup (port of the Rubin path): one pushed row per (token, destination rank).
            "combine_dedup": OptionalRequirement(bool),
            # Combine dedup: epilogue warps join the combine after their last tile (default on under atomic_counter).
            "dedup_epilogue_tail_combine": OptionalRequirement(bool),
            # Combine dedup: copies per combine fill (default 1 for reuse_dispatch / hybrid, 2 for standalone).
            "dedup_copies_per_fill": OptionalRequirement(int),
            # Combine dedup: fc2_done poll backoff cap in ns (default 1000; MEGA_DEDUP_WAIT_SLEEP_MAX_NS overrides).
            "dedup_poll_max_sleep_ns": OptionalRequirement(int),
            "ring_token_blocks": OptionalRequirement(Optional[int]),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)
        problem_desc, impl_desc = resolve_dlb_expert_counts(
            problem_desc, impl_desc, world_size=problem_desc["world_size"]
        )
        # The Blackwell epilogue publishes expert counters; it counts per token tile only with the local FC2
        # output in the ring workspace, where the kernel asks for that itself.
        if impl_desc.get("token_back_ready_granularity", "expert") != "expert":
            raise ValueError("Blackwell requires token_back_ready_granularity='expert'.")

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
        self.gate_activation = problem_desc.get("gate_activation") or "swiglu"
        self.swiglu_alpha = problem_desc.get("swiglu_alpha")
        self.swiglu_beta = problem_desc.get("swiglu_beta")
        self.world_size = problem_desc["world_size"]
        self.topk = problem_desc["topk"]
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]
        self.fc1_activation_per_token_scale = bool(problem_desc.get("fc1_activation_per_token_scale", False))
        self.shared_intermediate_size = problem_desc.get("shared_intermediate_size")
        self.shared_gate_up_clamp = problem_desc.get("shared_gate_up_clamp")
        self.fuse_shared = self.shared_intermediate_size is not None
        # Set by _init_shared_expert once the scheduler is known.
        self.shared_fc2_claims = False
        # One-round push dispatch (``_OneRoundRouter``) for buckets of at most 256 tokens per rank at EP16 and
        # wider replaces the size broadcast, the metadata push and the token-in pull with one push round. FC1 reads
        # the same bytes. At EP8 the pushes cost more than the round trips they save. Its router does not carry
        # per-token activation scales. It is measured on the DeepSeek-V4 MoE only, the one that applies the top-k
        # weights at FC1.
        self.one_round_dispatch = (
            _ONE_ROUND_MIN_WORLD_SIZE <= self.world_size <= _OneRoundRouter.max_world_size
            and self.max_tokens_per_rank <= _ONE_ROUND_MAX_TOKENS_PER_RANK
            and not self.fc1_activation_per_token_scale
            and self.apply_topk_at_fc1
        )
        # The sorter waits on one counter per (source rank, source CTA), one thread each (512 per CTA).
        self.one_round_cta_count = max(1, min(32, 512 // max(self.world_size, 1)))

        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
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
        # Ring workspace: the ring_token_blocks the caller asked for (None or 0: none). The ring in effect is
        # self.ring_token_blocks, resolved below once the geometry is validated.
        self.requested_ring_token_blocks = impl_desc.get("ring_token_blocks")
        # Read by the router, which takes it from the caller's descriptor. Kept here for the check of a requested
        # ring, which applies the router's limits with the settings the router gets.
        self.comm_stage_depths = tuple(impl_desc.get("comm_stage_depths", (1, 1)))
        self.skip_topk_reduce = impl_desc.get("skip_topk_reduce", False)
        self.helper_expert_count = impl_desc.get("helper_expert_count", 0)
        if type(self.helper_expert_count) is not int or self.helper_expert_count < 0:
            raise ValueError("helper_expert_count must be a non-negative exact int")
        # Opt-in persistent scheduler for the separate top-k reduce.
        self.topk_reduce_persistent = bool(
            impl_desc.get("topk_reduce_persistent") and not self.reduce_topk_in_kernel
        )
        self.combine_dedup = bool(impl_desc.get("combine_dedup", False))
        requested_tail = impl_desc.get("dedup_epilogue_tail_combine")
        self.dedup_epilogue_tail_combine = self.combine_dedup and (
            impl_desc["work_id_mode"] == "atomic_counter" if requested_tail is None else bool(requested_tail)
        )
        requested_copies_per_fill = impl_desc.get("dedup_copies_per_fill")
        self.dedup_copies_per_fill = (
            int(requested_copies_per_fill)
            if requested_copies_per_fill is not None
            else 1 if self.token_back_mode in ("reuse_dispatch_warps", "hybrid_warps") else 2
        )
        requested_poll_cap = impl_desc.get("dedup_poll_max_sleep_ns")
        self.dedup_poll_max_sleep_ns = int(
            requested_poll_cap
            if requested_poll_cap is not None
            else os.environ.get("MEGA_DEDUP_WAIT_SLEEP_MAX_NS", self.default_dedup_poll_max_sleep_ns)
        )
        # Diagnostic only (MEGA_DIAG_SKIP_TOKEN_BACK=1): the combine warps do nothing and the output is wrong.
        self._diag_skip_token_back = os.environ.get("MEGA_DIAG_SKIP_TOKEN_BACK", "0") == "1"
        if self.combine_dedup:
            # The one-round router and tail parity are TRT-LLM-local and exist only in the NVLink token comm, not in
            # the dedup one; the fused shared expert is not wired into the dedup epilogue.
            self.one_round_dispatch = False
            if self.fuse_shared:
                raise ValueError("The fused shared expert does not support combine_dedup.")
        self.tail_parity = not self.combine_dedup

        self.occupancy = 1
        self.architecture = "sm_100"
        self.local_expert_count = self.expert_count // self.world_size
        if self.helper_expert_count > 0:
            helper_topology = derive_helper_weight_ready_topology(
                self.local_expert_count,
                self.helper_expert_count,
            )
            self.home_expert_count = helper_topology.home_expert_count
        else:
            # Preserve the baseline's runtime-expert-count specialization.  A
            # static M is required only when an S>0 helper boundary must be
            # compiled into the scheduler gate.
            self.home_expert_count = self.local_expert_count
        # Master switch of every internode (multi-node) code path. False (the default) is the
        # NVLink-domain kernel of main: the nvlink component, one logical node, no GIN handles, no
        # relay warps, no component token in the compile-cache name; the internode keys must be absent. True selects
        # the internode_gin component (communication/internode/).
        self.internode = bool(impl_desc.get("internode", False))
        self.token_comm_kind = impl_desc.get("token_comm_kind", "internode_gin" if self.internode else "nvlink")
        if self.token_comm_kind not in ("nvlink", "internode_gin"):
            raise ValueError(f"token_comm_kind must be 'nvlink' or 'internode_gin', got {self.token_comm_kind!r}.")
        if self.internode != (self.token_comm_kind == "internode_gin"):  # nosec B105
            raise ValueError(
                "internode=True selects token_comm_kind='internode_gin' and internode=False (the default) "
                f"'nvlink'; got internode={self.internode}, token_comm_kind={self.token_comm_kind!r}."
            )
        if self.internode:
            # Not supported in this copy: with more than one node the relay weights are zero when top-k
            # is applied after FC2, reuse_dispatch_warps and the in-kernel reduce drop the remote
            # partials, and the tail reset can race late fc2_done mirror increments.
            raise NotImplementedError("internode=True is not supported by this MegaMoE kernel package.")
        if not self.internode:
            stray = [key for key in _INTERNODE_IMPL_KEYS if key in impl_desc]
            if impl_desc.get("ranks_per_node", self.world_size) != self.world_size:
                stray.append("ranks_per_node")
            if stray:
                raise ValueError(
                    "internode=False compiles every internode code path out; the ImplDesc must not carry "
                    f"{stray} (ranks_per_node, when given, must equal the world size)."
                )

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
            self.cluster_shape_mn if self.fallback_cluster_shape_mn is None else self.fallback_cluster_shape_mn
        )
        if self.launch_cluster_count != self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred:
            raise ValueError(
                "launch_cluster_count must equal the canonical preferred-cluster slot count resolved from "
                "preferred_cluster_count and fallback_cluster_count."
            )

        self._validate_geometry()
        # Ring workspace (0: off). The per-launch pools on the way of a routed token are sized for
        # ring_token_blocks token blocks instead of the worst-case routed tokens: the dispatched-token pool, the
        # FC1 output and, where the kernel keeps the FC2 result in a local FC2 output for combine warps of its
        # own, that output (ring_covers_fc2_output). Pool token block b lives in slot b % ring_token_blocks of
        # each and keeps it until the block is retired, that is until its last reader is done (the slot lease).
        # token_in alone waits for the ring: it writes a block once the block that held its slot before is
        # retired. A requested ring that would not be smaller than the worst case is not in effect: the kernel
        # is then the kernel without the key, in everything it sizes, generates and names.
        self.ring_token_blocks = ring_token_blocks_in_effect(
            self.requested_ring_token_blocks,
            tokens_per_block=self.tokens_per_token_block,
            source_tokens=self.world_size * self.max_tokens_per_rank,
            local_experts=self.local_expert_count,
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
        # The ring in effect takes two tuning values over, and name() shows what is in effect. Everything else
        # that conflicts with a ring is refused (_validate_ring_workspace).
        if self.ring_token_blocks:
            # A held FC1 completion flag could be the very one a ring slot waits for: publish each tile at once.
            # The prologue of a caller who left the hint to the kernel is then the smallest safe one, not that
            # one rounded up to the caller's batch; a hint the caller gave stays the caller's.
            self.epi_flag_batches = (1, self.epi_flag_batches[1])
            self.hint = self._ring_schedule()["hint"]
        if self.ring_covers_fc2_output:
            # Likewise for FC2: a held completion could be the one the combine warps need to push, and with
            # that retire, the block a ring slot waits for.
            self.epi_flag_batches = (self.epi_flag_batches[0], 1)
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=(self.mma_tiler_mnk[0], self.mma_tiler_mnk[1], self.quant_kind.instruction_k("1x")),
            participates=mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        tmem_plan = make_tcgen05_tmem_plan(mma_instruction, self.architecture, self.mma_tiler_mnk)
        tokens_per_fc1_ready_slot = self.tokens_per_token_block
        hidden_per_fc2_cluster_tile = self.mma_tiler_mnk[0] // mma_cta_count * self.cluster_shape_mn[0]
        fc2_done_signals_per_token_tile = (
            ceil_div(self.hidden_size, hidden_per_fc2_cluster_tile)
            * self.cluster_shape_mn[0]
            * self.cluster_shape_mn[1]
        )
        if self.ring_covers_fc2_output:
            # A local FC2 output in the ring workspace is read out block by block, so FC2 completion is counted
            # per token tile: every CTA tile of a block publishes once. The helper checks that a ready slot is
            # one token tile (hidden-only clusters, complete MMA CTA groups).
            tokens_per_fc1_ready_slot, fc2_done_signals_per_token_tile = derive_fc2_ready_geometry(
                hidden_size=self.hidden_size,
                mma_tiler_mnk=self.mma_tiler_mnk,
                cluster_shape_mn=self.cluster_shape_mn,
                use_2cta_instrs=self.use_2cta_instrs,
                fallback_cluster_shape_mn=self.fallback_cluster_shape_mn,
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
                "token_back_schedule_mode": self.token_back_schedule_mode,
                "tail_parity": self.tail_parity,
                "one_round_dispatch": self.one_round_dispatch,
                "one_round_cta_count": self.one_round_cta_count,
                # sm_100 / sm_103 have no fma.f32x2.bf16x2: the dedup combine reduce widens bf16 first.
                **(
                    {
                        "dedup_mixed_precision_fma": False,
                        "dedup_epilogue_tail_combine": self.dedup_epilogue_tail_combine,
                        "dedup_copies_per_fill": self.dedup_copies_per_fill,
                        "dedup_poll_max_sleep_ns": self.dedup_poll_max_sleep_ns,
                    }
                    if self.combine_dedup
                    else {}
                ),
                "token_back_ready_granularity": "token_tile" if self.ring_covers_fc2_output else "expert",
                **self._ring_lease_impl_fields(tokens_per_fc1_ready_slot),
            }
        )
        # One assignment on purpose: tests/test_dlb_contracts.py executes the constructor's
        # self.token_comm binding in isolation, so the class choice cannot live in a local.
        # combine_dedup with internode is rejected by _validate_geometry.
        self.token_comm = (
            TokenCommDedup
            if self.combine_dedup
            else _internode_component() if self.internode else TokenCommNonDeterministic
        )(problem_desc, token_comm_impl_desc)
        # CTA layout: 12 warps, or 16 when warps 12-15 host standalone token-back or the
        # component's relay group. Both wide layouts use the standalone register budget
        # (4 x 256 + 12 x 64 registers per thread-slot fits the 64K register file).
        self.relay_warp_count = self.token_comm.relay_warp_count
        if self.relay_warp_count not in (0, 4):
            raise ValueError(f"relay_warp_count must be 0 or 4, got {self.relay_warp_count}.")
        if self.relay_warp_count > 0 and self.token_back_mode in ("standalone_warps", "hybrid_warps"):  # nosec B105
            raise ValueError(f"token_back_mode={self.token_back_mode!r} and a relay warp group both need warps 12-15.")
        if self.relay_warp_count > 0:
            # The internode combine gather also runs on the epilogue warps after their last tile.
            epilogue_workers = getattr(self.token_comm, "combine_epilogue_worker_warps", None)
            if epilogue_workers is not None and epilogue_workers != len(self.epilogue_warp_ids):
                raise ValueError(
                    "The component's combine_epilogue_worker_warps must equal the kernel's epilogue warp "
                    f"count ({len(self.epilogue_warp_ids)})."
                )
        # hybrid_warps (dedup only): the standalone combine warps 12-15 plus the dispatch warps after token_in.
        wide_layout = self.token_back_mode in ("standalone_warps", "hybrid_warps") or self.relay_warp_count > 0  # nosec B105
        self.threads_per_cta = 16 * 32 if wide_layout else 12 * 32
        self.other_warp_register_count = 64 if wide_layout else 72
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
                "epi_async_release": impl_desc.get("epi_async_release", True),
                "fc2_epi_flag_batch": self.epi_flag_batches[1],
                # With the local FC2 output in the ring workspace an FC2 tile stores into the ring slot of its
                # block and publishes its completion per block.
                "fc2_output_ring_blocks": self.ring_token_blocks if self.ring_covers_fc2_output else 0,
                "fc2_done_release_scope": self.token_comm.fc2_done_release_scope,
                # Optional int: absent when the component has no relay records (validation rejects None).
                **(
                    {"relay_record_topk_marker": self.token_comm.relay_record_topk_marker}
                    if self.token_comm.relay_record_topk_marker is not None
                    else {}
                ),
                "publish_fc2_done": self.token_comm.publishes_fc2_done,
                # Combine dedup: the FC2 epilogue scatters rows into TokenCommDedup's order buffer.
                **(
                    {
                        "dedup_world_size": self.world_size,
                        "dedup_topk": self.topk,
                        "dedup_chunk_bytes": self.token_comm.order_chunk_bytes,
                        "dedup_chunks_per_row": self.token_comm.order_chunks_per_row,
                    }
                    if self.combine_dedup
                    else {}
                ),
            }
        )
        scheduler_type = (
            BlackwellFusedFc12Scheduler if self.schedule_mode == "grouped" else PhaseInterleavedFc12Scheduler
        )
        self.scheduler = scheduler_type(fc12_problem_desc, resolved_impl_desc)
        if self.combine_dedup:
            self.epilogue = SwapABGatedActEpilogueDedup(fc12_problem_desc, resolved_impl_desc)
        else:
            self.epilogue = SwapABGatedActEpilogue(fc12_problem_desc, resolved_impl_desc)
        if self.fuse_shared:
            self._init_shared_expert()
        if self.ring_covers_fc2_output:
            self._validate_token_tile_ready_contract()
        if self.epilogue.fc1_output_dtype is not self.b_dtype:
            raise ValueError("Epilogue FC1 output dtype must match the Mainloop B dtype.")
        if self.epilogue.fc1_output_sf_dtype is not self.sf_dtype:
            raise ValueError("Epilogue FC1 output scale dtype must match the Mainloop scale dtype.")
        if self.epilogue.sf_vec_size != self.sf_vec_size:
            raise ValueError("Epilogue and Mainloop scale vector sizes must match.")
        self._device_workspace = self._build_device_workspace()
        self._mainloop, self._smem_workspace = self._build_mainloop_and_smem(fc12_problem_desc, resolved_impl_desc)
        if self.dedup_epilogue_tail_combine:
            self._check_epilogue_tail_ring()
        self.masks_invalid_combine_slots = not self.reduce_topk_in_kernel and not self.internode
        # With relay tokens the reduce adds one returned BF16 node partial per remote node (I11).
        internode_reduce = (
            (self.local_expert_count, self.token_comm.ranks_per_node, self.token_comm.relay_slot_count)
            if self.token_comm.relay_slot_count > 0
            else None
        )
        self._topk_reduce = (
            None
            if self.reduce_topk_in_kernel or self.skip_topk_reduce
            # Dedup: weights are folded at FC1; sum the rank slots named by the valid-rank mask.
            else TopkReduceDedup(self.hidden_size, self.world_size, self.combine_format)
            if self.combine_dedup
            else TopkReduce(
                self.hidden_size,
                self.topk,
                self.combine_format,
                num_experts=self.expert_count,
                internode=internode_reduce,
                persistent=self.topk_reduce_persistent,
            )
        )
        self._route_reset = ClearInvalidRoutes(
            self.max_tokens_per_rank, self.topk, self.expert_count,
            self.hidden_size * self.combine_format.act_dtype.width // 8,
            self.token_comm.combine_sf_hidden_padded * self.combine_format.scale_dtype.width // 8
            if self.combine_format.is_quantized else 0,
        )

    # Kernel-argument order of the shared expert's TMA tensors, atoms and live token count.
    _shared_param_keys: ClassVar[Tuple[str, ...]] = (
        "fc1_a_tensor",
        "fc1_a_atom",
        "fc1_sfa_tensor",
        "fc1_sfa_atom",
        "fc2_a_tensor",
        "fc2_a_atom",
        "fc2_sfa_tensor",
        "fc2_sfa_atom",
        "fc1_b_tensor",
        "fc1_b_atom_or_pair",
        "fc1_sfb_tensor",
        "fc1_sfb_atom_or_pair",
        "fc2_b_tensor",
        "fc2_b_atom_or_pair",
        "fc2_sfb_tensor",
        "fc2_sfb_atom_or_pair",
        "fc1_output_tma_atom",
        "fc1_output_tma_tensor",
        "num_tokens",
    )

    def _init_shared_expert(self) -> None:
        """Plan the shared expert's tiles; its output joins the top-k reduction."""
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        cta_tile_m = self.mma_tiler_mnk[0] // mma_cta_count
        cluster_tile_m = cta_tile_m * self.cluster_shape_mn[0]
        shared_gateup = 2 * self.shared_intermediate_size
        # The down projection waits for a count of gate/up tiles, so no tile may be partial.
        if shared_gateup % cluster_tile_m or self.hidden_size % cluster_tile_m:
            raise ValueError(
                f"The shared expert's projections ({shared_gateup}, {self.hidden_size}) must be "
                f"multiples of the cluster's {cluster_tile_m} rows."
            )
        self._shared_schedule = SharedExpertTileSchedule(
            fc1_output_blocks=ceil_div(shared_gateup, cluster_tile_m),
            fc2_output_blocks=ceil_div(self.hidden_size, cluster_tile_m),
            tokens_per_tile=self.mma_tiler_mnk[1],
            cluster_m=self.cluster_shape_mn[0],
            slot_count=self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred,
            max_tokens=self.max_tokens_per_rank,
        )
        # Every in-bound gate/up CTA tile of a token block signals once.
        self.shared_fc2_spin_threshold = ceil_div(shared_gateup, cta_tile_m)
        self.shared_token_rows = round_up(self.max_tokens_per_rank, 128)
        # The phase-interleaved FC2 claim stream continues into the down-projection tiles, so the clusters that
        # finish their routed work first take more of them, instead of a static share per cluster slot.
        self.shared_fc2_claims = isinstance(self.scheduler, PhaseInterleavedFc12Scheduler)
        if self.shared_fc2_claims:
            self.scheduler.claim_shared_fc2_tiles(
                output_blocks=ceil_div(self.hidden_size, cluster_tile_m),
                tokens_per_tile=self.mma_tiler_mnk[1],
                cluster_m=self.cluster_shape_mn[0],
            )
        # The static share's slots without gate/up tiles run their down-projection tiles in the dispatch window, and
        # those wait on gate/up tiles whose CTAs only get their next tile once the router is done.
        self.epilogue.shared_fc1_signal_early = not self.shared_fc2_claims

    def _validate_geometry(self) -> None:
        if self.skip_topk_reduce and self.internode:
            raise ValueError("skip_topk_reduce does not expose internode partials.")
        if self.skip_topk_reduce and self.topk_reduce_persistent:
            raise ValueError("skip_topk_reduce and topk_reduce_persistent are mutually exclusive.")
        if self.skip_topk_reduce and (self.reduce_topk_in_kernel or self.combine_format.is_quantized):
            raise ValueError("skip_topk_reduce requires BF16 combine and reduce_topk_in_kernel=False.")
        static_expert_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_expert_dimensions) and not all(static_expert_dimensions):
            raise ValueError("MegaMoE expert dimensions must be either all static or all runtime.")
        if not all(static_expert_dimensions):
            raise NotImplementedError("The MegaMoE Kernel currently requires static expert dimensions.")
        if self.expert_count <= 0 or self.intermediate_gateup_size <= 0 or self.hidden_size <= 0:
            raise ValueError("MegaMoE expert dimensions must be positive.")
        if self.world_size <= 0 or self.expert_count % self.world_size != 0:
            raise ValueError("expert_count must be divisible by a positive world_size.")
        if self.topk <= 0 or self.topk > self.expert_count:
            raise ValueError("topk must be positive and no greater than expert_count.")
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}.")
        if self.reduce_topk_in_kernel and not self.apply_topk_at_fc1:
            raise ValueError("In-kernel top-k reduction requires apply_topk_at_fc1=True.")
        if self.token_back_mode == "hybrid_warps" and not self.combine_dedup:
            raise ValueError("token_back_mode='hybrid_warps' exists only in the combine-dedup token_back.")
        if self.combine_dedup:
            if self.topk > TokenSrcMetadata.MAX_TOPK or self.world_size > TokenSrcMetadata.MAX_RANKS:
                raise ValueError(
                    f"TokenSrcMetadata keeps topk and rank in {TokenSrcMetadata.FIELD_BITS} bits each; "
                    f"topk must be <= {TokenSrcMetadata.MAX_TOPK} and world_size <= {TokenSrcMetadata.MAX_RANKS}."
                )
            if self.max_tokens_per_rank > TokenSrcMetadata.MAX_TOKENS:
                raise ValueError(
                    f"TokenSrcMetadata keeps the token index in {TokenSrcMetadata.TOKEN_BITS} bits; "
                    f"max_tokens_per_rank must be <= {TokenSrcMetadata.MAX_TOKENS}."
                )
            if not self.apply_topk_at_fc1:
                raise ValueError("combine_dedup applies top-k scores at FC1 only (pre_fc1_quant).")
            if self.combine_format.is_quantized or self.combine_format.act_dtype is not cutlass.BFloat16:
                raise ValueError("combine_dedup supports the BF16 combine format only.")
            if self.world_size > 32:
                raise ValueError("combine_dedup keeps the destination-rank mask in an Int32; world_size must be <= 32.")
            if self.token_back_mode == "epi_warps":
                raise ValueError(
                    "combine_dedup pushes the reduced rank-major rows from the combine warps; use "
                    "reuse_dispatch_warps, standalone_warps or hybrid_warps."
                )
            if not self.fc2_use_bulk:
                raise ValueError("combine_dedup stages FC2 subtiles in SMEM for per-token bulk stores; fc2_use_bulk must be on.")
            if self.internode:
                raise ValueError("combine_dedup is an NVLink-domain path; internode must be off.")
            if self.skip_topk_reduce or self.topk_reduce_persistent:
                raise ValueError("combine_dedup has its own rank-slot reduce; skip_topk_reduce / topk_reduce_persistent must be off.")
            if self.fc1_activation_per_token_scale:
                raise ValueError("combine_dedup's router does not carry per-token FC1 activation scales.")
        if self.intermediate_gateup_size % 2 != 0:
            raise ValueError("The SwiGLU intermediate dimension must be even.")
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
        if self.hint is not None and (isinstance(self.hint, bool) or not isinstance(self.hint, int) or self.hint <= 0):
            raise ValueError(f"schedule_policy hint must be a positive Python int, got {self.hint!r}.")
        if self.schedule_mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError("phase_interleave currently requires work_id_mode='atomic_counter'.")
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches must contain FC1 and FC2 batch sizes.")
        if any(batch < 1 or batch > 32 for batch in self.epi_flag_batches):
            raise ValueError("epi_flag_batches values must be in [1, 32].")
        if self.sf_vec_size <= 0:
            raise ValueError("sf_vec_size must be positive.")
        if self.quant_kind.needs_unpack_tma(self.architecture):
            # The fp4 weight reaches SMEM through the unpacking TMA, whose inner tensor dimension
            # must be a whole number of 64 B chunks, i.e. 128 fp4 elements. Both weights are
            # K-major, so the constrained extent is FC1's hidden and FC2's intermediate_downproj.
            # The latter is stricter than the block-scale rule already enforced below, which only
            # gets intermediate_downproj to a multiple of 64.
            for name, extent in (
                ("hidden_size", self.hidden_size),
                ("intermediate_downproj", self.intermediate_gateup_size // 2),
            ):
                if extent % 128 != 0:
                    raise ValueError(
                        f"{self.quant_kind} loads its fp4 weight through the unpacking TMA, which "
                        f"requires {name} to be a multiple of 128, got {extent}."
                    )
        mma_m, mma_n, mma_k = self.mma_tiler_mnk
        cluster_m, cluster_n = self.cluster_shape_mn
        expected_mma_m = 256 if self.use_2cta_instrs else 128
        if mma_m != expected_mma_m:
            raise ValueError(f"mma_tiler M must be {expected_mma_m}, got {mma_m}.")
        if mma_n not in (64, 128, 256):
            raise ValueError(f"mma_tiler N must be 64, 128, or 256, got {mma_n}.")
        if mma_k % (self.sf_vec_size * 4) != 0:
            raise ValueError("mma_tiler K must be divisible by four scale-factor vectors.")
        if self.intermediate_gateup_size % (self.sf_vec_size * 4) != 0:
            raise ValueError("The intermediate dimension must be divisible by four scale-factor vectors.")
        if cluster_n != 1:
            raise ValueError(f"The swap-AB FC12 path requires cluster N=1, got {cluster_n}.")
        if cluster_m <= 0 or cluster_m > 16 or cluster_m & (cluster_m - 1):
            raise ValueError("cluster M must be a power of two no greater than 16.")
        if self.use_2cta_instrs and cluster_m % 2 != 0:
            raise ValueError("Two-CTA MMA requires an even cluster M.")
        if self.fallback_cluster_shape_mn is not None:
            fallback_cluster_m, fallback_cluster_n = self.fallback_cluster_shape_mn
            if fallback_cluster_n != 1:
                raise ValueError(f"The swap-AB FC12 fallback path requires cluster N=1, got {fallback_cluster_n}.")
            if fallback_cluster_m <= 0 or fallback_cluster_m > 16 or fallback_cluster_m & (fallback_cluster_m - 1):
                raise ValueError("fallback cluster M must be a power of two no greater than 16.")
            if self.use_2cta_instrs and fallback_cluster_m % 2 != 0:
                raise ValueError("Two-CTA MMA requires an even fallback cluster M.")

    def _resolve_schedule_hint(self) -> None:
        """Resolve a concrete grouped budget or safe phase-interleave prologue."""
        canonical_cluster_count = self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        if self.schedule_mode != "phase_interleave":
            if self.hint is None:
                self.hint = canonical_cluster_count
            return

        mma_cta_count = 2 if self.use_2cta_instrs else 1
        cluster_feature_tile = self.mma_tiler_mnk[0] // mma_cta_count * self.cluster_shape_mn[0]
        raw_minimum_hint = minimum_phase_interleave_hint(
            blocks_fc1=ceil_div(self.intermediate_gateup_size, cluster_feature_tile),
            blocks_fc2=ceil_div(self.hidden_size, cluster_feature_tile),
            launch_cluster_cnt_merge_as_preferred=canonical_cluster_count,
        )
        # A safe prologue must end on an FC1 completion-batch boundary.
        minimum_safe_hint = round_up(raw_minimum_hint, self.epi_flag_batches[0])
        if self.hint is None:
            self.hint = minimum_safe_hint
        elif self.hint < minimum_safe_hint:
            raise ValueError(
                f"phase_interleave hint {self.hint} is unsafe with "
                f"fc1_epi_flag_batch={self.epi_flag_batches[0]}; "
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
        clusters = self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred
        schedule = dict(blocks_fc1=blocks_fc1, blocks_fc2=blocks_fc2, launch_cluster_cnt_merge_as_preferred=clusters)
        hint = self.schedule_policy[1]
        if hint is None:
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
        if self.tail_parity or self.one_round_dispatch or self.fuse_shared:
            # TRT-LLM local features: their workspace epochs and extra tiles were never checked with a ring.
            raise NotImplementedError(
                "The ring workspace does not support tail parity, the one-round router or the fused shared expert."
            )
        if self.schedule_mode != "phase_interleave":
            # grouped runs the FC1 tiles of a whole expert group before its FC2 tiles: every block of the
            # group is live at once, and a single expert can hold every routed token.
            raise ValueError("The ring workspace requires schedule_policy phase_interleave.")
        if self.internode:
            raise NotImplementedError("The ring workspace does not support the internode component.")
        if self.token_back_mode == "reuse_dispatch_warps":
            # Not supported, with combine_dedup or without. Without it the local FC2 output of this mode, the
            # largest pool, could never join the ring.
            raise NotImplementedError(
                "The ring workspace does not support token_back_mode reuse_dispatch_warps: its dispatch warps "
                "start pushing only after they have received every token, while the one wait of the ring is in "
                "token_in."
            )
        if self.ring_would_cover_fc2_output:
            # The ring has the local FC2 output in it: FC2 TMA stores put a tile into the slot of its block, and
            # the combine warps read the slots block by block.
            if not self.fc2_use_bulk:
                raise NotImplementedError(
                    "The ring workspace with standalone_warps requires fc2_use_bulk (FC2 TMA stores into the "
                    "local FC2 output)."
                )
            if self._diag_skip_token_back:
                raise ValueError(
                    "The ring workspace with standalone_warps cannot run under MEGA_DIAG_SKIP_TOKEN_BACK: its "
                    "combine warps would retire no block."
                )
        if not self.combine_dedup:
            # The limits of the router's own paths, in the router's one definition and words (the staged
            # token_in, and with the local FC2 output in the ring the multi-stage token-back and the static
            # token-back schedule). The combine_dedup router has no such path.
            TokenCommNonDeterministic.validate_ring_paths(
                dispatch_stage_depth=self.comm_stage_depths[0],
                combine_stage_depth=self.comm_stage_depths[1],
                token_back_schedule_mode=self.token_back_schedule_mode,
                token_back_mode=self.token_back_mode,
            )
        token_tile = self.tokens_per_token_block
        if round_up(token_tile, 128) % token_tile != 0:
            # A ring slot is one token tile of data rows and that tile rounded up to whole 128-row scale atoms;
            # the kernel extension derives the data row of a slot from its scale row (get_gmem_tensor).
            raise ValueError(
                f"The ring workspace needs a token tile that divides the scale rows of its slot, got {token_tile}."
            )
        # combine_dedup needs no check of its own: its router has the same single-stage token_in, with the ring
        # rows and the admission, and what it adds (the order buffer, the rank mask) is addressed by source
        # token and rank, never by a pool row. Nor do DLB helper experts: a helper slot is a physical expert
        # of the rank, numbered in the pool like every other, and the scheduler warp's wait for its weights
        # holds a claimed tile back without changing the claim order.
        minimum = self.minimum_ring_token_blocks()
        if ring_token_blocks < minimum:
            raise ValueError(
                f"ring_token_blocks {ring_token_blocks} cannot hold the FC1 lead of this schedule "
                f"(hint {self._ring_schedule()['hint']}, "
                f"{self.mixed_cga_config.launch_cluster_cnt_merge_as_preferred} clusters); the minimum is {minimum}."
            )

    def _validate_token_tile_ready_contract(self) -> None:
        """Check the FC2 completion count per token tile against the built components.

        With the local FC2 output in the ring the combine warps read a block once its counter shows
        fc2_done_signals_per_token_tile, so the scheduler's logical CTA geometry, the epilogue's tiles and that
        target have to agree.
        """
        mapping_tile = self.scheduler.mapping_cta_tile_shape_mnk
        mapping_cluster = self.scheduler.mapping_cluster_shape_mn
        if (
            mapping_cluster != (1, self.cluster_shape_mn[0])
            or mapping_tile[:2] != (self.epilogue.cta_tile_n, self.epilogue.cta_tile_m)
        ):
            raise ValueError(
                "The ring workspace with standalone_warps needs the scheduler and the epilogue to share the logical "
                "CTA geometry."
            )
        expected_signals = ceil_div(self.hidden_size, mapping_tile[1] * mapping_cluster[1]) * mapping_cluster[1]
        if (
            self.token_comm.tokens_per_fc1_ready_slot != mapping_tile[0]
            or self.token_comm.fc2_done_signals_per_token_tile != expected_signals
        ):
            raise ValueError(
                "The ring workspace with standalone_warps needs the combine warps and the FC2 epilogue to share "
                "ready slots and completion counts."
            )

    @property
    def token_back_schedule_mode(self) -> str:
        """The token-back schedule this kernel hands its router: it follows the work-id mode."""
        return "atomic_counter" if self.work_id_mode == "atomic_counter" else "static"

    @property
    def ring_would_cover_fc2_output(self) -> bool:
        """True when a ring of this kernel, of whatever size, has the local FC2 output in it: a property of the
        caller's configuration, with a ring in effect or without. The checks of a requested ring read this one.

        With token_back_mode standalone_warps and no combine_dedup the kernel keeps the FC2 result in a local
        FC2 output for combine warps of its own: that output is on the way of a block, and the last reader of
        the block is the combine. In every other mode a ring goes with, the last reader is the FC2 operand load.
        """
        return self.token_back_mode == "standalone_warps" and not self.combine_dedup

    @property
    def ring_covers_fc2_output(self) -> bool:
        """True when the local FC2 output is in the ring in effect: a ring of the same token blocks as the other
        pools. Everything that sizes a pool or generates code reads this one."""
        return bool(self.ring_token_blocks) and self.ring_would_cover_fc2_output

    def _ring_lease_impl_fields(self, tokens_per_block: int) -> dict:
        """token_comm's ring keys: the ring size, whether its local FC2 output is in the ring, the retire counter
        and the count at which a block is retired.

        The last reader of a block adds to the retire counter. Without the local FC2 output in the ring that is
        the FC2 operand load: the MMA warp adds one for every in-bound FC2 MMA tile of the block (the mainloop's
        run_mma). With it, it is the combine: the combine warps add the rows they have pushed out of that
        output, a whole token block of rows per block, and the MMA warp adds nothing.
        """
        retire_region, retire_threshold = None, None
        if self.ring_token_blocks:
            retire_region = self.ring_retire_region
            retire_threshold = ceil_div(self.hidden_size, self.mma_tiler_mnk[0])
            if self.ring_covers_fc2_output:
                retire_threshold = tokens_per_block
        return {
            "ring_token_blocks": self.ring_token_blocks,
            "ring_retire_region": retire_region,
            "ring_retire_threshold": retire_threshold,
        }

    @property
    def tokens_per_token_block(self) -> int:
        """Routed tokens of one token block: one token tile of one expert, the ready slot of the router."""
        return self.mma_tiler_mnk[1] * self.cluster_shape_mn[1]

    @property
    def separate_tmp_buffer(self) -> bool:
        """True: the per-launch data lives in the caller-owned tmp buffer (third pointer operand)."""
        # The internode component keeps main's single local buffer: its router data regions were
        # never validated on a recycled buffer.
        return not self.internode

    def _build_device_workspace(self) -> DeviceWorkspace:
        intermediate_downproj = self.intermediate_gateup_size // 2
        sf_column_count = round_up(ceil_div(intermediate_downproj, self.sf_vec_size), 4)
        # The FC1 output has the rows of the router's dispatch pool: the worst case, or the ring.
        max_sf_rows = self.token_comm.fc1_pool_sf_token_count
        counter_slot_count = self.token_comm.max_fc1_ready_slot_count

        device_workspace = DeviceWorkspace(separate_tmp=self.separate_tmp_buffer)
        if self.tail_parity:
            device_workspace.enable_tail_parity()
        device_workspace.register(
            self.fc1_output_region,
            self.epilogue.fc1_output_dtype,
            (self.token_comm.fc1_pool_token_count, intermediate_downproj),
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
        if self.topk_reduce_persistent:
            # Work cursor of the persistent top-k reduce: tail-reset like the
            # other per-launch counters, so it is zero before every reduce.
            device_workspace.register(
                self.topk_reduce_counter_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                byte_alignment=16,
                reset="tail_reset",
            )
        if self.fuse_shared:
            shared_sf_vec_size = QuantKind.mxfp8_e4m3.sf_vec_size
            device_workspace.register(
                self.shared_fc1_output_region,
                QuantKind.mxfp8_e4m3.activation_dtype,
                (self.shared_token_rows, self.shared_intermediate_size),
                buffer_space="local",
                mem_order=(1, 0),
                byte_alignment=128,
            )
            device_workspace.register(
                self.shared_fc1_output_sf_region,
                QuantKind.mxfp8_e4m3.sf_dtype,
                (
                    self.shared_token_rows,
                    round_up(ceil_div(self.shared_intermediate_size, shared_sf_vec_size), 4),
                ),
                buffer_space="local",
                mem_order=(1, 0),
                byte_alignment=128,
            )
            device_workspace.register(
                self.shared_output_region,
                cutlass.BFloat16,
                (self.shared_token_rows, self.hidden_size),
                buffer_space="local",
                mem_order=(1, 0),
                byte_alignment=128,
            )
            self._shared_schedule.register_device_workspace(device_workspace)
        self.scheduler.register_device_workspace(device_workspace)
        self.token_comm.register_device_workspace(device_workspace)
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

    def topk_reduce_input_region(self) -> dict:
        """Describe borrowed BF16 partials in the shared workspace.

        If apply_routing_weights is true, multiply by top-k scores before
        summing. Otherwise the scores were already applied in FC1. For routes
        outside [0, expert_count), mask both partials and scores to zero before
        multiplying: zero times NaN or Inf is not zero. Caller scores are unchanged.
        Consume on the launch stream before reusing or releasing the workspace.
        """
        if not self.skip_topk_reduce:
            raise RuntimeError("skip_topk_reduce mode is not enabled")
        region = self.token_comm.pre_reduced_activation_region
        return {
            "name": region,
            "byte_offset": self._device_workspace.offset(region),
            "nbytes": self._device_workspace.nbytes(region),
            "shape": (self.max_tokens_per_rank, self.topk, self.hidden_size),
            "stride": (self.topk * self.hidden_size, self.hidden_size, 1),
            "dtype": "bfloat16",
            "alignment": self._device_workspace.byte_alignment(region),
            "apply_routing_weights": not self.apply_topk_at_fc1,
        }

    def skip_topk_reduce_region(self) -> dict:
        return self.topk_reduce_input_region()

    def name(self) -> str:
        """Canonical encoding of every constexpr: the compiled-kernel cache key.

        Two candidates that differ in any construction knob must produce different names, otherwise a tuning sweep
        silently collapses to whichever one compiled first.
        """

        def dtype_name(dtype: type) -> str:
            return getattr(dtype, "__name__", str(dtype)).lower()

        tile = "x".join(str(dimension) for dimension in self.mma_tiler_mnk)
        cluster = "x".join(str(dimension) for dimension in self.cluster_shape_mn)
        fallback_cluster = (
            "none"
            if self.fallback_cluster_shape_mn is None
            else "x".join(str(dimension) for dimension in self.fallback_cluster_shape_mn)
        )
        epi_flags = "x".join(str(batch) for batch in self.epi_flag_batches)
        return (
            f"sm100_block_scaled_swap_ab_mega_moe_{self.quant_kind}_"
            f"{dtype_name(self.a_dtype)}_{dtype_name(self.b_dtype)}_{dtype_name(self.acc_dtype)}_"
            f"sfvec{self.sf_vec_size}_a{self.a_major_mode.name.lower()}_b{self.b_major_mode.name.lower()}_"
            f"e{self.expert_count}_ep{self.world_size}_topk{self.topk}_"
            f"helpers{self.helper_expert_count}_"
            f"topkidx{dtype_name(self.topk_index_dtype)}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_maxtoken{self.max_tokens_per_rank}_"
            f"tile{tile}_cluster{cluster}_{'2cta' if self.use_2cta_instrs else '1cta'}_"
            f"fallback{fallback_cluster}_prefclusters{self.preferred_cluster_count}_"
            f"fallbackclusters{self.fallback_cluster_count}_"
            f"sched{self.schedule_mode}_hint{self.hint}_pad{self.token_padding_block}x{self.sf_padding_block}_"
            f"work{self.work_id_mode}_"
            f"fc2store{'tma' if self.epilogue.fc2_use_tma else 'ublk' if self.epilogue.fc2_use_ublk else 'stg'}_"
            f"fc2tmastages{self.epilogue.fc2_tma_stages}_asyncrel{int(self.epilogue.epi_async_release)}_"
            f"epiflag{epi_flags}_clusters{self.launch_cluster_count}_ctas{self.mixed_cga_config.total_cta_cnt}_"
            f"tokeninflag{self.token_in_flag_batch}_tokenback{self.token_back_mode}_"
            f"combine{self.combine_format.name}_clamp{self.gate_up_clamp}_"
            # The SiTU betas are codegen-time constants, so they must key the compiled kernel.
            f"situ{self.situ_beta}x{self.situ_linear_beta}_"
            f"swiglu{self.swiglu_alpha}x{self.swiglu_beta}_skipreduce{self.skip_topk_reduce}_"
            f"persistent{self.topk_reduce_persistent}_"
            f"{'apply_topk_fc1' if self.apply_topk_at_fc1 else 'apply_topk_fc2'}_"
            f"{'inkernel_redg' if self.reduce_topk_in_kernel else 'separate_reduce'}"
            f"{'_weightevictfirst' if self._mainloop.weight_evict_first else ''}"
            f"{self._shared_name_suffix()}"
            f"{f'_oneround{self.one_round_cta_count}' if self.one_round_dispatch else ''}"
            f"{'_tailpar' if self.tail_parity else ''}"
            # Empty for the NVLink-domain component: the internode=False kernel carries no component token.
            + (f"_{self.token_comm.name_token}" if self.token_comm.name_token else "")
            + "_dlb_v1"
            # Off / SwiGLU keep the historical name so existing compiled-kernel caches stay valid.
            + ("_fc1actscale" if self.fc1_activation_per_token_scale else "")
            + (f"_act{self.gate_activation}" if self.gate_activation != "swiglu" else "")
            # Every build takes the tmp_buffer operand (None with the internode component): an ABI change, so it
            # keys the compiled kernel and its AOT symbol.
            + "_tmpbuf_v1"
            + ("_tanhsigmoid" if self.epilogue.use_tanh_sigmoid else "")
            + ("_dedup" if self.combine_dedup else "")
            + (f"_c{self.token_comm.order_chunk_bytes}" if self.combine_dedup else "")
            + ("_epitail" if self.dedup_epilogue_tail_combine else "")
            + (f"_cpf{self.token_comm.token_back_copies_per_fill}" if self.combine_dedup else "")
            + (f"_poll{self.token_comm.fc2_done_wait_sleep_max_ns}" if self.combine_dedup else "")
            + ("_diagskiptb" if self._diag_skip_token_back else "")
            # The ring size is baked into the pool layouts and the slot arithmetic.
            + (f"_ring{self.ring_token_blocks}_v1" if self.ring_token_blocks else "")
        )

    def _shared_name_suffix(self) -> str:
        if not self.fuse_shared:
            return ""
        return f"_sharedfp8i{self.shared_intermediate_size}clamp{self.shared_gate_up_clamp}"

    default_dedup_poll_max_sleep_ns: ClassVar[int] = 1000

    def _check_epilogue_tail_ring(self) -> None:
        """The epilogue-tail ring reuses the FC2 staging SMEM: check that it fits."""
        needed = self.token_comm.epilogue_tail_ring_bytes
        staging = 0 if self.epilogue.fc2_staging_spec is None else self.epilogue.fc2_staging_spec.nbytes
        if staging < needed:
            raise ValueError(
                f"The combine-dedup epilogue tail needs {needed} B of FC2 staging SMEM, the staging region has "
                f"{staging} B."
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
                sym_int64(divisibility=math.gcd(int(extent), 128)) if axis in dynamic_axes else int(extent)
                for axis, extent in enumerate(shape)
            )
            return make_fake_compact_tensor(dtype, extents, stride_order=stride_order, assumed_align=alignment)

        tokens = self.max_tokens_per_rank
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        experts = self.local_expert_count
        sf_vec_size = self.sf_vec_size
        # Weight SF is atom-swizzled and opaque, so its extent is whatever the caller staged, not a kernel choice.
        fc1_weight_sf_columns = round_up(intermediate_gateup, 128) * round_up(hidden // sf_vec_size, 4)
        fc2_weight_sf_columns = round_up(hidden, 128) * round_up(intermediate_downproj // sf_vec_size, 4)
        # TopkReduce and the in-kernel REDG both take their output dtype from this tensor.
        output_dtype = cutlass.BFloat16

        fake_arguments = dict(
            # Swap-AB puts the weights on the MMA A operand, so the token wire dtype is TokenComm's, not a_dtype.
            activation=fake_tensor(self.token_comm.activation_dtype, (tokens, hidden), (1, 0), {0}, 16),
            activation_sf=fake_tensor(
                self.token_comm.activation_sf_dtype,
                (tokens, self.token_comm.activation_sf_hidden_padded),
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
            # 32 B under combine dedup: TopkReduceDedup stores 256-bit rows.
            output_activation=fake_tensor(
                output_dtype, (tokens, hidden), (1, 0), {0}, REDUCE_OPERAND_ALIGN_BYTES if self.combine_dedup else 16
            ),
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
        # Caller-owned tmp buffer: a bare base pointer with the same 128-byte promise as the workspaces.
        # The internode component has none: its per-launch data stays in local_workspace.
        fake_arguments.update(
            tmp_buffer=make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128)
            if self.separate_tmp_buffer
            else None
        )
        if self.helper_expert_count > 0:
            fake_arguments.update(
                hot_expert_weight_ready_flags=make_ptr(
                    cutlass.Uint64,
                    0,
                    AddressSpace.gmem,
                    assumed_align=8,
                ),
                hot_expert_weight_ready_generation=cutlass.Uint64(1),
            )
        if self.topk_reduce_persistent:
            # The persistent reduce reads its token count from device memory, so
            # the AOT ABI carries the int32[1] operand; static-reduce entries omit it.
            fake_arguments.update(num_valid_tokens=fake_tensor(cutlass.Int32, (1,), (0,), set(), 4))
        if self.quant_kind.uses_global_scale:
            # nvfp4 carries per-expert dequant scalars. Under an e8m0 scale they do not exist at
            # all, and declaring them here would put three tensors in the ABI that the caller has
            # no value for.
            fake_arguments.update(
                fc1_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc2_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc1_norm_const=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
            )
        if self.fuse_shared:
            shared_kind = QuantKind.mxfp8_e4m3
            shared_down = self.shared_intermediate_size
            shared_gateup = 2 * shared_down
            sf_columns = round_up(hidden // shared_kind.sf_vec_size, 4)
            fake_arguments.update(
                shared_fc1_weight=fake_tensor(shared_kind.weight_dtype, (shared_gateup, hidden), (1, 0), set(), 16),
                shared_fc1_weight_sf=fake_tensor(
                    cutlass.Uint8, (round_up(shared_gateup, 128) * sf_columns,), (0,), set(), 16
                ),
                shared_fc2_weight=fake_tensor(shared_kind.weight_dtype, (hidden, shared_down), (1, 0), set(), 16),
                shared_fc2_weight_sf=fake_tensor(
                    cutlass.Uint8,
                    (round_up(hidden, 128) * round_up(shared_down // shared_kind.sf_vec_size, 4),),
                    (0,),
                    set(),
                    16,
                ),
                shared_activation=fake_tensor(shared_kind.activation_dtype, (tokens, hidden), (1, 0), {0}, 16),
                shared_activation_sf=fake_tensor(
                    cutlass.Uint8, (self.shared_token_rows * sf_columns,), (0,), set(), 16
                ),
                shared_num_tokens=Int32(0),
            )
        if self.fc1_activation_per_token_scale:
            # Local (tokens,) fp32 read by the router on the source rank; only in the ABI when enabled.
            fake_arguments.update(
                fc1_activation_per_token_scale=fake_tensor(cutlass.Float32, (tokens,), (0,), {0}, 4),
            )

        compiled = cute.compile[cute.EnableTVMFFI(True)](self, **fake_arguments)
        if out_path is None:
            return compiled
        compiled.export_to_c(
            out_path,
            function_name=self._aot_symbol_name(),
            export_only_tvm_ffi_symbols=True,
        )
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
        smem_limit = utils.get_smem_capacity_in_bytes() // self.occupancy
        smem_workspace = SmemWorkspace()
        self.token_comm.register_smem_regions(smem_workspace)
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
        shared_workspace: cute.Pointer,  # symmetric GMEM byte workspace
        # Caller-owned per-launch buffer of get_tmp_buffer_bytes() bytes, 128-byte aligned: any content
        # at launch, nothing in it survives the launch, may be a different buffer on every launch.
        # None iff the internode component is selected (its per-launch data stays in local_workspace).
        tmp_buffer: Optional[cute.Pointer],
        peer_rank_ptr_mapper_host,
        stream: cuda.CUstream,
        fc1_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc2_alpha: Optional[cute.Tensor] = None,  # (local_experts,)
        fc1_norm_const: Optional[cute.Tensor] = None,  # (local_experts,)
        shared_fc1_weight: Optional[cute.Tensor] = None,  # (2 * shared_intermediate, hidden)
        shared_fc1_weight_sf: Optional[cute.Tensor] = None,  # (packed scale factors,)
        shared_fc2_weight: Optional[cute.Tensor] = None,  # (hidden, shared_intermediate)
        shared_fc2_weight_sf: Optional[cute.Tensor] = None,  # (packed scale factors,)
        shared_activation: Optional[cute.Tensor] = None,  # (tokens, hidden)
        shared_activation_sf: Optional[cute.Tensor] = None,  # (packed scale factors,)
        shared_num_tokens: Optional[Int32] = None,
        hot_expert_weight_ready_flags: Optional[cute.Pointer] = None,  # uint64[EP]
        hot_expert_weight_ready_generation: Optional[cutlass.Uint64] = None,
        gin_dev_comm=None,  # nccl.core.device.cute.DevComm; internode component with enable_nccl_gin only
        gin_window=None,  # nccl.core.device.cute.Window over the symmetric (LSA) workspace arena
        gin_shared_window_offset: Optional[Int64] = None,  # byte offset of shared_workspace inside gin_window (None: 0)
        # (tokens,) fp32, this rank's own tokens, plain local memory (the router reads it here and
        # pushes it to the pool). Required iff the descriptor enables fc1_activation_per_token_scale.
        # Last on purpose: the AOT kwargs wrapper right-aligns its None defaults, so appending keeps
        # every existing positional caller valid.
        fc1_activation_per_token_scale: Optional[cute.Tensor] = None,
        # Caller-owned int32[1] live-token count for persistent finalization.
        num_valid_tokens: Optional[cute.Tensor] = None,
    ) -> None:
        """Launch router, fused MegaMoE compute, and optional TopK reduce."""
        if cutlass.const_expr(self.combine_dedup):
            # TopkReduceDedup stores 256-bit rows: the caller declares 32 B (as on Rubin).
            check_reduce_operand_alignment(output_activation=output_activation)
        # One constexpr bool: the DSL preprocessor only treats a plain ``const_expr(...)`` test as
        # static, so a ``raise`` under ``const_expr(a) or const_expr(b)`` is rejected as an early exit.
        gin_argument_mismatch: cutlass.Constexpr = ((gin_dev_comm is None) != (gin_window is None)) or (
            (gin_dev_comm is not None and gin_window is not None) != self.token_comm.requires_gin_resources
        )
        if cutlass.const_expr(gin_argument_mismatch):
            raise ValueError(
                "gin_dev_comm and gin_window must both be present iff the communication component enables NCCL GIN"
            )
        # ``None`` (AOT builds and single-domain JIT calls never pass it) means offset 0. The router
        # and the fused kernel take the offset as a dynamic ``Int64`` operand, so resolve it here once
        # instead of letting ``None`` reach a kernel argument cast.
        # Without GIN resources no offset argument exists at all (router and component kernels keep main's
        # signatures); with GIN the offset defaults to 0.
        if cutlass.const_expr(not self.token_comm.requires_gin_resources):
            shared_window_offset = None
        elif cutlass.const_expr(gin_shared_window_offset is None):
            shared_window_offset = Int64(0)
        else:
            shared_window_offset = gin_shared_window_offset

        def rewrite_tensor_shape(tensor: cute.Tensor, shape: Tuple) -> cute.Tensor:
            return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=tensor.stride))

        if cutlass.const_expr(self.separate_tmp_buffer != (tmp_buffer is not None)):
            raise ValueError(
                "tmp_buffer must be a pointer iff the kernel keeps its per-launch data in the tmp buffer "
                f"(separate_tmp_buffer={self.separate_tmp_buffer})."
            )
        if cutlass.const_expr(self.fc1_activation_per_token_scale != (fc1_activation_per_token_scale is not None)):
            raise ValueError(
                "fc1_activation_per_token_scale must be passed iff the descriptor enables it "
                f"(enabled={self.fc1_activation_per_token_scale})."
            )
        if cutlass.const_expr(
            fc1_activation_per_token_scale is not None
            and fc1_activation_per_token_scale.element_type is not cutlass.Float32
        ):
            raise TypeError(
                f"fc1_activation_per_token_scale must be Float32, got {fc1_activation_per_token_scale.element_type}."
            )
        if cutlass.const_expr(fc1_activation_per_token_scale is not None):
            # The router gathers scale[token] for every routed token, so the scale needs one entry
            # per topk_indices row. Only checkable when both extents are static (AOT leaves them
            # dynamic); a short tensor would otherwise read past its end silently.
            scale_rows = fc1_activation_per_token_scale.shape[0]
            token_rows = topk_indices.shape[0]
            if cutlass.const_expr(isinstance(scale_rows, int) and isinstance(token_rows, int) and scale_rows < token_rows):
                raise ValueError(
                    f"fc1_activation_per_token_scale has {scale_rows} entries but topk_indices has {token_rows} rows."
                )

        helper_ready_bundle = (
            hot_expert_weight_ready_flags,
            hot_expert_weight_ready_generation,
        )
        if cutlass.const_expr(
            (
                self.helper_expert_count > 0
                and not all(value is not None for value in helper_ready_bundle)
            )
            or (
                self.helper_expert_count == 0
                and any(value is not None for value in helper_ready_bundle)
            )
        ):
            raise ValueError(
                "helper READY flags/generation must both be present iff helper_expert_count > 0"
            )

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
        # The internode router packs the Stage 1 relay messages from the caller's FP4 rows and scale
        # factors and the one-round router pushes them with the routing; the NVLink-domain router takes no
        # payload inputs (keeps its specialization).
        router_takes_payload = self.token_comm.requires_gin_resources or self.one_round_dispatch
        router_activation = activation if cutlass.const_expr(router_takes_payload) else None
        router_activation_sf = activation_sf if cutlass.const_expr(router_takes_payload) else None
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
            gin_dev_comm,
            gin_window,
            shared_window_offset,
            router_activation,
            router_activation_sf,
            fc1_activation_per_token_scale,
            tmp_buffer=tmp_buffer,
        )

        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_object()
        self._device_workspace.assign_device_members(
            local_workspace, shared_workspace, tmp_buffer, read_epoch=False
        )
        # Only external partial-output consumers need invalid rows cleared.
        if cutlass.const_expr(not self.reduce_topk_in_kernel and self.skip_topk_reduce):
            self._route_reset(
                topk_indices,
                self.token_comm.pre_reduced_activation_tensor(self._device_workspace),
                self.token_comm.pre_reduced_activation_sf_tensor(self._device_workspace),
                stream,
            )
        activation_pool = self.token_comm.fc1_activation_tensor(self._device_workspace)
        activation_sf_pool = self.token_comm.fc1_activation_sf_tensor(self._device_workspace)
        fc1_output = self._device_workspace.tensor(self.fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.fc1_output_sf_region)

        topk_reduce_counters = (
            self._device_workspace.tensor(self.topk_reduce_counter_region)
            if cutlass.const_expr(self.topk_reduce_persistent)
            else None
        )

        experts = self.local_expert_count
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        fc1_weight = rewrite_tensor_shape(fc1_weight, (experts, hidden, intermediate_gateup))
        fc2_weight = rewrite_tensor_shape(fc2_weight, (experts, intermediate_downproj, hidden))

        singleton = cutlass.Int32(1)
        token_rows = self.token_comm.fc1_pool_token_count

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
        padded_token_rows = self.token_comm.fc1_pool_sf_token_count
        padded_hidden = hidden
        fc1_sfb = cute.make_tensor(
            activation_sf_pool.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (padded_token_rows, cutlass.Int32(padded_hidden), singleton), sf_vec_size
            ),
        )
        padded_intermediate_gateup = cute.round_up(intermediate_gateup, 128)
        expected_fc1_weight_sf_columns = padded_intermediate_gateup * cute.round_up(hidden // sf_vec_size, 4)
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
                    (self.max_tokens_per_rank, 1, hidden),
                    stride=(output_activation.stride[0], output_activation.stride[0], output_activation.stride[1]),
                ),
            )
            pre_reduced_activation_sf = None
        else:
            pre_reduced_activation = self.token_comm.pre_reduced_activation_tensor(self._device_workspace)
            pre_reduced_activation_sf = self.token_comm.pre_reduced_activation_sf_tensor(self._device_workspace)
        dedup_valid_rank_mask = None
        if cutlass.const_expr(self.combine_dedup):
            dedup_valid_rank_mask = self.token_comm.valid_rank_mask_tensor(self._device_workspace)
        # Internode combine inputs of TopkReduce, resolved while the workspace members are bound.
        combine_landing = self.token_comm.combine_landing_tensor(self._device_workspace)
        combine_message_index = self.token_comm.message_index_tensor(self._device_workspace)

        shared_tensors = None
        if cutlass.const_expr(self.fuse_shared):
            shared_tensors = self._shared_expert_tensors(
                shared_fc1_weight,
                shared_fc1_weight_sf,
                shared_fc2_weight,
                shared_fc2_weight_sf,
                shared_activation,
                shared_activation_sf,
            )

        if cutlass.const_expr(self.token_comm.token_back_push_data):
            # Local FC2 output: the expert-major pool, or the (T*R*K, 1, H) order buffer under dedup.
            fc2_output = self.token_comm.fc2_activation_tensor(self._device_workspace)
        else:
            fc2_output = rewrite_tensor_shape(
                pre_reduced_activation, (pre_reduced_activation.shape[0], pre_reduced_activation.shape[1], hidden)
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
        shared_params = None
        if cutlass.const_expr(self.fuse_shared):
            shared_params = self._mainloop.prepare_shared_tma_load_params(
                fc1_a=shared_tensors["fc1_a"],
                fc1_sfa=shared_tensors["fc1_sfa"],
                fc2_a=shared_tensors["fc2_a"],
                fc2_sfa=shared_tensors["fc2_sfa"],
                fc1_b=shared_tensors["fc1_b"],
                fc1_sfb=shared_tensors["fc1_sfb"],
                fc2_b=shared_tensors["fc2_b"],
                fc2_sfb=shared_tensors["fc2_sfb"],
            )
            shared_params["fc1_output_tma_atom"], shared_params["fc1_output_tma_tensor"] = (
                self.epilogue.prepare_shared_tma_store_params(shared_tensors["fc2_b"])
            )
            shared_params["num_tokens"] = Int32(shared_num_tokens)
            shared_params = tuple(shared_params[key] for key in self._shared_param_keys)

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
            tmp_buffer,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            shared_params,
            hot_expert_weight_ready_flags,
            hot_expert_weight_ready_generation,
            gin_dev_comm,
            gin_window,
            shared_window_offset,
        )
        if cutlass.const_expr(self.mixed_cga_config.is_mixed):
            kernel.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                fallback_cluster=(*self.resolved_fallback_cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=self.occupancy,
                use_pdl=self.use_pdl,
            )
        else:
            kernel.launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
                min_blocks_per_mp=self.occupancy,
                use_pdl=self.use_pdl,
            )
        if cutlass.const_expr(self.topk_reduce_persistent != (num_valid_tokens is not None)):
            raise ValueError("num_valid_tokens is required exactly when topk_reduce_persistent=True.")
        if cutlass.const_expr(not self.reduce_topk_in_kernel and self.combine_dedup):
            # Weights are already folded at FC1; sum the rank slots named by the valid-rank mask.
            self._topk_reduce(pre_reduced_activation, dedup_valid_rank_mask, output_activation, stream)
        elif cutlass.const_expr(not self.reduce_topk_in_kernel and not self.skip_topk_reduce):
            reduce_scores = None if cutlass.const_expr(self.apply_topk_at_fc1) else topk_scores
            shared_term = None
            if cutlass.const_expr(self.fuse_shared):
                self._device_workspace.assign_device_members(
                    local_workspace, shared_workspace, tmp_buffer, read_epoch=False
                )
                shared_term = self._device_workspace.tensor(self.shared_output_region)
                self._device_workspace.remove_device_members()
            if cutlass.const_expr(self.topk_reduce_persistent):
                # Fixed grid; the token count is read from device memory
                # (num_valid_tokens), so one captured launch covers every T.
                self._topk_reduce(
                    pre_reduced_activation,
                    pre_reduced_activation_sf,
                    output_activation,
                    reduce_scores,
                    stream,
                    counters=topk_reduce_counters,
                    num_tokens_dev=num_valid_tokens,
                    topk_indices=topk_indices,
                    shared_term=shared_term,
                )
            else:
                self._topk_reduce(
                    pre_reduced_activation,
                    pre_reduced_activation_sf,
                    output_activation,
                    reduce_scores,
                    stream,
                    remote_partials=combine_landing,
                    message_index=combine_message_index,
                    topk_indices=topk_indices,
                    local_rank=local_rank,
                    shared_term=shared_term,
                )

    @cute.jit
    def _shared_expert_tensors(
        self,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
    ) -> dict:
        """GEMM-domain (M, K, 1) views of the shared expert's operands and workspace buffers."""
        sf_vec_size = QuantKind.mxfp8_e4m3.sf_vec_size
        hidden = self.hidden_size
        gateup = 2 * self.shared_intermediate_size
        down = self.shared_intermediate_size
        rows = self.shared_token_rows
        singleton = Int32(1)

        def k_major(tensor: cute.Tensor, m, k) -> cute.Tensor:
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout((m, Int32(k), singleton), stride=(tensor.stride[0], 1, 0)),
            )

        def scale_factors(tensor: cute.Tensor, m, k) -> cute.Tensor:
            # Scale-factor bytes may arrive as uint8; the MMA reads them as UE8M0.
            return cute.make_tensor(
                cute.recast_ptr(tensor.iterator, dtype=QuantKind.mxfp8_e4m3.sf_dtype),
                blockscaled_utils.tile_atom_to_shape_SF((m, Int32(k), singleton), sf_vec_size),
            )

        fc1_output = self._device_workspace.tensor(self.shared_fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.shared_fc1_output_sf_region)
        return {
            "fc1_a": k_major(fc1_weight, Int32(gateup), hidden),
            "fc1_sfa": scale_factors(fc1_weight_sf, Int32(round_up(gateup, 128)), hidden),
            "fc2_a": k_major(fc2_weight, Int32(hidden), down),
            "fc2_sfa": scale_factors(fc2_weight_sf, Int32(round_up(hidden, 128)), down),
            "fc1_b": k_major(activation, activation.shape[0], hidden),
            "fc1_sfb": scale_factors(activation_sf, Int32(rows), hidden),
            "fc2_b": k_major(fc1_output, Int32(rows), down),
            "fc2_sfb": scale_factors(fc1_output_sf, Int32(rows), down),
        }

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
        tmp_buffer: Optional[cute.Pointer],
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        shared_params: Optional[tuple],
        hot_expert_weight_ready_flags: Optional[cute.Pointer],
        hot_expert_weight_ready_generation: Optional[cutlass.Uint64],
        gin_dev_comm,
        gin_window,
        gin_shared_window_offset,  # Optional[Int64]: None without GIN resources
    ):
        """Compose TokenComm, Scheduler, Mainloop, and Epilogue."""
        # First statement on purpose: a by-value NCCL DevComm argument must be materialized where it
        # dominates the whole kernel; the component keeps only scalar pointers for nested JIT regions.
        self.token_comm.prepare_device_resources(gin_dev_comm, gin_window, gin_shared_window_offset)
        self._mainloop.materialize_codegen_members()
        storage_type = self._smem_workspace.storage_class()
        smem_allocator = utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()
        self._device_workspace.assign_device_members(local_workspace, shared_workspace, tmp_buffer)
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
                (fc1_output_sf_storage.shape[0], intermediate_gateup // 2, cutlass.Int32(1)), self.epilogue.sf_vec_size
            ),
        )
        shared = None
        if cutlass.const_expr(self.fuse_shared):
            shared = dict(zip(self._shared_param_keys, shared_params))
            shared_fc1_done_counter = self._device_workspace.tensor(
                SharedExpertTileSchedule.done_counter_region
            )
            shared_output_storage = self._device_workspace.tensor(self.shared_output_region)
            shared_fc1_output_sf_storage = self._device_workspace.tensor(
                self.shared_fc1_output_sf_region
            )
            shared_sf_vec_size = QuantKind.mxfp8_e4m3.sf_vec_size
            shared_args = {
                "kernel_extension": BlockScaledSwapAbFc12Extension(
                    sf_vec_size=shared_sf_vec_size,
                    fc1_done_counter_pointer=shared_fc1_done_counter.iterator,
                    fc2_spin_threshold=self.shared_fc2_spin_threshold,
                ),
                "fc1_output_tma_atom": shared["fc1_output_tma_atom"],
                "fc1_output_tma_tensor": shared["fc1_output_tma_tensor"],
                "fc1_output_sf": cute.make_tensor(
                    shared_fc1_output_sf_storage.iterator,
                    blockscaled_utils.tile_atom_to_shape_SF(
                        (
                            shared_fc1_output_sf_storage.shape[0],
                            self.shared_intermediate_size,
                            cutlass.Int32(1),
                        ),
                        shared_sf_vec_size,
                    ),
                ),
                "fc1_done_counter": shared_fc1_done_counter,
                # (token, 1, hidden), the routed FC2 output domain.
                "output": cute.make_tensor(
                    shared_output_storage.iterator,
                    cute.make_layout(
                        (shared_output_storage.shape[0], 1, shared_output_storage.shape[1]),
                        stride=(
                            shared_output_storage.stride[0],
                            shared_output_storage.stride[0],
                            shared_output_storage.stride[1],
                        ),
                    ),
                ),
            }

        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = thread_idx // 32
        block_idx = cute.arch.block_idx()
        active_cluster_m = Int32(self.cluster_shape_mn[0])
        is_fallback_cluster = Boolean(False)
        if cutlass.const_expr(self.mixed_cga_config.is_mixed):
            active_cluster_m, _, _ = cute.arch.block_in_cluster_dim()
            is_fallback_cluster = active_cluster_m == Int32(self.resolved_fallback_cluster_shape_mn[0])
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
        pool_activation_scales = self.token_comm.fc1_activation_per_token_scales_tensor(self._device_workspace)

        # Bind the active cluster before deriving the AB pipeline arrival count.
        self._mainloop.assign_device_members(
            self._smem_workspace, smem_base, cta_coord_in_cluster, is_fallback_cluster, hidden, intermediate_gateup
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

        fc2_spin_threshold = (intermediate_gateup + self._mainloop.cta_tile_m - 1) // self._mainloop.cta_tile_m
        helper_weight_ready_gate = None
        if cutlass.const_expr(self.helper_expert_count > 0):
            helper_weight_ready_gate = make_helper_weight_ready_generation_gate(
                memory_slot_count=self.local_expert_count,
                helper_count=self.helper_expert_count,
                source_count=self.world_size,
                terminal_flags=hot_expert_weight_ready_flags,
                expected_generation=hot_expert_weight_ready_generation,
            )
        kernel_extension = BlockScaledSwapAbFc12Extension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=self.token_comm.fc1_ready_counter_pointer(self._device_workspace),
            ring_token_blocks=self.ring_token_blocks,
            ring_data_rows_per_block=self.token_comm.tokens_per_fc1_ready_slot if self.ring_token_blocks else 0,
            ring_sf_rows_per_block=self.token_comm.ring_sf_rows_per_block if self.ring_token_blocks else 0,
        )
        if cutlass.const_expr(self.combine_dedup):
            optional_epilogue_args = GatedActEpilogueArgsDedup(
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                topk_scores=pool_topk_scores,
                fc1_activation_per_token_scale=pool_activation_scales,
                fc2_remote_output=pre_reduced_activation,
                local_rank=local_rank,
            )
        else:
            optional_epilogue_args = GatedActEpilogueArgs(
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                topk_scores=pool_topk_scores,
                fc1_activation_per_token_scale=pool_activation_scales,
            )

        pipeline_init_arrive_mixed_cga(
            self.cluster_shape_mn, self.resolved_fallback_cluster_shape_mn, is_fallback_cluster, is_relaxed=True
        )
        pipeline_init_wait_mixed_cga(
            self.cluster_shape_mn, self.resolved_fallback_cluster_shape_mn, is_fallback_cluster
        )
        finalize_barrier_thread_count = (
            12 * 32 if self.token_back_mode in ("standalone_warps", "hybrid_warps") else 8 * 32  # nosec B105
        )
        finalize_barrier = pipeline.NamedBarrier(barrier_id=13, num_threads=finalize_barrier_thread_count)

        if warp_idx == self.scheduler_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.scheduler_register_count)
            iket.range_push("mega.scheduler")
            iket.range_push("scheduler.register_group")
            scheduler.initialize_fallback_group()
            iket.range_pop()
            if cutlass.const_expr(self.fuse_shared):
                # Local tokens only: these tiles cover the router and dispatch latency.
                scheduler = self._shared_schedule.publish(
                    scheduler,
                    BlockPhase.SharedLinear1,
                    shared["num_tokens"],
                    block_idx[2],
                    block_idx[0],
                )
                if cutlass.const_expr(not self.shared_fc2_claims):
                    # The static share's slots without gate/up tiles run their down-projection tiles here.
                    scheduler = self._shared_schedule.publish(
                        scheduler,
                        BlockPhase.SharedLinear2,
                        shared["num_tokens"],
                        block_idx[2],
                        block_idx[0],
                        slots="without_gate_up",
                    )
            iket.range_push("scheduler.wait_sizes_ready")
            self.token_comm.wait_for_sizes_ready(self._device_workspace)
            iket.range_pop()
            if cutlass.const_expr(self.fuse_shared and self.shared_fc2_claims):
                scheduler.set_shared_fc2_tokens(shared["num_tokens"])
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
            if cutlass.const_expr(self.fuse_shared and not self.shared_fc2_claims):
                scheduler = self._shared_schedule.publish(
                    scheduler,
                    BlockPhase.SharedLinear2,
                    shared["num_tokens"],
                    block_idx[2],
                    block_idx[0],
                    slots="with_gate_up",
                )
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
                shared_params=shared,
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
                shared_params=shared,
                shared_fc1_done_counter_pointer=(
                    shared_args["fc1_done_counter"].iterator
                    if cutlass.const_expr(self.fuse_shared)
                    else None
                ),
                shared_fc2_spin_threshold=(
                    self.shared_fc2_spin_threshold if cutlass.const_expr(self.fuse_shared) else None
                ),
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
                ring_device_workspace=self._device_workspace,
                # The MMA warp retires the blocks unless the combine does.
                ring_retire_region=(
                    self.ring_retire_region
                    if self.ring_token_blocks and not self.ring_covers_fc2_output
                    else None
                ),
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
                self.token_comm.relay_landing_tensor(self._device_workspace),
                self.token_comm.fc2_done_fanout(self._device_workspace, peer_rank_ptr_mapper),
                shared_args if cutlass.const_expr(self.fuse_shared) else None,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(tmem_allocator.retrieve_ptr(self.acc_dtype), self._mainloop.num_tmem_alloc_cols)
            iket.range_pop()
            if cutlass.const_expr(self.dedup_epilogue_tail_combine and not self._diag_skip_token_back):
                # Combine dedup: the epilogue warps reduce leftover blocks in their finished FC2 staging SMEM.
                tail_ring = self._smem_workspace.ptr(self.epilogue.fc2_staging_region, smem_base)
                iket.range_push("mega.token_back_epilogue_tail")
                self.token_comm.token_back_epilogue_tail(self._smem_workspace, smem_base, warp_idx, tail_ring)
                iket.range_pop()
            else:
                # Internode combine: the epilogue warps join the relay warps' gather after their last tile.
                self.token_comm.epilogue_tail_body(self._smem_workspace, smem_base, warp_idx)
            finalize_barrier.arrive_unaligned()

        if warp_idx >= Int32(self.transfer_warp_idx_start):
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            if warp_idx < Int32(self.token_in_end_warp_idx):
                iket.range_push("mega.token_in")
                self.token_comm.token_in(self._smem_workspace, smem_base)
                iket.range_pop()
                if cutlass.const_expr(
                    self.token_comm.token_back_enabled
                    and self.token_back_mode not in ("standalone_warps",)  # nosec B105
                    and not self._diag_skip_token_back
                ):
                    # reuse_dispatch_warps, epi_warps (quantized scale push) and hybrid_warps (after token_in).
                    iket.range_push("mega.token_back")
                    self.token_comm.token_back(self._smem_workspace, smem_base)
                    iket.range_pop()
                finalize_barrier.arrive_and_wait()
                iket.range_push("mega.tail_reset")
                self.token_comm.reset_tail()
                iket.range_pop()
            elif cutlass.const_expr(self.token_back_mode in ("standalone_warps", "hybrid_warps")):  # nosec B105
                iket.range_push("mega.token_back_standalone")
                if cutlass.const_expr(not self._diag_skip_token_back):
                    self.token_comm.token_back(self._smem_workspace, smem_base)
                iket.range_pop()
                finalize_barrier.arrive_unaligned()
            elif cutlass.const_expr(self.relay_warp_count > 0):
                # Relay warps 12-15 of the internode component (no finalize-barrier participation).
                self.token_comm.relay_warp_body(self._smem_workspace, smem_base)
        self.token_comm.remove_device_members()
        self._device_workspace.remove_device_members()
