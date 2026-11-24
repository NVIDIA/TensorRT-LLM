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
instead of per tile (before its work loop, or on its first routed tile when shared
expert tiles run ahead), and the FC12 extension runs with no FC1 ready counter.

There are no transfer warps. Nothing here pulls payloads into an expert pool and
nothing pushes results back: FC1 gathers straight from the dense received plane,
and the FC2 epilogue warps send results to their owning rank themselves through
the bulk reduce-add. Twelve warps cover every role.

The kernel owns the workspace tail reset. Every CTA joins its warp roles and
arrives once; the last arriver clears the tail-reset counters (the scheduler's
claim counters and the fc1_done slots among them) and advances the epoch (in-kernel
reduce: after one cross-rank barrier; standalone reduce: alone, the cross-rank ready counters
are never reset so no peer is signalled), which the next
call's communication kernel reads. That kernel waits at entry until the epoch
shows the previous call's tail complete (a standalone top-k reduce may end before
the tail, and the next launch may start right after the reduce). The tail has no
grid sync, so every CTA passes one release / acquire cluster barrier between
the join of its warp roles and the tail (peers' remote arrives land before a CTA
can exit; the tail itself touches global memory only), and the scheduler warp
drains the cluster claim channel before that join.

Residency: the communication kernel fires ``launch_dependents`` at entry, and a
programmatic dependent is only released once every primary CTA has done so, so
all of its CTAs are resident before this kernel starts; this kernel itself needs
no co-residency of its own CTAs (arrivals do not block, tiles are claimed
dynamically), except n_split, whose liveness needs a cluster on each die (the
scheduler's K6; host guard: more CTAs than the larger die has SMs). Remaining
timing assumptions are listed at the exchange's LL write in
token_comm_gen_specialized.py.

Schedule (``schedule_policy``). Every build runs the decode scheduler,
``("decode", L)`` (``GenphaseDecodeFc12Scheduler``, L an optional static FC1
lead, None = all FC1 tiles of a stream before its FC2 tiles). With a lead the
claim layout is exact for each stream's token blocks K, which the scheduler warp
counts once right after the ``sizes_ready`` wait and before its first claim
(``compute_stream_token_blocks``, mandatory in lead builds; no code without a lead),
so no claim below the end of a stream maps to no tile (each cluster still claims
one past-the-end id per counter it visits, as without a lead). The decode
scheduler itself never waits for in-kernel progress; the scheduler warp's claim
gate (``claim_gate_on_tma_a``) sets how far its claims run ahead of this CTA's
TMA-A. True (default) claims unit j+1 once TMA-A has taken unit j; False does not
wait; (FC1, FC2) (die-blind builds only) is a late gate: it claims unit j+1 once
TMA-A starts the load of the k tile that many k tiles before the end of FC1 / FC2
unit j (0: before its first k tile). A late gate keeps a claim from running far
ahead of the weight stream, so the last units of a launch go to the cluster that
frees up first; it pays off for one or two tokens per rank, where the weight
stream is the whole launch. With a fused shared expert the scheduler warp
publishes the shared FC1 (and early shared FC2) units before its ``sizes_ready``
wait, and tail shared FC2 units after the routed ones. The only FC1 -> FC2
synchronization is still the FC2 TMA-B wait on the fc1_done counters, and the
claim order keeps that wait live (see the decode scheduler's docstring, blocking
point W5 for the gate). The cluster shape is (4, 1) or (2, 1) (with the N128 /
N256 tiles; N64 is 4x1 only).

The die-aware forms split the work between the GPU's two dies; each die claims
its own work first and then steals the other die's (n_split: only its FC2 tiles;
see the scheduler's docstring).
``{"die_aware": "on"}`` (n_split) halves every expert's weight rows: the weight
and scale-factor operands are die-major, ``(2E, K, N/2)`` and ``(2E, cols/2)``
with entry ``half * E + e`` placed on die ``half`` (needs both widths divisible
by two cluster tiles; with a lead each die claims only its own tiles). ``{"die_aware": "expert_half"}`` halves
the local expert list instead: standard operands, experts ``[0, E/2)`` placed
on die 0 and ``[E/2, E)`` on die 1 (needs an even local expert count). A CTA
reads its die from ``cute.arch.locality_domain_id``. A fused shared expert keeps
its standard operands in both forms; by default (``shared_die_aware`` auto / on)
each die claims the tiles of its own part of the shared weight rows first, which
the host places on that die, and ``off`` leaves the shared tiles die-blind.
"""

import hashlib
import math
from typing import ClassVar, Optional, Tuple, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .....api import ImplDesc, KernelClass, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferDevice
from .....helpers.helper_weight_ready_gate import (
    derive_helper_weight_ready_topology,
    make_helper_weight_ready_generation_gate,
)
from .....helpers.ptx_helpers import barrier_sync_unaligned
from .....helpers.cute_py_helpers import Tcgen05MmaInstruction, make_tcgen05_tmem_plan, tcgen05_block_scaled_acc_dtype
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dlb_expert_count import resolve_dlb_expert_counts
from .....helpers.dsl_helpers import ReadyFlag, wait_ready, spin_wait
from .....helpers.iket_compat import iket
from .....helpers.megamoe_aot import make_megamoe_aot_callable
from .....helpers.ptx_helpers import read_smid
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.topk_reduce import TopkReduce, check_reduce_operand_alignment
from .....helpers.utils import ceil_div, round_up
from .....quant_def import CombineFormat, QuantKind
from ....schedulers.base import WorkIdAcquisitionMode
from ....schedulers.genphase_decode_scheduler import GenphaseDecodeFc12Scheduler
from .block_scaled_swap_ab_fc12_epilogue import GatedActEpilogueArgs, SwapABGatedActEpilogue
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .block_scaled_swap_ab_fc12_mainloop_gen_specialized import BlockScaledSwapAbFc12MainloopGenSpecialized
from .shared_expert_helper import (
    SharedExpertKernelMixin,
    shared_kernel_impl_requirements,
    shared_kernel_problem_requirements,
)
from .token_comm_gen_specialized import GenphaseTokenComm, GenphaseTokenCommArgs
from .genphase_tail import GenphaseTail
from .weight_prefetch import prefetch_span


_aot_symbol_prefix = "rubin_genphase_mega_moe_aot"


class BlockScaledSwapAbGenphaseMoeKernel(SharedExpertKernelMixin, KernelClass):
    """Compose generation-phase dispatch, persistent FC12, and the return path."""

    fc1_output_region: ClassVar[str] = "rubin.genphase_moe.fc1_output"
    fc1_output_sf_region: ClassVar[str] = "rubin.genphase_moe.fc1_output_sf"
    fc1_done_counter_region: ClassVar[str] = "rubin.genphase_moe.fc1_done_counter"
    pre_reduced_activation_region: ClassVar[str] = "rubin.genphase_moe.pre_reduced_activation"
    shared_fc1_output_region: ClassVar[str] = "rubin.genphase_moe.shared_fc1_output"
    shared_fc1_output_sf_region: ClassVar[str] = "rubin.genphase_moe.shared_fc1_output_sf"
    shared_fc1_done_counter_region: ClassVar[str] = "rubin.genphase_moe.shared_fc1_done_counter"

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
    largest_die_sm_count: ClassVar[int] = 112  # upper bound on the SM count of one die

    # Tuned for Rubin, not portable. VR200 carries 212 SMs and this kernel's 184 CTAs
    # (46 four-CTA or 92 two-CTA clusters) leave 28 for the communication grid; go over that
    # and the main kernel's last cluster starts only once communication CTAs leave (the
    # last-arriver tail needs no co-residency, tiles are claimed dynamically), except
    # that n_split's liveness needs a cluster on each die (the scheduler's K6). The
    # communication exchange CTAs exit as soon as their count rows are published, so the
    # resident set is one helper plus these pushers. Porting to Blackwell means deriving
    # this number again from that part's SM count and cluster capacity.
    pusher_cta_count: ClassVar[int] = 27
    refine_participant_groups: ClassVar[int] = 2
    refine_output_stages: ClassVar[int] = 4

    # Cluster N carries no tokens under swap-AB, so only cluster M varies. A cluster tile is
    # cta_tile_m x M weight rows; (2, 1) halves it to 256, which n_split needs where a die's
    # half is not a whole number of 512-row tiles (e.g. hidden 3584).
    default_cluster_shape_mn: ClassVar[Tuple[int, int]] = (4, 1)
    supported_cluster_shapes_mn: ClassVar[Tuple[Tuple[int, int], ...]] = ((4, 1), (2, 1))
    # The class attribute, kept for callers that read the cluster shape off the class; a
    # built kernel's own shape is the instance attribute set in __init__.
    cluster_shape_mn: ClassVar[Tuple[int, int]] = default_cluster_shape_mn
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
            "swiglu_alpha": OptionalRequirement(Optional[float]),
            "swiglu_beta": OptionalRequirement(Optional[float]),
            "world_size": int,
            "topk": int,
            "topk_index_dtype": type,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
            **shared_kernel_problem_requirements(),
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
            # True, False or (FC1, FC2); the tuple's contents are checked by value below.
            "claim_gate_on_tma_a": OptionalRequirement(Union[bool, tuple]),
            "sf_plain_transport": OptionalRequirement(bool),
            "helper_expert_count": OptionalRequirement(int),
            "total_helper_slots": OptionalRequirement(int),
            # None = off; cluster_shape_mn is not declared here: it is checked by value below.
            "die_aware": OptionalRequirement(Optional[str]),
            # TensorRT-LLM local patch, removed once the pinned CuTe DSL provides
            # cute.arch.locality_domain_id: the die of each %smid, four 64-bit words.
            "die_smid_mask": OptionalRequirement(Optional[tuple]),
            **shared_kernel_impl_requirements(),
            # Die-aware forms with a shared expert: auto (default) / on split its tiles between the dies, off does not.
            "shared_die_aware": OptionalRequirement(Optional[str]),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)
        # The gen-phase token comm reads the ids as slot ids; the key is refused, not declared, so a framework that
        # probes impl_desc_require() for the route does not take this kernel for one that maps expert ids.
        if impl_desc.get("adaptive_route_experts"):
            raise NotImplementedError("The adaptive route is implemented in token_comm.py's router only.")
        problem_desc, impl_desc = resolve_dlb_expert_counts(
            problem_desc, impl_desc, world_size=problem_desc["world_size"]
        )
        schedule_policy = impl_desc["schedule_policy"]
        if len(schedule_policy) != 2 or schedule_policy[0] != "decode":
            raise ValueError(
                "The generation-phase kernel runs the decode scheduler: schedule_policy must be ('decode', None) or "
                f"('decode', L), got {schedule_policy!r}; grouped and phase_interleave are not schedules of this "
                "kernel."
            )
        # The scheduler's own keys: the lead comes only from schedule_policy, so a caller's copy would be overwritten.
        for key in ("decode_lead", "hint", "schedule_mode"):
            if key in impl_desc:
                raise ValueError(
                    f"{key!r} is not an impl_desc field of the generation-phase kernel; the schedule is "
                    "schedule_policy = ('decode', None) or ('decode', L)."
                )
        supplied_cluster_shape_mn = impl_desc.get("cluster_shape_mn")
        self.cluster_shape_mn = (
            self.default_cluster_shape_mn if supplied_cluster_shape_mn is None else supplied_cluster_shape_mn
        )
        if self.cluster_shape_mn not in self.supported_cluster_shapes_mn:  # a list never equals a tuple
            raise ValueError(
                f"The generation-phase kernel supports cluster_shape_mn in {self.supported_cluster_shapes_mn} "
                f"(default {self.default_cluster_shape_mn}), got {self.cluster_shape_mn}."
            )
        for field_name, fixed_value, supplied_value in (
            ("mma_k_mode", self.mma_k_mode, impl_desc.get("mma_k_mode")),
            ("a_major_mode", self.a_major_mode, problem_desc.get("a_major_mode")),
            ("b_major_mode", self.b_major_mode, problem_desc.get("b_major_mode")),
            ("epilogue_warpgroups", 1, impl_desc.get("epilogue_warpgroups")),
        ):
            if supplied_value is not None and supplied_value != fixed_value:
                raise ValueError(
                    f"The generation-phase kernel fixes {field_name} at {fixed_value}, got {supplied_value}."
                )

        fallback_cluster_shape_mn = impl_desc.get("fallback_cluster_shape_mn")
        if fallback_cluster_shape_mn is not None and fallback_cluster_shape_mn != self.cluster_shape_mn:
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
        self.swiglu_alpha = problem_desc.get("swiglu_alpha")
        self.swiglu_beta = problem_desc.get("swiglu_beta")
        self.world_size = problem_desc["world_size"]
        self.topk = problem_desc["topk"]
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.active_gather_b_warp_count = 2 if self.mma_tiler_mnk[1] == 64 and self.cluster_shape_mn == (4, 1) else 4
        self.scheduler_consumer_thread_count = (7 + self.active_gather_b_warp_count) * 32
        self.schedule_policy = schedule_policy
        # ("decode", L) with the static FC1 lead L (None: every FC1 tile of a stream before its FC2 tiles).
        self.decode_lead = schedule_policy[1]
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.fc2_use_bulk = impl_desc["fc2_use_bulk"]
        self.epi_flag_batches = impl_desc["epi_flag_batches"]
        self.launch_cluster_count = impl_desc["launch_cluster_count"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]
        # The scheduler warp's claim gate. True (default): claim the next unit once TMA-A has taken the previous one.
        # False: do not wait. (FC1, FC2), die-blind builds only: once TMA-A starts the load of the k tile that many k
        # tiles before the end of the previous FC1 / FC2 unit (0: before its first k tile).
        self.claim_gate_on_tma_a = impl_desc.get("claim_gate_on_tma_a", True)
        gate = self.claim_gate_on_tma_a
        if not (type(gate) is bool or (
            type(gate) is tuple and len(gate) == 2 and all(type(k_tiles) is int and k_tiles >= 0 for k_tiles in gate)
        )):
            raise ValueError(
                f"claim_gate_on_tma_a must be True, False or (FC1, FC2) non-negative ints, got {gate!r}."
            )
        self.claim_gate_enabled = gate is not False
        self.late_claim_gate_k_tiles: Optional[Tuple[int, int]] = gate if type(gate) is tuple else None
        self.helper_expert_count = impl_desc.get("helper_expert_count", 0)
        if type(self.helper_expert_count) is not int or self.helper_expert_count < 0:
            raise ValueError("helper_expert_count must be a non-negative exact int")
        self.die_aware = impl_desc.get("die_aware") or "off"
        if self.die_aware not in ("off", "on", "expert_half"):
            raise ValueError(f"die_aware must be off, on or expert_half, got {self.die_aware!r}.")
        # on: each expert's weight rows are halved between the dies (die-major weights, n_split);
        # expert_half: the local expert list is halved between the dies (standard weights).
        self.n_split = self.die_aware == "on"
        self.expert_half = self.die_aware == "expert_half"
        self.die_smid_mask = impl_desc.get("die_smid_mask")
        if self.die_smid_mask is not None and (
            len(self.die_smid_mask) != 4
            or not all(type(word) is int and 0 <= word < 1 << 64 for word in self.die_smid_mask)
        ):
            raise ValueError("die_smid_mask must be four 64-bit words (bit s: the die of SM s).")
        if self.die_aware != "off" and self.helper_expert_count > 0:
            # The per-die placement covers this rank's local experts; DLB helper slots are not placed or owned.
            raise ValueError("die_aware is not supported together with DLB helper slots (helper_expert_count > 0).")

        self.occupancy = 1
        self.architecture = "sm_107"
        self.threads_per_cta = 12 * 32
        self.local_expert_count = self.expert_count // self.world_size
        if self.helper_expert_count > 0:
            self.home_expert_count = derive_helper_weight_ready_topology(
                self.local_expert_count, self.helper_expert_count
            ).home_expert_count
        else:
            self.home_expert_count = self.local_expert_count
        self.cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        self.promised_launchable_sm_count = self.launch_cluster_count * self.cluster_size
        self._resolve_shared_expert(problem_desc, impl_desc)
        shared_die_aware = impl_desc.get("shared_die_aware")
        if shared_die_aware not in (None, "auto", "on", "off"):
            raise ValueError(f"shared_die_aware must be 'auto', 'on' or 'off', got {shared_die_aware!r}.")
        if shared_die_aware is not None and not self.shared_expert:
            raise ValueError("shared_die_aware requires shared_expert=True.")

        if self.die_aware != "off" and self.promised_launchable_sm_count <= self.largest_die_sm_count:
            # Requiring more CTAs than one die can hold guarantees both dies only under K6's full-residency premise.
            # n_split needs this for liveness; expert_half imposes the same guard for locality.
            needs = (
                "n_split needs more than {count} resident CTAs (K6: a cluster on each die)"
                if self.n_split
                else "expert_half is only supported with more than {count} resident CTAs (a cluster on each die; "
                "liveness does not need it, the die-aware placement does)"
            )
            raise ValueError(
                f"{needs.format(count=self.largest_die_sm_count)}, got {self.launch_cluster_count} x "
                f"{self.cluster_size}."
            )

        self._validate_geometry()
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
        if self.shared_expert and self.heterogeneous_shared:
            tmem_plan = self._merge_shared_tmem_plan(tmem_plan, mma_cta_count)
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
        self.tail = GenphaseTail(world_size=self.world_size)

        fc12_problem_desc = ProblemDesc({**problem_desc, "expert_count": self.local_expert_count})
        resolved_impl_desc = ImplDesc(
            {
                # die_aware is forwarded below only when not off (an explicit None reads as off).
                **{key: value for key, value in impl_desc.items() if key not in ("die_aware", "die_smid_mask")},
                # Resolved or fixed here, but the scheduler, the epilogue and the mainloop
                # read them from the descriptor, so they have to be forwarded rather than assumed.
                "cluster_shape_mn": self.cluster_shape_mn,
                "mma_k_mode": self.mma_k_mode,
                "decode_lead": self.decode_lead,
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
                # The standalone reduce below polls the self-validating combine plane.
                "combine_ll_encoded": not self.reduce_topk_in_kernel,
                "fc1_epi_flag_batch": self.epi_flag_batches[0],
                "fc2_epi_flag_batch": self.epi_flag_batches[1],
                **({"die_aware": self.die_aware} if self.die_aware != "off" else {}),
                # The scheduler owns the shared tile stream.
                **self._shared_scheduler_impl_fields(),
            }
        )
        self.scheduler = GenphaseDecodeFc12Scheduler(fc12_problem_desc, resolved_impl_desc)
        # Each die claims its own part of the shared tiles; the host places the weight rows below the boundary on die 0.
        self.shared_die_split = bool(getattr(self.scheduler, "shared_die_split", False))
        self.shared_die_boundary_rows = getattr(self.scheduler, "shared_die_boundary_rows", None)
        self.shared_die_split_reason = getattr(self.scheduler, "shared_die_split_reason", None)
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
            raise ValueError("The generation-phase communication component provides no token-back path.")
        self._make_shared_epilogue(fc12_problem_desc, resolved_impl_desc)
        self._device_workspace = self._build_device_workspace()
        self._mainloop, self._smem_workspace = self._build_mainloop_and_smem(fc12_problem_desc, resolved_impl_desc)
        if self.epilogue.combine_ll_encoded != (not self.reduce_topk_in_kernel):
            raise ValueError("The standalone top-k reduce expects the self-validating bf16 combine plane.")
        self._check_shared_tile_k()
        self._topk_reduce = (
            None
            if self.reduce_topk_in_kernel
            else TopkReduce(
                self.hidden_size,
                self.topk,
                self.combine_format,
                ll_encoded=True,
                num_experts=self.expert_count,
                shared_term=self.reduce_shared_output,
                outlive_producer=self.shared_expert,
            )
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
            raise ValueError("Genphase MegaMoE expert dimensions must be either all static or all runtime.")
        if not all(static_expert_dimensions):
            raise NotImplementedError("The Genphase MegaMoE kernel currently requires static expert dimensions.")
        if self.expert_count <= 0 or self.intermediate_gateup_size <= 0 or self.hidden_size <= 0:
            raise ValueError("Genphase MegaMoE expert dimensions must be positive.")
        if self.world_size <= 0 or self.expert_count % self.world_size != 0:
            raise ValueError("expert_count must be divisible by a positive world_size.")
        if self.topk <= 0 or self.topk > self.expert_count:
            raise ValueError("topk must be positive and no greater than expert_count.")
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}.")
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
        # The decode scheduler checks its lead and work_id_mode itself.
        if self.late_claim_gate_k_tiles is not None and self.die_aware != "off":
            raise ValueError("A late claim gate (claim_gate_on_tma_a = (FC1, FC2)) needs the die-blind build.")
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches must contain FC1 and FC2 batch sizes.")
        if any(batch < 1 or batch > len(self.epilogue_warp_ids) for batch in self.epi_flag_batches):
            raise ValueError(f"epi_flag_batches values must be in [1, {len(self.epilogue_warp_ids)}].")
        if self.sf_vec_size <= 0:
            raise ValueError("sf_vec_size must be positive.")

        for field_name, dimensions, expected_rank in (
            ("mma_instruction_mnk", self.mma_instruction_mnk, 3),
            ("mma_tiler_mnk", self.mma_tiler_mnk, 3),
        ):
            if len(dimensions) != expected_rank:
                raise ValueError(f"{field_name} must contain {expected_rank} dimensions.")
            if not all(isinstance(dimension, int) and not isinstance(dimension, bool) for dimension in dimensions):
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
            raise NotImplementedError("Rubin Genphase MegaMoE does not implement M/N instruction repetition.")
        # The additional decode geometry is deliberately bounded (see the mainloop's
        # n64_decode_tile): tile N 64 with the K of one 128-byte token row, two-CTA MMA,
        # cluster 4x1.
        n64_tile = (256, 64, 128 * 8 // int(self.quant_kind.activation_dtype.width))
        # N64 is qualified on 4x1 clusters only (two gather warps, B multicast across both CTA
        # pairs); the (2, 1) decode cluster keeps the N128 / N256 tiles.
        if self.mma_tiler_mnk[1] == 64 and self.cluster_shape_mn != (4, 1):
            raise ValueError(f"GenPhase N64 runs on 4x1 clusters, got {self.cluster_shape_mn}.")
        if self.mma_tiler_mnk[1] == 64 and not (self.mma_tiler_mnk == n64_tile and self.use_2cta_instrs):
            raise ValueError(
                f"GenPhase N64 requires tile {n64_tile} for {self.quant_kind.value} with two-CTA MMA."
            )
        if mma_n not in (64, 128, 256):
            raise ValueError(f"mma_tiler N must be 64, 128, or 256, got {mma_n}.")
        if mma_k % instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if mma_k % (self.sf_vec_size * 4) != 0:
            raise ValueError("mma_tiler K must be divisible by four scale-factor vectors.")
        if self.intermediate_gateup_size % (self.sf_vec_size * 4) != 0:
            raise ValueError("The intermediate dimension must be divisible by four scale-factor vectors.")
        if mma_m // mma_cta_count != 128:
            raise ValueError(f"The epilogue fixes the per-CTA output tile at 128, got {mma_m // mma_cta_count}.")
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

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def _build_device_workspace(self) -> DeviceWorkspace:
        intermediate_downproj = self.intermediate_gateup_size // 2
        sf_column_count = round_up(ceil_div(intermediate_downproj, self.sf_vec_size), 4)
        max_sf_rows = self.token_comm.worst_case_sf_rows
        # One slot per token block across every local expert, which is the space
        # `cumulative_token_block_count + tile_n_idx` indexes.
        counter_slot_count = self.token_comm.worst_case_padded_rows(self.cta_tile_n) // self.cta_tile_n

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
            # TopkReduce launch folds it into the caller's output afterwards. Cells
            # are stored bit-inverted, so zero means "not arrived": zeroed once here,
            # and re-zeroed by the reduce as it consumes them.
            device_workspace.register(
                self.pre_reduced_activation_region,
                self.combine_format.act_dtype,
                (self.max_tokens_per_rank, self.topk, self.hidden_size),
                buffer_space="shared",
                mem_order=(2, 1, 0),
                byte_alignment=128,
                reset="zero_on_first_allocate",
            )
        self.scheduler.register_device_workspace(device_workspace)
        self.tail.register_device_workspace(device_workspace)
        self.token_comm.register_device_workspace(device_workspace)
        self._register_shared_workspace(device_workspace)
        device_workspace.finalize()
        return device_workspace

    def _build_mainloop_and_smem(
        self, problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc
    ) -> Tuple[BlockScaledSwapAbFc12MainloopGenSpecialized, SmemWorkspace]:
        # The communication component builds its own SMEM workspace for its own
        # launch, so nothing of it is charged against this kernel's budget.
        smem_limit = utils.get_smem_capacity_in_bytes(self.architecture) // self.occupancy
        smem_workspace = SmemWorkspace()
        self.scheduler.register_smem_regions(smem_workspace)
        self.epilogue.register_smem_regions(smem_workspace, **self._shared_smem_kwargs())
        smem_workspace.register_mbarrier(self.tma_a_progress_region, 1)
        fixed_component_bytes = smem_workspace.estimate_total_bytes()
        alignment_shift_bytes = max(region.byte_alignment for region in smem_workspace.regions())
        mainloop_smem_budget_bytes = smem_limit - fixed_component_bytes - alignment_shift_bytes
        if mainloop_smem_budget_bytes <= 0:
            raise ValueError("Fixed kernel components leave no SMEM budget for the Mainloop.")

        mainloop_impl_desc = ImplDesc({**resolved_impl_desc, "mainloop_smem_budget_bytes": mainloop_smem_budget_bytes})
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
        schedule = f"scheddecode_lead{self.decode_lead}"
        die_tag = {"off": "", "on": "die_aware_", "expert_half": "die_aware_expert_half_"}[self.die_aware]
        if self.die_smid_mask is not None:
            die_tag += f"smidmask{hashlib.sha256(repr(self.die_smid_mask).encode()).hexdigest()[:12]}_"
        # claimgate1 / claimgate0 for the on / off gate; a late gate names its k tiles.
        claim_gate_tag = (
            str(int(self.claim_gate_enabled))
            if self.late_claim_gate_k_tiles is None
            else "late{}x{}".format(*self.late_claim_gate_k_tiles)
        )
        return (
            f"sm107_block_scaled_swap_ab_gen_phase_specialized_moe_{self.quant_kind}_"
            f"{dtype_name(self.a_dtype)}_{dtype_name(self.b_dtype)}_{dtype_name(self.acc_dtype)}_"
            f"sfvec{self.sf_vec_size}_a{self.a_major_mode.name.lower()}_b{self.b_major_mode.name.lower()}_"
            f"w{self.world_size}_e{self.expert_count}_topk{self.topk}_helpers{self.helper_expert_count}_"
            f"topkidx{dtype_name(self.topk_index_dtype)}_"
            f"h{self.hidden_size}_i{self.intermediate_gateup_size}_maxtoken{self.max_tokens_per_rank}_"
            f"inst{instruction}_{self.mma_k_mode}_tile{tile}_"
            f"cluster{cluster}_{'2cta' if self.use_2cta_instrs else '1cta'}_"
            f"{schedule}_pad{self.token_padding_block}x{self.sf_padding_block}_"
            f"work{self.work_id_mode}_"
            f"fc2store{'ublk' if self.epilogue.fc2_use_ublk else 'stg'}_"
            f"fc2tmastages{self.epilogue.fc2_tma_stages}_"
            f"epiflag{epi_flags}_clusters{self.launch_cluster_count}_{die_tag}"
            f"pusher{self.pusher_cta_count}_refine{self.refine_participant_groups}x{self.refine_output_stages}_"
            f"combine{self.combine_format.name}_clamp{self.gate_up_clamp}_"
            f"swiglua{self.swiglu_alpha}_swiglub{self.swiglu_beta}_"
            f"{'apply_topk_fc1' if self.apply_topk_at_fc1 else 'apply_topk_fc2'}_"
            f"{'inkernel_reduce' if self.reduce_topk_in_kernel else 'separate_reduce'}_"
            f"claimgate{claim_gate_tag}_sfplain{int(self.token_comm.sf_plain_transport)}_"
            f"abstages{self._mainloop.num_weight_ab_stages}x{self._mainloop.num_token_ab_stages}"
            + ("_dlb_v1" if self.helper_expert_count > 0 else "")
            + self._shared_name_suffix()
            + ("_shareddiesplit" if self.shared_die_split else "")
            # The shared epilogue bakes the SiTU betas and the tanh sigmoid in at codegen time, so they must key the
            # compiled kernel; only non-default values change the name, so SwiGLU kernels keep their keys.
            + (
                f"_situ{self.epilogue.situ_beta}x{self.epilogue.situ_linear_beta}"
                if self.epilogue.situ_beta is not None
                else ""
            )
            + ("_tanhsigmoid" if self.epilogue.use_tanh_sigmoid else "")
        )

    @cute.jit
    def _die_from_smid_mask(self) -> Int32:
        """TensorRT-LLM local patch: this SM's die from die_smid_mask, for DSL builds without
        cute.arch.locality_domain_id (remove with die_smid_mask)."""
        smid = read_smid()
        word = Int64(0)
        for index in cutlass.range_constexpr(4):
            if (smid >> Int32(6)) == Int32(index):
                # A 64-bit word with the top bit set is passed as its two's-complement value.
                value = self.die_smid_mask[index]
                word = Int64(value - (1 << 64) if value >= 1 << 63 else value)
        return Int32((word >> Int64(smid & Int32(63))) & Int64(1))

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
        # n_split: die-major weight (2E, K, N/2) and SF (2E, cols/2) tensors, expert' = half * E + e.
        weight_experts = 2 * experts if self.n_split else experts
        fc1_weight_rows = intermediate_gateup // 2 if self.n_split else intermediate_gateup
        fc2_weight_rows = hidden // 2 if self.n_split else hidden
        # Weight SF is atom-swizzled and opaque, so its extent is whatever the caller staged.
        fc1_weight_sf_columns = round_up(fc1_weight_rows, 128) * round_up(hidden // sf_vec_size, 4)
        fc2_weight_sf_columns = round_up(fc2_weight_rows, 128) * round_up(intermediate_downproj // sf_vec_size, 4)

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
            fc1_weight=fake_tensor(self.a_dtype, (weight_experts, hidden, fc1_weight_rows), (2, 0, 1), {1, 2}, 16),
            fc1_weight_sf=fake_tensor(self.sf_dtype, (weight_experts, fc1_weight_sf_columns), (1, 0), {1}, 16),
            fc2_weight=fake_tensor(
                self.a_dtype, (weight_experts, intermediate_downproj, fc2_weight_rows), (2, 0, 1), {1, 2}, 16
            ),
            fc2_weight_sf=fake_tensor(self.sf_dtype, (weight_experts, fc2_weight_sf_columns), (1, 0), {1}, 16),
            # 32 B: the top-k reduce stores it in up to 256-bit accesses.
            output_activation=fake_tensor(cutlass.BFloat16, (tokens, hidden), (1, 0), {}, 32),
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
        if self.helper_expert_count > 0:
            fake_arguments.update(
                hot_expert_weight_ready_flags=make_ptr(cutlass.Uint64, 0, AddressSpace.gmem, assumed_align=8),
                hot_expert_weight_ready_generation=cutlass.Uint64(1),
            )
        if self.quant_kind.uses_global_scale:
            fake_arguments.update(
                fc1_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc2_alpha=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
                fc1_norm_const=fake_tensor(cutlass.Float32, (experts,), (0,), set(), 4),
            )
        fake_arguments.update(self._shared_fake_arguments(fake_tensor, tokens))

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
        # n_split: die-major tensors instead, (2 * local_experts, ...) with the last extent halved
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
        hot_expert_weight_ready_flags: Optional[cute.Pointer] = None,  # uint64[EP]
        hot_expert_weight_ready_generation: Optional[cutlass.Uint64] = None,
        shared_activation: Optional[cute.Tensor] = None,  # (max_tokens_per_rank, shared_hidden)
        shared_activation_sf: Optional[cute.Tensor] = None,  # flat canonical (atom-major) scale factors
        shared_fc1_weight: Optional[cute.Tensor] = None,  # (1, shared_hidden, shared_intermediate_gateup)
        shared_fc1_weight_sf: Optional[cute.Tensor] = None,  # (1, packed)
        shared_fc2_weight: Optional[cute.Tensor] = None,  # (1, shared_intermediate_downproj, shared_output)
        shared_fc2_weight_sf: Optional[cute.Tensor] = None,  # (1, packed)
        shared_output: Optional[cute.Tensor] = None,  # (max_tokens_per_rank, shared_output) BF16, local
    ) -> None:
        """Launch generation-phase dispatch, fused FC12, and the optional top-k reduce."""
        check_reduce_operand_alignment(output_activation=output_activation, shared_output=shared_output)

        helper_ready_bundle = (hot_expert_weight_ready_flags, hot_expert_weight_ready_generation)
        if cutlass.const_expr(
            (self.helper_expert_count > 0 and not all(value is not None for value in helper_ready_bundle))
            or (self.helper_expert_count == 0 and any(value is not None for value in helper_ready_bundle))
        ):
            raise ValueError("helper READY flags/generation must both be present iff helper_expert_count > 0")

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
        if cutlass.const_expr(self.quant_kind.uses_global_scale):
            for scalar_name, scalar in (
                ("fc1_alpha", fc1_alpha),
                ("fc2_alpha", fc2_alpha),
                ("fc1_norm_const", fc1_norm_const),
            ):
                if cutlass.const_expr(scalar is None):
                    raise ValueError(f"{self.quant_kind} requires {scalar_name}.")
        if cutlass.const_expr(
            self.die_aware != "off" and self.die_smid_mask is None and not hasattr(cute.arch, "locality_domain_id")
        ):
            raise RuntimeError(
                "die_aware reads the die of each SM with cute.arch.locality_domain_id (PTX %locality_domain_id), "
                "which this CuTe DSL build lacks: use one that provides cute.arch.locality_domain_id, "
                "or pass die_smid_mask."
            )

        self._check_shared_operands(
            dict(
                shared_activation=shared_activation,
                shared_activation_sf=shared_activation_sf,
                shared_fc1_weight=shared_fc1_weight,
                shared_fc1_weight_sf=shared_fc1_weight_sf,
                shared_fc2_weight=shared_fc2_weight,
                shared_fc2_weight_sf=shared_fc2_weight_sf,
                shared_output=shared_output,
            )
        )

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
        canonical_activation_sf = self.token_comm.canonical_activation_sf_tensor(self._device_workspace)
        scheduler_expert_sizes = self.token_comm.scheduler_expert_sizes_tensor(self._device_workspace)
        fc1_output = self._device_workspace.tensor(self.fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.fc1_output_sf_region)

        experts = self.local_expert_count
        hidden = self.hidden_size
        intermediate_gateup = self.intermediate_gateup_size
        intermediate_downproj = intermediate_gateup // 2
        # n_split: die-major weight and SF views; the extension adds half * E to the expert coordinate and rebases the
        # row block into the half, so the scheduler and epilogues still address the full (E, N) problem.
        weight_experts = 2 * experts if self.n_split else experts
        fc1_weight_rows = intermediate_gateup // 2 if self.n_split else intermediate_gateup
        fc2_weight_rows = hidden // 2 if self.n_split else hidden
        fc1_weight = rewrite_tensor_shape(fc1_weight, (weight_experts, hidden, fc1_weight_rows))
        fc2_weight = rewrite_tensor_shape(fc2_weight, (weight_experts, intermediate_downproj, fc2_weight_rows))

        singleton = cutlass.Int32(1)

        sf_vec_size = self.sf_vec_size

        fc1_a = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (cutlass.Int32(fc1_weight_rows), cutlass.Int32(hidden), cutlass.Int32(weight_experts)),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
            ),
        )
        # FC1's B is the dense (source rank, source token) plane the communication
        # kernel filled, indexed by `gather_index` rather than by a pool row, so it
        # carries no per-expert offset.
        # Gather4 always issues a 2D TMA instruction. A static singleton row
        # mode is otherwise dropped from the descriptor at EP1 capacity one.
        # Keep that extent dynamic so the descriptor retains both dimensions.
        gather_rows = dense_activation.shape[0]
        if cutlass.const_expr(gather_rows == 1):
            gather_rows = cutlass.Int32(gather_rows)
        fc1_b = cute.make_tensor(
            dense_activation.iterator,
            cute.make_layout(
                (gather_rows, cutlass.Int32(hidden), singleton),
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
        # TMA load views keep a dynamic non-zero L stride so all of a warp's tensor maps share one rank.
        fc2_b = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (cutlass.Int32(self.max_routed_rows), cutlass.Int32(intermediate_downproj), singleton),
                stride=(fc1_output.stride[0], fc1_output.stride[1], cutlass.Int32(fc1_output.stride[0])),
            ),
        )
        # The canonical scale-factor pool already carries the MMA's atom layout, so
        # FC1's SFB leg is an ordinary tiled TMA over it.
        fc1_sfb = cute.make_tensor(
            canonical_activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (cutlass.Int32(self.token_comm.worst_case_sf_rows), cutlass.Int32(hidden), singleton), sf_vec_size
            ),
        )
        padded_intermediate_gateup = cute.round_up(fc1_weight_rows, sf_vec_size * 4)
        expected_fc1_weight_sf_columns = padded_intermediate_gateup * hidden // sf_vec_size
        if cutlass.const_expr(
            isinstance(fc1_weight_sf.shape[1], int) and isinstance(expected_fc1_weight_sf_columns, int)
        ):
            if cutlass.const_expr(fc1_weight_sf.shape[1] != expected_fc1_weight_sf_columns):
                raise ValueError("fc1_weight_sf has an incompatible column count.")
        fc1_sfa = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (cutlass.Int32(padded_intermediate_gateup), cutlass.Int32(hidden), cutlass.Int32(weight_experts)),
                sf_vec_size,
            ),
        )

        experts_fc2, intermediate_fc2, hidden_fc2 = fc2_weight.shape
        fc2_a = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (cutlass.Int32(hidden_fc2), cutlass.Int32(intermediate_fc2), cutlass.Int32(experts_fc2)),
                stride=(fc2_weight.stride[2], fc2_weight.stride[1], fc2_weight.stride[0]),
            ),
        )
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
                (cutlass.Int32(fc1_output_sf.shape[0]), cutlass.Int32(fc1_output_sf.shape[1] * sf_vec_size), singleton),
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
                    stride=(output_activation.stride[0], output_activation.stride[0], output_activation.stride[1]),
                ),
            )
        else:
            pre_reduced_activation = self._device_workspace.tensor(self.pre_reduced_activation_region)
        fc2_output = rewrite_tensor_shape(
            pre_reduced_activation, (pre_reduced_activation.shape[0], pre_reduced_activation.shape[1], hidden)
        )

        self._mainloop.materialize_codegen_members()
        # TMA-A / TMA-B take every phase's coordinates from the FC1 template views, so only FC2's atoms are kept.
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

        (
            shared_tma_params,
            shared_fc1_output_tma_atom,
            shared_fc1_output_tma_tensor,
            shared_fc2_output_tma_atom,
            shared_fc2_output_tma_tensor,
            shared_fc2_output,
        ) = self._prepare_shared_operands(
            shared_activation,
            shared_activation_sf,
            shared_fc1_weight,
            shared_fc1_weight_sf,
            shared_fc2_weight,
            shared_fc2_weight_sf,
            shared_output,
        )

        grid = self.scheduler.get_grid_shape(max_active_clusters=self.launch_cluster_count)
        # The last-arriver tail counts promised_launchable_sm_count arrivals; another grid size would never reset
        # the tail span (or reset it while CTAs still run).
        grid_cta_count = grid[0] * grid[1] * grid[2]
        if cutlass.const_expr(grid_cta_count != self.promised_launchable_sm_count):
            raise ValueError(
                f"the last-arriver tail needs grid == launch_cluster_count * cluster_size == "
                f"promised_launchable_sm_count, got grid {grid} ({grid_cta_count} CTAs), "
                f"{self.launch_cluster_count} x {self.cluster_size}, {self.promised_launchable_sm_count}."
            )
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
            hot_expert_weight_ready_flags,
            hot_expert_weight_ready_generation,
            (fc1_weight.iterator.toint(), fc1_weight_sf.iterator.toint()),
            # Shared FC1's (tensor, atom) pairs, then shared FC2's atoms only (coordinates from the FC1 views).
            *shared_tma_params[:8],
            *shared_tma_params[9::2],
            shared_fc1_output_tma_atom,
            shared_fc1_output_tma_tensor,
            shared_fc2_output_tma_atom,
            shared_fc2_output_tma_tensor,
            shared_fc2_output,
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
                pre_reduced_activation,
                None,
                output_activation,
                reduce_scores,
                stream,
                topk_indices=topk_indices,
                shared_term=shared_output if cutlass.const_expr(self.reduce_shared_output) else None,
            )

    @property
    def lazy_comm_waits(self) -> bool:
        """Shared builds wait for the comm flags on the first routed tile and drain them at the tail."""
        return self.shared_expert

    def _payload_ready_flag(self) -> ReadyFlag:
        """The routed payload has landed; peers write it, hence system scope.

        Standalone-reduce builds never reset the counter: this call's slot and target come from the epoch, which
        only this rank's tail advances, after every CTA of this call has arrived. Read before that arrival it is
        this call's. In-kernel-reduce builds keep the original tail-reset counter (slot 0, fixed target).
        """
        epoch = Int32(0)
        if cutlass.const_expr(self.token_comm.ready_counters_never_reset):
            epoch = self.tail.read_epoch(self._device_workspace)
        return ReadyFlag(
            self._device_workspace.ptr(self.token_comm.token_data_ready_region)
            + self.token_comm.ready_counter_slot(epoch),
            self.token_comm.ready_counter_target(epoch, self.token_comm.payload_ready_target),
            "sys",
        )

    def _metadata_ready_flag(self) -> ReadyFlag:
        """This rank's routing metadata (gather index, scores, records) is final."""
        return ReadyFlag(
            self._device_workspace.ptr(self.token_comm.metadata_ready_region),
            self.token_comm.metadata_ready_target,
            "gpu",
        )

    # ------------------------------------------------------------------
    # Weight prefetch
    # ------------------------------------------------------------------

    # Total across all clusters, issued once the payload flag is seen (see profiles/NOTES.md).
    l2_prefetch_budget_bytes: ClassVar[int] = 64 << 20
    l2_prefetch_chunk_bytes: ClassVar[int] = 16384
    # Experts whose FC1 weights (+ SFA) are prefetched at the kernel start, before the sizes exist: local experts
    # 0 .. start_prefetch_expert_count-1, which the routing kernel pins to the first slots (pinned_expert_count).
    start_prefetch_expert_count: ClassVar[int] = 1

    @property
    def _start_prefetch_enabled(self) -> bool:
        return (
            self.l2_prefetch_budget_bytes > 0 and not self.lazy_comm_waits and self.token_comm.pins_experts > 0
            and self.die_aware == "off" and not self.n_split
        )

    @property
    def _start_prefetch_on_idle_warp(self) -> bool:
        """The last Gather-B warp is idle when fewer than four Gather-B warps are active (tile N 64 on 4x1)."""
        return self.active_gather_b_warp_count < len(self.gather_b_warp_ids)
    # TMA-A arrives once per work tile it takes; the scheduler claims unit j only after
    # TMA-A has taken unit j - 1, so claims follow the order units start in.
    tma_a_progress_region: ClassVar[str] = "genphase.smem.tma_a_progress"

    @cute.jit
    def _prefetch_leading_experts(self, linear_cta_idx: Int32, weight_bases: Tuple[Int64, Int64]) -> None:
        """Warm L2 with the FC1 weights + SFA of local experts 0 .. start_prefetch_expert_count-1 (one warp per CTA).

        Issued at the kernel start: these experts are the first slots of the processing order (the routing kernel
        pins them there), so the first claim wave's weights are known before any size is. Each expert's FC1 entry is
        one contiguous span, so the leading experts are one span split evenly over the CTAs (a remainder below one
        CTA share is left to the regular prefetch / the stream).
        """
        lane_idx = cute.arch.lane_idx()
        experts = min(self.token_comm.pins_experts, self.start_prefetch_expert_count, self.local_expert_count)
        row_bytes = self.hidden_size * int(self.a_dtype.width) // 8
        entry_rows = self.intermediate_gateup_size
        sf_block_bytes = 128 * round_up(self.hidden_size // self.sf_vec_size, 4)
        ctas = self.launch_cluster_count * self.cluster_size
        chunk = self.l2_prefetch_chunk_bytes
        for base, total in ((weight_bases[0], experts * entry_rows * row_bytes),
                            (weight_bases[1], experts * (round_up(entry_rows, 128) // 128) * sf_block_bytes)):
            share = (total // ctas) // 16 * 16
            if cutlass.const_expr(share > 0):
                prefetch_span(base + Int64(linear_cta_idx) * Int64(share), share, lane_idx, chunk)

    @cute.jit
    def _prefetch_first_waves(self, linear_cta_idx: Int32, weight_bases: Tuple[Int64, Int64]) -> None:
        """Warm L2 with this CTA's share of the first FC1 claim waves (one warp).

        Claims are dynamic, so a CTA does not prefetch what it will itself load:
        CTA ``c`` covers slice ``c % cluster_size`` of FC1 unit ``c // cluster_size``
        in every wave, and the waves together cover the first units in the order
        the FC1 stream hands them out (slot-major, output block minor).

        The die-aware forms have one FC1 stream per die (the decode scheduler's
        per-die counters): unit ``u`` is unit ``u >> 1`` of die ``u & 1``'s stream,
        so the waves cover the first units of both streams, half the budget each.
        expert_half: a die's stream holds only the slots whose expert lies in that
        die's half (the scheduler's ``_owner_owns_slot``), standard weight entries.
        n_split: every slot, ``blocks_fc1`` units per token block, rows of the
        die-major weight entry ``die * E + expert`` (``gateup / 2`` rows each).
        """
        hidden = self.hidden_size
        gateup = self.intermediate_gateup_size
        cluster_size = self.cluster_size
        slice_rows = self.cta_tile_m
        unit_rows = slice_rows * cluster_size
        die_aware = self.die_aware != "off"
        # Weight rows of one weight entry: n_split's die-major entries hold half an expert's rows.
        entry_rows = gateup // 2 if self.n_split else gateup
        blocks_fc1 = ceil_div(entry_rows, unit_rows)
        token_block = self.cta_tile_n
        # Weight row in bytes for this quant kind (NVFP4: hidden / 2, MXFP8: hidden).
        row_bytes = hidden * int(self.a_dtype.width) // 8
        expert_bytes = entry_rows * row_bytes
        sf_block_bytes = 128 * round_up(hidden // self.sf_vec_size, 4)
        sf_expert_bytes = round_up(entry_rows, 128) // 128 * sf_block_bytes
        unit_bytes = unit_rows * row_bytes + cluster_size * sf_block_bytes
        clusters = self.launch_cluster_count
        prefetch_units = self.l2_prefetch_budget_bytes // unit_bytes
        waves = ceil_div(prefetch_units, clusters)
        # expert_half filters the slots per die, so it scans once per die; the other forms share one scan.
        scans = 2 if self.expert_half else 1

        lane_idx = cute.arch.lane_idx()
        sizes_by_slot = self.token_comm.scheduler_expert_sizes_tensor(self._device_workspace)
        slot_to_expert = self.token_comm.slot_to_expert_tensor(self._device_workspace)
        part = linear_cta_idx % Int32(cluster_size)
        slot_chunks = ceil_div(self.local_expert_count, 32)
        # Each lane owns one slot per chunk: its unit range comes from a warp scan, and
        # the slot holding a target unit is found by ballot, so no lane walks serially.
        chunk_end = cute.make_rmem_tensor((scans * slot_chunks,), Int32)
        chunk_units = cute.make_rmem_tensor((scans * slot_chunks,), Int32)
        chunk_expert = cute.make_rmem_tensor((slot_chunks,), Int32)
        for scan in cutlass.range_constexpr(scans):
            carry = Int32(0)
            for chunk in cutlass.range_constexpr(slot_chunks):
                slot = Int32(chunk * 32) + lane_idx
                units = Int32(0)
                if cutlass.const_expr(scan == 0):
                    chunk_expert[chunk] = Int32(-1)
                if slot < Int32(self.local_expert_count):
                    units = ceil_div(sizes_by_slot[slot], Int32(token_block)) * Int32(blocks_fc1)
                    slot_expert = slot_to_expert[slot]
                    if cutlass.const_expr(scan == 0):
                        chunk_expert[chunk] = slot_expert
                    if cutlass.const_expr(self.expert_half):
                        if (slot_expert * Int32(2)) // Int32(self.local_expert_count) != Int32(scan):
                            units = Int32(0)
                inclusive = units
                for step_log in cutlass.range_constexpr(5):
                    step = Int32(1 << step_log)
                    previous = Int32(cute.arch.shuffle_sync(inclusive, lane_idx - step))
                    if lane_idx >= step:
                        inclusive = inclusive + previous
                chunk_end[scan * slot_chunks + chunk] = carry + inclusive
                chunk_units[scan * slot_chunks + chunk] = units
                carry = carry + Int32(cute.arch.shuffle_sync(inclusive, Int32(31)))

        for wave in cutlass.range_constexpr(waves):
            unit = linear_cta_idx // Int32(cluster_size) + Int32(wave * clusters)
            stream_unit = unit
            stream_die = Int32(0)
            if cutlass.const_expr(die_aware):
                stream_unit = unit >> Int32(1)
                stream_die = unit & Int32(1)
            expert = Int32(-1)
            output_block = Int32(0)
            for scan in cutlass.range_constexpr(scans):
                for chunk in cutlass.range_constexpr(slot_chunks):
                    index = scan * slot_chunks + chunk
                    begin = chunk_end[index] - chunk_units[index]
                    owns = (stream_unit >= begin) & (stream_unit < chunk_end[index])
                    if cutlass.const_expr(scans > 1):
                        owns = owns & (stream_die == Int32(scan))
                    ballot = Int32(cute.arch.vote_ballot_sync(owns))
                    if ballot != Int32(0):
                        owner = Int32(cute.arch.popc((ballot & (-ballot)) - Int32(1)))
                        owner_expert = Int32(cute.arch.shuffle_sync(chunk_expert[chunk], owner))
                        owner_begin = Int32(cute.arch.shuffle_sync(begin, owner))
                        expert = owner_expert
                        output_block = (stream_unit - owner_begin) % Int32(blocks_fc1)
            first_row = output_block * Int32(unit_rows) + part * Int32(slice_rows)
            if expert >= Int32(0) and first_row < Int32(entry_rows) and unit < Int32(prefetch_units):
                entry = expert
                if cutlass.const_expr(self.n_split):
                    entry = stream_die * Int32(self.local_expert_count) + expert
                a_base = weight_bases[0] + Int64(entry) * Int64(expert_bytes) + Int64(first_row) * Int64(row_bytes)
                sf_base = (
                    weight_bases[1]
                    + Int64(entry) * Int64(sf_expert_bytes)
                    + Int64(first_row // Int32(128)) * Int64(sf_block_bytes)
                )
                if cutlass.const_expr(entry_rows % unit_rows == 0):
                    prefetch_span(a_base, slice_rows * row_bytes, lane_idx, self.l2_prefetch_chunk_bytes)
                    prefetch_span(sf_base, sf_block_bytes, lane_idx, self.l2_prefetch_chunk_bytes)
                else:
                    if first_row + Int32(slice_rows) <= Int32(entry_rows):
                        prefetch_span(a_base, slice_rows * row_bytes, lane_idx, self.l2_prefetch_chunk_bytes)
                        prefetch_span(sf_base, sf_block_bytes, lane_idx, self.l2_prefetch_chunk_bytes)

    @cute.jit
    def _wait_claim_gate(self, tma_a_progress, published: Int32) -> None:
        """W5: the scheduler warp waits until this CTA's TMA-A opens the claim gate on the unit it just published (the
        n-th): when TMA-A takes the unit, or with a late gate when it starts the load of the gate's k tile."""
        if cutlass.const_expr(self.claim_gate_enabled):
            cute.arch.mbarrier_wait(tma_a_progress, (published - Int32(1)) % Int32(2))

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
        hot_expert_weight_ready_flags: Optional[cute.Pointer],
        hot_expert_weight_ready_generation: Optional[cutlass.Uint64],
        weight_bases: Tuple[Int64, Int64],
        # Shared expert operands (all None when disabled).
        shared_fc1_tma_a_tensor: Optional[cute.Tensor],
        shared_fc1_tma_a_atom: Optional[cute.CopyAtom],
        shared_fc1_tma_sfa_tensor: Optional[cute.Tensor],
        shared_fc1_tma_sfa_atom: Optional[cute.CopyAtom],
        shared_fc1_tma_b_tensor: Optional[cute.Tensor],
        shared_fc1_tma_b_atom: Optional[cute.CopyAtom],
        shared_fc1_tma_sfb_tensor: Optional[cute.Tensor],
        shared_fc1_tma_sfb_atom: Optional[cute.CopyAtom],
        shared_fc2_tma_a_atom: Optional[cute.CopyAtom],
        shared_fc2_tma_sfa_atom: Optional[cute.CopyAtom],
        shared_fc2_tma_b_atom: Optional[cute.CopyAtom],
        shared_fc2_tma_sfb_atom: Optional[cute.CopyAtom],
        shared_fc1_output_tma_atom: Optional[cute.CopyAtom],
        shared_fc1_output_tma_tensor: Optional[cute.Tensor],
        shared_fc2_output_tma_atom: Optional[cute.CopyAtom],
        shared_fc2_output_tma_tensor: Optional[cute.Tensor],
        shared_fc2_output: Optional[cute.Tensor],
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
                (fc1_output_sf_storage.shape[0], intermediate_gateup // 2, cutlass.Int32(1)), self.epilogue.sf_vec_size
            ),
        )
        shared_fc1_done_counter_pointer, shared_fc2_spin_threshold = self._shared_counter_bindings()

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
        # The scheduler enumerates slots; the extension maps a slot to its expert through this remap, and so
        # does the expert_half scheduler to tell which die owns a slot.
        expert_remap = self.token_comm.slot_to_expert_tensor(self._device_workspace)
        die = None
        if cutlass.const_expr(self.die_aware != "off"):
            # %locality_domain_id is the die of this CTA's SM; a cluster lies within one GPC, so within one die.
            if cutlass.const_expr(self.die_smid_mask is not None):
                die = cute.arch.make_warp_uniform(self._die_from_smid_mask())
            else:
                die = cute.arch.make_warp_uniform(cute.arch.locality_domain_id())
        self.scheduler.assign_device_members(
            expert_token_sizes=scheduler_expert_sizes,
            block_idx=block_idx,
            smem_workspace=self._smem_workspace,
            smem_base=smem_base,
            device_workspace=self._device_workspace,
            expert_remap=expert_remap,
            die=die,
        )
        scheduler = self.scheduler

        fc2_spin_threshold = (intermediate_gateup + self._mainloop.cta_tile_m - 1) // self._mainloop.cta_tile_m
        # No FC1 ready counter: readiness is published per rank, not per tile, and
        # the waits below cover it once per warp.
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
            fc1_ready_counter_pointer=None,
            expert_remap=expert_remap,
            n_split=self.n_split,
            local_expert_count=self.local_expert_count,
            fc1_half_cta_tiles=(self.intermediate_gateup_size // 2) // self._mainloop.cta_tile_m,
            fc2_half_cta_tiles=(self.hidden_size // 2) // self._mainloop.cta_tile_m,
            shared_fc1_done_counter_pointer=shared_fc1_done_counter_pointer,
            shared_fc2_spin_threshold=shared_fc2_spin_threshold,
        )
        optional_epilogue_args = GatedActEpilogueArgs(
            fc1_alpha=fc1_alpha, fc2_alpha=fc2_alpha, fc1_norm_const=fc1_norm_const, topk_scores=pool_topk_scores
        )
        shared_kernel_extension, shared_epilogue_bindings = self._shared_device_bindings(
            kernel_extension,
            shared_fc1_output_tma_atom,
            shared_fc1_output_tma_tensor,
            shared_fc2_output_tma_atom,
            shared_fc2_output_tma_tensor,
            shared_fc2_output,
        )

        tma_a_progress = self._smem_workspace.ptr(self.tma_a_progress_region, smem_base)
        if warp_idx == self.scheduler_warp_id:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tma_a_progress, 1)
        cute.arch.mbarrier_init_fence()
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        self._mainloop.assign_device_members(
            self._smem_workspace, smem_base, cta_coord_in_cluster, hidden, intermediate_gateup
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.scheduler_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            iket.range_push("genphase.scheduler")
            published = Int32(0)
            if cutlass.const_expr(self.shared_expert):
                # The shared FC1 and early shared FC2 units need no expert sizes: they run ahead of W3.
                iket.range_push("scheduler.shared_prologue")
                shared_tile = scheduler.gen_next_shared_work()
                while shared_tile.is_valid_tile:
                    scheduler.publish_work(kernel_extension.prepare_work_tile(shared_tile))
                    published = published + Int32(1)
                    self._wait_claim_gate(tma_a_progress, published)
                    shared_tile = scheduler.gen_next_shared_work()
                iket.range_pop()
            # The decode scheduler has no expert-sizes gate of its own: wait for the published sizes here.
            iket.range_push("scheduler.wait_sizes_ready")
            spin_wait(
                self._device_workspace.ptr(self.token_comm.sizes_ready_region),
                lambda value: value == Int32(1),
                scope="gpu",
                sleep_cycles=150,
            )
            iket.range_pop()
            if cutlass.const_expr(self.decode_lead is not None):
                # Mandatory for a decode lead: the exact layout's K per stream (decode scheduler docstring), once,
                # all lanes, before the first claim. Without a lead: no code.
                iket.range_push("scheduler.stream_token_blocks")
                scheduler.compute_stream_token_blocks()
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
                published = published + Int32(1)
                iket.range_pop()
                # Unit j is claimed once TMA-A has opened the gate on unit j - 1 (waiting from unit 1 keeps the parity
                # exact): at most one unit of lookahead for the weight stream (less with a late gate), and claims in
                # start order.
                if cutlass.const_expr(self.claim_gate_enabled):
                    if published >= Int32(1):
                        self._wait_claim_gate(tma_a_progress, published)
                iket.range_push("scheduler.gen_work")
                work_tile = scheduler.gen_next_work()
                iket.range_pop()
            iket.range_push("scheduler.publish_tail")
            scheduler.publish_work(work_tile)
            scheduler.produce_tail()
            # Drain peers' DSMEM arrives before the leader can exit; see finish_claim_channel.
            scheduler.finish_claim_channel()
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
                trigger_dependents_when_drained=not self.reduce_topk_in_kernel,
                progress_mbarrier=tma_a_progress if self.claim_gate_enabled else None,
                late_claim_gate_k_tiles=self.late_claim_gate_k_tiles,
                shared_fc1_tma_a_tensor=shared_fc1_tma_a_tensor,
                shared_fc1_tma_a_atom=shared_fc1_tma_a_atom,
                shared_fc1_tma_sfa_tensor=shared_fc1_tma_sfa_tensor,
                shared_fc1_tma_sfa_atom=shared_fc1_tma_sfa_atom,
                shared_fc2_tma_a_atom=shared_fc2_tma_a_atom,
                shared_fc2_tma_sfa_atom=shared_fc2_tma_sfa_atom,
                shared_kernel_extension=shared_kernel_extension,
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
                fc1_sf_ready_pointer=self._device_workspace.ptr(self.token_comm.fc1_sf_ready_region),
                expert_sizes=self._device_workspace.tensor(self.token_comm.expert_sizes_region),
                fc1_sf_parts_per_block=self.token_comm.refine_parts_per_block,
                shared_fc1_tma_b_tensor=shared_fc1_tma_b_tensor,
                shared_fc1_tma_b_atom=shared_fc1_tma_b_atom,
                shared_fc1_tma_sfb_tensor=shared_fc1_tma_sfb_tensor,
                shared_fc1_tma_sfb_atom=shared_fc1_tma_sfb_atom,
                shared_fc2_tma_b_atom=shared_fc2_tma_b_atom,
                shared_fc2_tma_sfb_atom=shared_fc2_tma_sfb_atom,
                shared_kernel_extension=shared_kernel_extension,
                shared_fc1_done_counter_pointer=shared_fc1_done_counter_pointer,
                shared_fc2_spin_threshold=shared_fc2_spin_threshold,
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
            # Lazy-wait builds acquire metadata_ready on the first routed tile instead.
            epilogue_ready_kwargs = {}
            if cutlass.const_expr(self.lazy_comm_waits):
                epilogue_ready_kwargs = dict(routed_ready_flag=self._metadata_ready_flag())
            else:
                iket.range_push("epilogue.wait_metadata_ready")
                wait_ready(self._metadata_ready_flag())
                iket.range_pop()
            sched_consumer = scheduler.make_consumer()
            tmem_allocator.allocate(self._mainloop.num_tmem_alloc_cols)
            barrier_sync_unaligned(self._mainloop.tmem_allocation_barrier_id, 32 * (1 + self._mainloop.num_accumulator_consumer_warps_per_cta))
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
                shared=shared_epilogue_bindings,
                **epilogue_ready_kwargs,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(tmem_allocator.retrieve_ptr(self.acc_dtype), self._mainloop.num_tmem_alloc_cols)

            iket.range_pop()

        if (warp_idx >= Int32(self.gather_b_warp_ids[0])) & (warp_idx <= Int32(self.gather_b_warp_ids[-1])):
            cute.arch.warpgroup_reg_dealloc(self.other_warp_register_count)
            if cutlass.const_expr(self._start_prefetch_enabled and self._start_prefetch_on_idle_warp):
                # An inactive Gather-B warp issues the start prefetch, so the L2-prefetch / gather warp is not
                # delayed by it on its way to the sizes / metadata / payload waits.
                if warp_idx == Int32(self.gather_b_warp_ids[-1]):
                    self._prefetch_leading_experts(linear_cta_idx, weight_bases)
            # Every gather warp deallocates its registers; only the active ones (two at
            # tile N 64, four otherwise) consume work and issue Gather4.
            if warp_idx < Int32(self.gather_b_warp_ids[0] + self.active_gather_b_warp_count):
                iket.range_push("genphase.gather_b1")
                if warp_idx == Int32(self.gather_b_warp_ids[0]) and cutlass.const_expr(
                    self.l2_prefetch_budget_bytes > 0 and not self.lazy_comm_waits
                ):
                    iket.range_push("gather_b1.l2_prefetch")
                    if cutlass.const_expr(self._start_prefetch_enabled and not self._start_prefetch_on_idle_warp):
                        self._prefetch_leading_experts(linear_cta_idx, weight_bases)
                    spin_wait(
                        self._device_workspace.ptr(self.token_comm.sizes_ready_region),
                        lambda value: value == Int32(1),
                        scope="gpu",
                    )
                    # Held until the metadata is ready: the weight stream then fills the idle
                    # window before the first MMA instead of slowing the dispatch's
                    # latency-bound memory round trips.
                    wait_ready(self._metadata_ready_flag())
                    self._prefetch_first_waves(linear_cta_idx, weight_bases)
                    iket.range_pop()
                # Lazy-wait builds take the payload / metadata waits on the first routed FC1 tile instead.
                gather_ready_kwargs = {}
                if cutlass.const_expr(self.lazy_comm_waits):
                    gather_ready_kwargs = dict(
                        routed_input_ready_flag=self._payload_ready_flag(),
                        routed_metadata_ready_flag=self._metadata_ready_flag(),
                    )
                else:
                    iket.range_push("gather_b1.wait_input_ready")
                    # `gather_index` is local, published by this rank's sort workers.
                    wait_ready(self._payload_ready_flag())
                    wait_ready(self._metadata_ready_flag())
                    iket.range_pop()
                sched_consumer = scheduler.make_consumer()
                self._mainloop.run_tma_gather_b1(
                    fc1_tma_b_atom=fc1_tma_b_atom,
                    gather_index=gather_index,
                    ab_pipeline=token_pipeline,
                    ab_pipeline_state=gather_b_pipeline_state,
                    sched_consumer=sched_consumer,
                    gather_warp_idx=warp_idx - Int32(self.gather_b_warp_ids[0]),
                    **gather_ready_kwargs,
                )
                iket.range_pop()

        # CTAs with no work can finish the epilogue before gather warps pass
        # their readiness waits. Join every role before clearing those flags.
        cute.arch.sync_threads()
        # No CTA may leave while a cluster peer can still touch its shared memory: the
        # scheduler's work-id broadcast releases, the pipelines' consumer releases and the
        # multicast legs arrive on peer CTAs, and the last of them are issued after the
        # peer's own last tile is done. A CTA pair with little or no work can finish its
        # epilogue before then, and a remote operation on a CTA that has already exited
        # faults (cluster target block not present). The tail below only touches global
        # memory, so no cluster barrier follows it.
        # Release / acquire (the defaults of barrier.cluster.arrive / wait): a relaxed
        # arrive would order nothing, so a peer's remote mbarrier arrive issued before its
        # arrive could still land after this CTA's wait returns.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if warp_idx < len(self.epilogue_warp_ids):
            epilogue_thread_count = 32 * len(self.epilogue_warp_ids)

            # Every epilogue warp of this CTA has drained its stores; one arrival per CTA.
            iket.range_push("tail.arrive")
            cute.arch.barrier(barrier_id=self.tail_barrier_id, number_of_threads=epilogue_thread_count)
            iket.range_pop()
            # Release the top-k reduce launch: it is a programmatic dependent that waits
            # for this grid's completion itself, so it can be resident before the tail ends.
            cute.arch.griddepcontrol_launch_dependents()
            if warp_idx == Int32(self.epilogue_warp_ids[0]):
                lane_idx = cute.arch.lane_idx()
                is_last = Int32(0)
                if lane_idx == Int32(0):
                    if self.tail.arrive(self._device_workspace, Int32(self.promised_launchable_sm_count)):
                        is_last = Int32(1)
                is_last = Int32(cute.arch.shuffle_sync(is_last, Int32(0)))
                if is_last != Int32(0):
                    # The last arriver has acquired every other CTA's arrival, so nothing on
                    # this rank reads the tail span any more: reset it, then advance the epoch
                    # (in-kernel reduce: after meeting the peers).
                    if cutlass.const_expr(self.lazy_comm_waits):
                        # No CTA may have waited on the comm flags yet: drain them before the reset clears them.
                        iket.range_push("tail.wait_comm_complete")
                        wait_ready(self._payload_ready_flag())
                        wait_ready(self._metadata_ready_flag())
                        iket.range_pop()
                    iket.range_push("tail.reset_workspace")
                    self.tail.reset_tail_space(self._device_workspace, lane_idx, 32)
                    cute.arch.sync_warp()
                    iket.range_pop()
                    iket.range_push("tail.nvlink_barrier")
                    if lane_idx == Int32(0):
                        epoch = self.tail.read_epoch(self._device_workspace)
                        if cutlass.const_expr(self.reduce_topk_in_kernel):
                            # Peers reduce-add straight into this rank's output, so the
                            # call is complete only once every rank is through.
                            self.tail.cross_rank_barrier(self._device_workspace, peer_rank_ptr_mapper, epoch)
                        else:
                            # The standalone reduce polls its cells itself, so nothing in this call waits for
                            # the peers, and the peers have nothing to wait for either: the only state they
                            # write here (the ready counters) is never reset.
                            self.tail.advance_epoch(self._device_workspace, epoch)
                    iket.range_pop()

        self._device_workspace.remove_device_members()


__all__ = ["BlockScaledSwapAbGenphaseMoeKernel"]
