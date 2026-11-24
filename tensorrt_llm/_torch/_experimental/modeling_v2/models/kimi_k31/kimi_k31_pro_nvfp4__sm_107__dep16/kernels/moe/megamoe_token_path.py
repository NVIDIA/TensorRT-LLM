# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""dep16's MegaMoE token-path levers: the engine's generation-phase kernel and its token exchange with
three more launch constants (Rubin).

``RubinTokenPathMegaMoE`` is the engine kernel ``BlockScaledSwapAbGenphaseMoeKernel``
(``tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src/rubin/inference/mega/``
``block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py``, which the package exports as
``RubinInferenceGenphaseMegaMoE``) with ``RubinTokenPathComm`` as its token exchange: the engine's
``GenphaseTokenComm`` (``token_comm_gen_specialized.py`` beside it) with three levers. They change
how the exchange signals and when its CTAs wait; every index, segment base and scale-factor block
that reaches the MMAs is the same:

* ``topk_sentinel``: a top-k plane carries no flag. Every index goes out as index + 1 into an inbox
  the main kernel's tail zeroes with the ready counters, and each sort thread polls its own words
  until none reads 0;
* ``sort_local_prefix``: each sort CTA takes its segment bases from the helper's slot order once
  sizes_ready is out (:meth:`RubinTokenPathComm._stage_local_segment_bases`) and releases
  metadata_ready only after prefix_ready; refine stages its expert table at prefix_ready;
* ``refine_sf_first``: refine polls the scale-factor planes' counter (sys scope) before metadata.

The exchange overrides the methods the levers touch, each the engine's with the levers' branches
(``cutlass.const_expr`` blocks where the kernel is traced): ``impl_desc_require``, ``__init__`` and
``_validate_configuration`` (the switches; the sentinel inbox refuses the FC1 score plane),
``_register_symmetric_regions`` (the inbox's reset), ``_build_smem_workspace`` (the sort CTAs' scan
regions), ``_run_exchange``, ``_canonicalize_wire_indices``, ``_run_sort`` and ``_run_refine``; it
adds ``wire_index_bias`` and ``_stage_local_segment_bases``. The kernel carries the switches as class
constants (``token_comm_*``, off), tags any that is on in ``name()`` and builds this exchange with
them. The target's test ``kernels/moe/test_megamoe_token_path.py`` pins the engine's version of every
overridden method, so an engine change to one fails there (the copy must follow it).

One more switch moves work out of the launch: under ``defer_topk_reduce`` (lever
``moe_back_fused_norm``) a launch leaves its combine plane (every routed slot's FC2 output of a row,
``[max tokens, top_k, hidden]`` BF16 in the shared workspace) unreduced, without the engine's
``TopkReduce`` launch, and the caller reduces and normalizes it in one kernel
(``ops/moe/rubin_moe_back_norm.py``). Every kernel this class builds records where its plane lies
(:attr:`RubinTokenPathMegaMoE.combine_plane_offsets`), and :func:`combine_plane` views it.

:func:`bind_token_path_levers` turns the levers on for the process, as the launcher binds the
pusher CTAs: it sets the class constants and makes this kernel the generation kernel the engine's
MegaMoE op builds (:func:`_install_generation_kernel`). dep16 calls it before its MegaMoE workspace's
first query. Nothing here registers a torch op.
"""

from types import MappingProxyType
from typing import ClassVar, Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import (
    ImplDesc,
    OptionalRequirement,
    ProblemDesc,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.nvlink_domain.symmetric_buffer import (  # noqa: E501
    SymmetricBufferDevice,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.token_protocol import (
    TokenSrcMetadata,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.device_workspace import DeviceWorkspace
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.dsl_helpers import (
    smem_exclusive_prefix,
    spin_wait,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.iket_compat import iket
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.ptx_helpers import (
    cp_async_bulk_s2g,
    red_async_add_release_gpu_s32,
    red_async_add_release_sys_u32,
    tma_load_1d,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.smem_workspace import SmemWorkspace
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.utils import (
    ceil_div,
    padded_expert_rows,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.rubin.inference.mega.block_scaled_swap_ab_mega_moe_kernel_gen_specialized import (  # noqa: E501
    BlockScaledSwapAbGenphaseMoeKernel,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.rubin.inference.mega.token_comm_gen_specialized import (  # noqa: E501
    GenphaseTokenComm,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def import QuantKind

__all__ = [
    "RubinTokenPathComm",
    "RubinTokenPathMegaMoE",
    "bind_deferred_topk_reduce",
    "bind_token_path_levers",
    "combine_plane",
]

#: The generation kernel's token-path class constants, by :func:`bind_token_path_levers` argument.
_TOKEN_LEVERS = MappingProxyType(
    {
        "topk_sentinel": "token_comm_topk_sentinel",
        "sort_local_prefix": "token_comm_sort_local_prefix",
        "refine_sf_first": "token_comm_refine_sf_first",
    }
)


class RubinTokenPathComm(GenphaseTokenComm):
    """The engine's generation-phase token exchange with the token-path levers.

    Descriptors as the engine exchange's, and three optional switches in ``impl_desc``
    (``topk_sentinel``, ``sort_local_prefix``, ``refine_sf_first``; left out, off). The sort CTAs'
    two scan regions exist only under ``sort_local_prefix``.
    """

    sort_scan_input_region: ClassVar[str] = "genphase.token_comm.smem.sort_scan_input"
    sort_scan_prefix_region: ClassVar[str] = "genphase.token_comm.smem.sort_scan_prefix"

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "sf_padding_block": int,
            "pusher_cta_count": int,
            "refine_participant_groups": int,
            "refine_output_stages": int,
            "smem_limit_bytes": OptionalRequirement(int),
            "topk_sentinel": OptionalRequirement(bool),
            "sort_local_prefix": OptionalRequirement(bool),
            "refine_sf_first": OptionalRequirement(bool),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.topk_index_dtype = problem_desc["topk_index_dtype"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.pusher_cta_count = impl_desc["pusher_cta_count"]
        self.refine_participant_groups = impl_desc["refine_participant_groups"]
        self.refine_output_stages = impl_desc["refine_output_stages"]
        self.smem_limit_bytes = impl_desc.get("smem_limit_bytes", self.default_smem_limit_bytes)
        # The token path's levers, all off by default:
        # * topk_sentinel -- a top-k plane carries no flag. Every index goes out as index + 1 into
        #   an inbox the main kernel's tail zeroes with the ready counters, and each receiving
        #   thread polls its own words until none reads 0. The flag's sys-scope release takes
        #   microseconds to land even on the rank's own copy; the words are there as soon as
        #   the bulk copy completes.
        # * sort_local_prefix -- the sort CTAs take their segment bases from the helper's slot
        #   order once sizes_ready is out, by the helper's arithmetic for their own source,
        #   instead of waiting for its prefixes; metadata_ready still waits for prefix_ready.
        #   Refine stages its expert table at prefix_ready, before the metadata wait.
        # * refine_sf_first -- refine polls the scale-factor counter (sys scope, in long before)
        #   before metadata (gpu scope), as the gather warps poll their two counters.
        self.topk_sentinel = bool(impl_desc.get("topk_sentinel", False))
        self.sort_local_prefix = bool(impl_desc.get("sort_local_prefix", False))
        self.refine_sf_first = bool(impl_desc.get("refine_sf_first", False))

        self._validate_configuration()
        self._smem_workspace = self._build_smem_workspace()

    def _validate_configuration(self) -> None:
        positive_fields = (
            "world_size",
            "expert_count",
            "topk",
            "max_tokens_per_rank",
            "hidden_size",
            "token_padding_block",
            "sf_padding_block",
            "pusher_cta_count",
            "refine_participant_groups",
            "refine_output_stages",
            "smem_limit_bytes",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive, got {getattr(self, field_name)}.")

        if self.expert_count % self.world_size != 0:
            raise ValueError(
                f"expert_count must be divisible by world_size, got {self.expert_count} and {self.world_size}."
            )
        if self.expert_count > 16384:
            raise NotImplementedError("Genphase token comm supports at most 16384 global experts.")
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )
        if self.max_tokens_per_rank > 1024:
            raise ValueError(
                f"Genphase token comm requires max_tokens_per_rank <= 1024, got {self.max_tokens_per_rank}."
            )
        if self.topk_index_dtype not in (cutlass.Int32, cutlass.Int64):
            raise ValueError(
                f"topk_index_dtype must be Int32 or Int64, got {self.topk_index_dtype}."
            )

        if self.sf_padding_block != 128:
            raise ValueError(
                f"sf_padding_block must be the 128-row SF atom, got {self.sf_padding_block}."
            )
        if self.sf_padding_block % self.token_padding_block != 0:
            raise ValueError("sf_padding_block must be divisible by token_padding_block.")

        # One SF value covers sf_vec_size activation elements and one atom spans
        # four of them along K, so the hidden extent has to fill whole atoms.
        elements_per_sf_atom = self.quant_kind.sf_vec_size * 4
        if self.hidden_size % elements_per_sf_atom != 0:
            raise ValueError(
                f"{self.quant_kind} requires hidden_size divisible by {elements_per_sf_atom}."
            )

        # The caller's top-k plane is exactly `routes_per_rank` elements with no
        # slack, so the bulk copy that stages it has to land on that boundary
        # rather than rounding up into memory we do not own.
        topk_plane_bytes = self.routes_per_rank * int(self.topk_index_dtype.width) // 8
        if topk_plane_bytes % 16 != 0:
            raise ValueError(
                f"max_tokens_per_rank * topk * sizeof({self.topk_index_dtype}) must be a multiple of 16 bytes, "
                f"got {topk_plane_bytes}."
            )
        if self.apply_topk_at_fc1 and (self.routes_per_rank * 4) % 16 != 0:
            raise ValueError(
                f"max_tokens_per_rank * topk * 4 must be a multiple of 16 bytes for the score plane, "
                f"got {self.routes_per_rank * 4}."
            )
        # Each refine group claims and finishes blocks on its own, so it must own
        # whole warps: the group's named barrier id is warp-uniform only if no warp
        # straddles two groups, and `bar.sync` is implicitly aligned.
        refine_threads = self.refine_participant_groups * self.refine_warps_per_group * 32
        if refine_threads != self.threads_per_cta:
            raise ValueError(
                f"{self.refine_participant_groups} refine groups of "
                f"{self.refine_warps_per_group} warps cover {refine_threads} threads, "
                f"which must equal the {self.threads_per_cta}-thread block."
            )
        # One thread stages one row of a 128-row block.
        if self.refine_warps_per_group * 32 != 128:
            raise ValueError(
                "A refine group must be exactly four warps so one thread owns one row."
            )
        # The staging load reads a run-aligned row straight out of the linear
        # plane, so the plane's row has to be at least that wide. Both extents
        # round the same value up to sixteen, from a payload alignment and from an
        # atom count respectively; this catches the two drifting apart.
        if self.plain_activation_sf_row_elements < self.refine_row_columns:
            raise ValueError(
                f"the linear scale-factor row covers {self.plain_activation_sf_row_elements} columns, "
                f"short of the {self.refine_row_columns} the staging load reads."
            )
        if self.refine_staging_row_stride < self.refine_row_columns:
            raise ValueError(
                f"the staging row stride {self.refine_staging_row_stride} is short of the "
                f"{self.refine_row_columns} columns the refine runs traverse."
            )
        if self.refine_barrier_base + self.refine_participant_groups > 16:
            raise ValueError(
                f"{self.refine_participant_groups} refine groups starting at barrier "
                f"{self.refine_barrier_base} exceed the 16 CTA barriers."
            )
        if self.topk_sentinel and self.apply_topk_at_fc1:
            raise ValueError(
                "the sentinel top-k inbox covers the index words only; the FC1 score plane needs "
                "the top-k flag."
            )

    @property
    def wire_index_bias(self) -> int:
        """Added to every top-k index on the wire: 1 under the sentinel inbox, so no word is 0."""
        return 1 if self.topk_sentinel else 0

    def _register_symmetric_regions(self, workspace: DeviceWorkspace) -> None:
        workspace.register(
            self.topk_index_inbox_region,
            cutlass.Int32,
            (self.world_size, self.max_tokens_per_rank, self.topk),
            buffer_space="shared",
            stride=(self.topk_slab_elements, self.topk, 1),
            byte_alignment=16,
            # The sentinel inbox is zeroed between launches with the ready counters.
            reset="tail_reset" if self.topk_sentinel else "data",
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.topk_score_inbox_region,
                cutlass.Float32,
                (self.world_size, self.max_tokens_per_rank, self.topk),
                buffer_space="shared",
                stride=(self.topk_slab_elements, self.topk, 1),
                byte_alignment=16,
            )
        workspace.register(
            self.topk_ready_region,
            cutlass.Int32,
            (self.world_size,),
            buffer_space="shared",
            reset="tail_reset",
        )
        # The shape is the logical column extent and the row padding is a stride.
        # Bulk copies move whole padded rows, so the trailing row's padding has to
        # be inside the region even though no index reaches it; the workspace sizes
        # a region by what its strides occupy for exactly this reason.
        for region_name, dtype, columns, row_elements in (
            (
                self.plain_activation_region,
                self.activation_dtype,
                self.hidden_size,
                self.plain_activation_row_elements,
            ),
            (
                self.plain_activation_sf_region,
                self.activation_sf_dtype,
                self.hidden_sf,
                self.plain_activation_sf_row_elements,
            ),
        ):
            workspace.register(
                region_name,
                dtype,
                (self.world_size, self.max_tokens_per_rank, columns),
                buffer_space="shared",
                stride=(self.max_tokens_per_rank * row_elements, row_elements, 1),
                byte_alignment=128,
            )
        for region_name in (self.token_data_ready_region, self.token_sf_plain_ready_region):
            workspace.register(
                region_name, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset"
            )

    def _build_smem_workspace(self) -> SmemWorkspace:
        """Place the five role lifetimes on one overlay.

        Every CTA walks a strict subset of the roles in time order (exchange then
        sort then refine, or pusher then refine), and no two roles are live at
        once, so the physical footprint is the largest single role rather than
        their sum.
        """
        warp_count = self.threads_per_cta // 32
        workspace = SmemWorkspace()
        workspace.register_mbarrier(self.exchange_mbarrier_region, 1)
        workspace.register_mbarrier(
            self.pusher_mbarrier_region, warp_count * self.pusher_pipeline_depth
        )
        workspace.register_mbarrier(self.refine_mbarrier_region, self.refine_participant_groups)
        overlay = workspace.create_overlay("genphase.token_comm.smem.role")

        # Staged at slab rather than route granularity: the wire transfer moves a
        # whole padded slab so that its byte count is bulk-copy aligned even when
        # `max_tokens_per_rank * topk` is not, and the staging buffer has to cover
        # what the transfer reads.
        exchange = overlay.add_lifetime("exchange")
        exchange.register_tensor(
            self.exchange_topk_stage_region,
            self.topk_index_dtype,
            (self.topk_slab_elements,),
            byte_alignment=16,
        )
        if self.apply_topk_at_fc1:
            exchange.register_tensor(
                self.exchange_score_stage_region,
                cutlass.Float32,
                (self.topk_slab_elements,),
                byte_alignment=16,
            )

        sort = overlay.add_lifetime("sort")
        sort.register_tensor(
            self.sort_histogram_region,
            cutlass.Int32,
            (self.histogram_bucket_count,),
            byte_alignment=16,
        )
        sort.register_tensor(
            self.sort_prefix_region,
            cutlass.Int32,
            (self.local_expert_bucket_count,),
            byte_alignment=16,
        )
        sort.register_tensor(
            self.sort_warp_totals_region,
            cutlass.Int32,
            (self.threads_per_cta // 32,),
            byte_alignment=16,
        )
        if self.sort_local_prefix:
            for region_name in (self.sort_scan_input_region, self.sort_scan_prefix_region):
                sort.register_tensor(
                    region_name, cutlass.Int32, (self.local_expert_bucket_count,), byte_alignment=16
                )

        helper = overlay.add_lifetime("helper")
        helper.register_tensor(
            self.helper_count_matrix_region,
            cutlass.Int32,
            (self.world_size, self.local_expert_bucket_count),
            stride=(self.local_expert_bucket_count, 1),
            byte_alignment=16,
        )
        for region_name in (
            self.helper_totals_region,
            self.helper_scan_input_region,
            self.helper_prefix_region,
        ):
            helper.register_tensor(
                region_name, cutlass.Int32, (self.local_expert_bucket_count,), byte_alignment=16
            )
        helper.register_tensor(
            self.helper_warp_totals_region,
            cutlass.Int32,
            (self.threads_per_cta // 32,),
            byte_alignment=16,
        )
        for region_name in (self.helper_slot_to_expert_region, self.helper_totals_by_slot_region):
            helper.register_tensor(
                region_name, cutlass.Int32, (self.local_expert_bucket_count,), byte_alignment=16
            )
        if not self.ranks_experts_in_warp:
            helper.register_tensor(
                self.helper_load_bin_region,
                cutlass.Int32,
                (self.load_bin_count,),
                byte_alignment=16,
            )

        pusher = overlay.add_lifetime("pusher")
        for region_name, dtype, row_elements in (
            (
                self.pusher_data_stage_region,
                self.activation_dtype,
                self.plain_activation_row_elements,
            ),
            (
                self.pusher_sf_stage_region,
                self.activation_sf_dtype,
                self.plain_activation_sf_row_elements,
            ),
        ):
            pusher.register_tensor(
                region_name,
                dtype,
                (warp_count, self.pusher_pipeline_depth, row_elements),
                stride=(self.pusher_pipeline_depth * row_elements, row_elements, 1),
                byte_alignment=16,
            )

        refine = overlay.add_lifetime("refine")
        refine.register_tensor(
            self.refine_claim_region,
            cutlass.Int32,
            (self.refine_participant_groups, self.refine_claim_fields),
            stride=(self.refine_claim_fields, 1),
            byte_alignment=16,
        )
        refine.register_tensor(
            self.refine_expert_table_region,
            cutlass.Int32,
            (self.refine_table_rows, self.local_expert_bucket_count),
            stride=(self.local_expert_bucket_count, 1),
            byte_alignment=16,
        )
        # The column extent is what the runs traverse rather than `hidden_sf`: a
        # staging row is written and read at that width, and `hidden_sf` is only
        # the live prefix of it.
        refine.register_tensor(
            self.refine_staging_region,
            self.activation_sf_dtype,
            (self.refine_participant_groups, 128, self.refine_row_columns),
            stride=(self.refine_staging_group_stride, self.refine_staging_row_stride, 1),
            byte_alignment=16,
        )
        refine.register_tensor(
            self.refine_output_region,
            self.activation_sf_dtype,
            (
                self.refine_participant_groups,
                self.refine_output_stages,
                self.refine_output_stage_elements,
            ),
            stride=(
                self.refine_output_stages * self.refine_output_stage_elements,
                self.refine_output_stage_elements,
                1,
            ),
            byte_alignment=16,
        )

        workspace.finalize(max_bytes=self.smem_limit_bytes)
        return workspace

    @cute.jit
    def _run_exchange(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        smem_base: cute.Pointer,
        destination_rank: Int32,
        thread_idx: Int32,
    ) -> None:
        """Send this rank's whole top-k plane to one peer.

        This is the only cross-rank round trip on the critical path, and it has no
        routing dependency, so it leaves at kernel entry. The plane is tiny
        (``max_tokens_per_rank * topk`` indices) and every peer needs all of it,
        so each of the W exchange CTAs stages the same source plane and unicasts
        it once. There is no cross-rank multicast to amortize that with.
        """
        iket.range_push("exchange")
        index_stage = self._smem_workspace.tensor(self.exchange_topk_stage_region, smem_base)
        load_mbarrier = self._smem_workspace.ptr(self.exchange_mbarrier_region, smem_base)

        index_plane_bytes = self.routes_per_rank * int(self.topk_index_dtype.width) // 8
        score_plane_bytes = self.routes_per_rank * 4
        staged_bytes = index_plane_bytes
        if cutlass.const_expr(self.apply_topk_at_fc1):
            staged_bytes = index_plane_bytes + score_plane_bytes

        if thread_idx == Int32(0):
            cute.arch.mbarrier_init(load_mbarrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if thread_idx == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(load_mbarrier, Int32(staged_bytes))
            tma_load_1d(
                index_stage.iterator, topk_indices.iterator, load_mbarrier, Int32(index_plane_bytes)
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                score_stage = self._smem_workspace.tensor(
                    self.exchange_score_stage_region, smem_base
                )
                tma_load_1d(
                    score_stage.iterator,
                    topk_scores.iterator,
                    load_mbarrier,
                    Int32(score_plane_bytes),
                )
        cute.arch.mbarrier_wait(load_mbarrier, 0)

        self._canonicalize_wire_indices(index_stage, thread_idx)
        # Every thread rewrote its own share of the staged plane; the single thread
        # that copies the plane out reads all of it.
        cute.arch.sync_threads()

        # The peer's slab address is this rank's slab address shifted by the
        # symmetric-heap delta, so the layout supplies the intra-region offset and
        # the mapper supplies the inter-rank one. Neither is computed by hand.
        index_inbox = device_workspace.tensor(self.topk_index_inbox_region)
        own_index_slab = index_inbox[local_rank, None, None]
        peer_offset = peer_rank_ptr_mapper.map(Int64(0), destination_rank, Int64(0))
        wire_indices = self._wire_index_view(index_stage)

        if thread_idx == Int32(0):
            cp_async_bulk_s2g(
                cute.make_ptr(
                    cutlass.Int32,
                    own_index_slab.iterator.toint() + peer_offset,
                    AddressSpace.gmem,
                    assumed_align=16,
                ),
                wire_indices.iterator,
                Int32(self.topk_slab_elements * 4),
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                score_inbox = device_workspace.tensor(self.topk_score_inbox_region)
                own_score_slab = score_inbox[local_rank, None, None]
                cp_async_bulk_s2g(
                    cute.make_ptr(
                        cutlass.Float32,
                        own_score_slab.iterator.toint() + peer_offset,
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    self._smem_workspace.tensor(
                        self.exchange_score_stage_region, smem_base
                    ).iterator,
                    Int32(self.topk_slab_elements * 4),
                )
            cute.arch.cp_async_bulk_commit_group()

        # The copy and the flag both belong to this one thread, so the wait belongs
        # to it too: no other thread committed a group.
        if thread_idx == Int32(0):
            cute.arch.cp_async_bulk_wait_group(0)
            # Each destination's `topk_ready[source]` slot has exactly one
            # producer, so the consumer can wait on an exact value.
            # Under the sentinel inbox the words are their own signal: no flag.
            if cutlass.const_expr(not self.topk_sentinel):
                own_ready_slot = device_workspace.ptr(self.topk_ready_region) + local_rank
                red_async_add_release_sys_u32(own_ready_slot.toint() + peer_offset, Int32(1))
        cute.arch.sync_threads()
        iket.range_pop()

    @cute.jit
    def _canonicalize_wire_indices(self, index_stage: cute.Tensor, thread_idx: Int32) -> None:
        """Rewrite the staged plane into the Int32 wire form."""
        wire_indices = self._wire_index_view(index_stage)
        narrow_rounds = ceil_div(self.routes_per_rank, self.threads_per_cta)
        narrowed = cute.make_rmem_tensor((narrow_rounds,), cutlass.Int32)

        for narrow_round in cutlass.range_constexpr(narrow_rounds):
            route = Int32(narrow_round * self.threads_per_cta) + thread_idx
            narrowed[narrow_round] = Int32(self.invalid_expert_marker)
            if route < Int32(self.routes_per_rank):
                raw_index = index_stage[route]
                if not self._is_unowned_expert(raw_index):
                    narrowed[narrow_round] = Int32(raw_index)

        cute.arch.sync_threads()

        for narrow_round in cutlass.range_constexpr(narrow_rounds):
            route = Int32(narrow_round * self.threads_per_cta) + thread_idx
            if route < Int32(self.routes_per_rank):
                if cutlass.const_expr(self.wire_index_bias != 0):
                    wire_indices[route] = narrowed[narrow_round] + Int32(self.wire_index_bias)
                else:
                    wire_indices[route] = narrowed[narrow_round]

    @cute.jit
    def _run_sort(
        self,
        device_workspace: DeviceWorkspace,
        local_rank: Int32,
        smem_base: cute.Pointer,
        source_rank: Int32,
        thread_idx: Int32,
    ) -> None:
        """Bucket one source rank's routes by local expert, then place them.

        Waiting on one source's slot rather than on all of them means an
        early-arriving peer gets counted while the last one is still on the wire.

        The bucket index and within-bucket offset stay in registers across the wait
        for the global prefix. Staging the sorted routes in SMEM instead would cost
        a round trip to save registers that are not scarce here.
        """
        histogram = self._smem_workspace.tensor(self.sort_histogram_region, smem_base)
        base_row = self._smem_workspace.tensor(self.sort_prefix_region, smem_base)

        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.histogram_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.histogram_bucket_count):
                histogram[bucket] = Int32(0)

        route_rounds = ceil_div(self.routes_per_rank, self.threads_per_cta)
        # Indexed with the (token, slot) pair rather than a flat route number. A
        # single integer coordinate into a rank-2 layout walks it leftmost-mode
        # fastest, which would classify one route while labelling another; the pair
        # is both the address and the route identity, so the two cannot drift.
        index_slab = device_workspace.tensor(self.topk_index_inbox_region)[source_rank, None, None]

        # Spans the release, not just the poll: what matters is when the whole CTA
        # is let through, and the same holds for every other wait range below.
        iket.range_push("sort.wait_topk")
        if cutlass.const_expr(self.topk_sentinel):
            # No flag: each word is index + 1 and reads 0 until this launch's word lands. The
            # main kernel's tail zeroed the inbox before the launch's last NVLink barrier, and a
            # peer writes it again only in its next launch, after it passed that barrier. An
            # aligned 32-bit word lands whole, so a non-zero word is this launch's; the thread
            # keeps what it read and never reads a word twice.
            polled = cute.make_rmem_tensor((route_rounds,), cutlass.Int32)
            for route_round in cutlass.range_constexpr(route_rounds):
                route = Int32(route_round * self.threads_per_cta) + thread_idx
                polled[route_round] = Int32(0)
                if route < Int32(self.routes_per_rank):
                    word_address = index_slab.iterator + route
                    word = cute.arch.load(word_address, cutlass.Int32, sem="relaxed", scope="sys")
                    while word == Int32(0):
                        word = cute.arch.load(
                            word_address, cutlass.Int32, sem="relaxed", scope="sys"
                        )
                    polled[route_round] = word
        else:
            if thread_idx == Int32(0):
                spin_wait(
                    device_workspace.ptr(self.topk_ready_region) + source_rank,
                    lambda value: value == Int32(self.topk_ready_target),
                    scope="sys",
                )
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("sort.count")
        buckets = cute.make_rmem_tensor((route_rounds,), cutlass.Int32)
        offsets = cute.make_rmem_tensor((route_rounds,), cutlass.Int32)
        local_expert_begin = local_rank * Int32(self.experts_per_rank)

        for route_round in cutlass.range_constexpr(route_rounds):
            route = Int32(route_round * self.threads_per_cta) + thread_idx
            buckets[route_round] = Int32(self.trash_bucket_index)
            if route < Int32(self.routes_per_rank):
                # One unsigned compare covers "not this rank's" and "invalid":
                # the wire marker and any wild value both land outside the window.
                if cutlass.const_expr(self.topk_sentinel):
                    wire_index = polled[route_round] - Int32(self.wire_index_bias)
                else:
                    wire_index = index_slab[route // Int32(self.topk), route % Int32(self.topk)]
                local_expert = wire_index - local_expert_begin
                if cutlass.Uint32(local_expert) < cutlass.Uint32(self.experts_per_rank):
                    buckets[route_round] = local_expert

        for route_round in cutlass.range_constexpr(route_rounds):
            offsets[route_round] = Int32(
                cute.arch.atomic_add(
                    histogram.iterator + buckets[route_round], Int32(1), sem="relaxed", scope="cta"
                )
            )
        cute.arch.sync_threads()

        expert_counts = device_workspace.tensor(self.expert_counts_by_rank_region)
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                expert_counts[source_rank, bucket] = histogram[bucket]
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("sort.wait_prefix")
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.rank_count_ready_region), Int32(1)
            )
            # With the local prefix this CTA needs only the helper's slot order, which
            # sizes_ready covers; the helper's own prefixes follow on prefix_ready.
            spin_wait(
                device_workspace.ptr(
                    self.sizes_ready_region if self.sort_local_prefix else self.prefix_ready_region
                ),
                lambda value: value == Int32(1),
                scope="gpu",
            )
        cute.arch.sync_threads()
        iket.range_pop()

        if cutlass.const_expr(self.sort_local_prefix):
            iket.range_push("sort.local_prefix")
            self._stage_local_segment_bases(
                device_workspace, smem_base, base_row, local_rank, source_rank, thread_idx
            )
            iket.range_pop()

        iket.range_push("sort.scatter")
        if cutlass.const_expr(not self.sort_local_prefix):
            # This source's segment start inside each expert, staged once: every thread
            # needs it for up to `route_rounds` unrelated buckets.
            source_expert_base = device_workspace.tensor(self.source_expert_base_region)
            for bucket_round in cutlass.range_constexpr(
                ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
            ):
                bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
                if bucket < Int32(self.local_expert_bucket_count):
                    base_row[bucket] = source_expert_base[source_rank, bucket]
            cute.arch.sync_threads()

        gather_index = device_workspace.tensor(self.gather_index_region)
        token_metadata = device_workspace.tensor(self.token_src_metadata_region)
        routed_scores = self.fc1_topk_scores_tensor(device_workspace)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_slab = device_workspace.tensor(self.topk_score_inbox_region)[
                source_rank, None, None
            ]
        dense_row_begin = source_rank * Int32(self.max_tokens_per_rank)

        for route_round in cutlass.range_constexpr(route_rounds):
            route = Int32(route_round * self.threads_per_cta) + thread_idx
            if buckets[route_round] != Int32(self.trash_bucket_index):
                source_token = route // Int32(self.topk)
                source_slot = route % Int32(self.topk)
                pool_row = base_row[buckets[route_round]] + offsets[route_round]
                gather_index[pool_row] = dense_row_begin + source_token
                token_metadata[pool_row] = TokenSrcMetadata(
                    src_rank=source_rank, src_token=source_token, src_topk=source_slot
                ).pack()
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    routed_scores[pool_row] = score_slab[source_token, source_slot]

        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            if cutlass.const_expr(self.sort_local_prefix):
                # Every reader of metadata_ready also reads the helper's prefixes, which this
                # CTA no longer waited for: the release waits until they are out as well.
                iket.range_push("sort.wait_helper_prefix")
                spin_wait(
                    device_workspace.ptr(self.prefix_ready_region),
                    lambda value: value == Int32(1),
                    scope="gpu",
                )
                iket.range_pop()
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.metadata_ready_region), Int32(1)
            )
        iket.range_pop()

    @cute.jit
    def _stage_local_segment_bases(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        base_row: cute.Tensor,
        local_rank: Int32,
        source_rank: Int32,
        thread_idx: Int32,
    ) -> None:
        """This source's segment start inside each expert, from the helper's slot order.

        The helper's arithmetic for one source, in its order: the data bases are the exclusive
        prefix of the padded loads in slot order (``_publish_padded_prefix``), and a source's
        segment follows the sources before it in ring order from this rank
        (``_publish_source_segment_bases``). sizes_ready covers the slot order, the per-slot
        loads and every count row, so the bases match the helper's to the row. The helper's own
        order is reused rather than ranked again here: with more experts than a warp it breaks
        ties between equal loads in atomic order, which a second ranking need not repeat.
        """
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        scan_input = self._smem_workspace.tensor(self.sort_scan_input_region, smem_base)
        scan_prefix = self._smem_workspace.tensor(self.sort_scan_prefix_region, smem_base)
        warp_totals = self._smem_workspace.tensor(self.sort_warp_totals_region, smem_base)
        loads_by_slot = device_workspace.tensor(self.expert_sizes_by_slot_region)
        slot_to_expert = device_workspace.tensor(self.slot_to_expert_region)
        expert_counts = device_workspace.tensor(self.expert_counts_by_rank_region)
        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)

        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            slot = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if slot < Int32(self.local_expert_bucket_count):
                # Slots past the live experts hold the padded buckets, which are empty.
                load = Int32(0)
                if slot < Int32(self.experts_per_rank):
                    load = loads_by_slot[slot]
                scan_input[slot] = padded_expert_rows(load, self.token_padding_block)
        cute.arch.sync_threads()

        smem_exclusive_prefix(
            scan_input,
            scan_prefix,
            warp_totals,
            self.threads_per_cta,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        cute.arch.sync_threads()

        ring_position = (source_rank - local_rank + Int32(self.world_size)) % Int32(self.world_size)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            slot = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if slot < Int32(self.local_expert_bucket_count):
                expert = slot_to_expert[slot]
                segment_begin = scan_prefix[slot]
                for earlier in cutlass.range_constexpr(self.world_size - 1):
                    if Int32(earlier) < ring_position:
                        earlier_rank = (local_rank + Int32(earlier)) % Int32(self.world_size)
                        segment_begin = segment_begin + expert_counts[earlier_rank, expert]
                base_row[expert] = segment_begin
        cute.arch.sync_threads()

    @cute.jit
    def _run_refine(
        self, device_workspace: DeviceWorkspace, smem_base: cute.Pointer, thread_idx: Int32
    ) -> None:
        """Rewrite the received scale factors into the layout the MMA reads."""
        group_threads = self.refine_warps_per_group * 32
        group_idx = thread_idx // Int32(group_threads)
        thread_in_group = thread_idx % Int32(group_threads)
        # Warp-uniform because a group owns whole warps, which is what makes an
        # implicitly aligned `bar.sync` on a runtime id legal.
        group_barrier = Int32(self.refine_barrier_base) + group_idx

        if cutlass.const_expr(self.sort_local_prefix):
            # The table reads the helper's prefixes only: stage it while the sort CTAs
            # still scatter.
            iket.range_push("refine.wait_prefix")
            if thread_idx == Int32(0):
                spin_wait(
                    device_workspace.ptr(self.prefix_ready_region),
                    lambda value: value == Int32(1),
                    scope="gpu",
                )
            cute.arch.sync_threads()
            iket.range_pop()
            iket.range_push("refine.table")
            expert_table = self._stage_refine_expert_table(device_workspace, smem_base, thread_idx)
            iket.range_pop()

        iket.range_push("refine.wait")
        if thread_idx == Int32(0):
            if cutlass.const_expr(self.refine_sf_first):
                # The scale-factor planes are in long before the metadata: polling their
                # counter (sys scope) first keeps its slower acquire off the path.
                spin_wait(
                    device_workspace.ptr(self.token_sf_plain_ready_region),
                    lambda value: value == Int32(self.payload_ready_target),
                    scope="sys",
                )
                spin_wait(
                    device_workspace.ptr(self.metadata_ready_region),
                    lambda value: value == Int32(self.metadata_ready_target),
                    scope="gpu",
                )
            else:
                spin_wait(
                    device_workspace.ptr(self.metadata_ready_region),
                    lambda value: value == Int32(self.metadata_ready_target),
                    scope="gpu",
                )
                spin_wait(
                    device_workspace.ptr(self.token_sf_plain_ready_region),
                    lambda value: value == Int32(self.payload_ready_target),
                    scope="sys",
                )
        cute.arch.sync_threads()
        iket.range_pop()

        group_mbarriers = self._smem_workspace.ptr(self.refine_mbarrier_region, smem_base)
        if thread_idx < Int32(self.refine_participant_groups):
            cute.arch.mbarrier_init(group_mbarriers + thread_idx, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        stage_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        if cutlass.const_expr(not self.sort_local_prefix):
            iket.range_push("refine.table")
            expert_table = self._stage_refine_expert_table(device_workspace, smem_base, thread_idx)
            iket.range_pop()
        claim = self._smem_workspace.tensor(self.refine_claim_region, smem_base)
        work_counter = device_workspace.ptr(self.refine_work_counter_region)
        block_total = device_workspace.tensor(self.refine_block_total_region)[0]

        claim_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
        )
        claimed = cute.make_rmem_tensor((self.refine_claim_fields,), cutlass.Int32)

        fc1_sf_ready = device_workspace.ptr(self.fc1_sf_ready_region)
        owning_expert = self._claim_next_block(
            claim,
            expert_table,
            work_counter,
            block_total,
            group_idx,
            thread_in_group,
            group_barrier,
            group_threads,
        )
        cute.copy(claim_atom, claim[group_idx, None], claimed)

        while claimed[self.refine_claim_block] < block_total:
            stage_state = self._refine_one_block(
                device_workspace,
                smem_base,
                stage_state,
                claimed[self.refine_claim_data_row],
                claimed[self.refine_claim_sf_row],
                claimed[self.refine_claim_live_rows],
                group_idx,
                thread_in_group,
                group_barrier,
                group_threads,
            )
            iket.range_push("refine.publish_block")
            cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)
            if thread_in_group == Int32(0):
                red_async_add_release_gpu_s32(fc1_sf_ready + owning_expert, Int32(1))
            iket.range_pop()
            owning_expert = self._claim_next_block(
                claim,
                expert_table,
                work_counter,
                block_total,
                group_idx,
                thread_in_group,
                group_barrier,
                group_threads,
            )
            cute.copy(claim_atom, claim[group_idx, None], claimed)


class RubinTokenPathMegaMoE(BlockScaledSwapAbGenphaseMoeKernel):
    """The engine's generation-phase MegaMoE kernel, its token exchange :class:`RubinTokenPathComm`.

    Descriptors as the engine kernel's. The switches are class constants that every build reads
    (:func:`bind_token_path_levers` sets them for the process).
    """

    # The token path's levers (RubinTokenPathComm documents them), off until bind_token_path_levers
    # turns them on for the process; name() tags any that is on. They change how the exchange
    # signals and when its CTAs wait, never what reaches the MMAs.
    token_comm_topk_sentinel: ClassVar[bool] = False
    token_comm_sort_local_prefix: ClassVar[bool] = False
    token_comm_refine_sf_first: ClassVar[bool] = False
    # A launch without the TopkReduce: the combine plane stays unreduced for the caller (off until
    # bind_deferred_topk_reduce turns it on for the process; name() tags it).
    defer_topk_reduce: ClassVar[bool] = False
    #: The combine plane's byte offset in the shared workspace of every kernel this class built in
    #: the process, by (world size, max tokens per rank, top-k, hidden).
    combine_plane_offsets: ClassVar[dict] = {}

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        # The engine constructor built and registered the engine exchange. Build this one with the
        # switches in its place, and the device workspace again: under the sentinel the top-k
        # inbox is a tail_reset region, which moves every symmetric region behind it. The routed-
        # row bound and the gather slack the scheduler, epilogue and mainloop were sized from are
        # the exchange's geometry, which no lever touches.
        self.token_comm = RubinTokenPathComm(
            problem_desc,
            ImplDesc(
                {
                    **impl_desc,
                    "pusher_cta_count": self.pusher_cta_count,
                    "refine_participant_groups": self.refine_participant_groups,
                    "refine_output_stages": self.refine_output_stages,
                    "topk_sentinel": self.token_comm_topk_sentinel,
                    "sort_local_prefix": self.token_comm_sort_local_prefix,
                    "refine_sf_first": self.token_comm_refine_sf_first,
                }
            ),
        )
        self._device_workspace = self._build_device_workspace()
        if not self.reduce_topk_in_kernel:
            region = self.pre_reduced_activation_region
            shape = (self.max_tokens_per_rank, self.topk, self.hidden_size)
            dense = (self.topk * self.hidden_size, self.hidden_size, 1)
            if tuple(self._device_workspace.stride(region)) != dense:
                raise ValueError(
                    f"MegaMoE combine plane strides {self._device_workspace.stride(region)}, "
                    f"not the dense {dense} of {shape}"
                )
            key = (self.world_size, *shape)
            RubinTokenPathMegaMoE.combine_plane_offsets[key] = self._device_workspace.offset(region)
            if self.defer_topk_reduce:
                self._topk_reduce = _no_topk_reduce

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
            f"{'tksent_' if self.token_comm_topk_sentinel else ''}"
            f"{'lpfx_' if self.token_comm_sort_local_prefix else ''}"
            f"{'rfsf_' if self.token_comm_refine_sf_first else ''}"
            f"{'defertopk_' if self.defer_topk_reduce else ''}"
            f"combine{self.combine_format.name}_clamp{self.gate_up_clamp}_"
            f"swiglua{self.swiglu_alpha}_swiglub{self.swiglu_beta}_"
            f"{'apply_topk_fc1' if self.apply_topk_at_fc1 else 'apply_topk_fc2'}_"
            f"{'inkernel_reduce' if self.reduce_topk_in_kernel else 'separate_reduce'}_"
            f"abstages{self._mainloop.num_weight_ab_stages}x{self._mainloop.num_token_ab_stages}"
        )


def bind_token_path_levers(
    *, topk_sentinel: bool, sort_local_prefix: bool, refine_sf_first: bool
) -> None:
    """Turn the generation kernel's token-path levers on for the process.

    Class constants, like the pusher CTAs (``kernels/moe/rubin_megamoe.py``'s
    ``_bind_pusher_ctas``): bind them before the first MegaMoE workspace query and keep them. All
    off leaves the engine kernel as shipped; any on sets those constants and makes
    :class:`RubinTokenPathMegaMoE` the kernel the engine's op builds. Turning one off in a process
    that already runs the kernel with it on raises.
    """
    switches = {
        "topk_sentinel": topk_sentinel,
        "sort_local_prefix": sort_local_prefix,
        "refine_sf_first": refine_sf_first,
    }
    if not any(switches.values()):
        return
    for arg, on in switches.items():
        if getattr(RubinTokenPathMegaMoE, _TOKEN_LEVERS[arg]) and not on:
            raise ValueError(
                f"MegaMoE lever {arg}: this process already runs the generation kernel with it on"
            )
    for arg, on in switches.items():
        if on:
            setattr(RubinTokenPathMegaMoE, _TOKEN_LEVERS[arg], True)
    _install_generation_kernel(RubinTokenPathMegaMoE)


def _no_topk_reduce(*_args) -> None:
    """The deferred TopkReduce: the launch traces no reduce (``defer_topk_reduce``)."""


def bind_deferred_topk_reduce(on: bool) -> None:
    """Turn ``defer_topk_reduce`` on for the process (lever ``moe_back_fused_norm``): every
    generation-phase launch then leaves its combine plane unreduced and returns no reduced rows.

    A class constant like the token-path levers: bind it before the first MegaMoE workspace query
    and keep it. Off leaves the kernel as it is; on sets the constant and makes
    :class:`RubinTokenPathMegaMoE` the kernel the engine's op builds. Turning it off in a process
    that already runs the kernel with it on raises.
    """
    if not on:
        if RubinTokenPathMegaMoE.defer_topk_reduce:
            raise ValueError(
                "MegaMoE lever defer_topk_reduce: this process already runs the generation kernel "
                "with it on"
            )
        return
    RubinTokenPathMegaMoE.defer_topk_reduce = True
    _install_generation_kernel(RubinTokenPathMegaMoE)


def combine_plane(
    shared_workspace: torch.Tensor, world_size: int, max_tokens: int, top_k: int, hidden: int
):
    """The combine plane of the generation kernel over a bucket's ``shared_workspace`` (U8):
    ``[max_tokens, top_k, hidden]`` BF16, a view; None when no kernel of this class was built for
    that bucket (another kernel serves it, and its launch reduces)."""
    offset = RubinTokenPathMegaMoE.combine_plane_offsets.get(
        (world_size, max_tokens, top_k, hidden)
    )
    if offset is None:
        return None
    nbytes = max_tokens * top_k * hidden * torch.bfloat16.itemsize
    if offset % 16 or offset + nbytes > shared_workspace.numel():
        raise ValueError(
            f"MegaMoE combine plane at byte {offset} (+{nbytes}) outside the shared workspace "
            f"of {shared_workspace.numel()} bytes"
        )
    flat = shared_workspace[offset : offset + nbytes].view(torch.bfloat16)
    return flat.view(max_tokens, top_k, hidden)


def _install_generation_kernel(kernel_cls: type) -> None:
    """Make ``kernel_cls`` the generation-phase kernel the engine's MegaMoE op builds, for the
    process: rebind the package's ``RubinInferenceGenphaseMegaMoE``.

    The op imports that name from the package each time it builds a kernel
    (``cute_dsl_megamoe_custom_op._construct_megamoe_kernel``); its import-time copy serves an
    ``isinstance`` check, which a subclass passes. Another class already in that place raises.
    """
    from tensorrt_llm._torch.moe.kernels import cutedsl_megamoe

    bound = cutedsl_megamoe.RubinInferenceGenphaseMegaMoE
    if bound is not kernel_cls and bound is not BlockScaledSwapAbGenphaseMoeKernel:
        raise RuntimeError(
            f"MegaMoE: the generation kernel is already {bound.__module__}.{bound.__qualname__}"
        )
    cutedsl_megamoe.RubinInferenceGenphaseMegaMoE = kernel_cls
