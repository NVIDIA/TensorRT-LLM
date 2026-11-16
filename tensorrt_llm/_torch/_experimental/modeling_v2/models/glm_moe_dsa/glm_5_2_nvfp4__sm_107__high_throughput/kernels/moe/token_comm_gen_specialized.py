# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Genphase MegaMoE token communication.

The design target is a single cross-rank round trip on the critical path. Both
outbound transfers leave at kernel entry with no routing dependency:

  exchange CTA b   sends this rank's top-k plane to peer b
  pusher CTA p     sends a slice of this rank's tokens to every peer that wants them

Everything after that is local: each exchange CTA turns into a counting-sort
worker for the peer it just talked to and then leaves, one helper CTA folds the W
count rows into per-expert sizes and prefixes, and the helper and the pushers
converge on rewriting the received scale factors into the layout the block-scaled
MMA wants.

The exchange CTAs leave instead of joining that last phase because the dependent
main kernel sizes its grid to fill the device: whatever this component holds comes
out of the main kernel's resident cluster count, so a role releases its SM as soon
as nothing depends on it.

Contrast with ``communication/nvlink_domain/token_comm.py``, which pays three
round trips (size broadcast, metadata push, activation pull) and cannot start the
bulk activation transfer until the second one lands.

Row spaces
----------
Two per-expert row spaces coexist and must not be merged:

  data rows   padded to ``token_padding_block`` (64), indexes gather_index,
              token metadata, the FC1 output pool and therefore FC2's B operand
  sf rows     padded to ``sf_padding_block``, indexes the canonical scale factor
              pool; it is pinned to 128, the row count of one block-scaled SF
              atom, because the refine block and its readiness counter are one
              padding block tall

The main kernel derives both cumulative bases from ``expert_sizes`` alone, so the
bases this component writes must reproduce the scheduler's formula bit for bit.

Payload representations
-----------------------
Activation scale factors exist twice, and the counters say which one is meant:

  token_sf_plain_ready   the linear per-token plane pushed over NVLink
  fc1_sf_ready[expert]   the canonical atom-major pool the MMA reads, counted in
                         refine blocks so one expert opens without the others
"""

import dataclasses
from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import (
    ImplDesc,
    KernelComponent,
    OptionalRequirement,
    ProblemDesc,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.nvlink_domain.symmetric_buffer import (
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
    exit,
    nanosleep,
    red_add_relaxed_sys_s32,
    red_add_release_gpu_s32,
    red_async_add_release_gpu_s32,
    red_async_add_release_sys_u32,
    tma_load_1d,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.smem_workspace import SmemWorkspace
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.helpers.utils import (
    ceil_div,
    padded_expert_rows,
    round_up,
)
from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.quant_def import QuantKind

from .constants import GEN_TOKEN_BUCKETS
from .ptx import ld_relaxed_sys_b64, st_relaxed_sys_b64


@cute.jit
def _first_matching_lane(predicate: cutlass.Boolean) -> Int32:
    """Lowest lane whose predicate holds, or -1 when none does.

    A local copy of the scheduler's primitive; it belongs beside the other warp
    helpers rather than in either caller, but moving it is a separate change.
    """
    ballot = Int32(cute.arch.vote_ballot_sync(predicate))
    first_lane = Int32(-1)
    if ballot != Int32(0):
        lowest_set_bit = ballot & (-ballot)
        first_lane = Int32(cute.arch.popc(lowest_set_bit - Int32(1)))
    return first_lane


@cute.jit
def _warp_inclusive_sum(value: Int32, lane_idx: Int32) -> Int32:
    """Inclusive prefix sum of one value per lane, across a whole warp.

    The out-of-range shuffle on the first steps is harmless: a lane below the
    step does not add what it read. Another local copy of a scheduler primitive,
    for the same reason as ``_first_matching_lane``.
    """
    inclusive = value
    for step_log in cutlass.range_constexpr(5):
        step = Int32(1 << step_log)
        previous = Int32(cute.arch.shuffle_sync(inclusive, lane_idx - step))
        if lane_idx >= step:
            inclusive = inclusive + previous
    return inclusive


@dataclasses.dataclass(frozen=True)
class GenphaseTokenCommArgs:
    """Caller-owned tensors the communication kernel reads.

    These are inputs only. Everything the component produces lives in the
    ``DeviceWorkspace`` so that the dependent main kernel can find it by region
    name without threading extra pointers through its launch.

    ``activation`` and ``activation_sf`` must already carry the padded row stride
    the component advertises (``plain_activation_row_elements`` and
    ``plain_activation_sf_row_elements``). Bulk copies move whole padded rows, so
    a tightly packed caller tensor would be read past its last row.
    """

    topk_indices: cute.Tensor  # (max_tokens_per_rank, topk), Int32 or Int64
    topk_scores: Optional[cute.Tensor]  # (max_tokens_per_rank, topk), Float32
    activation: cute.Tensor  # (max_tokens_per_rank, hidden), padded row stride
    activation_sf: cute.Tensor  # (max_tokens_per_rank, hidden / sf_vec_size), padded row stride


class GenphaseTokenComm(KernelComponent):
    """One-round-trip token dispatch for generation-phase MegaMoE.

    Owns a standalone kernel launch. ``griddepcontrol_launch_dependents`` fires at
    entry so the dependent main kernel becomes eligible immediately and streams
    weights while the payload is still in flight; the two kernels are expected to
    be co-resident, which the host must budget SMs for.
    """

    threads_per_cta: ClassVar[int] = 256
    # CTA-cluster width of the launch. No role here needs cluster-scoped anything;
    # this only changes placement, so the grid is padded up to a whole cluster and
    # the surplus CTAs leave at entry.
    launch_cluster_width: ClassVar[int] = 2
    refine_warps_per_group: ClassVar[int] = 4
    gather_index_tail_slack: ClassVar[int] = 256
    default_smem_limit_bytes: ClassVar[int] = 227 * 1024
    # highest bit 1 -> invalid
    load_bin_count: ClassVar[int] = 32
    # Per-warp payload staging depth. Two slots keep a second row in flight while
    # the first is being read out of SMEM, which matters because the wire is the
    # long pole; the slot index and mbarrier phase stay compile-time constants
    # because the token loop is fully unrolled.
    pusher_pipeline_depth: ClassVar[int] = 2
    # Named barrier ids the refine groups synchronize on. Barrier 0 belongs to the
    # block-wide `sync_threads`, so the groups start at 1.
    refine_barrier_base: ClassVar[int] = 1
    # Row-stride padding for the two linear payload planes and the top-k inbox
    # slabs. Sixteen bytes is `cp.async.bulk`'s address and length granularity,
    # which is the only reason this padding exists, so it is a constant rather
    # than a knob: nothing here benefits from a wider stride (the bank-conflict
    # skew in `refine_staging_row_stride` uses its own 128), and at sixteen the
    # padding is a no-op for any hidden extent that already fills whole atoms.
    # It must be known at construction because `register_device_workspace` sizes
    # the symmetric payload regions from it, before any caller tensor exists --
    # which is why it cannot simply be read off `args.activation`. If a caller
    # ever needs a wider stride, relax the equality check in `launch` to "a
    # multiple of sixteen, no smaller than the logical extent" rather than
    # reintroducing a descriptor field the two sides have to agree on.
    plain_row_byte_alignment: ClassVar[int] = 16

    # Symmetric regions: peers write into these through the peer pointer mapper.
    topk_index_inbox_region: ClassVar[str] = "genphase.token_comm.topk_index_inbox"
    topk_score_inbox_region: ClassVar[str] = "genphase.token_comm.topk_score_inbox"
    topk_ready_region: ClassVar[str] = "genphase.token_comm.topk_ready"
    plain_activation_region: ClassVar[str] = "genphase.token_comm.plain_activation"
    plain_activation_sf_region: ClassVar[str] = "genphase.token_comm.plain_activation_sf"
    token_data_ready_region: ClassVar[str] = "genphase.token_comm.token_data_ready"
    token_sf_plain_ready_region: ClassVar[str] = "genphase.token_comm.token_sf_plain_ready"

    # Local regions: produced and consumed on this rank only.
    expert_counts_by_rank_region: ClassVar[str] = "genphase.token_comm.expert_counts_by_rank"
    rank_count_ready_region: ClassVar[str] = "genphase.token_comm.rank_count_ready"
    expert_sizes_region: ClassVar[str] = "genphase.token_comm.expert_sizes"
    sizes_ready_region: ClassVar[str] = "genphase.token_comm.sizes_ready"
    source_expert_base_region: ClassVar[str] = "genphase.token_comm.source_expert_base"
    slot_to_expert_region: ClassVar[str] = "genphase.token_comm.slot_to_expert"
    expert_sizes_by_slot_region: ClassVar[str] = "genphase.token_comm.expert_sizes_by_slot"
    data_expert_base_region: ClassVar[str] = "genphase.token_comm.data_expert_base"
    sf_expert_base_region: ClassVar[str] = "genphase.token_comm.sf_expert_base"
    refine_block_base_region: ClassVar[str] = "genphase.token_comm.refine_block_base"
    refine_block_total_region: ClassVar[str] = "genphase.token_comm.refine_block_total"
    prefix_ready_region: ClassVar[str] = "genphase.token_comm.prefix_ready"
    token_src_metadata_region: ClassVar[str] = "genphase.token_comm.token_src_metadata"
    gather_index_region: ClassVar[str] = "genphase.token_comm.gather_index"
    fc1_topk_scores_region: ClassVar[str] = "genphase.token_comm.fc1_topk_scores"
    metadata_ready_region: ClassVar[str] = "genphase.token_comm.metadata_ready"
    canonical_activation_sf_region: ClassVar[str] = "genphase.token_comm.canonical_activation_sf"
    fc1_sf_ready_region: ClassVar[str] = "genphase.token_comm.fc1_sf_ready"
    refine_work_counter_region: ClassVar[str] = "genphase.token_comm.refine_work_counter"
    pushers_drained_region: ClassVar[str] = "genphase.token_comm.pushers_drained"

    # Tagged-word exchange (fenceless / sorted). The inboxes are symmetric and zeroed whenever the
    # barrier state is; no valid word is zero, because its tag never is.
    topk_word_inbox_region: ClassVar[str] = "genphase.token_comm.topk_word_inbox"
    count_word_inbox_region: ClassVar[str] = "genphase.token_comm.count_word_inbox"
    # Local, zeroed with the barrier state: [0] the epoch this launch tags its words with, [1] how many
    # exchange CTAs of this launch have read it.
    exchange_epoch_region: ClassVar[str] = "genphase.token_comm.exchange_epoch"
    # Local, tail-reset, on the mid-batch attention-DP decode: the helper and every pusher count themselves
    # here once their last access to the tail-reset span is behind them; the main kernel resets the span only
    # after all of them have (see `tail_reset_after_comm`).
    comm_done_region: ClassVar[str] = "genphase.token_comm.comm_done"

    # A tagged word is one aligned Int64: bits 0-15 a 16-bit payload, bits 16-31 the launch tag
    # (1..word_tag_period, so never zero), bits 32-63 a 32-bit payload (the score bits, or zero).
    word_payload_mask: ClassVar[int] = 0xFFFF
    word_tag_shift: ClassVar[int] = 16
    word_tag_period: ClassVar[int] = 0xFFFF
    # Payload of the words that fill a sorted slab past its last route.
    word_filler_payload: ClassVar[int] = 0xFFFF
    word_poll_sleep_cycles: ClassVar[int] = 64

    # SMEM regions, grouped into mutually exclusive role lifetimes.
    exchange_mbarrier_region: ClassVar[str] = "genphase.token_comm.smem.exchange_mbarrier"
    pusher_mbarrier_region: ClassVar[str] = "genphase.token_comm.smem.pusher_mbarrier"
    exchange_topk_stage_region: ClassVar[str] = "genphase.token_comm.smem.exchange_topk"
    exchange_score_stage_region: ClassVar[str] = "genphase.token_comm.smem.exchange_score"
    sort_histogram_region: ClassVar[str] = "genphase.token_comm.smem.sort_histogram"
    sort_prefix_region: ClassVar[str] = "genphase.token_comm.smem.sort_prefix"
    sort_warp_totals_region: ClassVar[str] = "genphase.token_comm.smem.sort_warp_totals"
    helper_count_matrix_region: ClassVar[str] = "genphase.token_comm.smem.helper_count_matrix"
    helper_totals_region: ClassVar[str] = "genphase.token_comm.smem.helper_totals"
    helper_scan_input_region: ClassVar[str] = "genphase.token_comm.smem.helper_scan_input"
    helper_prefix_region: ClassVar[str] = "genphase.token_comm.smem.helper_prefix"
    helper_warp_totals_region: ClassVar[str] = "genphase.token_comm.smem.helper_warp_totals"
    helper_slot_to_expert_region: ClassVar[str] = "genphase.token_comm.smem.helper_slot_to_expert"
    helper_totals_by_slot_region: ClassVar[str] = "genphase.token_comm.smem.helper_totals_by_slot"
    helper_load_bin_region: ClassVar[str] = "genphase.token_comm.smem.helper_load_bin"
    pusher_data_stage_region: ClassVar[str] = "genphase.token_comm.smem.pusher_data"
    pusher_sf_stage_region: ClassVar[str] = "genphase.token_comm.smem.pusher_sf"
    refine_mbarrier_region: ClassVar[str] = "genphase.token_comm.smem.refine_mbarrier"
    refine_claim_region: ClassVar[str] = "genphase.token_comm.smem.refine_claim"
    refine_expert_table_region: ClassVar[str] = "genphase.token_comm.smem.refine_expert_table"
    refine_staging_region: ClassVar[str] = "genphase.token_comm.smem.refine_staging"
    refine_output_region: ClassVar[str] = "genphase.token_comm.smem.refine_output"
    # Sorted exchange: the sender's per-expert counts and slab bases, and the receiver's bases of its
    # source's bins, of its source's segment in every expert, and of every expert in the data rows.
    exchange_send_count_region: ClassVar[str] = "genphase.token_comm.smem.exchange_send_count"
    exchange_send_base_region: ClassVar[str] = "genphase.token_comm.smem.exchange_send_base"
    sort_source_bin_base_region: ClassVar[str] = "genphase.token_comm.smem.sort_source_bin_base"
    sort_segment_base_region: ClassVar[str] = "genphase.token_comm.smem.sort_segment_base"
    sort_data_base_region: ClassVar[str] = "genphase.token_comm.smem.sort_data_base"

    # Fields of `refine_claim`. A claim carries everything its block implies, so
    # the block body never re-derives which expert it landed in.
    refine_claim_block: ClassVar[int] = 0
    refine_claim_data_row: ClassVar[int] = 1
    refine_claim_sf_row: ClassVar[int] = 2
    refine_claim_live_rows: ClassVar[int] = 3
    refine_claim_fields: ClassVar[int] = 4

    # Rows of `refine_expert_table`, staged once per CTA before the claim loop.
    refine_table_block_base: ClassVar[int] = 0
    refine_table_expert_size: ClassVar[int] = 1
    refine_table_data_base: ClassVar[int] = 2
    refine_table_sf_base: ClassVar[int] = 3
    refine_table_rows: ClassVar[int] = 4

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "hidden_size": int,
            "quant_kind": str,
            "topk_index_dtype": type,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "sf_padding_block": int,
            "pusher_cta_count": int,
            "refine_participant_groups": int,
            "refine_output_stages": int,
            "smem_limit_bytes": OptionalRequirement(int),
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
        mid_batch = self.max_tokens_per_rank in GEN_TOKEN_BUCKETS
        # One system-scope publication per step on that shape: one CTA sends the top-k plane to every peer and
        # publishes all ready flags after a single system fence, and the pushers count their drains on a GPU-scope
        # counter so that one of them publishes the payload flags for all of them after a single system fence.
        self.batched_publish = mid_batch
        # On the same shape, the routes pre-bucketed by the sender: each rank sends every peer only the routes that
        # peer owns, already bucketed by the peer's local expert, plus one count per expert, all as tagged words (it
        # implies the fenceless exchange). Every receiving CTA then folds the count matrix and lays out its rows
        # itself: no count-row hop to a helper CTA and no prefix hop back. Only where the expert ranking fits one
        # warp (ranks_experts_in_warp): above that the histogram ranking orders equal loads by atomic arrival, so
        # two CTAs could lay out different rows; there one helper CTA keeps publishing the layout for all of them.
        self.sorted_exchange = mid_batch and self.ranks_experts_in_warp
        # On the same shape, the top-k exchange as tagged words: each route travels as one 64-bit word that
        # carries its own launch tag next to the expert index and the score, stored with a relaxed system-scope
        # store. A receiver polls the words themselves, so the exchange needs no system fence and no ready flag.
        self.fenceless_exchange = mid_batch or self.sorted_exchange
        # On the same shape, the scale-factor refine prepares before the payload: a refine group claims its block as
        # soon as the helper's tables are out, reads the block's row sources once the routing metadata is, waits for
        # the scale-factor payload only before loading the rows, and writes the canonical block to the pool with
        # plain stores.
        self.early_refine = mid_batch
        # On the same shape the main kernel resets the tail span only after this launch's communication is over
        # on this rank. A rank whose local experts receive no token in a layer (an idle attention-DP rank's dummy
        # rows send every token to the same few experts) runs no tile: its epilogue waits for the routing metadata
        # alone and goes on to the tail reset, which zeroes the peers' payload flags (token_data_ready and
        # token_sf_plain_ready; its gather warps wait for both with an equality test) and counters its own helper
        # and pushers still use. Elsewhere nothing orders that reset either, but it comes late enough: the helper
        # and the pushers wait in the refine for every peer's scale-factor flags, and the main kernel's last
        # clusters start only on the SMs those CTAs free. The early refine lets a CTA without a block leave
        # without that wait (and the tagged exchange brings the metadata sooner), so a late peer's flags can land
        # after the reset: a gather warp never sees its target, the launch never completes, and the rank stops
        # there while its peers, which got every word and barrier signal of the launch, run on to their next
        # collective. The main kernel therefore waits, before the grid sync that precedes the reset, for its
        # gather warps (named barrier), for both payload counters and for `comm_done_target` arrivals on
        # `comm_done`.
        self.tail_reset_after_comm = self.fenceless_exchange or self.early_refine

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
        # A tagged word carries an expert index, a route index or a count in 16 bits, and the
        # sorted slab marks its unused tail with the all-ones payload.
        if self.fenceless_exchange and self.invalid_expert_marker > self.word_payload_mask:
            raise ValueError(
                f"The tagged exchange carries expert indices below {self.word_payload_mask + 1}."
            )
        if self.sorted_exchange and (
            self.routes_per_rank >= self.word_filler_payload
            or self.max_tokens_per_rank > self.word_payload_mask
        ):
            raise ValueError(
                f"The sorted exchange needs fewer than {self.word_filler_payload} routes per rank, "
                f"got {self.routes_per_rank}."
            )

    # ------------------------------------------------------------------
    # Static geometry
    # ------------------------------------------------------------------

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    @property
    def local_expert_bucket_count(self) -> int:
        """Histogram bucket count, padded because the CTA-wide scan reads four at a time."""
        return round_up(self.experts_per_rank, 4)

    @property
    def invalid_expert_marker(self) -> int:
        """Wire value for a route no rank owns.

        ``expert_count`` is outside every rank's local range, so the receiver's
        "is this mine" test sends it to the trash bucket without a second rule.
        """
        return self.expert_count

    @property
    def trash_bucket_index(self) -> int:
        return self.local_expert_bucket_count

    @property
    def histogram_bucket_count(self) -> int:
        return self.local_expert_bucket_count + 1

    @property
    def ranks_experts_in_warp(self) -> bool:
        """Whether the load ranking fits one warp at one lane per expert."""
        return self.local_expert_bucket_count <= 32

    @property
    def routes_per_rank(self) -> int:
        return self.max_tokens_per_rank * self.topk

    @property
    def topk_slab_elements(self) -> int:
        """Per-source-rank inbox slab, padded so every slab base is bulk-copy aligned."""
        return int(round_up(self.routes_per_rank, self.plain_row_byte_alignment // 4))

    @property
    def activation_dtype(self) -> type:
        return self.quant_kind.activation_dtype

    @property
    def activation_sf_dtype(self) -> type:
        return self.quant_kind.sf_dtype

    @property
    def hidden_sf(self) -> int:
        return self.hidden_size // self.quant_kind.sf_vec_size

    def _padded_row_elements(self, elements: int, dtype: type) -> int:
        elements_per_alignment = self.plain_row_byte_alignment * 8 // int(dtype.width)
        return int(round_up(elements, elements_per_alignment))

    @property
    def plain_activation_row_elements(self) -> int:
        return self._padded_row_elements(self.hidden_size, self.activation_dtype)

    @property
    def plain_activation_sf_row_elements(self) -> int:
        """``Pg``: the linear SF plane's row stride."""
        return self._padded_row_elements(self.hidden_sf, self.activation_sf_dtype)

    @property
    def pusher_warp_count(self) -> int:
        return self.threads_per_cta // 32

    @property
    def pusher_global_warp_count(self) -> int:
        return self.pusher_cta_count * self.pusher_warp_count

    @property
    def pusher_tokens_per_warp(self) -> int:
        """Tokens one warp is responsible for pushing, across both payload passes."""
        return int(ceil_div(self.max_tokens_per_rank, self.pusher_global_warp_count))

    @property
    def destination_mask_words(self) -> int:
        """Words one token's destination bitmask occupies, at one bit per peer."""
        return int(ceil_div(self.world_size, 32))

    @property
    def topk_lane_rounds(self) -> int:
        """Passes a warp needs for one token's top-k slots, at one slot per lane."""
        return int(ceil_div(self.topk, 32))

    @property
    def refine_staging_row_stride(self) -> int:
        """Byte stride between staging rows.

        128 bytes is exactly the 32 shared-memory banks, so the skew rotates each
        successive row by four banks. A lane reads sixteen bytes, the warp is
        served in four phases of eight lanes, and eight lanes four banks apart
        cover all thirty-two -- conflict-free.
        """
        return int(round_up(self.hidden_sf, 128)) + 16

    @property
    def refine_staging_group_stride(self) -> int:
        return 128 * self.refine_staging_row_stride

    @property
    def refine_atoms_per_stage(self) -> int:
        """Atoms flushed per bulk store; four keeps the 16-byte value mode whole."""
        return 4

    @property
    def refine_output_stage_elements(self) -> int:
        return self.refine_atoms_per_stage * 512

    @property
    def refine_run_columns(self) -> int:
        """Scale-factor columns one run covers: four per atom, four atoms."""
        return 4 * self.refine_atoms_per_stage

    @property
    def refine_total_atoms(self) -> int:
        """Atoms along one row. One atom spans four columns, and the hidden extent
        is required to fill whole atoms, so this is always exact."""
        return self.hidden_sf // 4

    @property
    def refine_run_count(self) -> int:
        """Runs per row, rounded up.

        The atom count need not be a multiple of the four atoms a run builds, so
        the last run is partial. It still reads and builds a whole run -- the extra
        columns live in the staging row's padding -- and only its live atoms are
        stored.
        """
        return int(ceil_div(self.refine_total_atoms, self.refine_atoms_per_stage))

    @property
    def refine_row_columns(self) -> int:
        """Padded column extent the runs traverse, so runs tile the row exactly."""
        return self.refine_run_count * self.refine_run_columns

    @property
    def route_rounds(self) -> int:
        """Routes one thread covers when a CTA walks a whole top-k plane."""
        return int(ceil_div(self.routes_per_rank, self.threads_per_cta))

    @property
    def bin_search_steps(self) -> Tuple[int, ...]:
        """Descending powers of two whose sum reaches the last bucket: the steps of a bucket search."""
        step = 1
        while step * 2 <= self.local_expert_bucket_count - 1:
            step *= 2
        steps = []
        while step >= 1:
            steps.append(step)
            step //= 2
        return tuple(steps)

    def worst_case_padded_rows(self, block: int) -> int:
        """Tight bound on ``sum_e round_up(size_e, block)`` for this rank's experts.

        Two bounds apply and the smaller wins: every expert wastes at most one
        block beyond its own routes, and no expert can hold more routes than the
        whole source token population.
        """
        source_token_capacity = self.world_size * self.max_tokens_per_rank
        routes_per_source_token = min(self.topk, self.experts_per_rank)
        route_capacity = source_token_capacity * routes_per_source_token
        active_expert_capacity = min(self.experts_per_rank, route_capacity)
        route_budget_blocks = (
            active_expert_capacity + (route_capacity - active_expert_capacity) // block
        )
        expert_bound_blocks = active_expert_capacity * int(ceil_div(source_token_capacity, block))
        return min(route_budget_blocks, expert_bound_blocks) * block

    @property
    def worst_case_data_rows(self) -> int:
        return self.worst_case_padded_rows(self.token_padding_block)

    @property
    def worst_case_sf_rows(self) -> int:
        return self.worst_case_padded_rows(self.sf_padding_block)

    @property
    def worst_case_refine_blocks(self) -> int:
        return self.worst_case_sf_rows // self.sf_padding_block

    # ------------------------------------------------------------------
    # Grid layout
    # ------------------------------------------------------------------

    @property
    def helper_cta_index(self) -> int:
        return self.world_size

    @property
    def pusher_cta_begin(self) -> int:
        return self.world_size + 1

    @property
    def grid_cta_count(self) -> int:
        return self.world_size + 1 + self.pusher_cta_count

    @property
    def launch_cluster_count(self) -> int:
        return ceil_div(self.grid_cta_count, self.launch_cluster_width)

    @property
    def topk_ready_target(self) -> int:
        """One producer per source rank, so the wait target is exact."""
        return 1

    @property
    def rank_count_ready_target(self) -> int:
        return self.world_size

    @property
    def metadata_ready_target(self) -> int:
        return self.world_size

    @property
    def comm_done_target(self) -> int:
        """The helper and the pushers: the communication CTAs that outlive the exchange CTAs."""
        return 1 + self.pusher_cta_count

    @property
    def payload_ready_target(self) -> int:
        """Every pusher publishes to every peer, whether or not it sent that peer bytes."""
        return self.pusher_cta_count * self.world_size

    # ------------------------------------------------------------------
    # GMEM workspace
    # ------------------------------------------------------------------

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        """Declare every region this component reads or writes.

        Ready counters are ``tail_reset`` so exactly the leading span can be
        zeroed between launches. This component never resets from inside: it exits
        while the dependent main kernel is still consuming ``token_data_ready``,
        so clearing counters here would strand any CTA that had not yet observed
        them. The reset therefore belongs to whoever outlives it -- normally the
        main kernel's tail, which zeroes the span between two NVLink barriers. A
        launch without that main kernel (a standalone test) has to zero on the
        host instead.
        """
        self._register_symmetric_regions(workspace)
        self._register_local_regions(workspace)

    def _register_symmetric_regions(self, workspace: DeviceWorkspace) -> None:
        if self.fenceless_exchange:
            # One tagged word per route of every source (sorted: per route the receiver owns, then
            # filler), and with the sorted exchange one per local expert of every source. Every word
            # is rewritten by every launch, and the inboxes are zeroed whenever the epoch is, so a
            # word that carries this launch's tag was written by this launch.
            workspace.register(
                self.topk_word_inbox_region,
                cutlass.Int64,
                (self.world_size, self.routes_per_rank),
                buffer_space="shared",
                stride=(self.routes_per_rank, 1),
                byte_alignment=16,
                reset="zero_on_first_allocate",
            )
            if self.sorted_exchange:
                workspace.register(
                    self.count_word_inbox_region,
                    cutlass.Int64,
                    (self.world_size, self.experts_per_rank),
                    buffer_space="shared",
                    stride=(self.experts_per_rank, 1),
                    byte_alignment=16,
                    reset="zero_on_first_allocate",
                )
        else:
            workspace.register(
                self.topk_index_inbox_region,
                cutlass.Int32,
                (self.world_size, self.max_tokens_per_rank, self.topk),
                buffer_space="shared",
                stride=(self.topk_slab_elements, self.topk, 1),
                byte_alignment=16,
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

    def _register_local_regions(self, workspace: DeviceWorkspace) -> None:
        # With the sorted exchange every count row arrives in the count inbox and every receiver
        # derives its own segment bases, so the count rows and the source bases have no GMEM copy.
        if not self.sorted_exchange:
            workspace.register(
                self.expert_counts_by_rank_region,
                cutlass.Int32,
                (self.world_size, self.local_expert_bucket_count),
                buffer_space="local",
                stride=(self.local_expert_bucket_count, 1),
                byte_alignment=16,
            )
        workspace.register(
            self.expert_sizes_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
            reset="tail_reset",
        )
        if not self.sorted_exchange:
            workspace.register(
                self.source_expert_base_region,
                cutlass.Int32,
                (self.world_size, self.local_expert_bucket_count),
                buffer_space="local",
                stride=(self.local_expert_bucket_count, 1),
                byte_alignment=16,
            )
        workspace.register(
            self.slot_to_expert_region,
            cutlass.Int32,
            (self.local_expert_bucket_count,),
            buffer_space="local",
            byte_alignment=16,
        )
        workspace.register(
            self.expert_sizes_by_slot_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
            byte_alignment=16,
        )
        # Sized to the padded bucket count rather than the live expert count: the
        # scan that fills them runs over padded buckets, and letting the tail
        # entries exist removes every boundary special case downstream.
        for region_name in (
            self.data_expert_base_region,
            self.sf_expert_base_region,
            self.refine_block_base_region,
        ):
            workspace.register(
                region_name,
                cutlass.Int32,
                (self.local_expert_bucket_count,),
                buffer_space="local",
                byte_alignment=16,
            )
        workspace.register(
            self.token_src_metadata_region,
            cutlass.Int64,
            (self.worst_case_data_rows,),
            buffer_space="local",
            byte_alignment=16,
        )
        # The FC1 gather window reaches up to a whole token tile past a tile base,
        # so the index array needs readable slack beyond the last expert -- unlike a
        # descriptor coordinate, an index is read before any bounds check. The
        # values there are irrelevant: a wild index is zero-filled by the TMA
        # descriptor bounds, and a valid-but-wrong one only lands on an N column the
        # dynamic instruction extent already excludes.
        #
        # 32-byte alignment, not 16: the gather warps read eight identifiers per
        # 256-bit access, and every run they address sits on an eight-row boundary,
        # so the base has to carry that alignment for the access to be legal.
        workspace.register(
            self.gather_index_region,
            cutlass.Int32,
            (self.worst_case_data_rows + self.gather_index_tail_slack,),
            buffer_space="local",
            byte_alignment=32,
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.fc1_topk_scores_region,
                cutlass.Float32,
                (self.worst_case_data_rows,),
                buffer_space="local",
                byte_alignment=16,
            )
        workspace.register(
            self.canonical_activation_sf_region,
            self.activation_sf_dtype,
            (self.worst_case_sf_rows * self.hidden_sf,),
            buffer_space="local",
            byte_alignment=128,
        )
        for region_name in (
            self.rank_count_ready_region,
            self.sizes_ready_region,
            self.prefix_ready_region,
            self.metadata_ready_region,
            self.refine_block_total_region,
            self.refine_work_counter_region,
        ):
            if self.sorted_exchange and region_name == self.rank_count_ready_region:
                continue
            workspace.register(
                region_name, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )
        if self.batched_publish:
            # Pushers that have drained their payload; the publishing pusher waits for all of them.
            workspace.register(
                self.pushers_drained_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                reset="tail_reset",
            )
        if self.fenceless_exchange:
            # Not tail-reset: the epoch persists from launch to launch, and the helper CTA clears the
            # reader count itself when it advances the epoch.
            workspace.register(
                self.exchange_epoch_region,
                cutlass.Int32,
                (2,),
                buffer_space="local",
                byte_alignment=16,
                reset="zero_on_first_allocate",
            )
        if self.tail_reset_after_comm:
            workspace.register(
                self.comm_done_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )
        # One slot per local expert
        workspace.register(
            self.fc1_sf_ready_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
            reset="tail_reset",
        )

    # ------------------------------------------------------------------
    # SMEM plan
    # ------------------------------------------------------------------

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
        if self.sorted_exchange:
            # The exchange CTAs run the helper's fold and ranking themselves, in this same lifetime,
            # before and after their own sends.
            for region_name in (
                self.exchange_send_count_region,
                self.exchange_send_base_region,
                self.sort_source_bin_base_region,
                self.sort_segment_base_region,
                self.sort_data_base_region,
            ):
                helper.register_tensor(
                    region_name, cutlass.Int32, (self.local_expert_bucket_count,), byte_alignment=16
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
        # The early refine writes blocks straight from the staging tile and needs no output stages.
        if not self.early_refine:
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

    @property
    def smem_workspace(self) -> SmemWorkspace:
        return self._smem_workspace

    @property
    def smem_bytes(self) -> int:
        return self._smem_workspace.total_bytes

    # ------------------------------------------------------------------
    # Typed views on workspace regions
    # ------------------------------------------------------------------

    @cute.jit
    def topk_index_inbox_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.topk_index_inbox_region)

    @cute.jit
    def topk_score_inbox_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.topk_score_inbox_region)

    @cute.jit
    def _payload_plane_by_dense_row(
        self, device_workspace: DeviceWorkspace, region_name: str
    ) -> cute.Tensor:
        """``(world * tokens, columns)`` view of a payload plane.

        The registered shape keeps the source rank as its own mode, which is how
        the push side addresses a row. The gather side wants the opposite: one flat
        row per (source rank, source token) pair, because that is what
        ``gather_index`` stores and what the FC1 gather4 descriptor indexes. Those
        two leading modes are contiguous, so merging them is a reshape rather than
        a reinterpretation, and both the column extent and the padded row stride
        are read back off the region so they stay stated only at its registration.
        """
        _, _, columns = device_workspace.region(region_name).shape
        row_elements = device_workspace.stride(region_name)[1]
        return cute.make_tensor(
            device_workspace.ptr(region_name),
            cute.make_layout(
                (self.world_size * self.max_tokens_per_rank, columns), stride=(row_elements, 1)
            ),
        )

    @cute.jit
    def plain_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.plain_activation_region)

    @cute.jit
    def plain_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.plain_activation_sf_region)

    @cute.jit
    def dense_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return self._payload_plane_by_dense_row(device_workspace, self.plain_activation_region)

    @cute.jit
    def dense_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return self._payload_plane_by_dense_row(device_workspace, self.plain_activation_sf_region)

    @cute.jit
    def canonical_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        """The SF pool in the layout the block-scaled MMA reads.

        ``tile_atom_to_shape_SF`` takes the *data* shape, not the SF shape: the
        broadcast mode it inserts already absorbs ``sf_vec_size``.
        """
        layout = tile_atom_to_shape_SF(
            (self.worst_case_sf_rows, self.hidden_size, 1), self.quant_kind.sf_vec_size
        )
        return cute.make_tensor(
            device_workspace.ptr(self.canonical_activation_sf_region),
            cute.select(layout, mode=[0, 1]),
        )

    @cute.jit
    def expert_sizes_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.expert_sizes_region)

    @cute.jit
    def slot_to_expert_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        """The processing order: which expert holds each slot the main kernel walks."""
        return device_workspace.tensor(self.slot_to_expert_region)

    @cute.jit
    def scheduler_expert_sizes_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        """Per-expert sizes in the order the FC12 scheduler must walk them.

        The scheduler rebuilds every pool base by scanning this array in index
        order, so handing it the sizes already permuted is what makes the whole
        remap invisible to it: its running prefix reproduces the layout this
        component wrote, unchanged. ``expert_sizes`` stays in expert order for
        the readers that address an expert -- the refine table and the main
        kernel's per-expert scale-factor gate.
        """
        return device_workspace.tensor(self.expert_sizes_by_slot_region)

    @cute.jit
    def token_src_metadata_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.token_src_metadata_region)

    @cute.jit
    def gather_index_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.gather_index_region)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.fc1_topk_scores_region)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    @cute.jit
    def launch(
        self,
        args: GenphaseTokenCommArgs,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        """Launch the communication kernel ahead of the dependent main kernel."""
        if cutlass.const_expr(self.apply_topk_at_fc1 and args.topk_scores is None):
            raise ValueError("apply_topk_at_fc1 requires topk_scores.")
        if cutlass.const_expr(args.topk_indices.element_type is not self.topk_index_dtype):
            raise TypeError(
                f"topk_indices must be the declared {self.topk_index_dtype}, got {args.topk_indices.element_type}."
            )
        if cutlass.const_expr(args.activation.element_type is not self.activation_dtype):
            raise TypeError(
                f"activation must be {self.activation_dtype}, got {args.activation.element_type}."
            )
        if cutlass.const_expr(args.activation_sf.element_type is not self.activation_sf_dtype):
            raise TypeError(
                f"activation_sf must be {self.activation_sf_dtype}, got {args.activation_sf.element_type}."
            )
        # Row addressing uses each plane's own layout, so the padded stride is a
        # requirement rather than something this component can paper over: a whole
        # padded row is bulk-copied, and a tighter caller stride would be read past
        # its last row. `from_dlpack` unpacks sub-byte tensors, so both strides are
        # already in logical elements and comparable.
        for plane_name, plane, row_elements in (
            ("activation", args.activation, self.plain_activation_row_elements),
            ("activation_sf", args.activation_sf, self.plain_activation_sf_row_elements),
        ):
            if cutlass.const_expr(isinstance(plane.stride[0], int)):
                if cutlass.const_expr(plane.stride[0] != row_elements):
                    raise ValueError(
                        f"{plane_name} needs the padded row stride {row_elements}, got {plane.stride[0]}."
                    )

        self._kernel(
            args.topk_indices,
            args.topk_scores,
            args.activation,
            args.activation_sf,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host.make_device_object(),
            device_workspace,
        ).launch(
            grid=[self.launch_cluster_width, 1, self.launch_cluster_count],
            cluster=(self.launch_cluster_width, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def _kernel(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        device_workspace: DeviceWorkspace,
    ) -> None:
        """Dispatch CTA roles.

        The dependent main kernel is released before anything else so its weight
        streaming overlaps the whole payload transfer.

        Every role takes the launch context explicitly rather than through
        instance attributes: the component is shared by all CTAs, and stashing
        per-CTA values on it makes the data flow invisible at the call site.
        """
        cute.arch.griddepcontrol_launch_dependents()

        cta_idx_in_cluster, _, cluster_idx = cute.arch.block_idx()
        linear_cta_idx = cluster_idx * Int32(self.launch_cluster_width) + cta_idx_in_cluster

        # Cluster padding: these CTAs hold no role and are counted by no readiness
        # target, so they leave before allocating anything rather than falling
        # through the whole body. The predicate is CTA-uniform and no barrier here
        # is cluster-scoped, so the cluster peer is unaffected; the dependent grid
        # was released above.
        if linear_cta_idx >= Int32(self.grid_cta_count):
            exit()

        thread_idx, _, _ = cute.arch.thread_idx()

        storage = cutlass.utils.SmemAllocator().allocate(self._smem_workspace.storage_class())
        smem_base = storage.buffer.data_ptr()

        device_workspace.assign_device_members(local_workspace, shared_workspace)

        if linear_cta_idx < Int32(self.world_size):
            if cutlass.const_expr(self.fenceless_exchange):
                # Dual identity: this CTA sends tagged words to peer `linear_cta_idx` and then places
                # the words that same peer sent here.
                self._run_tagged_exchange_and_sort(
                    device_workspace,
                    peer_rank_ptr_mapper,
                    local_rank,
                    topk_indices,
                    topk_scores,
                    smem_base,
                    linear_cta_idx,
                    thread_idx,
                )
            else:
                if cutlass.const_expr(self.batched_publish):
                    # CTA 0 sends the plane to every peer and publishes every ready flag
                    # once; each CTA then counting-sorts what peer `linear_cta_idx` sent.
                    if linear_cta_idx == Int32(0):
                        self._run_exchange_to_all_peers(
                            device_workspace,
                            peer_rank_ptr_mapper,
                            local_rank,
                            topk_indices,
                            topk_scores,
                            smem_base,
                            thread_idx,
                        )
                else:
                    # Dual identity: this CTA sends to peer `linear_cta_idx` and then
                    # counting-sorts what that same peer sent back.
                    self._run_exchange(
                        device_workspace,
                        peer_rank_ptr_mapper,
                        local_rank,
                        topk_indices,
                        topk_scores,
                        smem_base,
                        linear_cta_idx,
                        thread_idx,
                    )
                self._run_sort(device_workspace, local_rank, smem_base, linear_cta_idx, thread_idx)
            exit()
        elif linear_cta_idx == Int32(self.helper_cta_index):
            self._run_helper(device_workspace, local_rank, smem_base, thread_idx)
        else:
            self._run_pusher(
                device_workspace,
                peer_rank_ptr_mapper,
                local_rank,
                topk_indices,
                activation,
                activation_sf,
                smem_base,
                linear_cta_idx - Int32(self.pusher_cta_begin),
                thread_idx,
            )

        self._run_refine(device_workspace, smem_base, thread_idx)
        if cutlass.const_expr(self.tail_reset_after_comm):
            # Every access of this CTA to the tail-reset span is behind this barrier; the main kernel resets
            # the span only once the helper and every pusher have counted themselves here.
            cute.arch.sync_threads()
            if thread_idx == Int32(0):
                red_add_release_gpu_s32(device_workspace.ptr(self.comm_done_region), Int32(1))

        device_workspace.remove_device_members()

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

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
            own_ready_slot = device_workspace.ptr(self.topk_ready_region) + local_rank
            red_async_add_release_sys_u32(own_ready_slot.toint() + peer_offset, Int32(1))
        cute.arch.sync_threads()
        iket.range_pop()

    @cute.jit
    def _run_exchange_to_all_peers(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> None:
        """Send this rank's top-k plane to every peer, then publish all their ready flags at once.

        Thread ``t`` copies the staged plane to peer ``t`` and drains its own copy. One system
        fence by thread 0, between two CTA barriers, then orders every copy before the flag bumps,
        which thread ``t`` makes to peer ``t`` as relaxed reductions: the publication pattern of
        ``_publish_payload_flags``. While the payload is on the wire, a fence followed by relaxed
        reductions reaches the peers sooner than one asynchronous release per copy.
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
        # Every thread rewrote its own share of the staged plane; the copies read all of it.
        cute.arch.sync_threads()

        index_inbox = device_workspace.tensor(self.topk_index_inbox_region)
        own_index_slab = index_inbox[local_rank, None, None]
        wire_indices = self._wire_index_view(index_stage)
        # Thread t serves peer t; the modulo only keeps the other threads' unused offset in range.
        peer_offset = peer_rank_ptr_mapper.map(
            Int64(0), thread_idx % Int32(self.world_size), Int64(0)
        )

        if thread_idx < Int32(self.world_size):
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
            cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            cute.arch.fence_acq_rel_sys()
        cute.arch.sync_threads()
        if thread_idx < Int32(self.world_size):
            # Each destination's `topk_ready[source]` slot still has exactly one producer.
            own_ready_slot = device_workspace.ptr(self.topk_ready_region) + local_rank
            red_add_relaxed_sys_s32(own_ready_slot.toint() + peer_offset, Int32(1))
        iket.range_pop()

    # ------------------------------------------------------------------
    # Tagged-word exchange
    # ------------------------------------------------------------------
    #
    # A route travels as one aligned 64-bit word (see `word_tag_shift`) written with a relaxed
    # system-scope store and read with a relaxed system-scope load. Those are morally strong accesses
    # of one naturally aligned 64-bit location, so the reader sees the old word or the new word, never
    # a mix: a word that carries this launch's tag carries this launch's payload. The words are the
    # whole message (nothing else is read on the strength of having seen one), so no fence and no
    # release / acquire pair is needed between sender and receiver.
    #
    # A tag is the launch epoch plus one. Every word of every inbox is rewritten by every launch and the
    # inboxes are zeroed whenever the epoch is (allocation and the host barrier-reset fence), so before a
    # launch's word lands its slot holds the previous launch's word or zero, neither of which carries the
    # current tag. A sender cannot overwrite a word its receiver has not read yet: the next launch starts
    # on a rank only after the tail barrier, and every rank enters it only after the readers of its
    # inboxes (the exchange CTAs and, with the sorted exchange, the helper) published what they derived
    # from them, which its main kernel waits for.

    @cute.jit
    def _exchange_tag(self, device_workspace: DeviceWorkspace) -> Int32:
        """This launch's tag: the epoch plus one, so in 1..word_tag_period and never zero.

        The epoch moves on once per launch, only after every CTA that tags or checks words has read it
        (``_advance_exchange_epoch``), so all CTAs of a launch use one tag and consecutive launches differ.
        """
        epoch = cute.arch.load(
            device_workspace.ptr(self.exchange_epoch_region), Int32, sem="relaxed", scope="gpu"
        )
        return epoch + Int32(1)

    @cute.jit
    def _count_exchange_tag_reader(
        self, device_workspace: DeviceWorkspace, thread_idx: Int32
    ) -> None:
        """Count this exchange CTA as having read the epoch, once every thread of it holds its tag."""
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_add_release_gpu_s32(
                device_workspace.ptr(self.exchange_epoch_region) + Int32(1), Int32(1)
            )

    @cute.jit
    def _advance_exchange_epoch(self, device_workspace: DeviceWorkspace, thread_idx: Int32) -> None:
        """Move the epoch on for the next launch once every exchange CTA of this one has read it.

        The helper runs this after its own last use of the tag. The next launch reads the new epoch
        only after this grid has completed, and finds the reader count back at zero.
        """
        if thread_idx == Int32(0):
            epoch_pointer = device_workspace.ptr(self.exchange_epoch_region)
            spin_wait(
                epoch_pointer + Int32(1), lambda value: value == Int32(self.world_size), scope="gpu"
            )
            next_epoch = cute.arch.load(epoch_pointer, Int32, sem="relaxed", scope="gpu") + Int32(1)
            if next_epoch == Int32(self.word_tag_period):
                next_epoch = Int32(0)
            cute.arch.store(epoch_pointer, next_epoch, sem="relaxed", scope="gpu")
            cute.arch.store(epoch_pointer + Int32(1), Int32(0), sem="relaxed", scope="gpu")

    @cute.jit
    def _pack_word(self, tag: Int32, payload: Int32, high: Int32) -> Int64:
        low = (tag << Int32(self.word_tag_shift)) | (payload & Int32(self.word_payload_mask))
        return (Int64(high) << Int64(32)) | (Int64(low) & Int64(0xFFFFFFFF))

    @cute.jit
    def _word_payload(self, word: Int64) -> Int32:
        return Int32(word & Int64(self.word_payload_mask))

    @cute.jit
    def _word_high(self, word: Int64) -> Int32:
        return Int32(word >> Int64(32))

    @cute.jit
    def _poll_word(self, address: Int64, tag: Int32) -> Int64:
        """The word at ``address`` once it carries ``tag``."""
        word = ld_relaxed_sys_b64(address)
        while Int32((word >> Int64(self.word_tag_shift)) & Int64(self.word_payload_mask)) != tag:
            nanosleep(self.word_poll_sleep_cycles)
            word = ld_relaxed_sys_b64(address)
        return word

    @cute.jit
    def _poll_source_words(
        self, device_workspace: DeviceWorkspace, source_rank: Int32, thread_idx: Int32, tag: Int32
    ) -> cute.Tensor:
        """This thread's words (slots ``t, t + 256, ...``) of the slab ``source_rank`` sent here.

        Every first load is issued before the first wait so that the waits overlap.
        """
        slab_address = device_workspace.tensor(self.topk_word_inbox_region)[
            source_rank, None
        ].iterator.toint()
        words = cute.make_rmem_tensor((self.route_rounds,), cutlass.Int64)
        for route_round in cutlass.range_constexpr(self.route_rounds):
            route = Int32(route_round * self.threads_per_cta) + thread_idx
            words[route_round] = Int64(0)
            if route < Int32(self.routes_per_rank):
                words[route_round] = ld_relaxed_sys_b64(slab_address + Int64(route) * Int64(8))
        for route_round in cutlass.range_constexpr(self.route_rounds):
            route = Int32(route_round * self.threads_per_cta) + thread_idx
            if route < Int32(self.routes_per_rank):
                word = words[route_round]
                while (
                    Int32((word >> Int64(self.word_tag_shift)) & Int64(self.word_payload_mask))
                    != tag
                ):
                    nanosleep(self.word_poll_sleep_cycles)
                    word = ld_relaxed_sys_b64(slab_address + Int64(route) * Int64(8))
                words[route_round] = word
        return words

    @cute.jit
    def _wire_index(self, raw_index) -> Int32:
        """A caller index in wire form: the invalid marker for anything outside the global expert range."""
        wire = Int32(self.invalid_expert_marker)
        if not self._is_unowned_expert(raw_index):
            wire = Int32(raw_index)
        return wire

    @cute.jit
    def _score_bits_view(self, topk_scores: cute.Tensor) -> cute.Tensor:
        """Int32 view of the score plane: a score travels as its exact bits."""
        return cute.make_tensor(
            cute.recast_ptr(topk_scores.iterator, dtype=cutlass.Int32), topk_scores.layout
        )

    @cute.jit
    def _run_tagged_exchange_and_sort(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        smem_base: cute.Pointer,
        peer_rank: Int32,
        thread_idx: Int32,
    ) -> None:
        """Send peer ``peer_rank`` its tagged words, then place the words that peer sent here."""
        tag = self._exchange_tag(device_workspace)
        if cutlass.const_expr(self.sorted_exchange):
            self._run_sorted_exchange(
                device_workspace,
                peer_rank_ptr_mapper,
                local_rank,
                topk_indices,
                topk_scores,
                smem_base,
                peer_rank,
                thread_idx,
                tag,
            )
        else:
            self._run_fenceless_exchange(
                device_workspace,
                peer_rank_ptr_mapper,
                local_rank,
                topk_indices,
                topk_scores,
                peer_rank,
                thread_idx,
                tag,
            )
        self._count_exchange_tag_reader(device_workspace, thread_idx)
        if cutlass.const_expr(self.sorted_exchange):
            self._run_sorted_sort(
                device_workspace, local_rank, smem_base, peer_rank, thread_idx, tag
            )
        else:
            self._run_sort(device_workspace, local_rank, smem_base, peer_rank, thread_idx, tag)

    @cute.jit
    def _run_fenceless_exchange(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        destination_rank: Int32,
        thread_idx: Int32,
        tag: Int32,
    ) -> None:
        """Send this rank's whole top-k plane to one peer as tagged words.

        Thread ``t`` reads routes ``t, t + 256, ...`` straight from the caller's planes and stores each
        as one word (wire index, tag, score bits) into this rank's slab at the peer. A word is its own
        ready flag: no staging copy, no system fence, no counter.
        """
        iket.range_push("exchange")
        own_slab = device_workspace.tensor(self.topk_word_inbox_region)[local_rank, None]
        peer_slab_address = peer_rank_ptr_mapper.map(own_slab.iterator.toint(), destination_rank)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_bits = self._score_bits_view(topk_scores)
        for route_round in cutlass.range_constexpr(self.route_rounds):
            route = Int32(route_round * self.threads_per_cta) + thread_idx
            if route < Int32(self.routes_per_rank):
                token = route // Int32(self.topk)
                slot = route - token * Int32(self.topk)
                high = Int32(0)
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    high = score_bits[token, slot]
                word = self._pack_word(tag, self._wire_index(topk_indices[token, slot]), high)
                st_relaxed_sys_b64(peer_slab_address + Int64(route) * Int64(8), word)
        iket.range_pop()

    @cute.jit
    def _run_sorted_exchange(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        smem_base: cute.Pointer,
        destination_rank: Int32,
        thread_idx: Int32,
        tag: Int32,
    ) -> None:
        """Send one peer the routes it owns, bucketed by its local expert, and one count per expert.

        The slab holds the routes in bucket order, each as one tagged word (route index, score bits);
        the order inside a bucket is the order this CTA counted them in, and the receiver keeps it.
        The rest of the slab is filled with tagged filler words, so every word of every slab is
        rewritten by every launch. The count words go first: every receiver needs the whole count
        matrix before it can place anything.
        """
        iket.range_push("exchange")
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        send_count = self._smem_workspace.tensor(self.exchange_send_count_region, smem_base)
        send_base = self._smem_workspace.tensor(self.exchange_send_base_region, smem_base)
        warp_totals = self._smem_workspace.tensor(self.helper_warp_totals_region, smem_base)

        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                send_count[bucket] = Int32(0)
        cute.arch.sync_threads()

        expert_begin = destination_rank * Int32(self.experts_per_rank)
        local_experts = cute.make_rmem_tensor((self.route_rounds,), cutlass.Int32)
        offsets = cute.make_rmem_tensor((self.route_rounds,), cutlass.Int32)
        highs = cute.make_rmem_tensor((self.route_rounds,), cutlass.Int32)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_bits = self._score_bits_view(topk_scores)
        for route_round in cutlass.range_constexpr(self.route_rounds):
            route = Int32(route_round * self.threads_per_cta) + thread_idx
            local_experts[route_round] = Int32(-1)
            offsets[route_round] = Int32(0)
            highs[route_round] = Int32(0)
            if route < Int32(self.routes_per_rank):
                token = route // Int32(self.topk)
                slot = route - token * Int32(self.topk)
                local_expert = self._wire_index(topk_indices[token, slot]) - expert_begin
                if cutlass.Uint32(local_expert) < cutlass.Uint32(self.experts_per_rank):
                    local_experts[route_round] = local_expert
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        highs[route_round] = score_bits[token, slot]
        for route_round in cutlass.range_constexpr(self.route_rounds):
            if local_experts[route_round] >= Int32(0):
                offsets[route_round] = Int32(
                    cute.arch.atomic_add(
                        send_count.iterator + local_experts[route_round],
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                )
        cute.arch.sync_threads()
        sent_total = smem_exclusive_prefix(
            send_count, send_base, warp_totals, self.threads_per_cta, thread_idx, lane_idx, warp_idx
        )

        own_count_row = device_workspace.tensor(self.count_word_inbox_region)[local_rank, None]
        peer_count_row_address = peer_rank_ptr_mapper.map(
            own_count_row.iterator.toint(), destination_rank
        )
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.experts_per_rank, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.experts_per_rank):
                st_relaxed_sys_b64(
                    peer_count_row_address + Int64(bucket) * Int64(8),
                    self._pack_word(tag, send_count[bucket], Int32(0)),
                )

        own_slab = device_workspace.tensor(self.topk_word_inbox_region)[local_rank, None]
        peer_slab_address = peer_rank_ptr_mapper.map(own_slab.iterator.toint(), destination_rank)
        for route_round in cutlass.range_constexpr(self.route_rounds):
            if local_experts[route_round] >= Int32(0):
                route = Int32(route_round * self.threads_per_cta) + thread_idx
                position = send_base[local_experts[route_round]] + offsets[route_round]
                st_relaxed_sys_b64(
                    peer_slab_address + Int64(position) * Int64(8),
                    self._pack_word(tag, route, highs[route_round]),
                )
        for route_round in cutlass.range_constexpr(self.route_rounds):
            position = Int32(route_round * self.threads_per_cta) + thread_idx
            if (position < Int32(self.routes_per_rank)) & (position >= sent_total):
                st_relaxed_sys_b64(
                    peer_slab_address + Int64(position) * Int64(8),
                    self._pack_word(tag, Int32(self.word_filler_payload), Int32(0)),
                )
        iket.range_pop()

    @cute.jit
    def _load_count_matrix(
        self,
        device_workspace: DeviceWorkspace,
        count_matrix: cute.Tensor,
        thread_idx: Int32,
        tag: Int32,
    ) -> None:
        """Fill the SMEM (source, local expert) count matrix from the count words that carry ``tag``.

        Padded buckets read zero.
        """
        count_words_address = device_workspace.tensor(self.count_word_inbox_region).iterator.toint()
        entries = self.world_size * self.local_expert_bucket_count
        for entry_round in cutlass.range_constexpr(ceil_div(entries, self.threads_per_cta)):
            entry = Int32(entry_round * self.threads_per_cta) + thread_idx
            if entry < Int32(entries):
                source_rank = entry // Int32(self.local_expert_bucket_count)
                bucket = entry - source_rank * Int32(self.local_expert_bucket_count)
                count = Int32(0)
                if bucket < Int32(self.experts_per_rank):
                    word_index = source_rank * Int32(self.experts_per_rank) + bucket
                    count = self._word_payload(
                        self._poll_word(count_words_address + Int64(word_index) * Int64(8), tag)
                    )
                count_matrix[source_rank, bucket] = count

    @cute.jit
    def _fold_count_matrix(
        self, smem_base: cute.Pointer, count_matrix: cute.Tensor, thread_idx: Int32
    ) -> None:
        """Per-expert totals of the SMEM count matrix into the helper's totals, as the helper's fold."""
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                expert_total = Int32(0)
                for source_rank in cutlass.range_constexpr(self.world_size):
                    expert_total = expert_total + count_matrix[Int32(source_rank), bucket]
                totals[bucket] = expert_total
        cute.arch.sync_threads()

    @cute.jit
    def _bin_of_position(self, bin_base: cute.Tensor, position: Int32) -> Int32:
        """The last bucket whose base is at most ``position``.

        For a position below the source's total that is the non-empty bucket holding it: an empty bucket
        shares its base with the next one, so the search moves past it.
        """
        bucket = Int32(0)
        steps = self.bin_search_steps
        for step_index in cutlass.range_constexpr(len(steps)):
            candidate = bucket + Int32(steps[step_index])
            if candidate < Int32(self.local_expert_bucket_count):
                if bin_base[candidate] <= position:
                    bucket = candidate
        return bucket

    @cute.jit
    def _run_sorted_sort(
        self,
        device_workspace: DeviceWorkspace,
        local_rank: Int32,
        smem_base: cute.Pointer,
        source_rank: Int32,
        thread_idx: Int32,
        tag: Int32,
    ) -> None:
        """Place the routes peer ``source_rank`` sent here, with every base derived inside this CTA.

        The CTA waits on the tagged words themselves: its source's slab and the whole count matrix.
        It then runs the helper's fold, ranking and data-row prefix in SMEM (the helper publishes the
        same values for the scheduler and the refine), and its source's segment in each expert follows
        the ring order of ``_publish_source_segment_bases``. A route's row inside its segment is its
        position inside the sender's bucket.
        """
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        count_matrix = self._smem_workspace.tensor(self.helper_count_matrix_region, smem_base)
        warp_totals = self._smem_workspace.tensor(self.helper_warp_totals_region, smem_base)
        bin_base = self._smem_workspace.tensor(self.sort_source_bin_base_region, smem_base)
        segment_base = self._smem_workspace.tensor(self.sort_segment_base_region, smem_base)
        data_base = self._smem_workspace.tensor(self.sort_data_base_region, smem_base)

        iket.range_push("sort.wait_topk")
        words = self._poll_source_words(device_workspace, source_rank, thread_idx, tag)
        self._load_count_matrix(device_workspace, count_matrix, thread_idx, tag)
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("sort.layout")
        self._fold_count_matrix(smem_base, count_matrix, thread_idx)
        self._rank_experts_by_load(
            device_workspace, smem_base, thread_idx, lane_idx, warp_idx, publish=False
        )
        self._publish_padded_prefix(
            device_workspace,
            self._smem_workspace.tensor(self.helper_totals_by_slot_region, smem_base),
            self._smem_workspace.tensor(self.helper_slot_to_expert_region, smem_base),
            self._smem_workspace.tensor(self.helper_scan_input_region, smem_base),
            self._smem_workspace.tensor(self.helper_prefix_region, smem_base),
            warp_totals,
            self.token_padding_block,
            None,
            thread_idx,
            lane_idx,
            warp_idx,
            destination=data_base,
        )
        source_total = smem_exclusive_prefix(
            count_matrix[source_rank, None],
            bin_base,
            warp_totals,
            self.threads_per_cta,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        ring_position = (source_rank - local_rank + Int32(self.world_size)) % Int32(self.world_size)
        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                segment_begin = data_base[bucket]
                for ring_step in cutlass.range_constexpr(self.world_size - 1):
                    if Int32(ring_step) < ring_position:
                        earlier_source = (local_rank + Int32(ring_step)) % Int32(self.world_size)
                        segment_begin = segment_begin + count_matrix[earlier_source, bucket]
                segment_base[bucket] = segment_begin
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("sort.scatter")
        gather_index = device_workspace.tensor(self.gather_index_region)
        token_metadata = device_workspace.tensor(self.token_src_metadata_region)
        if cutlass.const_expr(self.apply_topk_at_fc1):
            routed_scores = self.fc1_topk_scores_tensor(device_workspace)
            routed_score_bits = cute.make_tensor(
                cute.recast_ptr(routed_scores.iterator, dtype=cutlass.Int32), routed_scores.layout
            )
        dense_row_begin = source_rank * Int32(self.max_tokens_per_rank)
        for route_round in cutlass.range_constexpr(self.route_rounds):
            position = Int32(route_round * self.threads_per_cta) + thread_idx
            if position < source_total:
                bucket = self._bin_of_position(bin_base, position)
                pool_row = segment_base[bucket] + (position - bin_base[bucket])
                route = self._word_payload(words[route_round])
                source_token = route // Int32(self.topk)
                source_slot = route - source_token * Int32(self.topk)
                gather_index[pool_row] = dense_row_begin + source_token
                token_metadata[pool_row] = TokenSrcMetadata(
                    src_rank=source_rank, src_token=source_token, src_topk=source_slot
                ).pack()
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    routed_score_bits[pool_row] = self._word_high(words[route_round])

        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.metadata_ready_region), Int32(1)
            )
        iket.range_pop()

    @cute.jit
    def _wire_index_view(self, index_stage: cute.Tensor) -> cute.Tensor:
        """Int32 view of the staged plane.

        For Int64 input this aliases the front half of the same bytes, which is
        where ``_canonicalize_wire_indices`` compacted the narrowed values.
        """
        if cutlass.const_expr(self.topk_index_dtype is cutlass.Int32):
            return index_stage
        return cute.make_tensor(
            cute.recast_ptr(index_stage.iterator, dtype=cutlass.Int32),
            cute.make_layout((self.topk_slab_elements,)),
        )

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
                wire_indices[route] = narrowed[narrow_round]

    @cute.jit
    def _is_unowned_expert(self, raw_index) -> cutlass.Boolean:
        """Whether a caller-supplied index falls outside the global expert range."""
        if cutlass.const_expr(self.topk_index_dtype is cutlass.Int32):
            return cutlass.Uint32(raw_index) >= cutlass.Uint32(self.expert_count)
        return cutlass.Uint64(raw_index) >= cutlass.Uint64(self.expert_count)

    @cute.jit
    def _classify_tokens(
        self, topk_indices: cute.Tensor, global_warp_idx: Int32, lane_idx: Int32
    ) -> cute.Tensor:
        """One bit per (token, peer): does this warp owe that peer this token?

        Both payload passes need the same answer from the same input, so deriving
        it twice is not just wasted work, it is a silent invariant that the two
        derivations agree. Derived once, what the passes are left with is a bit
        test against a warp-uniform word, which keeps their store predicate a
        uniform branch just as the ballot it replaces did.

        A lane owns one top-k slot and contributes the bit of that slot's owning
        rank; the warp-wide OR of those contributions is the token's mask. That
        turns one ballot per peer into one reduction per mask word, and it is what
        lets the slot count exceed a warp: further rounds only OR more bits into
        the same accumulator.

        The loads all issue before any reduction. A reduction is warp-synchronous,
        so a fused loop would put one in front of every load but the first, and the
        loads would stop overlapping.
        """
        tokens_per_warp = self.pusher_tokens_per_warp
        mask_words = self.destination_mask_words
        # Holds each lane's own contributions until the second loop reduces them
        # in place, so the two phases share one set of registers.
        masks = cute.make_rmem_tensor((tokens_per_warp, mask_words), cutlass.Uint32)

        for token_round in cutlass.range_constexpr(tokens_per_warp):
            token = global_warp_idx + Int32(token_round * self.pusher_global_warp_count)
            for word in cutlass.range_constexpr(mask_words):
                masks[token_round, word] = cutlass.Uint32(0)

            for slot_round in cutlass.range_constexpr(self.topk_lane_rounds):
                slot = Int32(slot_round * 32) + lane_idx
                lane_owns_slot = token < Int32(self.max_tokens_per_rank)
                # Only a trailing partial round can run off the top-k width, and
                # only when the width is not a whole number of warps.
                if cutlass.const_expr((slot_round + 1) * 32 > self.topk):
                    lane_owns_slot = lane_owns_slot & (slot < Int32(self.topk))

                if lane_owns_slot:
                    raw_index = topk_indices[token, slot]
                    if not self._is_unowned_expert(raw_index):
                        lane_rank = cutlass.Uint32(raw_index) // cutlass.Uint32(
                            self.experts_per_rank
                        )
                        if cutlass.const_expr(mask_words == 1):
                            masks[token_round, 0] = masks[token_round, 0] | (
                                cutlass.Uint32(1) << lane_rank
                            )
                        else:
                            lane_word = lane_rank // cutlass.Uint32(32)
                            lane_bit = cutlass.Uint32(1) << (lane_rank % cutlass.Uint32(32))
                            for word in cutlass.range_constexpr(mask_words):
                                if lane_word == cutlass.Uint32(word):
                                    masks[token_round, word] = masks[token_round, word] | lane_bit

        for token_round in cutlass.range_constexpr(tokens_per_warp):
            for word in cutlass.range_constexpr(mask_words):
                masks[token_round, word] = cutlass.Uint32(
                    cute.arch.warp_redux_sync(masks[token_round, word], "or")
                )
        return masks

    @cute.jit
    def _run_pusher(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        topk_indices: cute.Tensor,
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        smem_base: cute.Pointer,
        pusher_idx: Int32,
        thread_idx: Int32,
    ) -> None:
        """Push this rank's tokens to every peer that routed to them.

        Both planes go out before either flag does. Publishing the scale factors
        early -- so that a peer's rewrite could start while the activation was
        still on the wire -- cost a full ``cp_async_bulk_wait_group(0)`` and a
        block-wide barrier between the two passes, which held the activation pass
        behind the last scale-factor store. With a single publication point the
        pass order no longer reaches the consumer and the staging pipeline runs
        straight through.
        """
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = thread_idx % Int32(32)
        mbarriers = self._smem_workspace.ptr(self.pusher_mbarrier_region, smem_base)
        global_warp_idx = pusher_idx * Int32(self.pusher_warp_count) + warp_idx
        sf_stage = self._smem_workspace.tensor(self.pusher_sf_stage_region, smem_base)
        data_stage = self._smem_workspace.tensor(self.pusher_data_stage_region, smem_base)

        # Initialized once for both passes. The barriers used to be reset on entry
        # to each pass because the phase was a compile-time function of the round
        # index, which is only true of a pass that starts from a known phase; the
        # state below carries the real phase across both instead.
        if thread_idx < Int32(self.pusher_warp_count * self.pusher_pipeline_depth):
            cute.arch.mbarrier_init(mbarriers + thread_idx, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        iket.range_push("pusher.classify")
        destination_masks = self._classify_tokens(topk_indices, global_warp_idx, lane_idx)
        iket.range_pop()
        stage_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.pusher_pipeline_depth
        )

        iket.range_push("pusher.sf")
        stage_state = self._push_one_plane(
            peer_rank_ptr_mapper,
            local_rank,
            destination_masks,
            stage_state,
            activation_sf,
            self.plain_activation_sf_tensor(device_workspace),
            sf_stage,
            mbarriers,
            self.plain_activation_sf_row_elements * int(self.activation_sf_dtype.width) // 8,
            global_warp_idx,
            warp_idx,
            lane_idx,
        )
        iket.range_pop()

        iket.range_push("pusher.data")
        stage_state = self._push_one_plane(
            peer_rank_ptr_mapper,
            local_rank,
            destination_masks,
            stage_state,
            activation,
            self.plain_activation_tensor(device_workspace),
            data_stage,
            mbarriers,
            self.plain_activation_row_elements * int(self.activation_dtype.width) // 8,
            global_warp_idx,
            warp_idx,
            lane_idx,
        )
        iket.range_pop()

        iket.range_push("pusher.publish")
        self._publish_payload_flags(device_workspace, peer_rank_ptr_mapper, thread_idx, pusher_idx)
        iket.range_pop()

    @cute.jit
    def _push_one_plane(
        self,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        local_rank: Int32,
        destination_masks: cute.Tensor,
        stage_state: pipeline.PipelineState,
        source_plane: cute.Tensor,
        destination_plane: cute.Tensor,
        stage: cute.Tensor,
        mbarriers: cute.Pointer,
        row_bytes: int,
        global_warp_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
    ) -> pipeline.PipelineState:
        """Send one warp's tokens of one payload plane to their destination ranks.

        A token nobody asked for is skipped outright rather than staged, which
        covers both a round past the token count and a token whose whole top-k is
        invalid. The staging slot and barrier phase come from ``stage_state``,
        which advances only on the rounds that used a slot, so they stay in step
        with the barriers whichever rounds were skipped. It is returned rather than
        advanced in place: the state is a value, so the caller's copy would not see
        an advance made here, and the other pass shares these barriers.

        Every lane sees the same mask, so the skip and the advance are both
        warp-uniform and the state cannot diverge within a warp.
        """
        depth = self.pusher_pipeline_depth

        for token_round in cutlass.range_constexpr(self.pusher_tokens_per_warp):
            token = global_warp_idx + Int32(token_round * self.pusher_global_warp_count)
            any_destination = destination_masks[token_round, 0]
            for word in cutlass.range_constexpr(1, self.destination_mask_words):
                any_destination = any_destination | destination_masks[token_round, word]

            if any_destination != cutlass.Uint32(0):
                slot_mbarrier = mbarriers + (warp_idx * Int32(depth) + stage_state.index)
                slot_stage = stage[warp_idx, stage_state.index, None]

                iket.range_push("pusher.into_smem")
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(slot_mbarrier, Int32(row_bytes))
                    tma_load_1d(
                        slot_stage.iterator,
                        source_plane[token, None].iterator,
                        slot_mbarrier,
                        Int32(row_bytes),
                    )
                cute.arch.mbarrier_wait(slot_mbarrier, stage_state.phase)
                iket.range_pop()

                iket.range_push("pusher.out_of_smem")
                for destination_rank in cutlass.range_constexpr(self.world_size):
                    mask_word = destination_rank // 32
                    peer_bit = cutlass.Uint32(1 << (destination_rank % 32))
                    peer_offset = peer_rank_ptr_mapper.map(
                        Int64(0), Int32(destination_rank), Int64(0)
                    )
                    destination_row = destination_plane[local_rank, token, None]
                    if (destination_masks[token_round, mask_word] & peer_bit) != cutlass.Uint32(0):
                        with cute.arch.elect_one():
                            cp_async_bulk_s2g(
                                cute.make_ptr(
                                    destination_plane.element_type,
                                    destination_row.iterator.toint() + peer_offset,
                                    AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                slot_stage.iterator,
                                Int32(row_bytes),
                            )
                cute.arch.cp_async_bulk_commit_group()
                stage_state.advance()
                iket.range_pop()
            cute.arch.sync_warp()
            cute.arch.cp_async_bulk_wait_group(depth - 1, read=True)
        return stage_state

    @cute.jit
    def _publish_payload_flags(
        self,
        device_workspace: DeviceWorkspace,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        thread_idx: Int32,
        pusher_idx: Int32,
    ) -> None:
        """Drain both payload passes, then tell every peer this pusher is finished.

        The flags go to all peers, including ones this CTA sent nothing to: a peer
        counts publications, so a silent pusher would leave it short.

        The fan-out gives each lane a different peer, so the address is not
        warp-uniform and the asynchronous release reduction -- whose operands come
        from uniform registers -- cannot be used here. One system fence covering
        both counters is what replaces it, and covering both is why the two passes
        share this publication point.
        """
        cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.batched_publish):
            # Count this pusher as drained on a GPU-scope counter. Pusher 0 acquires the full
            # count, fences once at system scope and adds the pusher count to each peer's two
            # counters, so a peer still waits for the same total as with one publication per pusher.
            drained = device_workspace.ptr(self.pushers_drained_region)
            if thread_idx == Int32(0):
                red_add_release_gpu_s32(drained, Int32(1))
            if pusher_idx == Int32(0):
                if thread_idx == Int32(0):
                    spin_wait(
                        drained, lambda value: value == Int32(self.pusher_cta_count), scope="gpu"
                    )
                    cute.arch.fence_acq_rel_sys()
                cute.arch.sync_threads()
                if thread_idx < Int32(self.world_size):
                    for region_name in (
                        self.token_sf_plain_ready_region,
                        self.token_data_ready_region,
                    ):
                        counter_address = device_workspace.ptr(region_name).toint()
                        red_add_relaxed_sys_s32(
                            peer_rank_ptr_mapper.map(counter_address, thread_idx, Int64(0)),
                            Int32(self.pusher_cta_count),
                        )
        else:
            if thread_idx == Int32(0):
                cute.arch.fence_acq_rel_sys()
            cute.arch.sync_threads()
            if thread_idx < Int32(self.world_size):
                for region_name in (self.token_sf_plain_ready_region, self.token_data_ready_region):
                    counter_address = device_workspace.ptr(region_name).toint()
                    red_add_relaxed_sys_s32(
                        peer_rank_ptr_mapper.map(counter_address, thread_idx, Int64(0)), Int32(1)
                    )

    @cute.jit
    def _run_sort(
        self,
        device_workspace: DeviceWorkspace,
        local_rank: Int32,
        smem_base: cute.Pointer,
        source_rank: Int32,
        thread_idx: Int32,
        tag: Optional[Int32] = None,
    ) -> None:
        """Bucket one source rank's routes by local expert, then place them.

        Waiting on one source's slot rather than on all of them means an
        early-arriving peer gets counted while the last one is still on the wire.

        The bucket index and within-bucket offset stay in registers across the wait
        for the global prefix. Staging the sorted routes in SMEM instead would cost
        a round trip to save registers that are not scarce here.

        With the fenceless exchange (``tag`` given) each thread waits on its own
        tagged words instead of a ready flag, and keeps their score bits for the
        scatter.
        """
        histogram = self._smem_workspace.tensor(self.sort_histogram_region, smem_base)
        base_row = self._smem_workspace.tensor(self.sort_prefix_region, smem_base)

        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.histogram_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.histogram_bucket_count):
                histogram[bucket] = Int32(0)

        # Spans the release, not just the poll: what matters is when the whole CTA
        # is let through, and the same holds for every other wait range below.
        iket.range_push("sort.wait_topk")
        if cutlass.const_expr(self.fenceless_exchange):
            words = self._poll_source_words(device_workspace, source_rank, thread_idx, tag)
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
        route_rounds = ceil_div(self.routes_per_rank, self.threads_per_cta)
        buckets = cute.make_rmem_tensor((route_rounds,), cutlass.Int32)
        offsets = cute.make_rmem_tensor((route_rounds,), cutlass.Int32)

        # Indexed with the (token, slot) pair rather than a flat route number. A
        # single integer coordinate into a rank-2 layout walks it leftmost-mode
        # fastest, which would classify one route while labelling another; the pair
        # is both the address and the route identity, so the two cannot drift.
        if cutlass.const_expr(not self.fenceless_exchange):
            index_slab = device_workspace.tensor(self.topk_index_inbox_region)[
                source_rank, None, None
            ]
        local_expert_begin = local_rank * Int32(self.experts_per_rank)

        for route_round in cutlass.range_constexpr(route_rounds):
            route = Int32(route_round * self.threads_per_cta) + thread_idx
            buckets[route_round] = Int32(self.trash_bucket_index)
            if route < Int32(self.routes_per_rank):
                # One unsigned compare covers "not this rank's" and "invalid":
                # the wire marker and any wild value both land outside the window.
                if cutlass.const_expr(self.fenceless_exchange):
                    local_expert = self._word_payload(words[route_round]) - local_expert_begin
                else:
                    local_expert = (
                        index_slab[route // Int32(self.topk), route % Int32(self.topk)]
                        - local_expert_begin
                    )
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
            spin_wait(
                device_workspace.ptr(self.prefix_ready_region),
                lambda value: value == Int32(1),
                scope="gpu",
            )
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("sort.scatter")
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
            if cutlass.const_expr(self.fenceless_exchange):
                routed_score_bits = cute.make_tensor(
                    cute.recast_ptr(routed_scores.iterator, dtype=cutlass.Int32),
                    routed_scores.layout,
                )
            else:
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
                    if cutlass.const_expr(self.fenceless_exchange):
                        routed_score_bits[pool_row] = self._word_high(words[route_round])
                    else:
                        routed_scores[pool_row] = score_slab[source_token, source_slot]

        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(
                device_workspace.ptr(self.metadata_ready_region), Int32(1)
            )
        iket.range_pop()

    @cute.jit
    def _run_helper(
        self,
        device_workspace: DeviceWorkspace,
        local_rank: Int32,
        smem_base: cute.Pointer,
        thread_idx: Int32,
    ) -> None:
        """Fold the W count rows into per-expert sizes and both row-space prefixes.

        ``expert_sizes`` and its flag go out before the prefixes: the main kernel's
        scheduler only needs the sizes to lay out work and start streaming weights,
        so it should not wait for tables that only the sort workers read.

        The processing order is settled here as well, between the fold and that
        flag, because it is the earliest point at which every expert's load is
        known and the latest at which the per-slot sizes the flag releases can
        still be written. Everything the phase produces is covered by the flag's
        release, and the only reader of the order -- the main kernel's scheduler
        warp -- is also the only waiter on the flag, so the order needs no
        counter of its own.

        With the sorted exchange the count rows come straight from the count words
        the peers sent, and the source segment bases are left to the exchange CTAs.
        With any tagged exchange the helper also moves the exchange epoch on.
        """
        lane_idx = thread_idx % Int32(32)
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        count_matrix = self._smem_workspace.tensor(self.helper_count_matrix_region, smem_base)
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        scan_input = self._smem_workspace.tensor(self.helper_scan_input_region, smem_base)
        prefix = self._smem_workspace.tensor(self.helper_prefix_region, smem_base)
        warp_totals = self._smem_workspace.tensor(self.helper_warp_totals_region, smem_base)
        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)

        if cutlass.const_expr(self.sorted_exchange):
            iket.range_push("helper.wait_counts")
            self._load_count_matrix(
                device_workspace, count_matrix, thread_idx, self._exchange_tag(device_workspace)
            )
            cute.arch.sync_threads()
            iket.range_pop()

            iket.range_push("helper.fold")
            self._fold_count_matrix(smem_base, count_matrix, thread_idx)
        else:
            iket.range_push("helper.wait_counts")
            if thread_idx == Int32(0):
                spin_wait(
                    device_workspace.ptr(self.rank_count_ready_region),
                    lambda value: value == Int32(self.rank_count_ready_target),
                    scope="gpu",
                )
            cute.arch.sync_threads()
            iket.range_pop()

            iket.range_push("helper.fold")
            expert_counts = device_workspace.tensor(self.expert_counts_by_rank_region)
            for bucket_round in cutlass.range_constexpr(bucket_rounds):
                bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
                if bucket < Int32(self.local_expert_bucket_count):
                    expert_total = Int32(0)
                    for source_rank in cutlass.range_constexpr(self.world_size):
                        source_count = expert_counts[Int32(source_rank), bucket]
                        count_matrix[Int32(source_rank), bucket] = source_count
                        expert_total = expert_total + source_count
                    totals[bucket] = expert_total
            cute.arch.sync_threads()

        iket.range_push("helper.rank_by_load")
        self._rank_experts_by_load(device_workspace, smem_base, thread_idx, lane_idx, warp_idx)
        iket.range_pop()

        expert_sizes = device_workspace.tensor(self.expert_sizes_region)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.experts_per_rank):
                expert_sizes[bucket] = totals[bucket]
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(device_workspace.ptr(self.sizes_ready_region), Int32(1))
        iket.range_pop()

        iket.range_push("helper.prefix")
        # Every prefix below walks the order the pool is physically laid out in,
        # which is the slot order rather than the expert order. The bases
        # themselves stay keyed by expert, because every reader of them -- the
        # source segment bases, the refine expert table, and the main kernel's
        # gather -- addresses an expert rather than a slot.
        loads_in_scan_order = self._smem_workspace.tensor(
            self.helper_totals_by_slot_region, smem_base
        )
        scan_order_experts = self._smem_workspace.tensor(
            self.helper_slot_to_expert_region, smem_base
        )

        for padding_block, region_name in (
            (self.token_padding_block, self.data_expert_base_region),
            (self.sf_padding_block, self.sf_expert_base_region),
        ):
            self._publish_padded_prefix(
                device_workspace,
                loads_in_scan_order,
                scan_order_experts,
                scan_input,
                prefix,
                warp_totals,
                padding_block,
                region_name,
                thread_idx,
                lane_idx,
                warp_idx,
            )
        self._publish_refine_block_prefix(
            device_workspace,
            loads_in_scan_order,
            scan_order_experts,
            scan_input,
            prefix,
            warp_totals,
            thread_idx,
            lane_idx,
            warp_idx,
        )
        if cutlass.const_expr(not self.sorted_exchange):
            self._publish_source_segment_bases(
                device_workspace, count_matrix, local_rank, thread_idx
            )

        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            red_async_add_release_gpu_s32(device_workspace.ptr(self.prefix_ready_region), Int32(1))
        iket.range_pop()

        if cutlass.const_expr(self.fenceless_exchange):
            self._advance_exchange_epoch(device_workspace, thread_idx)

    @cute.jit
    def _rank_experts_by_load(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
        publish: bool = True,
    ) -> None:
        """Settle the order the main kernel processes this rank's experts in.

        Descending routed load, because the FC2 tiles of a token block cannot
        start until every FC1 tile of that block has finished. Leaving the biggest
        expert for last therefore ends the kernel on a long FC1 stretch followed
        by an FC2 stretch with nothing left to overlap it; taking it first moves
        that pair into the pipeline fill, and the tail becomes a run of small
        experts whose FC1-to-FC2 waits are short and cover each other.

        Two implementations, chosen by whether the expert count fits one warp.
        Both write the same four outputs through ``_place_expert_at_slot``, so
        nothing downstream can tell which one ran. Without ``publish`` only the
        two SMEM outputs are written (an exchange CTA laying out its own rows).
        """
        if cutlass.const_expr(self.ranks_experts_in_warp):
            self._rank_by_counting_in_warp(device_workspace, smem_base, lane_idx, warp_idx, publish)
        else:
            self._rank_by_load_histogram(
                device_workspace, smem_base, thread_idx, lane_idx, warp_idx, publish
            )

    @cute.jit
    def _place_expert_at_slot(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        slot: Int32,
        expert: Int32,
        routed_count: Int32,
        publish: bool = True,
    ) -> None:
        """Record one expert's position in the processing order.

        Four destinations, all built from values the caller already holds, which
        is why neither ranking implementation ever reads a load back through the
        permutation it just produced.

        A slot below ``experts_per_rank`` can name a padded bucket, and that is
        harmless rather than merely unlikely: descending order puts every expert
        holding at least one route ahead of every empty one, so such a slot always
        carries a zero size, the scheduler emits no tile for it, and the main
        kernel never translates it.
        """
        self._smem_workspace.tensor(self.helper_slot_to_expert_region, smem_base)[slot] = expert
        self._smem_workspace.tensor(self.helper_totals_by_slot_region, smem_base)[slot] = (
            routed_count
        )
        if cutlass.const_expr(publish):
            device_workspace.tensor(self.slot_to_expert_region)[slot] = expert
            if slot < Int32(self.experts_per_rank):
                device_workspace.tensor(self.expert_sizes_by_slot_region)[slot] = routed_count

    @cute.jit
    def _rank_by_counting_in_warp(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        lane_idx: Int32,
        warp_idx: Int32,
        publish: bool = True,
    ) -> None:
        """Rank the experts inside one warp by counting how many outrank each.

        Lane ``e`` carries expert ``e``'s load and each round broadcasts one
        lane's load to all of them, so a lane accumulates its own position. The
        rounds share no state and pipeline; a selection loop over the same values
        would instead serialize on masking out each round's winner, which costs a
        full cross-lane latency per round.

        The comparison is on the exact load rather than on its binade because
        exactness is free here -- the round count is the expert count either way
        -- so this branch resolves orderings the histogram branch leaves tied.
        """
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        if warp_idx == Int32(0):
            own_load = Int32(0)
            if lane_idx < Int32(self.local_expert_bucket_count):
                own_load = totals[lane_idx]
            slot = Int32(0)
            for other in cutlass.range_constexpr(self.local_expert_bucket_count):
                other_load = Int32(cute.arch.shuffle_sync(own_load, Int32(other)))
                if (other_load > own_load) | ((other_load == own_load) & (Int32(other) < lane_idx)):
                    slot = slot + Int32(1)
            if lane_idx < Int32(self.local_expert_bucket_count):
                self._place_expert_at_slot(
                    device_workspace, smem_base, slot, lane_idx, own_load, publish
                )
        cute.arch.sync_threads()

    @cute.jit
    def _rank_by_load_histogram(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
        publish: bool = True,
    ) -> None:
        """Bucket-sort the experts by descending load binade, at one thread per expert.

        Used once the expert count outgrows a warp, where counting how many
        experts outrank each would be quadratic. Cost here is instead linear in
        the expert count plus the bin count, so it barely moves as the expert
        count grows.

        A single pass, because the atomic does double duty: what it returns is
        this expert's index inside its own bin and what it leaves behind is that
        bin's population. ``_run_sort`` already carries the same trick; only the
        bucket definition differs.
        """
        totals = self._smem_workspace.tensor(self.helper_totals_region, smem_base)
        load_bins = self._smem_workspace.tensor(self.helper_load_bin_region, smem_base)

        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        keys = cute.make_rmem_tensor((bucket_rounds,), cutlass.Int32)
        offsets_in_bin = cute.make_rmem_tensor((bucket_rounds,), cutlass.Int32)

        if warp_idx == Int32(0):
            load_bins[lane_idx] = Int32(0)
        cute.arch.sync_threads()

        # Key and within-bin index stay in registers across the scan rather than
        # being recomputed after it. Rounds past the bucket count still get a
        # defined key so that no register is read on a path that never wrote it.
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            keys[bucket_round] = Int32(0)
            offsets_in_bin[bucket_round] = Int32(0)
            if bucket < Int32(self.local_expert_bucket_count):
                # No histogram spans the exact counts, so the bin is the binade;
                # inverting it is what makes the prefix below descend.
                binade = Int32(32) - Int32(cute.arch.clz(totals[bucket]))
                keys[bucket_round] = Int32(self.load_bin_count - 1) - binade
                offsets_in_bin[bucket_round] = Int32(
                    cute.arch.atomic_add(
                        load_bins.iterator + keys[bucket_round],
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                )
        cute.arch.sync_threads()

        # The bin count is exactly a warp, so the populations become slot bases in
        # five shuffles instead of through the CTA-wide shared-memory scan: no
        # warp-total round trip and no barrier inside. In place, because every
        # lane holds its own population in a register before any lane stores.
        if warp_idx == Int32(0):
            population = load_bins[lane_idx]
            load_bins[lane_idx] = _warp_inclusive_sum(population, lane_idx) - population
        cute.arch.sync_threads()

        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                slot = load_bins[keys[bucket_round]] + offsets_in_bin[bucket_round]
                self._place_expert_at_slot(
                    device_workspace, smem_base, slot, bucket, totals[bucket], publish
                )
        cute.arch.sync_threads()

    @cute.jit
    def _publish_padded_prefix(
        self,
        device_workspace: DeviceWorkspace,
        loads_in_scan_order: cute.Tensor,
        scan_order_experts: cute.Tensor,
        scan_input: cute.Tensor,
        prefix: cute.Tensor,
        warp_totals: cute.Tensor,
        padding_block: int,
        region_name: Optional[str],
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
        destination: Optional[cute.Tensor] = None,
    ) -> None:
        """Exclusive prefix of per-expert padded row counts, into one row space.

        The loads are read twice, once per row space, so the padded values go to a
        separate scan input instead of overwriting them.

        The padding call is the shared one the scheduler uses, because the main
        kernel rebuilds these same bases from the sizes it is handed and the two
        have to agree exactly.

        ``scan_order_experts`` is the slot-to-expert order, so the scan accumulates
        along slots and only the store side translates back to an expert.

        ``destination`` replaces the region, for a caller that keeps the bases in SMEM.
        """
        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if scan_entry < Int32(self.local_expert_bucket_count):
                scan_input[scan_entry] = padded_expert_rows(
                    loads_in_scan_order[scan_entry], padding_block
                )
        cute.arch.sync_threads()

        smem_exclusive_prefix(
            scan_input, prefix, warp_totals, self.threads_per_cta, thread_idx, lane_idx, warp_idx
        )
        cute.arch.sync_threads()

        if cutlass.const_expr(destination is None):
            destination = device_workspace.tensor(region_name)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if scan_entry < Int32(self.local_expert_bucket_count):
                destination[scan_order_experts[scan_entry]] = prefix[scan_entry]
        cute.arch.sync_threads()

    @cute.jit
    def _publish_refine_block_prefix(
        self,
        device_workspace: DeviceWorkspace,
        loads_in_scan_order: cute.Tensor,
        scan_order_experts: cute.Tensor,
        scan_input: cute.Tensor,
        prefix: cute.Tensor,
        warp_totals: cute.Tensor,
        thread_idx: Int32,
        lane_idx: Int32,
        warp_idx: Int32,
    ) -> None:
        """Per-expert refine block counts and their prefix.

        A block is one SF padding block of scale-factor rows, the row count of one
        canonical atom. Only blocks holding at least one live route are enumerated:
        padding rows carry scale factors the MMA's dynamic instruction extent never
        reads.

        This count is also what the main kernel waits for on ``fc1_sf_ready``, so
        the divisor here and the one it compares against are the same field.

        Scanning in the same order as the row-space bases is not just for
        symmetry: it makes the linear block index the refine workers claim run
        through the experts in the order the main kernel processes them, so the
        scale factors of the expert scheduled first are also refined first.
        """
        bucket_rounds = ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if scan_entry < Int32(self.local_expert_bucket_count):
                scan_input[scan_entry] = ceil_div(
                    loads_in_scan_order[scan_entry], Int32(self.sf_padding_block)
                )
        cute.arch.sync_threads()

        block_total = smem_exclusive_prefix(
            scan_input, prefix, warp_totals, self.threads_per_cta, thread_idx, lane_idx, warp_idx
        )
        cute.arch.sync_threads()

        destination = device_workspace.tensor(self.refine_block_base_region)
        for bucket_round in cutlass.range_constexpr(bucket_rounds):
            scan_entry = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if scan_entry < Int32(self.local_expert_bucket_count):
                destination[scan_order_experts[scan_entry]] = prefix[scan_entry]
        if thread_idx == Int32(0):
            device_workspace.tensor(self.refine_block_total_region)[0] = block_total
        cute.arch.sync_threads()

    @cute.jit
    def _publish_source_segment_bases(
        self,
        device_workspace: DeviceWorkspace,
        count_matrix: cute.Tensor,
        local_rank: Int32,
        thread_idx: Int32,
    ) -> None:
        """Where each (source rank, expert) segment starts in the data row space.

        Sources are laid out in ring order beginning at this rank, so peers write
        to different offsets within an expert rather than all piling onto its
        front.
        """
        data_expert_base = device_workspace.tensor(self.data_expert_base_region)
        source_expert_base = device_workspace.tensor(self.source_expert_base_region)

        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                segment_begin = data_expert_base[bucket]
                for ring_position in cutlass.range_constexpr(self.world_size):
                    source_rank = (local_rank + Int32(ring_position)) % Int32(self.world_size)
                    source_expert_base[source_rank, bucket] = segment_begin
                    segment_begin = segment_begin + count_matrix[source_rank, bucket]
        cute.arch.sync_threads()

    @cute.jit
    def _run_refine(
        self, device_workspace: DeviceWorkspace, smem_base: cute.Pointer, thread_idx: Int32
    ) -> None:
        """Rewrite the received scale factors into the layout the MMA reads.

        With the early refine a group claims its block from the helper's tables, then waits for the
        routing metadata and for the scale-factor payload in turn, and writes each block to the pool
        with plain stores.
        """
        group_threads = self.refine_warps_per_group * 32
        group_idx = thread_idx // Int32(group_threads)
        thread_in_group = thread_idx % Int32(group_threads)
        # Warp-uniform because a group owns whole warps, which is what makes an
        # implicitly aligned `bar.sync` on a runtime id legal.
        group_barrier = Int32(self.refine_barrier_base) + group_idx

        iket.range_push("refine.wait")
        if thread_idx == Int32(0):
            if cutlass.const_expr(self.early_refine):
                # The helper's tables are all a claim needs; a group waits for the routing metadata
                # and for the scale factors only once it holds a block.
                spin_wait(
                    device_workspace.ptr(self.prefix_ready_region),
                    lambda value: value == Int32(1),
                    scope="gpu",
                )
            else:
                spin_wait(
                    device_workspace.ptr(self.metadata_ready_region),
                    lambda value: value == Int32(self.metadata_ready_target),
                    scope="gpu",
                )
                if cutlass.const_expr(self.sorted_exchange):
                    # The exchange CTAs derive their own bases, so nothing on the metadata path
                    # acquires the helper's tables any more; the refine reads them and acquires them.
                    spin_wait(
                        device_workspace.ptr(self.prefix_ready_region),
                        lambda value: value == Int32(1),
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

        if cutlass.const_expr(self.early_refine):
            # The claim came from the helper's tables alone. The block's row sources need the routing
            # metadata and its rows need the scale factors; the group's first thread acquires each
            # once, and the group's later blocks find both already acquired.
            iket.range_push("refine.wait_metadata")
            if claimed[self.refine_claim_block] < block_total:
                if thread_in_group == Int32(0):
                    spin_wait(
                        device_workspace.ptr(self.metadata_ready_region),
                        lambda value: value == Int32(self.metadata_ready_target),
                        scope="gpu",
                    )
            cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)
            iket.range_pop()
            row_source = self._refine_row_source(
                device_workspace,
                claimed[self.refine_claim_data_row],
                claimed[self.refine_claim_live_rows],
                thread_in_group,
            )
            iket.range_push("refine.wait_payload")
            if claimed[self.refine_claim_block] < block_total:
                if thread_in_group == Int32(0):
                    spin_wait(
                        device_workspace.ptr(self.token_sf_plain_ready_region),
                        lambda value: value == Int32(self.payload_ready_target),
                        scope="sys",
                    )
            cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)
            iket.range_pop()

            while claimed[self.refine_claim_block] < block_total:
                iket.range_push("refine.into_smem")
                stage_state = self._stage_block_rows(
                    device_workspace,
                    smem_base,
                    stage_state,
                    claimed[self.refine_claim_data_row],
                    claimed[self.refine_claim_live_rows],
                    group_idx,
                    thread_in_group,
                    group_barrier,
                    group_threads,
                    row_source=row_source,
                )
                iket.range_pop()
                iket.range_push("refine.out_of_smem")
                self._store_block_direct(
                    device_workspace,
                    smem_base,
                    claimed[self.refine_claim_sf_row],
                    group_idx,
                    thread_in_group,
                    group_barrier,
                    group_threads,
                )
                iket.range_pop()
                iket.range_push("refine.publish_block")
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
                row_source = self._refine_row_source(
                    device_workspace,
                    claimed[self.refine_claim_data_row],
                    claimed[self.refine_claim_live_rows],
                    thread_in_group,
                )
        else:
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

    @cute.jit
    def _refine_row_source(
        self,
        device_workspace: DeviceWorkspace,
        data_row_begin: Int32,
        live_rows: Int32,
        thread_in_group: Int32,
    ) -> Int32:
        """The dense payload row this thread stages for a block: its gather index, or zero past the live rows."""
        row_source = Int32(0)
        if thread_in_group < live_rows:
            row_source = device_workspace.tensor(self.gather_index_region)[
                data_row_begin + thread_in_group
            ]
        return row_source

    @cute.jit
    def _store_block_direct(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        sf_row_begin: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
    ) -> None:
        """Write one whole canonical block from the staging tile straight to the pool.

        The permutation is ``_scatter_block_atoms``'s: thread ``t`` holds staging row ``t``, whose 32-bit
        word ``a`` is that row's four scale factors of atom ``a`` and belongs at word ``t`` of the atom,
        so every warp store covers 128 contiguous bytes. In the pool a block is one contiguous span: its
        atoms follow each other along K.

        The main kernel reads the pool through the async proxy, and the next block's staging load
        overwrites through it what was read here, so each thread fences both directions before the
        group meets; the caller then releases the block.
        """
        staging = self._smem_workspace.tensor(self.refine_staging_region, smem_base)
        canonical = self.canonical_activation_sf_tensor(device_workspace)
        row_words = cute.make_tensor(
            cute.recast_ptr(
                staging[group_idx, thread_in_group, None].iterator, dtype=cutlass.Int32
            ),
            cute.make_layout((self.refine_run_count * self.refine_atoms_per_stage,)),
        )
        row_runs = cute.zipped_divide(row_words, (self.refine_atoms_per_stage,))
        destination = cute.domain_offset((sf_row_begin, Int32(0)), canonical)
        # A block is one contiguous span of whole 512-byte atoms.
        block_words = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int32, destination.iterator.toint(), AddressSpace.gmem, assumed_align=16
            ),
            cute.make_layout((128, self.refine_total_atoms), stride=(1, 128)),
        )
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Int32,
            num_bits_per_copy=32 * self.refine_atoms_per_stage,
        )
        run_words = cute.make_rmem_tensor(
            cute.make_layout((self.refine_atoms_per_stage,)), cutlass.Int32
        )
        for run in cutlass.range_constexpr(self.refine_run_count):
            cute.copy(load_atom, row_runs[None, run], run_words)
            for atom in cutlass.range_constexpr(self.refine_atoms_per_stage):
                if cutlass.const_expr(
                    run * self.refine_atoms_per_stage + atom < self.refine_total_atoms
                ):
                    block_words[thread_in_group, run * self.refine_atoms_per_stage + atom] = (
                        run_words[atom]
                    )
        cute.arch.fence_proxy("async")
        cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)

    @cute.jit
    def _claim_next_block(
        self,
        claim: cute.Tensor,
        expert_table: cute.Tensor,
        work_counter: cute.Pointer,
        block_total: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
    ) -> Int32:
        """Take the next block and publish everything it implies.

        Only the group's first warp resolves the block, and the barrier that has
        to be here anyway hands the result to the rest. Resolving inside the block
        body instead would have every warp repeat the same ballot and every thread
        the same table lookups to reach one group-uniform answer.

        The fields are assembled in registers and published as one 16-byte store,
        so a reader either sees the whole claim or none of it.

        The owning expert is returned rather than stored with them: only the
        thread that publishes this block's readiness needs it, and that thread is
        in the warp that just resolved it. Widening the claim would cost the
        single-store property for a value no other thread reads.
        """
        owning_expert = Int32(-1)
        if thread_in_group < Int32(32):
            claimed_block = Int32(0)
            if thread_in_group == Int32(0):
                claimed_block = Int32(
                    cute.arch.atomic_add(work_counter, Int32(1), sem="relaxed", scope="gpu")
                )
            # Broadcast rather than staged through shared memory: the ballot below
            # needs the whole warp to agree on the block, and a shuffle gets it
            # there without a second barrier.
            claimed_block = Int32(cute.arch.shuffle_sync(claimed_block, Int32(0)))

            resolved = cute.make_rmem_tensor((self.refine_claim_fields,), cutlass.Int32)
            resolved[self.refine_claim_block] = claimed_block
            resolved[self.refine_claim_data_row] = Int32(0)
            resolved[self.refine_claim_sf_row] = Int32(0)
            resolved[self.refine_claim_live_rows] = Int32(0)
            if claimed_block < block_total:
                block_rows = Int32(self.sf_padding_block)
                owning_expert = self._resolve_block_expert(
                    expert_table, claimed_block, thread_in_group
                )
                rows_before = (
                    claimed_block - expert_table[self.refine_table_block_base, owning_expert]
                ) * block_rows
                expert_size = expert_table[self.refine_table_expert_size, owning_expert]
                data_base = expert_table[self.refine_table_data_base, owning_expert]
                sf_base = expert_table[self.refine_table_sf_base, owning_expert]
                resolved[self.refine_claim_data_row] = data_base + rows_before
                resolved[self.refine_claim_sf_row] = sf_base + rows_before
                resolved[self.refine_claim_live_rows] = cutlass.min(
                    expert_size - rows_before, block_rows
                )

            if thread_in_group == Int32(0):
                claim_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
                )
                cute.copy(claim_atom, resolved, claim[group_idx, None])
        cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)
        return owning_expert

    @cute.jit
    def _stage_refine_expert_table(
        self, device_workspace: DeviceWorkspace, smem_base: cute.Pointer, thread_idx: Int32
    ) -> cute.Tensor:
        """Bring the four per-expert columns the claim loop needs into SMEM.

        Every thread resolves its block's owning expert by scanning all buckets,
        so these belong in SMEM rather than being re-read from GMEM per block.
        """
        expert_table = self._smem_workspace.tensor(self.refine_expert_table_region, smem_base)
        block_base = device_workspace.tensor(self.refine_block_base_region)
        data_base = device_workspace.tensor(self.data_expert_base_region)
        sf_base = device_workspace.tensor(self.sf_expert_base_region)
        expert_sizes = device_workspace.tensor(self.expert_sizes_region)

        for bucket_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_bucket_count, self.threads_per_cta)
        ):
            bucket = Int32(bucket_round * self.threads_per_cta) + thread_idx
            if bucket < Int32(self.local_expert_bucket_count):
                expert_table[self.refine_table_block_base, bucket] = block_base[bucket]
                expert_table[self.refine_table_data_base, bucket] = data_base[bucket]
                expert_table[self.refine_table_sf_base, bucket] = sf_base[bucket]
                # `expert_sizes` only covers live experts; padded buckets read as
                # empty so the ownership test rejects them.
                expert_table[self.refine_table_expert_size, bucket] = Int32(0)
                if bucket < Int32(self.experts_per_rank):
                    expert_table[self.refine_table_expert_size, bucket] = expert_sizes[bucket]
        cute.arch.sync_threads()
        return expert_table

    @cute.jit
    def _refine_one_block(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        stage_state: pipeline.PipelineState,
        data_row_begin: Int32,
        sf_row_begin: Int32,
        live_rows: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
    ) -> pipeline.PipelineState:
        """Turn one 128-row block of plain scale factors into canonical atoms.

        Takes the rows to move rather than the block to resolve: which expert the
        block landed in was settled when it was claimed.

        Both global memory legs are fully coalesced and the whole permutation
        happens in shared memory addressing: rows arrive as contiguous per-token
        runs and leave as contiguous atom runs.
        """
        iket.range_push("refine.into_smem")
        stage_state = self._stage_block_rows(
            device_workspace,
            smem_base,
            stage_state,
            data_row_begin,
            live_rows,
            group_idx,
            thread_in_group,
            group_barrier,
            group_threads,
        )
        iket.range_pop()

        iket.range_push("refine.out_of_smem")
        self._scatter_block_atoms(
            device_workspace,
            smem_base,
            sf_row_begin,
            group_idx,
            thread_in_group,
            group_barrier,
            group_threads,
        )
        iket.range_pop()
        return stage_state

    @cute.jit
    def _resolve_block_expert(
        self, expert_table: cute.Tensor, block_index: Int32, lane_idx: Int32
    ) -> Int32:
        """Which local expert owns a linear refine block.

        One lane per bucket, so the answer comes out of a ballot rather than a scan
        every thread repeats. The whole group works on one block, so a scan would
        have all `group_threads` threads walk every bucket to reach the same
        answer, and would read one bucket broadcast to all lanes instead of one
        bucket per lane.

        Exactly one bucket satisfies the half-open test. Testing the upper bound
        too, rather than just taking the last bucket whose base fits, is what keeps
        empty experts from claiming a block that starts where they end.
        """
        resolved = Int32(0)
        for bucket_round in cutlass.range_constexpr(ceil_div(self.local_expert_bucket_count, 32)):
            bucket = Int32(bucket_round * 32) + lane_idx
            bucket_owns_block = cutlass.Boolean(False)
            if bucket < Int32(self.local_expert_bucket_count):
                bucket_base = expert_table[self.refine_table_block_base, bucket]
                bucket_blocks = ceil_div(
                    expert_table[self.refine_table_expert_size, bucket],
                    Int32(self.sf_padding_block),
                )
                bucket_owns_block = (bucket_base <= block_index) & (
                    block_index < bucket_base + bucket_blocks
                )
            owning_lane = _first_matching_lane(bucket_owns_block)
            if owning_lane >= Int32(0):
                resolved = Int32(bucket_round * 32) + owning_lane
        return resolved

    @cute.jit
    def _stage_block_rows(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        stage_state: pipeline.PipelineState,
        data_row_begin: Int32,
        live_rows: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
        row_source: Optional[Int32] = None,
    ) -> pipeline.PipelineState:
        """Pull 128 gathered scale-factor rows into the staging tile.

        One thread owns one row, so the padded row is a single bulk copy and the
        read side of the permutation never touches global memory again.
        ``row_source`` is this thread's gather index when the caller already read it.

        The permutation the canonical layout needs lives here, in which staging row
        a thread writes, rather than in which row it reads back. Both orders are
        equivalent, but this one leaves the ``gather_index`` read sequential and
        costs nothing: a bulk copy's destination address is per-thread anyway.

        Rows past the expert's live count are left holding whatever the previous
        block put there. They land in atoms the MMA does read, but only in N
        columns its dynamic instruction extent excludes.
        """
        plain_sf = self.dense_activation_sf_tensor(device_workspace)
        gather_index = device_workspace.tensor(self.gather_index_region)
        staging = self._smem_workspace.tensor(self.refine_staging_region, smem_base)
        mbarrier = self._smem_workspace.ptr(self.refine_mbarrier_region, smem_base) + group_idx
        # The staging tile's own column extent, which is also the destination row's
        # whole width. Reading it out of the linear plane goes past that plane's
        # `hidden_sf` columns into its row padding, which is why the plane's region
        # has to own that padding rather than just address it.
        row_bytes = self.refine_row_columns * int(self.activation_sf_dtype.width) // 8

        if thread_in_group == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, live_rows * Int32(row_bytes))
        cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)

        if thread_in_group < live_rows:
            staging_row = Int32(4) * (thread_in_group % Int32(32)) + thread_in_group // Int32(32)
            if cutlass.const_expr(row_source is None):
                tma_load_1d(
                    staging[group_idx, staging_row, None].iterator,
                    plain_sf[gather_index[data_row_begin + thread_in_group], None].iterator,
                    mbarrier,
                    Int32(row_bytes),
                )
            else:
                tma_load_1d(
                    staging[group_idx, staging_row, None].iterator,
                    plain_sf[row_source, None].iterator,
                    mbarrier,
                    Int32(row_bytes),
                )
        cute.arch.mbarrier_wait(mbarrier, stage_state.phase)
        # One stage, so every advance flips the phase. All 128 threads run the same
        # number of blocks, which is what keeps the phase group-uniform. Returned
        # because the state is a value: an advance made here is invisible upstream.
        stage_state.advance()
        return stage_state

    @cute.jit
    def _scatter_block_atoms(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        sf_row_begin: Int32,
        group_idx: Int32,
        thread_in_group: Int32,
        group_barrier: Int32,
        group_threads: int,
    ) -> None:
        """Build canonical atoms from the staging tile and store them.

        Thread ``t`` reads staging row ``t``, which the stage side arranged to hold
        the source row whose canonical position is word ``t`` of the atom. Both
        legs are therefore linear across the warp: the read walks consecutive rows
        four banks apart, the store walks consecutive words.
        """
        staging = self._smem_workspace.tensor(self.refine_staging_region, smem_base)
        output = self._smem_workspace.tensor(self.refine_output_region, smem_base)
        canonical = self.canonical_activation_sf_tensor(device_workspace)
        sf_vec_size = self.quant_kind.sf_vec_size
        columns_per_run = self.refine_run_columns

        atom_row_group = thread_in_group % Int32(4)
        atom_row = thread_in_group // Int32(4)
        # The runs tile the row exactly: the staging tile's column extent is the
        # run-aligned one, so a partial last run reads columns past `hidden_sf`
        # that belong to this row rather than to the next.
        staged_runs = cute.zipped_divide(
            staging[group_idx, thread_in_group, None], (columns_per_run,)
        )

        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.activation_sf_dtype,
            num_bits_per_copy=8 * columns_per_run,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.activation_sf_dtype, num_bits_per_copy=32
        )
        staged_values = cute.make_rmem_tensor(
            cute.make_layout((columns_per_run,)), self.activation_sf_dtype
        )
        value_atoms = cute.zipped_divide(staged_values, (4,))

        for run in cutlass.range_constexpr(self.refine_run_count):
            stage_slot = run % self.refine_output_stages
            live_atoms = min(
                self.refine_atoms_per_stage,
                self.refine_total_atoms - run * self.refine_atoms_per_stage,
            )
            # One run of atoms, addressed the way the MMA reads it: 32 rows by 4
            # row-groups against 4 scale factors by 4 atoms.
            run_output = cute.make_tensor(
                output[group_idx, stage_slot, None].iterator,
                cute.make_layout(
                    ((32, 4), (4, self.refine_atoms_per_stage)), stride=((16, 4), (1, 512))
                ),
            )
            cute.copy(load_atom, staged_runs[None, run], staged_values)
            for atom in cutlass.range_constexpr(self.refine_atoms_per_stage):
                cute.copy(
                    store_atom,
                    value_atoms[None, atom],
                    run_output[(atom_row, atom_row_group), (None, atom)],
                )
            cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)

            # `sf_row_begin` is atom-aligned and the run starts on an atom
            # boundary, so the tensor's own layout hands back the run base.
            destination = cute.domain_offset(
                (sf_row_begin, Int32(run * columns_per_run * sf_vec_size)), canonical
            )
            # Elected inside the group's first warp, not in every warp: the stage
            # slot is written cooperatively by all `group_threads` threads, so it
            # leaves as one store. A bare `elect_one` would issue the same address
            # once per warp.
            #
            # Only the live atoms are stored; a partial last run built the rest
            # from staging padding and they have no place in the pool.
            if thread_in_group // Int32(32) == Int32(0):
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(
                        destination.iterator,
                        output[group_idx, stage_slot, None].iterator,
                        Int32(live_atoms * 512),
                    )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(self.refine_output_stages - 1, read=True)
            cute.arch.barrier(barrier_id=group_barrier, number_of_threads=group_threads)

    # ------------------------------------------------------------------
    # DSL plumbing
    # ------------------------------------------------------------------

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "GenphaseTokenComm":
        if values:
            raise ValueError("GenphaseTokenComm carries no MLIR values.")
        return self


__all__ = ["GenphaseTokenComm", "GenphaseTokenCommArgs"]
