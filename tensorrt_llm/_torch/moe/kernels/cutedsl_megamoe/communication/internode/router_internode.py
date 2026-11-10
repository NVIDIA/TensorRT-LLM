"""Router of the internode component: node-local metadata push plus the remote-node histogram.

Extends :class:`_MetadataPushRouter` (already node-scoped through ``ranks_per_node``) with the
work the design document assigns to the routing prologue for remote nodes (``dispatch_prep``,
Steps 1 and 2 of the metadata phase): every DATA CTA builds, over the tokens it holds, the
node-deduplicated remote histogram ``send_count[slot][g]`` (``slot`` = remote-node slot,
``g`` = node-local expert index of the destination node, trailing entry = number of relay
messages), reserves a disjoint range of the per-node token list ``tokens_to_node[slot]`` with
one packed 64-bit atomic and fills it. A token that hits several experts of one remote node is
listed once (design invariant I4 covers the zeroing and the cross-CTA aggregation).

With ``enable_nccl_gin`` the exchange warp also performs Step 3 of the metadata phase: once every
DATA CTA has published ``list_done`` (histogram folded, token lists and header plane complete),
lanes ``0..N-2`` of its warp 0 each put this rank's slot block for one remote node, the
``send_count`` row followed by the 32-byte message headers (route words and top-k weights, the
"header plane" the DATA CTA fills while listing the tokens), into the rail peer's ``recv_splits``
block with a bundled INC signal (``splits_signal_base + source slot``). The receiver's Step 5a
(:meth:`launch_wait_remote_splits`, later the fused kernel) waits on those signals through the
NCCL-managed monotonic shadow targets (I8); its relay scan reads the landed headers, so the relay
records are pushed while the payload is still in flight.

Stage 1 (dispatch payload, decision 4 of the adaptation plan) also lives here. Every DATA CTA
publishes ``list_done`` once its ranges of ``tokens_to_node`` are filled; ``router_pack_cta_count``
pack CTAs appended after the base grid (DATA, HELPER, PUSH roles) then pack the relay messages
(``relay_message.py`` layout) into ``relay_send_buffer[slot][0, message_count)``, one warp per
message strided over all pack warps, every lane issuing all of its 16-byte loads before its first
store (one DRAM round trip per message: the pack is latency-bound). Each pack CTA publishes
``pack_done``; the HELPER lane of a remote node waits for all pack CTAs and then issues, on the
same queue pair right after its Step 3 row, ONE range put of the dense message range into the
rail peer's ``relay_recv_buffer[my slot]`` with a bundled ADD signal carrying the expert-hit
credit (sum of the row's expert counts), which is the receiver's completion target (I4, I5, I8;
the legacy coalesced ``internode_push``; "legacy" here and below means the standalone MegaMoE
implementation outside ``next/``, the non-next/ code path). The puts are issued by warp 0 of the
last pack CTA once its own share is packed (the HELPER's push duties wait for every node peer's
size broadcast first). The single-DATA-CTA pack this replaces cost about 200 us at N2R4 T=256.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, Uint64

import cuda.bindings.driver as cuda

from ...api import ImplDesc, OptionalRequirement, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.iket_compat import iket
from ...helpers.ptx_helpers import ldg_v4_b32, nanosleep, read_globaltimer_ns, stg_b64, stg_f32, stg_v4_b32
from ...helpers.utils import ceil_div
from ...quant_def import QuantKind
from ..nvlink_domain.token_comm import _MetadataPushRouter
from ..token_protocol import RELAY_RECORD_SLOT_SHIFT, RELAY_RECORD_TOPK_MARKER, RELAY_RECORD_TOPK_MASK, TokenSrcMetadata
from .lsa_peer_mapper import _device_trap
from .relay_message import RelayMessageLayout, build_relay_message_layout

# NCCL GIN signal operations (nccl4py CuTe device API): INC adds one, ADD adds signal_op_arg.
GIN_SIGNAL_OP_INC: int = 0
GIN_SIGNAL_OP_ADD: int = 1


class InternodeMetadataRouter(_MetadataPushRouter):
    """Node-scoped metadata router that also produces the remote-node dispatch metadata."""

    send_count_region: ClassVar[str] = "internode.router.send_count"
    recv_splits_region: ClassVar[str] = "internode.router.recv_splits"
    send_token_count_region: ClassVar[str] = "internode.router.send_token_count"
    histogram_done_region: ClassVar[str] = "internode.router.histogram_done"
    pack_done_region: ClassVar[str] = "internode.router.pack_done"
    list_done_region: ClassVar[str] = "internode.router.list_done"
    # Tool-free phase timers (impl key phase_stamps): one lane writes %globaltimer at each phase
    # boundary of the last launch; the bench prints them per rank (no tracer, no host jitter).
    phase_stamps_region: ClassVar[str] = "internode.debug.stamps"
    phase_stamp_names: ClassVar[Tuple[str, ...]] = (
        "router_histogram",
        "router_pack_done",
        "router_puts_issued",
        "fused_entry",
        "landing_done",
        "node_landed",
        "metadata_ready",
        "layout_computed",
        "layout_ready",
        "layout_published",
        "scan_done",
        "reduce_done",
        "return_issued",
        "origin_done",
        "tail_rail_done",
        "tail_end",
        "records_complete",
        "fc2_local_done",
    )
    # Stage 1 pack CTAs appended after the base router grid (16 warps each, one message per warp
    # per round); 48 CTAs cover the production N4R8 message count (about 640 per rank) in two rounds.
    default_router_pack_cta_count: ClassVar[int] = 48
    relay_send_buffer_region: ClassVar[str] = "internode.relay.send_buffer"
    relay_recv_buffer_region: ClassVar[str] = "internode.relay.recv_buffer"
    # Final pool layout of this rank's experts (Step 5c, written by SM0 of the fused kernel): the
    # scheduler, token_in, token_back and the FC2 epilogue read these instead of the router's
    # intra-only tables when relay tokens exist.
    layout_sizes_region: ClassVar[str] = "internode.layout.sizes"  # S(e) = I(e) + R(e)
    layout_pool_base_region: ClassVar[str] = "internode.layout.pool_base"  # P(e), padded prefix of S
    layout_intra_count_region: ClassVar[str] = "internode.layout.intra_count"  # I(e) snapshot
    layout_relay_base_region: ClassVar[str] = "internode.layout.relay_base"  # (e, source slot, rail peer) row base
    layout_push_fill_region: ClassVar[str] = "internode.layout.push_fill"  # (owner, expert, slot) rows pushed by this rank
    layout_ready_region: ClassVar[str] = "internode.layout.ready"
    # Stage 1 payload of every source slot acquired by SM0 (after layout_ready): the token_in relay
    # pulls read the message bytes only behind this flag; layout_ready / landed_ready cover the rows
    # and header planes alone (the relay scan needs nothing more).
    payload_landed_region: ClassVar[str] = "internode.layout.payload_landed"
    landed_ready_region: ClassVar[str] = "internode.layout.landed_ready"  # node peers' Step 5a completion
    final_records_region: ClassVar[str] = "internode.layout.records"  # TokenSrcMetadata per final pool token
    final_scores_region: ClassVar[str] = "internode.layout.scores"  # FC1 top-k weight per final pool token
    relay_written_region: ClassVar[str] = "internode.layout.relay_written"  # relay records written per expert
    # One 128-byte line per expert counter: the transfer warps poll them while the peers increment.
    relay_written_stride: ClassVar[int] = 32
    # Four-state generation counter of the tail's rail barrier (persistent across launches, I8).
    rail_barrier_counter_region: ClassVar[str] = "internode.rail_barrier_counter"
    # Combine: the rail peer's landing region (source slot, message, top-k slot, hidden) BF16; the
    # owners' FC2 epilogues push relay rows into it over the LSA mapping during FC2, so the rail
    # peer's reduce reads local memory (the pull-style gather over NVLink after the last FC2 tile
    # cost 35 us at N2R4 and 64 us at N4R8).
    combine_landing_rows_region: ClassVar[str] = "internode.combine.landing_rows"
    # Combine: local mirror of every node peer's per-expert fc2_done counters, indexed by node expert
    # (owner in node x experts + local expert). The owners' FC2 epilogues fan their per-tile
    # increments out to all node peers' mirrors, so the gather acquires local memory instead of
    # polling the owner over NVLink (one round trip per unseen route, plus the polling traffic).
    fc2_done_mirror_region: ClassVar[str] = "internode.combine.fc2_done_mirror"
    # Combine: node partials staged by the rail peer per (source slot, message) and returned as one
    # range put per source node into the source's landing buffer indexed by (contributor slot, message).
    intra_node_reduce_region: ClassVar[str] = "internode.combine.intra_node_reduce"
    combine_landing_region: ClassVar[str] = "internode.combine.landing"
    combine_reduce_done_region: ClassVar[str] = "internode.combine.reduce_done"  # finished gather warps
    layout_published_region: ClassVar[str] = "internode.layout.published"  # node peers' Step 5c publication
    # Pool records of relay tokens: rank = node-local rail peer holding the message, token = message
    # index, topk = marker | source slot << 4 | the message's top-k slot of this route (intra records
    # keep their plain top-k slot below the marker).
    relay_record_topk_marker: ClassVar[int] = RELAY_RECORD_TOPK_MARKER
    relay_record_slot_shift: ClassVar[int] = RELAY_RECORD_SLOT_SHIFT
    relay_record_topk_mask: ClassVar[int] = RELAY_RECORD_TOPK_MASK
    tokens_to_node_region: ClassVar[str] = "internode.router.tokens_to_node"
    message_index_region: ClassVar[str] = "internode.router.message_index"  # (slot, token) -> message index
    # GIN signal ids (legacy window layout order): Step 3 INC signals per source slot, two slots for
    # the rail barrier, the Stage 1 ADD credit per source slot, the combine arrival per source slot.
    splits_signal_base: ClassVar[int] = 0
    # Route word of a top-k slot that the destination node does not own (legacy wire contract).
    unrouted_route_word: ClassVar[int] = 0xFFFF
    smem_send_count_region: ClassVar[str] = "internode.router_smem.send_count"
    smem_token_count_region: ClassVar[str] = "internode.router_smem.token_count"
    smem_token_base_region: ClassVar[str] = "internode.router_smem.token_base"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {**super().problem_desc_require(), "hidden_size": int, "quant_kind": str}

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "ranks_per_node": int,
            "enable_nccl_gin": OptionalRequirement(bool),
            "router_pack_cta_count": OptionalRequirement(int),
            "phase_stamps": OptionalRequirement(bool),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        self.enable_nccl_gin = bool(impl_desc.get("enable_nccl_gin", False))
        self.phase_stamps = bool(impl_desc.get("phase_stamps", False))
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.activation_row_bytes = self.hidden_size * int(self.quant_kind.activation_dtype.width) // 8
        self.activation_sf_row_bytes = (
            (self.hidden_size // self.quant_kind.sf_vec_size) * int(self.quant_kind.sf_dtype.width) // 8
        )
        # One relay message per (token, remote node); a node receives at most max_tokens_per_rank
        # messages from one source rank, which sizes the per-slot send and receive ranges.
        self.max_relay_tokens_per_node = self.max_tokens_per_rank
        self.relay_message: RelayMessageLayout = build_relay_message_layout(
            hidden_bytes=self.activation_row_bytes, sf_bytes=self.activation_sf_row_bytes, topk=self.topk
        )
        self._gin_gpu_ptr = None
        self._team_rail_value = None
        self._gin_window = None
        self._gin_shared_window_offset = None
        self._activation = None
        self._activation_sf = None
        # Stage 1 pack CTAs follow the DATA / HELPER / PUSH roles of the base grid.
        self.router_pack_cta_count = self._resolve_pack_cta_count(impl_desc)
        self.router_grid_cta_count = self.router_base_grid_cta_count + self.router_pack_cta_count
        if self.router_pack_cta_count > 0:
            # The last pack CTA issues the Step 3 row and Stage 1 range puts right after the pack
            # instead of the HELPER, whose push duties first wait for every node peer's size
            # broadcast (13 to 32 us of skew under IKET; remote landing at 55 us at N4R8).
            self.router_exchange_cta_index = self.router_grid_cta_count - 1

    def _resolve_pack_cta_count(self, impl_desc: ImplDesc) -> int:
        if not self.exchanges_remote_splits:
            return 0
        free_sm_count = self.promised_launchable_sm_count - self.router_base_grid_cta_count
        requested = impl_desc.get("router_pack_cta_count", None)
        count = min(self.default_router_pack_cta_count, free_sm_count) if requested is None else int(requested)
        if count < 1 or count > free_sm_count:
            raise ValueError(
                f"router_pack_cta_count={count} must lie in [1, {free_sm_count}]: the base router grid holds "
                f"{self.router_base_grid_cta_count} of the {self.promised_launchable_sm_count} promised SMs and "
                "every router CTA must be concurrently resident."
            )
        return count

    @property
    def exchanges_remote_splits(self) -> bool:
        """True when the HELPER CTA posts Step 3 over GIN (remote nodes exist and GIN is on)."""
        return self.enable_nccl_gin and self.remote_node_count > 0

    @property
    def barrier_signal_base(self) -> int:
        return self.splits_signal_base + self.remote_node_slots

    @property
    def relay_node_signal_base(self) -> int:
        """Stage 1 ADD credit signal of each source slot (the receiver's payload completion target)."""
        return self.barrier_signal_base + 2

    @property
    def combine_arrival_signal_base(self) -> int:
        return self.relay_node_signal_base + self.remote_node_slots

    @property
    def required_signal_count(self) -> int:
        """GIN signals the dispatch path consumes (Step 3 INC, rail barrier, Stage 1 credit, combine)."""
        return self.combine_arrival_signal_base + self.remote_node_slots

    @property
    def relay_range_bytes(self) -> int:
        """Bytes of one per-slot message range (send or receive side)."""
        return self.max_relay_tokens_per_node * self.relay_message.message_bytes

    @property
    def combine_row_bytes(self) -> int:
        return self.hidden_size * 2  # BF16 node partial

    @property
    def combine_range_bytes(self) -> int:
        """Bytes of one per-slot partial range (staging or landing side)."""
        return self.max_relay_tokens_per_node * self.combine_row_bytes

    @property
    def remote_node_count(self) -> int:
        return self.num_nodes - 1

    @property
    def remote_node_slots(self) -> int:
        """Rows of the per-remote-node tables (at least one so the regions exist in every mode)."""
        return max(1, self.remote_node_count)

    @property
    def node_experts(self) -> int:
        return self.ranks_per_node * self.experts_per_rank

    @property
    def send_count_stride(self) -> int:
        """Node-local experts plus the trailing relay-message count (the legacy GIN stride)."""
        return self.node_experts + 1

    # ------------------------------------------------------------------ early header plane
    # Every slot of send_count / recv_splits is one contiguous block: the Step 3 row (padded to 16 B),
    # then the "header plane", 32 B per message (route words, then FP32 weights, each plane padded to
    # 16 B like the message planes), written by the DATA CTA while it lists the token. The early put
    # ships row and headers together behind one INC signal, so the receiver's relay scan runs during
    # the payload flight (the scan used to wait for the 4 KB messages only to read 32 B of each).

    @property
    def relay_header_plane_bytes(self) -> int:
        return ((self.topk * 4 + 15) // 16) * 16

    @property
    def relay_header_bytes(self) -> int:
        """Bytes of one message header: route words plane + weights plane."""
        return 2 * self.relay_header_plane_bytes

    @property
    def relay_header_weights_offset(self) -> int:
        return self.relay_header_plane_bytes

    @property
    def splits_row_bytes(self) -> int:
        """Bytes from a slot's block start to its header plane (the Step 3 row, padded to 16 B)."""
        return ((self.send_count_stride * 4 + 15) // 16) * 16

    @property
    def splits_row_pitch_bytes(self) -> int:
        """Bytes per slot block of send_count / recv_splits: row plus header plane, 128 B aligned."""
        raw = self.splits_row_bytes + self.max_relay_tokens_per_node * self.relay_header_bytes
        return ((raw + 127) // 128) * 128

    @property
    def splits_row_pitch_int32(self) -> int:
        return self.splits_row_pitch_bytes // 4

    def splits_rows_host(self, flat_view):
        """Host helper: the (slots, send_count_stride) rows of a flat Int32 view of send_count / recv_splits."""
        return flat_view.reshape(self.remote_node_slots, self.splits_row_pitch_int32)[:, : self.send_count_stride]

    def header_plane_host(self, flat_view, slot: int):
        """Host helper: the (max messages, relay_header_bytes) header plane of ``slot`` as uint8 rows."""
        import numpy as np

        block = np.ascontiguousarray(flat_view.reshape(self.remote_node_slots, self.splits_row_pitch_int32)[slot]).view(np.uint8)
        plane = block[self.splits_row_bytes : self.splits_row_bytes + self.max_relay_tokens_per_node * self.relay_header_bytes]
        return plane.reshape(self.max_relay_tokens_per_node, self.relay_header_bytes)

    def _register_extra_data_smem(self, data_lifetime) -> None:
        data_lifetime.register_tensor(
            self.smem_send_count_region, cutlass.Int32, (self.remote_node_slots * self.send_count_stride,)
        )
        data_lifetime.register_tensor(self.smem_token_count_region, cutlass.Int32, (self.remote_node_slots,))
        data_lifetime.register_tensor(self.smem_token_base_region, cutlass.Int32, (self.remote_node_slots,))

    def _register_router_workspace(self, workspace: DeviceWorkspace) -> None:
        super()._register_router_workspace(workspace)
        # send_count is the source of the Step 3 put, recv_splits its symmetric destination: both
        # live in the shared (LSA / GIN window) workspace so their addresses are window-relative.
        # The row pitch leaves room for the header plane behind every row (splits_row_pitch_bytes);
        # the tensor views address the rows only, the planes are addressed in bytes.
        workspace.register(
            self.send_count_region,
            cutlass.Int32,
            (self.remote_node_slots, self.send_count_stride),
            buffer_space="shared",
            stride=(self.splits_row_pitch_int32, 1),
            byte_alignment=128,
            reset="tail_reset",
        )
        workspace.register(
            self.recv_splits_region,
            cutlass.Int32,
            (self.remote_node_slots, self.send_count_stride),
            buffer_space="shared",
            stride=(self.splits_row_pitch_int32, 1),
            byte_alignment=128,
        )
        workspace.register(
            self.histogram_done_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
        )
        if self.enable_nccl_gin:
            # Stage 1 relay planes live in the window: the send ranges are the put sources, the receive
            # ranges are the symmetric destinations of the rail peers (one range per source slot).
            for region in (self.relay_send_buffer_region, self.relay_recv_buffer_region):
                workspace.register(
                    region,
                    cutlass.Uint8,
                    (self.remote_node_slots, self.max_relay_tokens_per_node, self.relay_message.message_bytes),
                    buffer_space="shared",
                    stride=(self.relay_range_bytes, self.relay_message.message_bytes, 1),
                    byte_alignment=128,
                )
            workspace.register(self.pack_done_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset")
            workspace.register(self.list_done_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset")
            if self.phase_stamps:
                workspace.register(
                    self.phase_stamps_region,
                    cutlass.Uint64,
                    (len(self.phase_stamp_names),),
                    buffer_space="local",
                    byte_alignment=16,
                )
            experts = self.experts_per_rank
            # Final sizes and bases are read by the node peers too (combine: the rail peer derives an
            # owner's fc2_done target from the owner's final expert size), so they live in the window.
            for region in (self.layout_sizes_region, self.layout_pool_base_region):
                workspace.register(region, cutlass.Int32, (experts,), buffer_space="shared", byte_alignment=16)
            workspace.register(self.layout_intra_count_region, cutlass.Int32, (experts,), buffer_space="local")
            # The node peers that received the messages push the relay records: they read this rank's
            # relay_base, store into its final records / scores and increment its relay_written
            # counters over the LSA arena, so these four regions live in the shared workspace.
            bucket_shape = (experts, self.remote_node_slots, self.ranks_per_node)
            bucket_stride = (self.remote_node_slots * self.ranks_per_node, self.ranks_per_node, 1)
            workspace.register(
                self.layout_relay_base_region, cutlass.Int32, bucket_shape, buffer_space="shared", stride=bucket_stride
            )
            # Rows this rank pushed into the (owner, expert, source slot) buckets of its node peers.
            workspace.register(
                self.layout_push_fill_region,
                cutlass.Int32,
                (self.ranks_per_node, experts, self.remote_node_slots),
                buffer_space="local",
                stride=(experts * self.remote_node_slots, self.remote_node_slots, 1),
                reset="tail_reset",
            )
            workspace.register(self.layout_ready_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset")
            workspace.register(self.payload_landed_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset")
            workspace.register(
                self.relay_written_region,
                cutlass.Int32,
                (experts, self.relay_written_stride),
                buffer_space="shared",
                stride=(self.relay_written_stride, 1),
                byte_alignment=128,
                reset="tail_reset",
            )
            workspace.register(
                self.landed_ready_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset"
            )
            workspace.register(
                self.final_records_region, cutlass.Int64, (self.worst_case_token_count,), buffer_space="shared"
            )
            for region in (self.intra_node_reduce_region, self.combine_landing_region):
                workspace.register(
                    region,
                    cutlass.BFloat16,
                    (self.remote_node_slots, self.max_relay_tokens_per_node, self.hidden_size),
                    buffer_space="shared",
                    stride=(self.max_relay_tokens_per_node * self.hidden_size, self.hidden_size, 1),
                    byte_alignment=128,
                )
            workspace.register(
                self.combine_reduce_done_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )
            workspace.register(
                self.layout_published_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset"
            )
            workspace.register(
                self.combine_landing_rows_region,
                cutlass.BFloat16,
                (self.remote_node_slots, self.max_relay_tokens_per_node, self.topk, self.hidden_size),
                buffer_space="shared",
                stride=(
                    self.max_relay_tokens_per_node * self.topk * self.hidden_size,
                    self.topk * self.hidden_size,
                    self.hidden_size,
                    1,
                ),
                byte_alignment=128,
            )
            workspace.register(
                self.fc2_done_mirror_region,
                cutlass.Int32,
                (self.node_experts,),
                buffer_space="shared",
                reset="tail_reset",
                byte_alignment=128,
            )
            workspace.register(
                self.rail_barrier_counter_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                byte_alignment=16,
                reset="zero_on_first_allocate",
            )
            if self.apply_topk_at_fc1:
                workspace.register(
                    self.final_scores_region, cutlass.Float32, (self.worst_case_token_count,), buffer_space="shared"
                )
        workspace.register(
            self.send_token_count_region,
            cutlass.Int64,
            (self.remote_node_slots,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )
        workspace.register(
            self.tokens_to_node_region,
            cutlass.Int32,
            (self.remote_node_slots, self.max_tokens_per_rank),
            buffer_space="local",
            stride=(self.max_tokens_per_rank, 1),
            byte_alignment=16,
        )
        # Inverse of tokens_to_node for the combine: the message index of a token per remote slot
        # (valid only for tokens that route to that node, which TopkReduce derives from topk_indices).
        workspace.register(
            self.message_index_region,
            cutlass.Int32,
            (self.remote_node_slots, self.max_tokens_per_rank),
            buffer_space="local",
            stride=(self.max_tokens_per_rank, 1),
            byte_alignment=16,
        )

    @cute.jit
    def send_count_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.send_count_region)

    @cute.jit
    def send_token_count_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.send_token_count_region)

    @cute.jit
    def tokens_to_node_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.tokens_to_node_region)

    @cute.jit
    def _remote_route(self, expert: Int32):
        """Destination node, remote-node slot and node-local expert index of a valid route."""
        destination_rank = expert // Int32(self.experts_per_rank)
        destination_node = destination_rank // Int32(self.ranks_per_node)
        my_node = self._router_local_rank // Int32(self.ranks_per_node)
        remote_slot = destination_node
        if destination_node > my_node:
            remote_slot = destination_node - Int32(1)
        node_expert = (destination_rank - destination_node * Int32(self.ranks_per_node)) * Int32(
            self.experts_per_rank
        ) + expert % Int32(self.experts_per_rank)
        return destination_node, my_node, remote_slot, node_expert

    @cute.jit
    def _is_first_hit_of_node(self, topk_indices: cute.Tensor, token_idx: Int32, topk_slot: Int32, node: Int32):
        """True when no earlier top-k slot of ``token_idx`` targets ``node`` (one relay message per token and node)."""
        seen_earlier = Int32(0)
        for previous_slot in cutlass.range_constexpr(self.topk):
            if Int32(previous_slot) < topk_slot:
                previous_expert = Int32(topk_indices[token_idx, previous_slot])
                if previous_expert >= Int32(0) and previous_expert < Int32(self.expert_count):
                    previous_node = (previous_expert // Int32(self.experts_per_rank)) // Int32(self.ranks_per_node)
                    if previous_node == node:
                        seen_earlier = Int32(1)
        return seen_earlier == Int32(0)

    @cute.jit
    def _internode_histogram(
        self,
        expert_registers: cute.Tensor,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        smem_base: cute.Pointer,
    ):
        if cutlass.const_expr(self.remote_node_count > 0):
            iket.range_push("router.remote_histogram")
            if self._router_linear_cta_idx == Int32(0):
                if self._router_thread_idx == Int32(0):
                    self.stamp(self._device_workspace, 0)
            block_thread_count = self.router_warps_per_cta * 32
            entry_count = self.remote_node_slots * self.send_count_stride
            smem_send_count = self._router_smem_workspace.tensor(self.smem_send_count_region, smem_base)
            smem_token_count = self._router_smem_workspace.tensor(self.smem_token_count_region, smem_base)
            smem_token_base = self._router_smem_workspace.tensor(self.smem_token_base_region, smem_base)
            # Persistent-kernel discipline (I4a): zero the CTA histogram before counting.
            for zero_round in cutlass.range_constexpr(ceil_div(entry_count, block_thread_count)):
                entry = Int32(zero_round * block_thread_count) + self._router_thread_idx
                if entry < Int32(entry_count):
                    smem_send_count[entry] = Int32(0)
            if self._router_thread_idx < Int32(self.remote_node_slots):
                smem_token_count[self._router_thread_idx] = Int32(0)
                smem_token_base[self._router_thread_idx] = Int32(0)
            cute.arch.sync_threads()
            # Phase 1: per-CTA counts over the routes this thread holds.
            register_count = cute.size(expert_registers)
            for register_idx in cutlass.range_constexpr(register_count):
                expert = expert_registers[register_idx]
                if expert < Int32(self.expert_count):
                    destination_node, my_node, remote_slot, node_expert = self._remote_route(expert)
                    if destination_node != my_node:
                        cute.arch.atomic_add(
                            smem_send_count.iterator + remote_slot * Int32(self.send_count_stride) + node_expert,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                        token_idx, topk_slot = self._router_value_coordinate(register_idx, topk_indices.dtype)
                        if self._is_first_hit_of_node(topk_indices, token_idx, topk_slot, destination_node):
                            cute.arch.atomic_add(
                                smem_token_count.iterator + remote_slot, Int32(1), sem="relaxed", scope="cta"
                            )
            cute.arch.sync_threads()
            # Phase 2 (I4b): reserve this CTA's range of every per-node token list with one packed
            # atomic (high word counts the reserving CTAs, low word is the running token total) and
            # fold the CTA histogram into the rank-wide send counts.
            send_token_count = self._device_workspace.tensor(self.send_token_count_region)
            send_count = self._device_workspace.tensor(self.send_count_region)
            if self._router_thread_idx < Int32(self.remote_node_slots):
                cta_token_count = smem_token_count[self._router_thread_idx]
                packed = cute.arch.atomic_add(
                    send_token_count.iterator + self._router_thread_idx,
                    (Int64(1) << Int64(32)) | (Int64(cta_token_count) & Int64(0xFFFFFFFF)),
                    sem="relaxed",
                    scope="gpu",
                )
                range_base = Int32(packed & Int64(0xFFFFFFFF))
                smem_token_base[self._router_thread_idx] = range_base
                smem_token_count[self._router_thread_idx] = range_base
                cute.arch.atomic_add(
                    send_count.iterator + self._router_thread_idx * Int32(self.splits_row_pitch_int32) + Int32(self.node_experts),
                    cta_token_count,
                    sem="relaxed",
                    scope="gpu",
                )
            for entry_round in cutlass.range_constexpr(ceil_div(entry_count, block_thread_count)):
                entry = Int32(entry_round * block_thread_count) + self._router_thread_idx
                if entry < Int32(entry_count):
                    entry_slot = entry // Int32(self.send_count_stride)
                    entry_expert = entry % Int32(self.send_count_stride)
                    if entry_expert < Int32(self.node_experts):
                        cta_count = smem_send_count[entry]
                        if cta_count > Int32(0):
                            cute.arch.atomic_add(
                                send_count.iterator + entry_slot * Int32(self.splits_row_pitch_int32) + entry_expert,
                                cta_count,
                                sem="relaxed",
                                scope="gpu",
                            )
            cute.arch.sync_threads()
            # Every global atomic of this CTA's fold is issued: publish the CTA for Step 3 (I4).
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.histogram_done_region), Int32(1), sem="release", scope="gpu"
                )
            # Phase 3: list every token once per remote node inside this CTA's reserved range, and
            # write the message's header (route words, weights) into the slot's header plane behind
            # the send_count row: the early put ships it with the row (header plane, see above).
            tokens_to_node = self._device_workspace.tensor(self.tokens_to_node_region)
            message_index = self._device_workspace.tensor(self.message_index_region)
            send_count_base = self._device_workspace.ptr(self.send_count_region).toint()
            for register_idx in cutlass.range_constexpr(register_count):
                expert = expert_registers[register_idx]
                if expert < Int32(self.expert_count):
                    destination_node, my_node, remote_slot, node_expert = self._remote_route(expert)
                    if destination_node != my_node:
                        token_idx, topk_slot = self._router_value_coordinate(register_idx, topk_indices.dtype)
                        if self._is_first_hit_of_node(topk_indices, token_idx, topk_slot, destination_node):
                            list_slot = Int32(
                                cute.arch.atomic_add(
                                    smem_token_count.iterator + remote_slot, Int32(1), sem="relaxed", scope="cta"
                                )
                            )
                            tokens_to_node[remote_slot, list_slot] = token_idx
                            message_index[remote_slot, token_idx] = list_slot
                            if cutlass.const_expr(self.exchanges_remote_splits):
                                header_address = (
                                    send_count_base
                                    + Int64(remote_slot) * Int64(self.splits_row_pitch_bytes)
                                    + Int64(self.splits_row_bytes)
                                    + Int64(list_slot) * Int64(self.relay_header_bytes)
                                )
                                for header_slot in cutlass.range_constexpr(self.topk):
                                    route_word, weight = self._route_word_and_weight(
                                        token_idx, Int32(header_slot), destination_node, topk_indices, topk_scores
                                    )
                                    self._store_route_word(header_address + Int64(header_slot * 4), route_word)
                                    stg_f32(
                                        header_address + Int64(self.relay_header_weights_offset + header_slot * 4), weight
                                    )
            if cutlass.const_expr(self.exchanges_remote_splits):
                # bar.warp.sync orders every lane's header stores before lane 0's cumulative system
                # fence (the NIC reads the plane); the CTA barrier plus the release on list_done carry
                # them to the issuing lane and the pack CTAs (I4), exactly like the pack publish.
                cute.arch.sync_warp()
                if self._router_lane_idx == Int32(0):
                    cute.arch.fence_acq_rel_sys()
            cute.arch.sync_threads()
            if cutlass.const_expr(self.exchanges_remote_splits):
                # This CTA's ranges of tokens_to_node and its header plane entries are complete: hand
                # them to the pack CTAs and the early put (I4).
                if self._router_thread_idx == Int32(0):
                    cute.arch.atomic_add(
                        self._device_workspace.ptr(self.list_done_region), Int32(1), sem="release", scope="gpu"
                    )
            iket.range_pop()

    @cute.jit
    def _route_word_and_weight(
        self,
        token_idx: Int32,
        topk_slot: Int32,
        dst_node: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
    ):
        """Route word and FP32 weight of top-k slot ``topk_slot`` of ``token_idx`` for destination node ``dst_node``.

        ``node_expert | topk_slot << 16`` and the top-k score for a route the destination node
        owns, ``0xFFFF`` and 0 otherwise (legacy wire contract). The header plane and the message
        planes are both built here, so they agree by construction.
        """
        expert = Int32(topk_indices[token_idx, topk_slot])
        route_word = Int32(self.unrouted_route_word)
        weight = cutlass.Float32(0.0)
        if expert >= Int32(0):
            if expert < Int32(self.expert_count):
                destination_rank = expert // Int32(self.experts_per_rank)
                if destination_rank // Int32(self.ranks_per_node) == dst_node:
                    rank_in_node = destination_rank - dst_node * Int32(self.ranks_per_node)
                    node_expert = rank_in_node * Int32(self.experts_per_rank) + expert % Int32(self.experts_per_rank)
                    route_word = node_expert | (topk_slot << Int32(16))
                    if cutlass.const_expr(topk_scores is not None):
                        weight = cutlass.Float32(topk_scores[token_idx, topk_slot])
        return route_word, weight

    @cute.jit
    def _store_route_word(self, address: Int64, route_word: Int32) -> None:
        route_pointer = cute.make_ptr(Int32, address, AddressSpace.gmem, assumed_align=4)
        cute.arch.store(route_pointer, route_word, scope="gpu")

    # ------------------------------------------------------------------ final layout accessors (fused kernel)

    @property
    def relay_bucket_count(self) -> int:
        return self.experts_per_rank * self.remote_node_slots * self.ranks_per_node

    @cute.jit
    def final_sizes_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.layout_sizes_region)

    @cute.jit
    def final_pool_base_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.layout_pool_base_region)

    @cute.jit
    def intra_count_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.layout_intra_count_region)

    @cute.jit
    def final_records_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.final_records_region)

    @cute.jit
    def final_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.final_scores_region)

    @cute.jit
    def wait_for_layout_ready(self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000) -> None:
        """Warp-level wait for SM0's Step 5c publication (final sizes, bases, intra records)."""
        thread_idx, _, _ = cute.arch.thread_idx()
        if thread_idx % Int32(32) == Int32(0):
            ready = device_workspace.ptr(self.layout_ready_region)
            while cute.arch.load(ready, Int32, sem="acquire", scope="gpu") != Int32(1):
                nanosleep(sleep_cycles)
        cute.arch.sync_warp()

    @cute.jit
    def wait_for_relay_records(self, device_workspace: DeviceWorkspace, local_expert: Int32, sleep_cycles: int = 500) -> None:
        """Warp-level wait until every relay record of ``local_expert`` is written.

        The expected count is the expert's final size minus its intra count (both from the final
        layout); the node peer that received a message pushes the record into this rank's final
        layout and releases one increment per record after its system fence, so an expert's relay
        tokens start as soon as its own records are complete instead of after the slowest relay
        warp of the grid (2026-09-28 IKET: the grid-wide count arrived 65 us after layout_ready at
        N2R4 T=256 while the mean scan warp took 8 us; at N4R8 the pull-style scan of every node
        message by every rank took up to 340 us).
        """
        thread_idx, _, _ = cute.arch.thread_idx()
        if thread_idx % Int32(32) == Int32(0):
            expected = (
                self.final_sizes_tensor(device_workspace)[local_expert]
                - self.intra_count_tensor(device_workspace)[local_expert]
            )
            # The increments arrive from the node peers that pushed the records (system-scope release).
            written = device_workspace.ptr(self.relay_written_region) + local_expert * Int32(self.relay_written_stride)
            while cute.arch.load(written, Int32, sem="acquire", scope="sys") < expected:
                nanosleep(sleep_cycles)
        cute.arch.sync_warp()

    # ------------------------------------------------------------------ GIN handles and Step 3

    def _prepare_router_gin_resources(self, gin_dev_comm, gin_window, shared_window_offset) -> None:
        if not self.enable_nccl_gin:
            if gin_dev_comm is not None or gin_window is not None:
                raise ValueError("gin_dev_comm / gin_window given but the router's enable_nccl_gin is False.")
            return
        if gin_dev_comm is None or gin_window is None:
            raise ValueError("enable_nccl_gin requires gin_dev_comm and gin_window router arguments.")
        from nccl.core.device import cute as nccl_cute

        gin_gpu = gin_dev_comm.gin(
            nccl_cute.GinBackendMask.GDAKI, context_id=0, resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU
        )
        self._gin_gpu_ptr = gin_gpu.ptr
        self._team_rail_value = gin_dev_comm.team_rail.__extract_mlir_values__()[0]
        self._gin_window = gin_window
        self._gin_shared_window_offset = Int64(0) if shared_window_offset is None else shared_window_offset

    def _release_router_gin_resources(self) -> None:
        self._gin_gpu_ptr = None
        self._team_rail_value = None
        self._gin_window = None
        self._gin_shared_window_offset = None
        self._activation = None
        self._activation_sf = None

    def _bind_router_payload_inputs(self, activation, activation_sf) -> None:
        if not self.exchanges_remote_splits:
            # Degenerate single-node GIN configuration: the fused kernel still passes the payload
            # inputs (it keys them on the component's GIN requirement); nothing is packed.
            return
        if activation is None or activation_sf is None:
            raise ValueError("The Stage 1 pack requires the activation and activation_sf router arguments.")
        self._activation = activation
        self._activation_sf = activation_sf

    # ------------------------------------------------------------------ Stage 1 pack (DATA CTAs)

    @cute.jit
    def stamp(self, device_workspace: DeviceWorkspace, index: int) -> None:
        """Write %globaltimer into phase stamp ``index`` (no-op unless impl_desc phase_stamps); caller picks the lane."""
        if cutlass.const_expr(self.phase_stamps):
            device_workspace.tensor(self.phase_stamps_region)[Int32(index)] = read_globaltimer_ns()

    @cute.jit
    def _router_extra_cta_body(self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor]) -> None:
        """CTAs appended after the base grid: the Stage 1 pack CTAs."""
        if cutlass.const_expr(self.exchanges_remote_splits):
            self._pack_relay_messages(topk_indices, topk_scores)

    @cute.jit
    def _pack_relay_messages(self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor]) -> None:
        """Pack CTA: the relay messages of every remote node, one warp per message.

        Waits until every DATA CTA published ``list_done`` (its ``tokens_to_node`` ranges and the
        folded ``send_count`` row are complete), then the pack warps of all pack CTAs stride over
        ``[0, message_count)`` of every slot: ``tokens_to_node[slot][m]`` names the token of message
        ``m``. Every lane moves 16-byte vectors of the token row and the scale factors, lanes below
        the top-k count write their route word and weight, lane 0 writes the metadata slot.
        """
        list_done = self._device_workspace.ptr(self.list_done_region)
        iket.range_push("router.pack_wait_lists")
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(list_done, Int32, sem="acquire", scope="gpu") != Int32(self.router_data_cta_count):
                nanosleep(100)
        cute.arch.sync_threads()
        iket.range_pop()
        iket.range_push("router.pack_messages")
        send_count = self._device_workspace.tensor(self.send_count_region)
        tokens_to_node = self._device_workspace.tensor(self.tokens_to_node_region)
        send_base = self._device_workspace.ptr(self.relay_send_buffer_region).toint()
        message_bytes = self.relay_message.message_bytes
        pack_warp_total = self.router_pack_cta_count * self.router_warps_per_cta
        pack_warp = (self._router_linear_cta_idx - Int32(self.router_base_grid_cta_count)) * Int32(
            self.router_warps_per_cta
        ) + self._router_warp_idx
        for slot in cutlass.range_constexpr(self.remote_node_slots):
            dst_node, _ = self._remote_slot_geometry(Int32(slot))
            message_count = send_count[Int32(slot), Int32(self.node_experts)]
            message_index = pack_warp
            while message_index < message_count:
                token_idx = tokens_to_node[Int32(slot), message_index]
                message_address = send_base + (
                    Int64(slot * self.max_relay_tokens_per_node) + Int64(message_index)
                ) * Int64(message_bytes)
                self._pack_one_message(message_address, token_idx, dst_node, topk_indices, topk_scores)
                message_index = message_index + Int32(pack_warp_total)
        iket.range_pop()
        # bar.warp.sync orders every lane's stores before lane 0's cumulative system fence; the
        # CTA barrier plus the release on pack_done carry them to the issuing HELPER lane (I4, I5).
        iket.range_push("router.pack_publish")
        cute.arch.sync_warp()
        if self._router_lane_idx == Int32(0):
            cute.arch.fence_acq_rel_sys()
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.pack_done_region), Int32(1), sem="release", scope="gpu"
            )
        iket.range_pop()

    @cute.jit
    def _pack_one_message(
        self,
        message_address: Int64,
        token_idx: Int32,
        dst_node: Int32,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
    ) -> None:
        lane = self._router_lane_idx
        layout = self.relay_message
        if lane == Int32(0):
            TokenSrcMetadata(
                src_rank=Int32(self._router_local_rank), src_token=token_idx, src_topk=Int32(0xFFFF)
            ).store(message_address + Int64(layout.src_metadata_offset), wide_topk=True)
            stg_b64(message_address + Int64(layout.src_metadata_offset + 8), Int64(0))
        if lane < Int32(self.topk):
            # Same construction as the header plane (_route_word_and_weight): the message keeps its
            # route and weight planes so the wire format and the legacy readers stay valid.
            route_word, weight = self._route_word_and_weight(token_idx, lane, dst_node, topk_indices, topk_scores)
            self._store_route_word(message_address + Int64(layout.route_words_offset) + Int64(lane) * Int64(4), route_word)
            stg_f32(message_address + Int64(layout.topk_weights_offset) + Int64(lane) * Int64(4), weight)
        lane_bytes = Int64(lane) * Int64(16)
        source_row = self._activation.iterator.toint() + Int64(token_idx) * Int64(self.activation_row_bytes)
        data_destination = message_address + Int64(layout.token_data_offset)
        # Latency, not bandwidth, bounds the pack: every lane issues all of its full-warp vector
        # loads (512 bytes per warp per round) before the first store, so one message costs one
        # DRAM round trip instead of one per vector.
        full_round_count = self.activation_row_bytes // 512
        loaded_words = []
        for row_k in cutlass.range_constexpr(full_round_count):
            loaded_words.append(ldg_v4_b32(source_row + Int64(row_k * 512) + lane_bytes))
        # Partial last round (rows that are not a multiple of 512 bytes) and the scale-factor row
        # are lane-predicated; their loads overlap the outstanding full-round loads.
        tail_offset = Int64(full_round_count * 512) + lane_bytes
        if tail_offset < Int64(self.activation_row_bytes):
            self._copy_vector16(source_row + tail_offset, data_destination + tail_offset)
        if lane_bytes < Int64(self.activation_sf_row_bytes):
            source_sf = self._activation_sf.iterator.toint() + Int64(token_idx) * Int64(self.activation_sf_row_bytes)
            self._copy_vector16(source_sf + lane_bytes, message_address + Int64(layout.sf_data_offset) + lane_bytes)
        for row_k in cutlass.range_constexpr(full_round_count):
            words = loaded_words[row_k]
            stg_v4_b32(data_destination + Int64(row_k * 512) + lane_bytes, words[0], words[1], words[2], words[3])

    @cute.jit
    def _copy_vector16(self, source_address: Int64, destination_address: Int64) -> None:
        """Copy 16 bytes (the planes are 16-byte aligned by layout)."""
        words = ldg_v4_b32(source_address)
        stg_v4_b32(destination_address, words[0], words[1], words[2], words[3])

    @cute.jit
    def _window_tensor(self, region_name: str, region_byte_offset: Int64, dtype, layout, alignment: int):
        """Window-relative view at ``region_byte_offset`` inside a shared region.

        The window's local mapping is an alias of the arena at another virtual address, so put
        tensors must derive from ``Window.tensor`` (the window's local base), not from the kernel's
        ``shared_workspace`` pointer: ``Gin.put`` recovers the transfer offset as
        ``tensor address - window local pointer`` (legacy pattern: base from ``Window.tensor`` at
        offset 0, then plain address arithmetic).
        """
        byte_offset = (
            self._gin_shared_window_offset + Int64(self._device_workspace.offset(region_name)) + region_byte_offset
        )
        window_base = self._gin_window.tensor(Int32, cute.make_layout((1,), stride=(1,))).iterator.toint()
        return cute.make_tensor(
            cute.make_ptr(dtype, window_base + byte_offset, AddressSpace.gmem, assumed_align=alignment), layout
        )

    def _device_gin(self):
        """Rewrap the entry-materialized GPU-shared GIN pointer inside a nested JIT region."""
        from nccl.core.device.cute.gin import Gin

        return Gin(ptr=self._gin_gpu_ptr)

    @cute.jit
    def _remote_slot_geometry_for(self, dst_slot: Int32, local_rank: Int32):
        """(destination node, this node's slot at the destination) of remote slot ``dst_slot`` for ``local_rank``."""
        my_node = local_rank // Int32(self.ranks_per_node)
        dst_node = dst_slot
        if dst_slot >= my_node:
            dst_node = dst_slot + Int32(1)
        src_slot_at_dst = my_node
        if my_node > dst_node:
            src_slot_at_dst = my_node - Int32(1)
        return dst_node, src_slot_at_dst

    @cute.jit
    def _remote_slot_geometry(self, dst_slot: Int32):
        """(destination node, this node's slot at the destination) of remote slot ``dst_slot``."""
        my_node = self._router_local_rank // Int32(self.ranks_per_node)
        dst_node = dst_slot
        if dst_slot >= my_node:
            dst_node = dst_slot + Int32(1)
        src_slot_at_dst = my_node
        if my_node > dst_node:
            src_slot_at_dst = my_node - Int32(1)
        return dst_node, src_slot_at_dst

    @cute.jit
    def _exchange_remote_splits(self) -> None:
        if cutlass.const_expr(self.exchanges_remote_splits):
            if self._router_warp_idx == Int32(0):
                from nccl.core.device import cute as nccl_cute

                gin = self._device_gin()
                coop = nccl_cute.thread()
                # Step 3 row plus header plane: both are final once every DATA CTA published
                # list_done (its histogram fold precedes its list and header stores in program order,
                # so list_done covers histogram_done). Same lane and queue pair as the payload put,
                # so the block is transmitted first; a put issue costs about 10 us (doorbell path).
                iket.range_push("router.wait_lists_done")
                if self._router_lane_idx == Int32(0):
                    list_done = self._device_workspace.ptr(self.list_done_region)
                    while cute.arch.load(list_done, Int32, sem="acquire", scope="gpu") != Int32(
                        self.router_data_cta_count
                    ):
                        nanosleep(100)
                cute.arch.sync_warp()
                iket.range_pop()
                iket.range_push("router.put_row")
                if self._router_lane_idx < Int32(self.remote_node_count):
                    cute.arch.fence_acq_rel_sys()
                    dst_slot = self._router_lane_idx
                    dst_node, src_slot_at_dst = self._remote_slot_geometry(dst_slot)
                    send_count = self._device_workspace.tensor(self.send_count_region)
                    message_count = send_count[dst_slot, Int32(self.node_experts)]
                    if message_count > Int32(self.max_relay_tokens_per_node):
                        _device_trap()
                    # One contiguous put of the slot block: the row, then the headers of the
                    # message_count messages; the bundled INC signal certifies the whole put, so the
                    # receiver's scan may read the headers as soon as the row signal is acquired.
                    block_layout = cute.make_layout(
                        (Int32(self.splits_row_bytes) + message_count * Int32(self.relay_header_bytes),), stride=(1,)
                    )
                    src_block = self._window_tensor(
                        self.send_count_region, Int64(dst_slot) * Int64(self.splits_row_pitch_bytes), cutlass.Uint8, block_layout, 16
                    )
                    # Symmetric window: the destination offset is this rank's recv_splits block, which is
                    # the same window offset on the rail peer (LsaPool carves every rank identically).
                    dst_block = self._window_tensor(
                        self.recv_splits_region,
                        Int64(src_slot_at_dst) * Int64(self.splits_row_pitch_bytes),
                        cutlass.Uint8,
                        block_layout,
                        16,
                    )
                    gin.put(
                        self._team_rail_value,
                        dst_node,
                        self._gin_window,
                        dst_block,
                        self._gin_window,
                        src_block,
                        coop,
                        is_signal=True,
                        signal_id=Int32(self.splits_signal_base) + src_slot_at_dst,
                        signal_op=GIN_SIGNAL_OP_INC,
                    )
                cute.arch.sync_warp()
                iket.range_pop()
                # Every pack CTA packed its share of the messages after every DATA CTA's list
                # (list_done), then published pack_done (release after the pack); lane 0's acquire
                # and the warp barrier order the issuing lanes' payload puts after those stores (I4,
                # I5; bar.warp.sync orders memory among the warp's threads).
                iket.range_push("router.wait_pack_done")
                if self._router_lane_idx == Int32(0):
                    pack_done = self._device_workspace.ptr(self.pack_done_region)
                    while cute.arch.load(pack_done, Int32, sem="acquire", scope="gpu") != Int32(
                        self.router_pack_cta_count
                    ):
                        nanosleep(100)
                    self.stamp(self._device_workspace, 1)
                cute.arch.sync_warp()
                iket.range_pop()
                iket.range_push("router.put_range")
                if self._router_lane_idx < Int32(self.remote_node_count):
                    cute.arch.fence_acq_rel_sys()
                    dst_slot = self._router_lane_idx
                    dst_node, src_slot_at_dst = self._remote_slot_geometry(dst_slot)
                    send_count = self._device_workspace.tensor(self.send_count_region)
                    message_count = send_count[dst_slot, Int32(self.node_experts)]
                    if message_count > Int32(0):
                        # Stage 1: the dense [0, message_count) range of this destination as one put; the
                        # bundled ADD carries the expert-hit credit the receiver waits for (I5, whole range).
                        credit = Int32(0)
                        for node_expert in cutlass.range_constexpr(self.node_experts):
                            credit = credit + send_count[dst_slot, Int32(node_expert)]
                        range_layout = cute.make_layout(
                            (message_count * Int32(self.relay_message.message_bytes),), stride=(1,)
                        )
                        src_range = self._window_tensor(
                            self.relay_send_buffer_region,
                            Int64(dst_slot) * Int64(self.relay_range_bytes),
                            cutlass.Uint8,
                            range_layout,
                            16,
                        )
                        dst_range = self._window_tensor(
                            self.relay_recv_buffer_region,
                            Int64(src_slot_at_dst) * Int64(self.relay_range_bytes),
                            cutlass.Uint8,
                            range_layout,
                            16,
                        )
                        gin.put(
                            self._team_rail_value,
                            dst_node,
                            self._gin_window,
                            dst_range,
                            self._gin_window,
                            src_range,
                            coop,
                            is_signal=True,
                            signal_id=Int32(self.relay_node_signal_base) + src_slot_at_dst,
                            signal_op=GIN_SIGNAL_OP_ADD,
                            signal_op_arg=Int64(credit),
                            given_release=nccl_cute.ThreadScope.THREAD,
                        )
                cute.arch.sync_warp()
                if self._router_lane_idx == Int32(0):
                    self.stamp(self._device_workspace, 2)
                iket.range_pop()

    # ------------------------------------------------------------------ Step 5a (receiver wait)

    @cute.jit
    def launch_wait_remote_splits(
        self,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
        gin_dev_comm,
        gin_window,
        stream: cuda.CUstream,
    ) -> None:
        """One-CTA kernel: wait until every source node's Step 3 row AND Stage 1 range have landed.

        Per source slot: the Step 3 INC signal reaches the advanced shadow target, then the row's
        expert-hit total is the credit the Stage 1 ADD signal must reach (I5, I8). The fused kernel's
        SM0 / relay warps perform the same waits in-kernel; the router-only test uses this launch to
        read ``recv_splits`` and the landed messages from the host afterwards.
        """
        self._wait_remote_splits_kernel(local_workspace, shared_workspace, device_workspace, gin_dev_comm, gin_window).launch(
            grid=[1, 1, 1], block=[32, 1, 1], min_blocks_per_mp=1, stream=stream
        )

    @cute.kernel
    def _wait_remote_splits_kernel(
        self,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
        gin_dev_comm,
        gin_window,
    ) -> None:
        self._prepare_router_gin_resources(gin_dev_comm, gin_window, Int64(0))
        thread_idx, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.exchanges_remote_splits):
            device_workspace.assign_device_members(local_workspace, shared_workspace)
            if thread_idx == Int32(0):
                from nccl.core.device import cute as nccl_cute

                gin = self._device_gin()
                coop = nccl_cute.thread()
                recv_splits = device_workspace.tensor(self.recv_splits_region)
                for src_slot in cutlass.range_constexpr(self.remote_node_count):
                    signal_id = Int32(self.splits_signal_base + src_slot)
                    # One producer per source slot and launch: advance the NCCL-managed shadow
                    # target by one and wait for the signal to reach it (monotonic across launches, I8).
                    shadow = gin.signal_shadow_pointer(signal=signal_id)
                    old_target = cute.arch.atomic_add(shadow, Uint64(1), sem="relaxed", scope="gpu")
                    gin.wait_signal(
                        coop,
                        signal=signal_id,
                        least=Uint64(old_target) + Uint64(1),
                        bits=64,
                        ord=nccl_cute.MemoryOrder.ACQUIRE,
                    )
                    # The row is visible after the acquire: its expert-hit total is the payload credit.
                    credit = Int32(0)
                    for node_expert in cutlass.range_constexpr(self.node_experts):
                        credit = credit + recv_splits[Int32(src_slot), Int32(node_expert)]
                    credit_signal = Int32(self.relay_node_signal_base + src_slot)
                    credit_shadow = gin.signal_shadow_pointer(signal=credit_signal)
                    old_credit = cute.arch.atomic_add(credit_shadow, Uint64(credit), sem="relaxed", scope="gpu")
                    gin.wait_signal(
                        coop,
                        signal=credit_signal,
                        least=Uint64(old_credit) + Uint64(credit),
                        bits=64,
                        ord=nccl_cute.MemoryOrder.ACQUIRE,
                    )
            device_workspace.remove_device_members()
        self._release_router_gin_resources()


__all__ = ["GIN_SIGNAL_OP_ADD", "GIN_SIGNAL_OP_INC", "InternodeMetadataRouter"]
