"""Internode token communication component: NVLink domain plus NCCL GIN rail relay.

Selected through the master switch ``ImplDesc["internode"] = True`` (``token_comm_kind == "internode_gin"``);
with the switch off (the default) the kernel never imports this module. Carries the internode
configuration (``ranks_per_node``, ``enable_nccl_gin``, ``relay_warp_count``). In the
single-node degenerate configuration (one NVLink domain, ``enable_nccl_gin`` false) every hook
delegates to :class:`TokenCommNonDeterministic`, so the fused kernel's output is bitwise
identical to the NVLink-domain component; the relay warp group (warps 12-15) only reserves its
place in the CTA layout. With several NVLink domains the component runs the internode dispatch
(Step 3 splits and the coalesced Stage 1 payload puts from the router, the relay records and the
relay pull in the fused kernel) and the internode combine (landing-region pushes, the rail
peer's gather-reduce, one coalesced return put per source node, the ordered TopkReduce at the
origin). "legacy" throughout this package means the standalone MegaMoE implementation outside
``next/`` (the non-next/ code path) that this component was ported from; its protocol
invariants I1 to I12 are documented in ``docs/internode_megamoe_design.md``.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Boolean, Int32, Int64, Uint64

import dataclasses
import functools

from ...helpers.flag_batch import SysReleaseFlagBatchTracker
from ...helpers.iket_compat import iket
from ...helpers.ptx_helpers import nanosleep, read_clock64, red_add_relaxed_sys_s32, stg_b64, stg_f32
from .lsa_peer_mapper import _device_trap
from .router_internode import GIN_SIGNAL_OP_ADD
from ..token_protocol import TokenSrcMetadata

from ...api import ImplDesc, OptionalRequirement, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.utils import ceil_div
from ..nvlink_domain.token_comm import TokenCommArgs, TokenCommNonDeterministic
from .node_local_barrier import NodeLocalNvlinkBarrier
from .router_internode import InternodeMetadataRouter


@dataclasses.dataclass(frozen=True)
class _FanoutSysReleaseFlagBatchTracker(SysReleaseFlagBatchTracker):
    """fc2_done flag tracker of the internode component: the owner's counter plus every node peer's mirror.

    ``fire`` publishes with the fence-based release pattern (one system fence, then relaxed
    system-scope adds): the flag counter itself as before, and, for flags inside the fc2_done region
    (``fc2_lo <= address < fc2_hi``; the epilogue batches the FC1 done flags through the same
    tracker), ``fc2_done_mirror[owner in node, e]`` of all ``fanout_ranks`` node peers (offsets
    ``off0..off7`` from the LSA mapper, the unused ones zero). The rail peers' gather then acquires
    its local mirror instead of polling the owner over NVLink (I9 chain unchanged: row stores ->
    fence -> increment -> acquire -> row loads). Per-count subclasses (``_fanout_tracker_class``)
    keep every dataclass field a DSL value.
    """

    mirror_delta: Int64
    fc2_lo: Int64
    fc2_hi: Int64
    off0: Int64
    off1: Int64
    off2: Int64
    off3: Int64
    off4: Int64
    off5: Int64
    off6: Int64
    off7: Int64

    fanout_ranks: ClassVar[int] = 8

    @cute.jit
    def _make(self, flag_address: Int64, accumulated_flags: Int32, phase: Int32) -> "_FanoutSysReleaseFlagBatchTracker":
        return type(self)(
            flag_address=flag_address,
            accumulated_flags=accumulated_flags,
            phase=phase,
            thread_idx=self.thread_idx,
            mirror_delta=self.mirror_delta,
            fc2_lo=self.fc2_lo,
            fc2_hi=self.fc2_hi,
            off0=self.off0,
            off1=self.off1,
            off2=self.off2,
            off3=self.off3,
            off4=self.off4,
            off5=self.off5,
            off6=self.off6,
            off7=self.off7,
        )

    @cute.jit
    def fire(self) -> None:
        if self.flag_address != Int64(0):
            cute.arch.fence_acq_rel_sys()
            red_add_relaxed_sys_s32(self.flag_address, Int32(1))
            # Only FC2 completion flags have mirrors; the FC1 done flags share this tracker.
            if self.flag_address >= self.fc2_lo:
                if self.flag_address < self.fc2_hi:
                    mirror_address = self.flag_address + self.mirror_delta
                    offsets = (self.off0, self.off1, self.off2, self.off3, self.off4, self.off5, self.off6, self.off7)
                    for peer in cutlass.range_constexpr(self.fanout_ranks):
                        red_add_relaxed_sys_s32(mirror_address + offsets[peer], Int32(1))


@functools.lru_cache(maxsize=None)
def _fanout_tracker_class(fanout_ranks: int):
    if not 1 <= fanout_ranks <= 8:
        raise ValueError(f"fc2_done fan-out supports 1 to 8 node peers, got {fanout_ranks}.")
    return type(f"FanoutSysReleaseFlagBatchTracker{fanout_ranks}", (_FanoutSysReleaseFlagBatchTracker,), {"fanout_ranks": fanout_ranks})


@dataclasses.dataclass(frozen=True)
class _Fc2DoneFanout:
    """Handed to the FC2 epilogue (``fc2_done_fanout``): builds the fan-out tracker with this rank's mirror delta."""

    mirror_delta: Int64  # peer mirror cell of expert e = fc2_done address of e + mirror_delta + peer offset
    fc2_lo: Int64  # [fc2_lo, fc2_hi): this rank's fc2_done counters (other flag addresses are FC1 done flags)
    fc2_hi: Int64
    node_base: Int32  # first rank of this node (peer j = node_base + j)

    fanout_ranks: ClassVar[int] = 8

    def make_tracker(self, peer_rank_ptr_mapper, *, phase: Int32, thread_idx: Int32):
        offsets = [
            peer_rank_ptr_mapper.map(Int64(0), self.node_base + Int32(peer), Int64(0)) for peer in range(self.fanout_ranks)
        ]
        offsets += [Int64(0)] * (8 - len(offsets))
        return _fanout_tracker_class(self.fanout_ranks)(
            flag_address=Int64(0),
            accumulated_flags=Int32(0),
            phase=phase,
            thread_idx=thread_idx,
            mirror_delta=self.mirror_delta,
            fc2_lo=self.fc2_lo,
            fc2_hi=self.fc2_hi,
            off0=offsets[0],
            off1=offsets[1],
            off2=offsets[2],
            off3=offsets[3],
            off4=offsets[4],
            off5=offsets[5],
            off6=offsets[6],
            off7=offsets[7],
        )


@functools.lru_cache(maxsize=None)
def _fanout_desc_class(fanout_ranks: int):
    return type(f"Fc2DoneFanout{fanout_ranks}", (_Fc2DoneFanout,), {"fanout_ranks": fanout_ranks})


class TokenCommInternodeGin(TokenCommNonDeterministic):
    """NVLink-domain token communication extended with an internode relay warp group."""

    # Warps 12-15 of the 16-warp CTA; mutually exclusive with ``token_back_mode == "standalone_warps"``.
    relay_warp_idx_start: ClassVar[int] = 12
    # Relay records carry marker | slot | top-k in the 16-bit top-k field.
    wide_topk_records: ClassVar[bool] = True
    supported_relay_warp_counts: ClassVar[Tuple[int, ...]] = (0, 4)
    # Named barrier of the relay warp group (ids 9-13 belong to the NVLink component and the finalize barrier).
    relay_group_barrier_id: ClassVar[int] = 14
    # Relay warps -> transfer warps handoff before the tail: the tail's counter reset must not race the
    # relay group's combine (its completion counters and this rank's send counts are still in use).
    relay_done_barrier_id: ClassVar[int] = 15
    # Rail barrier spin bound before trapping (about 30 s at 2 GHz): a hang here means a remote node
    # never reached its tail, which is a protocol error, not a slow peer.
    rail_barrier_timeout_cycles: ClassVar[int] = 60_000_000_000
    # Combine gather-reduce chunk: hidden elements per pass of one warp (32 per lane, 64 B BF16 per
    # top-k slot). 512 and 256 were measured neutral to worse at N2R4 and N4R8.
    combine_chunk_elements: ClassVar[int] = 1024
    # Per-CTA SMEM cache of the owners' fc2_done targets and "seen done" flags, indexed by node
    # expert (owner in node x experts + local expert): the gather decodes a route with SMEM reads
    # instead of two NVLink round trips per top-k slot (phase stamps: 69 us of gather after the last
    # FC2 tile at N4R8 against legacy's 8 us before the cache).
    relay_fc2_target_smem_region: ClassVar[str] = "internode.relay_smem.fc2_targets"
    relay_fc2_done_smem_region: ClassVar[str] = "internode.relay_smem.fc2_done"
    # Chunk groups per message in the gather: work item = (message, group); more groups = more NVLink
    # round trips in flight per message across warps (IKET: about 3 us per 1024-element chunk).
    combine_gather_groups: ClassVar[int] = 4
    # Named barrier on which the relay warps arrive after the SMEM cache build and the epilogue warps
    # wait before joining the gather. Id 11 is the NVLink component's standalone_size_barrier_id,
    # which only the standalone token-back mode uses (mutually exclusive with relay warps).
    combine_join_barrier_id: ClassVar[int] = 11
    # Epilogue warps of the mega kernel (its epilogue_warp_ids) that join the gather after their last tile.
    combine_epilogue_worker_warps: ClassVar[int] = 4

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            **super().impl_desc_require(),
            "ranks_per_node": int,
            "enable_nccl_gin": bool,
            "relay_warp_count": OptionalRequirement(int),
            "fc2_done_release_scope": OptionalRequirement(str),
            "router_pack_cta_count": OptionalRequirement(int),
            "phase_stamps": OptionalRequirement(bool),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        self.ranks_per_node = impl_desc["ranks_per_node"]
        self.enable_nccl_gin = impl_desc["enable_nccl_gin"]
        self.relay_warp_count = impl_desc.get("relay_warp_count", 4 if self.enable_nccl_gin else 0)
        # GIN mode publishes fc2_done for other ranks' system-scope acquires (I9); the degenerate
        # mode keeps the GPU scope unless a test forces "sys".
        self._fc2_done_release_scope = impl_desc.get("fc2_done_release_scope", "sys" if self.enable_nccl_gin else "gpu")
        self._validate_internode_configuration()
        # The node-scoped router that also builds the remote-node dispatch metadata (Steps 1/2).
        self._router = InternodeMetadataRouter(problem_desc, impl_desc)
        # The all-rank barrier of the NVLink-domain protocol becomes a node-local barrier:
        # same sense-reversing scheme, peers = the ranks_per_node ranks of this rank's node.
        self._nvlink_barrier = NodeLocalNvlinkBarrier(
            world_size=self.world_size, ranks_per_node=self.ranks_per_node, barrier_id=self.grid_sync_barrier_id
        )
        self._gin_gpu_ptr = None
        self._gin_thread_ptr = None
        self._team_rail_value = None
        self._gin_window = None
        self._gin_shared_window_offset = None

    def _validate_internode_configuration(self) -> None:
        if self.ranks_per_node <= 0 or self.world_size % self.ranks_per_node != 0:
            raise ValueError(
                f"ranks_per_node must divide world_size, got {self.ranks_per_node} and {self.world_size}."
            )
        self.num_nodes = self.world_size // self.ranks_per_node
        if self.relay_warp_count not in self.supported_relay_warp_counts:
            raise ValueError(
                f"relay_warp_count must be one of {self.supported_relay_warp_counts}, got {self.relay_warp_count}."
            )
        if self.num_nodes > 1 and not self.enable_nccl_gin:
            raise ValueError("More than one NVLink domain requires enable_nccl_gin.")
        if self.relay_warp_count > 0 and self.token_back_mode == "standalone_warps":
            raise ValueError("Relay warps 12-15 and token_back_mode='standalone_warps' both need warps 12-15.")
        if self._fc2_done_release_scope not in ("gpu", "sys"):
            raise ValueError(f"fc2_done_release_scope must be 'gpu' or 'sys', got {self._fc2_done_release_scope!r}.")
        if self.enable_nccl_gin and self._fc2_done_release_scope != "sys":
            raise ValueError("enable_nccl_gin requires fc2_done_release_scope='sys' (design invariant I9).")
        if self.num_nodes > 1 and self.relay_warp_count == 0:
            raise ValueError("More than one NVLink domain needs the relay warp group (relay_warp_count=4).")
        if self.num_nodes > 1 and self.hidden_size % self.combine_chunk_elements != 0:
            raise ValueError(
                f"The relay combine reduces {self.combine_chunk_elements}-element chunks; hidden_size={self.hidden_size}."
            )
        # The CTA-specialized and multi-stage token transfers of the NVLink-domain component read the
        # pool through the router directly; the relay hooks (final layout, relay records, landing rows)
        # are only wired into the single-stage token_in / token_back paths.
        if self.num_nodes > 1 and (
            getattr(self, "cta_specialized", False) or self.dispatch_stage_depth > 1 or self.combine_stage_depth > 1
        ):
            raise NotImplementedError(
                "The internode component supports neither comm_cta_count > 0 nor comm_stage_depths above (1, 1)."
            )
        # The relay message carries the FP4 row, its scale factors, the route words and the top-k
        # weights; the per-token FC1 activation scale is not part of the wire format yet.
        if self.num_nodes > 1 and self._router.fc1_activation_per_token_scale:
            raise NotImplementedError(
                "fc1_activation_per_token_scale is not relayed across NVLink domains by the internode component."
            )

    @property
    def name_token(self) -> str:
        """Compile-cache token: every constexpr of this component."""
        return (
            f"comminode_rpn{self.ranks_per_node}_gin{int(self.enable_nccl_gin)}_relay{self.relay_warp_count}"
            f"_fc2{self._fc2_done_release_scope}_pack{self._router.router_pack_cta_count}"
            f"_stamps{int(self._router.phase_stamps)}"
        )

    @property
    def fc2_done_release_scope(self) -> str:
        return self._fc2_done_release_scope

    @property
    def requires_gin_resources(self) -> bool:
        return self.enable_nccl_gin

    @property
    def relays(self) -> bool:
        """Relay tokens exist: GIN on and more than one logical node."""
        return self._router.exchanges_remote_splits

    @property
    def relay_record_topk_marker(self) -> Optional[int]:
        return self._router.relay_record_topk_marker if self.relays else None

    @property
    def relay_warp_total(self) -> int:
        return self.promised_launchable_sm_count * self.relay_warp_count

    @property
    def combine_worker_warps_per_cta(self) -> int:
        """Warps per CTA that take combine gather items: the relay warps plus the epilogue warps."""
        return self.relay_warp_count + self.combine_epilogue_worker_warps

    @property
    def combine_worker_total(self) -> int:
        """Arrivals SM0 waits for on ``combine_reduce_done`` before the return puts."""
        return self.promised_launchable_sm_count * self.combine_worker_warps_per_cta

    @cute.jit
    def relay_landing_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.relays):
            return None
        return device_workspace.tensor(self._router.combine_landing_rows_region)

    @cute.jit
    def fc2_done_fanout(self, device_workspace: DeviceWorkspace, peer_rank_ptr_mapper):
        """Fan-out descriptor for the FC2 epilogue's fc2_done tracker (None without relay tokens).

        The owner's increment of ``fc2_done[e]`` is mirrored into ``fc2_done_mirror[owner in node, e]``
        of every node peer; ``mirror_delta`` maps this rank's flag address of expert ``e`` to its own
        mirror cell, and the LSA peer offset shifts that address into the peer's arena (both regions
        live in the shared arena).
        """
        if cutlass.const_expr(not self.relays):
            return None
        ranks_per_node = Int32(self.ranks_per_node)
        owner_in_node = self._local_rank % ranks_per_node
        node_base = (self._local_rank // ranks_per_node) * ranks_per_node
        mirror_base = device_workspace.ptr(self._router.fc2_done_mirror_region).toint()
        fc2_done_base = device_workspace.ptr(self.fc2_done_region).toint()
        mirror_delta = mirror_base + Int64(owner_in_node) * Int64(self.experts_per_rank * 4) - fc2_done_base
        return _fanout_desc_class(self.ranks_per_node)(
            mirror_delta=mirror_delta,
            fc2_lo=fc2_done_base,
            fc2_hi=fc2_done_base + Int64(self.experts_per_rank * 4),
            node_base=node_base,
        )

    @property
    def relay_slot_count(self) -> int:
        return self._router.remote_node_count if self.relays else 0

    @cute.jit
    def combine_landing_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.relays):
            return None
        return device_workspace.tensor(self._router.combine_landing_region)

    @cute.jit
    def message_index_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.relays):
            return None
        return device_workspace.tensor(self._router.message_index_region)

    @property
    def publishes_fc2_done(self) -> bool:
        # The rail peers gather the owners' relay FC2 rows after the owners' completion counters (I9).
        return self.token_back_enabled or self.relays

    @property
    def fc2_done_buffer_space(self) -> str:
        return "shared" if self.relays else "local"

    # ---------------------------------------------------------------- pool-layout hooks (final layout)

    @cute.jit
    def local_expert_sizes(self, device_workspace: DeviceWorkspace, local_rank: Int32) -> cute.Tensor:
        if cutlass.const_expr(self.relays):
            return self._router.final_sizes_tensor(device_workspace)
        return self._router.local_expert_sizes(device_workspace, local_rank)

    @cute.jit
    def wait_for_sizes_ready(self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000) -> None:
        if cutlass.const_expr(self.relays):
            self._router.wait_for_layout_ready(device_workspace, sleep_cycles)
        else:
            self._router.wait_for_sizes_ready(device_workspace, sleep_cycles)

    @cute.jit
    def pool_expert_base_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        if cutlass.const_expr(self.relays):
            return self._router.final_pool_base_tensor(device_workspace)
        return self._router.pool_expert_base_tensor(device_workspace)

    @cute.jit
    def token_src_metadata_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        if cutlass.const_expr(self.relays):
            return self._router.final_records_tensor(device_workspace)
        return self._router.token_src_metadata_tensor(device_workspace)

    @cute.jit
    def token_src_metadata_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        if cutlass.const_expr(self.relays):
            return device_workspace.ptr(self._router.final_records_region)
        return self._router.token_src_metadata_pointer(device_workspace)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(self.relays):
            return self._router.final_scores_tensor(device_workspace)
        return self._router.fc1_topk_scores_tensor(device_workspace)

    @cute.jit
    def wait_for_pool_record(self, local_expert: Int32, token_in_expert: Int32) -> None:
        if cutlass.const_expr(self.relays):
            intra_count = self._router.intra_count_tensor(self._device_workspace)
            if token_in_expert >= intra_count[local_expert]:
                # relay_written[e] is released by the node peer that holds the message only after
                # that peer's own payload landed (publisher-side join in _scan_relay_routes), so the
                # relay position waits for this one counter and the row pull below may read the
                # message bytes from that peer. This rank's payload_landed says nothing about the
                # peer's receive range, so it is not consulted here.
                self._router.wait_for_relay_records(self._device_workspace, local_expert)

    @cute.jit
    def pull_source_addresses(self, metadata: TokenSrcMetadata, peer_offset: Int64) -> Tuple[Int64, Int64]:
        activation_address, sf_address = TokenCommNonDeterministic.pull_source_addresses(self, metadata, peer_offset)
        # The NVLink-domain component hands back the row pointer; relay tokens need an address. A
        # conditional expression, not an ``if`` statement: the DSL rewrites statements into regions
        # and the value must not change shape across one.
        sf_address = sf_address if isinstance(sf_address, Int64) else sf_address.toint()
        if cutlass.const_expr(self.relays):
            if metadata.src_topk >= Int32(self._router.relay_record_topk_marker):
                # Relay token: the payload is the landed message in the rail peer's receive range
                # (same window offset on every rank of the node); the top-k field carries the source slot.
                source_slot = (metadata.src_topk >> Int32(self._router.relay_record_slot_shift)) & Int32(0x7FF)
                message_address = (
                    self._device_workspace.ptr(self._router.relay_recv_buffer_region).toint()
                    + peer_offset
                    + Int64(source_slot) * Int64(self._router.relay_range_bytes)
                    + Int64(metadata.src_token) * Int64(self._router.relay_message.message_bytes)
                )
                activation_address = message_address + Int64(self._router.relay_message.token_data_offset)
                sf_address = message_address + Int64(self._router.relay_message.sf_data_offset)
        return activation_address, sf_address

    @cute.jit
    def token_back_pushes(self, metadata: TokenSrcMetadata) -> Boolean:
        if cutlass.const_expr(self.relays):
            return metadata.src_topk < Int32(self._router.relay_record_topk_marker)
        return Boolean(True)

    def prepare_device_resources(self, gin_dev_comm, gin_window, gin_shared_window_offset=None) -> None:
        """Materialize the GIN handles at kernel entry (the legacy token_comm_hook_prepare_device_resources).

        ``DevComm`` is a by-value kernel argument whose nccl4py pointer is lazy: it must be
        materialized while the argument dominates the full kernel, and only its scalar GIN
        pointers and the rail-team value may be carried into nested JIT regions (passing the
        native wrappers as jit operands or extracting them later violates dominance).
        """
        if not self.enable_nccl_gin:
            if gin_dev_comm is not None or gin_window is not None:
                raise ValueError("gin_dev_comm / gin_window given but enable_nccl_gin is False.")
            return
        if gin_dev_comm is None or gin_window is None:
            raise ValueError("enable_nccl_gin requires gin_dev_comm and gin_window kernel arguments.")
        from nccl.core.device import cute as nccl_cute

        gin_gpu = gin_dev_comm.gin(
            nccl_cute.GinBackendMask.GDAKI, context_id=0, resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU
        )
        gin_thread = gin_dev_comm.gin(
            nccl_cute.GinBackendMask.GDAKI, context_id=0, resource_sharing_mode=nccl_cute.GinResourceSharingMode.THREAD
        )
        self._gin_gpu_ptr = gin_gpu.ptr
        self._gin_thread_ptr = gin_thread.ptr
        self._team_rail_value = gin_dev_comm.team_rail.__extract_mlir_values__()[0]
        self._gin_window = gin_window
        # None (the fused kernel's default) means the shared workspace starts at the window base.
        self._gin_shared_window_offset = Int64(0) if gin_shared_window_offset is None else gin_shared_window_offset

    def remove_device_members(self) -> None:
        super().remove_device_members()
        self._gin_gpu_ptr = None
        self._gin_thread_ptr = None
        self._team_rail_value = None
        self._gin_window = None
        self._gin_shared_window_offset = None

    @cute.jit
    def assign_device_members(
        self,
        *,
        device_workspace: DeviceWorkspace,
        token_comm_args: TokenCommArgs,
        local_rank: Int32,
        linear_cta_idx: Int32,
    ) -> None:
        # Explicit base call (zero-argument super() inside a @cute.jit method resolves from the runtime type).
        TokenCommNonDeterministic.assign_device_members(
            self,
            device_workspace=device_workspace,
            token_comm_args=token_comm_args,
            local_rank=local_rank,
            linear_cta_idx=linear_cta_idx,
        )
        self._nvlink_barrier.assign_local_rank(local_rank)

    @cute.jit
    def reset_tail(self) -> None:
        """Kernel tail: node-local drain, rail barrier across the logical nodes, counter reset, publish.

        The rail barrier (legacy ``internode_rail_barrier``) keeps every logical node in the same
        launch generation before the registered buffers are reused: a node may only issue the next
        launch's Step 3 / Stage 1 puts once every rank of every other node has drained this launch,
        i.e. finished pulling from its landed messages (I8). Two signal slots alternate phase and
        direction (four-state counter), so a replayed launch cannot inherit a satisfied barrier.
        The arrival signals are issued early, by SM0 relay warp 0 right after its return puts
        (``_issue_rail_arrival``); the tail only waits for the peers' signals.
        """
        if cutlass.const_expr(not self.relays):
            TokenCommNonDeterministic.reset_tail(self)
        else:
            transfer_warp_idx = self._transfer_warp_idx
            lane_idx = self._lane_idx
            transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx
            iket.range_push("tail.wait_relay_warps")
            self._relay_done_barrier().arrive_and_wait()
            iket.range_pop()
            iket.range_push("tail.nvlink_drain")
            self._nvlink_barrier.arrive_and_wait(
                self.transfer_thread_count,
                Int32(self.promised_launchable_sm_count),
                self._linear_cta_idx,
                transfer_thread_idx,
                prologue_grid_sync=True,
                epilogue_grid_sync=False,
            )
            iket.range_pop()
            iket.range_push("tail.rail_barrier")
            if self._linear_cta_idx == Int32(0):
                if transfer_thread_idx == Int32(0):
                    self._rail_barrier()
                    self._router.stamp(self._device_workspace, 14)
            iket.range_pop()
            total_reset_threads = self.promised_launchable_sm_count * self.transfer_thread_count
            global_reset_thread = self._linear_cta_idx * Int32(self.transfer_thread_count) + transfer_thread_idx
            iket.range_push("tail.reset_workspace")
            self._device_workspace.reset_tail_space("shared", global_reset_thread, total_reset_threads)
            self._device_workspace.reset_tail_space("local", global_reset_thread, total_reset_threads)
            iket.range_pop()
            iket.range_push("tail.nvlink_publish")
            self._nvlink_barrier.arrive_and_wait(
                self.transfer_thread_count,
                Int32(self.promised_launchable_sm_count),
                self._linear_cta_idx,
                transfer_thread_idx,
                prologue_grid_sync=True,
                epilogue_grid_sync=False,
            )
            iket.range_pop()
            if self._linear_cta_idx == Int32(0):
                if transfer_thread_idx == Int32(0):
                    self._router.stamp(self._device_workspace, 15)

    @cute.jit
    def _rail_barrier(self) -> None:
        """SM0 transfer thread 0: wait for every remote node's generation signal of this phase, advance the counter.

        The signals themselves were issued by SM0 relay warp 0 right after its return puts
        (``_issue_rail_arrival``), so this wait usually finds them landed.
        """
        from nccl.core.device import cute as nccl_cute
        from nccl.core.device.cute.gin import Gin

        router = self._router
        gin = Gin(ptr=self._gin_thread_ptr)
        counter = self._device_workspace.ptr(router.rail_barrier_counter_region)
        status = cute.arch.load(counter, Int32, sem="relaxed", scope="gpu") & Int32(3)
        signal_phase = status & Int32(1)
        target = Uint64(self.num_nodes - 1)
        if (status >> Int32(1)) != Int32(0):
            target = Uint64(0)
        signal_id = Int32(router.barrier_signal_base) + signal_phase
        cute.arch.atomic_add(counter, Int32(1), sem="relaxed", scope="gpu")
        barrier_start = read_clock64()
        while gin.read_signal(signal=signal_id, bits=64, ord=nccl_cute.MemoryOrder.ACQUIRE) != target:
            if read_clock64() - barrier_start > Int64(self.rail_barrier_timeout_cycles):
                _device_trap()

    def register_smem_regions(self, workspace: SmemWorkspace) -> None:
        TokenCommNonDeterministic.register_smem_regions(self, workspace)
        if self.relays:
            entries = self.ranks_per_node * self.experts_per_rank
            workspace.register_tensor(self.relay_fc2_target_smem_region, cutlass.Int32, (entries,), byte_alignment=16)
            workspace.register_tensor(self.relay_fc2_done_smem_region, cutlass.Int32, (entries,), byte_alignment=16)

    @cute.jit
    def relay_warp_body(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Body of relay warps 12-15 (Stage 3c receiver side; the sends happen in the router).

        SM0's relay warps: Step 5a (wait for every source node's Step 3 block: row plus header
        plane), publish "rows landed" to the node peers and wait for theirs, Step 5c (final
        per-expert layout from the intra counts and the rows landed at all rail peers of the node),
        copy the intra records / scores into the final pool positions, publish ``layout_ready``;
        SM0 relay warp 0 then waits for every source node's Stage 1 credit and publishes
        ``payload_landed`` (the layout and the scan need the rows and headers only). Every other
        relay warp of the grid scans the landed header planes of this rank for routes to the node's
        experts and writes the relay records (``relay_written[e]`` counts the records of expert
        ``e``, released behind the scanning rank's ``payload_landed``; the token_in relay pulls wait
        for it alone). Degenerate
        configuration: exit.
        """
        if cutlass.const_expr(self.relays):
            thread_idx, _, _ = cute.arch.thread_idx()
            relay_thread = thread_idx - Int32(self.relay_warp_idx_start * 32)
            relay_warp = cute.arch.make_warp_uniform(relay_thread // Int32(32))
            lane = relay_thread % Int32(32)
            relay_threads = self.relay_warp_count * 32
            group_barrier = pipeline.NamedBarrier(barrier_id=self.relay_group_barrier_id, num_threads=relay_threads)
            if self._linear_cta_idx == Int32(0):
                if relay_warp == Int32(0):
                    if lane == Int32(0):
                        self._router.stamp(self._device_workspace, 3)
                iket.range_push("relay.sm0_metadata")
                self._sm0_metadata_phase(relay_warp, lane, group_barrier)
                iket.range_pop()
            iket.range_push("relay.wait_layout_ready")
            self._router.wait_for_layout_ready(self._device_workspace)
            iket.range_pop()
            # The pushed records need every owner's published relay_base (system fence + arrival).
            iket.range_push("relay.wait_layout_published")
            self._wait_layout_published(lane)
            iket.range_pop()
            # SM0 relay warp 0 owns the payload credit wait (end of _sm0_metadata_phase) and takes no
            # scan share, so no record waits for the payload; SM0 relay warp 1 takes the scan-side
            # stamps (9, 10, 16) instead.
            if self._linear_cta_idx == Int32(0):
                if relay_warp == Int32(1):
                    if lane == Int32(0):
                        self._router.stamp(self._device_workspace, 9)
            # The scan reads the landed header planes, which arrived with the rows (acquired by SM0
            # before layout_ready): it runs while the payload is still in flight.
            iket.range_push("relay.scan_routes")
            self._scan_relay_routes(relay_warp, lane)
            iket.range_pop()
            if self._linear_cta_idx == Int32(0):
                if relay_warp == Int32(1):
                    if lane == Int32(0):
                        self._router.stamp(self._device_workspace, 10)
                    self._stamp_records_and_fc2_completion(lane)
            iket.range_push("relay.gather_reduce")
            self._gather_reduce_and_return(relay_warp, lane, smem_workspace, smem_base, group_barrier)
            iket.range_pop()
            # Release this CTA's transfer warps into the tail only now (see reset_tail).
            self._relay_done_barrier().arrive()

    def _relay_done_barrier(self):
        return pipeline.NamedBarrier(
            barrier_id=self.relay_done_barrier_id, num_threads=self.transfer_thread_count + self.relay_warp_count * 32
        )

    @cute.jit
    def _sm0_metadata_phase(self, relay_warp: Int32, lane: Int32, group_barrier) -> None:
        router = self._router
        workspace = self._device_workspace
        mapper = self._token_comm_args.peer_rank_ptr_mapper
        experts = self.experts_per_rank
        slots = router.remote_node_count
        ranks_per_node = self.ranks_per_node
        node_base = (self._local_rank // Int32(ranks_per_node)) * Int32(ranks_per_node)
        in_node = self._local_rank - node_base
        from nccl.core.device import cute as nccl_cute
        from nccl.core.device.cute.gin import Gin

        # Step 5a (one thread): Step 3 rows of every source slot, monotonic shadows (I8). The layout
        # needs only the rows (the node peers read this rank's recv_splits rows after landed_ready);
        # the Stage 1 payload credits are waited after layout_ready (split wait, see the end of this
        # method), so the layout no longer sits behind the payload landing.
        if relay_warp == Int32(0):
            gin = Gin(ptr=self._gin_gpu_ptr)
            coop = nccl_cute.thread()
            # Three lane-0 waits, each closed by a warp barrier (bar.warp.sync orders memory among
            # the warp's threads, so the lanes of Step 5c read what lane 0 acquired).
            iket.range_push("relay.step5a_wait_rows")
            if lane == Int32(0):
                for src_slot in cutlass.range_constexpr(slots):
                    signal_id = Int32(router.splits_signal_base + src_slot)
                    shadow = gin.signal_shadow_pointer(signal=signal_id)
                    old_target = cute.arch.atomic_add(shadow, Uint64(1), sem="relaxed", scope="gpu")
                    gin.wait_signal(
                        coop, signal=signal_id, least=Uint64(old_target) + Uint64(1), bits=64, ord=nccl_cute.MemoryOrder.ACQUIRE
                    )
                # Landed rows are visible to this thread: publish "rows landed" to the node peers (I2).
                cute.arch.fence_acq_rel_sys()
                landed_address = workspace.ptr(router.landed_ready_region).toint()
                for peer_index in cutlass.range_constexpr(ranks_per_node):
                    red_add_relaxed_sys_s32(mapper.map(landed_address, node_base + Int32(peer_index), Int64(0)), Int32(1))
            cute.arch.sync_warp()
            iket.range_pop()
            iket.range_push("relay.wait_node_landed")
            if lane == Int32(0):
                landed = workspace.ptr(router.landed_ready_region)
                while cute.arch.load(landed, Int32, sem="acquire", scope="sys") != Int32(ranks_per_node):
                    nanosleep(200)
                router.stamp(workspace, 5)
            cute.arch.sync_warp()
            iket.range_pop()
            iket.range_push("relay.wait_metadata_ready")
            if lane == Int32(0):
                # The intra records of the node peers were pushed before metadata_ready.
                metadata_ready = workspace.ptr(router.metadata_ready_region)
                while cute.arch.load(metadata_ready, Int32, sem="acquire", scope="sys") != Int32(router.metadata_ready_target):
                    nanosleep(200)
                router.stamp(workspace, 6)
            cute.arch.sync_warp()
            iket.range_pop()
        group_barrier.arrive_and_wait()
        # Step 5c on relay warp 0: the experts of the rank are processed in chunks of 32, lane e of a
        # chunk owning expert chunk*32 + e; the exclusive prefix of the padded widths carries across
        # chunks (warp-uniform carry from lane 31). Before 2026-10-22 a single pass handled lane e ==
        # expert e only, so ranks with more than 32 experts (seed EP 16 x 640 experts = 40 per rank)
        # never laid out experts 32.., leaving zero pool records that the pulls then read.
        iket.range_push("relay.step5c_layout")
        if relay_warp == Int32(0):
            # The intra totals are complete once every node peer's size broadcast arrived.
            router.wait_for_sizes_ready(workspace)
            intra_sizes = router.local_expert_sizes(workspace, self._local_rank)
            recv_splits_address = workspace.ptr(router.recv_splits_region).toint()
            relay_base = workspace.tensor(router.layout_relay_base_region)
            carry = Int32(0)
            for chunk in cutlass.range_constexpr((experts + 31) // 32):
                chunk_expert = Int32(chunk * 32) + lane
                expert = cutlass.min(chunk_expert, Int32(experts - 1))
                active = chunk_expert < Int32(experts)
                intra = Int32(0)
                if active:
                    intra = intra_sizes[expert]
                # Relay rows of expert e per (source slot, rail peer), read from every node peer's landed row.
                relay_total = Int32(0)
                bucket_counts = []
                for src_slot in cutlass.range_constexpr(slots):
                    for peer_index in cutlass.range_constexpr(ranks_per_node):
                        count = Int32(0)
                        if active:
                            row_address = (
                                mapper.map(recv_splits_address, node_base + Int32(peer_index), Int64(0))
                                + Int64(src_slot * router.splits_row_pitch_bytes)
                                + Int64(in_node * experts + expert) * Int64(4)
                            )
                            count = cute.arch.load(
                                cute.make_ptr(Int32, row_address, AddressSpace.gmem, assumed_align=4), Int32, sem="relaxed", scope="sys"
                            )
                        bucket_counts.append(count)
                        relay_total = relay_total + count
                total = intra + relay_total
                padded = ((total + Int32(self.token_padding_block - 1)) // Int32(self.token_padding_block)) * Int32(
                    self.token_padding_block
                )
                # Exclusive prefix of the padded widths over the chunk's experts (lane order): Hillis-Steele
                # scan with index shuffles (the source lane is clamped; lanes below the distance keep their
                # value). Inactive lanes contribute 0.
                prefix = padded
                for shift in cutlass.range_constexpr(5):
                    distance = 1 << shift
                    source_lane = cutlass.max(lane - Int32(distance), Int32(0))
                    neighbour = cute.arch.shuffle_sync(prefix, source_lane)
                    if lane >= Int32(distance):
                        prefix = prefix + neighbour
                pool_base = carry + prefix - padded
                if active:
                    router.final_sizes_tensor(workspace)[expert] = total
                    router.final_pool_base_tensor(workspace)[expert] = pool_base
                    router.intra_count_tensor(workspace)[expert] = intra
                    running = pool_base + intra
                    bucket = 0
                    for src_slot in cutlass.range_constexpr(slots):
                        for peer_index in cutlass.range_constexpr(ranks_per_node):
                            relay_base[expert, Int32(src_slot), Int32(peer_index)] = running
                            running = running + bucket_counts[bucket]
                            bucket += 1
                # Chunk total (inclusive prefix of lane 31) becomes the next chunk's base.
                carry = carry + cute.arch.shuffle_sync(prefix, Int32(31))
            cute.arch.sync_warp()
        group_barrier.arrive_and_wait()
        iket.range_pop()
        if relay_warp == Int32(0):
            if lane == Int32(0):
                router.stamp(workspace, 7)
        iket.range_push("relay.copy_intra_records")
        # Copy the intra records (and scores) from the router's staging layout into the final layout:
        # one pass over the intra positions of the rank, the expert of a position found against the
        # running prefix of the intra counts held in registers, four positions in flight per thread
        # (the per-expert loop with its dependent base loads took 16 us at N2R4 T=256 under IKET).
        intra_count = router.intra_count_tensor(workspace)
        final_bases = router.final_pool_base_tensor(workspace)
        staging_bases = router.pool_expert_base_tensor(workspace)
        staging_records = router.token_src_metadata_tensor(workspace)
        final_records = router.final_records_tensor(workspace)
        relay_thread = relay_warp * Int32(32) + lane
        relay_threads = self.relay_warp_count * 32
        expert_counts = []
        expert_prefix_end = []
        expert_final_base = []
        expert_staging_base = []
        total_intra = Int32(0)
        for local_expert in cutlass.range_constexpr(experts):
            count = intra_count[local_expert]
            expert_counts.append(count)
            expert_final_base.append(final_bases[local_expert])
            expert_staging_base.append(staging_bases[local_expert])
            total_intra = total_intra + count
            expert_prefix_end.append(total_intra)
        copy_rounds = (total_intra + Int32(relay_threads - 1)) // Int32(relay_threads)
        for copy_round in cutlass.range(copy_rounds, unroll=4):
            position = Int32(copy_round) * Int32(relay_threads) + relay_thread
            if position < total_intra:
                expert_index = Int32(0)
                for local_expert in cutlass.range_constexpr(experts):
                    if position >= expert_prefix_end[local_expert]:
                        expert_index = expert_index + Int32(1)
                source_index = Int32(0)
                destination_index = Int32(0)
                for local_expert in cutlass.range_constexpr(experts):
                    if expert_index == Int32(local_expert):
                        offset = position - (expert_prefix_end[local_expert] - expert_counts[local_expert])
                        source_index = expert_staging_base[local_expert] + offset
                        destination_index = expert_final_base[local_expert] + offset
                final_records[destination_index] = staging_records[source_index]
                if cutlass.const_expr(self.apply_topk_at_fc1):
                    router.final_scores_tensor(workspace)[destination_index] = router.fc1_topk_scores_tensor(workspace)[
                        source_index
                    ]
        group_barrier.arrive_and_wait()
        iket.range_pop()
        if relay_thread == Int32(0):
            # The final sizes are read by the node peers' combine (I2): system fence, then one arrival per peer.
            cute.arch.fence_acq_rel_sys()
            published_address = workspace.ptr(router.layout_published_region).toint()
            for peer_index in cutlass.range_constexpr(ranks_per_node):
                red_add_relaxed_sys_s32(mapper.map(published_address, node_base + Int32(peer_index), Int64(0)), Int32(1))
            cute.arch.atomic_add(workspace.ptr(router.layout_ready_region), Int32(1), sem="release", scope="gpu")
            router.stamp(workspace, 8)
            # Step 5a, second half: the Stage 1 payload credit of every source slot (monotonic shadows,
            # I8; the credit is the expert-hit sum of the landed row). The scan warps of this rank
            # release their relay_written increments only behind payload_landed (gpu-scope release
            # here, acquire in _scan_relay_routes), which is what lets the owners' token_in read the
            # message bytes behind relay_written alone; the scan itself needs the header planes
            # only, which came with the rows. Stamp 4 keeps its meaning "payload landed".
            iket.range_push("relay.step5a_wait_payload")
            payload_gin = Gin(ptr=self._gin_gpu_ptr)
            payload_coop = nccl_cute.thread()
            recv_splits = workspace.tensor(router.recv_splits_region)
            for src_slot in cutlass.range_constexpr(slots):
                credit = Int32(0)
                for node_expert in cutlass.range_constexpr(router.node_experts):
                    credit = credit + recv_splits[Int32(src_slot), Int32(node_expert)]
                credit_signal = Int32(router.relay_node_signal_base + src_slot)
                credit_shadow = payload_gin.signal_shadow_pointer(signal=credit_signal)
                old_credit = cute.arch.atomic_add(credit_shadow, Uint64(credit), sem="relaxed", scope="gpu")
                payload_gin.wait_signal(
                    payload_coop,
                    signal=credit_signal,
                    least=Uint64(old_credit) + Uint64(credit),
                    bits=64,
                    ord=nccl_cute.MemoryOrder.ACQUIRE,
                )
            cute.arch.fence_acq_rel_sys()
            cute.arch.atomic_add(workspace.ptr(router.payload_landed_region), Int32(1), sem="release", scope="gpu")
            router.stamp(workspace, 4)
            iket.range_pop()

    @cute.jit
    def _window_tensor(self, region_name: str, region_byte_offset: Int64, dtype, layout, alignment: int):
        """Window-relative view inside a shared region (see InternodeMetadataRouter._window_tensor)."""
        byte_offset = (
            self._gin_shared_window_offset + Int64(self._device_workspace.offset(region_name)) + region_byte_offset
        )
        window_base = self._gin_window.tensor(Int32, cute.make_layout((1,), stride=(1,))).iterator.toint()
        return cute.make_tensor(
            cute.make_ptr(dtype, window_base + byte_offset, AddressSpace.gmem, assumed_align=alignment), layout
        )

    @cute.jit
    def _gather_reduce_and_return(
        self, relay_warp: Int32, lane: Int32, smem_workspace: SmemWorkspace, smem_base: cute.Pointer, group_barrier
    ) -> None:
        """Combine Stage 1 + 2 of the rail peer (relay warps, joined by the epilogue warps) and the origin's credit wait (SM0).

        Every landed message of this rank is reduced per (message, chunk group) work item: for its
        routed top-k slots, in slot order, the owner's FC2 row (BF16, pushed by the owner's epilogue
        into this rank's landing cell) is accumulated in FP32 and cast once to the BF16 node partial
        (I11); the owner's per-expert fc2_done target is derived from its published final size (I9).
        The items are strided over every combine worker of the grid: the relay warps, plus the
        epilogue warps once they ran out of tiles (``epilogue_tail_body``). SM0 then returns each
        source node's dense partial range with one put carrying the message count as the ADD credit,
        and waits for the credits of the nodes this rank sent to (I5, I8).
        """
        self._build_fc2_cache(relay_warp, lane, smem_workspace, smem_base, group_barrier)
        self._gather_items(relay_warp, lane, smem_workspace, smem_base)
        self._finish_gather_worker(lane)
        if self._linear_cta_idx == Int32(0):
            if relay_warp == Int32(0):
                self._return_partials_and_wait(lane)

    @cute.jit
    def epilogue_tail_body(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer, epilogue_warp: Int32) -> None:
        """Epilogue warp after its last tile: join the gather as combine worker ``relay_warp_count + epilogue_warp``.

        The epilogue warps are idle from their last FC2 tile to the kernel end, which is exactly the
        window in which the gather runs (phase_interleave finishes every expert together, so the items
        become ready only at the end of FC2): joining doubles the worker warps (N4R8 phase stamps
        before the join: 32 us after the last FC2 tile for about 4.3 items per relay warp). The join
        barrier orders the relay warps' SMEM cache build before the first epilogue-warp read; the
        relay warps only arrive on it.
        """
        if cutlass.const_expr(self.relays):
            thread_idx, _, _ = cute.arch.thread_idx()
            lane = thread_idx % Int32(32)
            if self._linear_cta_idx == Int32(0):
                if epilogue_warp == Int32(0):
                    if lane == Int32(0):
                        # fc2_local_done (stamp 17, phase_stamps only): the tail entry would follow this
                        # gather (finalize barrier), so the stamp is taken here.
                        self._stamp_local_fc2_done()
            iket.range_push("mega.epilogue_gather")
            # The relay warps arrive on the join barrier after their scan, i.e. behind layout_ready
            # (SM0 acquired the header planes before publishing it): the CTA barrier orders this
            # warp's header loads after that acquire (as for the cache).
            self._combine_join_barrier().arrive_and_wait()
            self._gather_items(Int32(self.relay_warp_count) + epilogue_warp, lane, smem_workspace, smem_base)
            self._finish_gather_worker(lane)
            iket.range_pop()

    def _combine_join_barrier(self):
        return pipeline.NamedBarrier(
            barrier_id=self.combine_join_barrier_id,
            num_threads=(self.relay_warp_count + self.combine_epilogue_worker_warps) * 32,
        )

    @cute.jit
    def _build_fc2_cache(
        self, relay_warp: Int32, lane: Int32, smem_workspace: SmemWorkspace, smem_base: cute.Pointer, group_barrier
    ) -> None:
        """Relay warps: per-CTA SMEM cache of every owner's fc2_done target per expert plus a "done observed" flag.

        One NVLink read of the owner's published final size per entry (every node peer published its
        final layout: system fence + arrival), spread over the CTA's relay threads. A warp that observes
        fc2_done >= target publishes the flag with a CTA-scope release; a warp that reads the flag with
        a CTA-scope acquire is ordered after that system acquire, hence after the owner's FC2 row
        stores (I9 chain through the CTA). Every relay thread then fences and arrives on the join
        barrier so the epilogue warps read a complete cache.
        """
        router = self._router
        mapper = self._token_comm_args.peer_rank_ptr_mapper
        experts = self.experts_per_rank
        ranks_per_node = self.ranks_per_node
        node_base = (self._local_rank // Int32(ranks_per_node)) * Int32(ranks_per_node)
        relay_threads = self.relay_warp_count * 32
        sizes_address = self._device_workspace.ptr(router.layout_sizes_region).toint()
        iket.range_push("relay.gather_cache_build")
        fc2_targets = smem_workspace.tensor(self.relay_fc2_target_smem_region, smem_base)
        fc2_seen = smem_workspace.tensor(self.relay_fc2_done_smem_region, smem_base)
        relay_thread = relay_warp * Int32(32) + lane
        entry_count = ranks_per_node * experts
        entry = relay_thread
        while entry < Int32(entry_count):
            owner_in_node = entry // Int32(experts)
            local_expert = entry % Int32(experts)
            owner_offset = mapper.map(Int64(0), node_base + owner_in_node, Int64(0))
            owner_size = cute.arch.load(
                cute.make_ptr(Int32, sizes_address + owner_offset + Int64(local_expert) * Int64(4), AddressSpace.gmem, assumed_align=4),
                Int32,
                sem="relaxed",
                scope="sys",
            )
            tiles = (owner_size + Int32(self.tokens_per_fc1_ready_slot - 1)) // Int32(self.tokens_per_fc1_ready_slot)
            fc2_targets[entry] = tiles * Int32(self.fc2_done_signals_per_token_tile)
            fc2_seen[entry] = Int32(0)
            entry = entry + Int32(relay_threads)
        group_barrier.arrive_and_wait()
        cute.arch.fence_acq_rel_cta()
        self._combine_join_barrier().arrive()
        iket.range_pop()

    @cute.jit
    def _finish_gather_worker(self, lane: Int32) -> None:
        """One combine worker warp is done: system fence (the NIC reads the partial rows after SM0 posts the
        range puts), then one arrival on ``combine_reduce_done`` (Combine Stage 2 rendezvous of all workers)."""
        cute.arch.sync_warp()
        if lane == Int32(0):
            cute.arch.fence_acq_rel_sys()
            cute.arch.atomic_add(
                self._device_workspace.ptr(self._router.combine_reduce_done_region), Int32(1), sem="release", scope="gpu"
            )

    @cute.jit
    def _gather_items(self, worker: Int32, lane: Int32, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Combine worker ``worker`` of this CTA: its grid-strided (message, chunk group) items of every source slot."""
        router = self._router
        workspace = self._device_workspace
        mapper = self._token_comm_args.peer_rank_ptr_mapper
        experts = self.experts_per_rank
        slots = router.remote_node_count
        ranks_per_node = self.ranks_per_node
        node_base = (self._local_rank // Int32(ranks_per_node)) * Int32(ranks_per_node)
        hidden = self.hidden_size
        chunk_elements = self.combine_chunk_elements
        lane_elements = chunk_elements // 32
        lane_bytes = lane_elements * 2
        # Worker numbering (2026-10-23, seed EP 64 gather): every relay warp of the grid comes first, then
        # every epilogue warp. The relay warps are free as soon as their scan is done, the epilogue warps
        # only after their last FC2 tile, so with few items (small T) the gather no longer waits for FC2 to
        # end. SM0's relay warps are numbered last among the relay warps: they also carry the layout,
        # return-put and origin-wait duties.
        relay_warps = self.relay_warp_count
        sm_count = Int32(self.promised_launchable_sm_count)
        relay_cta = self._linear_cta_idx - Int32(1)
        if self._linear_cta_idx == Int32(0):
            relay_cta = sm_count - Int32(1)
        # (DSL: a value assigned under a dynamic branch must exist before it.)
        global_worker = sm_count * Int32(relay_warps) + self._linear_cta_idx * Int32(
            self.combine_epilogue_worker_warps
        ) + (worker - Int32(relay_warps))
        if worker < Int32(relay_warps):
            global_worker = relay_cta * Int32(relay_warps) + worker
        total_workers = Int32(self.combine_worker_total)
        recv_splits = workspace.tensor(router.recv_splits_region)
        landing_address = workspace.ptr(router.combine_landing_rows_region).toint()
        # Local mirror of every node peer's fc2_done (fanned out by the owners' epilogues), indexed by node expert.
        mirror_address = workspace.ptr(router.fc2_done_mirror_region).toint()
        partial_address = workspace.ptr(router.intra_node_reduce_region).toint()
        load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
        chunk_count = hidden // chunk_elements
        gather_groups = self.combine_gather_groups
        chunks_per_group = ceil_div(chunk_count, gather_groups)
        fc2_targets = smem_workspace.tensor(self.relay_fc2_target_smem_region, smem_base)
        fc2_seen = smem_workspace.tensor(self.relay_fc2_done_smem_region, smem_base)
        # One flattened (slot, message, chunk group) index space over every source slot (the per-slot
        # loop restarted the grid stride at worker 0 for each slot, so with a few messages per slot the
        # whole gather landed on the first two CTAs, half of it on their epilogue warps).
        slot_counts = []
        total_messages = Int32(0)
        for slot_index in cutlass.range_constexpr(slots):
            slot_count = recv_splits[Int32(slot_index), Int32(router.node_experts)]
            slot_counts.append(slot_count)
            total_messages = total_messages + slot_count
        work = global_worker
        while work < total_messages * Int32(gather_groups):
            message = work // Int32(gather_groups)
            chunk_group = work % Int32(gather_groups)
            # Locate the slot of the flat message index (slots are constexpr, counts dynamic).
            src_slot = Int32(0)
            for slot_index in cutlass.range_constexpr(slots):
                if src_slot == Int32(slot_index):
                    if message >= slot_counts[slot_index]:
                        message = message - slot_counts[slot_index]
                        src_slot = src_slot + Int32(1)
            if True:
                # Route words and weights come from the landed header plane (local memory, visible
                # since layout_ready); the gather reads no message bytes.
                header_address = self._relay_header_address(Int32(src_slot), message)
                partial_row = partial_address + Int64(src_slot) * Int64(router.combine_range_bytes) + Int64(message) * Int64(router.combine_row_bytes)
                # Route decode once per message (warp-uniform values). Every route word and weight
                # is loaded before any dependent use (one round trip for the message instead of one per
                # slot), and the owners' completion is checked by one lane per top-k slot at the same
                # time (one round trip instead of four in sequence): the decode used to cost more
                # than the row gather itself (N4R8 phase stamps: 39 us after FC2 for about 4.3 items
                # per warp).
                iket.range_push("relay.gather_route_decode")
                route_words = []
                raw_weights = []
                for topk_slot in cutlass.range_constexpr(self.topk):
                    route_words.append(
                        cute.arch.load(
                            cute.make_ptr(Int32, header_address + Int64(topk_slot * 4), AddressSpace.gmem, assumed_align=4),
                            Int32,
                            sem="relaxed",
                            scope="gpu",
                        )
                    )
                if cutlass.const_expr(not router.apply_topk_at_fc1):
                    # Unrouted slots hold an unused weight inside the header; the load is unconditional.
                    for topk_slot in cutlass.range_constexpr(self.topk):
                        raw_weights.append(
                            cute.arch.load(
                                cute.make_ptr(
                                    cutlass.Float32,
                                    header_address + Int64(router.relay_header_weights_offset + topk_slot * 4),
                                    AddressSpace.gmem,
                                    assumed_align=4,
                                ),
                                cutlass.Float32,
                                sem="relaxed",
                                scope="gpu",
                            )
                        )
                # The owner's completion (I9), lane k for top-k slot k, read from this rank's local mirror
                # (the owner's epilogue fans its increments out to every node peer): the CTA cache
                # short-circuits the system-scope acquire once any warp of the CTA observed the target
                # (CTA release/acquire pair); the warp barrier below orders every lane's row loads after
                # the acquires.
                lane_route_word = route_words[0]
                for topk_slot in cutlass.range_constexpr(1, self.topk):
                    if lane == Int32(topk_slot):
                        lane_route_word = route_words[topk_slot]
                if lane < Int32(self.topk):
                    if lane_route_word != Int32(router.unrouted_route_word):
                        lane_node_expert = lane_route_word & Int32(0xFFFF)
                        seen_pointer = fc2_seen.iterator + lane_node_expert
                        if cute.arch.load(seen_pointer, Int32, sem="acquire", scope="cta") == Int32(0):
                            target = fc2_targets[lane_node_expert]
                            done_pointer = cute.make_ptr(
                                Int32, mirror_address + Int64(lane_node_expert) * Int64(4), AddressSpace.gmem, assumed_align=4
                            )
                            while cute.arch.load(done_pointer, Int32, sem="acquire", scope="sys") < target:
                                nanosleep(200)
                            cute.arch.atomic_add(seen_pointer, Int32(1), sem="release", scope="cta")
                cute.arch.sync_warp()
                route_valid = []
                route_row_address = []
                route_weight = []
                for topk_slot in cutlass.range_constexpr(self.topk):
                    valid = route_words[topk_slot] != Int32(router.unrouted_route_word)
                    # Unrouted slots read the message's own partial row (local, never used): every
                    # slot's loads are issued unconditionally so one chunk costs one round trip.
                    row_address = partial_row
                    weight = cutlass.Float32(1.0)
                    if valid:
                        # The owner pushed this route's FC2 row into this rank's landing cell (slot, message, k).
                        row_address = landing_address + (
                            (Int64(src_slot) * Int64(router.max_relay_tokens_per_node) + Int64(message)) * Int64(self.topk)
                            + Int64(topk_slot)
                        ) * Int64(hidden * 2)
                        if cutlass.const_expr(not router.apply_topk_at_fc1):
                            weight = raw_weights[topk_slot]
                    route_valid.append(valid)
                    route_row_address.append(row_address)
                    route_weight.append(weight)
                iket.range_pop()
                iket.range_push("relay.gather_rows")
                for chunk_local in cutlass.range_constexpr(chunks_per_group):
                    chunk_idx = chunk_group * Int32(chunks_per_group) + Int32(chunk_local)
                    chunk_valid = chunk_idx < Int32(chunk_count)
                    # Out-of-range chunks of the last group read and write chunk 0 with weight 0 (see below).
                    chunk_offset = Int64(chunk_idx) * Int64(chunk_elements * 2) + Int64(lane) * Int64(lane_bytes)
                    if chunk_idx >= Int32(chunk_count):
                        chunk_offset = Int64(lane) * Int64(lane_bytes)
                    acc = cute.make_rmem_tensor((lane_elements,), cutlass.Float32)
                    for element in cutlass.range_constexpr(lane_elements):
                        acc[element] = cutlass.Float32(0.0)
                    # Latency-bound gather (NVLink loads from the owners): issue every slot's vectors of
                    # the chunk before the first FMA so the chunk costs one round trip, not one per slot
                    # (2026-09-28 IKET: 25-30 us per message with per-slot load -> FMA -> load).
                    chunk_values = []
                    for topk_slot in cutlass.range_constexpr(self.topk):
                        values = cute.make_rmem_tensor((lane_elements,), cutlass.BFloat16)
                        source = cute.make_tensor(
                            cute.make_ptr(cutlass.BFloat16, route_row_address[topk_slot] + chunk_offset, AddressSpace.gmem, assumed_align=16),
                            cute.make_layout((lane_elements,), stride=(1,)),
                        )
                        cute.copy(load_atom, source, values)
                        chunk_values.append(values)
                    for topk_slot in cutlass.range_constexpr(self.topk):
                        if route_valid[topk_slot]:
                            values = chunk_values[topk_slot]
                            weight_pair = (route_weight[topk_slot], route_weight[topk_slot])
                            for element in cutlass.range_constexpr(0, lane_elements, 2):
                                value_pair = (cutlass.Float32(values[element]), cutlass.Float32(values[element + 1]))
                                acc[element], acc[element + 1] = cute.arch.fma_packed_f32x2(
                                    value_pair, weight_pair, (acc[element], acc[element + 1])
                                )
                    if chunk_valid:
                        partial = cute.make_rmem_tensor((lane_elements,), cutlass.BFloat16)
                        partial.store(acc.load().to(cutlass.BFloat16))
                        destination = cute.make_tensor(
                            cute.make_ptr(cutlass.BFloat16, partial_row + chunk_offset, AddressSpace.gmem, assumed_align=16),
                            cute.make_layout((lane_elements,), stride=(1,)),
                        )
                        cute.copy(load_atom, partial, destination)
                iket.range_pop()
                work = work + total_workers

    @cute.jit
    def _issue_return_put(self, src_slot: Int32) -> None:
        """One coalesced return put: this rank's partials of source slot ``src_slot`` to that node's landing range.

        The GIN handle is rebuilt here from the entry-materialized pointer: a ``Gin`` object must not
        cross a nested JIT boundary (MLIR dominance failure).
        """
        from nccl.core.device import cute as nccl_cute
        from nccl.core.device.cute.gin import Gin

        router = self._router
        workspace = self._device_workspace
        gin = Gin(ptr=self._gin_gpu_ptr)
        coop = nccl_cute.thread()
        source_node, my_slot_at_source = router._remote_slot_geometry_for(src_slot, self._local_rank)
        recv_splits = workspace.tensor(router.recv_splits_region)
        message_count = recv_splits[src_slot, Int32(router.node_experts)]
        if message_count > Int32(0):
            range_layout = cute.make_layout((message_count * Int32(router.combine_row_bytes),), stride=(1,))
            src_range = self._window_tensor(
                router.intra_node_reduce_region, Int64(src_slot) * Int64(router.combine_range_bytes), cutlass.Uint8, range_layout, 16
            )
            dst_range = self._window_tensor(
                router.combine_landing_region, Int64(my_slot_at_source) * Int64(router.combine_range_bytes), cutlass.Uint8, range_layout, 16
            )
            gin.put(
                self._team_rail_value,
                source_node,
                self._gin_window,
                dst_range,
                self._gin_window,
                src_range,
                coop,
                is_signal=True,
                signal_id=Int32(router.combine_arrival_signal_base) + my_slot_at_source,
                signal_op=GIN_SIGNAL_OP_ADD,
                signal_op_arg=Int64(message_count),
                given_release=nccl_cute.ThreadScope.THREAD,
            )

    @cute.jit
    def _return_partials_and_wait(self, lane: Int32) -> None:
        """SM0 relay warp 0: one coalesced return put per source node (lanes), then the origin's credit waits."""
        from nccl.core.device import cute as nccl_cute
        from nccl.core.device.cute.gin import Gin

        router = self._router
        workspace = self._device_workspace
        slots = router.remote_node_count
        gin = Gin(ptr=self._gin_gpu_ptr)
        coop = nccl_cute.thread()
        # Lane 0 acquires the rendezvous; the warp barrier orders the issuing lanes after it.
        iket.range_push("relay.wait_reduce_done")
        if lane == Int32(0):
            reduce_done = workspace.ptr(router.combine_reduce_done_region)
            while cute.arch.load(reduce_done, Int32, sem="acquire", scope="gpu") != Int32(self.combine_worker_total):
                nanosleep(200)
            router.stamp(workspace, 11)
        cute.arch.sync_warp()
        iket.range_pop()
        iket.range_push("relay.return_fence")
        if lane < Int32(slots):
            cute.arch.fence_acq_rel_sys()
        cute.arch.sync_warp()
        iket.range_pop()
        iket.range_push("relay.return_put")
        # One lane per source node; a sequential node-rotated issue order was measured to land no earlier.
        if lane < Int32(slots):
            self._issue_return_put(lane)
        cute.arch.sync_warp()
        if lane == Int32(0):
            router.stamp(workspace, 12)
        iket.range_pop()
        # Early rail arrive (I8; the legacy internode_rail_arrive), right after the return puts so the
        # signal crosses the fabric while this rank waits for its own partials: every read of this
        # launch's landed rows and payload on this rank and its node peers completed before
        # reduce_done (fc2_done and layout_published chains), so a peer that sees the signal may
        # overwrite this rank's receive ranges in the next launch. The return source needs no later
        # arrival: this rank rewrites staging slot s in launch i+1 only for messages node s sent in
        # launch i+1, which node s issues after its own tail of launch i, i.e. after its origin wait
        # observed this rank's return put of launch i. One lane per remote node issues the
        # generation signal; the tail's _rail_barrier only waits.
        iket.range_push("relay.rail_arrive")
        if lane < Int32(slots):
            self._issue_rail_arrival(lane)
        cute.arch.sync_warp()
        iket.range_pop()
        iket.range_push("relay.origin_wait")
        if lane == Int32(0):
            # Origin wait: every node this rank sent messages to returns exactly that many partials (I5, I8).
            send_count = workspace.tensor(router.send_count_region)
            for remote_slot in cutlass.range_constexpr(slots):
                expected = send_count[Int32(remote_slot), Int32(router.node_experts)]
                signal_id = Int32(router.combine_arrival_signal_base + remote_slot)
                shadow = gin.signal_shadow_pointer(signal=signal_id)
                old_target = cute.arch.atomic_add(shadow, Uint64(expected), sem="relaxed", scope="gpu")
                gin.wait_signal(
                    coop, signal=signal_id, least=Uint64(old_target) + Uint64(expected), bits=64, ord=nccl_cute.MemoryOrder.ACQUIRE
                )
            router.stamp(workspace, 13)
        cute.arch.sync_warp()
        iket.range_pop()

    @cute.jit
    def _issue_rail_arrival(self, dst_slot: Int32) -> None:
        """One lane: this launch's rail-barrier generation signal to the remote node of ``dst_slot``.

        Same two-slot phase derivation as ``_rail_barrier``; the four-state counter is advanced by
        the tail's wait alone, so arrive and wait derive the same slot and delta. The GIN handle is
        rebuilt from the entry-materialized pointer (nested JIT boundary); the lanes issue
        concurrently, hence the GPU-shared handle like the return puts.
        """
        from nccl.core.device import cute as nccl_cute
        from nccl.core.device.cute.gin import Gin

        router = self._router
        gin = Gin(ptr=self._gin_gpu_ptr)
        coop = nccl_cute.thread()
        counter = self._device_workspace.ptr(router.rail_barrier_counter_region)
        status = cute.arch.load(counter, Int32, sem="relaxed", scope="gpu") & Int32(3)
        signal_phase = status & Int32(1)
        signal_delta = Int64(1)
        if (status >> Int32(1)) != Int32(0):
            signal_delta = Int64(-1)
        signal_id = Int32(router.barrier_signal_base) + signal_phase
        peer_node, _ = router._remote_slot_geometry_for(dst_slot, self._local_rank)
        gin.signal(self._team_rail_value, peer_node, True, signal_id, GIN_SIGNAL_OP_ADD, signal_delta, coop)

    @cute.jit
    def _stamp_records_and_fc2_completion(self, lane: Int32) -> None:
        """Diagnostic stamp (phase_stamps only), SM0 relay warp 1: every local expert's relay records
        complete (stamp 16; a few us after the warp's own scan, so its gather items barely move)."""
        if cutlass.const_expr(self._router.phase_stamps):
            router = self._router
            workspace = self._device_workspace
            for local_expert in cutlass.range_constexpr(self.experts_per_rank):
                router.wait_for_relay_records(workspace, Int32(local_expert))
            if lane == Int32(0):
                router.stamp(workspace, 16)
            cute.arch.sync_warp()

    @cute.jit
    def _stamp_local_fc2_done(self) -> None:
        """Diagnostic stamp (phase_stamps only): every local expert's FC2 tiles published (stamp 17).

        Taken by SM0 epilogue warp 0 lane 0 when it runs out of tiles, where the poll distorts no
        gather work of other warps."""
        if cutlass.const_expr(self._router.phase_stamps):
            router = self._router
            workspace = self._device_workspace
            sizes = router.final_sizes_tensor(workspace)
            fc2_done = workspace.ptr(self.fc2_done_region)
            for local_expert in cutlass.range_constexpr(self.experts_per_rank):
                size = sizes[local_expert]
                tiles = (size + Int32(self.tokens_per_fc1_ready_slot - 1)) // Int32(self.tokens_per_fc1_ready_slot)
                target = tiles * Int32(self.fc2_done_signals_per_token_tile)
                while cute.arch.load(fc2_done + local_expert, Int32, sem="acquire", scope="sys") < target:
                    nanosleep(200)
            router.stamp(workspace, 17)

    @cute.jit
    def _wait_layout_published(self, lane: Int32) -> None:
        """Warp-level wait until every node peer published its final layout (system fence + arrival)."""
        if lane == Int32(0):
            published = self._device_workspace.ptr(self._router.layout_published_region)
            while cute.arch.load(published, Int32, sem="acquire", scope="sys") != Int32(self.ranks_per_node):
                nanosleep(200)
        cute.arch.sync_warp()

    @cute.jit
    def _relay_header_address(self, src_slot: Int32, message: Int32) -> Int64:
        """Address of the landed header (route words, then weights) of ``message`` of source slot ``src_slot``."""
        router = self._router
        return (
            self._device_workspace.ptr(router.recv_splits_region).toint()
            + Int64(src_slot) * Int64(router.splits_row_pitch_bytes)
            + Int64(router.splits_row_bytes)
            + Int64(message) * Int64(router.relay_header_bytes)
        )

    @cute.jit
    def _scan_relay_routes(self, relay_warp: Int32, lane: Int32) -> None:
        """Every relay warp of every CTA (but SM0 warp 0): push the relay records of this rank's landed messages to their owners.

        Thread ``g`` of the grid-wide scanner space handles messages ``g, g + G, ...`` of every
        source slot landed at THIS rank (the pull-style scan of every node message by every rank
        took up to 340 us at N4R8 under IKET). The route words and weights are read from the header
        plane that landed with the Step 3 row, so the scan overlaps the payload flight. A route word naming an expert of node-local
        owner ``o`` becomes one record at the next free row of the owner's (expert, slot, this rank)
        bucket: the owner's published ``relay_base`` plus this rank's fill counter of the (owner,
        expert, slot) bucket (rows of a bucket are filled in arrival order, which does not matter for
        correctness: every row is independent and returns to its own source). The record (and score)
        is stored into the owner's final layout over NVLink; then, behind this rank's
        ``payload_landed`` (the owner reads the message bytes from this rank's receive range behind
        the counter alone) and a system fence, the owner's ``relay_written[expert]`` receives one
        release increment per record (I9 chain:
        relay_written -> pull -> tile -> fc2_done -> the owner's FC2 epilogue pushes the row into this
        rank's landing cell (slot, message, k)).
        """
        router = self._router
        workspace = self._device_workspace
        mapper = self._token_comm_args.peer_rank_ptr_mapper
        experts = self.experts_per_rank
        slots = router.remote_node_count
        ranks_per_node = self.ranks_per_node
        node_base = (self._local_rank // Int32(ranks_per_node)) * Int32(ranks_per_node)
        in_node = self._local_rank - node_base
        relay_threads = self.relay_warp_count * 32
        # Scanner space = every relay thread of the grid except SM0 relay warp 0, which waits for the
        # payload credits instead (relay_warp_body); it must not hold back any message's records.
        global_thread = self._linear_cta_idx * Int32(relay_threads) + relay_warp * Int32(32) + lane - Int32(32)
        global_threads = Int32(self.promised_launchable_sm_count * relay_threads - 32)
        recv_splits = workspace.tensor(router.recv_splits_region)
        relay_base_address = workspace.ptr(router.layout_relay_base_region).toint()
        push_fill = workspace.tensor(router.layout_push_fill_region)
        records_address = workspace.ptr(router.final_records_region).toint()
        written_address = workspace.ptr(router.relay_written_region).toint()
        # One flattened (slot, message) index space: at T=256 the 525 messages of a rank at N4R8 fit one
        # per thread (the per-slot loop gave the same threads one message per slot, serially: SM0's
        # scan took 43 us in the N4R8 phase stamps against 17 us at N2R4).
        slot_counts = []
        total_messages = Int32(0)
        for slot_index in cutlass.range_constexpr(slots):
            slot_count = recv_splits[Int32(slot_index), Int32(router.node_experts)]
            slot_counts.append(slot_count)
            total_messages = total_messages + slot_count
        work = global_thread
        if global_thread < Int32(0):
            work = total_messages
        while work < total_messages:
            src_slot = Int32(0)
            message = work
            for slot_index in cutlass.range_constexpr(slots):
                if src_slot == Int32(slot_index):
                    if message >= slot_counts[slot_index]:
                        message = message - slot_counts[slot_index]
                        src_slot = src_slot + Int32(1)
            header_address = self._relay_header_address(src_slot, message)
            # Records this thread pushes for the message: released per owner expert after the fence.
            wrote_record = []
            wrote_owner_offset = []
            wrote_expert = []
            for topk_slot in cutlass.range_constexpr(self.topk):
                wrote_record.append(Boolean(False))
                wrote_owner_offset.append(Int64(0))
                wrote_expert.append(Int32(0))
            for topk_slot in cutlass.range_constexpr(self.topk):
                route_word = cute.arch.load(
                    cute.make_ptr(Int32, header_address + Int64(topk_slot * 4), AddressSpace.gmem, assumed_align=4),
                    Int32,
                    sem="relaxed",
                    scope="gpu",
                )
                if route_word != Int32(router.unrouted_route_word):
                    node_expert = route_word & Int32(0xFFFF)
                    owner_in_node = node_expert // Int32(experts)
                    local_expert = node_expert % Int32(experts)
                    owner_offset = mapper.map(Int64(0), node_base + owner_in_node, Int64(0))
                    fill = cute.arch.atomic_add(
                        push_fill.iterator + (owner_in_node * Int32(experts) + local_expert) * Int32(slots) + src_slot,
                        Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    # The owner's relay_base[(expert, slot, this rank)] from its published layout.
                    base = cute.arch.load(
                        cute.make_ptr(
                            Int32,
                            relay_base_address
                            + owner_offset
                            + Int64((local_expert * Int32(slots) + src_slot) * Int32(ranks_per_node) + in_node) * Int64(4),
                            AddressSpace.gmem,
                            assumed_align=4,
                        ),
                        Int32,
                        sem="relaxed",
                        scope="sys",
                    )
                    row = base + fill
                    stg_b64(
                        records_address + owner_offset + Int64(row) * Int64(TokenSrcMetadata.nbytes),
                        TokenSrcMetadata(
                            src_rank=self._local_rank,
                            src_token=message,
                            src_topk=Int32(router.relay_record_topk_marker + topk_slot)
                            + (src_slot << Int32(router.relay_record_slot_shift)),
                        ).pack(wide_topk=True),
                    )
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        weight = cute.arch.load(
                            cute.make_ptr(
                                cutlass.Float32,
                                header_address + Int64(router.relay_header_weights_offset + topk_slot * 4),
                                AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cutlass.Float32,
                            sem="relaxed",
                            scope="gpu",
                        )
                        stg_f32(
                            workspace.ptr(router.final_scores_region).toint() + owner_offset + Int64(row) * Int64(4), weight
                        )
                    wrote_record[topk_slot] = Boolean(True)
                    wrote_owner_offset[topk_slot] = owner_offset
                    wrote_expert[topk_slot] = local_expert
            # Fence-based release pattern (PTX: fence, then strong relaxed writes by the same thread):
            # one system fence per writing thread orders its record and score stores into the owners
            # and its local reverse-map stores; the relaxed increments of the owners' per-expert
            # counters are what the owners' acquire loads synchronize with (I9 chain:
            # relay_written -> pull -> tile -> fc2_done).
            wrote_any = Boolean(False)
            for topk_slot in cutlass.range_constexpr(self.topk):
                wrote_any = wrote_any | wrote_record[topk_slot]
            if wrote_any:
                # Publisher-side join: the owner's token_in reads the message bytes from THIS rank's
                # receive range behind relay_written alone, so the increments wait for this rank's
                # payload_landed (SM0's gpu-scope release after the Stage 1 credits). The record
                # stores above already overlapped the payload flight; only the publish is held to
                # max(scan, landing), and the owner polls one counter per expert.
                landed = workspace.ptr(router.payload_landed_region)
                while cute.arch.load(landed, Int32, sem="acquire", scope="gpu") != Int32(1):
                    nanosleep(200)
                cute.arch.fence_acq_rel_sys()
                for topk_slot in cutlass.range_constexpr(self.topk):
                    if wrote_record[topk_slot]:
                        red_add_relaxed_sys_s32(
                            written_address
                            + wrote_owner_offset[topk_slot]
                            + Int64(wrote_expert[topk_slot]) * Int64(router.relay_written_stride * 4),
                            Int32(1),
                        )
            work = work + global_threads


__all__ = ["TokenCommInternodeGin"]
