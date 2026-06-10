# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE fused dispatch + fc1 + fc2 + combine kernel.

The base class owns the local fc1/fc2 GEMM pipeline.  This subclass owns the
token-communication hooks, workspace partitioning, and the MegaMoE argument
bundle.  ``static_expert_shape`` is required because dispatch storage and pool
sizes are codegen-time quantities.

Shared / local workspace split:

  SHARED  : src_token_topk_idx, expert_recv_count, expert_recv_count_sum,
            nvlink_barrier_signal
  LOCAL   : expert_send_count, grid_sync_counter, l1_token_buffer,
            l1_sf_buffer, l1_topk_weights_buffer, l1_arrival_count,
            token_src_metadata, fc1_output, fc1_output_sf,
            fc1_done_counter, (optionally) load_balance_counter

User tensors are not in the opaque workspaces. ``activation``,
``activation_sf``, ``topk_weights``, and ``combine_output`` must be reachable
through the symmetric-heap peer mapper; ``topk_idx`` and weights are local.

Dispatch/pool alignment constraints are unified at construction time:
``token_padding_block`` (base) and ``block_m`` (dispatch) become the
same constant, similarly for ``sf_padding_block`` / ``sf_block_m``;
C3 reduces to a divisibility check that ``cluster_tile_tokens`` is a
multiple of ``token_padding_block``.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here.
# PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``'s
# element-type introspection (it reads ``__annotations__`` and demands the
# values be live ``cute.struct.MemRange[...] / struct / array / base_dsl
# scalar`` objects, not their string forms).  The lean fc1+fc2 base
# (``kernel_fc12.py``) and the dispatch standalone (``src/dispatch_kernel.py``)
# both already follow this convention.  Self-references (the single
# ``"TokenCommArgs"`` forward ref on ``__new_from_mlir_values__``) stay
# quoted explicitly.

import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from .kernel_fc12 import Sm100SwapABSwigluFp4Fc12Kernel
from .token_comm import TokenCommArgs as ExtractedTokenCommArgs
from .token_comm import TokenInPullTokenBackPush, TokenSrcMetadata

# =============================================================================
# Module-level constants.
# =============================================================================

# NamedBarrier IDs.  Base reserves 1-7; this subclass uses 8 and 9.
_KernelTailNamedBarrierId = 8  # 12-warp rendezvous (384 threads)
_DispatchToSchedNamedBarrierId = 9  # 4 dispatch + 1 sched (160 threads)

# Dispatch warp count.
_DispatchWarpCount = 4

# Per-pool-slot provenance record consumed by combine STG redirect (S3) and
# token-back; one i64 = {src_rank, src_token, src_topk} (see TokenSrcMetadata).
_TokenMetadataBytes = TokenSrcMetadata.nbytes

# NVLink signal slots used by the DeepGEMM-style phase/sign barrier.
# A separate local counter selects phase/sign; the signal slots are not reset
# by tail cleanup.
_NvlinkSlotCount = 2

# Grid-sync counter slot count.  ``software_grid_sync`` phase-flips bit 31
# so a single slot suffices; 2 slots keeps the layout 8-byte aligned.
_GridSyncSlotCount = 2

# =============================================================================
# Region spec + layout helpers
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _RegionSpec:
    """One region in either the local or shared workspace.

    Byte size = ``ceil(numel * cute_dtype.width / 8)``.  ``align`` is
    the region's start-byte alignment (TMA store / load destinations
    want 128 B; counters / metadata want 16 B).
    """

    name: str
    cute_dtype: Any
    shape: Tuple[int, ...]
    align: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def stride_row_major(self) -> Tuple[int, ...]:
        """Row-major stride matching ``shape`` (rightmost dim contiguous)."""
        if len(self.shape) == 0:
            return ()
        out: List[int] = [1]
        for d in reversed(self.shape[1:]):
            out.append(out[-1] * d)
        out.reverse()
        return tuple(out)

    @property
    def nbytes(self) -> int:
        bits = self.numel * int(self.cute_dtype.width)
        return (bits + 7) // 8


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _layout_regions(regions: List[_RegionSpec], ) -> Tuple[Dict[str, int], int]:
    """Place ``regions`` sequentially honouring each region's ``align``.
    Returns ``(name -> byte_offset)`` and the total byte count (rounded
    up to 16 B for downstream safety).

    Drives both ``get_workspace_sizes()`` (total only) and the
    ``__call__`` partition (offsets) -- keeping the host allocation
    and the device view construction in sync without any explicit
    handshake.
    """
    offsets: Dict[str, int] = {}
    cursor = 0
    for r in regions:
        cursor = _round_up(cursor, r.align)
        offsets[r.name] = cursor
        cursor += r.nbytes
    total = _round_up(cursor, 16)
    return offsets, total


# =============================================================================
# Sm100MegaMoEKernel
# =============================================================================


class Sm100MegaMoEKernel(Sm100SwapABSwigluFp4Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 + combine kernel."""

    def __init__(
        self,
        # Base-class kwargs (forwarded 1:1 to ``super().__init__``).
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: str = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 16,
        scenario: str = "2Dx3D",
        # MegaMoE-specific independent constants.
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        fc2_output_dtype: Type[cutlass.Numeric],
        non_ubulk_fc2_store: bool = True,
        in_kernel_fc2_reduce: bool = False,
        token_back_mode: Literal["epi_warps", "standalone_warps",
                                 "reuse_dispatch_warps"] = "epi_warps",
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        flag_batch: int = 1,
    ) -> None:
        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm100MegaMoEKernel currently requires "
                "static_expert_shape != None (dynamic-shape MegaMoE is "
                "not wired).")
        # Keep the explicit ``hidden`` kwarg in lockstep with static shape;
        # dispatch SMEM sizing reads it before tensor layouts are rewritten.
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal "
                f"static_expert_shape[2] ({static_expert_shape[2]}).")

        # token_back_mode selects where the cross-rank fc2 push-back runs:
        #   epi_warps            -> epilogue warps STG directly to the peer
        #   standalone_warps     -> dedicated warp group 12-15, concurrent
        #                           with dispatch_pull
        #   reuse_dispatch_warps -> dispatch warps 8-11 push after dispatch_pull
        # The two non-epi modes both stage fc2 to a local workspace first, i.e.
        # token_back_by_dispatch=True; epi_warps keeps the epilogue STG redirect.
        if token_back_mode not in ("epi_warps", "standalone_warps",
                                   "reuse_dispatch_warps"):
            raise ValueError(
                f"token_back_mode must be 'epi_warps', 'standalone_warps', "
                f"or 'reuse_dispatch_warps'; got {token_back_mode!r}.")
        token_back_by_dispatch = token_back_mode != "epi_warps"

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            acc_dtype=acc_dtype,
            sf_vec_size=sf_vec_size,
            scenario=scenario,
            fc2_output_dtype=fc2_output_dtype,
            non_ubulk_fc2_store=non_ubulk_fc2_store,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
            apply_topk_in_fc1=apply_topk_in_fc1,
            gate_up_clamp=gate_up_clamp,
            epi_flag_batch=epi_flag_batch,
        )

        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        # Standalone token-back: a dedicated 4-warp group (12-15) doing
        # token_back_by_push concurrently with dispatch_pull, selected by the
        # user-facing token_back_mode knob ("standalone_warps").
        self.token_back_mode = token_back_mode
        self.token_back_standalone = token_back_mode == "standalone_warps"
        self.token_back_warp_id = (12, 13, 14,
                                   15) if self.token_back_standalone else None
        num_token_back_warps = (len(self.token_back_warp_id)
                                if self.token_back_standalone else 0)
        self.threads_per_cta = 32 * (
            len(self.epilogue_warp_id) + 1  # mma
            + 1  # tma_a
            + 1  # tma_b
            + 1  # sched
            + len(self.dispatch_warp_id) + num_token_back_warps)

        # Independent MegaMoE-specific constants.
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden

        # static_expert_shape = (num_experts_per_rank, intermediate_gateup, hidden).
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2

        # NVFP4: 4 bits/elem -> 2 elements per byte.
        self.hidden_bytes = self.hidden // 2
        # Dispatch pulls SF in uint32 units; host activation_sf rows must pad
        # to this ceiling with zero-filled bytes.
        sf_atom_k_elements = 4 * self.sf_vec_size
        self.sf_uint32_per_token = ((self.hidden + sf_atom_k_elements - 1) //
                                    sf_atom_k_elements)
        # Cross-rank totals: per-rank count * world_size.
        self.num_total_experts = world_size * self.num_experts_per_rank

        # Per-task-tile release-counter granularity used by dispatch_pull.
        self.cluster_tile_tokens = mma_tiler_mnk[1] * cluster_shape_mnk[1]

        # One dispatch task tile must map to contiguous pool blocks.
        if self.cluster_tile_tokens % self.token_padding_block != 0:
            raise ValueError(
                f"C3 violated: cluster_tile_tokens "
                f"({self.cluster_tile_tokens}) must be a multiple of "
                f"token_padding_block ({self.token_padding_block}); "
                f"otherwise pool row offsets and release counter slots "
                f"will not align.")

        # Cache region sizing inputs used by workspace layout and __call__.
        (
            self.pool_token_capacity,
            self.pool_sf_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()
        # Cohabit warps in this CTA outside the dispatch group:
        # epilogue + mma + tma_a + tma_b + sched.
        num_other_warps = (len(self.epilogue_warp_id) + 1 + 1 + 1 + 1)
        # fc2 epi publishes once per CTA per work tile; edge hidden tiles
        # still publish (no in-bound gating), so ceil_div on the hidden axis.
        cluster_fc2_tile_hidden = (self.mma_tiler[0] *
                                   self.cluster_shape_mn[0] //
                                   (2 if self.use_2cta_instrs else 1))
        fc2_publishes_per_token_cluster_tile = (
            (self.hidden + cluster_fc2_tile_hidden - 1) //
            cluster_fc2_tile_hidden) * self.cluster_shape_mn[0]

        # Homomorphic to the fc1+fc2 scheduler: atomic_counter token-back only
        # nets a win with enough tokens, the same condition that selects the
        # atomic_counter fc1+fc2 scheduler.  Static when token-back is off.
        self.token_back_schedule_mode = (
            self.load_balance_mode if self.token_back_by_dispatch else "static")

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            local_rank=self.local_rank,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=cutlass.Float4E2M1FN,
            fc2_output_dtype=(self.fc2_output_dtype
                              if self.token_back_by_dispatch else None),
            fc2_publishes_per_token_cluster_tile=
            fc2_publishes_per_token_cluster_tile,
            token_back_reduce_topk=(self.token_back_by_dispatch
                                    and self.in_kernel_fc2_reduce),
            token_back_standalone=self.token_back_standalone,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            flag_batch=flag_batch,
            token_back_schedule_mode=self.token_back_schedule_mode,
        )

        # Region layout (same call drives both get_workspace_sizes() and
        # the __call__ partition).
        self._local_region_specs = self._build_local_region_specs()
        self._shared_region_specs = self._build_shared_region_specs()
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs)
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs)
        self._local_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r
            for r in self._local_region_specs
        }
        self._shared_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r
            for r in self._shared_region_specs
        }

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM bytes for dispatch pull mbarriers, expert scratch, and token buffer."""
        pull_mbar_bytes = _DispatchWarpCount * 8
        expert_count_bytes = self.num_total_experts * 4
        pull_buffer_bytes = _DispatchWarpCount * self.hidden_bytes
        total = (_round_up(pull_mbar_bytes, 16) +
                 _round_up(expert_count_bytes, 16) +
                 _round_up(pull_buffer_bytes, 128))
        if self.token_back_standalone:
            total += (_round_up(_DispatchWarpCount * 8, 16) + _round_up(
                _DispatchWarpCount * self.token_comm.tb_chunk_bytes, 128))
        return total

    def _smem_misc_budget_bytes(self) -> int:
        """Base misc reservation plus dispatch-warp SMEM."""
        return super()._smem_misc_budget_bytes() + self._dispatch_smem_bytes()

    # =========================================================================
    # Pool sizing (first-principles)
    # =========================================================================

    def _pool_shapes(self) -> Tuple[int, int, int]:
        """Worst-case pool sizes.

        ``pool_token_capacity``: every received token from any peer can
        replicate to ``min(num_topk, num_experts_per_rank)`` local
        experts; worst case is ``world_size * max_tokens_per_rank``
        tokens received, each replicated up to that bound.  Each of
        the ``num_experts_per_rank`` experts wastes up to
        ``token_padding_block - 1`` rows at its tail; round the whole
        sum up to the pool-layout granularity ``token_padding_block``.

        ``pool_sf_capacity``: same number of expert blocks as the data
        pool, each padded to ``sf_padding_block`` rows (UTCCP 4x32
        swizzle that the SF TMA load expects).

        ``pool_task_tile_capacity``: ``ceil(pool_token_capacity,
        cluster_tile_tokens)``.  C3 makes ``cluster_tile_tokens`` a
        multiple of ``token_padding_block`` so this stays exact.
        """
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        sf_padding_block = self.sf_padding_block
        cluster_tile_tokens = self.cluster_tile_tokens

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        raw = (max_recv * max_per_token + num_experts_per_rank *
               (token_padding_block - 1))
        pool_token_capacity = _round_up(raw, token_padding_block)
        pool_sf_capacity = ((pool_token_capacity // token_padding_block) *
                            sf_padding_block)
        # Upper bound for sum_e ceil(valid_e, cluster_tile_tokens).  The
        # per-expert slack covers each expert's final partial task tile.
        pool_task_tile_capacity = (
            (pool_token_capacity + cluster_tile_tokens - 1) //
            cluster_tile_tokens + num_experts_per_rank)
        return (
            pool_token_capacity,
            pool_sf_capacity,
            pool_task_tile_capacity,
        )

    # =========================================================================
    # Region tables
    # =========================================================================

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        """Local-only regions (no peer access via ``peer_rank_ptr_mapper.map`` in
        ``src/dispatch_kernel.py``).
        """
        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        sf_uint32_per_token = self.sf_uint32_per_token
        intermediate_downproj = self.intermediate_downproj
        mma_tiler_n = self.mma_tiler_mnk[1]
        sf_vec_size = self.sf_vec_size
        sf_padding_block = self.sf_padding_block

        # fc1_output_sf / fc1_done_counter sizing mirrors base
        # ``get_workspace_size_in_bytes`` (kernel_fc12.py ~lines 525-543).
        sf_total_rows_upper = (pool_token_capacity +
                               num_experts_per_rank * sf_padding_block)
        sf_block_cols = ((((intermediate_downproj // sf_vec_size) + 3) // 4) *
                         4)
        fc1_done_slots = (
            (pool_token_capacity + mma_tiler_n - 1) // mma_tiler_n +
            num_experts_per_rank)

        specs: List[_RegionSpec] = [
            # L1 input pool (dispatch_pull writes -> fc1 reads).  Stored
            # as Uint8 bytes; the NVFP4 view at the same offset is
            # built inside ``__call__``.
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            # Stored as Int32 (dispatch_pull's 32 b read/write); the FP8
            # view for activation_sf is built at the same offset.
            # 1D Int32 atom-flat buffer.  Total Int32 count = pool_sf_capacity
            # (M-axis token positions) * sf_uint32_per_token (K-atom count),
            # laid out atom-by-atom per cute SFA layout.  dispatch writes
            # individual Int32 slots via the linear offset returned by
            # ``src/sf_swizzle.py:sf_atom_int32_offset``; the mma side
            # re-views this same byte buffer through ``tile_atom_to_shape_SF``
            # which reads back the atom-swizzled bytes.
            _RegionSpec(
                "l1_sf_buffer",
                cutlass.Int32,
                (pool_sf_capacity * sf_uint32_per_token, ),
                16,
            ),
            _RegionSpec(
                "l1_topk_weights_buffer",
                cutlass.Float32,
                (pool_token_capacity, ),
                16,
            ),
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity, ),
                16,
            ),
            _RegionSpec(
                "token_src_metadata",
                cutlass.Uint8,
                (pool_token_capacity, _TokenMetadataBytes),
                16,
            ),
            _RegionSpec(
                "expert_send_count",
                cutlass.Int64,
                (num_total_experts, ),
                16,
            ),
            _RegionSpec(
                "grid_sync_counter",
                cutlass.Int32,
                (_GridSyncSlotCount, ),
                16,
            ),
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1, ),
                16,
            ),
            _RegionSpec(
                "fc1_output",
                cutlass.Float4E2M1FN,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
            _RegionSpec(
                "fc1_output_sf",
                cutlass.Float8E4M3FN,
                (sf_total_rows_upper, sf_block_cols),
                128,
            ),
            _RegionSpec(
                "fc1_done_counter",
                cutlass.Int32,
                (fc1_done_slots, ),
                16,
            ),
        ]
        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    self.fc2_output_dtype,
                    (pool_token_capacity, 1, self.hidden),
                    128,
                ))
            specs.append(
                _RegionSpec(
                    "fc2_done_counter",
                    cutlass.Int32,
                    (num_experts_per_rank, ),
                    16,
                ))
            if self.token_back_schedule_mode == "atomic_counter":
                specs.append(
                    _RegionSpec(
                        "token_back_schedule_counter",
                        cutlass.Int32,
                        (1, ),
                        16,
                    ))

        if self.load_balance_mode == "atomic_counter":
            specs.append(
                _RegionSpec(
                    "load_balance_counter",
                    cutlass.Int32,
                    (1, ),
                    16,
                ))

        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        """Shared (peer-mapped) regions -- every entry is reached from
        some ``peer_rank_ptr_mapper.map(local_ptr, peer_rank, byte_off)``
        call site inside ``src/dispatch_kernel.py``:

          * ``src_token_topk_idx`` -- ``_dispatch_prep`` round 3
          * ``expert_recv_count`` / ``expert_recv_count_sum``
            -- ``_dispatch_barrier`` step 2 (b64 store + sys-atomic-add)
          * ``nvlink_barrier_signal``
            -- ``_nvlink_barrier_3stage`` stage B (two reusable phase slots)
        """
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        # ``MAX_SLOT`` in ``_dispatch_prep`` round 3: every (token, topk)
        # edge any peer might publish for this rank's local experts.
        max_slot = max_tokens_per_rank * num_topk

        return [
            _RegionSpec(
                "src_token_topk_idx",
                cutlass.Int32,
                (num_experts_per_rank, world_size, max_slot),
                16,
            ),
            _RegionSpec(
                "expert_recv_count",
                cutlass.Int64,
                (world_size, num_experts_per_rank),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_sum",
                cutlass.Int64,
                (num_experts_per_rank, ),
                16,
            ),
            _RegionSpec(
                "nvlink_barrier_signal",
                cutlass.Int32,
                (_NvlinkSlotCount, ),
                16,
            ),
        ]

    # =========================================================================
    # Public: workspace size query
    # =========================================================================

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)`` -- the byte
        budgets for the two opaque workspaces the host must allocate.
        Both totals are invariant across launches; per-launch ``T``
        may be <= ``max_tokens_per_rank``.
        """
        return self._local_total, self._shared_total

    # =========================================================================
    # Workspace partition helpers
    # =========================================================================

    @staticmethod
    def _make_typed_view(
        byte_workspace: cute.Tensor,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        assumed_align: int,
    ) -> cute.Tensor:
        """Build a typed cute view at ``byte_offset`` of the opaque workspace."""
        # Large MegaMoE problems can place later workspace regions above the
        # 2 GiB / 4 GiB boundary.  Keep the base adjustment in 64-bit pointer
        # arithmetic so region starts such as fc1_output_sf / counters do not
        # wrap before the typed view is built.
        byte_ptr = byte_workspace.iterator + Int64(byte_offset)
        typed_iter = cute.make_ptr(
            cute_dtype,
            byte_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        return cute.make_tensor(typed_iter,
                                cute.make_layout(shape, stride=stride))

    def _view_local(
        self,
        local_workspace: cute.Tensor,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        """Partition a region of the local workspace.  With no overrides,
        uses the region's declared dtype + shape + row-major stride;
        overrides let dual-view callers build alternate-dtype views at
        the same byte offset.
        """
        return self._partition_region(
            local_workspace,
            self._local_offsets,
            self._local_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_shared(
        self,
        shared_workspace: cute.Tensor,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._partition_region(
            shared_workspace,
            self._shared_offsets,
            self._shared_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _partition_region(
        self,
        byte_workspace: cute.Tensor,
        offsets: Dict[str, int],
        spec: _RegionSpec,
        *,
        cute_dtype: Optional[Any],
        shape: Optional[Tuple[int, ...]],
        stride: Optional[Tuple[int, ...]],
    ) -> cute.Tensor:
        dt = cute_dtype if cute_dtype is not None else spec.cute_dtype
        sh = shape if shape is not None else spec.shape
        st = stride
        if st is None:
            if cute_dtype is None and shape is None:
                st = spec.stride_row_major
            else:
                # Derive row-major from the (possibly overridden) shape.
                out: List[int] = [1]
                for d in reversed(list(sh)[1:]):
                    out.append(out[-1] * d)
                out.reverse()
                st = tuple(out)
        return self._make_typed_view(
            byte_workspace,
            offsets[spec.name],
            dt,
            sh,
            st,
            spec.align,
        )

    # =========================================================================
    # __call__
    # =========================================================================

    @cute.jit
    def __call__(
        self,
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,  # (T, hidden) NVFP4
        activation_sf: cute.
        Tensor,  # (T, round_up(hidden, sf_atom_block_k)) FP8
        topk_idx: cute.Tensor,  # (T, num_topk) Int64
        topk_weights: cute.Tensor,  # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_norm_const: cute.Tensor,
        # Combine destination (peer write target under S3; local fc2
        # output region under S2 -- same memory, same caller).
        combine_output: cute.Tensor,  # (T, num_topk, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Tensor,  # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,  # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``
        # before entering the device kernel.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch the MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``activation_sf`` / ``topk_weights`` MUST
            point into memory reachable via ``peer_rank_ptr_mapper.map(...)``
            (typically NVSHMEM symmetric heap).  Single-rank degenerate
            runs (``peer_rank_ptr_mapper.offsets[local_rank] == 0`` by NVSHMEM
            convention) are allowed.
          * ``topk_idx`` is read on the local rank only; placement is
            unconstrained (cuda local or sym heap).
          * ``fc1_weight`` / ``fc1_weight_sf`` / ``fc2_weight`` /
            ``fc2_weight_sf`` are local-only.
          * ``combine_output`` is the per-rank S3 combine STG target;
            under S2 it acts as the rank's local BF16 fc2 output.
            Placement: sym heap (peer write target) or local in the
            single-rank degenerate case.

        Workspace zero-init contract: caller is currently expected to
        zero ``shared_workspace`` before launch (the dispatch
        primitives' counters / signals rely on a clean state).  This
        contract may be tightened later to have the kernel take
        ownership of the reset.
        """
        # ``max_active_clusters`` and ``cluster_size`` are both Python ints
        # at trace time, so the product folds to a Python int that flows
        # cleanly to every dispatch primitive's ``num_sms: Constexpr[int]``
        # slot.
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        hidden = self.hidden
        sf_per_token_fp8 = self.sf_uint32_per_token * 4  # 4 FP8 SFs per Int32

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + NVFP4
        # view (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(
            local_workspace,
            "l1_token_buffer",
        )
        l1_token_buffer_nvfp4 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            cutlass.Float4E2M1FN,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

        # L1 SF buffer: Int32 view (dispatch_pull's [j, t] 2D indexing) +
        # FP8 view (base.activation_sf re-views via tile_atom_to_shape_SF
        # off the iterator, so the stride here is informational only).
        l1_sf_buffer_i32 = self._view_local(
            local_workspace,
            "l1_sf_buffer",
        )
        l1_sf_buffer_fp8 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_sf_buffer"],
            cutlass.Float8E4M3FN,
            (pool_sf_capacity, sf_per_token_fp8),
            (sf_per_token_fp8, 1),
            self._local_region_by_name["l1_sf_buffer"].align,
        )

        l1_topk_weights_buffer = self._view_local(
            local_workspace,
            "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local(
            local_workspace,
            "l1_arrival_count",
        )
        # token_src_metadata storage = (pool_token_capacity, 12) Uint8;
        # dispatch_pull writes three Uint32 fields per pool token row via
        # byte-stepped pointer arithmetic on this Uint8 view (so its
        # element-width-1 ``+ pool_token_idx * 12`` matches a 12-byte row
        # stride).  The fc2 epilogue's metadata-LDG path wants a logical
        # ``(N, 3) Uint32`` view of the same bytes -- it does that recast
        # itself inside ``_run_fc2_task_tile`` to keep the dispatch-side
        # Uint8 ABI intact (dispatch_kernel.py is a standalone module
        # whose API the fused kernel does not mutate).
        token_src_metadata = self._view_local(
            local_workspace,
            "token_src_metadata",
        )
        expert_send_count = self._view_local(
            local_workspace,
            "expert_send_count",
        )
        grid_sync_counter = self._view_local(
            local_workspace,
            "grid_sync_counter",
        )
        nvlink_barrier_counter = self._view_local(
            local_workspace,
            "nvlink_barrier_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local(
            local_workspace,
            "fc1_done_counter",
        )

        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace,
                "load_balance_counter",
            )

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace,
            "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(
            shared_workspace,
            "expert_recv_count",
        )
        expert_recv_count_sum = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace,
            "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum``
        # buffer -- low32 bits hold per-expert total token count after
        # _dispatch_barrier; zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank, ),
            stride=(2, ),
        )

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace,
                "fc2_output_workspace",
            )
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                (pool_token_capacity * self.hidden *
                 (int(self.fc2_output_dtype.width) // 8), ),
                None,
                self._local_region_by_name["fc2_output_workspace"].align,
            )
            fc2_done_counter = self._view_local(
                local_workspace,
                "fc2_done_counter",
            )
            combine_output_u8 = cute.recast_tensor(
                combine_output,
                cutlass.Uint8,
            )
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_u8 = combine_output

        if cutlass.const_expr(
                self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace,
                "token_back_schedule_counter",
            ).iterator
        else:
            token_back_schedule_counter = None

        token_comm_args = ExtractedTokenCommArgs(
            input_token_buffer=activation,
            input_sf_buffer=activation_sf,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=l1_sf_buffer_i32,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_u8,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_done_counter=fc2_done_counter,
            token_back_schedule_counter=token_back_schedule_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=self.local_rank,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            sm_count=sm_count,
        )

        # C1 / C2 are tautological (token_padding_block == "block_m";
        # sf_padding_block == "sf_block_m") so the pool layout and the
        # sched cumulative-row offsets align by construction.
        #
        # ``combine_output`` is MoE-domain storage. Non-reduce modes use
        # ``(max_tokens_per_rank, num_topk, hidden)`` and host-reduce topk;
        # REDG modes use ``(max_tokens_per_rank, 1, hidden)`` and reduce in
        # kernel.  The epilogue return tile maps local pool rows back to the
        # source rank's token row through ``token_comm_args``.
        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_target = combine_output

        super().__call__(
            activation=l1_token_buffer_nvfp4,
            fc1_weight=fc1_weight,
            activation_sf=l1_sf_buffer_fp8,
            fc1_weight_sf=fc1_weight_sf,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_output=fc2_output_target,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            offs=None,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )

    # =========================================================================
    # TokenComm delegation surface consumed by the fc1/fc2 base kernel
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> type:
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self,
        token_comm_args,
        work_tile_info,
    ):
        self.token_comm.fc1_tma_b_predispatch_spin(
            token_comm_args,
            work_tile_info,
        )

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
        self.token_comm.dispatch_warp_body(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

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
        self.token_comm.token_back_warp_body(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

    @cute.jit
    def token_comm_hook_tail_reset_shared_counters(
        self,
        token_comm_args,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        self.token_comm.tail_reset_shared_counters(
            token_comm_args,
            cta_linear_id=cta_linear_id,
            local_warp_idx=local_warp_idx,
            lane_idx=lane_idx,
        )

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        self.token_comm.kernel_tail(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )
