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
from .token_comm import CombineFormat
from .token_comm import TokenCommArgs as ExtractedTokenCommArgs
from .token_comm import TokenInPullTokenBackPush, TokenSrcMetadata
from .topk_reduce import TopkReduce

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
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        fc2_output_dtype: Type[cutlass.Numeric],
        combine_format: CombineFormat = CombineFormat.parse("bf16"),
        non_ubulk_fc2_store: bool = True,
        in_kernel_fc2_reduce: bool = False,
        token_back_mode: Literal["epi_warps", "standalone_warps",
                                 "reuse_dispatch_warps"] = "epi_warps",
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        flag_batch: int = 1,
    ) -> None:
        # The combine wire format drives the fc2 epilogue encoder, token_comm
        # push, and the combine_quant/combine_sf workspace sizing. The dataflow
        # (workspace carve, views, arg threading, epilogue encode, topk_reduce
        # receiver) is wired for every format, quantized included -- see the
        # closed-loop note above the fused launch in __call__. The guards below
        # only reject combinations the kernel cannot express
        # (in_kernel_fc2_reduce is bf16-only; FP4 needs non_ubulk_fc2_store).
        self.combine_format = combine_format
        if in_kernel_fc2_reduce and combine_format.is_quantized:
            raise ValueError(
                f"in_kernel_fc2_reduce requires a non-quantized (bf16) combine "
                f"format; got {combine_format}.")
        if in_kernel_fc2_reduce and not apply_topk_in_fc1:
            raise ValueError(
                "in_kernel_fc2_reduce requires apply_topk_in_fc1=True; "
                "the REDG path can only atomic-add terms whose topk score "
                "was already absorbed before fc2.")
        if (combine_format.act_dtype is cutlass.Float4E2M1FN
                and not non_ubulk_fc2_store):
            raise ValueError(
                f"{combine_format} combine requires non_ubulk_fc2_store=True "
                "(the UBLK fc2 store path cannot scalar-dereference FP4).")
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
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.flag_batch = flag_batch  # stored so name() can encode it

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

        # Per-(token, topk) SF block padded to a 16 B multiple so the token-back
        # cp.async.bulk push moves one aligned block per shot; only the first
        # hidden//scale_block entries of each block are valid (the rest is the
        # alignment gap).  ``None`` for the bf16 baseline (no SF plane).
        self.sf_block_pad = (_round_up(
            self.hidden // self.combine_format.scale_block,
            16 // (int(self.combine_format.scale_dtype.width) // 8),
        ) if self.combine_format.is_quantized else None)

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

        # Token-back warps run when they push the DATA plane (dispatch modes) OR
        # the SF plane (any quantized combine, including the epi_warps data path
        # where the epilogue STGs the data straight to the peer but SF still
        # needs the staged token-contiguous push).
        self.token_back_enabled = (self.token_back_by_dispatch
                                   or self.combine_format.is_quantized)
        # Homomorphic to the fc1+fc2 scheduler: atomic_counter token-back only
        # nets a win with enough tokens, the same condition that selects the
        # atomic_counter fc1+fc2 scheduler.  Static when token-back is off.
        self.token_back_schedule_mode = (self.load_balance_mode if
                                         self.token_back_enabled else "static")

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=cutlass.Float4E2M1FN,
            combine_format=self.combine_format,
            token_back_by_dispatch=self.token_back_by_dispatch,
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
            is_swap_ab=True,
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

        # Counter-prefix byte extents: the accumulating counters are front-placed
        # (see _build_*_region_specs), so the leading bytes up to the first data
        # region cover exactly the per-launch zero set. ``tail_reset_counters``
        # bulk-zeros these each launch; only the first launch needs a caller-zeroed
        # workspace. 128B-aligned offsets => multiples of 4 (Int32 zeroing exact).
        local_leading = self._local_offsets[
            "l1_token_buffer"]  # first data region
        shared_leading = self._shared_offsets[
            "src_token_topk_idx"]  # first data region
        self.require_zero_workspace_leading_bytes: Tuple[int, int] = (
            local_leading,
            shared_leading,
        )
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4

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

        # === Accumulating-counter prefix ===========================================
        # Front-placed so the per-launch reset is a single bulk zero of
        # ``[0:local_leading]`` (the bytes up to the first data region). These hold
        # spin thresholds / write cursors / phase-flip counters that the kernel
        # accumulates across the launch and that MUST start at 0 each launch; the
        # kernel tail (``tail_reset_counters``) bulk-zeros this prefix, so only the
        # FIRST launch relies on a caller-zeroed workspace.
        specs: List[_RegionSpec] = [
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity, ),
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
                "fc1_done_counter",
                cutlass.Int32,
                (fc1_done_slots, ),
                16,
            ),
        ]
        if self.token_back_enabled:
            # Per-expert fc2 completion gate consumed by the token-back push
            # (DATA and/or SF).  Published by the fc2 epilogue for every enabled
            # token-back path, including the epi_warps SF-only push.
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

        # === Data buffers (overwritten each launch; NOT zeroed) ====================
        # ``l1_token_buffer`` MUST be the first data region: ``__init__`` derives
        # ``local_leading`` from its offset (= end of the counter prefix).
        specs += [
            # L1 input pool (dispatch_pull writes -> fc1 reads), Uint8 bytes; the
            # NVFP4 view at the same offset is built inside ``__call__``.
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            # Persisted across launches (deliberately OUT of the zero prefix): the
            # sense-reversing nvlink barrier rides this phase counter across launch
            # boundaries (non-ncu back-to-back), and ncu kernel replay restores it
            # via its local-memory snapshot. Only the FIRST launch relies on the
            # caller-zeroed workspace.
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1, ),
                16,
            ),
            # Int32 atom-flat SF buffer (dispatch_pull 32b read/write; FP8 view at
            # the same offset). Count = pool_sf_capacity * sf_uint32_per_token.
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
                "token_src_metadata",
                cutlass.Uint8,
                (pool_token_capacity, _TokenMetadataBytes),
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
        ]
        if self.token_back_by_dispatch:
            # Local fc2 DATA staging (token_back_by_dispatch modes only); the
            # wire-format dtype sizes the plane.
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    self.combine_format.act_dtype,
                    (pool_token_capacity, 1, self.hidden),
                    128,
                ))
        # The per-block SF plane is ALWAYS staged locally (then pushed
        # token-contiguously by the dispatch / standalone warps), independent of
        # whether the DATA path goes local or straight to a peer: writing SF
        # per-token to a peer would scatter one warp's 32 lanes across up to 32
        # ranks and explode the NVLink request count. So it is allocated for
        # every quantized format, not only the dispatch data path.
        if self.combine_format.is_quantized:
            # Flat padded capacity (pool_token * sf_block_pad scale entries); the
            # (pool_token, 1, hidden//scale_block) logical shape + 16 B-aligned
            # per-block stride is assembled where the view is built in __call__.
            specs.append(
                _RegionSpec(
                    "fc2_output_sf",
                    self.combine_format.scale_dtype,
                    (pool_token_capacity * self.sf_block_pad, ),
                    128,
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

        # Accumulating counters first (zero-prefix -> tail bulk-zeros
        # ``[0:shared_leading]``); then the data/signal regions that must persist
        # across launches (``src_token_topk_idx`` overwritten each launch;
        # ``nvlink_barrier_signal`` is phase-flip and must NOT be zeroed).
        # ``src_token_topk_idx`` MUST be the first non-counter region: ``__init__``
        # derives ``shared_leading`` from its offset (= end of the counter prefix).
        specs: List[_RegionSpec] = [
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
                "src_token_topk_idx",
                cutlass.Int32,
                (num_experts_per_rank, world_size, max_slot),
                16,
            ),
            _RegionSpec(
                "nvlink_barrier_signal",
                cutlass.Int32,
                (_NvlinkSlotCount, ),
                16,
            ),
        ]
        # separate-kernel-reduce only: the per-topk fc2 combine staging buffer is
        # internalized here instead of being a caller tensor, so the public output
        # is the 2D (T, hidden) reduce result.  It is the cross-rank combine STG
        # target, hence must live on the symmetric heap (= this shared workspace);
        # appended after the data regions so it stays out of the per-launch zero
        # prefix (only the first launch relies on the caller-zeroed workspace).
        if not self.in_kernel_fc2_reduce:
            # combine_quant: the cross-rank combine data plane, one cell per
            # (token, topk). dtype follows the wire format -- the bf16 baseline is
            # byte-identical to the old ``combine_partial``; fp4/e4m3 shrink it.
            specs.append(
                _RegionSpec(
                    "combine_quant",
                    self.combine_format.act_dtype,
                    (max_tokens_per_rank, num_topk, self.hidden),
                    128,
                ))
            # combine_sf: per-block scale plane; only quantized formats carry one.
            if self.combine_format.is_quantized:
                # Flat padded capacity (max_tokens * num_topk * sf_block_pad scale
                # entries); the (max_tokens, num_topk, hidden//scale_block) logical
                # shape + 16 B-aligned per-block stride is assembled where the view
                # is built in __call__.
                specs.append(
                    _RegionSpec(
                        "combine_sf",
                        self.combine_format.scale_dtype,
                        (max_tokens_per_rank * num_topk * self.sf_block_pad, ),
                        128,
                    ))
        return specs

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
        byte_base: cute.Pointer,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        assumed_align: int,
    ) -> cute.Tensor:
        """Build a typed cute view at ``byte_offset`` of the opaque workspace.

        The workspace is a raw ``cute.Pointer`` (uint8 gmem base), not a tensor:
        the kernel only ever needs the base address + its own byte-offset table, so
        a tensor's shape would be both ignored AND, for >2 GiB workspaces (the
        internalized combine staging), overflow cute's 32-bit memref shape field.
        """
        # Large MegaMoE problems place later regions above the 2 GiB / 4 GiB
        # boundary; keep the base adjustment in 64-bit pointer arithmetic so region
        # starts (fc1_output_sf / counters) do not wrap before the typed view.
        byte_ptr = byte_base + Int64(byte_offset)
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
        local_workspace: cute.Pointer,
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
        shared_workspace: cute.Pointer,
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
        byte_workspace: cute.Pointer,
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

    def name(self) -> str:
        """Full compiled-kernel cache key. Self-contained on purpose (no shared
        helper): mirrors the base fc12 fields and appends the MegaMoE-specific
        ones -- keep the shared part in sync with the base ``name()``.
        ``local_rank`` excluded (deployment detail); same dropped set as base."""
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn
        exp = "x".join(map(
            str,
            self.static_expert_shape)) if self.static_expert_shape else "dyn"
        epiflag = "x".join(map(
            str, self.epi_flag_batch)) if self.epi_flag_batch else "none"
        cta = "2_cta" if self.use_2cta_instrs else "1_cta"
        fc2store = "fc2store_stg" if self.non_ubulk_fc2_store else "fc2store_ublk"
        inkred = "inkernel_redg" if self.in_kernel_fc2_reduce else "no_inkernel_redg"
        apply_topk = "apply_topk_fc1_pre_quant" if self.apply_topk_in_fc1 else "apply_topk_after_fc2"
        token_back = {
            "epi_warps": "epiwarps",
            "standalone_warps": "standalone",
            "reuse_dispatch_warps": "reuse_dispatch",
        }.get(self.token_back_mode, self.token_back_mode)
        return (
            "megamoe_nvfp4"
            f"_mmatiler_{m}x{n}x{k}_cluster_{cm}x{cn}_{cta}_sched_{self.load_balance_mode}"
            f"_expert_shape_{exp}_grouphint_{self.group_hint}"
            f"_padding_{self.token_padding_block}x{self.sf_padding_block}"
            f"_{fc2store}_{inkred}_token_back_by_{token_back}_{apply_topk}"
            f"_fc2out{self.fc2_output_dtype.__name__}_combine{self.combine_format}_sfvec{self.sf_vec_size}"
            f"_acc{self.acc_dtype.__name__}_clamp{self.gate_up_clamp}_epiflag{epiflag}"
            # MegaMoE-specific constexpr:
            f"_ep_{self.world_size}_topk_{self.num_topk}_maxtoken_{self.max_tokens_per_rank}"
            f"_flagbatch_{self.flag_batch}")

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
        # Final combined output the caller consumes: 2D (T, hidden).  Under
        # in_kernel_reduce it is the cross-rank REDG target (symmetric heap);
        # under separate_kernel_reduce it is the local tail-reduce destination
        # while the per-topk staging lives in the internal ``combine_quant``.
        output_activation: cute.Tensor,  # (T, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Pointer,  # uint8 gmem base of (local_ws_bytes,)
        shared_workspace: cute.Pointer,  # uint8 gmem base of (shared_ws_bytes,)
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
          * ``output_activation`` is the 2D (T, hidden) result.  Under
            ``in_kernel_reduce`` it is the per-rank cross-rank combine STG
            target and MUST be reachable via the peer mapper (sym heap, or
            local in the single-rank degenerate case).  Under
            ``separate_kernel_reduce`` the cross-rank target is the internal
            ``combine_quant`` staging region and ``output_activation`` only
            receives the local tail reduce, so it may be plain local memory.

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

        # MoE-domain ``(token_max, topk, hidden)`` cross-rank combine STG target.
        #   * in_kernel_reduce: REDG collapses topk on the fly, so this is a
        #     ``(T, 1, hidden)`` view of the caller's 2D ``output_activation``
        #     (the epilogue's topk index is a constexpr 0 in this mode).
        #   * separate_kernel_reduce: peers write one cell per (token, topk) into
        #     the internal ``combine_quant`` staging; the tail reduce below
        #     collapses topk into ``output_activation``.
        if cutlass.const_expr(self.in_kernel_fc2_reduce):
            combine_target = cute.make_tensor(
                output_activation.iterator,
                cute.make_layout(
                    (self.max_tokens_per_rank, 1, self.hidden),
                    stride=(self.hidden, self.hidden, 1),
                ),
            )
        else:
            combine_target = self._view_shared(shared_workspace,
                                               "combine_quant")

        # Per-block scale plane parallel to combine_quant; quantized formats only
        # (bf16 carries no SF, and in_kernel_reduce is bf16-by-construction).
        sf_blocks = (self.hidden // self.combine_format.scale_block
                     if self.combine_format.is_quantized else 0)
        if cutlass.const_expr(self.combine_format.is_quantized
                              and not self.in_kernel_fc2_reduce):
            # Assemble the (token, topk, valid_blocks) logical view + 16 B-aligned
            # per-block stride over the flat combine_sf capacity.
            combine_sf = self._view_shared(
                shared_workspace,
                "combine_sf",
                shape=(self.max_tokens_per_rank, self.num_topk, sf_blocks),
                stride=(self.num_topk * self.sf_block_pad, self.sf_block_pad,
                        1),
            )
        else:
            combine_sf = None

        if cutlass.const_expr(self.combine_format.is_quantized):
            # Same: logical (pool_token, 1, valid_blocks) + padded per-block stride
            # over the flat fc2_output_sf capacity.
            fc2_output_sf_phys = self._view_local(
                local_workspace,
                "fc2_output_sf",
                shape=(self.pool_token_capacity, 1, sf_blocks),
                stride=(self.sf_block_pad, self.sf_block_pad, 1),
            )
            sf_vec = self.combine_format.scale_block
            sf_layout = fc2_output_sf_phys.layout
            fc2_output_sf = cute.make_tensor(
                fc2_output_sf_phys.iterator,
                cute.make_layout(
                    (sf_layout.shape[0], sf_layout.shape[1],
                     (sf_vec, sf_layout.shape[2])),
                    stride=(sf_layout.stride[0], sf_layout.stride[1],
                            (0, sf_layout.stride[2])),
                ),
            )
        else:
            fc2_output_sf = None

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace,
                "fc2_output_workspace",
            )
            # Byte count = elements * dtype.width // 8 (multiply before divide so
            # the 4-bit fp4 data plane is not truncated to zero bytes/element).
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                ((pool_token_capacity * self.hidden *
                  int(self.combine_format.act_dtype.width)) // 8, ),
                None,
                self._local_region_by_name["fc2_output_workspace"].align,
            )
            combine_output_u8 = cute.recast_tensor(
                combine_target,
                cutlass.Uint8,
            )
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            combine_output_u8 = combine_target

        # fc2 completion gate: present whenever token-back runs (DATA and/or SF
        # push), so the epi_warps SF-only push gates per expert just like the
        # dispatch DATA path.
        if cutlass.const_expr(self.token_back_enabled):
            fc2_done_counter = self._view_local(
                local_workspace,
                "fc2_done_counter",
            )
        else:
            fc2_done_counter = None

        if cutlass.const_expr(
                self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace,
                "token_back_schedule_counter",
            ).iterator
        else:
            token_back_schedule_counter = None

        # Int32 views over each workspace's front counter prefix; the kernel tail
        # bulk-zeros them every launch (tail_reset_counters).
        local_zero_prefix = self._make_typed_view(
            local_workspace,
            0,
            cutlass.Int32,
            (self.local_zero_i32_count, ),
            (1, ),
            16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace,
            0,
            cutlass.Int32,
            (self.shared_zero_i32_count, ),
            (1, ),
            16,
        )
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
            combine_sf=combine_sf,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_output_sf=fc2_output_sf,
            fc2_done_counter=fc2_done_counter,
            token_back_schedule_counter=token_back_schedule_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            local_zero_prefix=local_zero_prefix,
            shared_zero_prefix=shared_zero_prefix,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=peer_rank_ptr_mapper_host.rank_idx,
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
        # ``combine_target`` is MoE-domain storage. separate_kernel_reduce uses
        # ``(max_tokens_per_rank, num_topk, hidden)`` (internal staging) and a
        # tail reduce; in_kernel_reduce uses ``(max_tokens_per_rank, 1, hidden)``
        # and reduces in kernel.  The epilogue return tile maps local pool rows
        # back to the source rank's token row through ``token_comm_args``.
        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_target = combine_target

        # Quantized combine loop is closed: the fc2 epilogue encodes the data
        # plane (STG-to-peer in epi_warps, or local staging in the dispatch
        # modes) and writes per-block scales to the rank-local SF plane
        # (fc2_output_sf); the token-back warps push that SF plane
        # token-contiguously to the peers' combine_sf (and the data plane too in
        # the dispatch modes); TopkReduce below dequantizes and reduces.
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

        # separate_kernel_reduce: collapse the per-topk ``combine_quant`` staging
        # into the public 2D output via the shared TopkReduce launcher (sized from
        # codegen-time hidden / num_topk / combine_format -- it dequantizes per
        # format and reduces over topk). Same stream, so it is ordered strictly
        # after the cross-rank combine writes landed (the mega kernel's nvlink
        # barrier guarantees all peer combine STGs to this rank completed before it
        # exits). Weighting follows the compute graph: deepgemm (apply_topk_in_fc1)
        # folded the routing weight into fc1 -> plain K-sum; transformers applies
        # topk_weights here.
        if cutlass.const_expr(not self.in_kernel_fc2_reduce):
            score = (topk_weights if
                     cutlass.const_expr(not self.apply_topk_in_fc1) else None)
            TopkReduce(self.hidden, self.num_topk, self.combine_format)(
                combine_target,
                combine_sf,
                output_activation,
                score,
                stream,
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
        self.token_comm.tail_reset_counters(
            token_comm_args,
            token_comm_args.shared_zero_prefix,
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
