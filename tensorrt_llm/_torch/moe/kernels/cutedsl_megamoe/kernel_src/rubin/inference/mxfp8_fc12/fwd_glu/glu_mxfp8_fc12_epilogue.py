# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous epilogue for the fused fc1+fc2 swap-AB MegaMoE kernel.

The B-reuse tile uses ``GluMxfp8BreuseEpilogue`` (glu_mxfp8_fc12_epilogue_breuse.py).

TensorRT-LLM-local file, not part of the upstream cutedsl_megamoe export: adapted
from the upstream training forward GLU kernel
(``sources/kernel_src/rubin/training/mega/fwd_glu/``). A re-export must carry it forward.
"""

import dataclasses
from typing import Any, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import Int64

from ......communication.token_protocol import TokenSrcMetadata
from ......helpers.flag_batch import GpuReleaseFlagBatchTracker
from ......helpers.iket_compat import iket
from ......helpers.ptx_helpers import cp_async_bulk_s2g, stg_e8m0_from_f32, stg_e8m0x8_from_f32
from ......helpers.ptx_helpers import (
    red_add_relaxed_sys_v2_bf16x2 as _red_add_relaxed_sys_v2_bf16x2,
)
from ......helpers.tmem_transpose import _TmemTranspose16x32Core
from ......quant_def import CombineFormat
from .....function_mapping import CoordinateSpace, FunctionMapping
from .....schedulers.fc12_mapping import BlockPhase
from ..helpers.utils import quant_sfd_row, swiglu_act

Fc1GateUpInterleave = 32
EpilogueTileN = 32
Fc1EpilogueOutputTileM = 256
Fc1EpilogueOutputTileN = 128
WarpThreadCount = 32
EpiWarpCount = 4
Fc1CTMAStages = 1


# =============================================================================
# FC2 output routing and training-specific store mappings
# =============================================================================


def make_fc2_stg_store_out_mapping(
    wire_width: int,
    cta_token_tile_size: int,
    cta_hidden_tile_size: int,
) -> FunctionMapping:
    """Per-vector-STG destination mapping for training's token-major FC2 tile."""
    elems_per_stg = min(256 // wire_width, EpilogueTileN)
    copy_iters = cta_hidden_tile_size // elems_per_stg
    return FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "iter_idx"), (cta_token_tile_size, copy_iters)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"),
            (cta_token_tile_size, cta_hidden_tile_size),
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": epi_tid,
            "hidden_in_cta_tile": iter_idx * elems_per_stg,
        },
    )


def make_fc2_ublk_store_out_mapping(
    wire_width: int,
    cta_token_tile_size: int,
    cta_hidden_tile_size: int,
) -> FunctionMapping:
    """Per-128B UBLK destination mapping.

    MXFP8 combines four 32-hidden accumulator subtiles per issue; BF16 combines
    two.  In both cases one epilogue thread owns one token and every issue stays
    within that token's contiguous hidden row.
    """
    elems_per_ublk = 128 * 8 // wire_width
    copy_iters = cta_hidden_tile_size // elems_per_ublk
    return FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "iter_idx"), (cta_token_tile_size, copy_iters)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"),
            (cta_token_tile_size, cta_hidden_tile_size),
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": epi_tid,
            "hidden_in_cta_tile": iter_idx * elems_per_ublk,
        },
    )


def make_fc2_redg_store_out_mapping(
    cta_token_tile_size: int,
    cta_hidden_tile_size: int,
) -> FunctionMapping:
    """Per-red.v2 destination mapping for training's coalesced BF16 REDG."""
    tokens_per_pass = 16
    hidden_per_issue = 4
    passes_per_subtile = cta_token_tile_size // tokens_per_pass
    subtile_count = cta_hidden_tile_size // EpilogueTileN
    return FunctionMapping(
        domain=CoordinateSpace(
            ("epi_tid", "iter_idx"),
            (cta_token_tile_size, subtile_count * passes_per_subtile),
        ),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"),
            (cta_token_tile_size, cta_hidden_tile_size),
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": (
                (iter_idx % passes_per_subtile) * tokens_per_pass
                + epi_tid // (EpilogueTileN // hidden_per_issue)
            ),
            "hidden_in_cta_tile": (
                (iter_idx // passes_per_subtile) * EpilogueTileN
                + (epi_tid % (EpilogueTileN // hidden_per_issue)) * hidden_per_issue
            ),
        },
    )


def make_fc2_sf_store_out_mapping(
    cta_token_tile_size: int,
    cta_hidden_tile_size: int,
) -> FunctionMapping:
    """Per-E8M0-block local destination mapping."""
    block_count = cta_hidden_tile_size // EpilogueTileN
    return FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "iter_idx"), (cta_token_tile_size, block_count)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_block_in_cta_tile"),
            (cta_token_tile_size, block_count),
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": epi_tid,
            "hidden_block_in_cta_tile": iter_idx,
        },
    )


@dataclasses.dataclass(frozen=True)
class Fc2OutputDest:
    """fc2 output destination in MoE-domain ``(token_max, topk, hidden)`` layout."""

    tensor: cute.Tensor
    metadata: Optional[cute.Tensor] = None
    peer_rank_ptr_mapper: Any = None
    reduce_topk_in_kernel: bool = False

    def __post_init__(self) -> None:
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2OutputDest: ``metadata`` and ``peer_rank_ptr_mapper`` must be "
                "both None (direct mode) or both non-None (MegaMoE / indirect "
                "mode).  Got metadata="
                f"{'set' if self.metadata is not None else 'None'}, "
                f"peer_rank_ptr_mapper="
                f"{'set' if self.peer_rank_ptr_mapper is not None else 'None'}."
            )

    @cute.jit
    def resolve_token_row(self, pool_token_global) -> cute.Tensor:
        """Return the ``(hidden,)`` BF16 GMEM row this pool token's STG lands on."""
        if cutlass.const_expr(self.metadata is None):
            # Int64 token coord: the (token, topk, hidden) row offset is token*(topk*hidden),
            # which overflows int32 once max_tokens*topk*hidden > 2^31 (mirrors inference).
            return cute.slice_(self.tensor, (Int64(pool_token_global), 0, None))

        md = TokenSrcMetadata.load(
            self.metadata.iterator.toint()
            + Int64(pool_token_global) * Int64(TokenSrcMetadata.nbytes)
        )
        src_rank = md.src_rank
        src_token = md.src_token
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            src_topk = cutlass.Int32(0)
        else:
            src_topk = md.src_topk
        # Int64 token coord: src_token*(topk*hidden) overflows int32 once
        local_row = cute.slice_(self.tensor, (Int64(src_token), src_topk, None))
        # next's SymmetricBufferDevice exposes ``map_pointer``
        peer_iter = self.peer_rank_ptr_mapper.map_pointer(
            local_row.iterator,
            src_rank,
            byte_alignment=32,
        )
        return cute.make_tensor(peer_iter, local_row.layout)


@dataclasses.dataclass(frozen=True)
class Fc2OutputRouter:
    """Unified training FC2 data/SF destination router.

    Direct STG/UBLK stores prefetch one invariant token row.  REDG distributes
    one prefetched row to each lane and shuffles row bases within 8-lane groups.
    """

    output_mapping: FunctionMapping
    pool_token_base: cutlass.Int32
    hidden_base_this_cta_tile: cutlass.Int32
    valid_tokens_this_cta_tile: cutlass.Int32
    valid_hidden: cutlass.Int32
    epi_tid: cutlass.Int32
    wire_bytes: int
    output: Any = None
    metadata: Any = None
    peer_rank_ptr_mapper: Any = None
    local_data_base_addr: Any = None
    local_data_token_base: Any = None
    local_data_row_stride: Any = None
    reduce_topk_in_kernel: bool = False
    prefetch_data_row: bool = True
    sf_mapping: Any = None
    sf_base_addr: Any = None
    sf_token_base: Any = None
    sf_row_stride: Any = None
    sf_hidden_block_base: Any = None
    sf_bytes: int = 1
    row_base: Any = None
    row_valid: Any = None

    def __post_init__(self) -> None:
        peer = self.metadata is not None
        local = self.local_data_base_addr is not None
        if peer == local:
            raise ValueError("Fc2OutputRouter requires exactly one peer or local data destination.")
        if peer != (self.peer_rank_ptr_mapper is not None):
            raise ValueError("Fc2OutputRouter requires peer_rank_ptr_mapper iff metadata is set.")
        if self.reduce_topk_in_kernel and not peer:
            raise ValueError("FC2 REDG routing requires a peer destination.")
        sf_fields = (
            self.sf_mapping,
            self.sf_base_addr,
            self.sf_token_base,
            self.sf_row_stride,
            self.sf_hidden_block_base,
        )
        if any(value is None for value in sf_fields) != all(value is None for value in sf_fields):
            raise ValueError(
                "Fc2OutputRouter SF mapping, base, token base, row stride, "
                "and hidden-block base must be provided together."
            )

    @cute.jit
    def prefetch(self) -> "Fc2OutputRouter":
        """Resolve one invariant DATA row before the store loop."""
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            redg_lanes_per_token: cutlass.Constexpr[int] = EpilogueTileN // 4
            prefetch_iter = self.epi_tid % cutlass.Int32(redg_lanes_per_token)
            coord = self.output_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=prefetch_iter)
        else:
            if cutlass.const_expr(not self.prefetch_data_row):
                return self
            coord = self.output_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=cutlass.Int32(0))

        token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
        row_base = Int64(0)
        row_valid = cutlass.Int32(0)
        token_valid = token_in_tile < self.valid_tokens_this_cta_tile
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            prefetch_valid = token_valid
        else:
            prefetch_valid = token_valid and self.hidden_base_this_cta_tile < self.valid_hidden
        if prefetch_valid:
            if cutlass.const_expr(self.metadata is None):
                local_token = self.local_data_token_base + token_in_tile
                row_base = self.local_data_base_addr + Int64(local_token) * Int64(
                    self.local_data_row_stride
                ) * Int64(self.wire_bytes)
            else:
                output_dest = Fc2OutputDest(
                    tensor=self.output,
                    metadata=self.metadata,
                    peer_rank_ptr_mapper=self.peer_rank_ptr_mapper,
                    reduce_topk_in_kernel=self.reduce_topk_in_kernel,
                )
                row_base = output_dest.resolve_token_row(
                    self.pool_token_base + token_in_tile
                ).iterator.toint()
            row_valid = cutlass.Int32(1)
        return dataclasses.replace(self, row_base=row_base, row_valid=row_valid)

    @cute.jit
    def get_data_addr_from_coord(self, token_in_tile, hidden_in_tile):
        """Return one DATA address from an already evaluated store coordinate."""
        hidden = self.hidden_base_this_cta_tile + hidden_in_tile
        addr = Int64(0)
        pred = cutlass.Int32(0)
        if token_in_tile < self.valid_tokens_this_cta_tile and hidden < self.valid_hidden:
            pred = cutlass.Int32(1)
            if cutlass.const_expr(self.prefetch_data_row):
                addr = self.row_base + Int64(hidden) * Int64(self.wire_bytes)
                pred = self.row_valid
            elif cutlass.const_expr(self.metadata is None):
                local_token = self.local_data_token_base + token_in_tile
                addr = (
                    self.local_data_base_addr
                    + Int64(local_token)
                    * Int64(self.local_data_row_stride)
                    * Int64(self.wire_bytes)
                    + Int64(hidden) * Int64(self.wire_bytes)
                )
            else:
                output_dest = Fc2OutputDest(
                    tensor=self.output,
                    metadata=self.metadata,
                    peer_rank_ptr_mapper=self.peer_rank_ptr_mapper,
                    reduce_topk_in_kernel=self.reduce_topk_in_kernel,
                )
                addr = output_dest.resolve_token_row(
                    self.pool_token_base + token_in_tile
                ).iterator.toint() + Int64(hidden) * Int64(self.wire_bytes)
        return addr, pred

    @cute.jit
    def get_data_addr(self, iter_idx):
        """Return the mapped byte address and validity predicate for one issue."""
        coord = self.output_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
        return self.get_data_addr_from_coord(
            cutlass.Int32(coord["token_in_cta_tile"]),
            cutlass.Int32(coord["hidden_in_cta_tile"]),
        )

    @cute.jit
    def get_sf_addr(self, iter_idx):
        """Return the local E8M0 destination for one mapped scale block."""
        coord = self.sf_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
        token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
        hidden_block_in_tile = cutlass.Int32(coord["hidden_block_in_cta_tile"])
        hidden_block = self.sf_hidden_block_base + hidden_block_in_tile
        addr = Int64(0)
        pred = cutlass.Int32(0)
        if (
            token_in_tile < self.valid_tokens_this_cta_tile
            and hidden_block * cutlass.Int32(EpilogueTileN) < self.valid_hidden
        ):
            pred = cutlass.Int32(1)
            sf_token = self.sf_token_base + token_in_tile
            addr = (
                self.sf_base_addr
                + Int64(sf_token) * Int64(self.sf_row_stride) * Int64(self.sf_bytes)
                + Int64(hidden_block) * Int64(self.sf_bytes)
            )
        return addr, pred


@dataclasses.dataclass(frozen=True)
class Fc2TmaTileRouter:
    """Own tile-level addressing for the rank-local FC2 TMA destination."""

    output: cute.Tensor
    pool_token_base: cutlass.Int32
    hidden_base_this_cta_tile: cutlass.Int32
    valid_tokens_this_cta_tile: cutlass.Int32
    cta_token_tile_size: int

    @cute.jit
    def get_data_tile(self, subtile_idx):
        tiled_output = cute.flat_divide(self.output, (self.cta_token_tile_size, EpilogueTileN))
        token_tile_idx = self.pool_token_base // cutlass.Int32(self.cta_token_tile_size)
        hidden_tile_idx = (
            self.hidden_base_this_cta_tile + subtile_idx * cutlass.Int32(EpilogueTileN)
        ) // cutlass.Int32(EpilogueTileN)
        return (
            tiled_output[None, None, token_tile_idx, hidden_tile_idx],
            self.valid_tokens_this_cta_tile > 0,
        )


# =============================================================================
# GluMxfp8Epilogue
# =============================================================================


class GluMxfp8Epilogue:
    """Epilogue of the fused FC1 (+ SwiGLU + MXFP8 quant) / FC2 kernel: 4 warps, one
    128-row task tile per work tile. Tile variants override the ``_get_task_tile``,
    ``_run_fc1_subtiles`` and ``_run_fc2_task_tile`` hooks instead of branching here.
    """

    _SubtileBarIdBase = 4
    # Named barrier for cross-warp sync during raw-C TMA stores.
    _CStoreBarId = 10

    def _validate_tile_shape(self) -> None:
        assert self._cta_tile_m == self._epi_tile_m, (
            f"{type(self).__name__} needs a per-CTA tile M of "
            f"{self._epi_tile_m}, got {self._cta_tile_m}"
        )
        if len(self._epilogue_warp_ids) != EpiWarpCount:
            raise ValueError(
                f"epilogue requires {EpiWarpCount} warps, got {len(self._epilogue_warp_ids)}."
            )

    def __init__(
        self,
        *,
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        use_2cta_instrs: bool,
        sf_vec_size: int,
        fc1_output_dtype: Type[cutlass.Numeric],
        fc1_output_layout: utils.LayoutEnum,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
        c_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        glu_clamp: Optional[float] = None,
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        epilog_sync_bar_id: int = 1,
        epilogue_warp_ids: Tuple[int, ...] = (0, 1, 2, 3),
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        apply_topk_in_fc1: bool = False,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
        combine_format: Optional[Any] = None,
        act_func: str = "swiglu",
        fc2_use_bulk: bool = False,
        fc2_tma_stages: Optional[int] = None,
        one_task_per_cta: bool = False,
        fc1_done_slot_m: Optional[int] = None,
    ) -> None:
        """``one_task_per_cta``: one work tile per CTA, so a single accumulator stage suffices."""
        self._act_func = act_func
        self._situ_beta = situ_beta
        self._situ_linear_beta = situ_linear_beta
        self.fc1_output_dtype = fc1_output_dtype
        self.fc1_output_layout = fc1_output_layout
        self.acc_dtype = acc_dtype
        self.sf_dtype = sf_dtype
        self._sf_vec_size = sf_vec_size
        self._c_dtype = c_dtype
        self._epilog_sync_bar_id = epilog_sync_bar_id
        self._epilogue_warp_ids = epilogue_warp_ids
        self._use_2cta_instrs = use_2cta_instrs

        self._atom_thr_size = 2 if use_2cta_instrs else 1
        self._cluster_m = cluster_shape_mn[0]
        # CTA tiles folded into one fc1_done slot; the kernel owns the choice (MMA band on the
        # standalone path, cluster token tile under MegaMoE) and both sides must agree.
        self._fc1_done_slot_m = self._atom_thr_size if fc1_done_slot_m is None else fc1_done_slot_m
        if self._cluster_m % self._fc1_done_slot_m != 0:
            raise ValueError(
                f"fc1_done_slot_m ({self._fc1_done_slot_m}) must divide "
                f"cluster_m ({self._cluster_m})."
            )
        self._cta_tile_m = mma_tiler_mnk[0] // self._atom_thr_size
        self._epi_tile_m = EpiWarpCount * WarpThreadCount
        self._validate_tile_shape()
        self._cta_tile_n = mma_tiler_mnk[1]
        self._mma_tiler_k = mma_tiler_mnk[2]
        self._cta_tile_n_sfb = ((mma_tiler_mnk[1] + 127) // 128) * 128
        self._static_expert_shape = static_expert_shape
        if (
            static_expert_shape is not None
            and static_expert_shape[2] % (self._cta_tile_m * cluster_shape_mn[0]) == 0
        ):
            self._fc2_stg_needs_predicate: bool = False
        else:
            self._fc2_stg_needs_predicate: bool = True

        # SwiGLU folds the accumulator's N columns (gate and up) into N // 2 outputs.
        self._epi_tile = (EpilogueTileN, self._cta_tile_n // 2)
        self._subtile_cnt = self._cta_tile_n // 2 // EpilogueTileN

        self._num_acc_stage = 1 if one_task_per_cta else 2
        self._num_acc_pipeline_stages = self._num_acc_stage

        k = self._mma_tiler_k
        self._num_sfa_tmem_cols = self._cta_tile_m * k // sf_vec_size * 4 // 4 // 128
        self._num_sfb_tmem_cols = self._cta_tile_n_sfb * k // sf_vec_size * 4 // 4 // 128

        self._num_accumulator_tmem_cols = self._cta_tile_n * self._num_acc_stage

        self._fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self._token_back_by_dispatch = token_back_by_dispatch
        self._apply_topk_in_fc1 = apply_topk_in_fc1
        self._generate_c = generate_c
        self._use_stg_fc1 = use_stg_fc1
        # combine_format determines fc2 output encoding: bf16 (default) or quantized.
        if combine_format is None:
            combine_format = CombineFormat.parse("bf16")
        self._combine_format = combine_format
        self._combine_mxfp8 = combine_format.is_quantized
        if self._combine_mxfp8 and (
            combine_format.act_dtype is not cutlass.Float8E4M3FN
            or combine_format.scale_dtype is not cutlass.Float8E8M0FNU
            or combine_format.scale_block != EpilogueTileN
        ):
            raise NotImplementedError(
                "The training GLU epilogue supports only 32e4m3xe8m0 for a quantized combine."
            )
        self._hidden_fc2 = static_expert_shape[2] if static_expert_shape is not None else 0
        # sf_block_pad for fc2 MXFP8 combine
        if self._combine_mxfp8 and static_expert_shape is not None:
            _sf_blocks_fc2 = self._hidden_fc2 // EpilogueTileN
            self._fc2_sf_block_pad = ((_sf_blocks_fc2 + 15) // 16) * 16
        else:
            self._fc2_sf_block_pad = 0
        # batching stg.64 SF
        self._fc2_sf_batch8 = (
            self._combine_mxfp8
            and self._hidden_fc2 > 0
            and (self._hidden_fc2 % self._cta_tile_n == 0)
            and (self._cta_tile_n // EpilogueTileN == 8)
        )
        self._epi_tile_c = (self._epi_tile_m, 2 * Fc1GateUpInterleave)
        self._epi_fc1_batch = max(1, epi_flag_batch[0])
        self._epi_fc2_batch = max(1, epi_flag_batch[1])

        self.glu_clamp = cutlass.Float32(glu_clamp) if glu_clamp is not None else None

        # When FC1 and FC2 work tiles are separated by a bulk-group drain and a
        # 128-thread rendezvous, so their scratch can share one physical slab.
        d_staged_layout = self.staged_smem_layout(self._subtile_cnt)
        self._fc1_d_scratch_bytes = (
            0
            if self._use_stg_fc1
            else int(cute.size_in_bytes(self.fc1_output_dtype, d_staged_layout))
        )
        self._fc1_c_scratch_offset = ((self._fc1_d_scratch_bytes + 127) // 128) * 128
        self._fc1_c_scratch_bytes = (
            int(
                cute.size_in_bytes(
                    self._c_dtype,
                    self.staged_c_smem_layout(Fc1CTMAStages),
                )
            )
            if self._generate_c
            else 0
        )
        self._fc1_scratch_bytes = (
            self._fc1_c_scratch_offset + self._fc1_c_scratch_bytes
            if self._generate_c
            else self._fc1_d_scratch_bytes
        )

        self._fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8
        bulk_hidden_doable = (
            static_expert_shape is not None and self._hidden_fc2 % self._cta_tile_n == 0
        )
        if fc2_use_bulk and static_expert_shape is not None and not bulk_hidden_doable:
            raise ValueError(
                "FC2 bulk store requires hidden to be divisible by the FC2 CTA tile N."
            )
        self._fc2_use_tma = (
            bool(fc2_use_bulk)
            and token_back_by_dispatch
            and not self._fc2_in_kernel_topk_reduce
            and bulk_hidden_doable
        )
        self._fc2_use_ublk = (
            bool(fc2_use_bulk)
            and not token_back_by_dispatch
            and not self._fc2_in_kernel_topk_reduce
            and bulk_hidden_doable
        )
        self._fc2_needs_staging = self._fc2_use_tma or self._fc2_use_ublk
        # Phases that commit bulk (TMA / cp.async.bulk) stores and need the drain in run().
        self._fc1_bulk_store = (not self._use_stg_fc1) or self._generate_c
        self._fc2_bulk_store = self._fc2_needs_staging
        self._fc2_wire_dtype = (
            self._combine_format.act_dtype if self._combine_mxfp8 else cutlass.BFloat16
        )
        self._fc2_wire_bytes = self._fc2_wire_dtype.width // 8
        self._fc2_ublk_bytes = 128
        self._fc2_ublk_elems = self._fc2_ublk_bytes // self._fc2_wire_bytes
        self._fc2_ublk_subtiles = self._fc2_ublk_elems // EpilogueTileN
        self._fc2_ublk_issue_count = self._fc2_subtile_cnt // self._fc2_ublk_subtiles
        self._fc2_staging_hidden = self._fc2_ublk_elems if self._fc2_use_ublk else EpilogueTileN
        max_store_stages = (
            self._fc2_ublk_issue_count if self._fc2_use_ublk else self._fc2_subtile_cnt
        )
        if fc2_tma_stages is not None and not 1 <= fc2_tma_stages <= max_store_stages:
            raise ValueError(
                f"fc2_tma_stages must be in [1, {max_store_stages}], got {fc2_tma_stages}."
            )
        single_stage_bytes = (
            self._epi_tile_m * self._fc2_staging_hidden * self._fc2_wire_dtype.width // 8
        )
        if not self._fc2_needs_staging:
            self._fc2_tma_stages = 0
        elif fc2_tma_stages is not None:
            self._fc2_tma_stages = fc2_tma_stages
        elif self._fc2_use_ublk:
            # The 128B peer path measured best at one stage.
            self._fc2_tma_stages = 1
        else:
            # TMA-only auto sizing: consume the FC1 scratch lifetime that is free
            # during FC2, with at least two BF16 store stages.
            bf16_baseline_bytes = (
                min(2, max_store_stages)
                * self._epi_tile_m
                * EpilogueTileN
                * cutlass.BFloat16.width
                // 8
            )
            available_staging_bytes = max(self._fc1_scratch_bytes, bf16_baseline_bytes)
            auto_stage_cap = min(4, max_store_stages)
            self._fc2_tma_stages = max(
                1,
                min(
                    auto_stage_cap,
                    available_staging_bytes // single_stage_bytes,
                ),
            )
        self._fc2_store_out_mapping = (
            make_fc2_ublk_store_out_mapping(
                self._fc2_wire_dtype.width, self._epi_tile_m, self._cta_tile_n
            )
            if self._fc2_use_ublk
            else make_fc2_stg_store_out_mapping(
                self._fc2_wire_dtype.width, self._epi_tile_m, self._cta_tile_n
            )
        )
        self._fc2_redg_store_out_mapping = make_fc2_redg_store_out_mapping(
            self._epi_tile_m, self._cta_tile_n
        )
        self._fc2_sf_store_out_mapping = make_fc2_sf_store_out_mapping(
            self._epi_tile_m, self._cta_tile_n
        )
        self._fc2_tma_stage_bytes = single_stage_bytes
        self._fc2_tma_staging_bytes = self._fc2_tma_stages * self._fc2_tma_stage_bytes

        self._fc2_reduce_coalesce = bool(self._fc2_in_kernel_topk_reduce)

        # bf16 staging tile: (cta_tile_m tokens x EpilogueTileN hidden), 2 B/elem.
        self._fc2_reduce_staging_bytes = (
            self._epi_tile_m * EpilogueTileN * cutlass.BFloat16.width // 8
            if self._fc2_reduce_coalesce
            else 0
        )
        self._fc2_scratch_bytes = max(
            self._fc2_tma_staging_bytes,
            self._fc2_reduce_staging_bytes,
        )
        self._scratch_bytes = max(
            self._fc1_scratch_bytes,
            self._fc2_scratch_bytes,
        )

    # -- Codegen-time queries  --

    @property
    def epi_tile(self) -> Tuple[int, int]:
        return self._epi_tile

    @property
    def num_acc_pipeline_stages(self) -> int:
        return self._num_acc_pipeline_stages

    @property
    def num_acc_stage(self) -> int:
        return self._num_acc_stage

    @property
    def subtile_cnt(self) -> int:
        return self._subtile_cnt

    @property
    def cta_tile_n(self) -> int:
        return self._cta_tile_n

    @property
    def num_sfa_tmem_cols(self) -> int:
        return self._num_sfa_tmem_cols

    @property
    def num_sfb_tmem_cols(self) -> int:
        return self._num_sfb_tmem_cols

    @property
    def num_accumulator_tmem_cols(self) -> int:
        return self._num_accumulator_tmem_cols

    def staged_smem_layout(
        self,
        n_stages: int,
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        return sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            self.fc1_output_layout,
            self._epi_tile,
            n_stages,
        )

    @property
    def smem_layout_one_stage(self) -> Union[cute.Layout, cute.ComposedLayout]:
        staged = self.staged_smem_layout(1)
        return cute.select(staged, mode=[0, 1])

    @property
    def bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self.fc1_output_dtype, self.smem_layout_one_stage)

    @property
    def epi_tile_c(self) -> Tuple[int, int]:
        """TMA tile for raw gate+up output: (cta_tile_m=128, 2*Fc1GateUpInterleave=64) fp32."""
        return self._epi_tile_c

    def staged_c_smem_layout(self, n_stages: int):
        """SMEM layout for n_stages of raw gate+up (Float32, row-major, epi_tile_c)."""
        return sm100_utils.make_smem_layout_epi(
            self._c_dtype,
            self.fc1_output_layout,  # same N-major direction as fc1 output
            self._epi_tile_c,
            n_stages,
        )

    @property
    def c_smem_layout_one_stage(self):
        return cute.select(self.staged_c_smem_layout(1), mode=[0, 1])

    @property
    def c_bytes_per_stage(self) -> int:
        return cute.size_in_bytes(self._c_dtype, self.c_smem_layout_one_stage)

    # ── FC2 TMASTG staging ───────────────────────────────────────────────
    @property
    def fc2_use_tma(self) -> bool:
        return self._fc2_use_tma

    @property
    def fc2_use_ublk(self) -> bool:
        return self._fc2_use_ublk

    @property
    def fc2_needs_staging(self) -> bool:
        """True when the FC2 store stages into the SMEM tile: either the dispatch
        TMASTG path (``fc2_use_tma``) or the epi_warps UBLK path (``fc2_use_ublk``)."""
        return self._fc2_needs_staging

    @property
    def fc2_tma_stages(self) -> int:
        return self._fc2_tma_stages

    @property
    def fc2_tma_staging_bytes(self) -> int:
        return self._fc2_tma_staging_bytes

    @property
    def fc1_c_scratch_offset(self) -> int:
        return self._fc1_c_scratch_offset

    @property
    def scratch_bytes(self) -> int:
        """Physical bytes for the FC1/FC2 lifetime-overlaid scratch slab."""
        return self._scratch_bytes

    @property
    def fc2_tma_tile(self) -> Tuple[int, int]:
        """TMA store tile: (epi_tile_m=128 tokens, EpilogueTileN=32 hidden)."""
        return (self._epi_tile_m, EpilogueTileN)

    def fc2_tma_staged_smem_layout(self, n_stages: int):
        """Token-major FC2 staging.

        Dispatch TMA retains a 128x32 tile.  Peer UBLK expands each token row
        to exactly 128 bytes: 128 MXFP8 or 64 BF16 elements.
        """
        epi_tile_m = self._epi_tile_m
        stage_stride = epi_tile_m * self._fc2_staging_hidden
        layout = cute.make_layout(
            (epi_tile_m, self._fc2_staging_hidden, n_stages),
            stride=(
                self._fc2_staging_hidden,
                1,
                stage_stride if n_stages > 1 else 0,
            ),
        )
        return layout

    @property
    def fc2_tma_smem_layout_one_stage(self):
        return cute.select(self.fc2_tma_staged_smem_layout(1), mode=[0, 1])

    # ── FC2 in-kernel reduce coalescing ──────────────────────────────────
    @property
    def fc2_reduce_coalesce(self) -> bool:
        return self._fc2_reduce_coalesce

    @property
    def fc2_reduce_staging_bytes(self) -> int:
        return self._fc2_reduce_staging_bytes

    def fc2_reduce_smem_layout(self):
        """Row-major (cta_tile_m tokens, EpilogueTileN hidden) bf16 transpose tile.
        Each epi thread writes its token's contiguous hidden row (64 B STS); the
        coalesced-issue re-partition reads 4-elem chunks hidden-major."""
        return cute.make_layout((self._epi_tile_m, EpilogueTileN), stride=(EpilogueTileN, 1))

    @staticmethod
    @cute.jit
    def tma_store_fc1_output(
        warp_idx,
        sC,
        store_idx,
        tma_atom_fc1_output: cute.CopyAtom,
        g_fc1_output_subtile_view: cute.Tensor,
        valid_tokens,
    ) -> None:
        """Per-warp TMA store for FC1 output."""
        cute.arch.fence_proxy("async.shared", space="cta")
        sC_stage = cute.slice_(sC, (None, None, store_idx))
        g_fc1_output_2d = cute.slice_(g_fc1_output_subtile_view, (None, None, 0))
        bSG_sC, bSG_g = cpasync.tma_partition(
            tma_atom_fc1_output,
            0,
            cute.make_layout(1),
            cute.group_modes(sC_stage, 0, 2),
            cute.group_modes(g_fc1_output_2d, 0, 2),
        )

        leader_warp = store_idx
        tile_has_valid = store_idx * cutlass.Int32(EpilogueTileN) < valid_tokens

        bar_id = store_idx + cutlass.Int32(GluMxfp8Epilogue._SubtileBarIdBase)
        bar = pipeline.NamedBarrier(
            barrier_id=bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        if warp_idx == leader_warp:
            bar.arrive_and_wait()
            # TMA bulk-tensor stores are all-or-nothing per issue (no per-element
            # `pred` mask, no scalar predicate arg), so guard the whole copy with
            # a runtime `if`.  Skips fully-padding token-tiles that would alias
            # the next expert's region.
            if tile_has_valid:
                cute.copy(tma_atom_fc1_output, bSG_sC, bSG_g)
        else:
            bar.arrive()

    @cute.jit
    def _store_fc1_c_subtile(
        self,
        r_gate: cute.Tensor,
        r_up: cute.Tensor,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c_subtile_view: cute.Tensor,
        c_buffer_idx,
        work_tile_info,
        warp_idx: int,
        tidx,
        c_pipeline,
    ) -> None:
        """Store pre-SwiGLU gate/up accumulators to the global C tensor via SMEM staging."""
        r_layout = cute.make_layout((((Fc1GateUpInterleave,), 1),), stride=(((1,), 0),))

        # Cast acc_dtype (Float32) → c_dtype (e.g. BFloat16) before R2S.
        r_gate_c = cute.make_rmem_tensor(r_layout.shape, self._c_dtype)
        r_up_c = cute.make_rmem_tensor(r_layout.shape, self._c_dtype)
        r_gate_c.store(r_gate.load().to(self._c_dtype))
        r_up_c.store(r_up.load().to(self._c_dtype))

        r2s_c_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._c_dtype,
            num_bits_per_copy=128,
        )
        thread_in_warp_c = tidx % cutlass.Int32(WarpThreadCount)
        c_row = cutlass.Int32(warp_idx * EpilogueTileN) + thread_in_warp_c
        sC_raw_stage = cute.slice_(smem_c_buffer, (None, None, c_buffer_idx))
        c_gate_smem = cute.local_tile(
            sC_raw_stage,
            (1, Fc1GateUpInterleave),
            (c_row, cutlass.Int32(0)),
        )
        cute.copy(r2s_c_atom, cute.coalesce(r_gate_c), cute.coalesce(c_gate_smem))
        c_up_smem = cute.local_tile(
            sC_raw_stage,
            (1, Fc1GateUpInterleave),
            (c_row, cutlass.Int32(1)),
        )
        cute.copy(r2s_c_atom, cute.coalesce(r_up_c), cute.coalesce(c_up_smem))

        # Fence + barrier: ensure all warps have written before TMA issue.
        cute.arch.fence_proxy("async.shared", space="cta")
        c_store_bar = pipeline.NamedBarrier(
            barrier_id=self._c_store_bar_id(work_tile_info),
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        c_store_bar.arrive_and_wait()

        # Warp 0 issues TMA S2G, then the one-stage pipeline overlaps the store.
        if warp_idx == 0:
            if work_tile_info.valid_tokens_in_cta_tile > cutlass.Int32(0):
                g_c_2d = cute.slice_(gmem_c_subtile_view, (None, None, 0))
                bSG_sC, bSG_gC = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC_raw_stage, 0, 2),
                    cute.group_modes(g_c_2d, 0, 2),
                )
                cute.copy(tma_atom_c, bSG_sC, bSG_gC)
            c_pipeline.producer_commit()

    def _c_store_bar_id(self, work_tile_info):
        """Named-barrier ID for one raw-C producer group.

        The base epilogue has one four-warp producer group. Tile variants with
        multiple independent producer groups override this hook.
        """
        return cutlass.Int32(self._CStoreBarId)

    def _subtile_local_tmem_tensor_pair(
        self,
        tmem_acc_tensor: cute.Tensor,
        subtile_idx,
        warp_idx,
    ) -> cute.Tensor:
        """
        Build a (gate, up) pair of TMEM tensor views for the MXFP8 fc1 epilogue.
        """
        base = tmem_acc_tensor.iterator
        warp_lane_off = warp_idx * WarpThreadCount
        subtile_col_off = subtile_idx * EpilogueTileN * 2
        total = (warp_lane_off << 16) + subtile_col_off
        subtile_gate_ptr = base + cute.assume(total, divby=16)
        subtile_up_ptr = base + cute.assume(total + Fc1GateUpInterleave, divby=16)
        return (
            cute.make_tensor(
                subtile_gate_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
            cute.make_tensor(
                subtile_up_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
        )

    def _subtile_forward_tmem_tensor(
        self,
        tmem_gate_tensor: cute.Tensor,
        tmem_up_tensor: cute.Tensor,
        col_offset: int,
    ) -> (cute.Tensor, cute.Tensor):
        """Move the tmem tensor to the correct position."""
        tmem_gate_ptr = tmem_gate_tensor.iterator + cute.assume(col_offset, divby=16)
        tmem_up_ptr = tmem_up_tensor.iterator + cute.assume(col_offset, divby=16)
        return (
            cute.make_tensor(
                tmem_gate_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
            cute.make_tensor(
                tmem_up_ptr,
                _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
            ),
        )

    # -- fc1 subtile: SM100 path --
    @cute.jit
    def _run_fc1_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        sched_ext,
        gmem_fc1_output: cute.Tensor,
        gmem_fc1_output_sf: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c: cute.Tensor,
        c_pipeline,
        acc_wait: bool = True,
        acc_release: bool = True,
    ) -> None:
        """MXFP8 fc1 task-tile: TmemTranspose16x32 TMEM loading + cross-warp E8M0 exchange."""
        real_fc1_output, _ = sched_ext.get_gmem_tensor("d", gmem_fc1_output, work_tile_info)
        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            real_fc1_output_sf, _ = sched_ext.get_gmem_tensor(
                "sfd", gmem_fc1_output_sf, work_tile_info
            )
        else:
            real_fc1_output_sf = None

        real_topk_scores = gmem_topk_scores
        if cutlass.const_expr(self._apply_topk_in_fc1):
            real_topk_scores, _ = sched_ext.get_gmem_tensor(
                "topk", gmem_topk_scores, work_tile_info
            )

        if cutlass.const_expr(self._generate_c):
            real_c, _ = sched_ext.get_gmem_tensor("c", gmem_c, work_tile_info)
            c_n_base = work_tile_info.tile_n_idx * cutlass.Int32(self._subtile_cnt)
        else:
            real_c = None
            c_n_base = None

        if cutlass.const_expr(acc_wait):
            acc_pipeline.consumer_wait(acc_consumer_state)
        if warp_idx == 0:
            iket.range_push("fc1_epi_tile")

        subtile_cnt = self._subtile_cnt
        tmem_gate, tmem_up = self._subtile_local_tmem_tensor_pair(
            tmem_acc_tensor,
            0,
            warp_idx,
        )
        tmem_forward_cols = Fc1GateUpInterleave * 2

        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            layout_sf = cute.make_layout(4)
            rmem_sf = cute.make_rmem_tensor(layout_sf.shape, self.acc_dtype)
        else:
            rmem_sf = None

        # Set up RMEM→SMEM copy atom (direct CopyUniversalOp)
        r2s_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=128,
        )

        # Keyword bundles shared by every subtile call; ``lean_c_args`` are the raw-C placeholders.
        subtile_args = dict(
            real_fc1_output=real_fc1_output,
            real_fc1_output_sf=real_fc1_output_sf,
            real_topk_scores=real_topk_scores,
            work_tile_info=work_tile_info,
            smem_fc1_output_buffer=smem_fc1_output_buffer,
            tma_atom_fc1_output=tma_atom_fc1_output,
            r2s_copy_atom=r2s_copy_atom,
            warp_idx=warp_idx,
            tidx=tidx,
            alpha=alpha,
            norm_const=norm_const,
            rmem_sf=rmem_sf,
            c_pipeline=c_pipeline,
        )
        lean_c_args = dict(
            smem_c_buffer=smem_fc1_output_buffer,
            tma_atom_c=tma_atom_fc1_output,
            gmem_c_subtile_view=real_fc1_output,
            c_buffer_idx=cutlass.Int32(0),
        )

        self._run_fc1_subtiles(
            tmem_gate=tmem_gate,
            tmem_up=tmem_up,
            tmem_forward_cols=tmem_forward_cols,
            subtile_cnt=subtile_cnt,
            subtile_args=subtile_args,
            lean_c_args=lean_c_args,
            real_c=real_c,
            c_n_base=c_n_base,
            smem_c_buffer=smem_c_buffer,
            tma_atom_c=tma_atom_c,
            work_tile_info=work_tile_info,
            acc_pipeline=acc_pipeline,
            acc_consumer_state=acc_consumer_state,
            acc_release=acc_release,
        )

        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            self._stg_sf_fc1(rmem_sf, real_fc1_output_sf, work_tile_info, tidx)

        # TMA store: 4 stores (one per warp group) after all subtiles
        if cutlass.const_expr(not self._use_stg_fc1):
            base_token_tile = work_tile_info.tile_m_idx * cutlass.Int32(
                self._epi_tile_m // EpilogueTileN
            )
            for idx in cutlass.range_constexpr(EpiWarpCount):
                g_fc1_output_warp_view = cute.local_tile(
                    real_fc1_output,
                    (EpilogueTileN, self._epi_tile[1], 1),
                    (base_token_tile + idx, work_tile_info.tile_n_idx, 0),
                )
                GluMxfp8Epilogue.tma_store_fc1_output(
                    warp_idx,
                    smem_fc1_output_buffer,
                    idx,
                    tma_atom_fc1_output,
                    g_fc1_output_warp_view,
                    work_tile_info.valid_tokens_in_cta_tile,
                )

        if warp_idx == 0:
            iket.range_pop()

    @cute.jit
    def _run_fc1_subtiles(
        self,
        tmem_gate,
        tmem_up,
        tmem_forward_cols,
        subtile_cnt,
        subtile_args,
        lean_c_args,
        real_c,
        c_n_base,
        smem_c_buffer,
        tma_atom_c,
        work_tile_info,
        acc_pipeline,
        acc_consumer_state,
        acc_release: bool,
    ) -> None:
        """fc1 subtile schedule up to and including the accumulator release (here: all in order).

        Overrides must not wait on the accumulator, must release it exactly once after their
        last TMEM read, and must have issued every subtile's stores and filled ``rmem_sf``."""
        for i in cutlass.range(0, subtile_cnt, 1, unroll=1):
            subtile_idx = cutlass.Int32(i)

            if cutlass.const_expr(self._generate_c):
                g_c_subtile = cute.local_tile(
                    real_c,
                    (self._epi_tile_m, 2 * Fc1GateUpInterleave, 1),
                    (work_tile_info.tile_m_idx, c_n_base + subtile_idx, cutlass.Int32(0)),
                )
                c_args = dict(
                    smem_c_buffer=smem_c_buffer,
                    tma_atom_c=tma_atom_c,
                    gmem_c_subtile_view=g_c_subtile,
                    c_buffer_idx=cutlass.Int32(i % Fc1CTMAStages),
                )
            else:
                c_args = lean_c_args

            self._run_fc1_subtile(
                subtile_idx=subtile_idx,
                tmem_gate_tensor=tmem_gate,
                tmem_up_tensor=tmem_up,
                **subtile_args,
                **c_args,
            )

            tmem_gate, tmem_up = self._subtile_forward_tmem_tensor(
                tmem_gate, tmem_up, tmem_forward_cols
            )

        self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, acc_release)

    @cute.jit
    def _swiglu_act(
        self,
        t_swiglu: cute.Tensor,
        t_up: cute.Tensor,
        t_gate: cute.Tensor,
        prob: Optional[Float32] = None,
    ) -> None:
        """SwiGLU hook consumed by ``_run_fc1_subtile``"""
        swiglu_act(t_swiglu, t_up, t_gate, prob)

    @cute.jit
    def _situ_act(
        self,
        t_situ: cute.Tensor,
        t_up: cute.Tensor,
        t_gate: cute.Tensor,
        prob: Optional[Float32] = None,
    ) -> None:
        """SiTU activation used by Kimi K3."""
        one_pair = (Float32(1.0), Float32(1.0))
        neg_log2e_pair = (Float32(-1.4426950408889634),) * 2
        inv_2beta = Float32(2.0 / self._situ_beta)
        two_beta = Float32(2.0 * self._situ_beta)
        neg_beta = Float32(-self._situ_beta)
        inv_2linear_beta = Float32(2.0 / self._situ_linear_beta)
        two_linear_beta = Float32(2.0 * self._situ_linear_beta)
        neg_linear_beta = Float32(-self._situ_linear_beta)

        def _sigmoid(values):
            scaled = cute.arch.mul_packed_f32x2(
                values,
                neg_log2e_pair,
                rnd="rn",
                ftz=False,
            )
            exponent = (
                cute.math.exp2(scaled[0], fastmath=True),
                cute.math.exp2(scaled[1], fastmath=True),
            )
            denominator = cute.arch.add_packed_f32x2(exponent, one_pair)
            return (
                cute.arch.rcp_approx(denominator[0]),
                cute.arch.rcp_approx(denominator[1]),
            )

        for i in cutlass.range_constexpr(0, cute.size(t_situ), 2):
            gate = (t_gate[i], t_gate[i + 1])
            up = (t_up[i], t_up[i + 1])
            gate_sigmoid = _sigmoid(gate)

            gate_scaled = cute.arch.mul_packed_f32x2(
                gate,
                (inv_2beta, inv_2beta),
                rnd="rn",
                ftz=False,
            )
            gate_tanh = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(
                    _sigmoid(gate_scaled),
                    (two_beta, two_beta),
                    rnd="rn",
                    ftz=False,
                ),
                (neg_beta, neg_beta),
            )

            up_scaled = cute.arch.mul_packed_f32x2(
                up,
                (inv_2linear_beta, inv_2linear_beta),
                rnd="rn",
                ftz=False,
            )
            up_tanh = cute.arch.add_packed_f32x2(
                cute.arch.mul_packed_f32x2(
                    _sigmoid(up_scaled),
                    (two_linear_beta, two_linear_beta),
                    rnd="rn",
                    ftz=False,
                ),
                (neg_linear_beta, neg_linear_beta),
            )

            situ_gate = cute.arch.mul_packed_f32x2(
                gate_tanh,
                gate_sigmoid,
                rnd="rn",
                ftz=False,
            )
            output = cute.arch.mul_packed_f32x2(
                situ_gate,
                up_tanh,
                rnd="rn",
                ftz=False,
            )
            if cutlass.const_expr(prob is not None):
                output = cute.arch.mul_packed_f32x2(
                    output,
                    (prob, prob),
                    rnd="rn",
                    ftz=False,
                )
            (t_situ[i], t_situ[i + 1]) = output

    @cute.jit
    def _fc1_subtile_load(
        self,
        tmem_gate_tensor: cute.Tensor,
        tmem_up_tensor: cute.Tensor,
    ) -> (cute.Tensor, cute.Tensor):
        """LDTM one fc1 subtile's gate and up columns into f32 registers."""
        r_layout = cute.make_layout((((Fc1GateUpInterleave,), 1),), stride=(((1,), 0),))
        r_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            self.acc_dtype,
        )
        cute.copy(atom_t2r, tmem_gate_tensor, r_gate)
        cute.copy(atom_t2r, tmem_up_tensor, r_up)
        return r_gate, r_up

    @cute.jit
    def _run_fc1_subtile(
        self,
        subtile_idx,
        tmem_gate_tensor: cute.Tensor,
        tmem_up_tensor: cute.Tensor,
        real_fc1_output: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        real_topk_scores: cute.Tensor,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        r2s_copy_atom: cute.CopyAtom,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        rmem_sf: cute.Tensor,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c_subtile_view: cute.Tensor,
        c_buffer_idx,
        c_pipeline,
    ) -> None:
        """MXFP8 fc1 subtile: LDTM gate/up, then GLU + E8M0 SF + fp8 store."""
        r_gate, r_up = self._fc1_subtile_load(tmem_gate_tensor, tmem_up_tensor)

        self._fc1_subtile_compute_store(
            subtile_idx=subtile_idx,
            r_gate=r_gate,
            r_up=r_up,
            real_fc1_output=real_fc1_output,
            real_fc1_output_sf=real_fc1_output_sf,
            real_topk_scores=real_topk_scores,
            work_tile_info=work_tile_info,
            smem_fc1_output_buffer=smem_fc1_output_buffer,
            tma_atom_fc1_output=tma_atom_fc1_output,
            r2s_copy_atom=r2s_copy_atom,
            warp_idx=warp_idx,
            tidx=tidx,
            alpha=alpha,
            norm_const=norm_const,
            rmem_sf=rmem_sf,
            smem_c_buffer=smem_c_buffer,
            tma_atom_c=tma_atom_c,
            gmem_c_subtile_view=gmem_c_subtile_view,
            c_buffer_idx=c_buffer_idx,
            c_pipeline=c_pipeline,
        )

    @cute.jit
    def _fc1_subtile_compute_store(
        self,
        subtile_idx,
        r_gate: cute.Tensor,
        r_up: cute.Tensor,
        real_fc1_output: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        real_topk_scores: cute.Tensor,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        r2s_copy_atom: cute.CopyAtom,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        rmem_sf: cute.Tensor,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c_subtile_view: cute.Tensor,
        c_buffer_idx,
        c_pipeline,
    ) -> None:
        """fc1 subtile from registers: (raw-C store) + GLU + E8M0 SF + fp8 store.

        Split from ``_run_fc1_subtile`` so the deferred-subtile path can run
        it after the accumulator release on gate/up held in registers."""
        if warp_idx == 0:
            iket.range_push("fc1_epilogue_subtile")

        r_layout = cute.make_layout((((Fc1GateUpInterleave,), 1),), stride=(((1,), 0),))

        # ── generate_c: store raw gate+up to GMEM C tensor via SMEM staging ──
        if cutlass.const_expr(self._generate_c):
            self._store_fc1_c_subtile(
                r_gate=r_gate,
                r_up=r_up,
                smem_c_buffer=smem_c_buffer,
                tma_atom_c=tma_atom_c,
                gmem_c_subtile_view=gmem_c_subtile_view,
                c_buffer_idx=c_buffer_idx,
                work_tile_info=work_tile_info,
                warp_idx=warp_idx,
                tidx=tidx,
                c_pipeline=c_pipeline,
            )

        if cutlass.const_expr(self.glu_clamp is not None):
            for i in cutlass.range_constexpr(cute.size(r_up)):
                r_gate[i] = cute.arch.fmin(r_gate[i], self.glu_clamp)
                r_up[i] = cute.arch.fmin(r_up[i], self.glu_clamp)
                r_up[i] = cute.arch.fmax(r_up[i], -self.glu_clamp)

        topk = None
        if cutlass.const_expr(self._apply_topk_in_fc1):
            thread_in_warp = tidx % cutlass.Int32(WarpThreadCount)
            token_in_tile = (
                work_tile_info.tile_m_idx * cutlass.Int32(self._epi_tile_m)
                + cutlass.Int32(warp_idx * WarpThreadCount)
                + thread_in_warp
            )
            topk = Float32(real_topk_scores[token_in_tile])

        swiglu = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        if cutlass.const_expr(self._act_func == "swiglu"):
            self._swiglu_act(swiglu, r_up, r_gate, topk)
        elif cutlass.const_expr(self._act_func == "situ"):
            self._situ_act(swiglu, r_up, r_gate, topk)

        c = cute.make_rmem_tensor(r_layout.shape, self.fc1_output_dtype)
        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            # Quantized hand-off: fp8 data + E8M0 block scale.
            qpvscale = quant_sfd_row(
                swiglu, c, norm_const, self._sf_vec_size, self.sf_dtype, self.fc1_output_dtype
            )
            if subtile_idx == 0:
                rmem_sf[0] = qpvscale
            elif subtile_idx == 1:
                rmem_sf[1] = qpvscale
            elif subtile_idx == 2:
                rmem_sf[2] = qpvscale
            elif subtile_idx == 3:
                rmem_sf[3] = qpvscale
        else:
            # Plain-data hand-off: direct cast to the fc1 output dtype (the
            # fc1_output workspace is reloaded as fc2's A operand).
            c.store(swiglu.load().to(self.fc1_output_dtype))

        thread_in_warp = tidx % WarpThreadCount
        if cutlass.const_expr(self._use_stg_fc1):
            # Direct STG.256 to GMEM — no SMEM staging or TMA store needed.
            token_in_tile = cutlass.Int32(warp_idx * EpilogueTileN) + thread_in_warp
            if token_in_tile < work_tile_info.valid_tokens_in_cta_tile:
                abs_token = (
                    work_tile_info.tile_m_idx * cutlass.Int32(self._epi_tile_m)
                    + cutlass.Int32(warp_idx * EpilogueTileN)
                    + thread_in_warp
                )
                # absolute column start (element index in the intermediate axis)
                col_elem = (
                    work_tile_info.tile_n_idx * cutlass.Int32(self._subtile_cnt) + subtile_idx
                ) * cutlass.Int32(Fc1GateUpInterleave)
                g_base = cute.local_tile(
                    real_fc1_output,
                    (1, 1, 1),
                    (abs_token, col_elem, cutlass.Int32(0)),
                )
                stg_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.fc1_output_dtype,
                    num_bits_per_copy=256,
                )
                # col_elem is always a multiple of Fc1GateUpInterleave=32 (FP8 elements)
                aligned_iter = cute.make_ptr(
                    self.fc1_output_dtype,
                    g_base.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                g_vec = cute.make_tensor(aligned_iter, cute.make_layout(Fc1GateUpInterleave))
                cute.copy(stg_atom, cute.coalesce(c), g_vec)
        else:
            sC_stage = cute.slice_(smem_fc1_output_buffer, (None, None, warp_idx))
            sC_thread_row = cute.local_tile(
                sC_stage, (1, Fc1GateUpInterleave), (thread_in_warp, subtile_idx)
            )
            cute.copy(r2s_copy_atom, cute.coalesce(c), cute.coalesce(sC_thread_row))

        if cutlass.const_expr(self._generate_c):
            if warp_idx == 0:
                c_pipeline.producer_acquire()
            c_store_bar = pipeline.NamedBarrier(
                barrier_id=self._c_store_bar_id(work_tile_info),
                num_threads=EpiWarpCount * WarpThreadCount,
            )
            c_store_bar.arrive_and_wait()

        if warp_idx == 0:
            iket.range_pop()

    @cute.jit
    def _subtile_fc2_tmem_tensor(
        self,
        tmem_acc_tensor: cute.Tensor,
        subtile_idx,
        warp_idx,
    ) -> cute.Tensor:
        """
        Per-warp TMEM view for one fc2 subtile (EpilogueTileN=32 cols).
        """
        base = tmem_acc_tensor.iterator
        warp_lane_off = warp_idx * WarpThreadCount
        subtile_col_off = subtile_idx * EpilogueTileN
        total = (warp_lane_off << 16) + subtile_col_off
        subtile_ptr = base + cute.assume(total, divby=16)
        return cute.make_tensor(
            subtile_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
        )

    @cute.jit
    def _advance_fc2_tmem_tensor(
        self,
        tmem_tensor: cute.Tensor,
        col_offset: int,
    ) -> cute.Tensor:
        """Advance the fc2 TMEM tensor by col_offset cols (mirrors _subtile_forward_tmem_tensor)."""
        new_ptr = tmem_tensor.iterator + cute.assume(col_offset, divby=16)
        return cute.make_tensor(
            new_ptr,
            _TmemTranspose16x32Core._tmem_layout(32, EpilogueTileN),
        )

    @cute.jit
    def _acc_pipeline_consumer_release(
        self,
        acc_pipeline,
        acc_consumer_state,
        is_release: bool,
    ) -> None:
        """Release the acc pipeline consumer."""
        if is_release:
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _fc2_subtile_cols(self, work_tile_info, subtile_idx):
        """(hidden_group, hidden_col_start) of one fc2 subtile of the current task tile."""
        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8
        hidden_group = work_tile_info.tile_n_idx * cutlass.Int32(fc2_subtile_cnt) + subtile_idx
        hidden_col_start = work_tile_info.tile_n_idx * cutlass.Int32(
            self._cta_tile_n
        ) + subtile_idx * cutlass.Int32(EpilogueTileN)
        return hidden_group, hidden_col_start

    @cute.jit
    def _fc2_subtile_load(self, tmem_subtile_tensor: cute.Tensor) -> cute.Tensor:
        """LDTM one fc2 subtile (EpilogueTileN f32 per thread) into registers."""
        r_acc_layout = cute.make_layout((((EpilogueTileN,), 1),), stride=(((1,), 0),))
        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            self.acc_dtype,
        )
        r_acc = cute.make_rmem_tensor(r_acc_layout.shape, self.acc_dtype)
        cute.copy(atom_t2r, tmem_subtile_tensor, r_acc)
        return r_acc

    @cute.jit
    def _fc2_lean_mxfp8_store(
        self,
        r_fp8: cute.Tensor,
        real_fc2_output: cute.Tensor,
        work_tile_info,
        hidden_group,
        token_row_in_cta,
    ) -> None:
        """Lean path: STG.256 of one thread's EpilogueTileN fp8 fc2 elements; the caller predicates."""
        fp8_dtype = self._combine_format.act_dtype
        g_fc2_output_tile = cute.local_tile(
            real_fc2_output,
            (self._epi_tile_m, EpilogueTileN, 1),
            (work_tile_info.tile_m_idx, hidden_group, 0),
        )
        g_fc2_slice = cute.slice_(g_fc2_output_tile, (None, None, 0))
        g_thread_row = cute.local_tile(
            g_fc2_slice,
            (1, EpilogueTileN),
            (token_row_in_cta, 0),
        )
        g_flat = cute.coalesce(g_thread_row)
        aligned_iter = cute.make_ptr(
            fp8_dtype,
            g_flat.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        stg_fp8_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            fp8_dtype,
            num_bits_per_copy=256,
        )
        r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(EpilogueTileN))
        cute.copy(
            stg_fp8_atom,
            r_fp8_flat,
            cute.make_tensor(aligned_iter, cute.make_layout(EpilogueTileN)),
        )

    @cute.jit
    def _run_fc2_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        real_fc2_output: cute.Tensor,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
        fc2_output_router=None,
        rmem_sf_fc2=None,
        smem_fc2_tma_buffer=None,
        tma_atom_fc2_output=None,
        fc2_tma_tile_router=None,
        smem_fc2_reduce_buffer=None,
    ) -> None:
        """fc2 subtile: LDTM + encode + STG."""
        if warp_idx == 0:
            iket.range_push("fc2_epi_subtile")

        hidden_col_start = work_tile_info.tile_n_idx * cutlass.Int32(
            self._cta_tile_n
        ) + subtile_idx * cutlass.Int32(EpilogueTileN)
        r_acc_layout = cute.make_layout((((EpilogueTileN,), 1),), stride=(((1,), 0),))
        r_acc = self._fc2_subtile_load(tmem_subtile_tensor)

        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        if token_row_in_cta < valid_tokens and hidden_col_start < valid_hidden:
            if cutlass.const_expr(
                token_comm_args is not None
                and not self._token_back_by_dispatch
                and self._combine_mxfp8
            ):
                # MegaMoE Form A, quantized combine:
                # 1. Quantize fp32 → fp8 + compute E8M0 block scale.
                # 2. STG fp8 data to peer's combine_output.
                # 3. Write E8M0 scale to local fc2_output_sf for token-back push.
                fp8_dtype = self._combine_format.act_dtype
                r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
                qpvscale = quant_sfd_row(
                    r_acc,
                    r_fp8,
                    1.0,
                    EpilogueTileN,
                    cutlass.Float8E8M0FNU,
                    fp8_dtype,
                )
                r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(EpilogueTileN))
                if cutlass.const_expr(self._fc2_use_ublk and token_comm_args is not None):
                    # Four 32-hidden MXFP8 subtiles form one contiguous 128B
                    # peer UBLK for this token.
                    subtile_in_ublk = subtile_idx % cutlass.Int32(self._fc2_ublk_subtiles)
                    ublk_iter_idx = subtile_idx // cutlass.Int32(self._fc2_ublk_subtiles)
                    stage_idx = ublk_iter_idx % cutlass.Int32(self._fc2_tma_stages)
                    smem_row = cute.slice_(smem_fc2_tma_buffer, (token_row_in_cta, None, stage_idx))
                    smem_chunk = cute.local_tile(
                        smem_row,
                        (EpilogueTileN,),
                        (subtile_in_ublk,),
                    )
                    sts_fp8_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        fp8_dtype,
                        num_bits_per_copy=256,
                    )
                    cute.copy(sts_fp8_atom, r_fp8_flat, smem_chunk)
                    if subtile_in_ublk == self._fc2_ublk_subtiles - 1:
                        dest_addr, dest_valid = fc2_output_router.get_data_addr(ublk_iter_idx)
                        if dest_valid != 0:
                            dest_fp8_ptr = cute.make_ptr(
                                fp8_dtype,
                                dest_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=32,
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cp_async_bulk_s2g(
                                dest_fp8_ptr,
                                smem_row.iterator,
                                cutlass.Int32(self._fc2_ublk_bytes),
                            )
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(self._fc2_tma_stages - 1, read=True)
                else:
                    # STG 32 fp8 elements = 256 bits in one shot.
                    dest_addr, dest_valid = fc2_output_router.get_data_addr(subtile_idx)
                    stg_fp8_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        fp8_dtype,
                        num_bits_per_copy=256,
                    )
                    if dest_valid != 0:
                        dest_fp8_ptr = cute.make_ptr(
                            fp8_dtype,
                            dest_addr,
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        cute.copy(
                            stg_fp8_atom,
                            r_fp8_flat,
                            cute.make_tensor(dest_fp8_ptr, cute.make_layout(EpilogueTileN)),
                        )
                # Buffer the E8M0 scale; the whole task tile's 8 scales are
                # flushed together by _stg_sf_fc2 (single stg.64 when aligned).
                self._write_sf_fc2_buffer(rmem_sf_fc2, subtile_idx, qpvscale)
            elif cutlass.const_expr(self._token_back_by_dispatch and self._combine_mxfp8):
                # MegaMoE token-back-by-dispatch + quantized combine:
                # Epilogue writes fp8 data to local pool; dispatch warps push
                # both data (fc2_output_workspace) and SF (fc2_output_sf) to peers.
                fp8_dtype = self._combine_format.act_dtype
                r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
                qpvscale = quant_sfd_row(
                    r_acc,
                    r_fp8,
                    1.0,
                    EpilogueTileN,
                    cutlass.Float8E8M0FNU,
                    fp8_dtype,
                )
                r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(EpilogueTileN))
                if cutlass.const_expr(self._fc2_use_tma):
                    stage_idx = subtile_idx % cutlass.Int32(self._fc2_tma_stages)
                    smem_row = cute.slice_(smem_fc2_tma_buffer, (token_row_in_cta, None, stage_idx))
                    sts_fp8_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        fp8_dtype,
                        num_bits_per_copy=256,
                    )
                    cute.copy(sts_fp8_atom, r_fp8_flat, smem_row)
                else:
                    # Local token-back pool STG uses affine Router addressing.
                    dest_addr, dest_valid = fc2_output_router.get_data_addr(subtile_idx)
                    stg_fp8_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        fp8_dtype,
                        num_bits_per_copy=256,
                    )
                    if dest_valid != 0:
                        aligned_fp8_iter = cute.make_ptr(
                            fp8_dtype,
                            dest_addr,
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        cute.copy(
                            stg_fp8_atom,
                            r_fp8_flat,
                            cute.make_tensor(
                                aligned_fp8_iter,
                                cute.make_layout(EpilogueTileN),
                            ),
                        )
                # Buffer the E8M0 scale; flushed together by _stg_sf_fc2 after
                # the subtile loop (single stg.64 when hidden-aligned).
                self._write_sf_fc2_buffer(rmem_sf_fc2, subtile_idx, qpvscale)
            elif cutlass.const_expr(token_comm_args is None and self._combine_mxfp8):
                # Lean path (no token-comm), quantized fc2 output
                fp8_dtype = self._combine_format.act_dtype
                r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
                qpvscale = quant_sfd_row(
                    r_acc,
                    r_fp8,
                    1.0,
                    EpilogueTileN,
                    cutlass.Float8E8M0FNU,
                    fp8_dtype,
                )
                stg_fp8_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    fp8_dtype,
                    num_bits_per_copy=256,
                )
                r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(EpilogueTileN))
                dest_addr, dest_valid = fc2_output_router.get_data_addr(subtile_idx)
                if dest_valid != 0:
                    aligned_iter = cute.make_ptr(
                        fp8_dtype,
                        dest_addr,
                        cute.AddressSpace.gmem,
                        assumed_align=32,
                    )
                    cute.copy(
                        stg_fp8_atom,
                        r_fp8_flat,
                        cute.make_tensor(aligned_iter, cute.make_layout(EpilogueTileN)),
                    )
                # Buffer the E8M0 scale; flushed together by _stg_sf_fc2 after
                # the subtile loop (single stg.64 when hidden-aligned).
                self._write_sf_fc2_buffer(rmem_sf_fc2, subtile_idx, qpvscale)
            else:
                # BF16 path: either stage a 128x32 tile for TMA/UBLK or use
                # a pair of 256-bit vector stores.
                r_bf16 = cute.make_rmem_tensor(r_acc_layout.shape, cutlass.BFloat16)
                r_bf16.store(r_acc.load().to(cutlass.BFloat16))
                r_bf16_flat = cute.make_tensor(r_bf16.iterator, cute.make_layout(EpilogueTileN))
                if cutlass.const_expr(self._fc2_reduce_coalesce):
                    # In-kernel reduce: stage the full bf16 token row token-major into
                    # SMEM (row start is 64 B aligned).
                    cute.autovec_copy(
                        r_bf16_flat,
                        cute.slice_(smem_fc2_reduce_buffer, (token_row_in_cta, None)),
                    )
                if cutlass.const_expr(self._fc2_use_ublk and token_comm_args is not None):
                    # Two 32-hidden BF16 subtiles form one contiguous 128B peer UBLK for this token.
                    subtile_in_ublk = subtile_idx % cutlass.Int32(self._fc2_ublk_subtiles)
                    ublk_iter_idx = subtile_idx // cutlass.Int32(self._fc2_ublk_subtiles)
                    stage_idx = ublk_iter_idx % cutlass.Int32(self._fc2_tma_stages)
                    smem_row = cute.slice_(smem_fc2_tma_buffer, (token_row_in_cta, None, stage_idx))
                    smem_chunk = cute.local_tile(
                        smem_row,
                        (EpilogueTileN,),
                        (subtile_in_ublk,),
                    )
                    cute.autovec_copy(r_bf16_flat, smem_chunk)
                    if subtile_in_ublk == self._fc2_ublk_subtiles - 1:
                        dest_addr, dest_valid = fc2_output_router.get_data_addr(ublk_iter_idx)
                        if dest_valid != 0:
                            dest_ptr = cute.make_ptr(
                                cutlass.BFloat16,
                                dest_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=32,
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cp_async_bulk_s2g(
                                dest_ptr,
                                smem_row.iterator,
                                cutlass.Int32(self._fc2_ublk_bytes),
                            )
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(self._fc2_tma_stages - 1, read=True)
                elif cutlass.const_expr(self._fc2_use_tma):
                    # Dispatch token-back path: TMA stores this staged tile to
                    # the rank-local FC2 pool after all 128 rows rendezvous.
                    stage_idx = subtile_idx % cutlass.Int32(self._fc2_tma_stages)
                    smem_row = cute.slice_(smem_fc2_tma_buffer, (token_row_in_cta, None, stage_idx))
                    cute.autovec_copy(r_bf16_flat, smem_row)
                else:
                    stg_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.BFloat16,
                        num_bits_per_copy=256,
                    )
                    if cutlass.const_expr(not self._fc2_reduce_coalesce):
                        for stg_half in cutlass.range_constexpr(EpilogueTileN // 16):
                            reg_view = cute.make_tensor(
                                r_bf16.iterator + stg_half * 16,
                                cute.make_layout(16),
                            )
                            stg_iter_idx = subtile_idx * cutlass.Int32(
                                EpilogueTileN // 16
                            ) + cutlass.Int32(stg_half)
                            dest_addr, dest_valid = fc2_output_router.get_data_addr(stg_iter_idx)
                            if dest_valid != 0:
                                dest_ptr = cute.make_ptr(
                                    cutlass.BFloat16,
                                    dest_addr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=32,
                                )
                                cute.copy(
                                    stg_atom,
                                    reg_view,
                                    cute.make_tensor(dest_ptr, cute.make_layout(16)),
                                )

        if cutlass.const_expr(self._fc2_use_tma):
            self._issue_fc2_tma_store(
                subtile_idx=subtile_idx,
                smem_fc2_tma_buffer=smem_fc2_tma_buffer,
                tma_atom_fc2_output=tma_atom_fc2_output,
                fc2_tma_tile_router=fc2_tma_tile_router,
                warp_idx=warp_idx,
            )

        # In-kernel reduce: the token-major SMEM tile is now populated; all 128 epi
        # threads re-partition hidden-major and issue coalesced peer red.add.
        if cutlass.const_expr(self._fc2_reduce_coalesce):
            self._issue_fc2_reduce_coalesced(
                subtile_idx=subtile_idx,
                smem_fc2_reduce_buffer=smem_fc2_reduce_buffer,
                warp_idx=warp_idx,
                tidx=tidx,
                fc2_output_router=fc2_output_router,
            )

        if warp_idx == 0:
            iket.range_pop()

    @cute.jit
    def _issue_fc2_tma_store(
        self,
        subtile_idx,
        smem_fc2_tma_buffer: cute.Tensor,
        tma_atom_fc2_output: cute.CopyAtom,
        fc2_tma_tile_router: Fc2TmaTileRouter,
        warp_idx: int,
    ) -> None:
        """Issue the FC2 bulk-tensor store for one subtile's staged SMEM tile."""
        stage_idx = subtile_idx % cutlass.Int32(self._fc2_tma_stages)
        smem_stage = cute.slice_(smem_fc2_tma_buffer, (None, None, stage_idx))
        gmem_subtile, tile_valid = fc2_tma_tile_router.get_data_tile(subtile_idx)
        bSG_s, bSG_g = cpasync.tma_partition(
            tma_atom_fc2_output,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_stage, 0, 2),
            cute.group_modes(gmem_subtile, 0, 2),
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        bar.arrive_and_wait()
        if warp_idx == 0:
            if tile_valid:
                cute.copy(tma_atom_fc2_output, bSG_s, bSG_g)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(self._fc2_tma_stages - 1, read=True)
        bar.arrive_and_wait()

    @cute.jit
    def _issue_fc2_reduce_coalesced(
        self,
        subtile_idx,
        smem_fc2_reduce_buffer: cute.Tensor,
        warp_idx: int,
        tidx,
        fc2_output_router: Fc2OutputRouter,
    ) -> None:
        """Coalesced peer red.add of one subtile's staged reduce tile."""
        redg_width: cutlass.Constexpr[int] = 4  # bf16 per red.v2.bf16x2
        lanes_per_token: cutlass.Constexpr[int] = EpilogueTileN // redg_width  # 8
        tokens_per_warp: cutlass.Constexpr[int] = WarpThreadCount // lanes_per_token  # 4
        tokens_per_pass: cutlass.Constexpr[int] = tokens_per_warp * EpiWarpCount  # 16
        passes: cutlass.Constexpr[int] = self._epi_tile_m // tokens_per_pass  # 8

        # Rendezvous: make every thread's staged SMEM row visible before the reads.
        bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=EpiWarpCount * WarpThreadCount,
        )
        bar.arrive_and_wait()

        thread_in_warp = tidx % WarpThreadCount
        chunk = thread_in_warp % cutlass.Int32(lanes_per_token)  # 0..lanes_per_token-1
        epi_tid = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        token_group = epi_tid // cutlass.Int32(lanes_per_token)  # 0..15
        group_first_lane = thread_in_warp - chunk
        hidden_in_tile = subtile_idx * cutlass.Int32(EpilogueTileN) + chunk * cutlass.Int32(
            redg_width
        )
        hidden = fc2_output_router.hidden_base_this_cta_tile + hidden_in_tile
        hidden_valid = cutlass.Int32(0)
        if hidden < fc2_output_router.valid_hidden:
            hidden_valid = cutlass.Int32(1)

        # Every lane prefetched one distinct token row for its 8-lane group.
        # Broadcast the selected pass's 64-bit row base as two i32 values.
        prefetched_row = Int64(fc2_output_router.row_base)
        prefetched_row_lo = cutlass.Int32(prefetched_row & Int64(0xFFFFFFFF))
        prefetched_row_hi = cutlass.Int32((prefetched_row >> Int64(32)) & Int64(0xFFFFFFFF))
        prefetched_valid = cutlass.Int32(fc2_output_router.row_valid)

        for p in cutlass.range_constexpr(passes):
            token_in_cta = cutlass.Int32(p * tokens_per_pass) + token_group
            source_lane = group_first_lane + cutlass.Int32(p)
            row_lo = cutlass.Int32(
                cute.arch.shuffle_sync(
                    prefetched_row_lo,
                    source_lane,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
            )
            row_hi = cutlass.Int32(
                cute.arch.shuffle_sync(
                    prefetched_row_hi,
                    source_lane,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
            )
            dest_valid = (
                cutlass.Int32(
                    cute.arch.shuffle_sync(
                        prefetched_valid,
                        source_lane,
                        mask=0xFFFFFFFF,
                        mask_and_clamp=31,
                    )
                )
                & hidden_valid
            )
            if dest_valid != 0:
                dest_row = ((Int64(row_hi) & Int64(0xFFFFFFFF)) << Int64(32)) | (
                    Int64(row_lo) & Int64(0xFFFFFFFF)
                )
                dest_addr = dest_row + Int64(hidden) * Int64(fc2_output_router.wire_bytes)
                # LDS this lane's 4 staged bf16 (contiguous hidden chunk).
                smem_chunk_ptr = cute.slice_(
                    smem_fc2_reduce_buffer, (token_in_cta, None)
                ).iterator + chunk * cutlass.Int32(redg_width)
                r4 = cute.make_rmem_tensor((redg_width,), cutlass.BFloat16)
                cute.autovec_copy(
                    cute.make_tensor(smem_chunk_ptr, cute.make_layout(redg_width)), r4
                )
                r4_u32 = cute.recast_tensor(r4, cutlass.Uint32)  # 2 u32 = 4 bf16
                dest_ptr = cute.make_ptr(
                    cutlass.BFloat16,
                    dest_addr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                _red_add_relaxed_sys_v2_bf16x2(
                    dest_ptr,
                    cutlass.Uint32(r4_u32[0]),
                    cutlass.Uint32(r4_u32[1]),
                )
        bar.arrive_and_wait()

    @cute.jit
    def _write_sf_fc2_buffer(self, rmem_sf_fc2, subtile_idx, qpvscale) -> None:
        """Scatter one subtile's E8M0 scale into the per-tile SF buffer."""
        for j in cutlass.range_constexpr(self._cta_tile_n // EpilogueTileN):
            if subtile_idx == cutlass.Int32(j):
                rmem_sf_fc2[j] = qpvscale

    @cute.jit
    def _stg_sf_row_group(
        self,
        real_sf: cute.Tensor,
        token_idx,
        sf_col,
        rmem_src_iter,
        count: int,
        assumed_align: int,
    ) -> None:
        """Convert ``count`` scales to ``sf_dtype`` and store them contiguously at (token_idx, sf_col)."""
        sf_base = cute.local_tile(real_sf, (1, 1, 1), (token_idx, sf_col, cutlass.Int32(0)))
        sf_ptr = cute.make_ptr(
            self.sf_dtype,
            sf_base.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        r_src = cute.make_tensor(rmem_src_iter, cute.make_layout(count))
        r_sf = cute.make_rmem_tensor(cute.make_layout(count).shape, self.sf_dtype)
        r_sf.store(r_src.load().to(self.sf_dtype))
        cute.autovec_copy(r_sf, cute.make_tensor(sf_ptr, cute.make_layout(count)))

    @cute.jit
    def _stg_sf_fc2_blocked(
        self,
        rmem_sf_fc2: cute.Tensor,
        real_fc2_output_sf: cute.Tensor,
        work_tile_info,
        tidx,
    ) -> None:
        """Store one task tile's fc2 E8M0 scales into the blocked SF plane, one token row per thread."""
        token_idx = (
            work_tile_info.cumulative_sf_physical_row
            + work_tile_info.tile_m_idx * cutlass.Int32(self._epi_tile_m)
            + tidx
        )
        if tidx < work_tile_info.valid_tokens_in_cta_tile:
            if cutlass.const_expr(self._cta_tile_n < 4 * self._sf_vec_size):
                self._stg_sf_row_group(
                    real_fc2_output_sf,
                    token_idx,
                    work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n),
                    rmem_sf_fc2.iterator,
                    count=self._cta_tile_n // self._sf_vec_size,
                    assumed_align=2,
                )
            for g in cutlass.range_constexpr(self._cta_tile_n // (4 * self._sf_vec_size)):
                self._stg_sf_row_group(
                    real_fc2_output_sf,
                    token_idx,
                    work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
                    + cutlass.Int32(g * 4 * self._sf_vec_size),
                    rmem_sf_fc2.iterator + 4 * g,
                    count=4,
                    assumed_align=4,
                )

    @cute.jit
    def _stg_sf_fc2(
        self,
        rmem_sf_fc2: cute.Tensor,
        fc2_output_router: Fc2OutputRouter,
    ) -> None:
        """Flush a task tile's fc2 E8M0 scales to the local ``fc2_output_sf``."""
        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN
        sf_byte_addr, sf_valid = fc2_output_router.get_sf_addr(cutlass.Int32(0))
        if sf_valid != 0:
            if cutlass.const_expr(self._fc2_sf_batch8):
                stg_e8m0x8_from_f32(
                    sf_byte_addr,
                    rmem_sf_fc2[0],
                    rmem_sf_fc2[1],
                    rmem_sf_fc2[2],
                    rmem_sf_fc2[3],
                    rmem_sf_fc2[4],
                    rmem_sf_fc2[5],
                    rmem_sf_fc2[6],
                    rmem_sf_fc2[7],
                )
            else:
                for j in cutlass.range_constexpr(fc2_subtile_cnt):
                    hidden_block = fc2_output_router.sf_hidden_block_base + cutlass.Int32(j)
                    if hidden_block * cutlass.Int32(EpilogueTileN) < fc2_output_router.valid_hidden:
                        stg_e8m0_from_f32(
                            sf_byte_addr + Int64(j) * Int64(fc2_output_router.sf_bytes),
                            rmem_sf_fc2[j],
                        )

    @cute.jit
    def _stg_sf_fc2_flat(
        self,
        rmem_sf_fc2: cute.Tensor,
        real_fc2_output_sf: cute.Tensor,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
    ) -> None:
        """Flush FC2 scales through the legacy flat-SF address path.

        Tile variants that do not construct an FC2 data router, such as the
        drain-first B-reuse epilogue, use this fallback.
        """
        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        if token_row_in_cta < work_tile_info.valid_tokens_in_cta_tile:
            pool_token_global = (
                work_tile_info.cumulative_data_physical_row
                + work_tile_info.tile_m_idx * cutlass.Int32(self._epi_tile_m)
                + token_row_in_cta
            )
            hidden_group_base = work_tile_info.tile_n_idx * cutlass.Int32(fc2_subtile_cnt)
            sf_byte_addr = (
                real_fc2_output_sf.iterator.toint()
                + Int64(pool_token_global) * Int64(real_fc2_output_sf.stride[0])
                + Int64(hidden_group_base)
            )
            if cutlass.const_expr(self._fc2_sf_batch8):
                stg_e8m0x8_from_f32(
                    sf_byte_addr,
                    rmem_sf_fc2[0],
                    rmem_sf_fc2[1],
                    rmem_sf_fc2[2],
                    rmem_sf_fc2[3],
                    rmem_sf_fc2[4],
                    rmem_sf_fc2[5],
                    rmem_sf_fc2[6],
                    rmem_sf_fc2[7],
                )
            else:
                for j in cutlass.range_constexpr(fc2_subtile_cnt):
                    block_hidden_start = work_tile_info.tile_n_idx * cutlass.Int32(
                        self._cta_tile_n
                    ) + cutlass.Int32(j * EpilogueTileN)
                    if block_hidden_start < valid_hidden:
                        stg_e8m0_from_f32(sf_byte_addr + Int64(j), rmem_sf_fc2[j])

    @cute.jit
    def _make_fc2_output_router(
        self,
        *,
        work_tile_info,
        real_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
        gmem_fc2_output_sf=None,
    ) -> Fc2OutputRouter:
        """Build one data/SF router for this FC2 task tile and epilogue thread."""
        epi_tid = cutlass.Int32(cutlass.Int32(warp_idx * WarpThreadCount) + tidx % WarpThreadCount)
        hidden_tile_start = work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
        local_token_base = work_tile_info.tile_m_idx * cutlass.Int32(self._epi_tile_m)
        pool_token_base = work_tile_info.cumulative_data_physical_row + local_token_base

        output = None
        metadata = None
        peer_rank_ptr_mapper = None
        local_data_base_addr = real_fc2_output.iterator.toint()
        local_data_row_stride = real_fc2_output.stride[0]
        if cutlass.const_expr(token_comm_args is not None):
            if cutlass.const_expr(not self._token_back_by_dispatch):
                output = token_comm_args.combine_output
                metadata = cute.recast_tensor(
                    token_comm_args.token_src_metadata,
                    cutlass.Uint32,
                )
                peer_rank_ptr_mapper = token_comm_args.peer_rank_ptr_mapper
                local_data_base_addr = None
                local_data_row_stride = None

        sf_base_addr = None
        sf_token_base = None
        sf_row_stride = None
        if cutlass.const_expr(self._combine_mxfp8):
            if cutlass.const_expr(token_comm_args is not None):
                if cutlass.const_expr(token_comm_args.fc2_output_sf is None):
                    raise ValueError("MXFP8 token communication requires fc2_output_sf.")
                sf_base_addr = token_comm_args.fc2_output_sf.iterator.toint()
                sf_token_base = pool_token_base
                sf_row_stride = self._fc2_sf_block_pad
            else:
                if cutlass.const_expr(gmem_fc2_output_sf is None):
                    raise ValueError("MXFP8 FC2 output requires an SF tensor.")
                sf_base_addr = gmem_fc2_output_sf.iterator.toint()
                sf_token_base = pool_token_base
                sf_row_stride = gmem_fc2_output_sf.stride[0]

        output_mapping = (
            self._fc2_redg_store_out_mapping
            if self._fc2_reduce_coalesce
            else self._fc2_store_out_mapping
        )
        return Fc2OutputRouter(
            output_mapping=output_mapping,
            pool_token_base=pool_token_base,
            hidden_base_this_cta_tile=hidden_tile_start,
            valid_tokens_this_cta_tile=work_tile_info.valid_tokens_in_cta_tile,
            valid_hidden=valid_hidden,
            epi_tid=epi_tid,
            wire_bytes=self._fc2_wire_bytes,
            output=output,
            metadata=metadata,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            local_data_base_addr=local_data_base_addr,
            local_data_token_base=local_token_base,
            local_data_row_stride=local_data_row_stride,
            reduce_topk_in_kernel=self._fc2_reduce_coalesce,
            prefetch_data_row=(not self._fc2_reduce_coalesce and not self._fc2_use_tma),
            sf_mapping=(self._fc2_sf_store_out_mapping if self._combine_mxfp8 else None),
            sf_base_addr=sf_base_addr,
            sf_token_base=sf_token_base,
            sf_row_stride=sf_row_stride,
            sf_hidden_block_base=(
                (work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n // EpilogueTileN))
                if self._combine_mxfp8
                else None
            ),
        ).prefetch()

    @cute.jit
    def _run_fc2_task_tile(
        self,
        work_tile_info,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
        gmem_fc2_output_sf=None,
        gmem_fc2_output_sf_blocked=None,
        smem_fc2_tma_buffer=None,
        tma_atom_fc2_output=None,
        gmem_fc2_tma_output=None,
        smem_fc2_reduce_buffer=None,
        acc_wait: bool = True,
        acc_release: bool = True,
    ) -> None:
        """fc2 task-tile body for one 128-row tile (here: LDTM + quantize + store per subtile).

        Overrides must open with ``_begin_fc2_task_tile``, release the accumulator exactly once
        after the last TMEM read, and close with ``_finish_fc2_task_tile``."""
        real_fc2_output = self._begin_fc2_task_tile(
            work_tile_info=work_tile_info,
            acc_pipeline=acc_pipeline,
            acc_consumer_state=acc_consumer_state,
            sched_ext=sched_ext,
            gmem_fc2_output=gmem_fc2_output,
            warp_idx=warp_idx,
            acc_wait=acc_wait,
        )

        fc2_output_router = self._make_fc2_output_router(
            work_tile_info=work_tile_info,
            real_fc2_output=real_fc2_output,
            valid_hidden=valid_hidden,
            warp_idx=warp_idx,
            tidx=tidx,
            token_comm_args=token_comm_args,
            gmem_fc2_output_sf=gmem_fc2_output_sf,
        )
        fc2_tma_tile_router = None
        if cutlass.const_expr(self._fc2_use_tma):
            fc2_tma_tile_router = Fc2TmaTileRouter(
                output=gmem_fc2_tma_output,
                pool_token_base=(
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._epi_tile_m)
                ),
                hidden_base_this_cta_tile=(
                    work_tile_info.tile_n_idx * cutlass.Int32(self._cta_tile_n)
                ),
                valid_tokens_this_cta_tile=(work_tile_info.valid_tokens_in_cta_tile),
                cta_token_tile_size=self._epi_tile_m,
            )

        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8

        tmem_t = self._subtile_fc2_tmem_tensor(
            tmem_acc_tensor,
            cutlass.Int32(0),
            warp_idx,
        )

        tmem_forward_cols = EpilogueTileN

        # Quantized combine: buffer the per-subtile E8M0 scales and flush them in
        # one stg.64 after the loop (see _stg_sf_fc2).  Indexed by subtile_idx, so
        # the reversed odd-turn walk fills the same slots.  Both the token-comm
        # combine plane and the lean stand-alone fc2_output_sf plane need it.
        if cutlass.const_expr(self._combine_mxfp8):
            layout_sf_fc2 = cute.make_layout(fc2_subtile_cnt)
            rmem_sf_fc2 = cute.make_rmem_tensor(layout_sf_fc2.shape, self.acc_dtype)
        else:
            rmem_sf_fc2 = None

        for i in cutlass.range(0, fc2_subtile_cnt, 1, unroll=1):
            subtile_idx = cutlass.Int32(i)

            self._run_fc2_subtile(
                subtile_idx=subtile_idx,
                tmem_subtile_tensor=tmem_t,
                real_fc2_output=real_fc2_output,
                work_tile_info=work_tile_info,
                valid_hidden=valid_hidden,
                warp_idx=warp_idx,
                tidx=tidx,
                token_comm_args=token_comm_args,
                fc2_output_router=fc2_output_router,
                rmem_sf_fc2=rmem_sf_fc2,
                smem_fc2_tma_buffer=smem_fc2_tma_buffer,
                tma_atom_fc2_output=tma_atom_fc2_output,
                fc2_tma_tile_router=fc2_tma_tile_router,
                smem_fc2_reduce_buffer=smem_fc2_reduce_buffer,
            )

            tmem_t = self._advance_fc2_tmem_tensor(tmem_t, tmem_forward_cols)

        self._acc_pipeline_consumer_release(acc_pipeline, acc_consumer_state, acc_release)

        self._finish_fc2_task_tile(
            rmem_sf_fc2=rmem_sf_fc2,
            work_tile_info=work_tile_info,
            valid_hidden=valid_hidden,
            warp_idx=warp_idx,
            tidx=tidx,
            token_comm_args=token_comm_args,
            fc2_output_router=fc2_output_router,
            gmem_fc2_output_sf=gmem_fc2_output_sf,
            gmem_fc2_output_sf_blocked=gmem_fc2_output_sf_blocked,
        )

    @cute.jit
    def _begin_fc2_task_tile(
        self,
        work_tile_info,
        acc_pipeline,
        acc_consumer_state,
        sched_ext,
        gmem_fc2_output: cute.Tensor,
        warp_idx: int,
        acc_wait: bool,
    ) -> cute.Tensor:
        """Open one fc2 task tile: resolve the output tile, wait on the accumulator, push profiling."""
        real_fc2_output, _ = sched_ext.get_gmem_tensor(
            "d",
            gmem_fc2_output,
            work_tile_info,
        )
        if cutlass.const_expr(acc_wait):
            acc_pipeline.consumer_wait(acc_consumer_state)
        if warp_idx == 0:
            iket.range_push("fc2_epi_tile")
        return real_fc2_output

    @cute.jit
    def _finish_fc2_task_tile(
        self,
        rmem_sf_fc2,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
        fc2_output_router=None,
        gmem_fc2_output_sf=None,
        gmem_fc2_output_sf_blocked=None,
    ) -> None:
        """Close one fc2 task tile: flush the buffered E8M0 scales and pop the profiling range."""
        # Flush the buffered E8M0 scales
        if cutlass.const_expr(self._combine_mxfp8 and token_comm_args is not None):
            self._stg_sf_fc2(
                rmem_sf_fc2=rmem_sf_fc2,
                fc2_output_router=fc2_output_router,
            )
        elif cutlass.const_expr(self._combine_mxfp8 and gmem_fc2_output_sf_blocked is not None):
            self._stg_sf_fc2_blocked(
                rmem_sf_fc2=rmem_sf_fc2,
                real_fc2_output_sf=gmem_fc2_output_sf_blocked,
                work_tile_info=work_tile_info,
                tidx=tidx,
            )
        elif cutlass.const_expr(self._combine_mxfp8 and gmem_fc2_output_sf is not None):
            if cutlass.const_expr(fc2_output_router is not None):
                self._stg_sf_fc2(
                    rmem_sf_fc2=rmem_sf_fc2,
                    fc2_output_router=fc2_output_router,
                )
            else:
                self._stg_sf_fc2_flat(
                    rmem_sf_fc2=rmem_sf_fc2,
                    real_fc2_output_sf=gmem_fc2_output_sf,
                    work_tile_info=work_tile_info,
                    valid_hidden=valid_hidden,
                    warp_idx=warp_idx,
                    tidx=tidx,
                )

        if warp_idx == 0:
            iket.range_pop()

    @cute.jit
    def _stg_sf_fc1(
        self,
        rmem_sf_f32: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        work_tile_info,
        tidx,
    ) -> None:
        """Compute gmem SF tile coords and store fc1 scale factors to gmem."""
        sf_idx = work_tile_info.tile_n_idx
        token_idx = work_tile_info.tile_m_idx * self._epi_tile_m + tidx
        if tidx < work_tile_info.valid_tokens_in_cta_tile:
            # Store exactly subtile_cnt scales; a fixed 4-byte store would clobber the neighbor N-tile.
            self._stg_sf_row_group(
                real_fc1_output_sf,
                token_idx,
                sf_idx * cutlass.Int32(self._cta_tile_n // 2),
                rmem_sf_f32.iterator,
                count=self._subtile_cnt,
                assumed_align=(
                    4 if self._subtile_cnt % 4 == 0 else (2 if self._subtile_cnt % 2 == 0 else 1)
                ),
            )

    @staticmethod
    @cute.jit
    def _drain_bulk_stores() -> None:
        """Commit and wait for all in-flight bulk stores, then make them GPU-visible."""
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.fence_proxy("async")
        cute.arch.fence_acq_rel_gpu()

    def _get_task_tile(self, work_tile_info, tmem_acc_tensor, acc_stage_index, warp_idx, tidx):
        """Entries ``run()`` consumes for one work tile (here: one, the whole 128-row tile).

        Overrides return ``(tile_info, tmem_acc_stage_tensor, warp_idx, tidx, acc_wait,
        acc_release)`` entries that wait on and release the accumulator stage exactly once."""
        return [
            (
                work_tile_info,
                tmem_acc_tensor[(None, None, None, acc_stage_index)],
                warp_idx,
                tidx,
                True,
                True,
            )
        ]

    @cute.jit
    def run(
        self,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        sched_consumer,
        sched_ext,
        smem_fc1_output_buffer: Optional[cute.Tensor],
        tma_atom_fc1_output: cute.CopyAtom,
        gmem_fc1_output: cute.Tensor,
        gmem_topk_scores: cute.Tensor,
        gmem_fc2_output: cute.Tensor,
        gmem_fc1_done_counter: cute.Tensor,
        warp_idx: int,
        tidx,
        gmem_fc1_output_sf: Optional[cute.Tensor] = None,
        alpha=None,
        norm_const=None,
        token_comm_args=None,
        gmem_fc2_output_sf: Optional[cute.Tensor] = None,
        gmem_fc2_output_sf_blocked: Optional[cute.Tensor] = None,
        smem_c_buffer: cute.Tensor = None,
        tma_atom_c: cute.CopyAtom = None,
        gmem_c: cute.Tensor = None,
        smem_fc2_tma_buffer: Optional[cute.Tensor] = None,
        tma_atom_fc2_output: cute.CopyAtom = None,
        gmem_fc2_tma_output: cute.Tensor = None,
        smem_fc2_reduce_buffer: Optional[cute.Tensor] = None,
    ) -> None:
        """
        Run the full fc1+fc2-fused epilogue task-tile loop.
        """
        assert self._use_stg_fc1 or smem_fc1_output_buffer is not None, (
            "smem_fc1_output_buffer=None requires use_stg_fc1=True (the TMA "
            "fc1-output store path consumes the sD staging buffer)"
        )
        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._num_acc_pipeline_stages
        )

        if cutlass.const_expr(self._generate_c):
            _c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=Fc1CTMAStages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    EpiWarpCount * WarpThreadCount,
                ),
            )
        else:
            _c_pipeline = None

        task_tile_boundary_bar = pipeline.NamedBarrier(
            barrier_id=self._epilog_sync_bar_id,
            num_threads=32 * len(self._epilogue_warp_ids),
        )

        valid_hidden = cutlass.Int32(gmem_fc2_output.shape[1])

        work_tile_info = sched_consumer.consume_work()

        flag_tracker = GpuReleaseFlagBatchTracker(
            flag_address=Int64(0),
            accumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            thread_idx=tidx % (len(self._epilogue_warp_ids) * WarpThreadCount),
        )

        while work_tile_info.is_valid_tile:
            acc_stage_index = acc_consumer_state.index

            # ``_get_task_tile`` decides how the tile is consumed; the real tile feeds the fc1_done slot.
            for (
                half_tile_info,
                tmem_acc_stage_tensor,
                warpgroup_warp_idx,
                warpgroup_tidx,
                _acc_wait,
                _acc_release,
            ) in self._get_task_tile(
                work_tile_info, tmem_acc_tensor, acc_stage_index, warp_idx, tidx
            ):
                # fc2 keyword bundle, built outside the dynamic phase branch.
                _fc2_kwargs = dict(
                    work_tile_info=half_tile_info,
                    tmem_acc_tensor=tmem_acc_stage_tensor,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    sched_ext=sched_ext,
                    gmem_fc2_output=gmem_fc2_output,
                    valid_hidden=valid_hidden,
                    warp_idx=warpgroup_warp_idx,
                    tidx=warpgroup_tidx,
                    gmem_fc2_output_sf=gmem_fc2_output_sf,
                    gmem_fc2_output_sf_blocked=gmem_fc2_output_sf_blocked,
                    acc_wait=_acc_wait,
                    acc_release=_acc_release,
                )
                if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                    if cutlass.const_expr(self._generate_c):
                        _smem_c_buf = smem_c_buffer
                        _tma_atom_c = tma_atom_c
                        _gmem_c = gmem_c
                    else:
                        _smem_c_buf = smem_fc1_output_buffer
                        _tma_atom_c = tma_atom_fc1_output
                        _gmem_c = gmem_fc1_output
                    self._run_fc1_task_tile(
                        work_tile_info=half_tile_info,
                        tmem_acc_tensor=tmem_acc_stage_tensor,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        smem_fc1_output_buffer=smem_fc1_output_buffer,
                        tma_atom_fc1_output=tma_atom_fc1_output,
                        sched_ext=sched_ext,
                        gmem_fc1_output=gmem_fc1_output,
                        gmem_fc1_output_sf=gmem_fc1_output_sf,
                        gmem_topk_scores=gmem_topk_scores,
                        warp_idx=warpgroup_warp_idx,
                        tidx=warpgroup_tidx,
                        alpha=alpha,
                        norm_const=norm_const,
                        smem_c_buffer=_smem_c_buf,
                        tma_atom_c=_tma_atom_c,
                        gmem_c=_gmem_c,
                        c_pipeline=_c_pipeline,
                        acc_wait=_acc_wait,
                        acc_release=_acc_release,
                    )
                else:
                    self._run_fc2_task_tile(
                        token_comm_args=token_comm_args,
                        smem_fc2_tma_buffer=smem_fc2_tma_buffer,
                        tma_atom_fc2_output=tma_atom_fc2_output,
                        gmem_fc2_tma_output=gmem_fc2_tma_output,
                        smem_fc2_reduce_buffer=smem_fc2_reduce_buffer,
                        **_fc2_kwargs,
                    )

            acc_consumer_state.advance()

            cur_was_linear1 = work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            # The cumulative count is in cluster-tile units, so scale it by slots per cluster
            # tile. The kernel's fc2 consumer computes the same slot.
            cur_fc1_counter_slot = work_tile_info.cumulative_token_block_count * cutlass.Int32(
                self._cluster_m // self._fc1_done_slot_m
            ) + work_tile_info.tile_m_idx // cutlass.Int32(self._fc1_done_slot_m)
            cur_fc2_expert_idx = work_tile_info.expert_idx

            work_tile_info = sched_consumer.consume_work()

            # Drain only the phases that issued asynchronous bulk stores.
            if cutlass.const_expr(self._fc1_bulk_store and self._fc2_bulk_store):
                GluMxfp8Epilogue._drain_bulk_stores()
            elif cutlass.const_expr(self._fc1_bulk_store):
                if cur_was_linear1:
                    GluMxfp8Epilogue._drain_bulk_stores()
            elif cutlass.const_expr(self._fc2_bulk_store):
                if not cur_was_linear1:
                    GluMxfp8Epilogue._drain_bulk_stores()

            task_tile_boundary_bar.arrive_and_wait()

            if cur_was_linear1:
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc1_batch,
                    (gmem_fc1_done_counter.iterator + cur_fc1_counter_slot).toint(),
                )
            else:
                _fire_fc2_counter: cutlass.Constexpr = (
                    self._token_back_by_dispatch or self._combine_mxfp8
                ) and token_comm_args is not None
                if cutlass.const_expr(_fire_fc2_counter):
                    # Fence before (deferred) counter release: make the fc2
                    # pool-output STG writes device-visible.
                    cute.arch.fence_acq_rel_gpu()
                    fc2_flag_addr = (
                        token_comm_args.fc2_done_counter.iterator + cur_fc2_expert_idx
                    ).toint()
                else:
                    fc2_flag_addr = Int64(0)
                no_fire: cutlass.Constexpr = not _fire_fc2_counter
                flag_tracker = flag_tracker.accumulate(
                    work_tile_info.phase,
                    self._epi_fc2_batch,
                    fc2_flag_addr,
                    no_fire,
                )

        flag_tracker.fire()
