"""Dedup FC2 epilogue: SMEM-staged subtiles pushed with one ``cp.async.bulk`` per token. ``dst_token_cnt == 1``
rows go straight to the source rank's ``pre_reduced[src_token][my_rank]``, others into the local order buffer
``[T, R, chunks_per_row, K, chunk]`` for the combine warps. Top-k scores are applied at FC1 only."""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from .....api import ImplDesc, ProblemDesc
from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferDevice
from .....communication.token_protocol import TokenSrcMetadata
from .....helpers.iket_compat import iket
from .....helpers.ptx_helpers import (
    cp_async_bulk_s2g,
    cp_reduce_async_bulk_add_bf16_s2g,
)
from ....function_mapping import FunctionMapping
from ....schedulers import SwapAbFc12WorkTileInfo
from .block_scaled_swap_ab_fc12_epilogue import (
    GatedActEpilogueArgs,
    SwapABFc2Epilogue,
    SwapABGatedActEpilogue,
    make_fc2_ublk_process_pipeline,
)


@dataclasses.dataclass(frozen=True)
class GatedActEpilogueArgsDedup(GatedActEpilogueArgs):
    """Base epilogue args plus the pre-reduced (T, R, H) buffer and this rank's index (the kernel's host argument,
    which selects the ``pre_reduced[src_token][local_rank]`` slot of single-copy rows)."""

    fc2_remote_output: Optional[cute.Tensor] = None
    local_rank: Optional[Int32] = None


@dataclasses.dataclass(frozen=True)
class Fc2DedupBulkRouter:
    """Per work tile: the bulk-copy destination of every token this thread owns. Order-buffer slot =
    ``(((src_token * R + src_rank) * chunks_per_row + hidden_chunk) * K + dst_order_id) * chunk``; single-copy
    tokens map to ``pre_reduced[src_token][slot][hidden_base]`` (slot = my rank, 0 under in-kernel REDG)."""

    metadata: cute.Tensor  # domain-offset to the tile's first pool row
    order_base: Int64
    remote_output: cute.Tensor  # local pre-reduced buffer (T, R, H); (T, 1, H) view of the output under REDG
    peer_rank_ptr_mapper: SymmetricBufferDevice
    local_rank: Int32
    hidden_base: Int32  # first hidden element of this CTA tile
    valid_tokens_this_cta_tile: Int32
    valid_hidden_this_cta_tile: Int32
    world_size: int
    topk: int
    chunk_bytes: int
    chunks_per_row: int
    element_bytes: int
    reduce_topk_in_kernel: bool
    mapping: FunctionMapping  # per-issue (epi_tid, iter_idx) -> token_in_cta_tile
    epi_tid: Int32
    dst_ptrs: Optional[cute.Tensor] = None  # i64 x copy_iters
    valid: Optional[cute.Tensor] = None  # i32 x copy_iters: 0 skip, 1 bulk copy, 2 bulk reduce-add
    copy_iters: int = 0

    @cute.jit
    def prefetch(self) -> "Fc2DedupBulkRouter":
        copy_iters: cutlass.Constexpr[int] = self.mapping.domain.axis_size("iter_idx")
        valid = cute.make_rmem_tensor((copy_iters,), Int32)
        dst_ptrs = cute.make_rmem_tensor((copy_iters,), Int64)
        hidden_base_bytes = Int64(self.hidden_base) * Int64(self.element_bytes)
        chunk_index = hidden_base_bytes // Int64(self.chunk_bytes)
        byte_in_chunk = hidden_base_bytes - chunk_index * Int64(self.chunk_bytes)
        remote_slot = Int32(0) if cutlass.const_expr(self.reduce_topk_in_kernel) else self.local_rank
        for iter_idx in cutlass.range_constexpr(copy_iters):
            token_in_tile = Int32(self.mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)["token_in_cta_tile"])
            valid[iter_idx] = Int32(0)
            dst_ptrs[iter_idx] = Int64(0)
            if (token_in_tile < self.valid_tokens_this_cta_tile) and (self.valid_hidden_this_cta_tile > Int32(0)):
                md = TokenSrcMetadata.load(
                    self.metadata.iterator.toint() + Int64(token_in_tile) * Int64(TokenSrcMetadata.nbytes)
                )
                if md.dst_token_cnt == Int32(1):
                    # Single copy on this rank: push straight to the source rank's pre-reduced row.
                    local_row = cute.domain_offset(
                        (Int64(md.src_token), remote_slot, self.hidden_base), self.remote_output
                    ).iterator.toint()
                    dst_ptrs[iter_idx] = self.peer_rank_ptr_mapper.map(local_row, md.src_rank)
                    valid[iter_idx] = Int32(2) if cutlass.const_expr(self.reduce_topk_in_kernel) else Int32(1)
                else:
                    block_index = Int64(md.src_token) * Int64(self.world_size) + Int64(md.src_rank)
                    slot_index = (block_index * Int64(self.chunks_per_row) + chunk_index) * Int64(self.topk) + Int64(
                        md.dst_order_id
                    )
                    dst_ptrs[iter_idx] = self.order_base + slot_index * Int64(self.chunk_bytes) + byte_in_chunk
                    valid[iter_idx] = Int32(1)
        return dataclasses.replace(self, dst_ptrs=dst_ptrs, valid=valid, copy_iters=copy_iters)

    @cute.jit
    def get_data_dst(self, iter_idx: int) -> Tuple[cute.Pointer, Int32]:
        ptr = cute.make_ptr(cutlass.BFloat16, self.dst_ptrs[iter_idx], AddressSpace.gmem, assumed_align=16)
        return ptr, self.valid[iter_idx]


@cute.jit
def fc2_dedup_bulk_store_function(
    *,
    epi,
    subtile: cute.Tensor,  # (64,) bf16: one element per token of the subtile (hidden = warp*32 + lane)
    epilogue_iter_idx: Int32,
    subtile_idx: Int32,
    fc2_output_router: Fc2DedupBulkRouter,
    **_,
):
    smem_tensor = epi.smem_tensor
    if cutlass.const_expr(smem_tensor is None):
        raise ValueError("The dedup bulk store requires epi.smem_tensor.")
    data_subtile = subtile
    if cutlass.const_expr(data_subtile.element_type is not cutlass.BFloat16):
        raise ValueError("The dedup FC2 store expects a BF16 payload.")
    if cutlass.const_expr(cute.size(data_subtile) != epi._EpilogueTokenTileSize):
        raise ValueError("BF16 bulk staging requires one register per token in the 64-token subtile.")
    warp_idx = epi.warp_idx
    lane_idx = epi.lane_idx
    smem_read_write_bar = SwapABGatedActEpilogue.epilogue_sync_barrier()
    stage_idx = epilogue_iter_idx % Int32(epi.fc2_tma_stages)
    smem_stage = smem_tensor[None, None, stage_idx]

    warp_hidden_base = Int32(warp_idx * 32)
    for token_idx in cutlass.range_constexpr(epi._EpilogueTokenTileSize):
        smem_stage[token_idx, warp_hidden_base + lane_idx] = data_subtile[token_idx]

    cute.arch.fence_proxy("async.shared", space="cta")
    smem_read_write_bar.arrive_and_wait()

    copy_elems = Int32(epi._EpilogueFc2HiddenTileSize)
    if cutlass.const_expr(epi.fc2_hidden_needs_predicate):
        copy_elems = fc2_output_router.valid_hidden_this_cta_tile
    copy_bytes = copy_elems * Int32(epi.combine_format.act_dtype.width // 8)
    scratch_row = warp_idx * Int32(16) + lane_idx % Int32(16)
    copy_iters: cutlass.Constexpr[int] = fc2_output_router.mapping.domain.axis_size("iter_idx")
    for ublk_iter_idx in cutlass.range_constexpr(copy_iters):
        owned_subtile = Int32(ublk_iter_idx * 2) + lane_idx // Int32(16)
        if owned_subtile == subtile_idx:
            dst_ptr, kind = fc2_output_router.get_data_dst(ublk_iter_idx)
            if kind != Int32(0):
                src_row = cute.slice_(smem_stage, (scratch_row, None))
                if cutlass.const_expr(fc2_output_router.reduce_topk_in_kernel):
                    if kind == Int32(2):
                        cp_reduce_async_bulk_add_bf16_s2g(dst_ptr, src_row.iterator, copy_bytes)
                    else:
                        cp_async_bulk_s2g(dst_ptr, src_row.iterator, copy_bytes)
                else:
                    cp_async_bulk_s2g(dst_ptr, src_row.iterator, copy_bytes)

    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(epi.fc2_tma_stages - 1, read=True)
    smem_read_write_bar.arrive_and_wait()


class SwapABFc2EpilogueDedup(SwapABFc2Epilogue):
    """FC2 epilogue with SMEM-staged, metadata-routed bulk stores (order buffer or direct peer push)."""

    def __init__(
        self,
        base: "SwapABGatedActEpilogueDedup",
        tidx: Int32,
        smem_tensor: Optional[cute.Tensor],
        tma_atom_fc2_output: Optional[cute.CopyAtom],
        fc2_tma_output: Optional[cute.Tensor],
        fc2_output: cute.Tensor,  # order buffer base as a (T*R*K, 1, hidden) tensor
        token_src_metadata: Optional[cute.Tensor],
        fc2_done_counter: Optional[cute.Tensor],
        fc2_output_sf: Optional[cute.Tensor],
        peer_rank_ptr_mapper,
        optional_epi_args: GatedActEpilogueArgs,
    ):
        # Not super().__init__(): the base picks its store pipeline and freezes; dedup installs its own.
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        self.tma_atom_fc2_output = None
        self.fc2_tma_output = None
        self.fc2_output = fc2_output
        self.token_src_metadata = token_src_metadata
        self.fc2_done_counter = fc2_done_counter
        self.fc2_output_sf = None
        self.peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self.optional_epi_args = optional_epi_args
        if smem_tensor is None:
            raise ValueError("The dedup FC2 epilogue needs the FC2 staging SMEM (fc2_use_bulk).")
        self.smem_tensor = smem_tensor
        self.process_pipeline = dataclasses.replace(
            make_fc2_ublk_process_pipeline(
                combine_format=base.combine_format,
                cta_token_tile_size=base.cta_tile_n,
                cta_hidden_tile_size=base.cta_tile_m,
            ),
            store_function=fc2_dedup_bulk_store_function,
        )
        self._freeze()

    @cute.jit
    def _make_bulk_router(self, work_tile_info: SwapAbFc12WorkTileInfo) -> Fc2DedupBulkRouter:
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row + work_tile_info.tile_n_idx * Int32(self.cta_tile_n)
        )
        hidden_base_this_cta_tile = work_tile_info.tile_m_idx * Int32(self.cta_tile_m)
        valid_hidden = Int32(self.fc2_output.shape[2]) - hidden_base_this_cta_tile
        if valid_hidden < 0:
            valid_hidden = 0
        if valid_hidden > self._EpilogueFc2HiddenTileSize:
            valid_hidden = self._EpilogueFc2HiddenTileSize
        remote_output = self.optional_epi_args.fc2_remote_output
        if cutlass.const_expr(remote_output is None):
            raise ValueError("The dedup FC2 epilogue needs fc2_remote_output (the pre-reduced (T, R, H) buffer).")
        # The pre_reduced slot of single-copy rows is the kernel's local_rank argument (not inferred from the mapper).
        local_rank = self.optional_epi_args.local_rank
        if cutlass.const_expr(local_rank is None):
            raise ValueError("The dedup FC2 epilogue needs local_rank (the kernel's host-provided rank argument).")
        return Fc2DedupBulkRouter(
            metadata=cute.domain_offset((task_tile_data_row_start,), self.token_src_metadata),
            order_base=self.fc2_output.iterator.toint(),
            remote_output=remote_output,
            peer_rank_ptr_mapper=self.peer_rank_ptr_mapper,
            local_rank=Int32(local_rank),
            hidden_base=hidden_base_this_cta_tile,
            valid_tokens_this_cta_tile=work_tile_info.valid_tokens_in_cta_tile,
            valid_hidden_this_cta_tile=valid_hidden,
            world_size=self.dedup_world_size,
            topk=self.dedup_topk,
            chunk_bytes=self.dedup_chunk_bytes,
            chunks_per_row=self.dedup_chunks_per_row,
            element_bytes=self.combine_format.act_dtype.width // 8,
            reduce_topk_in_kernel=self.reduce_topk_in_kernel,
            mapping=self.process_pipeline.store_out_mapping,
            epi_tid=self.tidx,
        )

    @cute.jit
    def __call__(
        self, work_tile_info: SwapAbFc12WorkTileInfo, tmem_acc_tensor: cute.Tensor, acc_pipeline, acc_consumer_state
    ):
        if cutlass.const_expr(self.optional_epi_args.fc2_alpha is not None):
            alpha_val = self.optional_epi_args.fc2_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        router = self._make_bulk_router(work_tile_info)
        acc_ready = False
        if not work_tile_info.peek_ready:
            acc_ready = True
            acc_pipeline.consumer_wait(acc_consumer_state)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor, (self._EpilogueFc2HiddenTileSize, self._EpilogueTokenTileSize)
        )[None, None, 0, None]
        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        # Read the metadata only after the accumulator wait: under PDL the router may still be pushing records.
        router = router.prefetch()
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        for subtile_idx in cutlass.range(self.subtile_cnt, unroll=1):
            if subtile_idx * Int32(self._EpilogueTokenTileSize) < valid_tokens:
                self.run_subtile(
                    work_tile_info=work_tile_info,
                    epilogue_iter_idx=subtile_idx,
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx],
                    fc2_output_router=router,
                    alpha_val=alpha_val,
                )
        cute.arch.fence_view_async_tmem_load()
        acc_pipeline.consumer_release(acc_consumer_state)


class SwapABGatedActEpilogueDedup(SwapABGatedActEpilogue):
    """``SwapABGatedActEpilogue`` whose FC2 half is ``SwapABFc2EpilogueDedup`` (bulk stores routed by metadata)."""

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {**super().problem_desc_require(), "apply_topk_at_fc1": bool}

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            **super().impl_desc_require(),
            "dedup_world_size": int,
            "dedup_topk": int,
            # Order buffer layout [T, R, chunks_per_row, K, chunk_bytes] (from TokenCommDedup).
            "dedup_chunk_bytes": int,
            "dedup_chunks_per_row": int,
        }

    def _make_fc2_epilogue(self, *args) -> SwapABFc2EpilogueDedup:
        return SwapABFc2EpilogueDedup(self, *args)

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        super().__init__(problem_desc, impl_desc)
        self.dedup_world_size = int(impl_desc["dedup_world_size"])
        self.dedup_topk = int(impl_desc["dedup_topk"])
        self.dedup_chunk_bytes = int(impl_desc["dedup_chunk_bytes"])
        self.dedup_chunks_per_row = int(impl_desc["dedup_chunks_per_row"])
        self.apply_topk_at_fc1 = bool(problem_desc["apply_topk_at_fc1"])
        if not self.apply_topk_at_fc1:
            raise ValueError("Combine dedup applies top-k scores at FC1 only (pre_fc1_quant); post_fc2 is not supported.")
        if not self.communication_enabled:
            raise ValueError("Combine dedup requires token communication.")
        if not self.token_back_push_data:
            raise ValueError(
                "Combine dedup reduces and pushes the order-buffer blocks from the combine warps; epi_warps is not supported."
            )
        if not self.fc2_use_bulk:
            raise ValueError("Combine dedup stages FC2 subtiles in SMEM for per-token bulk stores; fc2_use_bulk must be on.")
        if self.combine_format.is_quantized or self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("Combine dedup supports the BF16 combine format only.")
        if self.reduce_topk_in_epilogue:
            raise ValueError("Combine dedup never reduces in the epilogue.")
        if self.dedup_world_size <= 0 or self.dedup_topk <= 0 or self.dedup_chunks_per_row <= 0:
            raise ValueError("Combine dedup needs positive dedup_world_size, dedup_topk and dedup_chunks_per_row.")
        tile_bytes = self._EpilogueFc2HiddenTileSize * self.combine_format.act_dtype.width // 8
        if self.dedup_chunk_bytes <= 0 or self.dedup_chunk_bytes % tile_bytes != 0:
            raise ValueError(
                f"Combine dedup needs dedup_chunk_bytes as a multiple of the {tile_bytes} B hidden tile, got "
                f"{self.dedup_chunk_bytes}."
            )
        # The base resolved fc2_use_bulk + local output as the TMA tile store; dedup stores per token with UBLK.
        self.fc2_use_tma = False
        self.fc2_use_ublk = True
        single_stage_region = self._make_fc2_single_stage_region()
        if single_stage_region.nbytes % 16 != 0:
            raise ValueError("Each FC2 UBLK staging stage must occupy a multiple of 16 bytes.")
        requested_fc2_tma_stages = impl_desc.get("fc2_tma_stages")
        if requested_fc2_tma_stages is not None:
            self.fc2_tma_stages = int(requested_fc2_tma_stages)
        else:
            bf16_stage_bytes = self._EpilogueTokenTileSize * self._EpilogueFc2HiddenTileSize * cutlass.BFloat16.width // 8
            available_staging_bytes = max(self.fc1_staging_bytes, min(2, self.subtile_cnt) * bf16_stage_bytes)
            self.fc2_tma_stages = min(self.subtile_cnt, available_staging_bytes // single_stage_region.nbytes)
        self.fc2_staging_spec = self._make_fc2_staging_region(self.fc2_tma_stages)


__all__ = [
    "Fc2DedupBulkRouter",
    "GatedActEpilogueArgsDedup",
    "SwapABFc2EpilogueDedup",
    "SwapABGatedActEpilogueDedup",
    "fc2_dedup_bulk_store_function",
]
