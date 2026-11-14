# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-scheduled QSA indexer scoring over the paged index cache.

A CTA owns one tile of 32 consecutive query rows of one request (x 4 indexer
heads interleaved into the MMA M axis), holds its Q tile in SMEM and streams a
range of 128-block key tiles through a TmaUmma ring:

    SmemQResource, SmemKResource -> TmemSResource -> GmemScoreResource

Tiles are derived on device from the request-sorted rows: request ``r`` owns
rows ``[req_start[r], req_start[r+1])`` and tile ``t`` of the launch maps to
request ``r`` with ``tile_offsets[r-1] <= t < tile_offsets[r]``. The load task
TMA-fills Q once and one 128-block x 128-dim K-bar stage per iteration (a box
per 8-row core group, both 64-dim halves adjacent, resolved through the block
table); the MMA task accumulates S in a two-stage TMEM ring; the store task
applies scale and ReLU, sums the four head rows of each query with warp
shuffles and writes the fp32 scores straight from registers.
"""

from dataclasses import dataclass, field

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.pipeline as pipeline
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    SmemAllocation,
    SmemAllocator,
    TmemAllocation,
    TmemAllocator,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from cutlass.experimental.task_scheduling.task_manager import TaskManager

from tensorrt_llm._torch.attention.backends.prims_ts.kernels.placeholder_helpers import (
    _placeholder_smem_array,
)

io_dtype = cutlass.BFloat16
acc_dtype = cutlass.Float32

HEAD_DIM = 128
NUM_HEADS = 4
QUERY_ROWS_PER_TILE = 32
TILE_M = QUERY_ROWS_PER_TILE * NUM_HEADS
TILE_N = 128
STAGE_K = 64
Q_HALVES = HEAD_DIM // STAGE_K
MMA_K = 16
SMEM_STAGES = 5
# SW128 K-major core group: 8 rows x 128 B. A K stage holds all 128 dims of
# 128 blocks; when a page holds at least 8 compressed rows each TMA box
# brings one 8-row group with both 64-dim halves adjacent (16 boxes per
# stage), otherwise each half of the stage is filled page by page.
GROUP_ROWS = 8
TMEM_STAGES = 2
TMEM_COLS = TILE_N * TMEM_STAGES
T2R_COLS = 32
NUM_SUBTILES = TILE_N // T2R_COLS
COLS_PER_LANE = T2R_COLS // NUM_HEADS
SEARCH_STEPS = 16
# Named barriers below 12 are reserved for the task-scheduling runtime; 12
# and 13 bracket the task run.
NEG_INF = float("-inf")
# Page ids for the CTA's whole block-tile range are staged in SMEM once; the
# host bounds n_splits so a CTA never needs more than this many.
PAGE_ID_CAPACITY = 4096

_Q_STAGE_ELEMS = TILE_M * STAGE_K
_Q_ELEMS = TILE_M * HEAD_DIM
_K_STAGE_ELEMS = TILE_N * HEAD_DIM
_K_GROUP_ELEMS = GROUP_ROWS * HEAD_DIM
_K_HALF_ELEMS = TILE_N * STAGE_K
_Q_BYTES = _Q_ELEMS * io_dtype.width // 8
_K_STAGE_BYTES = _K_STAGE_ELEMS * io_dtype.width // 8


@cute.jit
def _tile_coords(
    tile_offsets: cute.Tensor,
    req_start: cute.Tensor,
    positions: cute.Tensor,
    num_rows: cutlass.Int32,
    n_splits: cutlass.Int32,
    compress_ratio: cutlass.Constexpr[int],
):
    """Resolve this CTA's (request, first row, row count, block-tile range).

    Launch tiles past the real tile count repeat the last tile; their stores
    are idempotent.
    """
    bx, _, _ = cute.arch.block_idx()
    num_requests = cutlass.Int32(tile_offsets.shape[0])
    total_tiles = cutlass.Int32(tile_offsets[num_requests - 1])
    tile = bx // n_splits
    if tile >= total_tiles:
        tile = total_tiles - 1
    split = bx % n_splits
    lo = cutlass.Int32(0)
    hi = num_requests
    for _ in cutlass.range_constexpr(SEARCH_STEPS):
        mid = (lo + hi) >> 1
        probe = mid
        if probe > num_requests - 1:
            probe = num_requests - 1
        if cutlass.Int32(tile_offsets[probe]) > tile:
            hi = mid
        else:
            lo = mid + 1
    request = lo
    if request > num_requests - 1:
        request = num_requests - 1
    tiles_before = cutlass.Int32(0)
    if request > 0:
        tiles_before = cutlass.Int32(tile_offsets[request - 1])
    row_begin = cutlass.Int32(req_start[request])
    row_end = num_rows
    if request + 1 < num_requests:
        row_end = cutlass.Int32(req_start[request + 1])
    first_row = row_begin + (tile - tiles_before) * QUERY_ROWS_PER_TILE
    tile_rows = row_end - first_row
    if tile_rows > QUERY_ROWS_PER_TILE:
        tile_rows = QUERY_ROWS_PER_TILE
    if tile_rows < 0:
        tile_rows = cutlass.Int32(0)
    lane = cute.arch.lane_idx()
    visible = cutlass.Int32(0)
    if lane < tile_rows:
        visible = (cutlass.Int32(positions[first_row + lane]) + 1) // compress_ratio
    tile_visible = cutlass.Int32(cute.arch.warp_redux_sync(visible, "max"))
    n_count = (tile_visible + TILE_N - 1) // TILE_N
    chunk = (n_count + n_splits - 1) // n_splits
    n_begin = split * chunk
    n_end = n_begin + chunk
    if n_end > n_count:
        n_end = n_count
    n_iters = n_end - n_begin
    if n_iters < 1:
        n_iters = cutlass.Int32(1)
        n_begin = n_count - 1
        if n_begin < 0:
            n_begin = cutlass.Int32(0)
    return request, first_row, tile_rows, n_begin, n_iters


@dataclass
class SmemQResource(MemoryResource):
    """The CTA's Q tile, TMA-loaded once and held for every K tile."""

    tma_desc_q: cutlass.Pointer = field(init=False, default=None)
    first_row: cutlass.Int32 = field(init=False, default=None)
    smem_q: cutlass.Array = field(init=False, default=None)
    desc_q_base: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    _alloc_q: cutlass.Constexpr = field(init=False, default=None)

    def __init__(
        self, tma_desc_q: cutlass.Pointer, first_row: cutlass.Int32, **kwargs: object
    ) -> None:
        super().__init__(**kwargs)
        self.tma_desc_q = tma_desc_q
        self.first_row = first_row
        self.desc_q_base = TaskLocalVariable(dtype=cutlass.Int64, default=cutlass.Int64(0))
        self.smem_q = _placeholder_smem_array(io_dtype, _Q_ELEMS)
        self._alloc_q = SmemAllocation("smem_q", _Q_BYTES, alignment=1024)

    def get_smem_requirements(self):
        return [self._alloc_q]

    @cute.jit
    def _init_state(self, stage_info: StageInfo) -> None:
        self.smem_q = cutlass.Array(
            stage_info.context.smem_base.data_ptr() + self._alloc_q.offset,
            dtype=io_dtype,
            shape=(_Q_ELEMS,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_state(stage_info)

    @producer_work
    @cute.jit
    def tma_load(self, stage_info: StageInfo) -> None:
        if prims.elect_sync():
            for half in cutlass.range_constexpr(Q_HALVES):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    self.smem_q.subview(_Q_STAGE_ELEMS * half),
                    self.tma_desc_q,
                    (cutlass.Int32(half * STAGE_K), self.first_row * NUM_HEADS),
                    stage_info.barrier,
                )

    @consumer_work(returns=desc_q_base)
    @cute.jit
    def build_descriptor(self, stage_info: StageInfo) -> cutlass.Int64:
        return prims.Tcgen05SmemDesc.build(
            self.smem_q, leading_byte_offset=16, stride_byte_offset=1024, layout=2
        )


@dataclass
class SmemKResource(MemoryResource):
    """Ring of K-bar operand tiles: one 128-block x 64-dim stage per TMA group."""

    tma_desc_k: cutlass.Pointer = field(init=False, default=None)
    block_table: cute.Tensor = field(init=False, default=None)
    blocks_per_page: cutlass.Constexpr[int] = field(init=False, default=None)
    compress_ratio: cutlass.Constexpr[int] = field(init=False, default=None)
    merged_halves: cutlass.Constexpr[bool] = field(init=False, default=None)
    stride_byte_offset: cutlass.Constexpr[int] = field(init=False, default=None)
    half_bytes: cutlass.Constexpr[int] = field(init=False, default=None)
    smem_k: cutlass.Array = field(init=False, default=None)
    smem_pages: cutlass.Array = field(init=False, default=None)
    request: cutlass.Int32 = field(init=False, default=None)
    n_begin: cutlass.Int32 = field(init=False, default=None)
    n_iters: cutlass.Int32 = field(init=False, default=None)
    desc_k_base: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    _alloc_k: cutlass.Constexpr = field(init=False, default=None)
    _alloc_pages: cutlass.Constexpr = field(init=False, default=None)

    def __init__(
        self,
        tma_desc_k: cutlass.Pointer,
        block_table: cute.Tensor,
        request: cutlass.Int32,
        n_begin: cutlass.Int32,
        n_iters: cutlass.Int32,
        blocks_per_page: int,
        compress_ratio: int,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.tma_desc_k = tma_desc_k
        self.block_table = block_table
        self.blocks_per_page = blocks_per_page
        self.compress_ratio = compress_ratio
        self.merged_halves = blocks_per_page >= GROUP_ROWS
        self.stride_byte_offset = (
            GROUP_ROWS * HEAD_DIM * 2 if self.merged_halves else GROUP_ROWS * STAGE_K * 2
        )
        self.half_bytes = GROUP_ROWS * STAGE_K * 2 if self.merged_halves else _K_HALF_ELEMS * 2
        self.request = request
        self.n_begin = n_begin
        self.n_iters = n_iters
        self.desc_k_base = TaskLocalVariable(dtype=cutlass.Int64, default=cutlass.Int64(0))
        # Placeholders keep the field structure stable across the warp-gated
        # task dispatch; the real views are bound in init work.
        self.smem_k = _placeholder_smem_array(io_dtype, _K_STAGE_ELEMS * SMEM_STAGES)
        self.smem_pages = _placeholder_smem_array(cutlass.Int32, PAGE_ID_CAPACITY)
        self._alloc_k = SmemAllocation("smem_k", _K_STAGE_BYTES * SMEM_STAGES, alignment=1024)
        self._alloc_pages = SmemAllocation("smem_pages", PAGE_ID_CAPACITY * 4, alignment=16)

    def get_smem_requirements(self):
        return [self._alloc_k, self._alloc_pages]

    @cute.jit
    def _init_state(self, stage_info: StageInfo) -> None:
        smem_base = stage_info.context.smem_base
        self.smem_k = cutlass.Array(
            smem_base.data_ptr() + self._alloc_k.offset,
            dtype=io_dtype,
            shape=(_K_STAGE_ELEMS * SMEM_STAGES,),
            addrspace=3,
        )
        self.smem_pages = cutlass.Array(
            smem_base.data_ptr() + self._alloc_pages.offset,
            dtype=cutlass.Int32,
            shape=(PAGE_ID_CAPACITY,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_state(stage_info)
        # Stage the physical page of every block tile this CTA will visit.
        # Pages past the table or unallocated (negative) entries are masked
        # in the epilogue; clamp so the TMA stays in bounds.
        pages_per_tile = TILE_N // self.blocks_per_page
        max_pages = cutlass.Int32(self.block_table.shape[1])
        total = self.n_iters * pages_per_tile
        lane = cute.arch.lane_idx()
        for i in cutlass.range(lane, total, 32):
            logical_page = self.n_begin * pages_per_tile + i
            if logical_page >= max_pages:
                logical_page = max_pages - 1
            physical_page = cutlass.Int32(self.block_table[(self.request, logical_page)])
            if physical_page < 0:
                physical_page = cutlass.Int32(0)
            self.smem_pages[i] = physical_page
        cute.arch.sync_warp()

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_state(stage_info)

    @consumer_work(returns=desc_k_base)
    @cute.jit
    def build_descriptor(self, stage_info: StageInfo) -> cutlass.Int64:
        sk = self.smem_k.subview(_K_STAGE_ELEMS * stage_info.stage_idx)
        return prims.Tcgen05SmemDesc.build(
            sk,
            leading_byte_offset=16,
            stride_byte_offset=self.stride_byte_offset,
            layout=2,
        )

    @producer_work
    @cute.jit
    def tma_load(self, stage_info: StageInfo) -> None:
        sk = self.smem_k.subview(_K_STAGE_ELEMS * stage_info.stage_idx)
        pages_per_tile = TILE_N // self.blocks_per_page
        page_base = stage_info.loop_offset * pages_per_tile
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.merged_halves):
            groups_per_page = self.blocks_per_page // GROUP_ROWS
            if lane < TILE_N // GROUP_ROWS:
                physical_page = cutlass.Int32(self.smem_pages[page_base + lane // groups_per_page])
                row0 = cutlass.Int32((lane % groups_per_page) * GROUP_ROWS)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sk.subview(_K_GROUP_ELEMS * lane),
                    self.tma_desc_k,
                    (cutlass.Int32(0), row0, cutlass.Int32(0), physical_page),
                    stage_info.barrier,
                )
        else:
            if lane < pages_per_tile:
                physical_page = cutlass.Int32(self.smem_pages[page_base + lane])
                for half in cutlass.range_constexpr(Q_HALVES):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sk.subview(_K_HALF_ELEMS * half + self.blocks_per_page * STAGE_K * lane),
                        self.tma_desc_k,
                        (
                            cutlass.Int32(half * STAGE_K),
                            cutlass.Int32(self.compress_ratio - 1),
                            cutlass.Int32(0),
                            physical_page,
                        ),
                        stage_info.barrier,
                    )


@dataclass
class TmemSResource(MemoryResource):
    """FP32 score accumulator ring in TMEM between the MMA and store tasks."""

    tmem_raw_addr: cutlass.Int32 = field(init=False, default=None)
    _alloc_acc: cutlass.Constexpr = field(init=False, default=None)
    t2r_rmem: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._alloc_acc = TmemAllocation("tmem_s", TMEM_COLS)
        self.tmem_raw_addr = cutlass.Int32(0)
        self.t2r_rmem = TaskLocalVariable(
            dtype=cutlass.Float32,
            default=cutlass.full([T2R_COLS], 0.0, cutlass.Float32),
        )

    def get_tmem_requirements(self):
        return [self._alloc_acc]

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        self.tmem_raw_addr = cutlass.Int32(stage_info.context.tmem_ptr_i32.load())

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_accumulator_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        self._init_tmem_state(stage_info)

    @consumer_work(returns=t2r_rmem)
    @cute.jit
    def load_subtile(
        self, stage_info: StageInfo, *, subtile_idx: cutlass.Constexpr[int]
    ) -> cutlass.Float32:
        # Routed by variable name, so one 32-column load per call; the four
        # round trips per tile are hidden behind the loader, not the limiter.
        base_col = (
            (self.tmem_raw_addr & 0xFFFF) + stage_info.stage_idx * TILE_N + subtile_idx * T2R_COLS
        )
        tmem_ptr = prims.make_tmem_ptr_from_warp_row_col(
            self.tmem_raw_addr, cute.arch.warp_idx(), base_col, cutlass.Float32
        )
        t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ptr, num=T2R_COLS)
        cute.arch.fence_view_async_tmem_load()
        return t2r_rmem

    @producer_work
    @cute.jit
    def mma(
        self,
        stage_info: StageInfo,
        *,
        desc_q_base: cutlass.Int64,
        desc_k_base: cutlass.Int64,
        k_half_bytes: cutlass.Constexpr[int],
    ) -> None:
        idesc = prims.Tcgen05InstrDesc.build(
            a_dtype=io_dtype,
            b_dtype=io_dtype,
            c_dtype=cutlass.Float32,
            n_dim=TILE_N,
            m_dim=TILE_M,
        )
        tmem_ptr = prims.make_tmem_ptr(
            self.tmem_raw_addr + stage_info.stage_idx * TILE_N, acc_dtype
        )
        tmem_acc = cutlass.Array(tmem_ptr.data_ptr(), dtype=cutlass.Int32, addrspace=6)
        blocks_per_half = STAGE_K // MMA_K
        for k_block in cutlass.range_constexpr(HEAD_DIM // MMA_K):
            half = k_block // blocks_per_half
            increment = (MMA_K * io_dtype.width // 8 * (k_block % blocks_per_half)) >> 4
            q_offset = ((_Q_STAGE_ELEMS * io_dtype.width // 8) * half) >> 4
            k_offset = (k_half_bytes * half) >> 4
            accumulate = cutlass.Boolean(k_block > 0)
            if prims.elect_sync():
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16,
                    prims.CTAGroup.CTA_1,
                    tmem_acc,
                    desc_q_base + q_offset + increment,
                    desc_k_base + k_offset + increment,
                    idesc,
                    accumulate,
                )


@dataclass
class GmemScoreResource(MemoryResource):
    """Score sink.

    Relu, head-sum, scale and causal mask straight from registers into the
    fp32 score rows with 16-byte vector stores.
    """

    scores: cute.Tensor = field(init=False, default=None)
    positions: cute.Tensor = field(init=False, default=None)
    scale: cutlass.Float32 = field(init=False, default=None)
    compress_ratio: cutlass.Constexpr[int] = field(init=False, default=None)
    first_row: cutlass.Int32 = field(init=False, default=None)
    tile_rows: cutlass.Int32 = field(init=False, default=None)
    n_begin: cutlass.Int32 = field(init=False, default=None)
    visible: cutlass.Int32 = field(init=False, default=None)
    row_ptr: cutlass.Int64 = field(init=False, default=None)

    def __init__(
        self,
        scores: cute.Tensor,
        positions: cute.Tensor,
        scale: cutlass.Float32,
        first_row: cutlass.Int32,
        tile_rows: cutlass.Int32,
        n_begin: cutlass.Int32,
        compress_ratio: int,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.scores = scores
        self.positions = positions
        self.scale = scale
        self.compress_ratio = compress_ratio
        self.first_row = first_row
        self.tile_rows = tile_rows
        self.n_begin = n_begin
        self.visible = cutlass.Int32(0)
        self.row_ptr = cutlass.Int64(0)

    def get_smem_requirements(self):
        return []

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        tx, _, _ = cute.arch.thread_idx()
        tile_row = tx // NUM_HEADS
        self.visible = cutlass.Int32(0)
        if tile_row < self.tile_rows:
            self.visible = (
                cutlass.Int32(self.positions[self.first_row + tile_row]) + 1
            ) // self.compress_ratio
        num_cols = cutlass.Int64(self.scores.shape[1])
        self.row_ptr = self.scores.iterator.toint() + (
            cutlass.Int64(self.first_row + tile_row) * num_cols * (acc_dtype.width // 8)
        )

    @producer_work
    @cute.jit
    def stage(
        self,
        stage_info: StageInfo,
        *,
        t2r_rmem: cutlass.Float32,
        subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        tx, _, _ = cute.arch.thread_idx()
        lane = tx % 32
        upper_half = (lane & 1) != 0
        upper_quarter = (lane & 2) != 0
        # The four head rows of one query sit in four adjacent lanes. Reduce-
        # scatter over the two lane bits so each lane ends with 8 fully summed
        # columns: lanes with bit0 set keep columns 16..31, then bit1 selects
        # the upper 8 of those 16.
        half = cutlass.Array(cutlass.Float32, 16, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(16):
            lo = t2r_rmem[i]
            hi = t2r_rmem[16 + i]
            if lo < cutlass.Float32(0.0):
                lo = cutlass.Float32(0.0)
            if hi < cutlass.Float32(0.0):
                hi = cutlass.Float32(0.0)
            send = hi
            keep = lo
            if upper_half:
                send = lo
                keep = hi
            recv = cutlass.Float32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=send,
                    offset=1,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            )
            half[i] = keep + recv
        quarter = cutlass.Array(cutlass.Float32, COLS_PER_LANE, space=cutlass.AddressSpace.rmem)
        n_tile = self.n_begin + stage_info.loop_offset
        col0 = n_tile * TILE_N + subtile_idx * T2R_COLS + (lane & 1) * 16 + (lane & 2) * 4
        for i in cutlass.range_constexpr(COLS_PER_LANE):
            lo = half[i]
            hi = half[COLS_PER_LANE + i]
            send = hi
            keep = lo
            if upper_quarter:
                send = lo
                keep = hi
            recv = cutlass.Float32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=send,
                    offset=2,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            )
            value = (keep + recv) * self.scale
            if col0 + i >= self.visible:
                value = cutlass.Float32(NEG_INF)
            quarter[i] = value
        num_cols = cutlass.Int32(self.scores.shape[1])
        tile_row = tx // NUM_HEADS
        if tile_row < self.tile_rows and col0 < num_cols:
            dst = cutlass.inttoptr(
                self.row_ptr + cutlass.Int64(col0) * (acc_dtype.width // 8),
                mem_space=1,
                dtype=cutlass.Float32,
            )
            if col0 + COLS_PER_LANE <= num_cols:
                dst.store(quarter.data_ptr().load(count=4, alignment=16), alignment=16)
                (dst + 4).store((quarter.data_ptr() + 4).load(count=4, alignment=16), alignment=16)
            else:
                for i in cutlass.range_constexpr(COLS_PER_LANE):
                    if col0 + i < num_cols:
                        (dst + i).store(quarter[i])


@cute.jit
def create_smem_q_resource(tma_desc_q, first_row) -> SmemQResource:
    cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=1,
        num_bytes=_Q_BYTES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        cta_layout_vmnk=(1, 1, 1, 1),
    )
    return SmemQResource(
        tma_desc_q=tma_desc_q, first_row=first_row, pipeline_config=cfg, name="SmemQ"
    )


@cute.jit
def create_smem_k_resource(
    tma_desc_k, block_table, request, n_begin, n_iters, blocks_per_page, compress_ratio
) -> SmemKResource:
    cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=SMEM_STAGES,
        num_bytes=_K_STAGE_BYTES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        cta_layout_vmnk=(1, 1, 1, 1),
    )
    return SmemKResource(
        tma_desc_k=tma_desc_k,
        block_table=block_table,
        request=request,
        n_begin=n_begin,
        n_iters=n_iters,
        blocks_per_page=blocks_per_page,
        compress_ratio=compress_ratio,
        pipeline_config=cfg,
        name="SmemK",
    )


@cute.jit
def create_tmem_s_resource(num_epilogue_warps: int) -> TmemSResource:
    cfg = PipelineConfig.create_umma_async_pipeline_cfg(
        num_stages=TMEM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, size=num_epilogue_warps * 32
        ),
        cta_layout_vmnk=(1, 1, 1, 1),
    )
    return TmemSResource(pipeline_config=cfg, name="TmemS")


@cute.jit
def create_load_task(smem_q, smem_k, n_iters, num_load_warps: int) -> Task:
    @schedule
    def load_schedule(q: MemoryResource, k: MemoryResource) -> None:
        q.init_load_state()
        k.init_load_state()
        q.try_acquire()
        q.acquire()
        q.tma_load()
        q.commit()
        with domain_loop(0, n_iters, 1, unroll=1):
            k.try_acquire()
            k.acquire()
            k.tma_load()
            k.commit()

    return Task(
        src_resources=[],
        dst_resources=[smem_q, smem_k],
        warp_idx=4,
        num_warps=num_load_warps,
        schedule=load_schedule(smem_q, smem_k),
        num_registers=40,
        name="LoadTask",
    )


@cute.jit
def create_mma_task(smem_q, smem_k, tmem_s, n_iters, num_mma_warps: int) -> Task:
    k_half_bytes = smem_k.half_bytes

    @schedule
    def mma_schedule(q: MemoryResource, k: MemoryResource, s: MemoryResource) -> None:
        q.init_mma_state()
        k.init_mma_state()
        s.init_accumulator_state()
        q.try_wait()
        q.wait()
        desc_q_base = q.build_descriptor()
        with domain_loop(0, n_iters, 1, unroll=1):
            s.try_acquire()
            s.acquire()
            k.try_wait()
            k.wait()
            desc_k_base = k.build_descriptor()
            s.mma(
                desc_q_base=desc_q_base,
                desc_k_base=desc_k_base,
                k_half_bytes=k_half_bytes,
            )
            k.release()
            s.commit()
        q.release()

    return Task(
        src_resources=[smem_q, smem_k],
        dst_resources=[tmem_s],
        warp_idx=5,
        num_warps=num_mma_warps,
        schedule=mma_schedule(smem_q, smem_k, tmem_s),
        num_registers=40,
        name="MmaTask",
    )


@cute.jit
def create_store_task(tmem_s, gmem_scores, n_iters, num_epilogue_warps: int) -> Task:
    @schedule
    def store_schedule(s: MemoryResource, out: MemoryResource) -> None:
        s.init_store_state()
        out.init_store_state()
        with domain_loop(0, n_iters, 1, unroll=1):
            s.try_wait()
            s.wait()
            for subtile_idx in cutlass.range_constexpr(NUM_SUBTILES):
                t2r_rmem = s.load_subtile(subtile_idx=subtile_idx)
                out.stage(t2r_rmem=t2r_rmem, subtile_idx=subtile_idx)
            s.release()

    return Task(
        src_resources=[tmem_s],
        dst_resources=[gmem_scores],
        warp_idx=0,
        num_warps=num_epilogue_warps,
        schedule=store_schedule(tmem_s, gmem_scores),
        num_registers=168,
        name="StoreTask",
    )


@cute.jit
def create_padding_task(n_iters, num_padding_warps: int, first_warp: int) -> Task:
    @schedule
    def padding_schedule() -> None:
        with domain_loop(0, n_iters, 1, unroll=1):
            pass

    return Task(
        src_resources=[],
        dst_resources=[],
        warp_idx=first_warp,
        num_warps=num_padding_warps,
        schedule=padding_schedule(),
        num_registers=40,
        name="PaddingTask",
    )


@cute.jit
def _build_pipeline(
    tma_desc_q,
    tma_desc_k,
    block_table,
    positions,
    scores,
    scale,
    request,
    first_row,
    tile_rows,
    n_begin,
    n_iters,
    blocks_per_page,
    compress_ratio,
):
    num_epilogue_warps = 4
    num_load_warps = 1
    num_mma_warps = 1

    allocator = SmemAllocator()
    smem_q = create_smem_q_resource(tma_desc_q, first_row)
    smem_k = create_smem_k_resource(
        tma_desc_k,
        block_table,
        request,
        n_begin,
        n_iters,
        blocks_per_page,
        compress_ratio,
    )
    tmem_s = create_tmem_s_resource(num_epilogue_warps)
    gmem_scores = GmemScoreResource(
        scores=scores,
        positions=positions,
        scale=scale,
        first_row=first_row,
        tile_rows=tile_rows,
        n_begin=n_begin,
        compress_ratio=compress_ratio,
        name="GmemScores",
    )
    allocator.add_resource(smem_q)
    allocator.add_resource(smem_k)
    allocator.add_tmem_ptr(SmemAllocation("tmem_ptr_i32", dtype=cutlass.Int32, alignment=4))
    allocator.compute_layout()
    allocator.allocate()
    tmem_ptr_i32 = allocator.get(allocator.tmem_ptr_alloc)
    tmem_allocator = TmemAllocator()
    tmem_allocator.add_resource(tmem_s)
    tmem_allocator.compute_layout()

    total_warps = num_epilogue_warps + num_load_warps + num_mma_warps
    num_padding_warps = (total_warps + 3) // 4 * 4 - total_warps
    tasks = [
        create_load_task(smem_q, smem_k, n_iters, num_load_warps),
        create_mma_task(smem_q, smem_k, tmem_s, n_iters, num_mma_warps),
        create_store_task(tmem_s, gmem_scores, n_iters, num_epilogue_warps),
    ]
    if cutlass.const_expr(num_padding_warps > 0):
        tasks.append(create_padding_task(n_iters, num_padding_warps, total_warps))
    task_manager = TaskManager(
        tasks=tasks,
        resource_dependency_graph={
            tmem_s: [smem_q, smem_k],
            gmem_scores: [tmem_s],
        },
        smem_allocator=allocator,
        tmem_allocator=tmem_allocator,
        tmem_ptr_i32=tmem_ptr_i32,
        verbose=False,
        exhaustive_deadlock_race_check=False,
    )
    return task_manager, tmem_ptr_i32


@cute.jit
def _run(task_manager, warp_idx, tmem_ptr_i32) -> None:
    task_manager.setup_resources_and_tasks()
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp_idx == 0:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.tcgen05_relinquish_alloc_permit()
    prims.barrier_cta_sync(12)
    task_manager.run()
    prims.barrier_cta_sync(13)
    if warp_idx == 0:
        tmem_ptr = prims.make_tmem_ptr(tmem_ptr_i32.load(), acc_dtype)
        prims.tcgen05_dealloc(tmem_ptr, TMEM_COLS)


@cute.kernel
def qsa_index_scores_kernel(
    tma_desc_q: cutlass.GridConstant[cuda.TensorMap],
    tma_desc_k: cutlass.GridConstant[cuda.TensorMap],
    block_table: cute.Tensor,
    tile_offsets: cute.Tensor,
    req_start: cute.Tensor,
    positions: cute.Tensor,
    scores: cute.Tensor,
    num_rows: cutlass.Int32,
    n_splits: cutlass.Int32,
    scale: cutlass.Float32,
    blocks_per_page: cutlass.Constexpr[int],
    compress_ratio: cutlass.Constexpr[int],
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 4:
        prims.prefetch_tensormap(tma_desc_q.get_ptr())
        prims.prefetch_tensormap(tma_desc_k.get_ptr())
    request, first_row, tile_rows, n_begin, n_iters = _tile_coords(
        tile_offsets, req_start, positions, num_rows, n_splits, compress_ratio
    )
    task_manager, tmem_ptr_i32 = _build_pipeline(
        tma_desc_q.get_ptr(),
        tma_desc_k.get_ptr(),
        block_table,
        positions,
        scores,
        scale,
        request,
        first_row,
        tile_rows,
        n_begin,
        n_iters,
        blocks_per_page,
        compress_ratio,
    )
    _run(task_manager, warp_idx, tmem_ptr_i32)


META_THREADS = 128


@cute.kernel
def qsa_index_tiles_kernel(
    request_indices: cute.Tensor,
    tile_meta: cute.Tensor,
    num_rows: cutlass.Int32,
) -> None:
    """Fill ``tile_meta = [req_start[0..R), tile_offsets[0..R))`` for request-sorted rows.

    ``req_start[r]`` is the first row of request ``r`` (``num_rows`` for an
    empty request) and ``tile_offsets`` is the inclusive scan of each
    request's 32-row tile count.
    """
    tx, _, _ = cute.arch.thread_idx()
    num_requests = cutlass.Int32(tile_meta.shape[0]) // 2
    for r in cutlass.range(tx, num_requests, META_THREADS):
        lo = cutlass.Int32(0)
        hi = num_rows
        for _ in cutlass.range_constexpr(32):
            if lo < hi:
                mid = (lo + hi) >> 1
                if cutlass.Int32(request_indices[mid]) < r:
                    lo = mid + 1
                else:
                    hi = mid
        tile_meta[r] = lo
    cute.arch.barrier()
    for r in cutlass.range(tx, num_requests, META_THREADS):
        row_begin = cutlass.Int32(tile_meta[r])
        row_end = num_rows
        if r + 1 < num_requests:
            row_end = cutlass.Int32(tile_meta[r + 1])
        tile_meta[num_requests + r] = (
            row_end - row_begin + QUERY_ROWS_PER_TILE - 1
        ) // QUERY_ROWS_PER_TILE
    cute.arch.barrier()
    if tx < 32:
        carry = cutlass.Int32(0)
        for base in cutlass.range(0, num_requests, 32):
            idx = base + tx
            value = cutlass.Int32(0)
            if idx < num_requests:
                value = cutlass.Int32(tile_meta[num_requests + idx])
            for step in cutlass.range_constexpr(5):
                other = cutlass.Int32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=value,
                        offset=1 << step,
                        mask_and_clamp=0,
                        kind=prims.Shfl.UP,
                    )
                )
                if tx >= (1 << step):
                    value = value + other
            total = cutlass.Int32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=value,
                    offset=31,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            )
            if idx < num_requests:
                tile_meta[num_requests + idx] = value + carry
            carry = carry + total


@cute.jit
def qsa_index_scores_host(
    q_ptr: cute.Pointer,
    index_ptr: cute.Pointer,
    block_table_ptr: cute.Pointer,
    request_indices_ptr: cute.Pointer,
    tile_meta_ptr: cute.Pointer,
    positions_ptr: cute.Pointer,
    scores_ptr: cute.Pointer,
    num_requests: cutlass.Int32,
    max_pages: cutlass.Int32,
    max_blocks: cutlass.Int32,
    num_pages: cutlass.Int32,
    page_stride: cutlass.Int32,
    num_rows: cutlass.Int32,
    n_splits: cutlass.Int32,
    scale: cutlass.Float32,
    grid_size: cutlass.Int32,
    blocks_per_page: cutlass.Constexpr[int],
    compress_ratio: cutlass.Constexpr[int],
    stream,
) -> None:
    q_rows = cute.make_tensor(
        q_ptr, cute.make_layout((num_rows * NUM_HEADS, HEAD_DIM), stride=(HEAD_DIM, 1))
    )
    block_table = cute.make_tensor(
        block_table_ptr,
        cute.make_layout((num_requests, max_pages), stride=(max_pages, 1)),
    )
    request_indices = cute.make_tensor(request_indices_ptr, cute.make_layout((num_rows,)))
    tile_meta = cute.make_tensor(tile_meta_ptr, cute.make_layout((2 * num_requests,)))
    req_start = cute.make_tensor(tile_meta_ptr, cute.make_layout((num_requests,)))
    tile_offsets = cute.make_tensor(tile_meta_ptr + num_requests, cute.make_layout((num_requests,)))
    positions = cute.make_tensor(positions_ptr, cute.make_layout((num_rows,)))
    scores = cute.make_tensor(
        scores_ptr, cute.make_layout((num_rows, max_blocks), stride=(max_blocks, 1))
    )
    qsa_index_tiles_kernel(request_indices, tile_meta, num_rows).launch(
        grid=(1, 1, 1), block=[META_THREADS, 1, 1], stream=stream
    )
    tma_desc_q = cuda.create_tensor_map_tiled_from_view(
        q_rows,
        box_dims=(TILE_M, STAGE_K),
        stride_order=(1, 0),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    if cutlass.const_expr(blocks_per_page >= GROUP_ROWS):
        # Compressed rows sit at token slot compress_ratio-1 of every block;
        # the two 64-dim halves are separate map dims so one box lands them
        # adjacently in SMEM.
        index_groups = cute.make_tensor(
            index_ptr + (compress_ratio - 1) * HEAD_DIM,
            cute.make_layout(
                (blocks_per_page, num_pages, Q_HALVES, STAGE_K),
                stride=(compress_ratio * HEAD_DIM, page_stride, STAGE_K, 1),
            ),
        )
        tma_desc_k = cuda.create_tensor_map_tiled_from_view(
            index_groups,
            box_dims=(GROUP_ROWS, 1, Q_HALVES, STAGE_K),
            stride_order=(3, 0, 2, 1),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
    else:
        index_groups = cute.make_tensor(
            index_ptr,
            cute.make_layout(
                (blocks_per_page, num_pages, compress_ratio, HEAD_DIM),
                stride=(compress_ratio * HEAD_DIM, page_stride, HEAD_DIM, 1),
            ),
        )
        tma_desc_k = cuda.create_tensor_map_tiled_from_view(
            index_groups,
            box_dims=(blocks_per_page, 1, 1, STAGE_K),
            stride_order=(3, 2, 0, 1),
            swizzle=cuda.TensorMapSwizzle.s128b,
        )
    qsa_index_scores_kernel(
        tma_desc_q,
        tma_desc_k,
        block_table,
        tile_offsets,
        req_start,
        positions,
        scores,
        num_rows,
        n_splits,
        scale,
        blocks_per_page,
        compress_ratio,
    ).launch(grid=(grid_size, 1, 1), block=[256, 1, 1], stream=stream)
