# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resource definitions for the TS FMHA kernel.

Maps the FMHA data-flow onto ``MemoryResource`` subclasses:

Each resource owns the work attached to one live buffer: producer methods fill
the buffer, consumer methods drain it, and the pipeline state records when the
next task may use the data. Task files only order these resource work calls.

Schedule phase terms follow TS schedule-builder naming. HEAD is the one-time
schedule before the repeated K/V tile loop, LOOP is the repeated K/V tile body,
and TAIL is the one-time cleanup and drain after LOOP exits.

SMEM resources (TMA pipelines)
------------------------------
- SmemQResource   : SMEM Q buffer, TmaUmma pipeline. Load -> MMA.
- SmemKVResource  : SMEM KV buffer, TmaUmma pipeline. Load -> MMA. The D256
                    depth is derived from the public SM100 SMEM capacity.
- SmemOResource   : One SMEM O buffer per Q/O instance, AsyncAsync pipeline.
                    Correction -> Epilogue.

TMEM resources (split from former TmemComputeResource)
------------------------------------------------------
- TmemSPResource  : S/P buffer, UmmaAsync. D256 uses a two-stage S/P ring and
                    an independent TmemPResource readiness handoff.
                    MMA writes S (Q*K scores). Softmax reads S, computes P,
                    and writes P back into the same slot for BMM2.
                    Self-edge in dependency graph enables ping-pong validation.
- TmemStatsResource : Correction statistics, AsyncAsync pipeline.
                    Softmax writes [old_max, new_max, row_sum], Correction reads.
- TmemOResource   : O accumulation, UmmaAsync pipeline.
                    MMA writes P*V -> O. Correction waits for O, rescales it
                    in-place, then releases the stage for the interleaved MMA.

Sequencing resources
--------------------
- S0S1SequenceResource : PipelineAsync (1 stage), Softmax0 → Softmax1.
                         Ensures S0 finishes P store to TMEM before S1 starts
                         P computation.  Prevents TMEM write contention.
                         Operations are inlined in TmemSPResource.consumer_work.

GMEM resources (no pipeline)
-----------------------------
- GmemQKVResource : TMA descriptors + per-tile coordinate resolution.
- GmemOResource   : TMA descriptor for O stores.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any, Optional, TypeAlias

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from ..tensor_map import transform_ragged_coords

from cutlass.experimental.task_scheduling.enums import WorkAttr
from ..stage import FmhaStage
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
)
from cutlass.experimental.task_scheduling.resources import consumer_work, producer_work


_SUPPORTED_CONTEXT_PAGE_SIZES = (16, 32, 64, 128)
from cutlass.pipeline import PipelineAsync, PipelineState
from cutlass.cutlass_dsl import Boolean, Constexpr, dsl_user_op, if_generate

from ..placeholder_helpers import _placeholder_smem_array, _placeholder_tmem_ptr
from .helpers import (
    bottom_right_window_left_bound,
    bottom_right_window_tile_start,
    freeze_smem_descriptor,
    variable_window_cta_min_start,
)
from cutlass.experimental import primitives as prims

SmemDescOffsets: TypeAlias = tuple[int, int]
TmemAddr: TypeAlias = int | Int32
TmemPtr: TypeAlias = cutlass.Array

SoftmaxScalar: TypeAlias = Float32
SoftmaxChunk: TypeAlias = cutlass.Vector
SoftmaxChunks: TypeAlias = list[SoftmaxChunk]
SoftmaxRowSumContribution: TypeAlias = SoftmaxChunks | SoftmaxScalar

# Trace-time storage for TmemSP s_data vectors (S/P chunks).
# Stored at module level (not on self) to avoid adding a non-dynamic-expression
# field to the dataclass, which breaks the framework's scf.if handling.
_tmem_sp_sdata: dict[int, list] = {}


@cute.jit
def _bmsk_clamp(start: Int32, width: Int32) -> Int32:
    """Create a contiguous 32-bit mask with clamped bounds."""
    return cute.arch.inline_ptx(
        "bmsk.clamp.b32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Int32],
        read_only_args=[start, width],
    )


@cute.jit
def _mask_score_quad(
    valid_bits: Int32,
    score0: Float32,
    score1: Float32,
    score2: Float32,
    score3: Float32,
) -> tuple[Float32, Float32, Float32, Float32]:
    """Expand four bitmap bits with setp and replace invalid scores."""
    return cute.arch.inline_ptx(
        """
        {
            .reg .pred valid<4>;
            .reg .b32 bit;
            mov.b32 {$w0}, {$r1};
            mov.b32 {$w1}, {$r2};
            mov.b32 {$w2}, {$r3};
            mov.b32 {$w3}, {$r4};
            and.b32 bit, {$r0}, 0x1;
            setp.ne.u32 valid0, bit, 0;
            and.b32 bit, {$r0}, 0x2;
            setp.ne.u32 valid1, bit, 0;
            and.b32 bit, {$r0}, 0x4;
            setp.ne.u32 valid2, bit, 0;
            and.b32 bit, {$r0}, 0x8;
            setp.ne.u32 valid3, bit, 0;
            @!valid0 mov.b32 {$w0}, 0xff800000;
            @!valid1 mov.b32 {$w1}, 0xff800000;
            @!valid2 mov.b32 {$w2}, 0xff800000;
            @!valid3 mov.b32 {$w3}, 0xff800000;
        }
        """,
        write_only_types=[Float32, Float32, Float32, Float32],
        read_only_args=[valid_bits, score0, score1, score2, score3],
    )


@cute.jit
def _pack_float4_to_fp8_e4m3(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
) -> Int32:
    """Pack four FP32 values with the public packed-conversion primitive."""
    lo = prims.cvt_packfloat_f32(
        v1,
        v0,
        Int32(0),
        prims.CVTPackFloat.E4M3X2,
        rnd=prims.FPRoundingMode.RN,
        sat=prims.SaturationModeKind.SATFINITE,
        extract_hi=False,
    )
    return prims.cvt_packfloat_f32(
        v3,
        v2,
        lo,
        prims.CVTPackFloat.E4M3X2,
        rnd=prims.FPRoundingMode.RN,
        sat=prims.SaturationModeKind.SATFINITE,
        extract_hi=True,
    )


def _placeholder_softmax_chunks(cfg: Any) -> SoftmaxChunks:
    """Build zero P chunks with the same structure as runtime softmax chunks."""
    try:
        tmem_x = cfg.tmem_x_load_s
        num_chunks = cfg.qk_mma_tiler[1] // tmem_x
        chunks = []
        for _ in range(num_chunks):
            zeros = tuple(cfg.qk_acc_dtype(0.0) for _ in range(tmem_x))
            chunks.append(cutlass.Vector.from_elements(zeros, cfg.qk_acc_dtype))
        return chunks
    except RuntimeError:
        return []


# ---------------------------------------------------------------------------
# FmhaConfig -- kernel-wide configuration
# ---------------------------------------------------------------------------


@dataclass
class FmhaConfig:
    """Compile-time and runtime configuration for the FMHA kernel.

    Mirrors the attributes from BlackwellFusedMultiHeadAttentionForward.__init__
    and _setup_attributes, collected into a single portable dataclass.

    All fields are marked Constexpr so that tree_flatten does not try to
    recursively extract MLIR values from dtype classes or plain Python ints/tuples.
    """

    # Data types
    q_dtype: type | None = None
    k_dtype: type | None = None
    v_dtype: type | None = None
    o_dtype: type | None = None
    qk_acc_dtype: type | None = None
    pv_acc_dtype: type | None = None

    # Tile shapes
    qk_mma_tiler: tuple[int, int, int] = (128, 128, 64)
    pv_mma_tiler: tuple[int, int, int] = (128, 64, 128)
    epi_tile: tuple[int, int] = (128, 64)
    # Number of interleaved Q/KV/O instances per CTA: two selects the paired
    # schedule, while one selects the single-instance schedule used for D>128.
    num_qkv_instances: int = 2

    # Pipeline stages
    q_stage: int = 2
    kv_stage: int = 3
    mma_softmax_stage: int = 1
    # Use the two-stage loop-carried S/P schedule and an independent P-ready
    # handoff, allowing QK(i+1) and PV(i) to operate on opposite S/P stages.
    has_tmem_p_pipeline: bool = False
    stats_via_smem: bool = False
    stage_scoped_tmem_stats: bool = False
    softmax_corr_stage: int = 1
    mma_corr_stage: int = 2

    # TMA copy granularity
    tma_copy_qkv_iters: int = 1
    tma_copy_q_granu_inner: int = 128
    tma_copy_q_elements: int = 0
    tma_copy_q_granu_elems: int = 0
    tma_copy_q_bytes: int = 0
    tma_copy_kv_granu_inner: int = 128
    tma_copy_kv_elements: int = 0
    tma_copy_kv_granu_elems: int = 0
    tma_copy_kv_bytes: int = 0
    tma_copy_o_iters: int = 1
    tma_copy_o_granu_inner: int = 0
    tma_copy_o_elements: int = 0
    tma_copy_o_granu_elems: int = 0
    q_tile_m: int = 128
    kv_tile_n: int = 128

    # Warp assignments
    softmax0_warp_ids: tuple[int, int, int, int] = (0, 1, 2, 3)
    softmax1_warp_ids: tuple[int, int, int, int] = (4, 5, 6, 7)
    correction_warp_ids: tuple[int, int, int, int] = (8, 9, 10, 11)
    mma_warp_id: int = 12
    load_warp_id: int = 13
    epilogue_warp_id: int = 14
    empty_warp_id: int = 15

    # Register budgets
    num_regs_softmax: int = 192
    num_regs_correction: int = 96
    num_regs_other: int = 32

    # TMEM layout
    tmem_alloc_cols: int = 512
    tmem_stats_cols: int = 4
    tmem_s0_offset: int = 0
    tmem_s1_offset: int = 128
    tmem_o0_offset: int = 256
    tmem_o1_offset: int = 384
    tmem_p0_offset: int = 32
    tmem_p1_offset: int = 160
    tmem_vec0_offset: int = 0
    tmem_vec1_offset: int = 128

    # SMEM shapes (set during __init__ of FmhaTs)
    sO_stage_elements: int = 0
    sQ_shape: tuple[int, int] = (2, 0)
    sK_shape: tuple[int, int] = (3, 0)

    # Misc
    buffer_align_bytes: int = 1024
    tmem_bar_id: int = 2
    cluster_shape_mn: tuple[int, int] = (1, 1)
    block_warps: int = 16

    # GQA: head ratio h_q // h_kv (1 = MHA, >1 = GQA)
    h_r: int = 1

    # Causal masking: when True, mask out positions where k_idx > q_idx
    is_causal: bool = False
    # Explicit packed-Q inclusive [start, end] bounds replace static masks.
    has_variable_window: bool = False
    # Causal balancing uses head_batch_seq logical tile order and reverses Q
    # sequence tiles.
    balance_causal_workload: bool = False
    num_seq_tiles: int | Int32 = 0
    # Skip correction optimization: when True, skip rescale if old_max == new_max
    enable_skip_correction: bool = True

    # Variable sequence length mode stores Q/K/V/O as flattened
    # [sum_seqlen, head, dim] tensors and uses cum_seqlen_* for per-batch
    # sequence offsets.
    has_varlen: bool = False
    # Uniform packed plans retain the ragged tensor-map ABI but derive their
    # cumulative offsets arithmetically, avoiding dependent indptr loads on
    # every persistent work tile.
    has_uniform_varlen: bool = False
    uniform_seq_len_q: int = 0
    uniform_seq_len_k: int = 0

    # When true, map each work tile to two Q heads sharing one K/V head.
    # This enables the grouped-query/sliding-window context flavor while
    # reusing the unified FMHA context implementation.
    head_paired: bool = False

    # Concrete kernel policy derived from paired geometry and V dtype.  The
    # resource consumes this flag directly so the row-sum algorithm and the
    # register budget selected by FmhaTs cannot diverge.
    enable_early_tile_sum: bool = False

    seq_tile_n: int = 128
    tmem_x_load_s: int = 32
    # Causal S_q < S_kv shifts Q rows right by q_offset = S_kv - S_q.
    # This flag selects the shifted causal mask; there is no second causal mode.
    has_q_offset: bool = False
    # Fixed causal attention with exactly one K/V tile does not need the
    # synthetic peer0 tail slot used by the general query-paired schedule.
    causal_single_kv_tile: bool = False
    window_size_left: int = 0
    # Number of valid K/V rows in the final fixed-length dense tile. Zero
    # means that the K/V extent is tile-aligned (or that this specialization
    # does not use the fixed dense-tail mask).
    fixed_dense_k_tail: int = 0
    # Packed-contiguous and paged dense attention normally mask scores past
    # each request's logical K length. Plans with uniform, tile-aligned K
    # lengths can compile that mask away because every K tile is fully valid.
    packed_dense_k_mask: bool = True

    # ------------------------------------------------------------------
    # Paged KV cache (vLLM-style logical->physical page indirection)
    #
    # When use_paged_kv is True, K/V live in a fixed-size page pool
    # [num_pages_in_pool, h_kv, num_tokens_per_page, d] and the kernel follows
    # a fixed row-strided block table to resolve logical (b, s) -> physical page
    # id at TMA-issue time.
    #
    # Staged D256 assigns page-offset prefetch to its empty/padding warp.
    # Paired D128 reads page IDs directly from the page table in its load task.
    # ------------------------------------------------------------------
    use_paged_kv: bool = False
    # D256 uses a single Q/KV instance and can issue the final O TMA store
    # from one correction warp after the four-warp correction group has
    # staged O.  This frees the standalone epilogue warp for scheduling.
    fuse_epilogue_into_correction: bool = False
    num_tokens_per_page: int = 32
    # Static upper bound derived from max_kv_len during kernel construction.
    # Runtime active-page bounds come from seq_lens_kv.
    max_num_pages_per_seq_kv: int = 1
    page_offsets_num_warps: int = 1
    # Selected internally from the staged topology, static page geometry, and
    # exact SMEM capacity. It is derived during kernel construction and is not
    # a public tuning input.
    page_table_window_entries: int = 32

    # Work-tile mapping for the two peer Q/O tiles handled by each CTA:
    # query-paired maps peers to two sequence tiles in one Q head, while
    # head-paired mode maps peers to two Q heads at one sequence tile.
    @property
    def single_qkv_instance(self) -> bool:
        """Return whether one work tile carries a single Q/KV/O instance."""
        return self.num_qkv_instances == 1

    @property
    def uses_early_tile_sum(self) -> bool:
        """Return whether paired M128 geometry supports early V-tile reduction."""
        return (
            not self.single_qkv_instance
            and self.q_tile_m == 128
            and self.v_dtype
            in (
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
            )
        )

    @property
    def uses_d256_fp8_softmax_cadence(self) -> bool:
        """Return whether staged D256 FP8 uses interleaved softmax retirement."""
        return (
            self.single_qkv_instance
            and self.has_tmem_p_pipeline
            and self.stage_kv_by_head_dim
            and self.qk_mma_tiler == (128, 128, 256)
            and self.q_dtype == cutlass.Float8E4M3FN
            and self.k_dtype == cutlass.Float8E4M3FN
            and self.v_dtype == cutlass.Float8E4M3FN
        )

    @property
    def uses_d128_fp8_softmax_cadence(self) -> bool:
        """Return whether paired D128 FP8 uses interleaved softmax retirement."""
        return (
            not self.stage_kv_by_head_dim
            and self.enable_early_tile_sum
            and self.q_dtype == cutlass.Float8E4M3FN
            and self.k_dtype == cutlass.Float8E4M3FN
            and self.v_dtype == cutlass.Float8E4M3FN
        )

    @property
    def reuses_page_table_windows(self) -> bool:
        """Whether dense paged-KV admits structural page-ID windows."""
        if (
            not self.use_paged_kv
            or self.is_causal
            or self.num_tokens_per_page <= 0
            or self.kv_tile_n % self.num_tokens_per_page != 0
        ):
            return False
        pages_per_tile = self.kv_tile_n // self.num_tokens_per_page
        window_entries = self.page_table_window_entries
        if (
            pages_per_tile <= 0
            or pages_per_tile > window_entries
            or window_entries % pages_per_tile != 0
        ):
            return False
        window_period = window_entries // pages_per_tile
        if self.single_qkv_instance and self.has_tmem_p_pipeline and window_period < 3:
            # The K-ahead/V-delayed HEAD requires distinct K0, K1, and tail
            # positions. Smaller periods retain the ordinary per-tile path.
            return False
        static_num_kv_tiles = (
            self.max_num_pages_per_seq_kv + pages_per_tile - 1
        ) // pages_per_tile
        return (
            static_num_kv_tiles >= window_period
            and static_num_kv_tiles % window_period == 0
        )

    @property
    def stages_page_offsets_in_smem(self) -> bool:
        """Whether a dedicated warp stages page IDs for the load warp.

        Staged D256 uses the coalesced SMEM page-window path. Paired D128 loads
        page IDs directly in its K/V producer.
        """
        return self.use_paged_kv and self.single_qkv_instance

    @property
    def page_table_window_candidate_entries(self) -> int:
        """Return the widest page-ID window admitted by static topology.

        A split-D K/V schedule consumes two head-dimension stages for each
        logical tile.  When the static domain can cover a complete window,
        let each producer lane fetch one ID per D stage so one page-window
        handoff spans both stages. Short domains admit only the natural
        one-warp window. The kernel's capacity pass makes the final selection.
        """
        natural_entries = cute.arch.WARP_SIZE
        if (
            not self.use_paged_kv
            or self.is_causal
            or self.num_tokens_per_page <= 0
            or not (self.single_qkv_instance and self.has_tmem_p_pipeline)
        ):
            return natural_entries
        pages_per_tile = self.kv_tile_n // self.num_tokens_per_page
        staged_entries = natural_entries * self.num_head_dim_stages_k
        staged_period = staged_entries // pages_per_tile
        static_num_kv_tiles = (
            self.max_num_pages_per_seq_kv + pages_per_tile - 1
        ) // pages_per_tile
        if (
            staged_entries % pages_per_tile == 0
            and static_num_kv_tiles >= staged_period
            and static_num_kv_tiles % staged_period == 0
        ):
            return staged_entries
        return natural_entries

    @property
    def page_offset_pipeline_stage_counts(self) -> tuple[int, ...]:
        """Return the physical page-ID ring depths for this topology."""
        if not self.use_paged_kv or not self.single_qkv_instance:
            return ()
        # The staged schedule holds one credit for every K/V head-dimension
        # slice plus one K-ahead boundary credit.  A reused page-table window
        # needs independent K-ahead and V-delayed rings; the ordinary path
        # shares the same total number of credits. Paired D128 loads page IDs
        # directly in its load task and therefore has no physical page ring.
        k_stages = self.num_head_dim_stages_k + 1
        v_stages = self.num_head_dim_stages_v
        if self.reuses_page_table_windows:
            return (k_stages, v_stages)
        return (k_stages + v_stages,)

    @property
    def cta_tiler(self) -> tuple[int, int, int]:
        """Derive the CTA tile from the MMA tile and work-tile mapping."""
        if self.single_qkv_instance or self.head_paired:
            return self.qk_mma_tiler
        return (
            self.num_qkv_instances * self.qk_mma_tiler[0],
            self.qk_mma_tiler[1],
            self.qk_mma_tiler[2],
        )

    @property
    def uses_causal_reversed_head_batch_seq_tile_order(self) -> bool:
        """Return whether causal head_batch_seq tiles reverse Q sequence order."""
        return self.is_causal and self.balance_causal_workload

    @property
    def uses_paired_fp8_head_batch_seq_tile_order(self) -> bool:
        """Return whether paired FP8 benefits from head-local tile order.

        Causal work uses this order for load balancing. Dense GQA uses it to
        keep Q-head groups that share the same K/V head adjacent. Dense MHA
        has no cross-head K/V reuse and retains its sequence-local order.
        """
        return (
            (self.is_causal or self.h_r > 1)
            and not self.single_qkv_instance
            and self.q_dtype is not None
            and self.k_dtype is not None
            and self.v_dtype is not None
            and self.q_dtype.width == 8
            and self.k_dtype.width == 8
            and self.v_dtype.width == 8
        )

    @property
    def uses_head_batch_seq_tile_order(self) -> bool:
        """Return whether work tiles use head_batch_seq coordinates."""
        return self.uses_paired_fp8_head_batch_seq_tile_order or (
            self.is_causal and self.balance_causal_workload
        )

    @property
    def work_tile_coord_indices(self) -> tuple[int, int, int]:
        """Return work-tile indices for logical ``(seq, head, batch)``."""
        if self.uses_head_batch_seq_tile_order:
            return 2, 0, 1
        return 0, 1, 2

    @property
    def pv_p_scale(self) -> float:
        """Return the P scale applied before PV MMA."""
        if self.v_dtype is not None and self.v_dtype.width == 8:
            # FP8 E4M3 has max finite magnitude 448; scaling P to that range
            # before PV MMA preserves dynamic range.
            return 448.0
        # Non-FP8 V uses P directly, so the PV-side P scale is identity.
        return 1.0

    @property
    def pv_p_scale_log2(self) -> float:
        """Return log2(P scale) for folding into exp2 softmax P."""
        return math.log2(self.pv_p_scale)

    @property
    def work_tile_q_heads(self) -> int:
        """Return the number of Q heads represented by one work tile."""
        if self.single_qkv_instance:
            return 1
        return 2 if self.head_paired else 1

    @property
    def work_tile_q_seq_tiles(self) -> int:
        """Return the number of Q sequence tiles represented by one work tile."""
        if self.single_qkv_instance:
            return 1
        return 1 if self.head_paired else 2

    @property
    def has_tile_aligned_uniform_q_offset(self) -> bool:
        """Whether a uniform causal shift preserves K/V tile boundaries.

        Uniform packed plans retain fixed Q/K lengths under their replay
        contract.  When their bottom-right shift is an exact K/V-tile
        multiple, every query tile's causal diagonal has the same placement
        as the zero-offset schedule: query-paired peer 0 can use its explicit
        diagonal/invalid-tail protocol, and all other diagonals remain in
        TAIL.  No LOOP iteration then needs a causal right mask.
        """
        return (
            self.has_q_offset
            and self.has_uniform_varlen
            and self.uniform_seq_len_q % self.cta_tiler[0] == 0
            and (self.uniform_seq_len_k - self.uniform_seq_len_q) % self.kv_tile_n == 0
        )

    @property
    def peer_q_head_stride(self) -> int:
        """Return the Q-head stride between the two peer Q/O tiles."""
        return 1 if self.head_paired and not self.single_qkv_instance else 0

    @property
    def peer_q_seq_tile_stride(self) -> int:
        """Return the Q-sequence tile stride between the two peer Q/O tiles."""
        return 0 if self.head_paired or self.single_qkv_instance else 1

    @property
    def gmem_o_store_wait_after_write(self) -> bool:
        """Return whether each O store must wait for the matching SMEM write."""
        return self.head_paired or self.stage_o_by_head_dim

    @property
    def skip_causal_invalid_peer0(self) -> bool:
        """Return whether query-paired causal peer0 may skip extra loop work."""
        if (
            not self.is_causal
            or self.head_paired
            or (self.has_q_offset and not self.has_tile_aligned_uniform_q_offset)
            or self.single_qkv_instance
            or self.causal_single_kv_tile
        ):
            return False
        peer0_kv_tiles = (self.q_tile_m + self.kv_tile_n - 1) // self.kv_tile_n
        paired_kv_tiles = (self.cta_tiler[0] + self.kv_tile_n - 1) // self.kv_tile_n
        extra_peer1_kv_tiles = paired_kv_tiles - peer0_kv_tiles
        if extra_peer1_kv_tiles > 2:
            raise ValueError(
                "query-paired causal scheduling supports peer1 at most two "
                "K/V tiles ahead of peer0; got "
                f"{extra_peer1_kv_tiles} extra K/V tiles"
            )
        return extra_peer1_kv_tiles > 0

    @property
    def kv_tile_start_window_size_left(self) -> int:
        """Return the left-window width used to compute the first K/V tile."""
        return self.window_size_left if self.head_paired else 0


# ---------------------------------------------------------------------------
# S0S1SequenceResource -- S0-S1 sequence barrier (PipelineAsync, 1 stage)
# ---------------------------------------------------------------------------


@cute.jit
def _resolve_work_tile_coords(
    cfg: Constexpr[FmhaConfig],
    tile_idx: cute.Coord,
) -> tuple[Int32, Int32, Int32]:
    """Return ``(seq, head, batch)`` for the configured tile order."""
    seq_idx, head_idx, batch_idx = cfg.work_tile_coord_indices
    seq_coord = tile_idx[seq_idx]
    head_coord = tile_idx[head_idx]
    batch_coord = tile_idx[batch_idx]
    if cutlass.const_expr(cfg.uses_causal_reversed_head_batch_seq_tile_order):
        seq_coord = cfg.num_seq_tiles - seq_coord - Int32(1)
    return seq_coord, head_coord, batch_coord


@dataclass(frozen=True)
class _StructuredWaitPipelineAsync(PipelineAsync):
    """PipelineAsync with an explicit public-primitive retry loop."""

    @cute.jit
    def _retry_wait(
        self,
        sync_object: object,
        state: PipelineState,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        while not sync_object.try_wait(
            state.index,
            state.phase,
            loc=loc,
            ip=ip,
        ):
            pass

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self._retry_wait(self.sync_object_empty, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_wait(
        self,
        state: PipelineState,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self._retry_wait(self.sync_object_full, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dataclass(kw_only=True)
class S0S1SequenceResource(MemoryResource):
    """Sequence barrier between Softmax0 (producer) and Softmax1 (consumer).

    Prevents both softmax groups from writing P to TMEM simultaneously.
    Matches the handwritten FMHA kernel's ``s0_s1_sequence_mbar``
    PipelineAsync.

    Single resource instance shared across tasks:
      - Softmax0's dst_resource (ProducerAcquire/Commit)
      - Softmax1's src_resource (ConsumerWait/Release)
    """

    is_barrier: cutlass.Constexpr[bool] = True

    def create_pipeline(self, pipeline_config: PipelineConfig) -> object:
        base = super().create_pipeline(pipeline_config)
        assert isinstance(base, PipelineAsync)
        return _StructuredWaitPipelineAsync(
            base.sync_object_full,
            base.sync_object_empty,
            base.num_stages,
            base.producer_mask,
            base.consumer_mask,
        )


# ---------------------------------------------------------------------------
# TmemStatsDoneResource -- cross-tile stats-read notification
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemStatsDoneResource(MemoryResource):
    """Notification barrier: Correction signals after reading stats from TMEM.

    Prevents cross-tile aliasing race where next tile's QK→S UMMA can
    overwrite TMEM columns overlapping TmemStats0/1 before correction reads them.

    Single resource instance shared across tasks:
      - MMA's dst_resource (ProducerAcquire/Commit)
      - Correction's src_resource (ConsumerWait/Release)
    """

    is_barrier: cutlass.Constexpr[bool] = True


# ---------------------------------------------------------------------------
# GmemQKVResource -- global memory Q/K/V source (no pipeline)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GmemQKVResource(MemoryResource):
    """Provides TMA descriptors and per-tile coordinates for Q/K/V loads.

    Consumer side resolves batch/head/seq coordinates from the work tile
    so that downstream SmemQ/SmemKV producer_work can issue TMA loads.
    """

    tma_q_desc: cutlass.Pointer | None = field(init=False, default=None)
    tma_k_desc: cutlass.Pointer | None = field(init=False, default=None)
    tma_v_desc: cutlass.Pointer | None = field(init=False, default=None)
    cum_seqlen_q: cute.Tensor | None = field(init=False, default=None)
    cum_seqlen_k: cute.Tensor | None = field(init=False, default=None)
    variable_window_token_starts: cute.Tensor | None = field(init=False, default=None)
    variable_window_cta_starts: cute.Tensor | None = field(init=False, default=None)
    variable_window_q_stride: int | Int32 = field(init=False, default=0)
    q_offset_default: int | Int32 = field(init=False, default=0)
    seqlens_kv: cute.Pointer | None = field(init=False, default=None)
    block_table_row_stride: int | Int32 = field(init=False, default=0)
    max_seq_len_kv: Optional[Int32 | int] = field(init=False, default=None)
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    seq_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    head_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_head_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    head_coord_kv: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    batch_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seq_coord_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cuseqlen_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cuseqlen_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seqlen_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seqlen_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_tile_start: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_request_begin: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_page_idx_ub: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        tma_q_desc: cutlass.Pointer | None,
        tma_k_desc: cutlass.Pointer | None,
        tma_v_desc: cutlass.Pointer | None,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        q_offset: int | Int32,
        cfg: FmhaConfig,
        seqlens_kv: cute.Pointer | None = None,
        block_table_row_stride: int | Int32 = 0,
        max_seq_len_kv: Int32 | int | None = None,
        variable_window_token_starts: cute.Tensor | None = None,
        variable_window_cta_starts: cute.Tensor | None = None,
        variable_window_q_stride: int | Int32 = 0,
        **kwargs: Any,
    ) -> None:
        """Bind Q/K/V descriptors, optional varlen metadata, and FMHA config."""
        super().__init__(**kwargs)
        self.tma_q_desc = tma_q_desc
        self.tma_k_desc = tma_k_desc
        self.tma_v_desc = tma_v_desc
        self.cum_seqlen_q = cum_seqlen_q
        self.cum_seqlen_k = cum_seqlen_k
        self.q_offset_default = q_offset
        self.seqlens_kv = seqlens_kv
        self.block_table_row_stride = block_table_row_stride
        self.max_seq_len_kv = max_seq_len_kv
        self.variable_window_token_starts = variable_window_token_starts
        self.variable_window_cta_starts = variable_window_cta_starts
        self.variable_window_q_stride = variable_window_q_stride
        self.cfg = cfg
        self.seq_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/K/V tile sequence coordinate.",
        )
        self.head_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/O head coordinate for the current work tile.",
        )
        self.kv_head_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K/V head coordinate for the current work tile.",
        )
        self.head_coord_kv = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K/V head coordinate mirrored for master FMHA context schedules.",
        )
        self.batch_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Batch coordinate for the current work tile.",
        )
        self.seq_coord_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/O row coordinate for the current work tile.",
        )
        self.cuseqlen_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q sequence cumulative offset for variable-length FMHA.",
        )
        self.cuseqlen_k = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K sequence cumulative offset for variable-length FMHA.",
        )
        self.seqlen_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q sequence length for variable-length FMHA.",
        )
        self.seqlen_k = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K sequence length for variable-length FMHA.",
        )
        self.kv_tile_start = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="First K/V loop tile for sliding-window FMHA.",
        )
        self.kv_request_begin = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Element offset of the request's block-table row.",
        )
        self.kv_page_idx_ub = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Inclusive logical-page upper bound for the request.",
        )

    @consumer_work(
        returns=(
            seq_coord,
            head_coord,
            kv_head_coord,
            head_coord_kv,
            batch_coord,
            seq_coord_q,
            cuseqlen_q,
            cuseqlen_k,
            seqlen_q,
            seqlen_k,
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_ub,
        )
    )
    @cute.jit
    def compute_coords(
        self, stage_info: StageInfo
    ) -> tuple[
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
    ]:
        """Resolve per-tile coordinates from work_tile for downstream use.

        Populates consumer variables with the batch/head/seq coordinates
        that SmemQ, SmemK, and SmemV producer_work methods need for TMA loads.
        GQA: head_coord indexes Q/O heads (h_q), kv_head_coord indexes K/V
        heads (h_kv). For MHA (h_r=1) they are identical.
        """
        seq_coord, head_coord, batch_coord = _resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )

        kv_head_coord = (head_coord * self.cfg.work_tile_q_heads) // self.cfg.h_r
        seq_coord_q = seq_coord * self.cfg.q_tile_m * self.cfg.work_tile_q_seq_tiles
        head_coord_kv = kv_head_coord
        cuseqlen_q = Int32(0)
        cuseqlen_k = Int32(0)
        seqlen_q = Int32(0)
        seqlen_k = Int32(0)
        window_q_offset = Int32(self.q_offset_default)
        kv_tile_start = Int32(0)
        kv_request_begin = Int32(0)
        kv_page_idx_ub = Int32(0)
        if cutlass.const_expr(self.cfg.has_varlen):
            if cutlass.const_expr(self.cfg.has_uniform_varlen):
                seqlen_q = Int32(self.cfg.uniform_seq_len_q)
                cuseqlen_q = batch_coord * seqlen_q
            else:
                cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
                next_cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
                seqlen_q = next_cuseqlen_q - cuseqlen_q
            if cutlass.const_expr(self.cfg.use_paged_kv):
                # Paged K/V is addressed through a block table rather than a
                # packed token buffer, so it has no cumulative token offset.
                from .helpers_paged import _load_runtime_seq_len_kv

                seqlen_k = _load_runtime_seq_len_kv(
                    self.seqlens_kv, self.max_seq_len_kv, batch_coord
                )
            elif cutlass.const_expr(self.cfg.has_uniform_varlen):
                seqlen_k = Int32(self.cfg.uniform_seq_len_k)
                cuseqlen_k = batch_coord * seqlen_k
            else:
                cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord])
                next_cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord + Int32(1)])
                seqlen_k = next_cuseqlen_k - cuseqlen_k
            seq_coord_q = cuseqlen_q + seq_coord_q
            # Each packed request uses its own bottom-right window origin. For
            # mixed causal plans the task manager also derives the request's
            # K-loop extent from these runtime Q/K lengths.
            window_q_offset = seqlen_k - seqlen_q
            if cutlass.const_expr(
                self.cfg.use_paged_kv and not self.cfg.stages_page_offsets_in_smem
            ):
                from .helpers_paged import _load_block_table_row_bounds

                kv_request_begin, kv_page_idx_ub = _load_block_table_row_bounds(
                    Int32(self.block_table_row_stride),
                    self.cfg,
                    seqlen_k,
                    batch_coord,
                )
        if cutlass.const_expr(self.cfg.kv_tile_start_window_size_left > 0):
            if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
                kv_tile_start = bottom_right_window_tile_start(
                    seq_coord=seq_coord,
                    q_tile_m=self.cfg.q_tile_m,
                    kv_tile_n=self.cfg.seq_tile_n,
                    q_offset=window_q_offset,
                    window_size_left=self.cfg.kv_tile_start_window_size_left,
                )
            else:
                # Preserve the minimal fixed equal-length specialization: its
                # bottom-right offset is statically zero.
                kv_tile_start = cute.math.max(
                    Int32(0),
                    (
                        seq_coord * self.cfg.q_tile_m
                        - self.cfg.kv_tile_start_window_size_left
                    )
                    // self.cfg.seq_tile_n,
                )
        if cutlass.const_expr(self.cfg.has_variable_window):
            min_window_start = variable_window_cta_min_start(
                self.variable_window_cta_starts,
                batch_coord=batch_coord,
                seq_coord=seq_coord,
                q_stride=self.variable_window_q_stride,
                tile_size_q=self.cfg.cta_tiler[0],
            )
            kv_tile_start = min_window_start // self.cfg.kv_tile_n
        return (
            seq_coord,
            head_coord,
            kv_head_coord,
            head_coord_kv,
            batch_coord,
            seq_coord_q,
            cuseqlen_q,
            cuseqlen_k,
            seqlen_q,
            seqlen_k,
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_ub,
        )

    @consumer_work(
        returns=(
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_ub,
        )
    )
    @cute.jit
    def compute_page_coords(self, stage_info: StageInfo) -> tuple[Int32, Int32, Int32]:
        """Resolve only the coordinates needed by paged-KV prefetch.

        The page-offset warp does not consume Q/head coordinates. Keeping its
        coordinate path narrow avoids materializing the full Q/K/V coordinate
        tuple once per persistent work tile.
        """
        seq_coord, _head_coord, batch_coord = _resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )

        from .helpers_paged import _load_runtime_seq_len_kv

        cached_seqlen_kv = _load_runtime_seq_len_kv(
            self.seqlens_kv, self.max_seq_len_kv, batch_coord
        )
        from .helpers_paged import _load_block_table_row_bounds

        kv_request_begin, kv_page_idx_ub = _load_block_table_row_bounds(
            Int32(self.block_table_row_stride),
            self.cfg,
            cached_seqlen_kv,
            batch_coord,
        )
        window_q_offset = Int32(self.q_offset_default)
        kv_tile_start = Int32(0)
        if cutlass.const_expr(self.cfg.kv_tile_start_window_size_left > 0):
            if cutlass.const_expr(self.cfg.has_varlen):
                if cutlass.const_expr(self.cfg.has_uniform_varlen):
                    seqlen_q = Int32(self.cfg.uniform_seq_len_q)
                else:
                    cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
                    next_cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
                    seqlen_q = next_cuseqlen_q - cuseqlen_q
                window_q_offset = cached_seqlen_kv - seqlen_q
            if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
                kv_tile_start = bottom_right_window_tile_start(
                    seq_coord=seq_coord,
                    q_tile_m=self.cfg.q_tile_m,
                    kv_tile_n=self.cfg.seq_tile_n,
                    q_offset=window_q_offset,
                    window_size_left=self.cfg.kv_tile_start_window_size_left,
                )
            else:
                kv_tile_start = cute.math.max(
                    Int32(0),
                    (
                        seq_coord * self.cfg.q_tile_m
                        - self.cfg.kv_tile_start_window_size_left
                    )
                    // self.cfg.seq_tile_n,
                )

        return (
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_ub,
        )


def _qkv_inner_dim_size_bytes(cfg: FmhaConfig) -> int:
    """Return the byte width of one Q/K/V tile inner dimension."""
    return cfg.qk_mma_tiler[2] * cfg.q_dtype.width // 8


def _qkv_smem_layout(cfg: FmhaConfig) -> int:
    """Return the tcgen05 descriptor layout selector for Q/K/V SMEM tiles."""
    inner_dim_size = _qkv_inner_dim_size_bytes(cfg)
    if inner_dim_size % 128 == 0:
        return 2
    if inner_dim_size == 64:
        return 4
    if inner_dim_size == 32:
        return 6
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")


def _qk_smem_desc_offsets(cfg: FmhaConfig) -> SmemDescOffsets:
    """Return Q/K descriptor leading and stride byte offsets."""
    leading_byte_offset = 0 if cfg.head_paired else 16
    stride_byte_offset = (
        cfg.qk_mma_tiler[2] * cfg.q_dtype.width // cfg.tma_copy_qkv_iters
    )
    return leading_byte_offset, stride_byte_offset


def _pv_smem_desc_offsets(cfg: FmhaConfig) -> SmemDescOffsets:
    """Return V descriptor leading and stride byte offsets for PV MMA."""
    leading_byte_offset = 0
    if cfg.tma_copy_qkv_iters != 1:
        tma_copy_kv_iters = (
            cfg.tma_copy_kv_stage_iters
            if cfg.stage_kv_by_head_dim
            else cfg.tma_copy_qkv_iters
        )
        leading_byte_offset = cfg.tma_copy_kv_bytes // tma_copy_kv_iters
    stride_byte_offset = (
        cfg.pv_mma_tiler[1] * cfg.v_dtype.width // cfg.tma_copy_qkv_iters
    )
    return leading_byte_offset, stride_byte_offset


def _qkv_smem_swizzle(cfg: FmhaConfig) -> cutlass.Swizzle:
    """Return the physical TMA swizzle used by Q/K/V SMEM fragments."""
    inner_dim_size = _qkv_inner_dim_size_bytes(cfg)
    if inner_dim_size % 128 == 0:
        return cutlass.Swizzle(3, 4, 3)
    if inner_dim_size == 64:
        return cutlass.Swizzle(2, 4, 3)
    if inner_dim_size == 32:
        return cutlass.Swizzle(1, 4, 3)
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")


def _smem_o_swizzle(cfg: FmhaConfig) -> cutlass.Swizzle:
    """Return the shared-memory swizzle used when staging O for TMA store."""
    return _qkv_smem_swizzle(cfg)


# ---------------------------------------------------------------------------
# SmemQResource -- SMEM Q tile buffer with TMA pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemQResource(MemoryResource):
    """SMEM buffer for Q tiles with a topology-derived TmaUmma pipeline.

    Producer: LoadTask (TMA loads Q0 and Q1 in the first K-loop iteration).
    Consumer: MmaTask (builds SMEM descriptors, holds Q across K-loop).
    """

    sQ_array: cutlass.Array = field(init=False, default=None)
    tma_q_desc: cutlass.Pointer | None = field(init=False, default=None)
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    desc_q0_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_q1_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        tma_q_desc: cutlass.Pointer | None,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        **kwargs: Any,
    ) -> None:
        """Bind the Q TMA descriptor and reserve SMEM for staged Q tiles."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.tma_q_desc = tma_q_desc
        self.cfg = cfg
        total_elements = cfg.sQ_shape[0] * cfg.sQ_shape[1]
        size_bytes = total_elements * cfg.q_dtype.width // 8
        self._alloc = SmemAllocation(
            "smem_q", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sQ_array = _placeholder_smem_array(cfg.q_dtype)
        self.desc_q0_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the first Q half.",
        )
        self.desc_q1_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the second Q half.",
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation required for staged Q tiles."""
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Materialize the Q SMEM array and descriptor dataflow slots."""
        smem_base = stage_info.context.smem_base
        total_elements = self.cfg.sQ_shape[0] * self.cfg.sQ_shape[1]
        self.sQ_array = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=self.cfg.q_dtype,
            shape=(total_elements,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @producer_work
    @cute.jit
    def tma_load(
        self,
        stage_info: StageInfo,
        *,
        seq_coord_q: Int32,
        head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_q: Int32,
        seqlen_q: Int32,
        inst_idx: cutlass.Constexpr[int],
    ) -> None:
        """TMA load one Q tile (Q0 or Q1) from GMEM to SMEM.

        inst_idx 0 = Q0, inst_idx 1 = Q1.
        Uses seq_coord_q from producer variables (forwarded from GmemQKV).
        """
        q_head_coord = (
            head_coord * self.cfg.work_tile_q_heads
            + inst_idx * self.cfg.peer_q_head_stride
        )
        q_seq_offset = (
            seq_coord_q + inst_idx * self.cfg.peer_q_seq_tile_stride * self.cfg.q_tile_m
        )
        q_seq_extent = Int32(0)
        if cutlass.const_expr(self.cfg.has_varlen):
            q_seq_extent = cuseqlen_q + seqlen_q - q_seq_offset
        smem_stage_elements = self.cfg.tma_copy_q_elements
        d_granu_inner = self.cfg.tma_copy_q_granu_inner

        sQ_curr = self.sQ_array.subview(stage_info.stage_idx * smem_stage_elements)
        if prims.elect_sync():
            for i in cutlass.range_constexpr(self.cfg.tma_copy_qkv_iters):
                d_offset = i * d_granu_inner
                q_coords = (d_offset, q_head_coord, q_seq_offset, batch_coord)
                if cutlass.const_expr(self.cfg.has_varlen):
                    q_coords = (d_offset, q_head_coord, q_seq_offset)
                    q_coords = transform_ragged_coords(
                        q_coords,
                        ragged_dim_idx=2,
                        ragged_box_size=self.cfg.qk_mma_tiler[0],
                        ragged_extent=q_seq_extent,
                    )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sQ_curr.subview(i * self.cfg.tma_copy_q_granu_elems),
                    self.tma_q_desc,
                    q_coords,
                    stage_info.barrier,
                )

    def _build_q_descriptor(self, inst_idx: int) -> prims.Tcgen05SmemDesc:
        """Build SMEM descriptor for the current Q tile.

        Uses inst_idx (not stage_idx) to compute the SMEM offset because
        Q is consumed twice in HEAD without an intervening ConsumerRelease,
        which would otherwise advance consumer_state.
        """
        sQ_curr = self.sQ_array.subview(inst_idx * self.cfg.tma_copy_q_elements)
        leading_byte_offset, stride_byte_offset = _qk_smem_desc_offsets(self.cfg)
        return prims.Tcgen05SmemDesc.build(
            sQ_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=_qkv_smem_layout(self.cfg),
        )

    @consumer_work(returns=desc_q0_base)
    @cute.jit
    def q0_desc(
        self, stage_info: StageInfo, *, inst_idx: cutlass.Constexpr[int]
    ) -> prims.Tcgen05SmemDesc:
        """Build Q0 SMEM descriptor -> desc_q0_base."""
        return self._build_q_descriptor(inst_idx)

    @consumer_work(returns=desc_q1_base)
    @cute.jit
    def q1_desc(
        self, stage_info: StageInfo, *, inst_idx: cutlass.Constexpr[int]
    ) -> prims.Tcgen05SmemDesc:
        """Build Q1 SMEM descriptor -> desc_q1_base."""
        return self._build_q_descriptor(inst_idx)


# ---------------------------------------------------------------------------
# SmemPageOffsetsKvResource -- paged-KV page-table cache in SMEM
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemPageOffsetsKvResource(MemoryResource):
    """Paged-KV logical-to-physical page IDs staged in SMEM (context kernel).

    The staged D256 path uses a dedicated warp to prefetch page-table
    entries for the next K/V tile so the TMA load warp can read SMEM-cached
    offsets. Each pipeline stage holds one topology-derived page-ID window
    from the request's fixed-table row; all 32 lanes co-load it. ``page_ids`` slices
    ``pages_per_tile`` entries for the current tile.

    Differences from decode:
    - Single ``load_k`` / ``load_v`` producer pair (context has no
      ``num_insts_kv > 1`` four-way split).
    - Consumer release labels bind to ``{"k_load", "v_load"}`` (matching
      ``SmemKVResource`` producer names).
    - Driven by the staged D256 ``cfg.empty_warp_id`` (warp 11).
      Paired D128 does not instantiate this resource.
    """

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    block_tables: cute.Pointer | None = field(init=False, default=None)
    page_table_is_v: Constexpr[bool] = field(init=False, default=False)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    _smem_page_offsets: cutlass.Array = field(init=False, default=None)
    cached_page_ids: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        block_tables: cute.Pointer | None,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        page_table_is_v: bool = False,
        **kwargs: Any,
    ) -> None:
        # ``page_ids`` runs from the downstream K/V resource after this
        # resource's ConsumerWait. Preserve the waited stage so the nested
        # lookup reads the matching page-table data.
        pipeline_config = replace(pipeline_config, advance_on_wait=True)
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.block_tables = block_tables
        self.page_table_is_v = page_table_is_v
        num_stages = pipeline_config.num_stages
        total_entries = num_stages * cfg.page_table_window_entries
        self._alloc = SmemAllocation(
            "smem_page_offsets_v" if page_table_is_v else "smem_page_offsets_k",
            size_bytes=total_entries * 4,
            # Page-size 16 consumes eight page IDs per 128-token K/V tile.
            # Align only that specialization for one 32-byte vector load;
            # preserve the established layout for larger page sizes.
            alignment=(32 if cfg.kv_tile_n // cfg.num_tokens_per_page == 8 else 16),
        )
        self._smem_page_offsets = _placeholder_smem_array(Int32, total_entries)
        self.cached_page_ids = TaskLocalVariable(
            dtype=cutlass.Array,
            default=None,
            docs="Page IDs retained while a delayed V tile crosses a page window.",
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        smem_base = stage_info.context.smem_base
        num_stages = self.pipeline_config.num_stages
        total_entries = num_stages * self.cfg.page_table_window_entries
        self._smem_page_offsets = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=cutlass.Int32,
            shape=(total_entries,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=cached_page_ids)
    @cute.jit
    def init_cached_read_state(self, stage_info: StageInfo) -> cutlass.Array:
        """Initialize the SMEM view and register cache for one tile's page IDs."""
        self._init_smem_state(stage_info)
        return cutlass.Array(
            Int32,
            self.cfg.kv_tile_n // self.cfg.num_tokens_per_page,
            space=cutlass.AddressSpace.rmem,
        )

    @cute.jit
    def page_ids(self, tile_idx: Int32) -> cutlass.Array:
        """Slice ``pages_per_tile`` entries from the cached page-ID stage.

        ``tile_idx`` is the runtime-resolved K/V tile index (same expression
        the page-offsets producer uses). The window-aligned base is implicit
        in the stage's contents; this LDS picks the per-tile entries.
        """
        cfg = self.cfg
        pages_per_tile = cfg.kv_tile_n // cfg.num_tokens_per_page
        window_entries = cfg.page_table_window_entries
        stage_idx = self.state_src.consumer_work_stage
        group_page_idx = (tile_idx * Int32(pages_per_tile)) & Int32(window_entries - 1)
        offset = stage_idx * Int32(window_entries) + group_page_idx
        if cutlass.const_expr(pages_per_tile == 8):
            return self._smem_page_offsets.load(offset, vector_size=8, alignment=32)
        if cutlass.const_expr(pages_per_tile == 4):
            return self._smem_page_offsets.load(offset, vector_size=4, alignment=16)
        if cutlass.const_expr(pages_per_tile == 2):
            return self._smem_page_offsets.load(offset, vector_size=2, alignment=8)
        return self._smem_page_offsets.load(offset, vector_size=1, alignment=4)

    @cute.jit
    def _producer_load_page_offsets(
        self,
        stage_info: StageInfo,
        tile_offset: cutlass.Constexpr[int] = 0,
        *,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        from .helpers_paged import _resolve_kv_tile_idx_context

        cfg = self.cfg
        # Context's K/V tile index is kv_tile_start + loop_offset; for V the
        # producer runs one tile ahead in TAIL, so reuse the same expression
        # but consult kv_tile_start with the loop's stage_info.
        tile_idx = _resolve_kv_tile_idx_context(
            stage_info, kv_tile_start, tile_offset=tile_offset
        )
        pages_per_tile = Int32(cfg.kv_tile_n // cfg.num_tokens_per_page)

        block_tables = self.block_tables
        smem_page_offsets = self._smem_page_offsets
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        # Lanes cooperatively fetch one topology-derived aligned window. A
        # split-D window gives each lane one scalar per D stage; consumers then
        # slice per-tile entries via ``page_ids``.
        window_entries = cfg.page_table_window_entries
        grouped_base_page_idx = (
            (tile_idx * pages_per_tile) // Int32(window_entries)
        ) * Int32(window_entries)
        grouped_smem_base = stage_info.stage_idx * Int32(window_entries)
        entries_per_lane = window_entries // cute.arch.WARP_SIZE
        for lane_group in cutlass.range_constexpr(entries_per_lane):
            lane_offset = lane_idx + Int32(lane_group * cute.arch.WARP_SIZE)
            grouped_logical_page_idx = cute.math.min(
                grouped_base_page_idx + lane_offset, kv_page_idx_ub
            )
            prims.cp_async_shared_global(
                smem_page_offsets.data_ptr() + grouped_smem_base + lane_offset,
                block_tables + kv_request_begin + grouped_logical_page_idx,
                4,
                "ca",
            )

    @producer_work
    @cute.jit
    def load_k(
        self,
        stage_info: StageInfo,
        *,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Prefetch K-side page IDs for the current K tile."""
        self._producer_load_page_offsets(
            stage_info,
            tile_offset=tile_offset,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def load_v(
        self,
        stage_info: StageInfo,
        *,
        previous: cutlass.Constexpr[bool] = False,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Prefetch V-side page IDs for the current V tile."""
        self._producer_load_page_offsets(
            stage_info,
            tile_offset=tile_offset + (-1 if previous else 0),
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @consumer_work
    @cute.jit
    def read_offsets(self, stage_info: StageInfo) -> None:
        return

    @consumer_work(returns=cached_page_ids)
    @cute.jit
    def cache_tile_page_ids(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids: cutlass.Array,
        kv_tile_start: Int32,
        tile_offset: cutlass.Constexpr[int] = 0,
    ) -> cutlass.Array:
        """Retain one tile's page IDs after its SMEM window is released."""
        from .helpers_paged import _resolve_kv_tile_idx_context

        tile_idx = _resolve_kv_tile_idx_context(
            stage_info, kv_tile_start, tile_offset=tile_offset
        )
        page_ids = self.page_ids(tile_idx)
        pages_per_tile = self.cfg.kv_tile_n // self.cfg.num_tokens_per_page
        for page_frag in cutlass.range_constexpr(pages_per_tile):
            cached_page_ids[page_frag] = Int32(page_ids[page_frag])
        return cached_page_ids

    def dma_consumer_release_labels_for(
        self, downstream: MemoryResource
    ) -> set[str] | None:
        """Bind page-offset releases to the K/V TMA loads that consumed them."""
        if isinstance(downstream, SmemKVResource):
            if self.cfg.reuses_page_table_windows:
                if self.page_table_is_v:
                    return {"v_load", "v_load_stage", "v_load_stage_cached"}
                return {"k_load", "k_load_stage"}
            return {
                "k_load",
                "v_load",
                "k_load_stage",
                "v_load_stage",
                "v_load_stage_cached",
            }
        return None


# ---------------------------------------------------------------------------
# SmemKVResource -- SMEM K/V tile buffer with TMA pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemKVResource(MemoryResource):
    """SMEM buffer for K and V tiles with a capacity-derived TmaUmma pipeline.

    K and V tiles alternate in the pipeline stages: K0, V0, K1, V1, ...
    Producer: LoadTask (TMA loads K/V tiles).
    Consumer: MmaTask (builds SMEM descriptors for QK and PV MMAs).
    """

    sK_array: cutlass.Array = field(init=False, default=None)
    tma_k_desc: cutlass.Pointer | None = field(init=False, default=None)
    tma_v_desc: cutlass.Pointer | None = field(init=False, default=None)
    page_offsets_kv: Optional["SmemPageOffsetsKvResource"] = field(
        init=False, default=None
    )
    page_offsets_v: Optional["SmemPageOffsetsKvResource"] = field(
        init=False, default=None
    )
    block_tables: cute.Pointer | None = field(init=False, default=None)
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    desc_k_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_v_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        tma_k_desc: cutlass.Pointer | None,
        tma_v_desc: cutlass.Pointer | None,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        page_offsets_kv: Optional["SmemPageOffsetsKvResource"] = None,
        page_offsets_v: Optional["SmemPageOffsetsKvResource"] = None,
        block_tables: cute.Pointer | None = None,
        **kwargs: Any,
    ) -> None:
        """Bind K/V TMA descriptors and reserve shared SMEM staging."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.tma_k_desc = tma_k_desc
        self.tma_v_desc = tma_v_desc
        self.page_offsets_kv = page_offsets_kv
        self.page_offsets_v = (
            page_offsets_v if page_offsets_v is not None else page_offsets_kv
        )
        self.block_tables = block_tables
        self.cfg = cfg
        total_elements = cfg.sK_shape[0] * cfg.sK_shape[1]
        size_bytes = total_elements * cfg.k_dtype.width // 8
        self._alloc = SmemAllocation(
            "smem_kv", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sK_array = _placeholder_smem_array(cfg.k_dtype)
        self.desc_k_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the current K tile.",
        )
        self.desc_v_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the current V tile.",
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation required for staged K/V tiles."""
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Materialize K/V SMEM storage and descriptor dataflow slots."""
        smem_base = stage_info.context.smem_base
        total_elements = self.cfg.sK_shape[0] * self.cfg.sK_shape[1]
        self.sK_array = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=self.cfg.k_dtype,
            shape=(total_elements,),
            addrspace=3,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @property
    def loop_offset_sensitive(self) -> bool:
        """Return true because K/V loads index the current loop tile."""
        # producer_work uses loop_offset to compute seq_coord_kv.
        return True

    @cute.jit
    def _tma_load(
        self,
        stage_info: StageInfo,
        tma_desc: cutlass.Pointer | None,
        is_v: cutlass.Constexpr[bool] = False,
        tile_offset: cutlass.Constexpr[int] = 0,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        cached_page_ids: cutlass.Array | None = None,
        *,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Issue TMA bulk-copy for one K or V tile."""
        seq_offset = (
            kv_tile_start + stage_info.loop_offset + tile_offset
        ) * self.cfg.kv_tile_n
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)

        if cutlass.const_expr(self.cfg.use_paged_kv):
            # Paged-KV path: read pre-staged page IDs and issue one TMA per
            # (page fragment, d fragment). K and V share the native rank-4
            # descriptor coordinates (d_off, 0, kv_head_coord, page_id).
            tile_idx = kv_tile_start + stage_info.loop_offset + tile_offset
            pages_per_tile = self.cfg.kv_tile_n // self.cfg.num_tokens_per_page
            d_granu_inner = self.cfg.tma_copy_kv_granu_inner
            page_d_elems = self.cfg.num_tokens_per_page * d_granu_inner
            d_iter_elems = self.cfg.tma_copy_kv_granu_elems
            if prims.elect_sync():
                # Only the elected TMA-issuing lane consumes page IDs. Loading
                # the vector outside this guard made every lane perform the
                # same SMEM read for every K and V tile.
                page_ids = cached_page_ids
                if cutlass.const_expr(page_ids is None):
                    page_offsets = (
                        self.page_offsets_v
                        if cutlass.const_expr(is_v)
                        else self.page_offsets_kv
                    )
                    if cutlass.const_expr(page_offsets is not None):
                        page_ids = page_offsets.page_ids(tile_idx)
                    else:
                        # The paired K/V schedule consumes K and V together.
                        # Reading its four contiguous page IDs directly avoids
                        # a producer warp spinning on an always-full auxiliary
                        # pipeline and leaves that warp available for CLC.
                        # K and V share the same fixed logical-to-physical page
                        # row. Clamp both to the pages covered by the request's
                        # runtime sequence length so padding IDs are untouched.
                        logical_page_idx = tile_idx * Int32(pages_per_tile)
                        page_ids = cutlass.Array(
                            Int32,
                            pages_per_tile,
                            space=cutlass.AddressSpace.rmem,
                        )
                        for frag in cutlass.range_constexpr(pages_per_tile):
                            clamped_page_idx = cute.math.min(
                                logical_page_idx + Int32(frag), kv_page_idx_ub
                            )
                            page_ids[frag] = Int32(
                                self.block_tables[kv_request_begin + clamped_page_idx]
                            )
                for frag in cutlass.range_constexpr(pages_per_tile):
                    page_id = Int32(page_ids[frag])
                    for i in cutlass.range_constexpr(self.cfg.tma_copy_kv_stage_iters):
                        d_offset = Int32(
                            head_dim_stage_idx * self.cfg.head_dim_per_stage_kv
                            + i * d_granu_inner
                        )
                        smem_offset = Int32(i * d_iter_elems + frag * page_d_elems)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sK_curr.subview(smem_offset),
                            tma_desc,
                            (d_offset, Int32(0), kv_head_coord, page_id),
                            stage_info.barrier,
                        )
            return

        if prims.elect_sync():
            d_granu_inner = self.cfg.tma_copy_kv_granu_inner
            seq_coord_kv = cuseqlen_k + seq_offset
            for i in cutlass.range_constexpr(self.cfg.tma_copy_kv_stage_iters):
                d_offset = (
                    head_dim_stage_idx * self.cfg.head_dim_per_stage_kv
                    + i * d_granu_inner
                )
                kv_coords = (d_offset, kv_head_coord, seq_coord_kv, batch_coord)
                if cutlass.const_expr(self.cfg.has_varlen):
                    kv_coords = (d_offset, kv_head_coord, seq_coord_kv)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sK_curr.subview(i * self.cfg.tma_copy_kv_granu_elems),
                    tma_desc,
                    kv_coords,
                    stage_info.barrier,
                )

    @producer_work
    @cute.jit
    def k_load(
        self,
        stage_info: StageInfo,
        *,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load K tile from GMEM to SMEM."""
        self._tma_load(
            stage_info,
            self.tma_k_desc,
            tile_offset=tile_offset,
            head_dim_stage_idx=head_dim_stage_idx,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def k_load_stage(
        self,
        stage_info: StageInfo,
        *,
        stage_id: Constexpr[int],
        tile_offset: Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load one K head-dim stage for split D scheduling."""
        self._tma_load(
            stage_info,
            self.tma_k_desc,
            False,
            tile_offset=tile_offset,
            head_dim_stage_idx=stage_id,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def v_load(
        self,
        stage_info: StageInfo,
        *,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load V tile from GMEM to SMEM."""
        self._tma_load(
            stage_info,
            self.tma_v_desc,
            True,
            tile_offset=tile_offset,
            head_dim_stage_idx=head_dim_stage_idx,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def v_load_stage(
        self,
        stage_info: StageInfo,
        *,
        stage_id: Constexpr[int],
        previous: Constexpr[bool] = False,
        tile_offset: Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load one current or previous V head-dim stage."""
        self._tma_load(
            stage_info,
            self.tma_v_desc,
            True,
            tile_offset=tile_offset + (-1 if previous else 0),
            head_dim_stage_idx=stage_id,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def v_load_stage_cached(
        self,
        stage_info: StageInfo,
        *,
        cached_v_page_ids: cutlass.Array,
        stage_id: Constexpr[int],
        tile_offset: Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Load one V head-dimension stage using register-cached page IDs."""
        self._tma_load(
            stage_info,
            self.tma_v_desc,
            True,
            tile_offset=tile_offset,
            head_dim_stage_idx=stage_id,
            cached_page_ids=cached_v_page_ids,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @consumer_work(returns=desc_k_base)
    @cute.jit
    def k_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build K SMEM descriptor (K-major layout for QK MMA) -> desc_k_base."""
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)
        leading_byte_offset, stride_byte_offset = _qk_smem_desc_offsets(self.cfg)
        desc_k_base = prims.Tcgen05SmemDesc.build(
            sK_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=_qkv_smem_layout(self.cfg),
        )
        return desc_k_base

    @cute.jit
    def _zero_paged_v_tail(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[FmhaStage],
        tile_offset: cutlass.Constexpr[int],
        seqlen_k: Int32,
        kv_tile_start: Int32,
    ) -> None:
        """Overwrite request-invalid V rows after TMA completion."""
        if cutlass.const_expr(section == FmhaStage.Head):
            domain_tile_idx = stage_info.loop_start
        elif cutlass.const_expr(section == FmhaStage.Tail):
            domain_tile_idx = stage_info.loop_end
        else:
            domain_tile_idx = stage_info.loop_offset
        logical_v_tile_idx = kv_tile_start + domain_tile_idx + tile_offset
        valid_rows = cute.math.min(
            cute.math.max(
                seqlen_k - logical_v_tile_idx * Int32(self.cfg.kv_tile_n),
                Int32(0),
            ),
            Int32(self.cfg.kv_tile_n),
        )

        if valid_rows < Int32(self.cfg.kv_tile_n):
            # Each paged TMA transaction writes one swizzled
            # (D-fragment, page-token) box. Pages are concatenated within a D
            # iteration, and D iterations are concatenated within the stage.
            # Mirror that exact physical layout: a flat row-major clear would
            # target the wrong bytes under the s128b swizzle.
            d_granu_inner = self.cfg.tma_copy_kv_granu_inner
            chunks_per_d_iter = d_granu_inner // 16
            chunks_per_v_row = self.cfg.tma_copy_kv_stage_iters * chunks_per_d_iter
            page_d_elems = self.cfg.num_tokens_per_page * d_granu_inner
            d_iter_elems = self.cfg.tma_copy_kv_granu_elems
            invalid_chunks = (Int32(self.cfg.kv_tile_n) - valid_rows) * Int32(
                chunks_per_v_row
            )
            zero_vec = cutlass.vector.full(
                [16], self.cfg.v_dtype(0.0), dtype=self.cfg.v_dtype
            )
            sV_curr = self.sK_array.subview(
                stage_info.stage_idx * self.cfg.tma_copy_kv_elements
            )
            lane_idx = cute.arch.lane_idx()
            for tail_chunk in cutlass.range(
                lane_idx,
                invalid_chunks,
                Int32(cute.arch.WARP_SIZE),
                unroll=1,
            ):
                invalid_row = tail_chunk // Int32(chunks_per_v_row)
                d_chunk = tail_chunk - invalid_row * Int32(chunks_per_v_row)
                d_iter = d_chunk // Int32(chunks_per_d_iter)
                d_chunk_in_iter = d_chunk - d_iter * Int32(chunks_per_d_iter)
                logical_row = valid_rows + invalid_row
                page_frag = logical_row // Int32(self.cfg.num_tokens_per_page)
                row_in_page = logical_row - page_frag * Int32(
                    self.cfg.num_tokens_per_page
                )
                smem_offset = (
                    d_iter * Int32(d_iter_elems)
                    + page_frag * Int32(page_d_elems)
                    + row_in_page * Int32(d_granu_inner)
                    + d_chunk_in_iter * Int32(16)
                )
                sV_curr.subview(smem_offset).data_ptr().store_swizzled(
                    zero_vec,
                    alignment=16,
                    swizzle=_qkv_smem_swizzle(self.cfg),
                )

            # v_desc is called only after this stage's skv.wait(), which makes
            # the TMA writes visible. Converge the one MMA warp after its
            # generic stores, then publish them to the async SMEM proxy before
            # tcgen05 consumes the descriptor.
            cute.arch.sync_warp()
            prims.fence_proxy(
                kind=prims.Proxy.ASYNC_SHARED,
                space=prims.SharedSpace.shared_cta,
            )

    def _build_v_descriptor(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build the current stage's V descriptor after any required clear."""
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)
        leading_byte_offset, stride_byte_offset = _pv_smem_desc_offsets(self.cfg)
        return prims.Tcgen05SmemDesc.build(
            sK_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=_qkv_smem_layout(self.cfg),
        )

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build a nonpaged V SMEM descriptor -> desc_v_base."""
        return self._build_v_descriptor(stage_info)

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc_paged(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[FmhaStage],
        tile_offset: cutlass.Constexpr[int] = 0,
        seqlen_k: Int32,
        kv_tile_start: Int32,
    ) -> prims.Tcgen05SmemDesc:
        """Clear invalid paged-V rows, then build its SMEM descriptor."""
        self._zero_paged_v_tail(
            stage_info,
            section=section,
            tile_offset=tile_offset,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
        )
        return self._build_v_descriptor(stage_info)


# ---------------------------------------------------------------------------
# TmemSPResource -- TMEM S/P ping-pong buffer with UmmaAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemSPResource(MemoryResource):
    """TMEM S/P ping-pong buffer with UmmaAsync pipeline.

    Producer: MMA warp writes S = Q*K scores, then reads P for P*V.
    Consumer: Softmax warp reads S, computes P = softmax(S), writes P back.

    Self-edge in dependency graph enables ping-pong validation:
    MMA acquires -> writes S -> commits -> Softmax waits -> reads S,
    writes P -> releases -> MMA re-acquires -> reads P for P*V -> commits.
    """

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_s_offset: Constexpr[int] = field(init=False, default=None)
    tmem_p_offset: Constexpr[int] = field(init=False, default=None)
    # 0 for SP0 (uses Q0), 1 for SP1 (uses Q1).
    q_half: Constexpr[int] = 0
    enable_early_tile_sum: Constexpr[bool] = False
    q_offset_default: int | Int32 = field(init=False, default=0)
    cum_seqlen_q: cute.Tensor | None = field(init=False, default=None)
    cum_seqlen_k: cute.Tensor | None = field(init=False, default=None)
    seq_lens_kv: cute.Pointer | None = field(init=False, default=None)
    variable_window_token_starts: cute.Tensor | None = field(init=False, default=None)
    variable_window_token_ends: cute.Tensor | None = field(init=False, default=None)
    variable_window_cta_starts: cute.Tensor | None = field(init=False, default=None)
    variable_window_q_stride: int | Int32 = field(init=False, default=0)
    scale_softmax_log2: cute.Tensor | None = field(init=False, default=None)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    # Precomputed TMEM pointers/addresses (set by auxiliary work). Avoids
    # per-iteration inttoptr + address math.
    # MMA warp pointer for QK to S.
    tmem_ptr_s_cached: TmemPtr | None = field(init=False, default=None)
    # Softmax warp per-warp S address.
    tmem_s_addr_cached: TmemAddr | None = field(init=False, default=None)
    # Softmax warp per-warp P address.
    tmem_p_addr_cached: TmemAddr | None = field(init=False, default=None)
    _alloc: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)
    old_row_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_chunk: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    q_offset: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seqlen_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    variable_window_start: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    variable_window_end: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_s_offset: int,
        tmem_p_offset: int,
        q_half: int = 0,
        q_offset: int | Int32 = 0,
        cum_seqlen_q: cute.Tensor | None = None,
        cum_seqlen_k: cute.Tensor | None = None,
        seq_lens_kv: cute.Pointer | None = None,
        variable_window_token_starts: cute.Tensor | None = None,
        variable_window_token_ends: cute.Tensor | None = None,
        variable_window_cta_starts: cute.Tensor | None = None,
        variable_window_q_stride: int | Int32 = 0,
        scale_softmax_log2: cute.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Bind S/P TMEM offsets, Q peer index, and optional varlen metadata."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.tmem_s_offset = tmem_s_offset
        self.tmem_p_offset = tmem_p_offset
        self.q_half = q_half
        self.enable_early_tile_sum = cfg.enable_early_tile_sum
        self.q_offset_default = q_offset
        self.cum_seqlen_q = cum_seqlen_q
        self.cum_seqlen_k = cum_seqlen_k
        self.seq_lens_kv = seq_lens_kv
        self.variable_window_token_starts = variable_window_token_starts
        self.variable_window_token_ends = variable_window_token_ends
        self.variable_window_cta_starts = variable_window_cta_starts
        self.variable_window_q_stride = variable_window_q_stride
        self.scale_softmax_log2 = scale_softmax_log2
        self._alloc = TmemAllocation(
            f"tmem_sp_q{q_half}",
            cfg.qk_mma_tiler[1] * cfg.mma_softmax_stage,
        )
        self.tmem_addr_cached = Int32(0)
        self.tmem_ptr_s_cached = _placeholder_tmem_ptr()
        self.tmem_s_addr_cached = Int32(0)
        self.tmem_p_addr_cached = Int32(0)
        self.old_row_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(-Float32.inf),
            docs="Softmax row maximum from the previous K/V tile.",
        )
        self.row_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(-Float32.inf),
            docs="Softmax row maximum for the current K/V tile.",
        )
        self.row_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Accumulated softmax denominator for the current row.",
        )
        if self.enable_early_tile_sum:
            self.p_chunk = TaskLocalVariable(
                dtype=Float32,
                default=Float32(0.0),
                docs="FP32 sum of the current probability tile.",
            )
        else:
            self.p_chunk = TaskLocalVariable(
                dtype=list,
                default_factory=lambda: _placeholder_softmax_chunks(cfg),
                docs="P fragments retained for post-release row-sum reduction.",
            )
        self.q_offset = TaskLocalVariable(
            dtype=Int32,
            default=Int32(self.q_offset_default),
            docs="Causal Q/K sequence offset for the current work tile.",
        )
        self.seqlen_k = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Request-local K/V sequence length for packed dense masking.",
        )
        self.variable_window_start = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Inclusive first K position for this Q row.",
        )
        self.variable_window_end = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Inclusive last K position for this Q row.",
        )
        self.scale_softmax_log2_value = TaskLocalVariable(
            dtype=Float32,
            # Placeholder before load_scale_softmax_log2 reads the runtime tensor.
            default=Float32(0.0),
            docs="Softmax scale cached from the runtime scale tensor.",
        )

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return the TMEM allocation for this S/P ping-pong resource."""
        return [self._alloc]

    @property
    def loop_offset_sensitive(self) -> bool:
        """Return true because MMA and masking decisions use loop_offset."""
        # producer_work and head-paired masking use loop_offset for K tile indices.
        return True

    @property
    def uses_left_window_loop_mask(self) -> bool:
        """Return whether loop iterations need the left sliding-window mask."""
        return self.cfg.kv_tile_start_window_size_left > 0

    @property
    def uses_varlen_loop_right_mask(self) -> bool:
        """Return whether loop iterations need mixed-varlen right masking."""
        return self.cfg.head_paired and self.cfg.has_varlen and self.cfg.has_q_offset

    @property
    def uses_varlen_q_offset_cache(self) -> bool:
        """Return whether masks need a per-work-tile varlen Q/K offset."""
        return self.cfg.has_varlen and self.cfg.has_q_offset

    @property
    def uses_variable_window(self) -> bool:
        """Return whether softmax consumes explicit packed-Q row bounds."""
        return self.cfg.has_variable_window

    @property
    def uses_fixed_dense_k_tail_mask(self) -> bool:
        """Return whether fixed dense attention has a partial final K/V tile."""
        return (
            not self.cfg.is_causal
            and not self.cfg.has_varlen
            and self.cfg.fixed_dense_k_tail > 0
        )

    @property
    def uses_packed_dense_k_mask(self) -> bool:
        """Return whether packed or paged dense attention needs local K bounds."""
        return (
            self.cfg.has_varlen
            and not self.cfg.is_causal
            and self.cfg.packed_dense_k_mask
        )

    @property
    def uses_query_paired_q_offset_loop_mask(self) -> bool:
        """Return whether query-paired loop iterations need q-offset masking.

        Mixed packed batches use a request-local domain, but paired-tail
        alignment and partial Q tiles can conservatively retain a K/V tile
        that crosses the causal right edge. Either peer can therefore need
        the right mask inside LOOP rather than only in peer0 TAIL. A uniform
        tile-aligned shift preserves the ordinary tail placement and compiles
        this per-iteration mask away.
        """
        return (
            self.cfg.has_q_offset
            and not self.cfg.head_paired
            and not self.cfg.has_tile_aligned_uniform_q_offset
        )

    @property
    def uses_head_paired_causal_tail_mask(self) -> bool:
        """Return whether TAIL should use head-paired causal masking."""
        return self.cfg.is_causal and self.cfg.head_paired

    @property
    def needs_window_tail_left_mask(self) -> bool:
        """Return whether a sliding-window TAIL can cross its left edge.

        For fixed equal-length 128x128 tiling, a window of at least M-1
        tokens places the entire final causal tile on or to the right of the
        left bound. Packed and bottom-right-offset inputs retain the general
        two-sided mask because their runtime tile origin can shift.
        """
        return self.cfg.window_size_left > 0 and (
            self.cfg.has_varlen
            or self.cfg.has_q_offset
            or self.cfg.q_tile_m != self.cfg.kv_tile_n
            or self.cfg.window_size_left < self.cfg.q_tile_m - 1
        )

    @property
    def uses_query_paired_causal_tail_mask(self) -> bool:
        """Return whether TAIL should use query-paired causal masking."""
        return self.cfg.is_causal and not self.cfg.head_paired and self.q_half == 0

    @property
    def uses_query_paired_invalid_tail(self) -> bool:
        """Return whether peer0 needs the extra wholly-invalid tail slot."""
        return self.cfg.skip_causal_invalid_peer0 and self.q_half == 0

    @cute.jit
    def _stage_col_offset(self, stage_info: StageInfo) -> Int32 | int:
        """Return the TMEM column offset for a pipelined S/P stage."""
        stage_col_offset = Int32(0)
        if cutlass.const_expr(self.cfg.mma_softmax_stage > 1):
            stage_col_offset = stage_info.stage_idx * self.cfg.qk_mma_tiler[1]
        return stage_col_offset

    @producer_work
    @cute.jit
    def qk_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_q_base: prims.Tcgen05SmemDesc,
        desc_k_base: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[FmhaStage],
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """QK MMA: compute Q*K -> S in TMEM.

        The schedule aliases either desc_q0_base or desc_q1_base into the
        logical desc_q_base producer arg based on which SP instance is being
        driven.

        In causal mode with no Q right offset, skips QK0→S0 MMA in the last
        LOOP iteration, since Softmax0's domain is N-2 but MMA's domain is N-1.
        The task domain pads partial final CTAs so this slot is always outside
        peer0's causal reach.
        """
        skip_qk0_invalid = False
        if cutlass.const_expr(self.cfg.skip_causal_invalid_peer0 and self.q_half == 0):
            if cutlass.const_expr(section == FmhaStage.Loop):
                if not is_tail:
                    skip_qk0_invalid = stage_info.loop_offset == (
                        stage_info.loop_end - 1
                    )

        if not skip_qk0_invalid:
            tmem_ptr_s = self.tmem_ptr_s_cached.subview(
                self._stage_col_offset(stage_info)
            )

            if cutlass.const_expr(self.cfg.q_dtype.width == 8):
                mma_kind = prims.Tcgen05MMAKind.F8F6F4
                # E4M3 operands use the Float16 encoding handle.
                ab_format = cutlass.Float16
            else:
                mma_kind = prims.Tcgen05MMAKind.F16
                if cutlass.const_expr(self.cfg.q_dtype == cutlass.BFloat16):
                    ab_format = cutlass.BFloat16
                else:
                    ab_format = cutlass.Float16

            idesc_qk = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=ab_format,
                b_dtype=ab_format,
                n_dim=self.cfg.qk_mma_tiler[1],
                m_dim=self.cfg.qk_mma_tiler[0],
            )

            k_dim_per_mma = 16
            if cutlass.const_expr(self.cfg.q_dtype.width != 16):
                k_dim_per_mma = 32
            num_kphases_qk = self.cfg.qk_mma_tiler[2] // k_dim_per_mma
            inc_bytes_qk = k_dim_per_mma * self.cfg.q_dtype.width // 8

            num_kphases_per_tma = num_kphases_qk // self.cfg.tma_copy_qkv_iters
            chunk_bytes_qk = inc_bytes_qk * num_kphases_per_tma
            if cutlass.const_expr(self.cfg.tma_copy_qkv_iters != 1):
                chunk_bytes_qk = (
                    self.cfg.tma_copy_kv_bytes // self.cfg.tma_copy_kv_stage_iters
                )
            num_tma_iters_qk = self.cfg.tma_copy_qkv_iters
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                num_tma_iters_qk = self.cfg.tma_copy_kv_stage_iters

            # Prevent LLVM from rematerializing descriptor
            # computations inside each elect_sync basic block.
            # Without this, NVPTX recomputes shr+and+cvt+or from
            # __dynamic_shmem__0 inside every elect BB (~5 extra
            # instructions per MMA call).
            desc_q_base_ = freeze_smem_descriptor(desc_q_base)
            desc_k_base_ = freeze_smem_descriptor(desc_k_base)

            scale_d = False
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                scale_d = head_dim_stage_idx != 0
            for tma_iter in cutlass.range_constexpr(num_tma_iters_qk):
                q_tma_iter = head_dim_stage_idx * num_tma_iters_qk + tma_iter
                q_tma_iter_offset = chunk_bytes_qk * q_tma_iter
                k_tma_iter_offset = chunk_bytes_qk * tma_iter
                for k_idx in cutlass.range_constexpr(num_kphases_per_tma):
                    local_increment = inc_bytes_qk * k_idx
                    dq = desc_q_base_ + ((local_increment + q_tma_iter_offset) >> 4)
                    dk = desc_k_base_ + ((local_increment + k_tma_iter_offset) >> 4)
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            prims.CTAGroup.CTA_1,
                            tmem_ptr_s,
                            dq,
                            dk,
                            idesc_qk,
                            scale_d,
                        )
                    scale_d = True

    @producer_work
    @cute.jit
    def p_read(self, stage_info: StageInfo) -> None:
        """P-read sync: no-op. SP handle held from QK, consumed by softmax."""
        pass

    @cute.jit
    def _init_function_state(self, stage_info: StageInfo) -> None:
        """Precompute TMEM pointers/addresses (once, before persistent loop).

        Runs on all warps via init_variables, after tmem_addr_cached is set.
        Only the MMA-warp pointer is computed here (needed ungated).
        Softmax-warp addresses are deferred to per-work-tile auxiliary work
        so they are computed after setmaxnreg and avoid crossing the register
        budget boundary.  The fields are initialized to Int32(0) here so the
        DSL sees a consistent type structure before the scf.while loop.

        Emits the softmax-side state variables (old_row_max, row_max,
        row_sum, p_chunk, q_offset) consumed by Softmax tasks; producer-side
        desc_q_base / desc_k_base slots are auto-mirrored from
        SmemQ / SmemKV by Task.init_variables (with explicit aliasing
        from desc_q0_base / desc_q1_base).
        """
        # MMA warp: tmem_ptr_s for QK→S producer_work
        self.tmem_ptr_s_cached = prims.make_tmem_ptr(
            self.tmem_addr_cached, cutlass.Int8
        ).subview(self.tmem_s_offset)
        # Initialize to establish DSL type; real values are set per work tile.
        self.tmem_s_addr_cached = Int32(0)
        self.tmem_p_addr_cached = Int32(0)
        _ = stage_info

    @cute.jit
    def _default_p_chunk(self) -> SoftmaxRowSumContribution:
        if cutlass.const_expr(self.enable_early_tile_sum):
            return Float32(0.0)
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x

        # PERF NOTE: These P chunk vectors become iter_args of the scf.while
        # persistent loop. The MLIR compiler materializes zero-initialization
        # HEAD code (~128 add.rn.f32x2 instructions adding 0+0) and
        # TAIL finalization code (runs once after LOOP exit). The K-loop
        # body instruction count is unaffected — identical with or without
        # these iter_args. The HEAD/TAIL overhead may affect performance
        # through i-cache pressure (~+3% PTX footprint), register allocation
        # changes (ptxas sees more live values at scf.while boundary), and
        # pipeline warm-up timing shifts.
        p_chunk = []
        for _chunk_idx in cutlass.range_constexpr(num_chunks):
            zeros = tuple(self.cfg.qk_acc_dtype(0.0) for _ in range(tmem_x))
            p_chunk.append(cutlass.Vector.from_elements(zeros, self.cfg.qk_acc_dtype))
        return p_chunk

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_function_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_softmax_state_early(self, stage_info: StageInfo) -> None:
        """Initialize softmax TMEM state without a function-lifetime P value."""
        self._init_function_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def init_softmax_state(self, stage_info: StageInfo) -> SoftmaxRowSumContribution:
        self._init_function_state(stage_info)
        return self._default_p_chunk()

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns="scale_softmax_log2_value")
    @cute.jit
    def load_scale_softmax_log2(self, stage_info: StageInfo) -> Float32:
        """Load the runtime softmax scale once before the K/V loop."""
        _ = stage_info
        if cutlass.const_expr(self.scale_softmax_log2 is None):
            # Safe fallback for validation-only resource construction.
            return Float32(0.0)
        return self.scale_softmax_log2[0]

    @cute.jit
    def _init_work_tile_state(self, stage_info: StageInfo) -> None:
        """Reset softmax state and recompute per-warp TMEM addresses each tile.

        Softmax-warp addresses are computed here (inside the persistent loop,
        after setmaxnreg) to avoid spilling them across the register-budget
        boundary in ungated HEAD. The returned q_offset defaults to the
        uniform kernel argument; varlen causal masks overwrite it once per
        work tile via cache_q_offset().
        """
        num_softmax_warps = 4
        warp_id_in_sg = cute.arch.warp_idx() % num_softmax_warps
        tmem_raw_addr = self.tmem_addr_cached
        tmem_base_row = tmem_raw_addr >> 16
        tmem_base_col = tmem_raw_addr & Int32(0xFFFF)
        row_id = tmem_base_row + warp_id_in_sg * cute.arch.WARP_SIZE
        self.tmem_s_addr_cached = (row_id << 16) | (tmem_base_col + self.tmem_s_offset)
        self.tmem_p_addr_cached = (row_id << 16) | (tmem_base_col + self.tmem_p_offset)
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_work_tile_state(self, stage_info: StageInfo) -> None:
        self._init_work_tile_state(stage_info)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_row_max, row_max, row_sum, q_offset),
    )
    @cute.jit
    def init_softmax_work_tile_state(
        self, stage_info: StageInfo
    ) -> tuple[Float32, Float32, Float32, Int32]:
        self._init_work_tile_state(stage_info)
        return (
            Float32(-Float32.inf),
            Float32(-Float32.inf),
            Float32(0.0),
            Int32(self.q_offset_default),
        )

    @cute.jit
    def _varlen_batch_coord(self, stage_info: StageInfo) -> Int32:
        """Return the batch coordinate for the active tile-order policy."""
        _, _, batch_coord = _resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        return batch_coord

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=q_offset)
    @cute.jit
    def cache_q_offset(self, stage_info: StageInfo) -> Int32:
        """Cache the per-work-tile causal Q/K sequence offset for masks.

        Mixed-varlen batches cannot use the uniform kernel q_offset because
        each batch can have a different S_kv - S_q. This pre-wait hook runs
        once in the softmax task HEAD, before the K/V loop, so loop and tail
        masks reuse the cached offset instead of rereading the request metadata.
        """
        if cutlass.const_expr(
            self.cfg.has_uniform_varlen and not self.cfg.use_paged_kv
        ):
            return Int32(self.cfg.uniform_seq_len_k - self.cfg.uniform_seq_len_q)
        batch_coord = self._varlen_batch_coord(stage_info)
        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            seqlen_q = Int32(self.cfg.uniform_seq_len_q)
        else:
            cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
            seqlen_q = Int32(self.cum_seqlen_q[batch_coord + Int32(1)]) - cuseqlen_q
        if cutlass.const_expr(self.cfg.use_paged_kv):
            seqlen_k = Int32(self.seq_lens_kv[batch_coord])
        elif cutlass.const_expr(self.cfg.has_uniform_varlen):
            seqlen_k = Int32(self.cfg.uniform_seq_len_k)
        else:
            cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord])
            seqlen_k = Int32(self.cum_seqlen_k[batch_coord + Int32(1)]) - cuseqlen_k
        return seqlen_k - seqlen_q

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=seqlen_k)
    @cute.jit
    def cache_seqlen_k(self, stage_info: StageInfo) -> Int32:
        """Cache the request-local K/V extent once per work tile."""
        if cutlass.const_expr(self.cfg.use_paged_kv):
            batch_coord = self._varlen_batch_coord(stage_info)
            return Int32(self.seq_lens_kv[batch_coord])
        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            return Int32(self.cfg.uniform_seq_len_k)
        batch_coord = self._varlen_batch_coord(stage_info)
        cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord])
        return Int32(self.cum_seqlen_k[batch_coord + Int32(1)]) - cuseqlen_k

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(variable_window_start, variable_window_end),
    )
    @cute.jit
    def cache_variable_window_bounds(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32]:
        """Load this lane's bounds relative to the CTA's first K/V tile."""
        seq_coord, _, batch_coord = _resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        warp_id_in_sg = cute.arch.warp_idx() % 4
        row_in_tile = warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx()
        local_q = (
            seq_coord * self.cfg.q_tile_m * self.cfg.work_tile_q_seq_tiles
            + self.q_half * self.cfg.peer_q_seq_tile_stride * self.cfg.q_tile_m
            + row_in_tile
        )
        local_q = cute.math.min(
            local_q,
            self.variable_window_q_stride - Int32(1),
        )
        packed_q = batch_coord * self.variable_window_q_stride + local_q
        min_window_start = variable_window_cta_min_start(
            self.variable_window_cta_starts,
            batch_coord=batch_coord,
            seq_coord=seq_coord,
            q_stride=self.variable_window_q_stride,
            tile_size_q=self.cfg.cta_tiler[0],
        )
        kv_base = (min_window_start // self.cfg.kv_tile_n) * self.cfg.kv_tile_n
        return (
            Int32(self.variable_window_token_starts[packed_q]) - kv_base,
            Int32(self.variable_window_token_ends[packed_q]) - kv_base,
        )

    @cute.jit
    def _load_s_chunks(self, stage_info: StageInfo) -> SoftmaxChunks:
        """Load ALL S chunks from TMEM into register vectors."""
        tmem_s_addr = self.tmem_s_addr_cached + self._stage_col_offset(stage_info)
        tmem_shape = "32x32b"
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        s_data = [None] * num_chunks
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            _chunk = cutlass.Array(self.cfg.qk_acc_dtype, tmem_x)
            _chunk[0:tmem_x] = prims.tcgen05_ld(
                tmem_shape,
                prims.make_tmem_ptr(
                    tmem_s_addr + chunk_idx * tmem_x, self.cfg.qk_acc_dtype
                ),
                num=tmem_x,
            )
            s_data[chunk_idx] = _chunk
        cute.arch.fence_view_async_tmem_load()
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            s_data[chunk_idx] = s_data[chunk_idx][0:tmem_x]
        return s_data

    @cute.jit
    def _reduce_row_max(
        self,
        s_data: SoftmaxChunks,
        row_max: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Reduce per-chunk maximums into row_max, stash s_data."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        old_row_max = row_max
        if cutlass.const_expr(
            self.cfg.uses_d128_fp8_softmax_cadence
            or self.cfg.uses_d256_fp8_softmax_cadence
        ):
            max_0 = row_max
            max_1 = row_max
            max_2 = row_max
            max_3 = row_max
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                for elem_idx in cutlass.range_constexpr(0, tmem_x, 4):
                    max_0 = cute.math.max(max_0, s_data[chunk_idx][elem_idx], ftz=True)
                    max_1 = cute.math.max(
                        max_1, s_data[chunk_idx][elem_idx + 1], ftz=True
                    )
                    max_2 = cute.math.max(
                        max_2, s_data[chunk_idx][elem_idx + 2], ftz=True
                    )
                    max_3 = cute.math.max(
                        max_3, s_data[chunk_idx][elem_idx + 3], ftz=True
                    )
            max_0 = cute.math.max(max_0, max_2, ftz=True)
            max_1 = cute.math.max(max_1, max_3, ftz=True)
            row_max = cute.math.max(max_0, max_1, ftz=True)
        else:
            row_values: tuple[Any, ...] = ()
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                for elem_idx in cutlass.range_constexpr(tmem_x):
                    row_values += (s_data[chunk_idx][elem_idx],)
            row_vector = cutlass.Vector.from_elements(row_values, self.cfg.qk_acc_dtype)
            tile_row_max = row_vector.reduce("max")
            row_max = cute.math.max(row_max, tile_row_max)
        _tmem_sp_sdata[id(self)] = s_data
        row_max_safe = row_max
        if row_max == -Float32.inf:
            row_max_safe = Float32(0.0)
        return old_row_max, row_max_safe

    @cute.jit
    def _exp2_p_store(
        self,
        stage_col_offset: TmemAddr,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """Apply exp2 softmax P, fold the PV P scale, and store P to TMEM."""
        tmem_p_addr = self.tmem_p_addr_cached + stage_col_offset
        tmem_shape = "32x32b"
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        p_packing_ratio = self.cfg.qk_acc_dtype.width // self.cfg.v_dtype.width
        scale = scale_softmax_log2
        if cutlass.const_expr(self.cfg.uses_d256_fp8_softmax_cadence):
            return self._exp2_p_store_d256_fp8_cadence(
                tmem_p_addr,
                row_max,
                scale,
            )
        if cutlass.const_expr(self.cfg.uses_d128_fp8_softmax_cadence):
            return self._exp2_p_store_d128_fp8_cadence(
                tmem_p_addr,
                row_max,
                scale,
            )
        p_data_f32 = cutlass.Array(self.cfg.qk_acc_dtype, tmem_x, alignment=16)
        p_data_packed = cutlass.Array(
            p_data_f32.data_ptr(),
            shape=(tmem_x * p_packing_ratio,),
            dtype=self.cfg.v_dtype,
        )
        p_scale_log2 = Float32(self.cfg.pv_p_scale_log2)
        minus_row_max_scale = (Float32(0.0) - row_max) * scale + p_scale_log2
        s_data = _tmem_sp_sdata.pop(id(self))
        if cutlass.const_expr(self.enable_early_tile_sum):
            # Keep four independent scalar dependency chains while expressing
            # them as two packed float2 values.  The explicit packed primitive
            # lowers to FADD2 for D128 instead of two scalar FADDs per pair.
            local_sum_pair_0 = (Float32(0.0), Float32(0.0))
            local_sum_pair_1 = (Float32(0.0), Float32(0.0))
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            p_vals = ()
            for elem_idx in cutlass.range_constexpr(0, tmem_x, 2):
                fma_pair = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][elem_idx],
                        s_data[chunk_idx][elem_idx + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                    rnd="rn",
                    ftz=False,
                )
                p0 = cute.math.exp2(fma_pair[0], fastmath=True)
                p1 = cute.math.exp2(fma_pair[1], fastmath=True)
                if cutlass.const_expr(self.enable_early_tile_sum):
                    pair_idx = chunk_idx * (tmem_x // 2) + elem_idx // 2
                    if cutlass.const_expr(pair_idx % 2 == 0):
                        local_sum_pair_0 = cute.arch.add_packed_f32x2(
                            local_sum_pair_0,
                            (p0, p1),
                            rnd="rn",
                            ftz=False,
                        )
                    else:
                        local_sum_pair_1 = cute.arch.add_packed_f32x2(
                            local_sum_pair_1,
                            (p0, p1),
                            rnd="rn",
                            ftz=False,
                        )
                p_vals += (p0, p1)
            s_data[chunk_idx] = cutlass.Vector.from_elements(
                p_vals, self.cfg.qk_acc_dtype
            )
        use_fused_d128_fp8x4_pack = (
            not self.cfg.stage_kv_by_head_dim
            and self.cfg.v_dtype == cutlass.Float8E4M3FN
        )
        for pair_idx in cutlass.range_constexpr(num_chunks // p_packing_ratio):
            if cutlass.const_expr(use_fused_d128_fp8x4_pack):
                # Match the handwritten D128 pack: merge both FP8x2
                # conversions in one side-effecting block so ptxas can retain
                # the 32-bit word without a PRMT between temporary vectors.
                packed_words: tuple[Any, ...] = ()
                for word_idx in cutlass.range_constexpr(tmem_x):
                    flat_idx = word_idx * 4
                    chunk_idx = pair_idx * p_packing_ratio + flat_idx // tmem_x
                    elem_idx = flat_idx % tmem_x
                    packed_word = _pack_float4_to_fp8_e4m3(
                        s_data[chunk_idx][elem_idx],
                        s_data[chunk_idx][elem_idx + 1],
                        s_data[chunk_idx][elem_idx + 2],
                        s_data[chunk_idx][elem_idx + 3],
                    )
                    packed_words += (packed_word,)
                store_fragment = cutlass.Vector.from_elements(packed_words, Int32)
            else:
                for slice_idx in cutlass.range_constexpr(p_packing_ratio):
                    chunk_idx = pair_idx * p_packing_ratio + slice_idx
                    p_chunk_dtype = s_data[chunk_idx].to(self.cfg.v_dtype)
                    if cutlass.const_expr(self.cfg.v_dtype.width == 8):
                        p_chunk_i8 = p_chunk_dtype.bitcast(cutlass.Int8)
                        p_data_packed[slice_idx * tmem_x : tmem_x] = p_chunk_i8
                    else:
                        p_data_packed[slice_idx * tmem_x : tmem_x] = p_chunk_dtype
                store_fragment = p_data_f32[0:tmem_x]
            prims.tcgen05_st(
                tmem_shape,
                prims.make_tmem_ptr(tmem_p_addr + pair_idx * tmem_x, cutlass.Int8),
                store_fragment,
            )
        if cutlass.const_expr(self.enable_early_tile_sum):
            local_sum_pair = cute.arch.add_packed_f32x2(
                local_sum_pair_0,
                local_sum_pair_1,
                rnd="rn",
                ftz=False,
            )
            tile_sum = local_sum_pair[0] + local_sum_pair[1]
        if cutlass.const_expr(
            self.enable_early_tile_sum or self.cfg.has_tmem_p_pipeline
        ):
            # Publish TMEM store through the task-pipeline barrier without a blocking
            # store wait. The P-ready consumer pipeline orders the UMMA warp
            # after every store in the staged D256 path.
            cute.arch.fence_view_async_tmem_store()
        else:
            # Preserve the legacy publication sequence for paths that retain
            # P fragments until after SP release.
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        if cutlass.const_expr(self.enable_early_tile_sum):
            return tile_sum
        result = []
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            result.append(s_data[chunk_idx])
        return result

    @cute.jit
    def _exp2_p_store_d128_fp8_cadence(
        self,
        tmem_p_addr: TmemAddr,
        row_max: SoftmaxScalar,
        scale: SoftmaxScalar,
    ) -> Float32:
        """Use TRT's D128 FP8 softmax arithmetic cadence.

        Prefetch eight FMA values and retire FP8 conversions and scalar sums
        eight values behind EXP2. Four independent sum chains preserve the
        dependency depth of the reference implementation.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_values = self.cfg.qk_mma_tiler[1]
        fma_lookahead = 8
        retirement_delay = 8
        store_group_words = 4
        words_per_chunk = 2 * store_group_words
        p_scale_log2 = Float32(self.cfg.pv_p_scale_log2)
        minus_row_max_scale = (Float32(0.0) - row_max) * scale + p_scale_log2
        s_data = _tmem_sp_sdata.pop(id(self))
        local_sum_chains = cutlass.Array(
            Float32,
            4,
            space=cutlass.AddressSpace.rmem,
        )
        for chain_idx in cutlass.range_constexpr(4):
            local_sum_chains[chain_idx] = Float32(0.0)

        fma_ring = cute.make_rmem_tensor((fma_lookahead,), Float32)
        exp_ring = cute.make_rmem_tensor((retirement_delay,), Float32)
        p_output_words_lo = cute.make_rmem_tensor((store_group_words,), Int32)
        p_output_words_hi = cute.make_rmem_tensor((store_group_words,), Int32)
        num_chunks = num_values // tmem_x
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            for local_idx in cutlass.range_constexpr(0, fma_lookahead, 2):
                fma_pair = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][local_idx],
                        s_data[chunk_idx][local_idx + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                    rnd="rn",
                    ftz=False,
                )
                fma_ring[local_idx] = fma_pair[0]
                fma_ring[local_idx + 1] = fma_pair[1]

            for local_idx in cutlass.range_constexpr(0, tmem_x, 2):
                fma_idx = local_idx % fma_lookahead
                exp_idx = local_idx % retirement_delay
                if cutlass.const_expr(
                    local_idx >= retirement_delay
                    and (local_idx - retirement_delay) % 4 == 0
                ):
                    delayed_idx = local_idx - retirement_delay
                    word_idx = delayed_idx // 4
                    if cutlass.const_expr(word_idx < store_group_words):
                        p_output_words_lo[word_idx] = _pack_float4_to_fp8_e4m3(
                            exp_ring[exp_idx],
                            exp_ring[exp_idx + 1],
                            exp_ring[exp_idx + 2],
                            exp_ring[exp_idx + 3],
                        )
                    else:
                        p_output_words_hi[word_idx - store_group_words] = (
                            _pack_float4_to_fp8_e4m3(
                                exp_ring[exp_idx],
                                exp_ring[exp_idx + 1],
                                exp_ring[exp_idx + 2],
                                exp_ring[exp_idx + 3],
                            )
                        )

                p_0 = cute.math.exp2(fma_ring[fma_idx], fastmath=True)
                # Preserve the odd value from the current pair before the
                # circular lookahead slot is refilled with the future pair.
                fma_1 = fma_ring[fma_idx + 1]
                if cutlass.const_expr(local_idx + fma_lookahead < tmem_x):
                    future_idx = local_idx + fma_lookahead
                    fma_pair = cute.arch.fma_packed_f32x2(
                        (
                            s_data[chunk_idx][future_idx],
                            s_data[chunk_idx][future_idx + 1],
                        ),
                        (scale, scale),
                        (minus_row_max_scale, minus_row_max_scale),
                        rnd="rn",
                        ftz=False,
                    )
                    fma_ring[fma_idx] = fma_pair[0]
                    fma_ring[fma_idx + 1] = fma_pair[1]
                p_1 = cute.math.exp2(fma_1, fastmath=True)

                if cutlass.const_expr(local_idx >= retirement_delay):
                    delayed_idx = local_idx - retirement_delay
                    chain_base = ((delayed_idx // 2) % 2) * 2
                    local_sum_chains[chain_base] += exp_ring[exp_idx]
                    local_sum_chains[chain_base + 1] += exp_ring[exp_idx + 1]
                exp_ring[exp_idx] = p_0
                exp_ring[exp_idx + 1] = p_1

            for delayed_idx in cutlass.range_constexpr(
                tmem_x - retirement_delay,
                tmem_x,
                4,
            ):
                exp_idx = delayed_idx % retirement_delay
                word_idx = delayed_idx // 4
                if cutlass.const_expr(word_idx < store_group_words):
                    p_output_words_lo[word_idx] = _pack_float4_to_fp8_e4m3(
                        exp_ring[exp_idx],
                        exp_ring[exp_idx + 1],
                        exp_ring[exp_idx + 2],
                        exp_ring[exp_idx + 3],
                    )
                else:
                    p_output_words_hi[word_idx - store_group_words] = (
                        _pack_float4_to_fp8_e4m3(
                            exp_ring[exp_idx],
                            exp_ring[exp_idx + 1],
                            exp_ring[exp_idx + 2],
                            exp_ring[exp_idx + 3],
                        )
                    )
            for delayed_idx in cutlass.range_constexpr(
                tmem_x - retirement_delay,
                tmem_x,
                2,
            ):
                exp_idx = delayed_idx % retirement_delay
                chain_base = ((delayed_idx // 2) % 2) * 2
                local_sum_chains[chain_base] += exp_ring[exp_idx]
                local_sum_chains[chain_base + 1] += exp_ring[exp_idx + 1]

            prims.tcgen05_st(
                "32x32b",
                prims.make_tmem_ptr(
                    tmem_p_addr + chunk_idx * words_per_chunk,
                    cutlass.Int8,
                ),
                p_output_words_lo.load(),
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            prims.tcgen05_st(
                "32x32b",
                prims.make_tmem_ptr(
                    tmem_p_addr + chunk_idx * words_per_chunk + store_group_words,
                    cutlass.Int8,
                ),
                p_output_words_hi.load(),
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)

        local_sum_pair = cute.arch.add_packed_f32x2(
            (local_sum_chains[0], local_sum_chains[1]),
            (local_sum_chains[2], local_sum_chains[3]),
            rnd="rn",
            ftz=False,
        )
        return local_sum_pair[0] + local_sum_pair[1]

    @cute.jit
    def _exp2_p_store_d256_fp8_cadence(
        self,
        tmem_p_addr: TmemAddr,
        row_max: SoftmaxScalar,
        scale: SoftmaxScalar,
    ) -> Float32:
        """Interleave D256 FP8 EXP2, conversion, and tile-sum retirement.

        The eight-value lookahead mirrors the handwritten D256 FMHA cadence
        while retaining immutable SSA values. This avoids the long all-EXP2
        burst without reintroducing the mutable cross-typed fragment that was
        nondeterministic under Task Scheduling control flow.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_values = self.cfg.qk_mma_tiler[1]
        p_scale_log2 = Float32(self.cfg.pv_p_scale_log2)
        minus_row_max_scale = (Float32(0.0) - row_max) * scale + p_scale_log2
        s_data = _tmem_sp_sdata.pop(id(self))

        fma_values: tuple[Any, ...] = ()
        for flat_idx in cutlass.range_constexpr(0, 8, 2):
            chunk_idx = flat_idx // tmem_x
            elem_idx = flat_idx % tmem_x
            fma_pair = cute.arch.fma_packed_f32x2(
                (
                    s_data[chunk_idx][elem_idx],
                    s_data[chunk_idx][elem_idx + 1],
                ),
                (scale, scale),
                (minus_row_max_scale, minus_row_max_scale),
                rnd="rn",
                ftz=False,
            )
            fma_values += (fma_pair[0], fma_pair[1])

        p_values: tuple[Any, ...] = ()
        packed_words: tuple[Any, ...] = ()
        local_sum_pair_0 = (Float32(0.0), Float32(0.0))
        local_sum_pair_1 = (Float32(0.0), Float32(0.0))
        use_two_sum_pairs = not self.cfg.stage_kv_by_head_dim
        for flat_idx in cutlass.range_constexpr(0, num_values, 2):
            if cutlass.const_expr(flat_idx >= 8):
                delayed_idx = flat_idx - 8
                if cutlass.const_expr(flat_idx % 4 == 0):
                    packed_word = _pack_float4_to_fp8_e4m3(
                        p_values[delayed_idx],
                        p_values[delayed_idx + 1],
                        p_values[delayed_idx + 2],
                        p_values[delayed_idx + 3],
                    )
                    packed_words += (packed_word,)
                if cutlass.const_expr(
                    use_two_sum_pairs and (delayed_idx // 2) % 2 == 1
                ):
                    local_sum_pair_1 = cute.arch.add_packed_f32x2(
                        local_sum_pair_1,
                        (p_values[delayed_idx], p_values[delayed_idx + 1]),
                        rnd="rn",
                        ftz=False,
                    )
                else:
                    local_sum_pair_0 = cute.arch.add_packed_f32x2(
                        local_sum_pair_0,
                        (p_values[delayed_idx], p_values[delayed_idx + 1]),
                        rnd="rn",
                        ftz=False,
                    )

            p0 = cute.math.exp2(fma_values[flat_idx], fastmath=True)
            if cutlass.const_expr(flat_idx + 8 < num_values):
                future_idx = flat_idx + 8
                chunk_idx = future_idx // tmem_x
                elem_idx = future_idx % tmem_x
                fma_pair = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][elem_idx],
                        s_data[chunk_idx][elem_idx + 1],
                    ),
                    (scale, scale),
                    (minus_row_max_scale, minus_row_max_scale),
                    rnd="rn",
                    ftz=False,
                )
                fma_values += (fma_pair[0], fma_pair[1])
            p1 = cute.math.exp2(fma_values[flat_idx + 1], fastmath=True)
            p_values += (p0, p1)

        for delayed_idx in cutlass.range_constexpr(num_values - 8, num_values, 2):
            if cutlass.const_expr(delayed_idx % 4 == 0):
                packed_word = _pack_float4_to_fp8_e4m3(
                    p_values[delayed_idx],
                    p_values[delayed_idx + 1],
                    p_values[delayed_idx + 2],
                    p_values[delayed_idx + 3],
                )
                packed_words += (packed_word,)
            if cutlass.const_expr(use_two_sum_pairs and (delayed_idx // 2) % 2 == 1):
                local_sum_pair_1 = cute.arch.add_packed_f32x2(
                    local_sum_pair_1,
                    (p_values[delayed_idx], p_values[delayed_idx + 1]),
                    rnd="rn",
                    ftz=False,
                )
            else:
                local_sum_pair_0 = cute.arch.add_packed_f32x2(
                    local_sum_pair_0,
                    (p_values[delayed_idx], p_values[delayed_idx + 1]),
                    rnd="rn",
                    ftz=False,
                )

        if cutlass.const_expr(use_two_sum_pairs):
            local_sum_pair_0 = cute.arch.add_packed_f32x2(
                local_sum_pair_0,
                local_sum_pair_1,
                rnd="rn",
                ftz=False,
            )

        store_fragment = cutlass.Vector.from_elements(packed_words, Int32)
        prims.tcgen05_st(
            "32x32b",
            prims.make_tmem_ptr(tmem_p_addr, cutlass.Int8),
            store_fragment,
        )
        cute.arch.fence_view_async_tmem_store()
        return local_sum_pair_0[0] + local_sum_pair_0[1]

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def compute_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Main K-loop stage: load S from TMEM and compute unmasked row_max."""
        s_data = self._load_s_chunks(stage_info)
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def fixed_dense_k_tail_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Exclude TMA zero-fill lanes in a partial fixed dense K/V tile."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        s_data = self._load_s_chunks(stage_info)

        if stage_info.loop_offset == stage_info.loop_end - Int32(1):
            neg_inf = cutlass.vector.full(
                [tmem_x],
                self.cfg.qk_acc_dtype(-Float32.inf),
                dtype=self.cfg.qk_acc_dtype,
            )
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                valid_in_chunk = cute.math.min(
                    cute.math.max(
                        Int32(self.cfg.fixed_dense_k_tail) - Int32(chunk_idx * tmem_x),
                        Int32(0),
                    ),
                    Int32(tmem_x),
                )
                mask = cutlass.vector.create_mask([tmem_x], [valid_in_chunk])
                s_data[chunk_idx] = cutlass.vector.where(
                    mask, s_data[chunk_idx], neg_inf
                )
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def packed_dense_k_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        seqlen_k: Int32,
        section: cutlass.Constexpr[FmhaStage],
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Mask scores beyond one packed request's K/V right edge."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        s_data = self._load_s_chunks(stage_info)
        if cutlass.const_expr(section == FmhaStage.Loop):
            kv_tile_idx = stage_info.loop_offset
        else:
            # Some schedules materialize their final score tile in TAIL.
            kv_tile_idx = stage_info.loop_end
        kv_base = kv_tile_idx * self.cfg.kv_tile_n
        kv_end = kv_base + Int32(self.cfg.kv_tile_n)
        if seqlen_k < kv_end:
            neg_inf = cutlass.vector.full(
                [tmem_x],
                self.cfg.qk_acc_dtype(-Float32.inf),
                dtype=self.cfg.qk_acc_dtype,
            )
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                chunk_base = kv_base + Int32(chunk_idx * tmem_x)
                valid_in_chunk = cute.math.min(
                    cute.math.max(seqlen_k - chunk_base, Int32(0)),
                    Int32(tmem_x),
                )
                mask = cutlass.vector.create_mask([tmem_x], [valid_in_chunk])
                s_data[chunk_idx] = cutlass.vector.where(
                    mask, s_data[chunk_idx], neg_inf
                )
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def variable_window_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        window_start: Int32,
        window_end: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Mask S using inclusive per-row VariableWindow bounds."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        s_data = self._load_s_chunks(stage_info)
        kv_tile_base = stage_info.loop_offset * self.cfg.kv_tile_n
        tile_n = self.cfg.qk_mma_tiler[1]
        left_oob = cute.math.min(
            cute.math.max(window_start - kv_tile_base, Int32(0)),
            Int32(tile_n),
        )
        right_valid = cute.math.min(
            cute.math.max(window_end + Int32(1) - kv_tile_base, Int32(0)),
            Int32(tile_n),
        )
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            chunk_base = Int32(chunk_idx * tmem_x)
            chunk_left = cute.math.min(
                cute.math.max(left_oob - chunk_base, Int32(0)),
                Int32(tmem_x),
            )
            chunk_right = cute.math.min(
                cute.math.max(right_valid - chunk_base, Int32(0)),
                Int32(tmem_x),
            )
            valid_bits = _bmsk_clamp(chunk_left, chunk_right - chunk_left)
            chunk = s_data[chunk_idx]
            masked_scores = []
            for quad_idx in cutlass.range_constexpr(tmem_x // 4):
                quad_base = quad_idx * 4
                masked_scores.extend(
                    _mask_score_quad(
                        valid_bits >> Int32(quad_base),
                        chunk[quad_base],
                        chunk[quad_base + 1],
                        chunk[quad_base + 2],
                        chunk[quad_base + 3],
                    )
                )
            s_data[chunk_idx] = cutlass.Vector.from_elements(
                tuple(masked_scores), self.cfg.qk_acc_dtype
            )
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def left_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Apply the bottom-right-aligned sliding-window left mask."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        num_softmax_warps = 4
        warp_id_in_sg = cute.arch.warp_idx() % num_softmax_warps
        tmem_row_id = warp_id_in_sg * cute.arch.WARP_SIZE
        row_in_tile = tmem_row_id + cute.arch.lane_idx()

        s_data = self._load_s_chunks(stage_info)

        seq_tile_coord, _, _ = _resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        index_q = seq_tile_coord * self.cfg.q_tile_m + row_in_tile
        if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
            window_bound_left = bottom_right_window_left_bound(
                index_q,
                q_offset,
                self.cfg.window_size_left,
            )
            kv_tile_start = bottom_right_window_tile_start(
                seq_coord=seq_tile_coord,
                q_tile_m=self.cfg.q_tile_m,
                kv_tile_n=self.cfg.kv_tile_n,
                q_offset=q_offset,
                window_size_left=self.cfg.window_size_left,
            )
        else:
            # Exact fixed equal-length fast path: q_offset is constexpr zero.
            window_bound_left = index_q - Int32(self.cfg.window_size_left)
            kv_tile_start = cute.math.max(
                Int32(0),
                (seq_tile_coord * self.cfg.q_tile_m - self.cfg.window_size_left)
                // self.cfg.kv_tile_n,
            )
        kv_tile_abs = kv_tile_start + stage_info.loop_offset

        neg_inf = cutlass.vector.full(
            [tmem_x], self.cfg.qk_acc_dtype(-Float32.inf), dtype=self.cfg.qk_acc_dtype
        )
        all_true_mask = cutlass.vector.create_mask([tmem_x], [tmem_x])

        for chunk_idx in cutlass.range_constexpr(num_chunks):
            base_k = kv_tile_abs * self.cfg.kv_tile_n + chunk_idx * tmem_x
            left_oob_end_idx = window_bound_left - base_k
            left_mask_inverted = cutlass.vector.create_mask(
                [tmem_x], [left_oob_end_idx]
            )
            mask = left_mask_inverted ^ all_true_mask
            if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
                # Packed requests share a worst-case window span, and fixed
                # bottom-right windows can begin at a non-aligned Q/K offset.
                window_bound_right = index_q + q_offset
                right_oob_start_idx = window_bound_right + Int32(1) - base_k
                right_oob_start_idx = cute.math.min(
                    cute.math.max(right_oob_start_idx, Int32(0)),
                    Int32(tmem_x),
                )
                right_mask = cutlass.vector.create_mask([tmem_x], [right_oob_start_idx])
                mask = mask & right_mask
            s_data[chunk_idx] = cutlass.vector.where(mask, s_data[chunk_idx], neg_inf)

        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def loop_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Loop stage: apply causal masking for mixed Q right-offset batches."""
        s_data = self._load_s_chunks(stage_info)
        s_data = self._apply_causal_mask_for_kv_tile(
            stage_info, s_data, kv_tile_idx=stage_info.loop_offset, q_offset=q_offset
        )
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=p_chunk)
    @cute.jit
    def exp2_p(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """Apply exp2 using the runtime scale cached before the K/V loop."""
        return self._exp2_p_store(
            self._stage_col_offset(stage_info), row_max, scale_softmax_log2
        )

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def right_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
        section: cutlass.Constexpr[FmhaStage],
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Head-paired stage: apply causal/window mask and compute row_max.

        Unlike the query-paired tail mask, this keeps Q0/Q1 on the same
        sequence tile; q_half selects a Q head, not a later sequence tile.
        Sliding-window tails need both bounds because the final tile can also
        contain keys to the left of the visible window.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        num_softmax_warps = 4
        warp_id_in_sg = cute.arch.warp_idx() % num_softmax_warps
        s_data = self._load_s_chunks(stage_info)
        if cutlass.const_expr(self.cfg.is_causal):
            seq_tile_coord, _, _ = _resolve_work_tile_coords(
                self.cfg, stage_info.work_tile.tile_idx
            )
            tmem_row_id = warp_id_in_sg * cute.arch.WARP_SIZE
            row_in_tile = tmem_row_id + cute.arch.lane_idx()
            index_q = seq_tile_coord * self.cfg.q_tile_m + row_in_tile
            if cutlass.const_expr(self.cfg.window_size_left > 0):
                if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
                    kv_tile_start = bottom_right_window_tile_start(
                        seq_coord=seq_tile_coord,
                        q_tile_m=self.cfg.q_tile_m,
                        kv_tile_n=self.cfg.kv_tile_n,
                        q_offset=q_offset,
                        window_size_left=self.cfg.window_size_left,
                    )
                else:
                    kv_tile_start = cute.math.max(
                        Int32(0),
                        (seq_tile_coord * self.cfg.q_tile_m - self.cfg.window_size_left)
                        // self.cfg.kv_tile_n,
                    )
                if cutlass.const_expr(self.needs_window_tail_left_mask):
                    window_bound_left = bottom_right_window_left_bound(
                        index_q,
                        q_offset,
                        self.cfg.window_size_left,
                    )
            else:
                kv_tile_start = Int32(0)
            if cutlass.const_expr(section == FmhaStage.Loop):
                base_k = (kv_tile_start + stage_info.loop_offset) * self.cfg.kv_tile_n
            else:
                # Tail uses the first tile after the loop domain.
                base_k = (kv_tile_start + stage_info.loop_end) * self.cfg.kv_tile_n
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                chunk_base_k = base_k + chunk_idx * tmem_x
                window_bound_right = index_q + q_offset
                right_oob_start_idx = window_bound_right + Int32(1) - chunk_base_k
                right_oob_start_idx = cute.math.min(
                    cute.math.max(right_oob_start_idx, Int32(0)),
                    Int32(tmem_x),
                )
                mask = cutlass.vector.create_mask([tmem_x], [right_oob_start_idx])
                if cutlass.const_expr(self.needs_window_tail_left_mask):
                    left_oob_end_idx = window_bound_left - chunk_base_k
                    left_mask_inverted = cutlass.vector.create_mask(
                        [tmem_x], [left_oob_end_idx]
                    )
                    all_true_mask = cutlass.vector.create_mask([tmem_x], [tmem_x])
                    left_mask = left_mask_inverted ^ all_true_mask
                    mask = mask & left_mask
                neg_inf = cutlass.vector.full(
                    [tmem_x],
                    self.cfg.qk_acc_dtype(-Float32.inf),
                    dtype=self.cfg.qk_acc_dtype,
                )
                s_data[chunk_idx] = cutlass.vector.where(
                    mask, s_data[chunk_idx], neg_inf
                )
        return self._reduce_row_max(s_data, row_max)

    @cute.jit
    def _apply_causal_mask_for_kv_tile(
        self,
        stage_info: StageInfo,
        s_data: SoftmaxChunks,
        kv_tile_idx: Int32,
        q_offset: Int32,
    ) -> SoftmaxChunks:
        """Apply the query-paired right-edge causal mask to a loaded S tile.

        Query-paired maps q_half=1 to the next sequence tile, so the row index
        includes q_half * q_tile_m. Head-paired causal tails use
        right_masked_row_max() instead. q_offset is cached once per work tile
        so varlen masking does not reload cum_seqlen_q/k in every K/V loop.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        num_softmax_warps = 4
        warp_id_in_sg = cute.arch.warp_idx() % num_softmax_warps
        seq_coord, _, _ = _resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        kv_base = kv_tile_idx * self.cfg.kv_tile_n
        q_min = (
            q_offset
            + seq_coord * self.cfg.cta_tiler[0]
            + self.q_half * self.cfg.q_tile_m
        )
        k_max = kv_base + self.cfg.qk_mma_tiler[1] - Int32(1)
        need_mask = q_min <= k_max
        if need_mask:
            q_idx = q_min + warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx()
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                k_chunk_base = kv_base + chunk_idx * tmem_x
                num_valid = cute.math.min(
                    cute.math.max(q_idx - k_chunk_base + Int32(1), Int32(0)),
                    Int32(tmem_x),
                )
                causal_mask = cutlass.vector.create_mask([tmem_x], [num_valid])
                neg_inf_vec = cutlass.vector.full_like(
                    s_data[chunk_idx], Float32(-Float32.inf)
                )
                s_data[chunk_idx] = cutlass.vector.where(
                    causal_mask, s_data[chunk_idx], neg_inf_vec
                )
        return s_data

    @cute.jit
    def _apply_causal_mask(
        self,
        stage_info: StageInfo,
        s_data: SoftmaxChunks,
        q_offset: Int32,
    ) -> SoftmaxChunks:
        """Apply the tail-stage causal mask to a loaded S tile."""
        return self._apply_causal_mask_for_kv_tile(
            stage_info, s_data, stage_info.loop_end, q_offset
        )

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        q_offset: Int32,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Tail stage: load S, apply causal mask, and compute row_max."""
        s_data = self._load_s_chunks(stage_info)
        if cutlass.const_expr(self.cfg.is_causal):
            s_data = self._apply_causal_mask(stage_info, s_data, q_offset)
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=p_chunk)
    @cute.jit
    def masked_exp2_p(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """Tail stage: apply exp2 using the cached runtime softmax scale."""
        return self._exp2_p_store(
            self._stage_col_offset(stage_info), row_max, scale_softmax_log2
        )

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def invalid_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Tail stage for softmax group 0: identity row_max, no S load."""
        row_max_safe = row_max
        if row_max == -Float32.inf:
            row_max_safe = Float32(0.0)
        _ = stage_info
        return row_max, row_max_safe

    @consumer_work
    @cute.jit
    def invalid_exp2_p(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> None:
        """Tail stage for softmax group 0: no-op because MMA will not read P."""
        pass

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=row_sum)
    @cute.jit
    def softmax_aux_reduce(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        p_chunk: SoftmaxRowSumContribution,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxScalar:
        """Accumulate row_sum from vector P fragments or their scalar sum."""
        _ = stage_info
        if cutlass.const_expr(self.enable_early_tile_sum):
            acc_scale = cute.math.exp2(
                scale_softmax_log2 * (old_row_max - row_max),
                fastmath=True,
            )
            return row_sum * acc_scale + p_chunk
        return self._row_sum_reduction(
            old_row_max=old_row_max,
            row_max=row_max,
            row_sum=row_sum,
            p_chunk=p_chunk,
            scale_softmax_log2=scale_softmax_log2,
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=old_row_max)
    @cute.jit
    def softmax_aux_identity(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
    ) -> SoftmaxScalar:
        """Auxiliary identity path (no P-chunk reduction)."""
        _ = stage_info
        return row_max

    @cute.jit
    def _row_sum_reduction(
        self,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        p_chunk: SoftmaxChunks,
        scale_softmax_log2: SoftmaxScalar,
    ) -> Float32:
        """Accumulate row_sum from P chunks saved by consumer_work."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        scale = scale_softmax_log2
        acc_scale_ = scale * (old_row_max - row_max)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        scaled_sum = row_sum * acc_scale
        if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
            # Use four independent float2 accumulation chains for D256. A
            # single 64-pair chain serializes every FADD behind the
            # preceding result and leaves no row-sum ILP after P publication.
            local_sum_0 = (scaled_sum, scaled_sum)
            local_sum_1 = (Float32(0.0), Float32(0.0))
            local_sum_2 = (Float32(0.0), Float32(0.0))
            local_sum_3 = (Float32(0.0), Float32(0.0))
            for chunk_idx in cutlass.range_constexpr(num_chunks):
                p_chunk_vec = p_chunk[chunk_idx]
                for elem_idx in cutlass.range_constexpr(0, tmem_x, 8):
                    local_sum_0 = cute.arch.add_packed_f32x2(
                        local_sum_0,
                        (p_chunk_vec[elem_idx], p_chunk_vec[elem_idx + 1]),
                        rnd="rn",
                        ftz=False,
                    )
                    local_sum_1 = cute.arch.add_packed_f32x2(
                        local_sum_1,
                        (p_chunk_vec[elem_idx + 2], p_chunk_vec[elem_idx + 3]),
                        rnd="rn",
                        ftz=False,
                    )
                    local_sum_2 = cute.arch.add_packed_f32x2(
                        local_sum_2,
                        (p_chunk_vec[elem_idx + 4], p_chunk_vec[elem_idx + 5]),
                        rnd="rn",
                        ftz=False,
                    )
                    local_sum_3 = cute.arch.add_packed_f32x2(
                        local_sum_3,
                        (p_chunk_vec[elem_idx + 6], p_chunk_vec[elem_idx + 7]),
                        rnd="rn",
                        ftz=False,
                    )
            local_sum_0 = cute.arch.add_packed_f32x2(
                local_sum_0, local_sum_1, rnd="rn", ftz=False
            )
            local_sum_2 = cute.arch.add_packed_f32x2(
                local_sum_2, local_sum_3, rnd="rn", ftz=False
            )
            local_sum_0 = cute.arch.add_packed_f32x2(
                local_sum_0, local_sum_2, rnd="rn", ftz=False
            )
            return local_sum_0[0] + local_sum_0[1]

        local_sum = (scaled_sum, scaled_sum)
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            p_chunk_vec = p_chunk[chunk_idx]
            for idx in cutlass.range_constexpr(tmem_x // 2):
                local_sum = cute.arch.add_packed_f32x2(
                    local_sum,
                    (p_chunk_vec[2 * idx], p_chunk_vec[2 * idx + 1]),
                    rnd="rn",
                    ftz=False,
                )
        return local_sum[0] + local_sum[1]


# ---------------------------------------------------------------------------
# TmemPResource -- P-ready handoff from softmax to UMMA
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemPResource(MemoryResource):
    """Pipeline-only P handoff for split S/P scheduling.

    Softmax stores P into the TMEM columns owned by ``TmemSPResource`` and
    commits this AsyncUmma resource. The MMA task waits on it before issuing
    PV, so the next QK can use the other S/P stage without using the S acquire
    as an implicit P-ready wait.
    """

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_p_offset: Constexpr[int] = field(init=False, default=None)
    tmem_p_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_p_offset: int,
        **kwargs: Any,
    ) -> None:
        """Bind the base P TMEM offset used by the split S/P pipeline."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.tmem_p_offset = tmem_p_offset
        self.tmem_p_base = TaskLocalVariable(
            dtype=Int32,
            default=Int32(tmem_p_offset),
            docs="Selected staged P TMEM column base for PV MMA.",
        )

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return no allocation because P aliases the TmemSP allocation."""
        return []

    @cute.jit
    def create_function_variables(self, context: Optional[Any] = None) -> Int32:
        """Create the staged P-base dataflow slot."""
        _ = context
        return Int32(self.tmem_p_offset)

    @consumer_work(returns=("tmem_p_base",))
    @cute.jit
    def p_base(self, stage_info: StageInfo) -> Int32:
        """Return the staged P column base for the MMA PV producer."""
        tmem_p_base = Int32(self.tmem_p_offset)
        if cutlass.const_expr(self.cfg.mma_softmax_stage > 1):
            tmem_p_base = tmem_p_base + stage_info.stage_idx * self.cfg.qk_mma_tiler[1]
        return tmem_p_base


# ---------------------------------------------------------------------------
# TmemStatsResource -- TMEM correction statistics with AsyncAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemStatsResource(MemoryResource):
    """Correction statistics with an AsyncAsync pipeline.

    Producer: Softmax writes old_max/row_max/row_sum stats. Consumer:
    Correction reads them for O rescaling. Persistent D256 keeps the payload
    in a small staged SMEM ring so the stats no longer alias S/P TMEM columns;
    other schedules retain the original TMEM storage.
    """

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_vec_offset: Constexpr[int] = field(init=False, default=None)
    scale_softmax_log2: cute.Tensor | None = field(init=False, default=None)
    output_scale: cute.Tensor | None = field(init=False, default=None)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    # Precomputed per-warp TMEM vec address (once, before persistent loop).
    tmem_vec_addr_cached: TmemAddr | None = field(init=False, default=None)
    tmem_ptr_vec_cached: TmemPtr | None = field(init=False, default=None)

    _alloc: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)
    _smem_alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    vec_old_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vec_new_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vec_row_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vec_scale: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_vec_offset: int,
        scale_softmax_log2: cute.Tensor | None = None,
        output_scale: cute.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Bind the correction-stat TMEM vector offset and allocation."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.tmem_vec_offset = tmem_vec_offset
        self.scale_softmax_log2 = scale_softmax_log2
        self.output_scale = output_scale
        self._alloc = TmemAllocation(f"tmem_vec_{tmem_vec_offset}", cfg.tmem_stats_cols)
        self._smem_alloc = None
        if cfg.stats_via_smem:
            stats_rows = len(cfg.softmax0_warp_ids) * cute.arch.WARP_SIZE
            # Each row publishes two packed FP32 values. Loop records hold
            # (old_max, new_max), while the final record reuses the pair for
            # (row_sum, new_max).
            stage_bytes = stats_rows * 2 * 4
            self._smem_alloc = SmemAllocation(
                f"smem_vec_{tmem_vec_offset}",
                pipeline_config.num_stages * stage_bytes,
                alignment=16,
            )
        self.tmem_addr_cached = Int32(0)
        self.tmem_vec_addr_cached = Int32(0)
        self.tmem_ptr_vec_cached = _placeholder_tmem_ptr()
        self.vec_old_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Previous row maximum read from TMEM stats.",
        )
        self.vec_new_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Current row maximum read from TMEM stats.",
        )
        self.vec_row_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Softmax denominator read from TMEM stats.",
        )
        self.vec_scale = TaskLocalVariable(
            dtype=Float32,
            default=Float32(1.0),
            docs="Correction scale derived from TMEM stats.",
        )
        self.scale_softmax_log2_value = TaskLocalVariable(
            dtype=Float32,
            # Placeholder before load_scale_softmax_log2 reads the runtime tensor.
            default=Float32(0.0),
            docs="Softmax scale cached from the runtime scale tensor.",
        )
        self.output_scale_value = TaskLocalVariable(
            dtype=Float32,
            # Placeholder before load_output_scale reads the runtime tensor.
            default=Float32(1.0),
            docs="Output scale cached from the runtime scale tensor.",
        )

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return the TMEM allocation for one correction-stat vector."""
        if self.cfg.stats_via_smem:
            return []
        return [self._alloc]

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the staged SMEM stats ring when TMEM aliasing is disabled."""
        if not self.cfg.stats_via_smem:
            return []
        return [self._smem_alloc]

    @cute.jit
    def _init_function_state(self, stage_info: StageInfo) -> None:
        """Initialize vec address fields to establish DSL type before scf.while.

        Real values computed by per-work-tile auxiliary work after setmaxnreg.

        Emits the correction stat slots (vec_old_max, vec_new_max,
        vec_row_sum, vec_scale) consumed by TmemO / SmemO via consumer-
        to-consumer routing; producer-side old_row_max / row_max /
        row_sum slots are auto-mirrored from TmemSP by
        Task.init_variables.
        """
        self.tmem_vec_addr_cached = Int32(0)
        self.tmem_ptr_vec_cached = prims.make_tmem_ptr(Int32(0), cutlass.Int8)
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        self._init_function_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        self._init_function_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns="scale_softmax_log2_value")
    @cute.jit
    def load_scale_softmax_log2(self, stage_info: StageInfo) -> Float32:
        """Load the runtime softmax scale once before the correction loop."""
        _ = stage_info
        if cutlass.const_expr(self.scale_softmax_log2 is None):
            # Safe fallback for validation-only resource construction.
            return Float32(0.0)
        return self.scale_softmax_log2[0]

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns="output_scale_value")
    @cute.jit
    def load_output_scale(self, stage_info: StageInfo) -> Float32:
        """Load the runtime output scale once before the correction loop."""
        _ = stage_info
        if cutlass.const_expr(self.output_scale is None):
            # Identity fallback for validation-only resource construction.
            return Float32(1.0)
        return self.output_scale[0]

    @cute.jit
    def _init_work_tile_state(self, stage_info: StageInfo) -> None:
        """Compute per-warp TMEM vec address and pointer each tile.

        Deferred from function-scope auxiliary work so the arithmetic runs
        after setmaxnreg and does not spill across the register-budget boundary.
        """
        if cutlass.const_expr(self.cfg.stats_via_smem):
            return
        # Softmax producer and correction consumer both use 4 warps.
        num_warps = 4
        warp_id_in_wg = cute.arch.warp_idx() % num_warps
        tmem_raw_addr = self.tmem_addr_cached
        tmem_base_row = tmem_raw_addr >> 16
        tmem_base_col = tmem_raw_addr & Int32(0xFFFF)
        row_id = tmem_base_row + warp_id_in_wg * cute.arch.WARP_SIZE
        self.tmem_vec_addr_cached = (row_id << 16) | (
            tmem_base_col + self.tmem_vec_offset
        )
        self.tmem_ptr_vec_cached = prims.make_tmem_ptr(
            self.tmem_vec_addr_cached, cutlass.Int8
        )
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_work_tile_state(self, stage_info: StageInfo) -> None:
        self._init_work_tile_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_work_tile_state(self, stage_info: StageInfo) -> None:
        self._init_work_tile_state(stage_info)

    @cute.jit
    def _stage_col_offset(self, stage_info: StageInfo) -> Int32 | int:
        """Return the TMEM column offset for stage-scoped stats."""
        stage_col_offset = Int32(0)
        if cutlass.const_expr(self.cfg.stage_scoped_tmem_stats):
            stage_col_offset = stage_info.stage_idx * self.cfg.qk_mma_tiler[1]
        return stage_col_offset

    @cute.jit
    def _stats_smem_ptr(self, stage_info: StageInfo) -> cute.Pointer:
        """Return this warp-group thread's staged SMEM stats slot."""
        context = stage_info.context
        assert context is not None and context.smem_base is not None
        assert self._smem_alloc is not None
        stats_rows = len(self.cfg.softmax0_warp_ids) * cute.arch.WARP_SIZE
        stats_elems_per_row = 2
        stage_elems = stats_rows * stats_elems_per_row
        tidx, _, _ = cute.arch.thread_idx()
        row_idx = tidx % stats_rows
        base_ptr = context.smem_base.data_ptr() + self._smem_alloc.offset
        view = cutlass.Array(
            base_ptr,
            dtype=Float32,
            shape=(self.pipeline_config.num_stages * stage_elems,),
            addrspace=3,
        )
        elem_offset = stage_info.stage_idx * stage_elems + row_idx * stats_elems_per_row
        return view.subview(elem_offset).data_ptr()

    @producer_work
    @cute.jit
    def store_vec(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        final_stats: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Softmax: publish correction statistics for one row.

        The TMEM-backed topology writes four elements per row:
          [0] = old_row_max (previous iteration's max)
          [1] = new_row_max (current iteration's max)
          [2] = row_sum (accumulated softmax denominator)
          [3] = padding

        The compact SMEM-backed topology writes ``[old_max, new_max]`` during
        the loop. Its final publication repurposes slot 0 for ``row_sum``.

        The Correction warp reads these to compute the rescale factor:
          scale = exp2(scale_log2 * (old_max - new_max))
        and to forward row_sum to SmemO for the final normalization.
        """
        if cutlass.const_expr(self.cfg.stats_via_smem):
            stat0 = old_row_max
            if cutlass.const_expr(final_stats):
                stat0 = row_sum
            vec_data = cutlass.Vector.from_elements(
                (stat0, row_max),
                self.cfg.qk_acc_dtype,
            )
            self._stats_smem_ptr(stage_info).store(vec_data, alignment=8)
        else:
            vec_data = cutlass.Vector.from_elements(
                (old_row_max, row_max, row_sum, Float32(0.0)),
                self.cfg.qk_acc_dtype,
            )
            tmem_ptr_vec = prims.make_tmem_ptr(
                self.tmem_vec_addr_cached + self._stage_col_offset(stage_info),
                cutlass.Int8,
            )
            prims.tcgen05_st(
                "32x32b",
                tmem_ptr_vec,
                vec_data,
            )
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _read_vec(
        self,
        stage_info: StageInfo,
        scale_softmax_log2: SoftmaxScalar,
        final_stats: cutlass.Constexpr[bool] = False,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar, SoftmaxScalar, SoftmaxScalar]:
        """Read correction stats from TMEM and cache in consumer_vars.

        CRITICAL: This read must happen here (immediately after the
        pipeline wait) rather than being deferred to TmemO.correct,
        because the TMEM stats region (cols 128-131 for TmemStats1, cols 0-3
        for TmemStats0) overlaps with the S0/S1 score regions. After the MMA warp
        commits O and continues to QK→S, the UMMA write to S can
        overwrite the stats data.  Reading here, before the O pipeline
        wait, ensures the stats are captured before any S overwrite.

        TmemO.correct retrieves the cached values from this
        resource's consumer_vars via a direct reference.
        """
        tmem_shape_vec = "32x32b"
        # Vector layout: [old_max, new_max, row_sum, pad].
        tmem_x_vec = self.cfg.tmem_stats_cols

        if cutlass.const_expr(self.cfg.stats_via_smem):
            vec_rmem = self._stats_smem_ptr(stage_info).load(
                count=2,
                alignment=8,
            )
        else:
            tmem_ptr_vec = prims.make_tmem_ptr(
                self.tmem_vec_addr_cached + self._stage_col_offset(stage_info),
                self.cfg.qk_acc_dtype,
            )
            vec_rmem = cutlass.Array(self.cfg.qk_acc_dtype, tmem_x_vec)
            vec_rmem[0:tmem_x_vec] = prims.tcgen05_ld(
                tmem_shape_vec, tmem_ptr_vec, num=tmem_x_vec
            )
            cute.arch.fence_view_async_tmem_load()

        vec_old_max = vec_rmem[0]
        vec_new_max = vec_rmem[1]
        vec_row_sum = Float32(0.0)
        if cutlass.const_expr(not self.cfg.stats_via_smem):
            vec_row_sum = vec_rmem[2]
        scale = Float32(1.0)
        if cutlass.const_expr(not (self.cfg.stats_via_smem and final_stats)):
            scale_ = scale_softmax_log2 * (vec_old_max - vec_new_max)
            scale = cute.math.exp2(scale_, fastmath=True)
        else:
            vec_row_sum = vec_rmem[0]
            vec_old_max = vec_new_max
        _ = stage_info
        return vec_old_max, vec_new_max, vec_row_sum, scale

    @consumer_work(returns=(vec_old_max, vec_new_max, vec_row_sum, vec_scale))
    @cute.jit
    def read_vec(
        self,
        stage_info: StageInfo,
        *,
        scale_softmax_log2: SoftmaxScalar,
        final_stats: cutlass.Constexpr[bool] = False,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar, SoftmaxScalar, SoftmaxScalar]:
        """Read correction stats using the cached runtime softmax scale."""
        return self._read_vec(stage_info, scale_softmax_log2, final_stats)


# ---------------------------------------------------------------------------
# TmemOResource -- TMEM O accumulation with UmmaAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemOResource(MemoryResource):
    """TMEM O accumulation with a topology-derived UmmaAsync pipeline.

    Producer: MMA writes P*V -> O (double-buffered O0/O1).
    Consumer: Correction rescales O in-place.

    Paired schedules use two stages so MMA can commit O0, work on O1,
    commit O1, then acquire O0. SMEM-stats D256 uses one stage because it
    writes one physical O accumulator.
    """

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_o0_offset: Constexpr[int] = field(init=False, default=None)
    tmem_o1_offset: Constexpr[int] = field(init=False, default=None)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    # Precomputed TMEM raw pointer (inttoptr of tmem_addr_cached).
    tmem_ptr_raw_cached: TmemPtr | None = field(init=False, default=None)
    # Precomputed per-warp TMEM O address base: (row_id << 16) | tmem_base_col.
    # consumer_work adds tmem_o_offset to get the final O0/O1 address.
    tmem_o_addr_base_cached: TmemAddr | None = field(init=False, default=None)
    # P-stage base supplied by TmemPResource for split S/P scheduling.
    tmem_p_base_cached: TmemAddr | None = field(init=False, default=None)
    # References to TmemStats resources for reading cached correction stats.
    # consumer_work reads stats from these instead of from TMEM, because
    # the stats TMEM region overlaps with S0/S1 and can be overwritten by
    # MMA's QK→S before correction reads it.
    tmem_vec0_resource: TmemStatsResource | None = field(init=False, default=None)
    tmem_vec1_resource: TmemStatsResource | None = field(init=False, default=None)

    _alloc_o0: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)
    _alloc_o1: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_o0_offset: int,
        tmem_o1_offset: int,
        tmem_vec0_resource: TmemStatsResource | None = None,
        tmem_vec1_resource: TmemStatsResource | None = None,
        **kwargs: Any,
    ) -> None:
        """Bind O TMEM offsets and correction-stat resources."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.tmem_o0_offset = tmem_o0_offset
        self.tmem_o1_offset = tmem_o1_offset
        self.tmem_vec0_resource = tmem_vec0_resource
        self.tmem_vec1_resource = tmem_vec1_resource
        self._alloc_o0 = TmemAllocation("tmem_o0", 128)
        self._alloc_o1 = TmemAllocation("tmem_o1", 128)
        self.tmem_addr_cached = Int32(0)
        self.tmem_ptr_raw_cached = _placeholder_tmem_ptr()
        self.tmem_o_addr_base_cached = Int32(0)

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return the TMEM allocations for the double-buffered O accumulators."""
        return [self._alloc_o0, self._alloc_o1]

    @property
    def loop_offset_sensitive(self) -> bool:
        """Return true because PV accumulation scale depends on loop_offset."""
        # producer_work uses loop_offset to compute scale_d.
        return True

    @cute.jit
    def _init_function_state(self, stage_info: StageInfo) -> None:
        """Precompute MMA-warp TMEM raw pointer (once, ungated).

        Per-warp O address base for correction warps is deferred to
        per-work-tile auxiliary work to avoid crossing the setmaxnreg boundary.
        tmem_o_addr_base_cached initialized to Int32(0) to establish DSL type.

        Pure consumer/producer of upstream emitters — emits no
        consumer vars itself.  Producer-side desc_v_base slot is
        auto-mirrored from SmemKV; consumer-side vec_old_max /
        vec_new_max slots are auto-mirrored from TmemStats via
        consumer-to-consumer routing in the captured schedule.
        """
        self.tmem_ptr_raw_cached = prims.make_tmem_ptr(
            self.tmem_addr_cached, cutlass.Int8
        )
        self.tmem_o_addr_base_cached = Int32(0)
        self.tmem_p_base_cached = Int32(0)
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self._init_function_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_correction_state(self, stage_info: StageInfo) -> None:
        self._init_function_state(stage_info)

    @cute.jit
    def _init_work_tile_state(self, stage_info: StageInfo) -> None:
        """Compute per-warp TMEM O address base each tile (after setmaxnreg)."""
        num_correction_warps = 4
        warp_id_in_wg = cute.arch.warp_idx() % num_correction_warps
        tmem_raw_addr = self.tmem_addr_cached
        tmem_base_row = tmem_raw_addr >> 16
        tmem_base_col = tmem_raw_addr & Int32(0xFFFF)
        row_id = tmem_base_row + warp_id_in_wg * cute.arch.WARP_SIZE
        self.tmem_o_addr_base_cached = (row_id << 16) | tmem_base_col
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_work_tile_state(self, stage_info: StageInfo) -> None:
        self._init_work_tile_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_correction_work_tile_state(self, stage_info: StageInfo) -> None:
        self._init_work_tile_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def set_p_base(self, stage_info: StageInfo, *, tmem_p_base: Int32) -> None:
        """Cache the P TMEM column base selected by TmemPResource."""
        _ = stage_info
        self.tmem_p_base_cached = tmem_p_base

    @producer_work
    @cute.jit
    def pv_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_v_base: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[FmhaStage],
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        inst_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """PV MMA: P*V -> O (double-buffered O0/O1).

        Uses captured schedule section and call index to select O0/O1 and
        scale_d statically.
        Reads P from TMEM (via tmem_p offset on the corresponding SP resource)
        and V descriptor from SmemV consumer vars.

        In causal mode with no Q right offset, skips P0*V→O0 MMA in the last
        LOOP iteration, since Softmax0's domain is N-2 but MMA's domain is N-1.
        The task domain pads partial final CTAs so this slot is always outside
        peer0's causal reach.
        """
        if cutlass.const_expr(section == FmhaStage.Head):
            writes_o0 = True
            first_o0_write = True
            first_o1_write_maybe = False
        elif cutlass.const_expr(section == FmhaStage.Loop):
            writes_o0 = inst_idx == 1
            first_o0_write = False
            first_o1_write_maybe = inst_idx == 0
        else:
            writes_o0 = False
            first_o0_write = False
            first_o1_write_maybe = True

        # In causal mode, check if O0 MMA should skip the last LOOP iteration.
        skip_o0_invalid = False
        if cutlass.const_expr(
            self.cfg.skip_causal_invalid_peer0
            and writes_o0
            and section == FmhaStage.Loop
        ):
            if not is_tail:
                skip_o0_invalid = stage_info.loop_offset == (stage_info.loop_end - 1)

        if not skip_o0_invalid:
            tmem_ptr_raw = self.tmem_ptr_raw_cached

            if cutlass.const_expr(self.cfg.v_dtype.width == 8):
                mma_kind = prims.Tcgen05MMAKind.F8F6F4
                # E4M3 operands use the Float16 encoding handle.
                ab_format = cutlass.Float16
            else:
                mma_kind = prims.Tcgen05MMAKind.F16
                if cutlass.const_expr(self.cfg.v_dtype == cutlass.BFloat16):
                    ab_format = cutlass.BFloat16
                else:
                    ab_format = cutlass.Float16

            idesc_pv = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=ab_format,
                b_dtype=ab_format,
                n_dim=(
                    self.cfg.head_dim_per_stage_kv
                    if self.cfg.single_qkv_instance and self.cfg.pv_mma_tiler[1] == 256
                    else self.cfg.pv_mma_tiler[1]
                ),
                m_dim=self.cfg.pv_mma_tiler[0],
                # V is row-major / MN-major.
                b_major=1,
            )

            pv_n_dim = self.cfg.pv_mma_tiler[1]
            num_head_dim_stages = 1
            if cutlass.const_expr(
                self.cfg.single_qkv_instance and self.cfg.pv_mma_tiler[1] == 256
            ):
                pv_n_dim = self.cfg.head_dim_per_stage_kv
                num_head_dim_stages = self.cfg.pv_mma_tiler[1] // pv_n_dim
            head_dim_stage_start = 0
            num_head_dim_stages_to_issue = num_head_dim_stages
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                num_head_dim_stages_to_issue = 1
                head_dim_stage_start = head_dim_stage_idx

            k_dim_per_mma = 16
            if cutlass.const_expr(self.cfg.v_dtype.width != 16):
                k_dim_per_mma = 32
            num_kphases_pv = self.cfg.pv_mma_tiler[2] // k_dim_per_mma
            inc_tmem_p = (
                k_dim_per_mma * self.cfg.v_dtype.width // self.cfg.qk_acc_dtype.width
            )
            tma_copy_iters_per_head_dim_stage = (
                self.cfg.tma_copy_qkv_iters // num_head_dim_stages
            )
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                tma_copy_iters_per_head_dim_stage = self.cfg.tma_copy_kv_stage_iters
            inc_bytes_v = (
                k_dim_per_mma
                * (pv_n_dim // tma_copy_iters_per_head_dim_stage)
                * self.cfg.v_dtype.width
                // 8
            )
            kv_chunk_bytes = (
                self.cfg.tma_copy_kv_bytes // self.cfg.tma_copy_kv_stage_iters
            )
            head_dim_stage_bytes_v = kv_chunk_bytes * tma_copy_iters_per_head_dim_stage

            # Select O buffer and P offset at trace time (compile-time constant)
            if cutlass.const_expr(self.cfg.single_qkv_instance or writes_o0):
                tmem_ptr_o = tmem_ptr_raw.subview(self.tmem_o0_offset)
                tmem_p_base = self.cfg.tmem_p0_offset
            else:
                tmem_ptr_o = tmem_ptr_raw.subview(self.tmem_o1_offset)
                tmem_p_base = self.cfg.tmem_p1_offset
            if cutlass.const_expr(self.cfg.single_qkv_instance):
                tmem_p_base = self.tmem_p_base_cached

            # scale_d at trace time.
            # O0 (even counters): HEAD initializes O0, so all
            # subsequent O0 writes (counter 2, 4, ...) always accumulate.
            # O1 (odd counters): first written in LOOP. Dynamic check needed
            # because with loop peeling + causal domain=1, the peeled iteration
            # may be the first O1 write (loop_offset=0 → scale_d=False).
            # TAIL O1: always accumulates because LOOP or the peeled iteration
            # already wrote O1 before TAIL runs.
            if cutlass.const_expr(self.cfg.single_qkv_instance):
                if cutlass.const_expr(self.cfg.has_tmem_p_pipeline):
                    if cutlass.const_expr(section == FmhaStage.Loop):
                        scale_d = stage_info.loop_offset > stage_info.loop_start
                    elif cutlass.const_expr(is_tail):
                        scale_d = stage_info.loop_end > stage_info.loop_start
                    else:
                        scale_d = False
                elif cutlass.const_expr(is_tail):
                    scale_d = stage_info.loop_end > 0
                elif cutlass.const_expr(section == FmhaStage.Loop):
                    scale_d = stage_info.loop_offset > 0
                else:
                    scale_d = False
            elif cutlass.const_expr(first_o0_write):
                # Head O0 is the first O0 write.
                scale_d = False
            elif cutlass.const_expr(first_o1_write_maybe and section == FmhaStage.Tail):
                # TAIL O1 accumulates if LOOP already wrote O1 (domain >= 1).
                # When domain=0, TAIL is the first O1 write.
                scale_d = stage_info.loop_end > 0
            elif cutlass.const_expr(first_o1_write_maybe):
                # Loop O1 initializes on the first iteration and accumulates later.
                scale_d = stage_info.loop_offset > 0
            else:
                # O0 after the head write always accumulates.
                scale_d = True
            # Prevent LLVM from rematerializing V descriptor inside
            # each elect_sync block (same pattern as QK MMA above).
            desc_v_base_ = freeze_smem_descriptor(desc_v_base)

            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                tmem_ptr_o_stage = tmem_ptr_o.subview(head_dim_stage_start * pv_n_dim)
                scale_d_stage = scale_d
                for k_idx in cutlass.range_constexpr(num_kphases_pv):
                    dp = tmem_ptr_raw.subview(tmem_p_base + k_idx * inc_tmem_p)
                    increment = (inc_bytes_v * k_idx) >> 4
                    dv = desc_v_base_ + increment
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            prims.CTAGroup.CTA_1,
                            tmem_ptr_o_stage,
                            dp,
                            dv,
                            idesc_pv,
                            scale_d_stage,
                        )
                    scale_d_stage = True
            else:
                for head_dim_stage_idx in cutlass.range_constexpr(
                    num_head_dim_stages_to_issue
                ):
                    tmem_ptr_o_stage = tmem_ptr_o.subview(head_dim_stage_idx * pv_n_dim)
                    v_stage_increment = (
                        head_dim_stage_bytes_v * head_dim_stage_idx
                    ) >> 4
                    scale_d_stage = scale_d
                    for k_idx in cutlass.range_constexpr(num_kphases_pv):
                        dp = tmem_ptr_raw.subview(tmem_p_base + k_idx * inc_tmem_p)
                        increment = v_stage_increment + ((inc_bytes_v * k_idx) >> 4)
                        dv = desc_v_base_ + increment
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                prims.CTAGroup.CTA_1,
                                tmem_ptr_o_stage,
                                dp,
                                dv,
                                idesc_pv,
                                scale_d_stage,
                            )
                        scale_d_stage = True

    @consumer_work
    @cute.jit
    def correct(
        self,
        stage_info: StageInfo,
        *,
        vec_old_max: SoftmaxScalar,
        vec_new_max: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        inst_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Correction using the scale cached by TmemStatsResource."""
        self._correct_impl(
            stage_info,
            vec_old_max=vec_old_max,
            vec_new_max=vec_new_max,
            scale_softmax_log2=Float32(0.0),
            vec_scale=vec_scale,
            use_cached_scale=True,
            inst_idx=inst_idx,
            is_tail=is_tail,
        )

    @cute.jit
    def _correct_impl(
        self,
        stage_info: StageInfo,
        *,
        vec_old_max: SoftmaxScalar,
        vec_new_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        use_cached_scale: Constexpr[bool],
        inst_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Correction: read cached stats, compute scale, rescale O.

        Reads the correction stats [old_max, new_max, row_sum] from the
        TmemStatsResource's consumer_vars (cached by TmemStats.read_vec
        right after the pipeline wait).  This avoids a TMEM race: the stats
        region overlaps with S0/S1, and MMA's QK->S can overwrite it
        between O commit and the time Correction reads here.

        Uses inst_idx to select O0/O1 and forwards row_sum to
        SmemO.producer_vars for the tail epilog.

        With skip_correction enabled: uses vote_ballot_sync to check if
        old_max == new_max across all threads. If true, scale=1.0 and
        we skip the expensive TMEM load+rescale+store.
        """
        # Select O offset from captured correction-call position.
        if cutlass.const_expr(self.cfg.single_qkv_instance or inst_idx == 0):
            tmem_o_offset = self.tmem_o0_offset
        else:
            tmem_o_offset = self.tmem_o1_offset

        # In causal mode with no Q right offset, skip the invalid O0 correction
        # in the last LOOP iteration. The task domain pads partial final CTAs so
        # this slot is always outside peer0's causal reach.
        skip_o0_invalid = False
        if cutlass.const_expr(
            self.cfg.skip_causal_invalid_peer0
            and not self.cfg.single_qkv_instance
            and inst_idx == 0
        ):
            # This is O0 correction; check if this is the last LOOP iteration.
            if not is_tail:
                skip_o0_invalid = stage_info.loop_offset == (stage_info.loop_end - 1)

        # Check if we should skip correction (when old_max == new_max)
        should_rescale = True
        if cutlass.const_expr(self.cfg.enable_skip_correction):
            vote_ballot_cnt = cute.arch.vote_ballot_sync(vec_old_max != vec_new_max)
            should_rescale = vote_ballot_cnt != Int32(0)

        scale = Float32(1.0)
        if should_rescale:
            if cutlass.const_expr(use_cached_scale):
                scale = vec_scale
            else:
                scale_ = scale_softmax_log2 * (vec_old_max - vec_new_max)
                scale = cute.math.exp2(scale_, fastmath=True)

        # PTX ISA 9.7.16.6.4.4: Non-pipelined instructions, different thread.
        # MMA (Thread 0) does tcgen05.mma → tcgen05.commit on O_full.
        # Correction (Thread 1) does mbarrier.try_wait on O_full → tcgen05.ld.
        # The fence orders the prior tcgen05.commit's completion with our tcgen05.ld.
        from cutlass.experimental import primitives as _prims

        _prims.tcgen05_fence("after")

        # Only rescale if old_max != new_max AND not in invalid O0 iteration
        if should_rescale and not skip_o0_invalid:
            # Load O, rescale, store back
            tmem_o_addr = self.tmem_o_addr_base_cached + tmem_o_offset

            tmem_shape = "32x32b"
            tmem_x = 16

            num_iters = self.cfg.cta_tiler[2] // tmem_x
            for i in cutlass.range_constexpr(num_iters):
                tmem_tile_addr = tmem_o_addr + i * tmem_x
                tmem_ptr = cutlass.inttoptr(
                    tmem_tile_addr,
                    mem_space=6,
                    dtype=self.cfg.pv_acc_dtype,
                )

                # Load from TMEM as vector, scale, store back
                o_vec = prims.tcgen05_ld(tmem_shape, tmem_ptr, num=tmem_x)
                cute.arch.fence_view_async_tmem_load()
                scale_vec = cutlass.vector.full_like(o_vec, scale)
                o_scaled = o_vec * scale_vec
                prims.tcgen05_st(tmem_shape, tmem_ptr, o_scaled)

            cute.arch.fence_view_async_tmem_store()
        # else: skip TMEM rescale entirely when scale=1.0


# ---------------------------------------------------------------------------
# SmemOResource -- SMEM O buffer with AsyncAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemOResource(MemoryResource):
    """SMEM buffer for one O-subtile (1-stage AsyncAsync pipeline).

    Each instance (``smem_o_0``, ``smem_o_1``) owns a distinct smem
    region for its subtile, so the checker can verify that stage-0
    and stage-1 accesses never conflict.

    Producer: CorrectionTask (correction_epilog writes converted O to SMEM).
    Consumer: EpilogueTask (TMA stores O from SMEM to GMEM).
    """

    sO_array: cutlass.Array = field(init=False, default=None)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_o_addr_base_cached: TmemAddr | None = field(init=False, default=None)
    # Reference to the TmemStats resource for this stage's correction stats.
    tmem_vec_resource: TmemStatsResource | None = field(init=False, default=None)
    stage_idx: Constexpr[int] = field(init=False, default=0)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    head_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    batch_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seq_coord_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        stage_idx: int = 0,
        tmem_vec_resource: TmemStatsResource | None = None,
        **kwargs: Any,
    ) -> None:
        """Bind one output subtile stage and reserve its SMEM staging buffer."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.stage_idx = stage_idx
        self.tmem_vec_resource = tmem_vec_resource
        stage_elements = cfg.sO_stage_elements
        size_bytes = stage_elements * cfg.o_dtype.width // 8
        self._alloc = SmemAllocation(
            f"smem_o_{stage_idx}", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sO_array = _placeholder_smem_array(cfg.o_dtype)
        self.tmem_addr_cached = Int32(0)
        self.tmem_o_addr_base_cached = Int32(0)
        self.head_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/O head coordinate for the output subtile.",
        )
        self.batch_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Batch coordinate for the output subtile.",
        )
        self.seq_coord_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Output row coordinate for the output subtile.",
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation for this O staging subtile."""
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Derive sO_array from context (once, ungated).

        Per-warp TMEM O address base deferred to per-work-tile auxiliary work
        to avoid crossing the setmaxnreg boundary.
        tmem_o_addr_base_cached initialized to Int32(0) to establish DSL type.

        Emits per-tile output coordinates consumed downstream by
        GmemO via the EpilogueTask; producer-side vec_row_sum /
        vec_scale slots are auto-mirrored from TmemStats by
        Task.init_variables.
        """
        smem_base = stage_info.context.smem_base
        stage_elements = self.cfg.sO_stage_elements
        self.sO_array = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=self.cfg.o_dtype,
            shape=(stage_elements,),
            addrspace=3,
        )
        self.tmem_o_addr_base_cached = Int32(0)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_output_state(self, stage_info: StageInfo) -> None:
        self._init_smem_state(stage_info)

    @cute.jit
    def _init_work_tile_state(self, stage_info: StageInfo) -> None:
        """Compute per-warp TMEM O address base each tile (after setmaxnreg)."""
        num_correction_warps = 4
        warp_id_in_wg = cute.arch.warp_idx() % num_correction_warps
        tmem_raw_addr = self.tmem_addr_cached
        tmem_base_row = tmem_raw_addr >> 16
        tmem_base_col = tmem_raw_addr & Int32(0xFFFF)
        row_id = tmem_base_row + warp_id_in_wg * cute.arch.WARP_SIZE
        self.tmem_o_addr_base_cached = (row_id << 16) | tmem_base_col
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_work_tile_state(self, stage_info: StageInfo) -> None:
        self._init_work_tile_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_output_work_tile_state(self, stage_info: StageInfo) -> None:
        self._init_work_tile_state(stage_info)

    @producer_work
    @cute.jit
    def store_o(
        self,
        stage_info: StageInfo,
        *,
        vec_row_sum: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        output_scale: SoftmaxScalar,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Store O using the output scale cached before the loop."""
        self._store_o(
            stage_info,
            vec_row_sum=vec_row_sum,
            vec_scale=vec_scale,
            output_scale=output_scale,
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @cute.jit
    def _store_o(
        self,
        stage_info: StageInfo,
        *,
        vec_row_sum: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        output_scale: SoftmaxScalar,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Correct and normalize O in one TMEM pass, then stage it in SMEM."""
        if cutlass.const_expr(self.stage_idx == 0):
            tmem_o_offset = self.cfg.tmem_o0_offset
        else:
            tmem_o_offset = self.cfg.tmem_o1_offset
        sO_base = self.sO_array

        # Read precomputed correction scale and row_sum from TmemStats.
        # vec_scale = exp2(scale_log2 * (old_max - new_max)) was computed
        # in TmemStats.consumer_work.
        correction_scale = vec_scale
        scale = output_scale * correction_scale / vec_row_sum

        num_correction_warps = 4
        tidx, _, _ = cute.arch.thread_idx()
        tid_in_wg = tidx % (cute.arch.WARP_SIZE * num_correction_warps)

        o_head_dim = self.cfg.epi_tile[1]
        tma_copy_o_iters = self.cfg.tma_copy_o_iters
        if cutlass.const_expr(self.cfg.stage_o_by_head_dim):
            o_head_dim = self.cfg.head_dim_per_stage_kv
            tma_copy_o_iters = self.cfg.tma_copy_o_stage_iters

        tmem_offset_o = (
            self.tmem_o_addr_base_cached
            + tmem_o_offset
            + head_dim_stage_idx * o_head_dim
        )

        tmem_shape = "32x32b"
        tmem_x = 16
        num_iters = o_head_dim // tmem_x

        smem_o_swizzle = _smem_o_swizzle(self.cfg)

        d_block_size = o_head_dim // tma_copy_o_iters
        row_offset = tid_in_wg * d_block_size

        for i in cutlass.range_constexpr(num_iters):
            tmem_offset_tile = tmem_offset_o + i * tmem_x

            tmem_ptr = cutlass.inttoptr(
                tmem_offset_tile,
                mem_space=6,
                dtype=self.cfg.pv_acc_dtype,
            )

            o_rmem = prims.tcgen05_ld(tmem_shape, tmem_ptr, num=tmem_x)
            cute.arch.fence_view_async_tmem_load()

            scale_vec = cutlass.vector.full_like(o_rmem, scale)
            o_rmem = o_rmem * scale_vec

            o_rmem_dtype = o_rmem.to(self.cfg.o_dtype)

            col_offset = (i * tmem_x) % d_block_size
            block_idx = (i * tmem_x) // d_block_size
            block_offset = block_idx * self.cfg.tma_copy_o_granu_elems
            smem_offset = block_offset + row_offset + col_offset
            smem_ptr = (sO_base.subview(smem_offset)).data_ptr()

            if cutlass.const_expr(self.cfg.o_dtype.width == 8):
                o_rmem_i8 = o_rmem_dtype.bitcast(cutlass.Int8)
                smem_ptr.store_swizzled(o_rmem_i8, alignment=64, swizzle=smem_o_swizzle)
            else:
                smem_ptr.store_swizzled(
                    o_rmem_dtype, alignment=64, swizzle=smem_o_swizzle
                )

        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

    @consumer_work(returns=(head_coord, batch_coord, seq_coord_q))
    @cute.jit
    def compute_output_coords(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32]:
        """Return output-tile coordinates for downstream GMEM TMA store."""
        seq_coord, head_coord, batch_coord = _resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        seq_coord_q = seq_coord * self.cfg.q_tile_m * self.cfg.work_tile_q_seq_tiles
        return head_coord, batch_coord, seq_coord_q


# ---------------------------------------------------------------------------
# GmemOResource -- global memory O output (no pipeline)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GmemOResource(MemoryResource):
    """TMA store for one O-subtile to global memory.

    Each instance (``gmem_o_0``, ``gmem_o_1``) has its own smem
    staging region aliased with the matching ``SmemOResource``
    instance, keeping per-subtile accesses independently trackable.
    No pipeline — point-access only.

    Producer: EpilogueTask stores O tiles from SMEM to GMEM via TMA.
    """

    tma_o_desc: cutlass.Pointer | None = field(init=False, default=None)
    cum_seqlen_q: cute.Tensor | None = field(init=False, default=None)
    sO_array: cutlass.Array = field(init=False, default=None)
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    stage_idx: Constexpr[int] = field(init=False, default=0)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    def __init__(
        self,
        tma_o_desc: cutlass.Pointer | None,
        cum_seqlen_q: cute.Tensor | None,
        cfg: FmhaConfig,
        stage_idx: int = 0,
        **kwargs: Any,
    ) -> None:
        """Bind the O TMA descriptor and reserve store-side SMEM staging."""
        super().__init__(**kwargs)
        self.tma_o_desc = tma_o_desc
        self.cum_seqlen_q = cum_seqlen_q
        self.cfg = cfg
        self.stage_idx = stage_idx
        stage_elements = cfg.sO_stage_elements
        size_bytes = stage_elements * cfg.o_dtype.width // 8
        self._alloc = SmemAllocation(
            f"gmem_o_{stage_idx}_smem", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sO_array = _placeholder_smem_array(cfg.o_dtype)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation used as the O TMA store source."""
        return [self._alloc]

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        """Materialize the O store staging buffer; this resource emits no slots."""
        # Pure sink — producer-side head_coord / batch_coord / seq_coord_q
        # slots are auto-mirrored from upstream SmemO by Task.init_variables.
        smem_base = stage_info.context.smem_base
        stage_elements = self.cfg.sO_stage_elements
        self.sO_array = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=self.cfg.o_dtype,
            shape=(stage_elements,),
            addrspace=3,
        )

    @producer_work
    @cute.jit
    def tma_store(
        self,
        stage_info: StageInfo,
        *,
        head_coord: Int32,
        batch_coord: Int32,
        seq_coord_q: Int32,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        correction_fused: cutlass.Constexpr[bool] = False,
    ) -> None:
        """TMA store O from SMEM to GMEM.

        Coordinates are produced by SmemOResource.consumer_work() and routed
        via schedule dataflow into this producer call.  In head-paired mode,
        the two output stages map to consecutive Q heads instead of consecutive
        Q sequence tiles.
        """
        head_coord = (
            head_coord * self.cfg.work_tile_q_heads
            + self.stage_idx * self.cfg.peer_q_head_stride
        )
        seq_offset_o = (
            seq_coord_q
            + self.stage_idx * self.cfg.peer_q_seq_tile_stride * self.cfg.q_tile_m
        )
        sO_base = self.sO_array
        should_store = True
        q_seq_extent = Int32(0)
        if cutlass.const_expr(self.cfg.has_varlen):
            if cutlass.const_expr(self.cfg.has_uniform_varlen):
                cuseqlen_q = batch_coord * Int32(self.cfg.uniform_seq_len_q)
                seq_end = cuseqlen_q + Int32(self.cfg.uniform_seq_len_q)
            else:
                cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
                seq_end = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
            seq_offset_o = cuseqlen_q + seq_offset_o
            q_seq_extent = seq_end - seq_offset_o
            should_store = seq_offset_o < seq_end

        tma_copy_o_iters = self.cfg.tma_copy_o_iters
        if cutlass.const_expr(self.cfg.stage_o_by_head_dim):
            tma_copy_o_iters = self.cfg.tma_copy_o_stage_iters

        is_store_warp = True
        if cutlass.const_expr(correction_fused):
            # elect_sync elects one lane *per warp*.  A correction-fused call
            # runs on four warps, so only the first correction warp may enter
            # the TMA issue/commit/wait body.
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            is_store_warp = warp_idx == self.cfg.correction_warp_ids[0]
        if is_store_warp:
            if should_store:
                if prims.elect_sync():
                    for i in cutlass.range_constexpr(tma_copy_o_iters):
                        d_offset = (
                            head_dim_stage_idx * self.cfg.head_dim_per_stage_kv
                            + i * self.cfg.tma_copy_o_granu_inner
                        )
                        o_coords = (d_offset, head_coord, seq_offset_o, batch_coord)
                        if cutlass.const_expr(self.cfg.has_varlen):
                            o_coords = (d_offset, head_coord, seq_offset_o)
                            o_coords = transform_ragged_coords(
                                o_coords,
                                ragged_dim_idx=2,
                                ragged_box_size=self.cfg.epi_tile[0],
                                ragged_extent=q_seq_extent,
                            )
                        prims.cp_async_bulk_tensor_global_shared_cta(
                            self.tma_o_desc,
                            sO_base.subview(i * self.cfg.tma_copy_o_granu_elems),
                            o_coords,
                        )
            # should_store is CTA-uniform because it depends only on batch and
            # Q tile coordinates. Keep commit paired with an actual store.
            if should_store:
                prims.cp_async_bulk_commit_group()
                if cutlass.const_expr(self.cfg.gmem_o_store_wait_after_write):
                    prims.cp_async_bulk_wait_group(0, read=True)
