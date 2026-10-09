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
- SmemKVResource  : SMEM K/V buffer, TmaUmma pipeline. Load -> MMA. The D256
                    depth is derived from the public SM100 SMEM capacity. One
                    instance holds both K and V when they share a dtype; a
                    QK/PV dtype mismatch instantiates one K-only and one V-only
                    instance, each sized from its own dtype.
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
from cutlass import Float32, Int16, Int32, Int64
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
from ...vc_attention import VC_MEAN_GROUP_TILES, VC_MEAN_MMA_K, VC_MEAN_OPERANDS
from .vc_resources import (
    E4M3_CODES_PER_OCTAVE,
    VC_EXPCAST_CODE_BIAS,
    VC_MEAN_TILE_LBO,
    VC_MEAN_TILE_SBO,
    _expcast_e4m3_quad_relu,
    _expcast_e4m3_quad_relu_init,
    _f16x2_sum4,
    _split_bf16_hi_lo_word,
)
from .helpers import (
    bottom_right_window_left_bound,
    bottom_right_window_tile_start,
    freeze_smem_descriptor,
    load_tmem_32x32b_max,
    variable_window_cta_min_start,
)
from cutlass.experimental import primitives as prims
from cutlass._mlir.dialects import arith as arith_dialect

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
# Packed fp8 P words carried from exp2_p to store_p when P is staged in SMEM.
_tmem_sp_pwords: dict[int, tuple] = {}
# VC-Attention-QK16: the log2 softmax scale carried from the row-max pass to exp2_p
# and the fp32 tile row sum carried from exp2_p to store_p.
_tmem_sp_tile_scale: dict[int, Any] = {}
_tmem_sp_tile_sum: dict[int, Any] = {}


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
    """Pack four FP32 values into one e4m3x4 word. Two b16 conversions
    are joined by mov.b32, which ptxas emits as F2FP pairs with no PRMT."""
    return cute.arch.inline_ptx(
        """{
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};
            cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};
            mov.b32 {$w0}, {lo, hi};
        }""",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )


@cute.jit
def _f32_bits(x: Float32) -> Int32:
    return cutlass.Vector.from_elements((x,), Float32).bitcast(Int32)[0]


@cute.jit
def _f32_from_bits(x: Int32) -> Float32:
    return cutlass.Vector.from_elements((x,), Int32).bitcast(Float32)[0]


@cute.jit
def _exp2_fma_packed(x0: Float32, x1: Float32) -> tuple[Float32, Float32]:
    """exp2 of two fp32 values on the FMA pipe, 2^x = 2^floor(x) * poly(x - floor(x)).
    Same routine as ex2_emulation_f32x2_value in the SM110 GQA decode kernel."""
    # Below -127 the result flushes to zero in e4m3 anyway.
    x0 = cute.math.max(x0, Float32(-127.0), ftz=True)
    x1 = cute.math.max(x1, Float32(-127.0), ftz=True)
    # Adding 1.5 * 2^23 with round-down leaves floor(x) in the low mantissa bits.
    bias = Float32(1.5 * 2.0**23)
    t = cute.arch.add_packed_f32x2((x0, x1), (bias, bias), rnd="rm", ftz=False)
    n = cute.arch.add_packed_f32x2(t, (-bias, -bias), rnd="rn", ftz=False)
    f = cute.arch.fma_packed_f32x2(
        n, (Float32(-1.0), Float32(-1.0)), (x0, x1), rnd="rn", ftz=False
    )
    # Degree-3 fit of 2^f on [0, 1).
    c1 = Float32(0.695146143436431884765625)
    c2 = Float32(0.227564394474029541015625)
    c3 = Float32(0.077119089663028717041015625)
    p = cute.arch.fma_packed_f32x2(f, (c3, c3), (c2, c2), rnd="rn", ftz=False)
    p = cute.arch.fma_packed_f32x2(p, f, (c1, c1), rnd="rn", ftz=False)
    p = cute.arch.fma_packed_f32x2(
        p, f, (Float32(1.0), Float32(1.0)), rnd="rn", ftz=False
    )
    # The bias has zero low bits, so bits(t) << 23 is floor(x) << 23.
    return (
        _f32_from_bits(_f32_bits(p[0]) + (_f32_bits(t[0]) << 23)),
        _f32_from_bits(_f32_bits(p[1]) + (_f32_bits(t[1]) << 23)),
    )


@cute.jit
def _f32_bits(x: Float32) -> Int32:
    return cutlass.Vector.from_elements((x,), Float32).bitcast(Int32)[0]


@cute.jit
def _f32_from_bits(x: Int32) -> Float32:
    return cutlass.Vector.from_elements((x,), Int32).bitcast(Float32)[0]


@cute.jit
def _exp2_fma_packed(x0: Float32, x1: Float32) -> tuple[Float32, Float32]:
    """exp2 of two fp32 values on the FMA pipe, 2^x = 2^floor(x) * poly(x - floor(x)).
    Same routine as ex2_emulation_f32x2_value in the SM110 GQA decode kernel."""
    # Below -127 the result flushes to zero in e4m3 anyway.
    x0 = cute.math.max(x0, Float32(-127.0), ftz=True)
    x1 = cute.math.max(x1, Float32(-127.0), ftz=True)
    # Adding 1.5 * 2^23 with round-down leaves floor(x) in the low mantissa bits.
    bias = Float32(1.5 * 2.0**23)
    t = cute.arch.add_packed_f32x2((x0, x1), (bias, bias), rnd="rm", ftz=False)
    n = cute.arch.add_packed_f32x2(t, (-bias, -bias), rnd="rn", ftz=False)
    f = cute.arch.fma_packed_f32x2(
        n, (Float32(-1.0), Float32(-1.0)), (x0, x1), rnd="rn", ftz=False
    )
    # Degree-3 fit of 2^f on [0, 1).
    c1 = Float32(0.695146143436431884765625)
    c2 = Float32(0.227564394474029541015625)
    c3 = Float32(0.077119089663028717041015625)
    p = cute.arch.fma_packed_f32x2(f, (c3, c3), (c2, c2), rnd="rn", ftz=False)
    p = cute.arch.fma_packed_f32x2(p, f, (c1, c1), rnd="rn", ftz=False)
    p = cute.arch.fma_packed_f32x2(
        p, f, (Float32(1.0), Float32(1.0)), rnd="rn", ftz=False
    )
    # The bias has zero low bits, so bits(t) << 23 is floor(x) << 23.
    return (
        _f32_from_bits(_f32_bits(p[0]) + (_f32_bits(t[0]) << 23)),
        _f32_from_bits(_f32_bits(p[1]) + (_f32_bits(t[1]) << 23)),
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
    logical_head_dim_qk: int = 128
    head_dim_per_stage_o: int = 128
    qk_mma_tiler: tuple[int, int, int] = (128, 128, 64)
    pv_mma_tiler: tuple[int, int, int] = (128, 64, 128)
    epi_tile: tuple[int, int] = (128, 64)
    # Number of interleaved Q/KV/O instances per CTA: two selects the paired
    # schedule, while one selects the single-instance schedule used for D>128.
    num_qkv_instances: int = 2

    # Pipeline stages
    q_stage: int = 2
    kv_stage: int = 3
    # Hand the S0/S1 pacing token back before the exp2/P work on the fp8 P-in-SMEM path
    # (as the TMEM-P fp8 cadence does), so the peer softmax group starts its row max
    # while this group computes P.
    fp8_psmem_early_token: bool = False
    # One TMA pipeline has one expected-transaction byte count per stage, so K
    # and V share a ring of kv_stage stages only while their dtype widths
    # match. Mixed widths set split_kv_pipelines and size one ring per side.
    split_kv_pipelines: bool = False
    kv_stage_k: int = 3
    kv_stage_v: int = 3
    mma_softmax_stage: int = 1
    # Stage fp8 P in SMEM so softmax releases the S stage right after loading it.
    p_in_smem: bool = False
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
    # V-specific TMA copy granularity. V may use a different dtype than K,
    # e.g. QK-BF16/PV-FP8, but these fields are otherwise ignored.
    tma_copy_v_iters: int = 1
    tma_copy_v_granu_inner: int = 128
    tma_copy_v_stage_iters: int = 0
    tma_copy_v_granu_elems: int = 0
    tma_copy_v_bytes: int = 0
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
    # Keep the running row max while a tile raises it by at most this many log2 units.
    corr_skip_threshold_log2: float = 0.0
    two_cta_umma: bool = False
    # exp2 pairs per 16-pair softmax chunk computed on the FMA pipe.
    exp2_fma_pairs: int = 0
    # VC-Attention-QK16. A nonzero ``vc_k_block_size`` (the 128-token K/V tile)
    # stores E4M3 V as per-tile residuals whose bf16 tile means one K=16 UMMA
    # step per tile restores into O.
    vc_k_block_size: int = 0
    # VC-Attention-QK16 V repair tiles after the whole K/V tiles. Non-zero
    # replaces the tile-mean restoration.
    vc_repair_tiles: int = 0
    # Strides of the per-(batch, head, channel) VC output scale table.
    vc_num_q_heads: int = 0
    vc_head_dim_v: int = 128

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
    # Opt-in to `tcgen05.ld.red.max` (LDTM.STAT) for the non-masked
    # `compute_row_max` path: fuses the per-chunk max into the TMEM load.
    # Requires SM103+/SM110+ with tcgen05.ld.red support; the primitive is
    # emitted with the 32dp x 32bit x 32rep shape only.
    uses_ldtm_stat: bool = False
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
    # The caller guarantees that request-invalid rows in every active final V
    # page contain zero. This lets consumers omit the defensive post-TMA clear.
    paged_v_tail_is_zero: bool = False
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
    def vc_attention(self) -> bool:
        return self.vc_k_block_size != 0

    @property
    def vc_restores_means(self) -> bool:
        """Whether VC-Attention-QK16 restores the V tile means rather than using V repair rows."""
        return self.vc_attention and self.vc_repair_tiles == 0

    def validate_vc_profile(self) -> None:
        """Validate the VC-Attention-QK16 recipe against the configured kernel."""
        if not self.vc_attention:
            if self.vc_num_q_heads != 0:
                raise ValueError("vc_num_q_heads requires vc_k_block_size")
            return
        if self.vc_k_block_size != self.qk_mma_tiler[1]:
            raise ValueError(
                f"vc_k_block_size must equal the K/V tile ({self.qk_mma_tiler[1]}), "
                f"got {self.vc_k_block_size}"
            )
        if self.vc_num_q_heads < 1:
            raise ValueError("VC-Attention-QK16 requires vc_num_q_heads")
        if (
            self.is_causal
            or self.has_variable_window
            or self.head_paired
            or self.use_paged_kv
            or self.h_r != 1
        ):
            raise ValueError(
                "VC-Attention-QK16 requires the dense contiguous query-paired context "
                "kernel with equal Q and K/V head counts"
            )
        if self.vc_head_dim_v != self.pv_mma_tiler[1]:
            raise ValueError(
                f"vc_head_dim_v must equal the V head dim ({self.pv_mma_tiler[1]}), "
                f"got {self.vc_head_dim_v}"
            )
        # The packed mean operand and the row-sum operand are laid out for 128
        # channels and 128 query rows.
        if self.logical_head_dim_qk != 128 or self.vc_head_dim_v != 128:
            raise ValueError("VC-Attention-QK16 requires head_dim 128")
        if self.q_dtype.width != 16 or self.v_dtype.width != 8:
            raise ValueError("VC-Attention-QK16 requires 16-bit Q/K and E4M3 V")
        if not self.p_in_smem:
            raise ValueError("VC-Attention-QK16 requires P staged in SMEM")

    @property
    def vc_mean_operand_bytes(self) -> int:
        """Bytes of one bf16 [D x 16] K-major mean operand (B of one mean UMMA step)."""
        return self.pv_mma_tiler[1] * VC_MEAN_MMA_K * 2

    @property
    def vc_mean_tile_bytes(self) -> int:
        """Bytes of the mean operands of one tile group (one ring stage)."""
        return VC_MEAN_OPERANDS * self.vc_mean_operand_bytes

    @property
    def vc_mean_stages(self) -> int:
        """Tile-mean ring depth: one operand per staged V tile."""
        return self.kv_stage_v if self.split_kv_pipelines else self.kv_stage_k

    @property
    def vc_rowsum_operand_bytes(self) -> int:
        """Bytes of one bf16 [128 x 16] K-major row-sum operand (A of one mean UMMA step)."""
        return self.qk_mma_tiler[0] * VC_MEAN_MMA_K * 2

    @property
    def vc_rowsum_tile_bytes(self) -> int:
        """Bytes of the row-sum operands of one tile group."""
        return VC_MEAN_OPERANDS * self.vc_rowsum_operand_bytes

    @property
    def smem_q_head_dim(self) -> int:
        """Q storage rounded to one 128-byte TMA fragment."""
        if self.logical_head_dim_qk == 192:
            fragment_elements = 1024 // self.q_dtype.width
            return (
                (self.logical_head_dim_qk + fragment_elements - 1)
                // fragment_elements
                * fragment_elements
            )
        return self.qk_mma_tiler[2]

    @property
    def single_qkv_instance(self) -> bool:
        """Return whether one work tile carries a single Q/KV/O instance."""
        return self.num_qkv_instances == 1

    @property
    def cta_group_size(self) -> int:
        """CTAs cooperating on one UMMA: 2 in the two-CTA form, else 1."""
        return 2 if self.two_cta_umma else 1

    @property
    def kv_tile_rows_per_cta(self) -> int:
        """K rows one CTA stages per K/V tile."""
        return self.kv_tile_n // self.cta_group_size

    @property
    def pv_n_per_cta(self) -> int:
        """V head-dim columns one CTA stages per tile."""
        return self.pv_mma_tiler[1] // self.cta_group_size

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
    def smem_p_bytes(self) -> int:
        """Bytes of one query group's SMEM P tile: q rows by k keys of the V dtype."""
        return self.qk_mma_tiler[0] * self.qk_mma_tiler[1] * self.v_dtype.width // 8

    @property
    def pv_half_overlap(self) -> bool:
        """Publish P in two 64-key halves so PV can start on the first half.
        Not used when P is staged in SMEM."""
        return (
            not self.single_qkv_instance
            and self.enable_early_tile_sum
            and not self.is_causal
            and not self.has_varlen
            and not self.has_tmem_p_pipeline
            and not self.p_in_smem
            and self.v_dtype is not None
            and self.v_dtype.width in (8, 16)
            and not self.uses_d128_fp8_softmax_cadence
            and not self.uses_d256_fp8_softmax_cadence
            # The head-dim-staged PV path issues every K slice per call.
            and not self.stage_kv_by_head_dim
            # Softmax halves the QK N tile and PV halves its K slices, so the
            # two extents must match and split evenly.
            and self.qk_mma_tiler[1] == self.pv_mma_tiler[2]
            and (self.pv_mma_tiler[2] // (16 if self.v_dtype.width == 16 else 32)) % 2
            == 0
        )

    @property
    def uses_d128_fp8_softmax_cadence(self) -> bool:
        """Paired D128 fp8 with P in TMEM retires softmax interleaved. Only the
        shapes that do not stage P in SMEM take this path."""
        return (
            not self.single_qkv_instance
            and not self.p_in_smem
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
    def needs_paged_v_tail_clear(self) -> bool:
        """Whether paged V tiles can contain request-invalid rows.

        A caller-owned zero-tail contract makes the clear redundant. Otherwise,
        only exact-full causal grids omit it. Requiring complete Q work tiles
        also excludes a padded final query-paired domain.
        """
        if not self.use_paged_kv or self.paged_v_tail_is_zero:
            return False
        q_work_tile_m = self.q_tile_m * self.work_tile_q_seq_tiles
        return not (
            self.has_uniform_varlen
            and self.is_causal
            and not self.has_q_offset
            and self.uniform_seq_len_k % self.kv_tile_n == 0
            and self.uniform_seq_len_q % q_work_tile_m == 0
        )

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
        has no cross-head K/V reuse and retains its sequence-local order. So
        does the two-CTA form under GQA, whose clusters must pair adjacent Q
        tiles of one head to share one K/V head.
        """
        return (
            (self.is_causal or self.h_r > 1)
            and not self.two_cta_umma
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
        """Return whether work tiles use head_batch_seq coordinates. Two-CTA
        stays seq-first so the grid padding and the cluster axis agree."""
        return self.uses_paired_fp8_head_batch_seq_tile_order or (
            self.is_causal and self.balance_causal_workload and not self.two_cta_umma
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
            # E4M3 saturates at 448. Exact correction keeps P <= 1, so P scales
            # by 448. Lazy correction lets the softmax row max lag the true one
            # by up to corr_skip_threshold_log2, so P <= 2^threshold. Keep a
            # power-of-two scale there. The largest with P * scale <= 448 is
            # 2^floor(log2(448) - threshold), which is 1 at threshold 8.
            if self.corr_skip_threshold_log2 == 0.0:
                return 448.0
            return 2.0 ** max(
                0.0, math.floor(math.log2(448.0) - self.corr_skip_threshold_log2)
            )
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

    Paces peer iterations. The paired FP8 main loop hands the token back
    before computing P, allowing overlap without letting one peer run ahead.
    Each peer's SP pipeline separately protects its P stores and MMA reads.
    Other paths retain serialized P computation.

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
# TmemPPrefixReadyResource -- leading keys of P stored
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemPPrefixReadyResource(MemoryResource):
    """Barrier: Softmax signals after storing the leading half of a P tile.

    Lets the MMA warp issue the PV MMAs for the first 64 keys while softmax
    is still computing the rest. There is no matching barrier for the rest of
    P; the existing SP stage guards the full tile and the S buffer reuse.

    Single resource instance shared across tasks:
      - Softmax's dst_resource (ProducerAcquire/Commit)
      - MMA's src_resource (ConsumerWait/Release)
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
                if cutlass.const_expr(self.cfg.has_uniform_varlen):
                    seqlen_k = Int32(self.cfg.uniform_seq_len_k)
                else:
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
            # K-loop extent from these live Q/K lengths.
            window_q_offset = seqlen_k - seqlen_q
            if cutlass.const_expr(
                self.cfg.use_paged_kv and not self.cfg.stages_page_offsets_in_smem
            ):
                if cutlass.const_expr(self.cfg.has_uniform_varlen):
                    kv_request_begin = batch_coord * Int32(self.block_table_row_stride)
                    kv_page_idx_ub = Int32(self.cfg.max_num_pages_per_seq_kv - 1)
                else:
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

        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            cached_seqlen_kv = Int32(self.cfg.uniform_seq_len_k)
            kv_request_begin = batch_coord * Int32(self.block_table_row_stride)
            kv_page_idx_ub = Int32(self.cfg.max_num_pages_per_seq_kv - 1)
        else:
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


def _qk_inner_dim_size_bytes(cfg: FmhaConfig) -> int:
    """Return the byte width of one Q/K tile inner dimension."""
    return cfg.qk_mma_tiler[2] * cfg.q_dtype.width // 8


def _pv_inner_dim_size_bytes(cfg: FmhaConfig) -> int:
    """Return the byte width of one P/V tile inner dimension."""
    return cfg.pv_n_per_cta * cfg.v_dtype.width // 8


def _o_inner_dim_size_bytes(cfg: FmhaConfig) -> int:
    """Return the byte width of one O tile inner dimension."""
    return cfg.qk_mma_tiler[2] * cfg.o_dtype.width // 8


def _smem_layout_for_inner_bytes(inner_dim_size: int) -> int:
    """Return the tcgen05 descriptor swizzle selector for an SMEM row of this many bytes."""
    if inner_dim_size % 128 == 0:
        return 2
    if inner_dim_size == 64:
        return 4
    if inner_dim_size == 32:
        return 6
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")


def _qk_smem_layout(cfg: FmhaConfig) -> int:
    """Return the tcgen05 descriptor layout selector for Q/K SMEM tiles."""
    return _smem_layout_for_inner_bytes(_qk_inner_dim_size_bytes(cfg))


def _pv_smem_layout(cfg: FmhaConfig) -> int:
    """Return the tcgen05 descriptor layout selector for V SMEM tiles."""
    return _smem_layout_for_inner_bytes(_pv_inner_dim_size_bytes(cfg))


def _qk_smem_desc_offsets(cfg: FmhaConfig) -> SmemDescOffsets:
    """Return Q/K descriptor leading and stride byte offsets."""
    leading_byte_offset = 0 if cfg.head_paired else 16
    stride_byte_offset = cfg.tma_copy_q_granu_inner * cfg.q_dtype.width
    return leading_byte_offset, stride_byte_offset


def _pv_smem_desc_offsets(cfg: FmhaConfig) -> SmemDescOffsets:
    """Return V descriptor leading and stride byte offsets for PV MMA."""
    if cfg.two_cta_umma:
        # One fragment of pv_n_per_cta columns per CTA.
        return 0, cfg.tma_copy_v_granu_inner * cfg.v_dtype.width
    leading_byte_offset = 0
    if cfg.tma_copy_v_iters != 1:
        tma_copy_v_iters = (
            cfg.tma_copy_v_stage_iters
            if cfg.stage_kv_by_head_dim
            else cfg.tma_copy_v_iters
        )
        leading_byte_offset = cfg.tma_copy_v_bytes // tma_copy_v_iters
    stride_byte_offset = cfg.pv_mma_tiler[1] * cfg.v_dtype.width // cfg.tma_copy_v_iters
    return leading_byte_offset, stride_byte_offset


def _pv_smem_swizzle(cfg: FmhaConfig) -> cutlass.Swizzle:
    """Return the physical TMA swizzle used by V SMEM fragments."""
    inner_dim_size = _pv_inner_dim_size_bytes(cfg)
    if inner_dim_size % 128 == 0:
        return cutlass.Swizzle(3, 4, 3)
    if inner_dim_size == 64:
        return cutlass.Swizzle(2, 4, 3)
    if inner_dim_size == 32:
        return cutlass.Swizzle(1, 4, 3)
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")


def _smem_o_swizzle(cfg: FmhaConfig) -> cutlass.Swizzle:
    """Return the shared-memory swizzle used when staging O for TMA store."""
    inner_dim_size = _o_inner_dim_size_bytes(cfg)
    if inner_dim_size % 128 == 0:
        return cutlass.Swizzle(3, 4, 3)
    if inner_dim_size == 64:
        return cutlass.Swizzle(2, 4, 3)
    if inner_dim_size == 32:
        return cutlass.Swizzle(1, 4, 3)
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")


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
                if cutlass.const_expr(self.cfg.two_cta_umma):
                    # Each CTA loads only its own 128 Q rows (multicast mask = own
                    # rank), but as a cta_group::2 copy, so the byte completion
                    # signals the mbarrier in the leader CTA's SMEM. The leader's
                    # MMA waits there for both CTAs' Q before the M=256 UMMA.
                    cta_rank = cute.arch.make_warp_uniform(
                        cute.arch.block_idx_in_cluster()
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sQ_curr.subview(i * self.cfg.tma_copy_q_granu_elems),
                        self.tma_q_desc,
                        q_coords,
                        cutlass.Array(
                            stage_info.barrier.data_ptr(), dtype=cutlass.Int64
                        ),
                        [],
                        multicast_mask=Int16(Int32(1) << cta_rank),
                        group=prims.CTAGroup.CTA_2,
                    )
                else:
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
            layout=_qk_smem_layout(self.cfg),
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
            labels: set[str] = set()
            binds_k = not (self.cfg.reuses_page_table_windows and self.page_table_is_v)
            binds_v = not (
                self.cfg.reuses_page_table_windows and not self.page_table_is_v
            )
            if downstream.holds_k and binds_k:
                labels |= {"k_load", "k_load_stage"}
            if downstream.holds_v and binds_v:
                labels |= {"v_load", "v_load_stage", "v_load_stage_cached"}
            return labels or None
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

    ``role="kv"`` shares one buffer and pipeline, which requires equal K and V
    element widths. Mixed QK/PV dtypes instead build one ``role="k"`` and one
    ``role="v"`` instance, each sized from its own dtype.
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
    role: Constexpr[str] = field(init=False, default="kv")
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)
    desc_k_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_k_stage_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
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
        role: str = "kv",
        **kwargs: Any,
    ) -> None:
        """Bind K/V TMA descriptors and reserve this role's SMEM staging."""
        if role not in ("kv", "k", "v"):
            raise ValueError(f"unsupported SmemKVResource role: {role}")
        if role == "kv" and cfg.k_dtype.width != cfg.v_dtype.width:
            raise ValueError(
                "a shared K/V buffer requires K and V to have the same element "
                f"width; got {cfg.k_dtype.width} and {cfg.v_dtype.width}"
            )
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.role = role
        self.tma_k_desc = tma_k_desc
        self.tma_v_desc = tma_v_desc
        self.page_offsets_kv = page_offsets_kv
        self.page_offsets_v = (
            page_offsets_v if page_offsets_v is not None else page_offsets_kv
        )
        self.block_tables = block_tables
        self.cfg = cfg
        # Every role stages pipeline_config.num_stages tiles; for "kv" that is
        # cfg.kv_stage, which is cfg.sK_shape[0].
        total_elements = pipeline_config.num_stages * cfg.sK_shape[1]
        size_bytes = total_elements * self._buffer_dtype(cfg, role).width // 8
        self._alloc = SmemAllocation(
            f"smem_{role}", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sK_array = _placeholder_smem_array(self._buffer_dtype(cfg, role))
        self.desc_k_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the current K tile.",
        )
        self.desc_k_stage_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor for the later K slice retained by both Q tiles.",
        )
        self.desc_v_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the current V tile.",
        )

    @staticmethod
    def _buffer_dtype(cfg: FmhaConfig, role: str) -> type:
        """Return the element type staged by a buffer with this role."""
        return cfg.v_dtype if role == "v" else cfg.k_dtype

    @property
    def holds_k(self) -> bool:
        """Return whether this buffer stages K tiles."""
        return self.role in ("kv", "k")

    @property
    def holds_v(self) -> bool:
        """Return whether this buffer stages V tiles."""
        return self.role in ("kv", "v")

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation required for staged K/V tiles."""
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Materialize K/V SMEM storage and descriptor dataflow slots."""
        smem_base = stage_info.context.smem_base
        total_elements = self.pipeline_config.num_stages * self.cfg.sK_shape[1]
        self.sK_array = cutlass.Array(
            smem_base.data_ptr() + self._alloc.offset,
            dtype=self._buffer_dtype(self.cfg, self.role),
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
        # V carries its own copy granularity because v_dtype may be narrower.
        # The two sets coincide when the widths match.
        if cutlass.const_expr(is_v):
            d_granu_inner = self.cfg.tma_copy_v_granu_inner
            d_iter_elems = self.cfg.tma_copy_v_granu_elems
            stage_iters = self.cfg.tma_copy_v_stage_iters
        else:
            d_granu_inner = self.cfg.tma_copy_kv_granu_inner
            d_iter_elems = self.cfg.tma_copy_kv_granu_elems
            stage_iters = self.cfg.tma_copy_kv_stage_iters

        if cutlass.const_expr(self.cfg.use_paged_kv):
            # Paged-KV path: read pre-staged page IDs and issue one TMA per
            # (page fragment, d fragment). The descriptor is shaped
            # (d_inner, num_tokens_per_page, h_kv, total_pages); coords are
            # (d_off, 0, kv_head_coord, page_id). SMEM layout per stage matches
            # the contiguous path: two d-halves (d_iter_elems each) with page
            # fragments concatenated along the seq axis inside each d-half.
            tile_idx = kv_tile_start + stage_info.loop_offset + tile_offset
            pages_per_tile = self.cfg.kv_tile_n // self.cfg.num_tokens_per_page
            page_d_elems = self.cfg.num_tokens_per_page * d_granu_inner
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
                    for i in cutlass.range_constexpr(stage_iters):
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
            seq_coord_kv = cuseqlen_k + seq_offset
            d_base = Int32(0)
            if cutlass.const_expr(self.cfg.two_cta_umma):
                # This CTA stages its half of the K rows or V columns.
                cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                if cutlass.const_expr(is_v):
                    d_base = cta_rank * Int32(self.cfg.pv_n_per_cta)
                else:
                    seq_coord_kv = seq_coord_kv + cta_rank * Int32(
                        self.cfg.kv_tile_rows_per_cta
                    )
            for i in cutlass.range_constexpr(stage_iters):
                d_offset = (
                    head_dim_stage_idx * self.cfg.head_dim_per_stage_kv
                    + i * d_granu_inner
                )
                kv_coords = (
                    d_offset + d_base,
                    kv_head_coord,
                    seq_coord_kv,
                    batch_coord,
                )
                if cutlass.const_expr(self.cfg.has_varlen):
                    kv_coords = (d_offset + d_base, kv_head_coord, seq_coord_kv)
                if cutlass.const_expr(self.cfg.two_cta_umma):
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sK_curr.subview(i * d_iter_elems),
                        tma_desc,
                        kv_coords,
                        cutlass.Array(
                            stage_info.barrier.data_ptr(), dtype=cutlass.Int64
                        ),
                        [],
                        multicast_mask=Int16(Int32(1) << cta_rank),
                        group=prims.CTAGroup.CTA_2,
                    )
                else:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sK_curr.subview(i * d_iter_elems),
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
            layout=_qk_smem_layout(self.cfg),
        )
        return desc_k_base

    @consumer_work(returns=desc_k_stage_base)
    @cute.jit
    def k_stage_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Keep the later K descriptor live alongside the first K and V."""
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)
        leading_byte_offset, stride_byte_offset = _qk_smem_desc_offsets(self.cfg)
        return prims.Tcgen05SmemDesc.build(
            sK_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=_qk_smem_layout(self.cfg),
        )

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
            d_granu_inner = self.cfg.tma_copy_v_granu_inner
            chunks_per_d_iter = d_granu_inner // 16
            chunks_per_v_row = self.cfg.tma_copy_v_stage_iters * chunks_per_d_iter
            page_d_elems = self.cfg.num_tokens_per_page * d_granu_inner
            d_iter_elems = self.cfg.tma_copy_v_granu_elems
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
                    swizzle=_pv_smem_swizzle(self.cfg),
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
            layout=_pv_smem_layout(self.cfg),
        )

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build a V SMEM descriptor that needs no paged-tail clear."""
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
# SmemPResource -- fp8 P tile staged in SMEM per query group
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class SmemPResource(MemoryResource):
    """One K-major SW128 fp8 P tile per query group in SMEM: softmax produces,
    the PV UMMA consumes, so softmax can release the S stage early."""

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    _alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        group_idx: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        # VC-Attention-QK16 appends the group's bf16 [128 x 16] row-sum operand
        # behind the P tile.
        extra_bytes = cfg.vc_rowsum_tile_bytes if cfg.vc_restores_means else 0
        self._alloc = SmemAllocation(
            f"smem_p{group_idx}",
            cfg.smem_p_bytes + extra_bytes,
            alignment=cfg.buffer_align_bytes,
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        return [self._alloc]

    @property
    def rowsum_tile_offset(self) -> int:
        """Byte offset of the VC-Attention-QK16 row-sum operand inside this allocation."""
        return self.cfg.smem_p_bytes

    @property
    def row_bytes(self) -> int:
        return self.cfg.qk_mma_tiler[1] * self.cfg.v_dtype.width // 8

    def descriptor_offsets(self) -> SmemDescOffsets:
        """LBO and SBO of the K-major swizzled tile: eight rows per swizzle atom."""
        leading_byte_offset = 16
        return leading_byte_offset, 8 * self.row_bytes

    def descriptor_layout(self) -> int:
        return _smem_layout_for_inner_bytes(self.row_bytes)


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
    p_lo: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    q_offset: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seqlen_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    # p_in_smem: this group's SMEM P tile.
    smem_p: Optional[SmemPResource] = field(init=False, default=None)
    variable_window_start: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    variable_window_end: Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    # VC-Attention-QK16: the row's softmax scale in log2 units (per work tile) and
    # the previous tile's fp32 row sum awaiting its deferred mean step.
    vc_row_scale: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_prev_tile_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_kept_row_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    # Row sums of the current mean group, kept in the running max's units.
    vc_pend0: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend1: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend2: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend3: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend4: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend5: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend6: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend7: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend8: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend9: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend10: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend11: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend12: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend13: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend14: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vc_pend15: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

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
        smem_p: Optional[SmemPResource] = None,
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
        self.smem_p = smem_p
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
        self.p_lo = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="FP32 sum of the first half of the current probability tile.",
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
        self.vc_row_scale = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: the softmax scale in log2 units.",
        )
        self.vc_prev_tile_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: previous tile's row sum, stored one tile late.",
        )
        self.vc_kept_row_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16 V repair, row sum over the original tokens.",
        )
        self.vc_pend0 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 0 of the current mean group.",
        )
        self.vc_pend1 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 1 of the current mean group.",
        )
        self.vc_pend2 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 2 of the current mean group.",
        )
        self.vc_pend3 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 3 of the current mean group.",
        )
        self.vc_pend4 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 4 of the current mean group.",
        )
        self.vc_pend5 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 5 of the current mean group.",
        )
        self.vc_pend6 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 6 of the current mean group.",
        )
        self.vc_pend7 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 7 of the current mean group.",
        )
        self.vc_pend8 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 8 of the current mean group.",
        )
        self.vc_pend9 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 9 of the current mean group.",
        )
        self.vc_pend10 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 10 of the current mean group.",
        )
        self.vc_pend11 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 11 of the current mean group.",
        )
        self.vc_pend12 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 12 of the current mean group.",
        )
        self.vc_pend13 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 13 of the current mean group.",
        )
        self.vc_pend14 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 14 of the current mean group.",
        )
        self.vc_pend15 = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="VC-Attention-QK16: row sum of tile 15 of the current mean group.",
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
            and not self.cfg.has_variable_window
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
    def qk_mma_stage(
        self,
        stage_info: StageInfo,
        *,
        desc_q_base: prims.Tcgen05SmemDesc,
        desc_k_stage: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[FmhaStage],
        head_dim_stage_idx: cutlass.Constexpr[int],
    ) -> None:
        """Accumulate a later K slice without replacing the first slice's binding."""
        self._qk_mma_impl(
            stage_info,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_stage,
            section=section,
            head_dim_stage_idx=head_dim_stage_idx,
        )

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
        """Issue one QK slice, clipping MMA phases to the logical head dimension."""
        self._qk_mma_impl(
            stage_info,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            section=section,
            head_dim_stage_idx=head_dim_stage_idx,
            is_tail=is_tail,
        )

    @cute.jit
    def _qk_mma_impl(
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
                m_dim=self.cfg.qk_mma_tiler[0] * self.cfg.cta_group_size,
            )

            k_dim_per_mma = 16
            if cutlass.const_expr(self.cfg.q_dtype.width != 16):
                k_dim_per_mma = 32
            inc_bytes_qk = k_dim_per_mma * self.cfg.q_dtype.width // 8

            num_kphases_per_tma = self.cfg.tma_copy_q_granu_inner // k_dim_per_mma
            # Byte stride between TMA fragments of the Q and K tiles. They differ only
            # in the two-CTA form, where a K stage holds half the rows.
            chunk_bytes_q = inc_bytes_qk * num_kphases_per_tma
            chunk_bytes_k = chunk_bytes_q
            if cutlass.const_expr(self.cfg.tma_copy_qkv_iters != 1):
                chunk_bytes_q = (
                    self.cfg.tma_copy_q_granu_elems * self.cfg.q_dtype.width // 8
                )
                chunk_bytes_k = (
                    self.cfg.tma_copy_kv_bytes // self.cfg.tma_copy_kv_stage_iters
                )
            cta_group = (
                prims.CTAGroup.CTA_2
                if cutlass.const_expr(self.cfg.two_cta_umma)
                else prims.CTAGroup.CTA_1
            )
            issue_mma = cutlass.Boolean(True)
            if cutlass.const_expr(self.cfg.two_cta_umma):
                issue_mma = (
                    cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
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
                q_tma_iter_offset = chunk_bytes_q * q_tma_iter
                k_tma_iter_offset = chunk_bytes_k * tma_iter
                # The final 128-wide K stage may contain only 64 logical
                # elements (non-absorbed MLA). Do not issue MMAs on padding.
                valid_kphases = min(
                    num_kphases_per_tma,
                    max(
                        0,
                        (
                            self.cfg.logical_head_dim_qk
                            - q_tma_iter * self.cfg.tma_copy_q_granu_inner
                        )
                        // k_dim_per_mma,
                    ),
                )
                for k_idx in cutlass.range_constexpr(valid_kphases):
                    local_increment = inc_bytes_qk * k_idx
                    dq = desc_q_base_ + ((local_increment + q_tma_iter_offset) >> 4)
                    dk = desc_k_base_ + ((local_increment + k_tma_iter_offset) >> 4)
                    if issue_mma:
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                cta_group,
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
        if cutlass.const_expr(self.cfg.vc_attention):
            self._zero_vc_rowsum_row(stage_info)

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

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            vc_row_scale,
            vc_prev_tile_sum,
            vc_pend0,
            vc_pend1,
            vc_pend2,
            vc_pend3,
            vc_pend4,
            vc_pend5,
            vc_pend6,
            vc_pend7,
            vc_pend8,
            vc_pend9,
            vc_pend10,
            vc_pend11,
            vc_pend12,
            vc_pend13,
            vc_pend14,
            vc_pend15,
            vc_kept_row_sum,
        ),
    )
    @cute.jit
    def vc_init_row_scale(self, stage_info: StageInfo) -> tuple[Float32, ...]:
        """VC-Attention-QK16 work-tile setup. Returns the softmax scale in log2 units,
        a zero previous-tile row sum, zero group row sums and a zero kept row sum."""
        _ = stage_info
        return (self.scale_softmax_log2[0], Float32(0.0)) + (Float32(0.0),) * 17

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def vc_compute_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        vc_row_scale: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """VC-Attention-QK16 K-loop row max in log2 units."""
        if cutlass.const_expr(self.cfg.uses_ldtm_stat):
            return self._load_s_chunks_and_reduce_row_max(
                stage_info, row_max, tile_scale=vc_row_scale
            )
        s_data = self._load_s_chunks(stage_info)
        return self._reduce_row_max(s_data, row_max, vc_row_scale)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def vc_fixed_dense_k_tail_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        vc_row_scale: SoftmaxScalar,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """VC-Attention-QK16 dense tail: mask the zero-filled lanes of the partial last
        K/V tile before the row max."""
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        tile_scale = vc_row_scale
        s_data = self._load_s_chunks(stage_info)
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
            s_data[chunk_idx] = cutlass.vector.where(mask, s_data[chunk_idx], neg_inf)
        return self._reduce_row_max(s_data, row_max, tile_scale)

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
        if cutlass.const_expr(self.cfg.has_uniform_varlen):
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
        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            return Int32(self.cfg.uniform_seq_len_k)
        if cutlass.const_expr(self.cfg.use_paged_kv):
            batch_coord = self._varlen_batch_coord(stage_info)
            return Int32(self.seq_lens_kv[batch_coord])
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
        tile_scale: SoftmaxScalar | None = None,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Reduce per-chunk maximums into row_max, stash s_data.

        VC-Attention-QK16 passes ``tile_scale``: the raw tile maximum is scaled into
        log2 units (the scale is positive, so max commutes with it) before the
        running maximum merge, and the scale is stashed for ``exp2_p``.
        """
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
            if cutlass.const_expr(self.cfg.vc_attention):
                tile_row_max = tile_row_max * tile_scale
                _tmem_sp_tile_scale[id(self)] = tile_scale
            row_max = cute.math.max(row_max, tile_row_max)
        _tmem_sp_sdata[id(self)] = s_data
        row_max_safe = row_max
        if row_max == -Float32.inf:
            row_max_safe = Float32(0.0)
        return old_row_max, row_max_safe

    @cute.jit
    def _load_s_chunks_and_reduce_row_max(
        self,
        stage_info: StageInfo,
        row_max: SoftmaxScalar,
        *,
        causal_loop_mask: cutlass.Constexpr[bool] = False,
        q_offset: Int32 = 0,
        tile_scale: SoftmaxScalar | None = None,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """LDTM.STAT: fuse S load and per-chunk row_max via tcgen05.ld.red.max.

        Instead of loading each S chunk with ``tcgen05.ld`` and then folding
        a per-lane max in registers, use ``tcgen05.ld.red.max`` so the
        reduction happens as part of the TMEM load. For causal loop masking,
        only tiles crossing the runtime right bound need a software reduction
        of masked scores; wholly visible tiles retain the hardware maximum.

        VC-Attention-QK16 keeps the row max in log2 units. ``tile_scale`` is
        the softmax scale, applied to the hardware tile maximum before the
        merge and carried to ``exp2_p`` like the loaded S chunks.
        """
        tmem_s_addr = self.tmem_s_addr_cached + self._stage_col_offset(stage_info)
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        old_row_max = row_max
        s_data: SoftmaxChunks = [None] * num_chunks
        chunk_maxima: tuple[Any, ...] = ()
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            if cutlass.const_expr(hasattr(prims, "tcgen05_ld_red")):
                loaded_words, red_word = prims.tcgen05_ld_red(
                    prims.Tcgen05LdStShape.SHAPE_32X32B,
                    prims.make_tmem_ptr(
                        tmem_s_addr + chunk_idx * tmem_x, self.cfg.qk_acc_dtype
                    ),
                    prims.ReductionKind.MAX,
                    num=tmem_x,
                )
                s_data[chunk_idx] = cutlass.Vector.from_elements(
                    tuple(
                        loaded_words[i].bitcast(self.cfg.qk_acc_dtype)
                        for i in range(tmem_x)
                    ),
                    dtype=self.cfg.qk_acc_dtype,
                )
                chunk_max = self.cfg.qk_acc_dtype(
                    arith_dialect.bitcast(
                        self.cfg.qk_acc_dtype.mlir_type, red_word.ir_value()
                    )
                )
            else:
                # PTX ISA 8.8 supports LDTM.STAT on SM103 before the DSL 4.8
                # convenience wrapper is available. The geometry is fixed at
                # 32 rows x 32 FP32 values for each context load fragment.
                assert tmem_x == 32
                loaded = load_tmem_32x32b_max(tmem_s_addr + chunk_idx * tmem_x)
                s_data[chunk_idx] = cutlass.Vector.from_elements(
                    loaded[:32], dtype=self.cfg.qk_acc_dtype
                )
                chunk_max = loaded[32]
            chunk_maxima += (chunk_max,)
        cute.arch.fence_view_async_tmem_load()
        # Reduction results are asynchronous, just like the loaded scores.
        tile_max = cutlass.Vector.from_elements(
            chunk_maxima, self.cfg.qk_acc_dtype
        ).reduce("max")
        if cutlass.const_expr(self.cfg.vc_attention):
            tile_max = tile_max * tile_scale
            _tmem_sp_tile_scale[id(self)] = tile_scale
        row_max = cute.math.max(row_max, tile_max)
        if cutlass.const_expr(causal_loop_mask):
            seq_coord, _, _ = _resolve_work_tile_coords(
                self.cfg, stage_info.work_tile.tile_idx
            )
            q_min = (
                q_offset
                + seq_coord * self.cfg.cta_tiler[0]
                + self.q_half * self.cfg.q_tile_m
            )
            kv_base = stage_info.loop_offset * self.cfg.kv_tile_n
            k_max = kv_base + self.cfg.qk_mma_tiler[1] - Int32(1)
            if q_min < k_max:
                q_idx = (
                    q_min
                    + (cute.arch.warp_idx() % 4) * cute.arch.WARP_SIZE
                    + cute.arch.lane_idx()
                )
                # Discard the unmasked maxima when any key can be invisible.
                masked_row_max = old_row_max
                for chunk_idx in cutlass.range_constexpr(num_chunks):
                    num_valid = cute.math.min(
                        cute.math.max(
                            q_idx - kv_base - chunk_idx * tmem_x + Int32(1),
                            Int32(0),
                        ),
                        Int32(tmem_x),
                    )
                    mask = cutlass.vector.create_mask([tmem_x], [num_valid])
                    neg_inf = cutlass.vector.full_like(
                        s_data[chunk_idx], Float32(-Float32.inf)
                    )
                    s_data[chunk_idx] = cutlass.vector.where(
                        mask, s_data[chunk_idx], neg_inf
                    )
                    masked_row_max = cute.math.max(
                        masked_row_max, s_data[chunk_idx].reduce("max")
                    )
                row_max = masked_row_max
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
        chunk_lo: cutlass.Constexpr[int] = 0,
        chunk_hi: cutlass.Constexpr[int | None] = None,
        sum_in: Float32 | None = None,
    ) -> SoftmaxRowSumContribution:
        """Apply exp2 softmax P, fold the PV P scale, and store P to TMEM.

        ``chunk_lo``/``chunk_hi`` select the key chunks to process; the whole
        tile by default. A partial range fences its stores and returns
        ``sum_in`` plus its row sum, for the two-half publish.
        """
        tmem_p_addr = self.tmem_p_addr_cached + stage_col_offset
        tmem_shape = "32x32b"
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        if cutlass.const_expr(chunk_hi is None):
            chunk_hi = num_chunks
        partial = chunk_lo > 0 or chunk_hi < num_chunks
        p_packing_ratio = self.cfg.qk_acc_dtype.width // self.cfg.v_dtype.width
        scale = scale_softmax_log2
        if cutlass.const_expr(self.cfg.vc_attention):
            # The row-max pass kept row_max in log2 units of the scaled scores.
            scale = _tmem_sp_tile_scale.pop(id(self))
        if cutlass.const_expr(not partial and self.cfg.uses_d256_fp8_softmax_cadence):
            return self._exp2_p_store_d256_fp8_cadence(
                tmem_p_addr,
                row_max,
                scale,
            )
        if cutlass.const_expr(not partial and self.cfg.uses_d128_fp8_softmax_cadence):
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
        if cutlass.const_expr(self.cfg.vc_attention):
            # The paper's 2^8 P scaling is part of VC_EXPCAST_CODE_BIAS, so
            # p_scale_log2 is not added here.
            return self._exp2_p_store_expcast(scale, Float32(0.0) - row_max)
        minus_row_max_scale = (Float32(0.0) - row_max) * scale + p_scale_log2
        if cutlass.const_expr(chunk_hi < num_chunks):
            # A later range reads the same S data.
            s_data = _tmem_sp_sdata[id(self)]
        else:
            s_data = _tmem_sp_sdata.pop(id(self))
        if cutlass.const_expr(self.enable_early_tile_sum or partial):
            # Four packed float2 accumulators keep eight independent FADD2
            # chains, so no add waits on the MUFU result it consumes.
            local_sum_pair_0 = (Float32(0.0), Float32(0.0))
            local_sum_pair_1 = (Float32(0.0), Float32(0.0))
            local_sum_pair_2 = (Float32(0.0), Float32(0.0))
            local_sum_pair_3 = (Float32(0.0), Float32(0.0))
        for chunk_idx in cutlass.range_constexpr(chunk_lo, chunk_hi):
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
                if cutlass.const_expr(
                    elem_idx // 2 >= tmem_x // 2 - self.cfg.exp2_fma_pairs
                ):
                    p0, p1 = _exp2_fma_packed(fma_pair[0], fma_pair[1])
                else:
                    p0 = cute.math.exp2(fma_pair[0], fastmath=True)
                    p1 = cute.math.exp2(fma_pair[1], fastmath=True)
                if cutlass.const_expr(self.enable_early_tile_sum or partial):
                    pair_idx = chunk_idx * (tmem_x // 2) + elem_idx // 2
                    if cutlass.const_expr(pair_idx % 4 == 0):
                        local_sum_pair_0 = cute.arch.add_packed_f32x2(
                            local_sum_pair_0,
                            (p0, p1),
                            rnd="rn",
                            ftz=False,
                        )
                    elif cutlass.const_expr(pair_idx % 4 == 1):
                        local_sum_pair_1 = cute.arch.add_packed_f32x2(
                            local_sum_pair_1,
                            (p0, p1),
                            rnd="rn",
                            ftz=False,
                        )
                    elif cutlass.const_expr(pair_idx % 4 == 2):
                        local_sum_pair_2 = cute.arch.add_packed_f32x2(
                            local_sum_pair_2, (p0, p1), rnd="rn", ftz=False
                        )
                    else:
                        local_sum_pair_3 = cute.arch.add_packed_f32x2(
                            local_sum_pair_3, (p0, p1), rnd="rn", ftz=False
                        )
                p_vals += (p0, p1)
            s_data[chunk_idx] = cutlass.Vector.from_elements(
                p_vals, self.cfg.qk_acc_dtype
            )
        use_fused_d128_fp8x4_pack = (
            not self.cfg.single_qkv_instance
            and self.cfg.v_dtype == cutlass.Float8E4M3FN
        )
        # One word holds four fp8 P values, so a pair of chunks is tmem_x words
        # and a chunk range inside a pair is a word sub-range of its store.
        for pair_idx in cutlass.range_constexpr(
            chunk_lo // p_packing_ratio,
            (chunk_hi + p_packing_ratio - 1) // p_packing_ratio,
        ):
            word_lo = max(chunk_lo - pair_idx * p_packing_ratio, 0) * tmem_x // 4
            word_hi = (
                min(chunk_hi - pair_idx * p_packing_ratio, p_packing_ratio)
                * tmem_x
                // 4
            )
            if cutlass.const_expr(use_fused_d128_fp8x4_pack):
                # Match the handwritten D128 pack: merge both FP8x2
                # conversions in one side-effecting block so ptxas can retain
                # the 32-bit word without a PRMT between temporary vectors.
                packed_words: tuple[Any, ...] = ()
                for word_idx in cutlass.range_constexpr(word_lo, word_hi):
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
                if cutlass.const_expr(self.cfg.p_in_smem):
                    _tmem_sp_pwords[id(self)] = packed_words
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
            if cutlass.const_expr(not self.cfg.p_in_smem):
                prims.tcgen05_st(
                    tmem_shape,
                    prims.make_tmem_ptr(
                        tmem_p_addr + pair_idx * tmem_x + word_lo, cutlass.Int8
                    ),
                    store_fragment,
                )
        if cutlass.const_expr(self.enable_early_tile_sum or partial):
            local_sum_pair_0 = cute.arch.add_packed_f32x2(
                local_sum_pair_0, local_sum_pair_2, rnd="rn", ftz=False
            )
            local_sum_pair_1 = cute.arch.add_packed_f32x2(
                local_sum_pair_1, local_sum_pair_3, rnd="rn", ftz=False
            )
            local_sum_pair = cute.arch.add_packed_f32x2(
                local_sum_pair_0, local_sum_pair_1, rnd="rn", ftz=False
            )
            tile_sum = local_sum_pair[0] + local_sum_pair[1]
        if cutlass.const_expr(partial):
            # Make this range's P visible before the barrier arrive that follows.
            cute.arch.fence_view_async_tmem_store()
            return sum_in + tile_sum
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
    def _exp2_p_store_expcast(
        self, scale: SoftmaxScalar, minus_row_max_scale: SoftmaxScalar
    ) -> Float32:
        """VC-Attention-QK16 ExpCast-FP8 P with one packed FFMA per element.

        ``code = 8 * (s * scale - m) + VC_EXPCAST_CODE_BIAS``

        Packed P words go to ``store_p``. Returns the tile row sum of the
        quantized P.
        """
        assert self.cfg.p_in_smem and self.enable_early_tile_sum
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        # The exact per-tile row max keeps u <= 0, so codes stay at or below
        # the code bias.
        code_scale = scale * Float32(E4M3_CODES_PER_OCTAVE)
        code_bias = minus_row_max_scale * Float32(E4M3_CODES_PER_OCTAVE) + Float32(
            VC_EXPCAST_CODE_BIAS
        )
        s_data = _tmem_sp_sdata.pop(id(self))
        packed_words: tuple[Any, ...] = ()
        acc: list[Any] = [None] * 4
        for chunk_idx in cutlass.range_constexpr(num_chunks):
            for elem_idx in cutlass.range_constexpr(0, tmem_x, 4):
                quad_idx = (chunk_idx * tmem_x + elem_idx) // 4
                chain = 2 * (quad_idx % 2)
                c0, c1 = cute.arch.fma_packed_f32x2(
                    (s_data[chunk_idx][elem_idx], s_data[chunk_idx][elem_idx + 1]),
                    (code_scale, code_scale),
                    (code_bias, code_bias),
                )
                c2, c3 = cute.arch.fma_packed_f32x2(
                    (
                        s_data[chunk_idx][elem_idx + 2],
                        s_data[chunk_idx][elem_idx + 3],
                    ),
                    (code_scale, code_scale),
                    (code_bias, code_bias),
                )
                if cutlass.const_expr(quad_idx < 2):
                    word, acc_lo, acc_hi = _expcast_e4m3_quad_relu_init(c0, c1, c2, c3)
                else:
                    word, acc_lo, acc_hi = _expcast_e4m3_quad_relu(
                        c0, c1, c2, c3, acc[chain], acc[chain + 1]
                    )
                acc[chain] = acc_lo
                acc[chain + 1] = acc_hi
                packed_words += (word,)
        _tmem_sp_pwords[id(self)] = packed_words
        return _f16x2_sum4(acc[0], acc[1], acc[2], acc[3])

    @cute.jit
    def _store_p_row_smem(
        self, stage_info: StageInfo, packed_words: tuple[Any, ...]
    ) -> None:
        """Write this thread's 128-byte P row in the SW128 K-major layout the PV
        descriptor expects: 16-byte chunk c of row r lands at chunk c xor (r mod 8)."""
        assert self.smem_p is not None
        context = stage_info.context
        assert context is not None and context.smem_base is not None
        view = cutlass.Array(
            context.smem_base.data_ptr() + self.smem_p._alloc.offset,
            dtype=Int32,
            shape=(self.cfg.smem_p_bytes // 4,),
            addrspace=3,
        )
        # One P row per softmax thread, indexed like its TMEM lane.
        warp_id_in_sg = cute.arch.warp_idx() % len(self.cfg.softmax0_warp_ids)
        row = Int32(warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx())
        row_words = row * Int32(32)
        swz = row & Int32(7)
        for chunk_idx in cutlass.range_constexpr(len(packed_words) // 4):
            chunk_words = cutlass.Vector.from_elements(
                packed_words[chunk_idx * 4 : chunk_idx * 4 + 4], Int32
            )
            phys_chunk = Int32(chunk_idx) ^ swz
            view.subview(row_words + phys_chunk * Int32(4)).data_ptr().store(
                chunk_words, alignment=16
            )

    @cute.jit
    def _vc_rowsum_view(self, stage_info: StageInfo) -> cutlass.Array:
        """This group's bf16 row-sum operand: K-major 32-byte rows in 8-row core
        matrices (``k`` 0-7 in the first 128 bytes of a group, 8-15 in the next)."""
        assert self.smem_p is not None
        context = stage_info.context
        return cutlass.Array(
            context.smem_base.data_ptr()
            + self.smem_p._alloc.offset
            + self.smem_p.rowsum_tile_offset,
            dtype=cutlass.BFloat16,
            shape=(self.cfg.vc_rowsum_tile_bytes // 2,),
            addrspace=3,
        )

    @cute.jit
    def _vc_row_base(self) -> Int32:
        """Element offset of this thread's row inside the row-sum operand."""
        warp_id_in_sg = cute.arch.warp_idx() % len(self.cfg.softmax0_warp_ids)
        row = Int32(warp_id_in_sg * cute.arch.WARP_SIZE + cute.arch.lane_idx())
        return (row >> Int32(3)) * Int32(128) + (row & Int32(7)) * Int32(8)

    @cute.jit
    def _zero_vc_rowsum_row(self, stage_info: StageInfo) -> None:
        """Clear this thread's 32-byte row of every row-sum operand once."""
        view = self._vc_rowsum_view(stage_info)
        base = self._vc_row_base()
        zero = cutlass.BFloat16(0.0)
        zeros = cutlass.Vector.from_elements((zero,) * 8, cutlass.BFloat16)
        for operand in cutlass.range_constexpr(VC_MEAN_OPERANDS):
            row = base + Int32(operand * self.cfg.vc_rowsum_operand_bytes // 2)
            view.subview(row).data_ptr().store(zeros, alignment=16)
            view.subview(row + Int32(64)).data_ptr().store(zeros, alignment=16)

    @cute.jit
    def _store_vc_rowsum_group(self, stage_info: StageInfo, sums: tuple) -> None:
        """Write the group's row sums as bf16 hi/lo pairs: tile ``8o+i`` in K slots
        ``2i``, ``2i+1`` of operand ``o``, as one 16-byte store per core matrix
        (slots 0-7 in the first, 8-15 in the next).

        The split truncates and the low half carries the remainder, so each
        pair is exact to 2^-16 and runs on the integer pipe.
        """
        view = cutlass.Array(
            self._vc_rowsum_view(stage_info).data_ptr(),
            dtype=Int32,
            shape=(self.cfg.vc_rowsum_tile_bytes // 4,),
            addrspace=3,
        )
        base = self._vc_row_base() >> Int32(1)
        tiles_per_operand = VC_MEAN_MMA_K // 2
        for operand in cutlass.range_constexpr(VC_MEAN_OPERANDS):
            words = [
                _split_bf16_hi_lo_word(sums[operand * tiles_per_operand + i])
                for i in range(tiles_per_operand)
            ]
            row = base + Int32(operand * self.cfg.vc_rowsum_operand_bytes // 4)
            for half in cutlass.range_constexpr(2):
                vec = cutlass.Vector.from_elements(
                    tuple(words[half * 4 : half * 4 + 4]), Int32
                )
                view.subview(row + Int32(half * 32)).data_ptr().store(vec, alignment=16)

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
        use_two_sum_pairs = not self.cfg.single_qkv_instance
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
        if cutlass.const_expr(self.cfg.uses_ldtm_stat):
            return self._load_s_chunks_and_reduce_row_max(stage_info, row_max)
        s_data = self._load_s_chunks(stage_info)
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(returns=(old_row_max, row_max))
    @cute.jit
    def fixed_dense_k_tail_masked_row_max(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        section: cutlass.Constexpr[FmhaStage] = FmhaStage.Loop,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar]:
        """Exclude TMA zero-fill lanes in a partial fixed dense K/V tile.

        ``section=Tail``: called once for the last tile, mask always applied.
        ``section=Loop``: used in loop, masks only on the last iteration.
        """
        tmem_x = self.cfg.tmem_x_load_s
        num_chunks = self.cfg.qk_mma_tiler[1] // tmem_x
        s_data = self._load_s_chunks(stage_info)

        if cutlass.const_expr(section == FmhaStage.Tail):
            is_tail_tile = True
        else:
            is_tail_tile = stage_info.loop_offset == stage_info.loop_end - Int32(1)
        if is_tail_tile:
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
        if cutlass.const_expr(self.cfg.uses_ldtm_stat):
            return self._load_s_chunks_and_reduce_row_max(
                stage_info, row_max, causal_loop_mask=True, q_offset=q_offset
            )
        s_data = self._load_s_chunks(stage_info)
        s_data = self._apply_causal_mask_for_kv_tile(
            stage_info, s_data, kv_tile_idx=stage_info.loop_offset, q_offset=q_offset
        )
        return self._reduce_row_max(s_data, row_max)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def store_p(self, stage_info: StageInfo) -> None:
        """Store the P row from exp2_p into SMEM after the P-tile acquire; the
        proxy fence orders the stores before the UMMA reads them. Auxiliary work
        because the S stage is already released and P goes to the SMEM tile."""
        self._store_p_row_smem(stage_info, _tmem_sp_pwords.pop(id(self)))
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

    @consumer_work(
        returns=(
            vc_prev_tile_sum,
            row_sum,
            vc_pend0,
            vc_pend1,
            vc_pend2,
            vc_pend3,
            vc_pend4,
            vc_pend5,
            vc_pend6,
            vc_pend7,
            vc_pend8,
            vc_pend9,
            vc_pend10,
            vc_pend11,
            vc_pend12,
            vc_pend13,
            vc_pend14,
            vc_pend15,
        )
    )
    @cute.jit
    def vc_store_p(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        vc_prev_tile_sum: SoftmaxScalar,
        p_chunk: SoftmaxRowSumContribution,
        vc_pend0: SoftmaxScalar,
        vc_pend1: SoftmaxScalar,
        vc_pend2: SoftmaxScalar,
        vc_pend3: SoftmaxScalar,
        vc_pend4: SoftmaxScalar,
        vc_pend5: SoftmaxScalar,
        vc_pend6: SoftmaxScalar,
        vc_pend7: SoftmaxScalar,
        vc_pend8: SoftmaxScalar,
        vc_pend9: SoftmaxScalar,
        vc_pend10: SoftmaxScalar,
        vc_pend11: SoftmaxScalar,
        vc_pend12: SoftmaxScalar,
        vc_pend13: SoftmaxScalar,
        vc_pend14: SoftmaxScalar,
        vc_pend15: SoftmaxScalar,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> tuple[Float32, ...]:
        """VC-Attention-QK16 P store. Writes the P row, rescales the group row sums
        by this tile's max correction and slots in the previous tile's sum. Every
        8th tile writes the completed group operand under the same P-ready
        handoff. Returns this tile's sum, the updated row sum and the group sums.
        """
        self._store_p_row_smem(stage_info, _tmem_sp_pwords.pop(id(self)))
        acc_scale = cute.math.exp2(old_row_max - row_max, fastmath=True)
        # The peeled tail entry carries the loop offset of the previous tile.
        if cutlass.const_expr(is_tail):
            tile = Int32(stage_info.loop_end)
        else:
            tile = Int32(stage_info.loop_offset)
        slot = (tile - Int32(1)) & Int32(VC_MEAN_GROUP_TILES - 1)
        new_sum = vc_prev_tile_sum * acc_scale
        sums = [
            vc_pend0,
            vc_pend1,
            vc_pend2,
            vc_pend3,
            vc_pend4,
            vc_pend5,
            vc_pend6,
            vc_pend7,
            vc_pend8,
            vc_pend9,
            vc_pend10,
            vc_pend11,
            vc_pend12,
            vc_pend13,
            vc_pend14,
            vc_pend15,
        ]
        for i in cutlass.range_constexpr(VC_MEAN_GROUP_TILES):
            scaled = sums[i] * acc_scale
            if slot == Int32(i):
                scaled = new_sum
            sums[i] = scaled
        keep = Float32(1.0)
        if ((tile & Int32(VC_MEAN_GROUP_TILES - 1)) == Int32(0)) & (tile > Int32(0)):
            self._store_vc_rowsum_group(stage_info, tuple(sums))
            keep = Float32(0.0)
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )
        return (p_chunk, row_sum * acc_scale + p_chunk) + tuple(
            sums[i] * keep for i in range(VC_MEAN_GROUP_TILES)
        )

    @consumer_work(returns=(vc_kept_row_sum, row_sum))
    @cute.jit
    def vc_store_p_repair(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        vc_kept_row_sum: SoftmaxScalar,
        p_chunk: SoftmaxRowSumContribution,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> tuple[Float32, Float32]:
        """VC-Attention-QK16 V repair P store. The last ``vc_repair_tiles`` of the
        loop domain stay out of the kept row sum that normalizes the output."""
        self._store_p_row_smem(stage_info, _tmem_sp_pwords.pop(id(self)))
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )
        acc_scale = cute.math.exp2(old_row_max - row_max, fastmath=True)
        kept_chunk = p_chunk
        if cutlass.const_expr(not is_tail):
            first_repair = Int32(stage_info.loop_end) - Int32(self.cfg.vc_repair_tiles)
            if Int32(stage_info.loop_offset) >= first_repair:
                kept_chunk = Float32(0.0)
        return (
            vc_kept_row_sum * acc_scale + kept_chunk,
            row_sum * acc_scale + p_chunk,
        )

    @consumer_work
    @cute.jit
    def vc_store_rowsum_final(
        self,
        stage_info: StageInfo,
        *,
        vc_prev_tile_sum: SoftmaxScalar,
        vc_pend0: SoftmaxScalar,
        vc_pend1: SoftmaxScalar,
        vc_pend2: SoftmaxScalar,
        vc_pend3: SoftmaxScalar,
        vc_pend4: SoftmaxScalar,
        vc_pend5: SoftmaxScalar,
        vc_pend6: SoftmaxScalar,
        vc_pend7: SoftmaxScalar,
        vc_pend8: SoftmaxScalar,
        vc_pend9: SoftmaxScalar,
        vc_pend10: SoftmaxScalar,
        vc_pend11: SoftmaxScalar,
        vc_pend12: SoftmaxScalar,
        vc_pend13: SoftmaxScalar,
        vc_pend14: SoftmaxScalar,
        vc_pend15: SoftmaxScalar,
    ) -> None:
        """VC-Attention-QK16: slot the last tile's row sum into its group and
        store the group operands for the tail mean step."""
        last_tile = Int32(stage_info.loop_end)
        if cutlass.const_expr(not self.uses_fixed_dense_k_tail_mask):
            last_tile = last_tile - Int32(1)
        slot = last_tile & Int32(VC_MEAN_GROUP_TILES - 1)
        sums = [
            vc_pend0,
            vc_pend1,
            vc_pend2,
            vc_pend3,
            vc_pend4,
            vc_pend5,
            vc_pend6,
            vc_pend7,
            vc_pend8,
            vc_pend9,
            vc_pend10,
            vc_pend11,
            vc_pend12,
            vc_pend13,
            vc_pend14,
            vc_pend15,
        ]
        for i in cutlass.range_constexpr(VC_MEAN_GROUP_TILES):
            value = sums[i]
            if slot == Int32(i):
                value = vc_prev_tile_sum
            sums[i] = value
        self._store_vc_rowsum_group(stage_info, tuple(sums))
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

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

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def exp2_p_smem(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """exp2_p with P in SMEM, run on the loaded S after the stage release."""
        return self._exp2_p_store(
            self._stage_col_offset(stage_info), row_max, scale_softmax_log2
        )

    @consumer_work(returns=p_lo)
    @cute.jit
    def exp2_p_lo(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> Float32:
        """First half of exp2_p: keys [0, N/2). Stores those P columns and
        fences so the MMA can start PV on them. Returns their row sum."""
        return self._exp2_p_store(
            self._stage_col_offset(stage_info),
            row_max,
            scale_softmax_log2,
            chunk_hi=self.cfg.qk_mma_tiler[1] // self.cfg.tmem_x_load_s // 2,
            sum_in=Float32(0.0),
        )

    @consumer_work(returns=p_chunk)
    @cute.jit
    def exp2_p_hi(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
        p_lo: Float32,
    ) -> Float32:
        """Second half of exp2_p: keys [N/2, N). Returns the full tile sum."""
        return self._exp2_p_store(
            self._stage_col_offset(stage_info),
            row_max,
            scale_softmax_log2,
            chunk_lo=self.cfg.qk_mma_tiler[1] // self.cfg.tmem_x_load_s // 2,
            sum_in=p_lo,
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

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=p_chunk)
    @cute.jit
    def masked_exp2_p_smem(
        self,
        stage_info: StageInfo,
        *,
        row_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
    ) -> SoftmaxRowSumContribution:
        """masked_exp2_p with P in SMEM, after the stage release."""
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
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention-QK16 keeps the row max in log2 units.
            scale_softmax_log2 = Float32(1.0)
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

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=row_max)
    @cute.jit
    def freeze_row_max(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        scale_softmax_log2: Float32,
    ) -> SoftmaxScalar:
        """Keep the old row max when the tile raises it by at most the threshold."""
        _ = stage_info
        threshold = Float32(self.cfg.corr_skip_threshold_log2)
        frozen = row_max
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention-QK16 keeps the row max in log2 units.
            scale_softmax_log2 = Float32(1.0)
        if (row_max - old_row_max) * scale_softmax_log2 <= threshold:
            frozen = old_row_max
        return frozen

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
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention-QK16 softmax publishes row maxima already in log2 units.
            return Float32(1.0)
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
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention-QK16: output_scale is the per-(batch, head, channel) V
            # residual scale table, applied per column in SmemOResource._store_o.
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

    Producer: MMA writes P*V into one O accumulator per Q instance.
    Consumer: Correction rescales O in-place.

    Paired schedules use two stages so MMA can commit O0, work on O1,
    commit O1, then acquire O0. Single-instance schedules use one stage
    because they write one physical O accumulator.
    """

    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_o0_offset: Constexpr[int] = field(init=False, default=None)
    tmem_o1_offset: Constexpr[int] = field(init=False, default=None)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    # Precomputed TMEM raw pointer (inttoptr of tmem_addr_cached).
    tmem_ptr_raw_cached: TmemPtr | None = field(init=False, default=None)
    vc_mean_issue_cached: Any | None = field(init=False, default=None)
    vc_desc_l_cached: Any | None = field(init=False, default=None)
    vc_idesc_mu_cached: Any | None = field(init=False, default=None)
    # Precomputed per-warp TMEM O address base: (row_id << 16) | tmem_base_col.
    # consumer_work adds tmem_o_offset to get the final O0/O1 address.
    tmem_o_addr_base_cached: TmemAddr | None = field(init=False, default=None)
    # P-stage base supplied by TmemPResource for split S/P scheduling.
    tmem_p_base_cached: TmemAddr | None = field(init=False, default=None)
    # p_in_smem: per-group SMEM P tiles read by the SS PV MMA.
    smem_p0_resource: Optional[SmemPResource] = field(init=False, default=None)
    smem_p1_resource: Optional[SmemPResource] = field(init=False, default=None)
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
        smem_p0_resource: Optional[SmemPResource] = None,
        smem_p1_resource: Optional[SmemPResource] = None,
        vc_ctrl: cute.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Bind O TMEM offsets, correction-stat resources, and SMEM P tiles.

        ``vc_ctrl`` is the VC-Attention-QK16 run control word: element 0 is 1 when
        the tile means are restored and 0 when the run is the plain low-bit
        kernel (V-Smooth off), which skips the mean UMMA steps.
        """
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.vc_ctrl = vc_ctrl
        self.smem_p0_resource = smem_p0_resource
        self.smem_p1_resource = smem_p1_resource
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
        if cutlass.const_expr(self.cfg.vc_restores_means):
            issue = self._vc_restore_means()
            if cutlass.const_expr(self.cfg.two_cta_umma):
                issue = issue & (
                    cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
                )
            self.vc_mean_issue_cached = issue
            self.vc_desc_l_cached = tuple(
                self._vc_rowsum_desc(stage_info, smem_p)
                for smem_p in (self.smem_p0_resource, self.smem_p1_resource)
            )
            self.vc_idesc_mu_cached = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=cutlass.BFloat16,
                b_dtype=cutlass.BFloat16,
                n_dim=self.cfg.pv_mma_tiler[1],
                m_dim=self.cfg.pv_mma_tiler[0] * self.cfg.cta_group_size,
            )
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
        k_half: cutlass.Constexpr[int | None] = None,
    ) -> None:
        """PV MMA: P*V -> O (double-buffered O0/O1). See ``_pv_mma_impl``."""
        self._pv_mma_impl(
            stage_info,
            desc_v_base=desc_v_base,
            section=section,
            head_dim_stage_idx=head_dim_stage_idx,
            inst_idx=inst_idx,
            is_tail=is_tail,
            k_half=k_half,
        )

    @producer_work
    @cute.jit
    def vc_mean_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_mu_base: prims.Tcgen05SmemDesc,
        inst_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """VC-Attention-QK16: ``O += sum_i rowsum(P_i) * mean_i`` over the 16 tiles
        before this one for query group ``inst_idx``, two bf16 K=16 UMMA steps
        after that group's PV steps every 16th tile."""
        if cutlass.const_expr(is_tail):
            tile = Int32(stage_info.loop_end)
        else:
            tile = Int32(stage_info.loop_offset)
        group_done = ((tile & Int32(VC_MEAN_GROUP_TILES - 1)) == Int32(0)) & (
            tile > Int32(0)
        )
        self._vc_mean_mma(
            inst_idx, desc_mu_base, self.vc_mean_issue_cached & group_done
        )

    @producer_work
    @cute.jit
    def vc_mean_mma_last(
        self,
        stage_info: StageInfo,
        *,
        desc_mu_last: prims.Tcgen05SmemDesc,
        inst_idx: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Tail variant of ``vc_mean_mma`` reading the last group's operand handle."""
        self._vc_mean_mma(inst_idx, desc_mu_last, self.vc_mean_issue_cached)

    @cute.jit
    def _pv_mma_impl(
        self,
        stage_info: StageInfo,
        *,
        desc_v_base: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[FmhaStage],
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        inst_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
        k_half: cutlass.Constexpr[int | None] = None,
    ) -> None:
        """PV MMA: P*V -> O (double-buffered O0/O1).

        ``k_half`` issues only the first (0) or second (1) half of the K slices
        so the schedule can run PV on the leading half of P before the rest is
        stored. The second half accumulates onto the first.

        Uses captured schedule section and call index to select O0/O1 and
        scale_d statically.
        Reads P from TMEM (via tmem_p offset on the corresponding SP resource)
        and V descriptor from SmemV consumer vars.

        In causal mode with no Q right offset, skips P0*V→O0 MMA in the last
        LOOP iteration, since Softmax0's domain is N-2 but MMA's domain is N-1.
        The task domain pads partial final CTAs so this slot is always outside
        peer0's causal reach.
        """
        if cutlass.const_expr(self.cfg.p_in_smem):
            # PV0(i), PV1(i) every iteration and in the tail, so inst_idx is the group.
            writes_o0 = inst_idx == 0
            first_o0_write = False
            first_o1_write_maybe = False
        elif cutlass.const_expr(section == FmhaStage.Head):
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

        # In causal mode, check if O0 MMA should skip peer 0's invalid last tile:
        # the last loop iteration, or the tail when P in SMEM moves PV0 there.
        skip_o0_invalid = False
        if cutlass.const_expr(self.cfg.skip_causal_invalid_peer0 and writes_o0):
            if cutlass.const_expr(self.cfg.p_in_smem):
                skip_o0_invalid = is_tail
            elif cutlass.const_expr(section == FmhaStage.Loop):
                if not is_tail:
                    skip_o0_invalid = stage_info.loop_offset == (
                        stage_info.loop_end - 1
                    )

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
                m_dim=self.cfg.pv_mma_tiler[0] * self.cfg.cta_group_size,
                # V is row-major / MN-major.
                b_major=1,
            )
            cta_group = (
                prims.CTAGroup.CTA_2
                if cutlass.const_expr(self.cfg.two_cta_umma)
                else prims.CTAGroup.CTA_1
            )
            issue_mma = cutlass.Boolean(True)
            if cutlass.const_expr(self.cfg.two_cta_umma):
                issue_mma = (
                    cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
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
                self.cfg.tma_copy_v_iters // num_head_dim_stages
            )
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                tma_copy_iters_per_head_dim_stage = self.cfg.tma_copy_v_stage_iters
            inc_bytes_v = (
                k_dim_per_mma
                * (pv_n_dim // tma_copy_iters_per_head_dim_stage)
                * self.cfg.v_dtype.width
                // 8
            )
            if cutlass.const_expr(self.cfg.two_cta_umma):
                # One V fragment of tma_copy_v_granu_inner columns per CTA.
                inc_bytes_v = (
                    k_dim_per_mma
                    * self.cfg.tma_copy_v_granu_inner
                    * self.cfg.v_dtype.width
                    // 8
                )
            v_chunk_bytes = self.cfg.tma_copy_v_bytes // self.cfg.tma_copy_v_stage_iters
            head_dim_stage_bytes_v = v_chunk_bytes * tma_copy_iters_per_head_dim_stage

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
            elif cutlass.const_expr(self.cfg.p_in_smem):
                if cutlass.const_expr(is_tail):
                    scale_d = stage_info.loop_end > 0
                else:
                    scale_d = stage_info.loop_offset > 0
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
                    if issue_mma:
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                cta_group,
                                tmem_ptr_o_stage,
                                dp,
                                dv,
                                idesc_pv,
                                scale_d_stage,
                            )
                    scale_d_stage = True
            else:
                if cutlass.const_expr(self.cfg.p_in_smem):
                    smem_p = (
                        self.smem_p0_resource if writes_o0 else self.smem_p1_resource
                    )
                    sP = cutlass.Array(
                        stage_info.context.smem_base.data_ptr() + smem_p._alloc.offset,
                        dtype=cutlass.Int8,
                        shape=(self.cfg.smem_p_bytes,),
                        addrspace=3,
                    )
                    p_lbo, p_sbo = smem_p.descriptor_offsets()
                    desc_p_base_ = freeze_smem_descriptor(
                        prims.Tcgen05SmemDesc.build(
                            sP,
                            leading_byte_offset=p_lbo,
                            stride_byte_offset=p_sbo,
                            layout=smem_p.descriptor_layout(),
                        )
                    )
                    inc_bytes_p = k_dim_per_mma * self.cfg.v_dtype.width // 8
                for head_dim_stage_idx in cutlass.range_constexpr(
                    num_head_dim_stages_to_issue
                ):
                    tmem_ptr_o_stage = tmem_ptr_o.subview(head_dim_stage_idx * pv_n_dim)
                    v_stage_increment = (
                        head_dim_stage_bytes_v * head_dim_stage_idx
                    ) >> 4
                    if cutlass.const_expr(k_half is None):
                        k_lo, k_hi = 0, num_kphases_pv
                    else:
                        k_lo = k_half * (num_kphases_pv // 2)
                        k_hi = k_lo + num_kphases_pv // 2
                    scale_d_stage = scale_d if k_lo == 0 else True
                    for k_idx in cutlass.range_constexpr(k_lo, k_hi):
                        if cutlass.const_expr(self.cfg.p_in_smem):
                            dp = desc_p_base_ + ((inc_bytes_p * k_idx) >> 4)
                        else:
                            dp = tmem_ptr_raw.subview(tmem_p_base + k_idx * inc_tmem_p)
                        increment = v_stage_increment + ((inc_bytes_v * k_idx) >> 4)
                        dv = desc_v_base_ + increment
                        if issue_mma:
                            if prims.elect_sync():
                                prims.tcgen05_mma(
                                    mma_kind,
                                    cta_group,
                                    tmem_ptr_o_stage,
                                    dp,
                                    dv,
                                    idesc_pv,
                                    scale_d_stage,
                                )
                        scale_d_stage = True

    def _vc_mean_cta_group(self):
        """cta_group::2 under two-CTA UMMA (M=256 over both CTAs' row sums)."""
        if self.cfg.two_cta_umma:
            return prims.CTAGroup.CTA_2
        return prims.CTAGroup.CTA_1

    @cute.jit
    def _vc_restore_means(self) -> cutlass.Boolean:
        """Whether this run restores the V tile means (``vc_ctrl[0] != 0``)."""
        if cutlass.const_expr(self.vc_ctrl is None):
            return cutlass.Boolean(True)
        return self.vc_ctrl[0] != Int32(0)

    @cute.jit
    def _vc_rowsum_desc(
        self, stage_info: StageInfo, smem_p: SmemPResource
    ) -> prims.Tcgen05SmemDesc:
        """K-major descriptor of a query group's row-sum operands (A of the mean steps)."""
        sL = cutlass.Array(
            stage_info.context.smem_base.data_ptr()
            + smem_p._alloc.offset
            + smem_p.rowsum_tile_offset,
            dtype=cutlass.Int8,
            shape=(self.cfg.vc_rowsum_tile_bytes,),
            addrspace=3,
        )
        return freeze_smem_descriptor(
            prims.Tcgen05SmemDesc.build(
                sL,
                leading_byte_offset=VC_MEAN_TILE_LBO,
                stride_byte_offset=VC_MEAN_TILE_SBO,
                layout=prims.Tcgen05SmemSwizzle.NONE,
            )
        )

    @cute.jit
    def _vc_mean_mma(
        self,
        inst_idx: cutlass.Constexpr[int],
        desc_mu_base: prims.Tcgen05SmemDesc,
        issue_mma: Any,
    ) -> None:
        """Issue ``O += sum_i rowsum(P_i) * mean_i`` for query group ``inst_idx`` as
        ``VC_MEAN_OPERANDS`` bf16 K=16 UMMA steps back to back.

        A is the query group's row-sum operands behind its P tile, B the
        TMA-staged mean operands of the tile group.
        """
        if issue_mma:
            if prims.elect_sync():
                if cutlass.const_expr(inst_idx == 0):
                    tmem_ptr_o = self.tmem_ptr_raw_cached.subview(self.tmem_o0_offset)
                else:
                    tmem_ptr_o = self.tmem_ptr_raw_cached.subview(self.tmem_o1_offset)
                desc_l = self.vc_desc_l_cached[inst_idx]
                desc_mu = freeze_smem_descriptor(desc_mu_base)
                for operand in cutlass.range_constexpr(VC_MEAN_OPERANDS):
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        self._vc_mean_cta_group(),
                        tmem_ptr_o,
                        desc_l + ((operand * self.cfg.vc_rowsum_operand_bytes) >> 4),
                        desc_mu
                        + (
                            (
                                operand
                                * self.cfg.vc_mean_operand_bytes
                                // self.cfg.cta_group_size
                            )
                            >> 4
                        ),
                        self.vc_idesc_mu_cached,
                        True,
                    )

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
        should_rescale = cutlass.Boolean(True)
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
            # Amortize TMEM handshakes with 32-value correction transfers.
            tmem_x = 32

            num_iters = self.cfg.pv_mma_tiler[1] // tmem_x
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

        vc_scale_base = Int64(0)
        vc_head_coord = Int32(0)
        vc_batch_coord = Int32(0)
        if cutlass.const_expr(self.cfg.vc_attention):
            assert self.tmem_vec_resource is not None
            assert self.tmem_vec_resource.output_scale is not None
            assert self.cfg.work_tile_q_heads == 1
            vc_scale_base = self.tmem_vec_resource.output_scale.iterator.toint()
            _, head_coord_wt, vc_batch_coord = _resolve_work_tile_coords(
                self.cfg, stage_info.work_tile.tile_idx
            )
            vc_head_coord = head_coord_wt * self.cfg.work_tile_q_heads

        o_head_dim = self.cfg.epi_tile[1]
        tma_copy_o_iters = self.cfg.tma_copy_o_iters
        if cutlass.const_expr(self.cfg.stage_o_by_head_dim):
            o_head_dim = self.cfg.head_dim_per_stage_o
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
            if cutlass.const_expr(self.cfg.vc_attention):
                # Per-channel V residual scale s[b, h, d]. The tile means were
                # stored divided by it, so the whole accumulator is rescaled.
                col0 = head_dim_stage_idx * o_head_dim + i * tmem_x
                chan_idx = (
                    Int64(vc_batch_coord) * Int64(self.cfg.vc_num_q_heads)
                    + Int64(vc_head_coord)
                ) * Int64(self.cfg.vc_head_dim_v) + Int64(col0)
                chan_scale = cutlass.inttoptr(
                    vc_scale_base + chan_idx * 4, mem_space=1, dtype=Float32
                ).load(count=tmem_x, alignment=64)
                o_rmem = o_rmem * chan_scale

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
                            head_dim_stage_idx * self.cfg.head_dim_per_stage_o
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
