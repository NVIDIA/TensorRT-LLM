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

"""Common helpers shared across FMHA decode TS resource files.

Holds the small primitives, type aliases, task-cache offsets, config-driven
shape/dtype/swizzle helpers, and ``DecodeGenResourceBase`` — anything that
multiple resource classes (and the other ``_helpers_*`` modules) need.
"""

from functools import partial
from typing import ClassVar


import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
)

from ..fmha_decode_config import FmhaDecodeConfig

Constexpr = cutlass.Constexpr
NEG_FLT_MAX = -3.4028235e38
fadd2 = partial(cute.arch.add_packed_f32x2, ftz=False, rnd="rn")
fmul2 = partial(cute.arch.mul_packed_f32x2, ftz=False, rnd="rn")
ffma2 = partial(cute.arch.fma_packed_f32x2, ftz=False, rnd="rn")

TaskCache = tuple[
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
]
DescriptorValue = prims.Tcgen05SmemDesc | cutlass.Int64 | Int32
ResourceVarValue = (
    Int32 | Float32 | Uint32 | cutlass.Int64 | cutlass.Array | DescriptorValue
)
ResourceVars = dict[str, ResourceVarValue]


@dsl_user_op
def _assume_nonnegative_i32(value: Int32, *, loc=None, ip=None) -> Int32:
    """Express a caller-guaranteed nonnegative Int32 contract to codegen."""

    condition = cutlass.Boolean(value >= Int32(0))
    llvm.intr_assume(
        condition.ir_value(loc=loc, ip=ip),
        [],
        [],
        loc=loc,
        ip=ip,
    )
    return value


# Offsets into DecodeGenTask.make_task_cache(). Keeping these symbolic makes
# resource code explicit about which task-local lane or address value it needs.
_TASK_CACHE_TMEM_BASE_OFFSET = 0
_TASK_CACHE_WARP_GRP_THREAD_IDX = 1
_TASK_CACHE_WARP_IDX = 2
_TASK_CACHE_LANE_IDX = 3
_TASK_CACHE_SEQ_LEN_KV = 4
_TASK_CACHE_KV_REQUEST_BEGIN = 5
_TASK_CACHE_KV_PAGE_IDX_UB = 6
_TASK_CACHE_KV_RAW_TILE_BASE = 7
_TASK_CACHE_KV_VALID_TILE_END = 8
_TASK_CACHE_KV_WINDOW_START = 9
# Block-sparse rows share the two generic KV-span words with paged decode.
# Semantic aliases keep sparse consumers independent of page-table naming
# without extending the stable ten-word task-cache ABI.
_TASK_CACHE_SPARSE_ROUTE_BEGIN = _TASK_CACHE_KV_REQUEST_BEGIN
_TASK_CACHE_SPARSE_ROUTE_COUNT = _TASK_CACHE_KV_PAGE_IDX_UB


@cute.jit
def _sparse_task_cache_route_begin(task_cache: TaskCache) -> Int32:
    """Load the first prepared-route ordinal for one sparse row."""

    return Int32(task_cache[_TASK_CACHE_SPARSE_ROUTE_BEGIN])


@cute.jit
def _sparse_task_cache_route_count(task_cache: TaskCache) -> Int32:
    """Load the live prepared-route count for one sparse row."""

    return Int32(task_cache[_TASK_CACHE_SPARSE_ROUTE_COUNT])


@cute.jit
def _warp_broadcast_i32(value: Int32, source_lane: Constexpr[int]) -> Int32:
    """Broadcast one source-lane scalar as a warp-uniform Int32 value."""

    return cute.arch.make_warp_uniform(
        Int32(
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=value,
                offset=source_lane,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
        )
    )


@cute.jit
def _swaps_routed_coordinate(
    cfg: Constexpr[FmhaDecodeConfig],
    lane_k_offset: Int32,
    origin0: Int32,
    origin1: Int32,
    origin2: Int32,
    origin3: Int32,
    *,
    token_group_idx: Constexpr[int],
) -> tuple[Int32, Int32]:
    """Map one SWAP register group to its staged atom and logical coordinate."""

    atom_size = min(cfg.kv_block_size, 32)
    groups_per_atom = atom_size // 8
    origin_idx = token_group_idx // groups_per_atom
    atom_origin = origin0
    if cutlass.const_expr(origin_idx == 1):
        atom_origin = origin1
    elif cutlass.const_expr(origin_idx == 2):
        atom_origin = origin2
    elif cutlass.const_expr(origin_idx == 3):
        atom_origin = origin3
    token_offset = (token_group_idx % groups_per_atom) * 8
    return atom_origin, atom_origin + Int32(token_offset) + lane_k_offset


def _mma_kind_for_qk(cfg: FmhaDecodeConfig) -> prims.Tcgen05MMAKind:
    """Select the tcgen05 MMA kind of the QK GEMM from ``qk_dtype``.

    Int8 accumulates INT32 scores; E4M3 and the 16-bit types accumulate FP32.
    The INT8 kind exists on SM100 alone, so planning admits Int8 Q/K only on
    that architecture (``_validate_int8_qk_device``).
    """
    if cfg.uses_int32_scores:
        return prims.Tcgen05MMAKind.INT8
    if cfg.use_fp8_qk:
        return prims.Tcgen05MMAKind.F8F6F4
    return prims.Tcgen05MMAKind.F16


def _mma_k_step_qk(cfg: FmhaDecodeConfig) -> int:
    """Return the K dimension advanced by one QK-GEMM MMA instruction."""
    return 32 if cfg.qk_dtype_bytes == 1 else 16


def _mma_kind_for_pv(cfg: FmhaDecodeConfig) -> prims.Tcgen05MMAKind:
    """Select the tcgen05 MMA opcode family for the PV GEMM."""
    if cfg.use_fp8_pv:
        return prims.Tcgen05MMAKind.F8F6F4
    return prims.Tcgen05MMAKind.F16


def _mma_k_step_pv(cfg: FmhaDecodeConfig) -> int:
    """Return the K dimension advanced by one PV-GEMM MMA instruction."""
    return cfg.pv_mma_k_step


def _qk_accumulator_dtype(cfg: FmhaDecodeConfig) -> type:
    """Return the type the QK GEMM accumulates one score as: Int32 for Int8 Q/K.

    INT32 scores are accumulated on top of ``INT32_SCORE_BIAS``, so the
    softmax reads every score tile as FP32.
    """
    return Int32 if cfg.uses_int32_scores else Float32


@cute.jit
def _freeze_smem_descriptor(desc):
    """Copy a SMEM descriptor through a register before MMA integer offsets."""
    return cute.arch.inline_ptx(
        "mov.b64 {$w0}, {$r0};",
        write_only_types=[Int64],
        read_only_args=[desc],
    )


@cute.jit
def _wait_for_mbarrier_phase(barrier, phase: Int32) -> None:
    """Wait for one parity of a CTA-local reusable mbarrier."""

    while not prims.mbarrier_try_wait_parity(
        barrier,
        phase,
        time_limit=10_000_000,
    ):
        pass


def _softmax_scale_pair_width(num_scale_groups: int, scale_base: int) -> int:
    """Return the number of live lanes in one packed two-group operation."""
    return min(2, num_scale_groups - scale_base)


def _shape_tuple(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    """Normalize placeholder shapes to a tuple form accepted by cutlass.Array."""
    if isinstance(shape, tuple):
        return shape
    return (shape,)


def _placeholder_smem_array(
    dtype: type, shape: int | tuple[int, ...] = 1
) -> cutlass.Array | None:
    """Create a fake SMEM array when resource construction needs a placeholder."""
    try:
        return cutlass.Array(
            cutlass.Int64(0),
            dtype=dtype,
            shape=_shape_tuple(shape),
            addrspace=3,
        )
    except (RuntimeError, ValueError):
        return None


def _placeholder_local_array(
    dtype: type, shape: int | tuple[int, ...] = 1, alignment: int | None = None
) -> cutlass.Array | None:
    """Create a fake register-space array when tracing needs a placeholder."""
    try:
        if alignment is None:
            return cutlass.Array(dtype, shape, space=cutlass.AddressSpace.rmem)
        return cutlass.Array(
            dtype, shape, space=cutlass.AddressSpace.rmem, alignment=alignment
        )
    except (RuntimeError, ValueError):
        return None


@cute.jit
def _keeps_q64_row_idx(warp_grp_thread_idx: Int32) -> Int32:
    """Map a correction lane to its q-row for tileSizeQ <= 64 keeps paths."""
    return (warp_grp_thread_idx >> Int32(5)) * Int32(16) + (
        warp_grp_thread_idx & Int32(0xF)
    )


@cute.jit
def _keeps_q64_col_base(lane_idx: Int32, half_cols: int) -> Int32:
    """Return the head-dim column base for one q64 keeps lane group."""
    return (lane_idx >> Int32(4)) * Int32(half_cols)


@cute.jit
def _keeps_spatial_half(
    cfg: Constexpr[FmhaDecodeConfig], warp_grp_thread_idx: Int32
) -> Int32:
    """Return the spatial half a Keeps warp-group thread belongs to.

    KV256 threads ``[0, 64)`` and ``[64, 128)`` own the two spatial KV128
    partials of one logical Q row (``FmhaDecodeConfig.keeps_spatial_halves``);
    every other Keeps profile has a single half.
    """
    if cutlass.const_expr(cfg.keeps_spatial_halves == 2):
        return warp_grp_thread_idx >> Int32(6)
    return Int32(0)


@cute.jit
def _keeps_route_atom(cfg: Constexpr[FmhaDecodeConfig], half: Int32, position) -> Int32:
    """Return the route atom a Keeps spatial half owns at ``position``.

    The halves interleave over the route's atoms: half ``h`` owns atoms ``h``,
    ``h + halves``, ... in order, so its ``position``-th atom is ``position *
    halves + h`` (``FmhaDecodeConfig.keeps_route_atom_owner`` is the inverse).
    """
    return position * Int32(cfg.keeps_spatial_halves) + half


@cute.jit
def _keeps_row_idx(cfg: Constexpr[FmhaDecodeConfig], warp_grp_thread_idx: Int32):
    """Map a correction lane to the logical output row it owns."""
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        # KV256's 2x2 datapath exposes two spatial KV128 partials for each
        # logical Q row. Threads [0, 64) and [64, 128) therefore share rows.
        return warp_grp_thread_idx & Int32(63)
    if cutlass.const_expr(cfg.tile_size_q == 128):
        return warp_grp_thread_idx
    return _keeps_q64_row_idx(warp_grp_thread_idx)


@cute.jit
def _keeps_col_base(
    cfg: Constexpr[FmhaDecodeConfig], lane_idx: Int32, half_cols: int
) -> Int32:
    """Return the lane's keepsMmaAb output-column base."""
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        return Int32(0)
    if cutlass.const_expr(cfg.tile_size_q == 128):
        return Int32(0)
    return _keeps_q64_col_base(lane_idx, half_cols)


@cute.jit
def _keeps_score_col(
    cfg: Constexpr[FmhaDecodeConfig],
    warp_grp_thread_idx: Int32,
    reg_idx: Constexpr[int],
    col_base: Int32,
) -> Int32:
    """Return the semantic KV column represented by one Keeps score register."""
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        # A physical thread owns four K32 fragments: ``keeps_fragments_per_atom``
        # per layout atom of its spatial half (``_keeps_route_atom``), each
        # fragment one K32 sub-block of that atom.
        fragment_regs = cfg.softmax_score_fragment_regs
        fragments_per_atom = cfg.keeps_fragments_per_atom
        spatial = _keeps_spatial_half(cfg, warp_grp_thread_idx)
        fragment = reg_idx // fragment_regs
        semantic_block = _keeps_route_atom(cfg, spatial, fragment // fragments_per_atom)
        return (
            semantic_block * Int32(cfg.keeps_atom_tokens)
            + Int32((fragment % fragments_per_atom) * fragment_regs)
            + Int32(reg_idx % fragment_regs)
        )
    return col_base + Int32(reg_idx)


@cute.jit
def _keeps_tcgen05_ld(
    cfg: Constexpr[FmhaDecodeConfig],
    tmem_addr,
    *,
    num: Constexpr[int],
    offset: Constexpr[int],
):
    """Load keepsMmaAb TMEM fragments using the tileSizeQ-specific shape."""
    if cutlass.const_expr(cfg.tile_size_kv == 256 or cfg.tile_size_q == 128):
        return prims.tcgen05_ld(
            "32x32b",
            tmem_addr,
            num=num,
        )
    # The 16x32bx2 variant has a required half-split offset operand.  Route
    # through the public primitive wrapper so Python constants are materialized
    # as MLIR values before reaching the low-level operation.
    return prims.tcgen05_ld(
        "16x32bx2",
        tmem_addr,
        num=num,
        offset=offset,
    )


@cute.jit
def _keeps_tcgen05_st(
    cfg: Constexpr[FmhaDecodeConfig],
    tmem_addr,
    val,
    *,
    offset: Constexpr[int],
) -> None:
    """Store keepsMmaAb TMEM fragments using the tileSizeQ-specific shape."""
    if cutlass.const_expr(cfg.tile_size_kv == 256 or cfg.tile_size_q == 128):
        prims.tcgen05_st(
            "32x32b",
            tmem_addr,
            val,
        )
    else:
        prims.tcgen05_st(
            "16x32bx2",
            tmem_addr,
            val,
            offset=offset,
        )


@cute.jit
def _pack_float2_to_fp16(v0: Float32, v1: Float32) -> Int32:
    """Pack two FP32 values into one FP16x2 register."""
    return cutlass.Vector.from_elements((v0, v1), Float32).to(Float16).bitcast(Int32)[0]


@cute.jit
def _pack_float2_to_bf16(v0: Float32, v1: Float32) -> Int32:
    """Pack two FP32 values into one BF16x2 register."""
    return (
        cutlass.Vector.from_elements((v0, v1), Float32).to(BFloat16).bitcast(Int32)[0]
    )


def _q_dtype_elements(cfg: FmhaDecodeConfig) -> int:
    """Return element count in one staged Q SMEM allocation."""
    return cfg.smem_q_tile_bytes // cfg.q_dtype_bytes


def _kv_dtype_elements(cfg: FmhaDecodeConfig) -> int:
    """Return element count in one staged K/V SMEM allocation."""
    return cfg.smem_kv_tile_elements


def _qkv_smem_swizzle(
    cfg: FmhaDecodeConfig,
) -> prims.Tcgen05SmemSwizzle:
    """Select the tcgen05 SMEM swizzle for staged Q tiles."""
    if cfg.q_dtype_bytes == 1 and cfg.headdim == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B


def _kv_smem_swizzle(
    cfg: FmhaDecodeConfig,
) -> prims.Tcgen05SmemSwizzle:
    """Select the tcgen05 SMEM swizzle for raw staged K/V tiles."""
    if cfg.k_dtype.width == 8 and cfg.v_dtype.width == 8 and cfg.headdim == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B


def _major_k_stride_bytes(dtype_bytes: int, headdim: int) -> int:
    """Return the descriptor K-major stride in bytes for one swizzle block."""
    # The descriptor swizzle shape depends on head dim and operand type, not
    # on the number of Q rows in the tile.
    num_smem_cols = 128 // dtype_bytes
    rows_per_smem_row = max(1, num_smem_cols // headdim)
    if rows_per_smem_row == 1:
        rows_per_swizzle_blk = 8
    elif rows_per_smem_row == 2:
        rows_per_swizzle_blk = 4
    elif rows_per_smem_row == 4:
        rows_per_swizzle_blk = 2
    else:
        rows_per_swizzle_blk = 1
    return 128 * rows_per_swizzle_blk


def _neg_max_f32() -> Float32:
    """Return the negative sentinel used for running softmax maxima."""
    return Float32(NEG_FLT_MAX)


@cute.jit
def _masked_weighted_sum(terms: tuple) -> Float32:
    """Return the sum of ``value * weight`` over four ``(uses, value, weight)`` terms.

    A term whose ``uses`` is false contributes zero, whatever its value. The
    fixed term order makes lanes that share a row agree bitwise.
    """
    assert len(terms) == 4
    (uses0, value0, weight0), (uses1, value1, weight1) = terms[0], terms[1]
    (uses2, value2, weight2), (uses3, value3, weight3) = terms[2], terms[3]
    masked0 = Float32(0.0)
    masked1 = Float32(0.0)
    masked2 = Float32(0.0)
    masked3 = Float32(0.0)
    if uses0:
        masked0 = value0
    if uses1:
        masked1 = value1
    if uses2:
        masked2 = value2
    if uses3:
        masked3 = value3
    lanes = ffma2((masked0, masked2), (weight0, weight2), (Float32(0.0), Float32(0.0)))
    lanes = ffma2((masked1, masked3), (weight1, weight3), lanes)
    return lanes[0] + lanes[1]


def _softmax_tile_idx(
    cfg: FmhaDecodeConfig, stage_info: StageInfo, inst_id: int
) -> Int32:
    """Tile index consumed by the softmax-side MMA loop (inst_id ∈ {0, 1})."""
    return stage_info.loop_offset * Int32(cfg.num_insts_kv) + Int32(inst_id)


@cute.jit
def _named_barrier_arrive(
    number_of_threads: Constexpr[int], barrier_id: Constexpr[int]
) -> None:
    """Arrive at a named barrier from the configured warp subset."""
    # The non-aligned primitive is required here: ordered softmax uses four
    # participating warps rather than a CTA-wide converged barrier.
    prims.barrier_cta_arrive(barrier_id, number_of_threads)


@cute.jit
def _named_barrier_sync(
    number_of_threads: Constexpr[int], barrier_id: Constexpr[int]
) -> None:
    """Wait at a named barrier from the configured warp subset."""
    prims.barrier_cta_sync(barrier_id, thread_count=number_of_threads)


def _is_last_loop_iteration(stage_info: StageInfo) -> cutlass.Boolean:
    """Return whether the current schedule loop iteration is the final one."""
    return stage_info.loop_offset + Int32(1) == stage_info.loop_end


def _clamp_valid_tile_idx(cfg: FmhaDecodeConfig, tile_idx: Int32) -> Int32:
    """Clamp a static K/V tile index to the last valid tile."""
    return cute.math.min(tile_idx, Int32(cfg.total_kv_tiles - 1))


@cute.jit
def _decode_gen_task_cache(stage_info: StageInfo) -> TaskCache:
    """Return the task cache or a zero-filled placeholder cache."""
    if cutlass.const_expr(stage_info.task_cache is None):
        zero = Int32(0)
        return (
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        )
    return stage_info.task_cache


@cute.jit
def _logical_head_batch(
    stage_info: StageInfo, fallback_h_k_idx: Int32, fallback_b_idx: Int32
) -> tuple[Int32, Int32]:
    """Resolve logical KV head and batch from work tile or static launch."""
    if cutlass.const_expr(stage_info.work_tile is not None):
        tile_idx = stage_info.work_tile.tile_idx
        return Int32(tile_idx[1]), Int32(tile_idx[2])
    return fallback_h_k_idx, fallback_b_idx


@cute.jit
def _logical_q_group_idx(
    cfg: Constexpr[FmhaDecodeConfig],
    stage_info: StageInfo,
    fallback_q_group_idx: Int32,
) -> Int32:
    """Resolve the q-group from the persistent work tile or static launch."""
    if cutlass.const_expr(cfg.has_single_q_cta):
        # A split coordinate may still vary in grid X, but every physical CTA
        # maps to logical Q group zero. State this explicitly because the
        # release compiler does not infer the range from the launch geometry.
        return Int32(0)
    if cutlass.const_expr(stage_info.work_tile is not None):
        q_group_cta_idx = Int32(stage_info.work_tile.tile_idx[0])
        if cutlass.const_expr(cfg.use_split_kv):
            return q_group_cta_idx // Int32(cfg.splits_kv)
        return q_group_cta_idx
    return fallback_q_group_idx


@cute.jit
def _q_tile_output_row_base(
    cfg: Constexpr[FmhaDecodeConfig], q_group_idx: Int32
) -> Int32:
    """Return the first packed output row owned by a logical Q CTA."""
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        return q_group_idx * Int32(cfg.q_tma_rows_per_cta)
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        token_idx = q_group_idx // head_ctas_per_token
        head_cta_idx = q_group_idx - token_idx * head_ctas_per_token
        return token_idx * Int32(cfg.heads_q_per_kv) + head_cta_idx * Int32(
            cfg.tile_size_q
        )
    return q_group_idx * Int32(cfg.tile_size_q)


@cute.jit
def _q_seq_bounds(
    cfg: Constexpr[FmhaDecodeConfig],
    cu_seqlens_q: cute.Pointer | None,
    batch_idx: Int32,
) -> tuple[Int32, Int32]:
    """Return packed Q token offset/length, or fixed-SQ neutral bounds."""
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        q_begin = Int32(cu_seqlens_q[batch_idx])
        q_end = Int32(cu_seqlens_q[batch_idx + Int32(1)])
        return q_begin, q_end - q_begin
    return Int32(0), Int32(cfg.max_seq_len_q)


@cute.jit
def _q_group_token_base(cfg: Constexpr[FmhaDecodeConfig], q_group_idx: Int32) -> Int32:
    """Return the first Q token owned by a logical Q CTA."""
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        return q_group_idx * Int32(cfg.q_tokens_per_cta)
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        return q_group_idx // head_ctas_per_token
    return Int32(0)


@cute.jit
def _q_tile_valid_rows_for_seq(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    q_group_idx: Int32,
    seq_len_q: Int32,
) -> Int32:
    """Return runtime-valid MMA rows in one logical Q CTA.

    ``h_r`` is the total packed output-row count supplied to correction. Grouped
    profiles advance by complete tokens, so structural padding and the final
    partial token group are excluded independently from that global bound.
    """
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        token_base = q_group_idx * Int32(cfg.q_tokens_per_cta)
        remaining_tokens = cute.math.max(seq_len_q - token_base, Int32(0))
        valid_tokens = cute.math.min(remaining_tokens, Int32(cfg.q_tokens_per_cta))
        return valid_tokens * Int32(cfg.heads_q_per_kv)
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        token_idx = q_group_idx // head_ctas_per_token
        head_cta_idx = q_group_idx - token_idx * head_ctas_per_token
        head_base = head_cta_idx * Int32(cfg.tile_size_q)
        return cute.math.min(
            cute.math.max(Int32(cfg.heads_q_per_kv) - head_base, Int32(0)),
            Int32(cfg.tile_size_q),
        )
    return cute.math.min(
        cute.math.max(
            h_r - q_group_idx * Int32(cfg.tile_size_q),
            Int32(0),
        ),
        Int32(cfg.tile_size_q),
    )


@cute.jit
def _q_logical_output_row_token_and_local_head(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    logical_output_row_idx: Int32,
) -> tuple[Int32, Int32]:
    """Decompose one batch-local scratch row into token and local Q head.

    Split-KV scratch stores logical rows densely as ``[token, local_head]``.
    Packed final output needs that pair to replace the scratch buffer's padded
    batch stride with the public cumulative-token offset.
    """
    heads_q_per_kv = h_r
    if cutlass.const_expr(cfg.heads_q_per_kv != 0):
        heads_q_per_kv = Int32(cfg.heads_q_per_kv)
    heads_q_per_kv_fdd = cute.fast_divmod_create_divisor(heads_q_per_kv)
    return divmod(logical_output_row_idx, heads_q_per_kv_fdd)


@cute.jit
def _q_logical_output_row_is_valid_for_seq(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    logical_output_row_idx: Int32,
    seq_len_q: Int32,
) -> cutlass.Boolean:
    """Return whether a batch-local split scratch row owns packed output."""
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        return logical_output_row_idx < seq_len_q * Int32(cfg.heads_q_per_kv)
    return logical_output_row_idx < h_r


@cute.jit
def _q_physical_output_row_from_token_and_local_head(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    num_heads_kv: Int32,
    batch_idx: Int32,
    kv_head_idx: Int32,
    q_token_idx: Int32,
    local_head_idx: Int32,
    q_token_offset: Int32,
) -> Int32:
    """Map one token/local-head pair to the selected public O ABI."""
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        heads_q_per_kv = Int32(cfg.heads_q_per_kv)
        num_heads_q = num_heads_kv * heads_q_per_kv
        global_head_idx = kv_head_idx * heads_q_per_kv + local_head_idx
        return (q_token_offset + q_token_idx) * num_heads_q + global_head_idx
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        heads_q_per_kv = Int32(cfg.heads_q_per_kv)
        num_heads_q = num_heads_kv * heads_q_per_kv
        global_head_idx = kv_head_idx * heads_q_per_kv + local_head_idx
        return (
            batch_idx * Int32(cfg.max_seq_len_q) + q_token_idx
        ) * num_heads_q + global_head_idx
    return (batch_idx * num_heads_kv + kv_head_idx) * h_r + local_head_idx


@cute.jit
def _q_physical_output_row_from_logical(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    num_heads_kv: Int32,
    batch_idx: Int32,
    kv_head_idx: Int32,
    logical_output_row_idx: Int32,
    q_token_offset: Int32,
) -> Int32:
    """Map a batch-local split scratch row to the physical output tensor."""
    q_token_idx, local_head_idx = _q_logical_output_row_token_and_local_head(
        cfg, h_r, logical_output_row_idx
    )
    return _q_physical_output_row_from_token_and_local_head(
        cfg,
        h_r,
        num_heads_kv,
        batch_idx,
        kv_head_idx,
        q_token_idx,
        local_head_idx,
        q_token_offset,
    )


@cute.jit
def _q_physical_output_row(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    num_heads_kv: Int32,
    batch_idx: Int32,
    kv_head_idx: Int32,
    q_group_idx: Int32,
    tile_row_idx: Int32,
    q_token_offset: Int32,
) -> Int32:
    """Map a CTA-local Q row to the public output tensor's flat row.

    Fixed output is token-major ``[B, SQ, Hq, D]`` and packed variable-Q output
    is token-major ``[sumQ, Hq, D]``. Resolve the common token/local-head pair
    once so direct, split-GMEM, separate, and cluster final stores share the same
    ABI.
    """
    q_token_idx, local_head_idx = _q_row_token_and_local_head(
        cfg, h_r, q_group_idx, tile_row_idx
    )
    return _q_physical_output_row_from_token_and_local_head(
        cfg,
        h_r,
        num_heads_kv,
        batch_idx,
        kv_head_idx,
        q_token_idx,
        local_head_idx,
        q_token_offset,
    )


@cute.jit
def _q_row_is_valid_for_seq(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    q_group_idx: Int32,
    tile_row_idx: Int32,
    seq_len_q: Int32,
) -> cutlass.Boolean:
    """Return whether a CTA-local Q row is valid for a runtime Q length."""
    return tile_row_idx < _q_tile_valid_rows_for_seq(cfg, h_r, q_group_idx, seq_len_q)


@cute.jit
def _q_row_token_and_local_head(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    q_group_idx: Int32,
    tile_row_idx: Int32,
) -> tuple[Int32, Int32]:
    """Map a valid CTA-local Q row to its token and KV-local Q head."""
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        heads_q_per_kv_value = cfg.heads_q_per_kv
        heads_q_per_kv = Int32(heads_q_per_kv_value)
        if cutlass.const_expr(
            heads_q_per_kv_value > 0
            and (heads_q_per_kv_value & (heads_q_per_kv_value - 1)) == 0
        ):
            # Ratio-32 grouped decode lowers directly to shift/mask. Keep a
            # FastDivmod fallback for future non-power-of-two grouped profiles.
            shift = heads_q_per_kv_value.bit_length() - 1
            token_offset = tile_row_idx >> Int32(shift)
            local_head = tile_row_idx & Int32(heads_q_per_kv_value - 1)
        else:
            heads_q_per_kv_fdd = cute.fast_divmod_create_divisor(heads_q_per_kv)
            token_offset, local_head = divmod(tile_row_idx, heads_q_per_kv_fdd)
        return (
            q_group_idx * Int32(cfg.q_tokens_per_cta) + token_offset,
            local_head,
        )
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        token_idx = q_group_idx // head_ctas_per_token
        head_cta_idx = q_group_idx - token_idx * head_ctas_per_token
        return (
            token_idx,
            head_cta_idx * Int32(cfg.tile_size_q) + tile_row_idx,
        )
    return (
        Int32(0),
        q_group_idx * Int32(cfg.tile_size_q) + tile_row_idx,
    )


@cute.jit
def _attention_sink_head_stride(cfg: Constexpr[FmhaDecodeConfig], h_r: Int32) -> Int32:
    """Return the per-token local-head stride for attention-sink indexing."""
    if cutlass.const_expr(cfg.heads_q_per_kv != 0):
        return Int32(cfg.heads_q_per_kv)
    return h_r


@cute.jit
def _local_head_from_q_output_row(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    output_row_idx: Int32,
) -> Int32:
    """Recover the local Q head from a packed output row."""
    _, local_head_idx = _q_logical_output_row_token_and_local_head(
        cfg, h_r, output_row_idx
    )
    return local_head_idx


class DecodeGenResourceBase(MemoryResource):
    """Base for decode-gen resources.

    Captured schedules let the framework manage variable lifecycle per work
    tile.

    consumer_vars / producer_vars are marked Constexpr so that the @cute.jit
    tracer's tree_flatten does NOT traverse them during dynamic-if
    serialization.  The framework accesses these dicts via
    object.__getattribute__ which bypasses both the Constexpr filter and
    the __getattribute__ guard.
    """

    consumer_vars: Constexpr[dict] = None
    producer_vars: Constexpr[dict] = None
    _task_local_specs: ClassVar[tuple[tuple, ...]] = ()

    def __post_init__(self) -> None:
        """Materialize task-local variables and placeholder resource state."""
        for name, dtype, default, docs in self._task_local_specs:
            object.__setattr__(
                self,
                name,
                TaskLocalVariable(dtype=dtype, default=default, docs=docs),
            )
        self._init_placeholder_state()

    def _init_placeholder_state(self) -> None:
        """Hook for subclasses to install placeholder arrays before tracing."""
        return


# Keeps softmax staging reserves bit 5 for the prepared route kind; the low
# four structural validity bits and bit 4 keep their meaning.
_SOFTMAX_ROUTE_IS_PROXY_FLAG = 1 << 5


@cute.jit
def _route_is_proxy(route_flags: Int32) -> cutlass.Boolean:
    """Return whether staged Softmax route flags mark a proxy route.

    Keeps stages the flags as an ``Int32`` word. SWAP forwards them as a
    bit-preserving ``Uint32`` dataflow token, which the consumer bitcasts
    before calling.
    """
    return cutlass.Boolean(
        (route_flags & Int32(_SOFTMAX_ROUTE_IS_PROXY_FLAG)) != Int32(0)
    )
