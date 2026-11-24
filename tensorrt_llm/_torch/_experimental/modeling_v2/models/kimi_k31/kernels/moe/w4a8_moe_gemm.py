# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts: W4A8 grouped GEMMs in CuTe DSL (SM100 family).

MXFP4 expert weights (E2M1 x UE8M0 per 32) times MXFP8 activations (E4M3 x UE8M0 per 32) in one
``tcgen05.mma.kind::mxf8f6f4`` per 32-deep K step, FP32 accumulation in tensor memory. The weights
are the MMA's A operand (128 output rows per tile), the tokens its B operand (``n_tile`` = 8..64
token rows per tile), so a decode step with one or two rows per expert streams every weight byte
once and wastes no MMA rows on padding tokens.

FC1 and FC2 as two launches or fused into one persistent launch (one class):

* FC1: ``acc = W13[e] @ x[rows of the tile]^T`` over K = 3584, then the SiTU gate
  ``beta * tanh(lin / beta) * alpha * tanh(gate / alpha) * sigmoid(gate)`` of each (linear, gate)
  row pair and the MXFP8 requantization of the 64 outputs of the 128-row tile (two 32-blocks, the
  ``mxfp8_quantize`` arithmetic), stored as permuted rows ``h [perm rows, inter]`` (E4M3) +
  ``h_sf [perm rows, inter / 32]`` (linear UE8M0).
* FC2: ``acc = W2[e] @ h[rows of the tile]^T`` over K = inter, stored as BF16
  permuted rows ``y [perm rows, 3584]`` (the expert outputs before the top-k weighted sum).

The weight tensors are the engine TRTLLM-Gen MXFP4 runner's own (``w3_w1_weight`` [E, 2 * inter,
3584 / 2], ``w2_weight`` [E, 3584, inter / 2], their 128x4-interleaved scales), read in place: their
rows are permuted (gated-act interleave + 32-row shuffle for FC1, 32-row shuffle for FC2), which
only changes which accumulator lane holds which output row; the epilogues undo it.

Routing comes as the TRTLLM-Gen runner's pre-routed layout (what the MoE front / route tail
write): tiles of ``n_tile`` permuted rows, ``tile_expert[t]`` (local expert),
``tile_limit[t]`` (exclusive permuted-row end), ``p2t[p]`` (token of permuted row ``p``) and
``num_tiles[0]``. Work unit = (tile, 128-row weight m-tile), step = one 128-deep K block of a unit;
a stream-K schedule splits each GEMM's steps evenly over one CTA per SM (``num_tiles`` read on the
device, so the launch is CUDA-graph safe); fused, FC2 waits on per-tile FC1 completion counters
(:class:`W4A8MoeKernel`).

Warp roles (224 threads): warps 0-3 epilogue (one TMEM lane each), warp 4 MMA + TMEM allocation,
warp 5 weight TMA (U4 -> U8 unpack into shared memory, as the f8f6f4 MMA reads 4-bit operands),
warp 6 token gather (cp.async of the tile's rows and their scales, the scales into the MMA's
32x4x4 scale-factor atom).
"""

from __future__ import annotations

import functools
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Boolean, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

#: Weight rows per MMA (UMMA M).
MMA_M = 128
#: K extent of one pipeline stage: four 32-deep MMAs; 128 bytes per row of either operand.
KBLK = 128
SF_VEC = 32
#: Bytes of one stage of scale factors (one 32x4x4 atom: 128 rows x 4 scales).
SF_STAGE_BYTES = 512
HIDDEN = 3584
#: Token tiles (UMMA N) supported.
SUPPORTED_N_TILES = (8, 16, 32, 64, 128)

_EPI_WARPS = 4
_MMA_WARP = 4
_TMA_WARP = 5
#: First token-gather warp; a tile of ``n_tile`` rows has ``max(1, n_tile // 32)`` of them (32 rows each).
_GATHER_WARP = 6
#: Named barrier of the epilogue warps.
_EPI_BAR = 1
#: L2 eviction hint for the streamed expert weights (CacheHintSm90::EVICT_FIRST).
_EVICT_FIRST = 0x12F0000000000000
#: Floats per token row of the FC1 epilogue's staging buffer: accumulator row r sits at r + 8 (r // 32), and
#: the odd pitch puts a warp's two token rows on opposite bank parities, so the SiTU job reads are
#: conflict-free (pitch 132 without the shift reads 4-way conflicted).
_EPI_ROW = 153
#: Accumulator columns (tokens) the epilogue loads and processes at a time.
_EPI_CHUNK = 32


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _i32(v, loc, ip):
    return Int32(v).ir_value(loc=loc, ip=ip)


def _i64(v, loc, ip):
    return Int64(v).ir_value(loc=loc, ip=ip)


def _f32(v, loc, ip):
    return Float32(v).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _cp_async_16(dst, src, *, loc=None, ip=None):
    _asm(None, [_i32(dst, loc, ip), _i64(src, loc, ip)], "cp.async.cg.shared.global [$0], [$1], 16;", "r,l",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _cp_async_4(dst, src, *, loc=None, ip=None):
    _asm(None, [_i32(dst, loc, ip), _i64(src, loc, ip)], "cp.async.ca.shared.global [$0], [$1], 4;", "r,l",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _shfl_xor_f32(v, mask, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_f32(v, loc, ip), _i32(mask, loc, ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _tanh_approx(x, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_f32(x, loc, ip)], "tanh.approx.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _ex2_approx(x, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_f32(x, loc, ip)], "ex2.approx.ftz.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _rcp_approx(x, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_f32(x, loc, ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _mxfp8_scale(amax, rcp448, *, loc=None, ip=None):
    """``(ue8m0 byte, 1 / scale)`` of a 32-block with absolute maximum ``amax``: ``mxfp8_quantize``'s
    arithmetic (scale = ``amax * rcp.approx(448)`` rounded up to a power of two; 0 for an all-zero block)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_f32(amax, loc, ip), _f32(rcp448, loc, ip)],
        "{\n"
        ".reg .f32 sf;\n.reg .b16 e;\n.reg .b32 eb, sb;\n.reg .pred pz, pe0, pe254;\n"
        "mul.rn.f32 sf, $2, $3;\n"
        "cvt.rp.satfinite.ue8m0x2.f32 e, sf, sf;\n"
        "cvt.u32.u16 eb, e;\n"
        "and.b32 eb, eb, 255;\n"
        "setp.eq.f32 pz, $2, 0f00000000;\n"
        "setp.eq.u32 pe0, eb, 0;\n"
        "setp.eq.u32 pe254, eb, 254;\n"
        "sub.u32 sb, 254, eb;\n"
        "shl.b32 sb, sb, 23;\n"
        "selp.b32 sb, 0x00400000, sb, pe254;\n"
        "selp.b32 sb, 0x3F800000, sb, pe0;\n"
        "selp.b32 sb, 0, sb, pz;\n"
        "mov.b32 $0, eb;\n"
        "mov.b32 $1, sb;\n"
        "}",
        "=r,=f,f,f",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e4m3x4(a, b, c, d, *, loc=None, ip=None):
    """Four FP32 values as E4M3 (round to nearest even, saturating), ``a`` in the low byte."""
    return Int32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip), _f32(b, loc, ip), _f32(c, loc, ip), _f32(d, loc, ip)],
            "{\n.reg .b16 lo, hi;\n"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
            "mov.b32 $0, {lo, hi};\n}",
            "=r,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2(a, b, *, loc=None, ip=None):
    """Two FP32 values as BF16 (round to nearest even), ``a`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [_f32(a, loc, ip), _f32(b, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _stg_b32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _f32(v, loc, ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_i32(addr, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _sts_b16(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(addr, loc, ip), _i32(v, loc, ip)],
        "{\n.reg .b16 t;\ncvt.u16.u32 t, $1;\nst.shared.b16 [$0], t;\n}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_i32(addr, loc, ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _stg_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [
            _i64(addr, loc, ip),
            _i32(a, loc, ip),
            _i32(b, loc, ip),
            _i32(c, loc, ip),
            _i32(d, loc, ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bar_sync(bar_id, nthreads, *, loc=None, ip=None):
    _asm(
        None,
        [_i32(bar_id, loc, ip), _i32(nthreads, loc, ip)],
        "bar.sync $0, $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _ld_acquire_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_i64(addr, loc, ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_add_release(addr, v, *, loc=None, ip=None):
    _asm(None, [_i64(addr, loc, ip), _i32(v, loc, ip)], "red.release.gpu.global.add.s32 [$0], $1;", "l,r",
         loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def _st_relaxed_u32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i32(v, loc, ip)],
        "st.relaxed.gpu.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_b64(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _i64(v, loc, ip)],
        "st.global.b64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


#: Debug: per-CTA globaltimer stamps into ``dbg`` (8 x int64 per CTA: entry, PDL wait, first weight TMA, first
#: full stage, MMA done, first accumulator, epilogue done, first token copy). Compiled in only when set.
_DEBUG_TIMING = False


def _smem_capacity() -> int:
    major, minor = torch.cuda.get_device_capability()
    return utils.get_smem_capacity_in_bytes(f"sm_{major}{minor}")


@dsl_user_op
def _atom_add_acq_rel(addr, v, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_i64(addr, loc, ip), _i32(v, loc, ip)],
            "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _stg_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_i64(addr, loc, ip), _f32(v, loc, ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_cg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [_i64(addr, loc, ip)], "ld.global.cg.f32 $0, [$1];", "=f,l", loc=loc, ip=ip)
    )


#: int32 words of the workspace header per unit / tile (see :func:`workspace_bytes`).
def workspace_layout(max_tiles: int, inter: int, hidden: int, n_tile: int, grid: int):
    """Byte offsets of the launch workspace: FC1 / FC2 per-unit tail counters, per-tile FC1-done and
    FC2-done counters, then the stream-K partial accumulators ``[2][grid][n_tile * 128]`` FP32. All
    counters are zero between launches (every launch re-arms the ones it used)."""
    u1 = max_tiles * (2 * inter // MMA_M)
    u2 = max_tiles * (hidden // MMA_M)
    off_f2 = 4 * u1
    off_ready = off_f2 + 4 * u2
    off_done2 = off_ready + 4 * max_tiles
    off_parts = (off_done2 + 4 * max_tiles + 127) // 128 * 128
    total = off_parts + 2 * grid * n_tile * MMA_M * 4
    return dict(
        off_f2=off_f2, off_ready=off_ready, off_done2=off_done2, off_parts=off_parts, bytes=total
    )


class W4A8MoeKernel:
    """The K3.1 routed experts' W4A8 GEMMs (see the module docstring): FC1 (SiTU + MXFP8 requant
    epilogue), FC2 (BF16 epilogue), or both in one persistent launch, with a stream-K schedule.

    Schedule: a phase (FC1, then FC2) is ``W = tiles * m-tiles * k-tiles`` k-tile steps in unit-major
    order; CTA ``b`` of ``G`` takes steps ``[b W / G, (b + 1) W / G)``, so every SM streams the same
    number of weight bytes whatever the tile count. A unit split over CTAs is finished by its head
    CTA (the one holding its k-tile 0, which is the last segment of that CTA's range): the other
    parts (each the first segment of its CTA's range) store their FP32 partial accumulator in the
    CTA's workspace slot and release the unit's tail counter; the head waits for the count, adds the
    partials in CTA order (deterministic) and runs the epilogue.

    Fused launch: an FC1 unit's epilogue releases ``ready[tile]`` (+1); an FC2 segment's token warp
    acquires ``ready[tile] == mt1`` before loading the tile's ``h`` rows; the FC2 unit that completes
    a tile's ``mt2``-th unit re-arms ``ready`` / ``done2``. No wait can deadlock: FC1 parts wait only
    on FC1 tails (first in their CTA's range), FC2 waits only on FC1, and the grid is one CTA per SM.

    Args:
        n_tile: token rows per tile (UMMA N), one of :data:`SUPPORTED_N_TILES`.
        inter: intermediate rows per expert on this rank (FC1 has 2 * inter weight rows).
        hidden: the latent width (FC1 K, FC2 rows), 3584.
        fc1 / fc2: which GEMMs this launch runs (both = fused).
        gate_cap / linear_cap: the SiTU soft caps (alpha, beta).
        use_pdl: programmatic dependent launch (wait before reading the routing / tokens, release
            the dependents at the end).
        early_weights: FC2-only launch: read the routing and start streaming the first weights
            before the grid dependency (the routing was written before the previous launch's
            dependency resolved, so it is complete once this grid runs).
        early_release: (with use_pdl, a launch that waits at its start: FC1-only or fused) release
            the dependents right after the grid dependency instead of at the end, so that an
            early-weights FC2 behind it streams its weights while this grid runs (every grid before
            this one has completed by then; the dependent's own wait still covers this grid).
    """

    def __init__(
        self,
        n_tile: int,
        inter: int,
        hidden: int = HIDDEN,
        fc1: bool = True,
        fc2: bool = True,
        gate_cap: float = 4.0,
        linear_cap: float = 25.0,
        use_pdl: bool = True,
        early_weights: bool = False,
        max_stages: int = 16,
        split1: bool = False,
        split2: bool = False,
        p2t_div: int = 1,
        early_release: bool = False,
    ):
        assert n_tile in SUPPORTED_N_TILES, n_tile
        assert fc1 or fc2
        assert (
            (2 * inter) % MMA_M == 0
            and hidden % MMA_M == 0
            and inter % KBLK == 0
            and hidden % KBLK == 0
        )
        self.n_tile = n_tile
        self.inter = inter
        self.hidden = hidden
        self.fc1 = fc1
        self.fc2 = fc2
        self.fused = fc1 and fc2
        self.gate_cap = float(gate_cap)
        self.linear_cap = float(linear_cap)
        self.use_pdl = use_pdl
        self.early_weights = early_weights and fc2 and not fc1
        self.early_release = early_release and use_pdl and not self.early_weights
        # p2t holds tokens (1) or expanded indices token * top_k + slot (top_k, the C++ moe_sort's layout)
        self.p2t_div = int(p2t_div)
        self.mt1 = 2 * inter // MMA_M
        self.kt1 = hidden // KBLK
        self.mt2 = hidden // MMA_M
        self.kt2 = inter // KBLK
        self.mma_tiler = (MMA_M, n_tile, KBLK)
        self.b_stage_bytes = n_tile * KBLK
        stage = MMA_M * KBLK + self.b_stage_bytes + 2 * SF_STAGE_BYTES
        self.gather_warps = max(1, n_tile // 32)
        self.threads = 32 * (_GATHER_WARP + self.gather_warps)
        self.epi_n = min(n_tile, _EPI_CHUNK)
        # epilogue staging: FC1 [epi_n][_EPI_ROW] FP32, FC2 [epi_n][128] BF16 (one buffer, both phases)
        self.epi_floats = max(
            self.epi_n * _EPI_ROW if fc1 else 4, self.epi_n * MMA_M // 2 if fc2 else 4
        )
        reserve = 4096 + self.epi_floats * 4
        self.num_stages = max(2, min(max_stages, (_smem_capacity() - reserve) // stage))
        cols = 2 * n_tile + 32
        self.tmem_cols = 32
        while self.tmem_cols < cols:
            self.tmem_cols *= 2
        self.acc_cols = 2 * n_tile
        # phases this launch runs: (FC1?, m-tiles, k-tiles, stream-K split?)
        self.phases = []
        if fc1:
            self.phases.append((True, self.mt1, self.kt1, bool(split1)))
        if fc2:
            self.phases.append((False, self.mt2, self.kt2, bool(split2)))

    # ------------------------------------------------------------------ host
    def _a_tensors(self, tiled_mma, a_layout, sfa_layout, cl, w_ptr, w_sf_ptr, m, k, num_experts):
        a = cute.make_tensor(w_ptr, cute.make_ordered_layout((m, k, num_experts), order=(1, 0, 2)))
        sfa = cute.make_tensor(w_sf_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC))
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        tma_a, tA = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int8
        )
        tma_sfa, tSFA = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, sfa, sfa_one, self.mma_tiler, tiled_mma, cl.shape, internal_type=cutlass.Int16
        )
        return tma_a, tA, tma_sfa, tSFA

    @cute.jit
    def __call__(
        self,
        w1_ptr: cute.Pointer,
        w1_sf_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w2_sf_ptr: cute.Pointer,
        num_experts: Int32,
        x_addr: Int64,
        x_sf_addr: Int64,
        h_addr: Int64,
        h_sf_addr: Int64,
        y_addr: Int64,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        p2t: cute.Tensor,
        num_tiles: cute.Tensor,
        ws_addr: Int64,
        off_f2: Int32,
        off_ready: Int32,
        off_done2: Int32,
        off_parts: Int32,
        dbg_addr: Int64,
        grid_ctas: Int32,
        stream: cuda.CUstream,
    ):
        S = self.num_stages
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, cutlass.Int8, S)
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, S
        )
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC, S)
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, SF_VEC, S)
        cl = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        # FC2-only launches pass FC2's weights in both slots (the FC1 atoms are never used)
        tma_a1, tA1, tma_sfa1, tSFA1 = self._a_tensors(
            tiled_mma, a_layout, sfa_layout, cl, w1_ptr, w1_sf_ptr,
            2 * self.inter if self.fc1 else self.hidden, self.hidden if self.fc1 else self.inter, num_experts,
        )  # fmt: skip
        tma_a2, tA2, tma_sfa2, tSFA2 = self._a_tensors(
            tiled_mma,
            a_layout,
            sfa_layout,
            cl,
            w2_ptr,
            w2_sf_ptr,
            self.hidden,
            self.inter,
            num_experts,
        )
        a_one = cute.slice_(a_layout, (None, None, None, 0))
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        self.tx_bytes = cute.size_in_bytes(cutlass.Float4E2M1FN, a_one) + cute.size_in_bytes(
            cutlass.Float8E8M0FNU, sfa_one
        )

        @cute.struct
        class SharedStorage:
            full: cute.struct.MemRange[cutlass.Int64, S]
            empty: cute.struct.MemRange[cutlass.Int64, S]
            acc_full: cute.struct.MemRange[cutlass.Int64, 2]
            acc_empty: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(b_layout.outer)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfa_layout)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfb_layout)], 128
            ]
            sEpi: cute.struct.Align[cute.struct.MemRange[Float32, self.epi_floats], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tma_a1,
            tA1,
            tma_sfa1,
            tSFA1,
            tma_a2,
            tA2,
            tma_sfa2,
            tSFA2,
            x_addr,
            x_sf_addr,
            h_addr,
            h_sf_addr,
            y_addr,
            tile_expert,
            tile_limit,
            p2t,
            num_tiles,
            ws_addr,
            off_f2,
            off_ready,
            off_done2,
            off_parts,
            dbg_addr,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            cl,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=[self.threads, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    # ------------------------------------------------------------------ device
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_a1: cute.CopyAtom,
        mA1: cute.Tensor,
        tma_sfa1: cute.CopyAtom,
        mSFA1: cute.Tensor,
        tma_a2: cute.CopyAtom,
        mA2: cute.Tensor,
        tma_sfa2: cute.CopyAtom,
        mSFA2: cute.Tensor,
        x_addr: Int64,
        x_sf_addr: Int64,
        h_addr: Int64,
        h_sf_addr: Int64,
        y_addr: Int64,
        tile_expert: cute.Tensor,
        tile_limit: cute.Tensor,
        p2t: cute.Tensor,
        num_tiles: cute.Tensor,
        ws_addr: Int64,
        off_f2: Int32,
        off_ready: Int32,
        off_done2: Int32,
        off_parts: Int32,
        dbg_addr: Int64,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        cl: cute.Layout,
    ):
        S = self.num_stages
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid_, _, _ = cute.arch.block_idx()
        nctas_, _, _ = cute.arch.grid_dim()
        bid = Int32(bid_)
        nctas = Int32(nctas_)
        dbg = dbg_addr + Int64(bid) * 64
        if cutlass.const_expr(_DEBUG_TIMING):
            if tidx == 0:
                _stg_b64(dbg, cute.arch.globaltimer())

        st = utils.SmemAllocator().allocate(self.shared_storage)
        if warp == _TMA_WARP:
            if cutlass.const_expr(self.fc1):
                cpasync.prefetch_descriptor(tma_a1)
                cpasync.prefetch_descriptor(tma_sfa1)
            if cutlass.const_expr(self.fc2):
                cpasync.prefetch_descriptor(tma_a2)
                cpasync.prefetch_descriptor(tma_sfa2)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(S):
                    # the weight TMA's expect-tx arrival + one cp.async arrival per token-gather lane
                    cute.arch.mbarrier_init(st.full.data_ptr() + i, 1 + 32 * self.gather_warps)
                    cute.arch.mbarrier_init(st.empty.data_ptr() + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(st.acc_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(st.acc_empty.data_ptr() + i, _EPI_WARPS)
                cute.arch.mbarrier_init(st.tmem_ready, 1)
                cute.arch.mbarrier_init(st.tmem_free, _EPI_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if warp == _MMA_WARP:
            cute.arch.alloc_tmem(self.tmem_cols, st.tmem_holding_buf)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(st.tmem_ready)
        if cutlass.const_expr(self.use_pdl and not self.early_weights):
            _griddepcontrol_wait()
            if cutlass.const_expr(self.early_release):
                _griddepcontrol_launch_dependents()
        nt = Int32(num_tiles[0])
        if cutlass.const_expr(_DEBUG_TIMING):
            if tidx == 32 * _TMA_WARP:
                _stg_b64(dbg + 8, cute.arch.globaltimer())
        if warp == _TMA_WARP:
            self._produce_weights(st, tiled_mma, tma_a1, mA1, tma_sfa1, mSFA1, tma_a2, mA2, tma_sfa2, mSFA2,
                                  a_layout, sfa_layout, cl, tile_expert, nt, bid, nctas, dbg)  # fmt: skip
        elif warp >= _GATHER_WARP:
            # an early-weights FC2 waits for the FC1 grid inside, after setting up its first tile's rows
            self._produce_tokens(st, x_addr, x_sf_addr, h_addr, h_sf_addr, tile_limit, p2t,
                                 ws_addr + Int64(off_ready), nt, bid, nctas, warp - _GATHER_WARP, dbg,
                                 self.use_pdl and self.early_weights)  # fmt: skip
        elif warp == _MMA_WARP:
            self._mma(
                st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, nt, bid, nctas, dbg
            )
        elif warp < _EPI_WARPS:
            if cutlass.const_expr(self.use_pdl and self.early_weights):
                _griddepcontrol_wait()
            self._epilogue(st, tiled_mma, tile_limit, h_addr, h_sf_addr, y_addr, ws_addr, off_f2, off_ready,
                           off_done2, off_parts, nt, bid, nctas, tidx, dbg)  # fmt: skip
        if cutlass.const_expr(self.use_pdl and not self.early_release):
            _griddepcontrol_launch_dependents()

    @cute.jit
    def _first(self, nt: Int32, mt: cutlass.Constexpr, kt: cutlass.Constexpr, split: cutlass.Constexpr,
               bid: Int32, nctas: Int32, after: cutlass.Constexpr = False):  # fmt: skip
        """``(cursor, end, W)`` of this CTA's walk over a phase of ``nt * mt`` units: stream-K
        (``split``): steps ``[b W / G, (b + 1) W / G)`` of ``W = units * kt``; else whole units
        ``b, b + G, ...`` (cursor = unit), or with ``after`` (fused FC2) the round-robin continued
        after the FC1 units, so the CTAs without FC1 work take the first FC2 units."""
        w = nt * (mt * kt)
        cur = bid
        if cutlass.const_expr(after):
            cur = (bid + nctas - (nt * self.mt1) % nctas) % nctas
        end = nt * mt
        if cutlass.const_expr(split):
            cur = (bid * w) // nctas
            end = ((bid + 1) * w) // nctas
        return cur, end, w

    @cute.jit
    def _segment(self, cur: Int32, end: Int32, mt: cutlass.Constexpr, kt: cutlass.Constexpr,
                 split: cutlass.Constexpr, nctas: Int32):  # fmt: skip
        """``(unit, tile, m-tile, k begin, k end, next cursor)`` of the segment at ``cur``."""
        u = cur
        ka = Int32(0)
        kb = Int32(kt)
        nxt = cur + nctas
        if cutlass.const_expr(split):
            u = cur // kt
            ka = cur - u * kt
            kb = cutlass.min(Int32(kt), ka + (end - cur))
            nxt = cur + (kb - ka)
        c = u // mt
        mi = u - c * mt
        return u, c, mi, ka, kb, nxt

    # ------------------------------------------------------------------ weight producer
    @cute.jit
    def _a_partition(self, st, tiled_mma, tma_a, mA, tma_sfa, mSFA, a_layout, sfa_layout, cl):
        sA = st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sSFA = st.sSFA.get_tensor(sfa_layout)
        thr_mma = tiled_mma.get_slice(0)
        tiler = cute.slice_(self.mma_tiler, (None, 0, None))
        tCgA = thr_mma.partition_A(cute.local_tile(mA, tiler, (None, None, None)))
        tCgSFA = thr_mma.partition_A(cute.local_tile(mSFA, tiler, (None, None, None)))
        a_cta = cute.make_layout(cute.slice_(cl, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_a, 0, a_cta, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3)
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_sfa, 0, a_cta, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3)
        )
        return tAsA, tAgA, cute.filter_zeros(tAsSFA), cute.filter_zeros(tAgSFA)

    @cute.jit
    def _produce_weights(self, st, tiled_mma, tma_a1, mA1, tma_sfa1, mSFA1, tma_a2, mA2, tma_sfa2, mSFA2,
                         a_layout, sfa_layout, cl, tile_expert, nt: Int32, bid: Int32, nctas: Int32,
                         dbg: Int64):  # fmt: skip
        S = self.num_stages
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        g = Int32(0)
        for ph in cutlass.range_constexpr(len(self.phases)):
            is1, mt, kt, split = self.phases[ph]
            if cutlass.const_expr(is1):
                tAsA, tAgA, tAsSFA, tAgSFA = self._a_partition(
                    st, tiled_mma, tma_a1, mA1, tma_sfa1, mSFA1, a_layout, sfa_layout, cl
                )
            else:
                tAsA, tAgA, tAsSFA, tAgSFA = self._a_partition(
                    st, tiled_mma, tma_a2, mA2, tma_sfa2, mSFA2, a_layout, sfa_layout, cl
                )
            tma_a = tma_a1 if is1 else tma_a2
            tma_sfa = tma_sfa1 if is1 else tma_sfa2
            pos, end, _w = self._first(
                nt, mt, kt, split, bid, nctas, self.fused and not is1 and not split
            )
            # each segment's expert is loaded one segment ahead (the first one together with the tile count)
            last_tile = cute.size(tile_expert) - 1
            _u, c, _mi, _ka, _kb, _nxt = self._segment(pos, end, mt, kt, split, nctas)
            e_next = Int32(tile_expert[cutlass.min(c, last_tile)])
            while pos < end:
                _u, c, mi, ka, kb, nxt = self._segment(pos, end, mt, kt, split, nctas)
                e = e_next
                if cutlass.const_expr(_DEBUG_TIMING):
                    if g == 0:
                        if cute.arch.lane_idx() == 0:
                            _stg_b64(dbg + 16, cute.arch.globaltimer())
                for kk in cutlass.range(ka, kb, 1, unroll=1):
                    s = g % S
                    if g >= S:
                        cute.arch.mbarrier_wait(empty + s, ((g // S) & 1) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + s, self.tx_bytes)
                    cute.copy(tma_a, tAgA[(None, mi, kk, e)], tAsA[(None, s)], tma_bar_ptr=full + s,
                              cache_policy=Int64(_EVICT_FIRST))  # fmt: skip
                    cute.copy(tma_sfa, tAgSFA[(None, mi, kk, e)], tAsSFA[(None, s)], tma_bar_ptr=full + s,
                              cache_policy=Int64(_EVICT_FIRST))  # fmt: skip
                    g += 1
                pos = nxt
                _u, c, _mi, _ka, _kb, _nxt = self._segment(pos, end, mt, kt, split, nctas)
                e_next = Int32(tile_expert[cutlass.min(c, last_tile)])

    # ------------------------------------------------------------------ token gather
    @cute.jit
    def _produce_tokens(self, st, x_addr: Int64, x_sf_addr: Int64, h_addr: Int64, h_sf_addr: Int64, tile_limit,
                        p2t, ready_addr: Int64, nt: Int32, bid: Int32, nctas: Int32, gw: Int32, dbg: Int64,
                        pdl_wait: cutlass.Constexpr = False):  # fmt: skip
        """cp.async of rows ``[32 gw, 32 gw + 32)`` of the tile (128 bytes of one K block per row, 16-byte
        chunks into the 128B-swizzled K-major layout the MMA reads; 8 lanes per row so a warp
        instruction reads 4 whole rows) and of their 4 scales (one 4-byte word per row into the scale
        atom). Rows past the tile's limit are not loaded (their accumulator columns are never stored).
        Each lane's ``cp.async.mbarrier.arrive.noinc`` signals the stage's full barrier once its copies
        landed, so every ring stage can be in flight. FC1 rows are the tile's tokens (``p2t``) of
        ``x``; FC2 rows are the tile's permuted rows of ``h`` (fused: once the tile's FC1 units stored
        them)."""
        S = self.num_stages
        n_tile = self.n_tile
        rows_w = min(32, n_tile)  # rows of this warp
        cpl = rows_w * (KBLK // 16) // 32  # 16-byte chunks per lane and stage
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        lane = cute.arch.lane_idx()
        sb_base = st.sB.data_ptr().toint()
        sfb_base = st.sSFB.data_ptr().toint()
        src = cute.make_rmem_tensor((cpl,), Int64)
        ok = cute.make_rmem_tensor((cpl,), cutlass.Boolean)
        r_first = gw * 32
        g = Int32(0)
        for ph in cutlass.range_constexpr(len(self.phases)):
            is1, mt, kt, split = self.phases[ph]
            base = x_addr if is1 else h_addr
            sf_base = x_sf_addr if is1 else h_sf_addr
            row_bytes = self.hidden if is1 else self.inter
            pos, end, _w = self._first(
                nt, mt, kt, split, bid, nctas, self.fused and not is1 and not split
            )
            # each segment's row sources are set up one segment ahead (the first one together with the tile
            # count, and for an early-weights FC2 before the wait on the FC1 grid)
            last_tile = cute.size(tile_limit) - 1
            _u, c, _mi, _ka, _kb, _nxt = self._segment(pos, end, mt, kt, split, nctas)
            sf_src, sf_ok = self._token_sources(cutlass.min(c, last_tile), is1, base, sf_base, row_bytes, tile_limit,
                                                p2t, lane, r_first, src, ok)  # fmt: skip
            if cutlass.const_expr(pdl_wait and ph == 0):
                _griddepcontrol_wait()
            while pos < end:
                _u, c, _mi, ka, kb, nxt = self._segment(pos, end, mt, kt, split, nctas)
                if cutlass.const_expr(self.fused and not is1):
                    # the tile's h rows are complete once its mt1 FC1 units released the counter
                    if lane == 0:
                        while _ld_acquire_u32(ready_addr + Int64(c) * 4) < self.mt1:
                            pass
                    cute.arch.sync_warp()
                for kk in cutlass.range(ka, kb, 1, unroll=1):
                    s = g % S
                    if g >= S:
                        cute.arch.mbarrier_wait(empty + s, ((g // S) & 1) ^ 1)
                    b_stage = sb_base + s * self.b_stage_bytes
                    for i in cutlass.range_constexpr(cpl):
                        if ok[i]:
                            idx = lane + 32 * i
                            n = r_first + idx // 8
                            ch = idx % 8
                            dst = b_stage + (n // 8) * 1024 + (n % 8) * 128 + ((ch ^ (n % 8)) * 16)
                            _cp_async_16(dst, src[i] + Int64(kk * KBLK))
                    if sf_ok:
                        n = r_first + lane
                        dst = sfb_base + s * SF_STAGE_BYTES + (n % 32) * 16 + (n // 32) * 4
                        _cp_async_4(dst, sf_src + Int64(kk * (KBLK // SF_VEC)))
                    # the stage's full barrier gets one arrival per lane once that lane's copies landed
                    cute.arch.cp_async_mbarrier_arrive_noinc(full + s)
                    if cutlass.const_expr(_DEBUG_TIMING):
                        if g == 0:
                            if lane == 0:
                                if gw == 0:
                                    _stg_b64(dbg + 56, cute.arch.globaltimer())
                    g += 1
                pos = nxt
                _u, c, _mi, _ka, _kb, _nxt = self._segment(pos, end, mt, kt, split, nctas)
                sf_src, sf_ok = self._token_sources(cutlass.min(c, last_tile), is1, base, sf_base, row_bytes,
                                                    tile_limit, p2t, lane, r_first, src, ok)  # fmt: skip

    @cute.jit
    def _token_sources(self, c: Int32, is1: cutlass.Constexpr, base: Int64, sf_base: Int64,
                       row_bytes: cutlass.Constexpr, tile_limit, p2t, lane: Int32, r_first: Int32, src,
                       ok):  # fmt: skip
        """This warp's row sources of tile ``c``: ``src`` / ``ok`` (16-byte chunks per lane, in the tile's
        limit) and the lane's scale-row source and predicate (returned). p2t is read for every row of
        the tile (in bounds: the tile's rows are allocated) without waiting for the limit, so the two
        loads overlap."""
        n_tile = self.n_tile
        rows_w = min(32, n_tile)
        cpl = rows_w * (KBLK // 16) // 32
        row0 = c * n_tile
        lim = Int32(tile_limit[c])
        for i in cutlass.range_constexpr(cpl):
            idx = lane + 32 * i
            p = row0 + r_first + idx // 8
            tok = p
            if cutlass.const_expr(is1):
                tok = Int32(p2t[p]) // self.p2t_div
            src[i] = base + Int64(tok) * row_bytes + Int64((idx % 8) * 16)
        p = row0 + r_first + lane
        tok = p
        if cutlass.const_expr(is1):
            tok = Int32(p2t[cutlass.min(p, row0 + n_tile - 1)]) // self.p2t_div
        sf_src = sf_base + Int64(tok) * (row_bytes // SF_VEC)
        for i in cutlass.range_constexpr(cpl):
            ok[i] = row0 + r_first + (lane + 32 * i) // 8 < lim
        sf_ok = (lane < rows_w) & (p < lim)
        return sf_src, sf_ok

    # ------------------------------------------------------------------ MMA
    @cute.jit
    def _mma(self, st, tiled_mma, a_layout, b_layout, sfa_layout, sfb_layout, nt: Int32, bid: Int32,
             nctas: Int32, dbg: Int64):  # fmt: skip
        S = self.num_stages
        full, empty = st.full.data_ptr(), st.empty.data_ptr()
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        cute.arch.mbarrier_wait(st.tmem_ready, 0)
        acc_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tAcc_base = cute.make_tensor(
            acc_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, 2)).layout
        )
        sfa_one = cute.slice_(sfa_layout, (None, None, None, 0))
        sfb_one = cute.slice_(sfb_layout, (None, None, None, 0))
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(acc_ptr + self.acc_cols, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, SF_VEC, sfa_one),
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(acc_ptr + self.acc_cols + 16, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, SF_VEC, sfb_one),
        )
        s2t_sfa, tCsSFA, tCtSFA_s2t = _s2t_copy_and_partition(
            st.sSFA.get_tensor(sfa_layout), tCtSFA
        )
        s2t_sfb, tCsSFB, tCtSFB_s2t = _s2t_copy_and_partition(
            st.sSFB.get_tensor(sfb_layout), tCtSFB
        )
        tCrA = tiled_mma.make_fragment_A(st.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner))
        tCrB = tiled_mma.make_fragment_B(st.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner))
        nkp = cute.size(tCrA, mode=[2])

        g = Int32(0)
        j = Int32(0)
        for ph in cutlass.range_constexpr(len(self.phases)):
            is1, mt, kt, split = self.phases[ph]
            pos, end, _w = self._first(
                nt, mt, kt, split, bid, nctas, self.fused and not is1 and not split
            )
            while pos < end:
                _u, _c, _mi, ka, kb, nxt = self._segment(pos, end, mt, kt, split, nctas)
                a = j % 2
                if j >= 2:
                    cute.arch.mbarrier_wait(acc_empty + a, ((j // 2) & 1) ^ 1)
                tAcc = tAcc_base[(None, None, None, a)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _kk in cutlass.range(ka, kb, 1, unroll=1):
                    s = g % S
                    cute.arch.mbarrier_wait(full + s, (g // S) & 1)
                    if cutlass.const_expr(_DEBUG_TIMING):
                        if g == 0:
                            if cute.arch.lane_idx() == 0:
                                _stg_b64(dbg + 24, cute.arch.globaltimer())
                    # the token rows came through cp.async (generic proxy): order them before the MMA reads
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.copy(s2t_sfa, tCsSFA[(None, None, None, None, s)], tCtSFA_s2t)
                    cute.copy(s2t_sfb, tCsSFB[(None, None, None, None, s)], tCtSFB_s2t)
                    for kp in cutlass.range_constexpr(nkp):
                        tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                        tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                        cute.gemm(
                            tiled_mma,
                            tAcc[(None, 0, 0)],
                            tCrA[(None, 0, kp, s)],
                            tCrB[(None, 0, kp, s)],
                            tAcc[(None, 0, 0)],
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + s)
                    g += 1
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + a)
                j += 1
                pos = nxt
        if cutlass.const_expr(_DEBUG_TIMING):
            if cute.arch.lane_idx() == 0:
                _stg_b64(dbg + 32, cute.arch.globaltimer())
        cute.arch.mbarrier_wait(st.tmem_free, 0)
        cute.arch.dealloc_tmem(acc_ptr, self.tmem_cols)

    # ------------------------------------------------------------------ epilogue
    @cute.jit
    def _epilogue(self, st, tiled_mma, tile_limit, h_addr: Int64, h_sf_addr: Int64, y_addr: Int64, ws_addr: Int64,
                  off_f2: Int32, off_ready: Int32, off_done2: Int32, off_parts: Int32, nt: Int32, bid: Int32,
                  nctas: Int32, tidx, dbg: Int64):  # fmt: skip
        """Per segment: the accumulator in chunks of ``epi_n`` token columns (TMEM -> registers), then a
        stream-K tail stores its FP32 partial, a split head adds the tails' partials, and a finished
        unit runs the FC1 (SiTU + MXFP8) or FC2 (BF16 rows) epilogue on each chunk."""
        n_tile = self.n_tile
        epi_n = self.epi_n
        n_chunks = n_tile // epi_n
        acc_full, acc_empty = st.acc_full.data_ptr(), st.acc_empty.data_ptr()
        cute.arch.mbarrier_wait(st.tmem_ready, 0)
        acc_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=st.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tAcc_base = cute.make_tensor(
            acc_ptr, tiled_mma.make_fragment_C(cute.append(acc_shape, 2)).layout
        )
        copy_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(epi_n)), Float32)
        tAcc_epi = cute.flat_divide(tAcc_base[((None, None), 0, 0, None)], (MMA_M, epi_n))
        tiled_t2r = tcgen05.make_tmem_copy(copy_atom, tAcc_epi[(None, None, 0, 0, 0)])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
        tTR_cAcc = thr_t2r.partition_D(cute.make_identity_tensor((MMA_M, epi_n)))
        rAcc = cute.make_rmem_tensor(tTR_cAcc.shape, Float32)
        epi_base = st.sEpi.data_ptr().toint()
        rcp448 = _rcp_approx(Float32(448.0))
        warp = tidx // 32
        lane = tidx % 32
        hidden_col = 32 * warp + 4 * (lane % 8) + lane // 8
        ready_addr = ws_addr + Int64(off_ready)
        done2_addr = ws_addr + Int64(off_done2)
        slot_bytes = n_tile * MMA_M * 4

        j = Int32(0)
        for ph in cutlass.range_constexpr(len(self.phases)):
            is1, mt, kt, split = self.phases[ph]
            flag_addr = ws_addr if is1 else ws_addr + Int64(off_f2)
            parts = ws_addr + Int64(off_parts) + Int64(ph * slot_bytes) * Int64(nctas)
            pos, end, w = self._first(
                nt, mt, kt, split, bid, nctas, self.fused and not is1 and not split
            )
            while pos < end:
                u, c, mi, ka, kb, nxt = self._segment(pos, end, mt, kt, split, nctas)
                row0 = c * n_tile
                lim = Int32(tile_limit[c])
                a = j % 2
                cute.arch.mbarrier_wait(acc_full + a, (j // 2) & 1)
                if cutlass.const_expr(_DEBUG_TIMING):
                    if j == 0:
                        if tidx == 0:
                            _stg_b64(dbg + 40, cute.arch.globaltimer())
                last_cta = bid
                if cutlass.const_expr(split):
                    if ka == 0:
                        if kb < kt:
                            # the head of a split unit: wait for the tails (CTAs bid + 1 .. of the unit's
                            # last step), then add their partials in CTA order
                            last = u * kt + (kt - 1)
                            last_cta = ((last + 1) * nctas + w - 1) // w - 1
                            if tidx == 0:
                                while _ld_acquire_u32(flag_addr + Int64(u) * 4) < last_cta - bid:
                                    pass
                                _st_relaxed_u32(flag_addr + Int64(u) * 4, 0)
                            _bar_sync(_EPI_BAR, 128)
                for ch in cutlass.range_constexpr(n_chunks):
                    cute.copy(tiled_t2r, tTR_tAcc[(None, None, None, 0, ch, a)], rAcc)
                    if cutlass.const_expr(ch == n_chunks - 1):
                        cute.arch.fence_view_async_tmem_load()
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(acc_empty + a)
                    n0 = ch * epi_n
                    is_tail = Boolean(False)
                    if cutlass.const_expr(split):
                        is_tail = ka > 0
                    if is_tail:
                        # a tail part: the FP32 partial into this CTA's slot ([n][row])
                        slot = parts + Int64(bid) * slot_bytes
                        for i in cutlass.range_constexpr(cute.size(rAcc)):
                            r, n = tTR_cAcc[i]
                            _stg_f32(slot + ((n0 + n) * MMA_M + r) * 4, rAcc[i])
                    else:
                        if cutlass.const_expr(split):
                            for q in cutlass.range(bid + 1, last_cta + 1, 1, unroll=1):
                                slot = parts + Int64(q) * slot_bytes
                                for i in cutlass.range_constexpr(cute.size(rAcc)):
                                    r, n = tTR_cAcc[i]
                                    rAcc[i] = rAcc[i] + _ldg_cg_f32(
                                        slot + ((n0 + n) * MMA_M + r) * 4
                                    )
                        if cutlass.const_expr(is1):
                            self._epi_fc1(rAcc, tTR_cAcc, epi_base, rcp448, h_addr, h_sf_addr, row0 + n0, lim,
                                          mi, tidx)  # fmt: skip
                        else:
                            self._epi_fc2(
                                rAcc,
                                tTR_cAcc,
                                epi_base,
                                hidden_col,
                                y_addr,
                                row0 + n0,
                                lim,
                                mi,
                                tidx,
                            )
                if cutlass.const_expr(split):
                    if ka > 0:
                        _bar_sync(_EPI_BAR, 128)
                        if tidx == 0:
                            _red_add_release(flag_addr + Int64(u) * 4, 1)
                if cutlass.const_expr(self.fused):
                    finished = Boolean(True)
                    if cutlass.const_expr(split):
                        finished = ka == 0
                    if finished:
                        if cutlass.const_expr(is1):
                            # every thread's stores of the unit precede the release (the CTA barrier
                            # that ends _epi_fc1)
                            if tidx == 0:
                                _red_add_release(ready_addr + Int64(c) * 4, 1)
                        else:
                            # the tile's last FC2 unit re-arms its FC1-done and FC2-done counters (every
                            # FC2 segment of the tile passed its wait before its unit completed)
                            if tidx == 0:
                                old = _atom_add_acq_rel(done2_addr + Int64(c) * 4, 1)
                                if old == self.mt2 - 1:
                                    _st_relaxed_u32(done2_addr + Int64(c) * 4, 0)
                                    _st_relaxed_u32(ready_addr + Int64(c) * 4, 0)
                j += 1
                pos = nxt
        if cutlass.const_expr(_DEBUG_TIMING):
            if tidx == 0:
                _stg_b64(dbg + 48, cute.arch.globaltimer())
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(st.tmem_free)

    @cute.jit
    def _epi_fc1(
        self, rAcc, tTR_cAcc, epi_base, rcp448, h_addr: Int64, h_sf_addr: Int64, row0, lim, mi, tidx
    ):
        """SiTU of the tile's 64 (linear, gate) row pairs + MXFP8 requant for ``epi_n`` token columns,
        through a token-major staging buffer: job = (token n, quad s) handles outputs 4 s .. 4 s + 3
        (pair p of 32-row block w: linear in stored lane (p >> 1) + 16 (p & 1), gate 8 lanes above).
        ``row0`` is the chunk's first permuted row. Ends with an epilogue-warp barrier."""
        epi_n = self.epi_n
        for i in cutlass.range_constexpr(cute.size(rAcc)):
            r, n = tTR_cAcc[i]
            _sts_f32(epi_base + (n * _EPI_ROW + r + 8 * (r // 32)) * 4, rAcc[i])
        _bar_sync(_EPI_BAR, 128)
        for rd in cutlass.range_constexpr(epi_n * 16 // 128):
            job = tidx + 128 * rd
            n = job // 16
            s = job % 16
            v = cute.make_rmem_tensor((4,), Float32)
            amax = Float32(0.0)
            for q4 in cutlass.range_constexpr(4):
                jl = 4 * s + q4
                w = jl // 16
                p = jl % 16
                r_lin = (
                    40 * w + (p >> 1) + ((p & 1) << 4)
                )  # stored lane of the pair's linear row, shifted
                lin = _lds_f32(epi_base + (n * _EPI_ROW + r_lin) * 4)
                gate = _lds_f32(epi_base + (n * _EPI_ROW + r_lin + 8) * 4)
                val = self._situ(gate, lin)
                v[q4] = val
                amax = cute.arch.fmax(amax, cute.math.absf(val))
            for m in cutlass.range_constexpr(3):
                amax = cute.arch.fmax(amax, _shfl_xor_f32(amax, 1 << m))
            sf, sc = _mxfp8_scale(amax, rcp448)
            packed = _e4m3x4(v[0] * sc, v[1] * sc, v[2] * sc, v[3] * sc)
            p = row0 + n
            if p < lim:
                _stg_b32(h_addr + Int64(p) * self.inter + mi * 64 + 4 * s, packed)
                if s % 8 == 0:
                    _stg_u8(h_sf_addr + Int64(p) * (self.inter // SF_VEC) + mi * 2 + s // 8, sf)
        _bar_sync(_EPI_BAR, 128)

    @cute.jit
    def _epi_fc2(self, rAcc, tTR_cAcc, epi_base, hidden_col, y_addr: Int64, row0, lim, mi, tidx):
        """BF16 rows of ``epi_n`` token columns: staged token-major at their output positions
        (``[n][128]``), then 16-byte stores (16 per token row of the m-tile). ``row0`` is the chunk's
        first permuted row. Ends with an epilogue-warp barrier."""
        epi_n = self.epi_n
        for i in cutlass.range_constexpr(cute.size(rAcc)):
            _r, n = tTR_cAcc[i]
            _sts_b16(epi_base + (n * MMA_M + hidden_col) * 2, _bf16x2(rAcc[i], Float32(0.0)))
        _bar_sync(_EPI_BAR, 128)
        for rd in cutlass.range_constexpr(epi_n * 16 // 128):
            q = tidx + 128 * rd
            n = q // 16
            c16 = q % 16
            p = row0 + n
            if p < lim:
                v0, v1, v2, v3 = _lds_v4(epi_base + (n * MMA_M + 8 * c16) * 2)
                _stg_v4(
                    y_addr + (Int64(p) * self.hidden + mi * MMA_M + 8 * c16) * 2, v0, v1, v2, v3
                )
        _bar_sync(_EPI_BAR, 128)

    @cute.jit
    def _situ(self, gate: Float32, lin: Float32) -> Float32:
        """``beta tanh(lin / beta) * alpha tanh(gate / alpha) sigmoid(gate)`` (TRTLLM-Gen SiTuGlu,
        gate on x1)."""
        alpha = Float32(self.gate_cap)
        beta = Float32(self.linear_cap)
        left = beta * _tanh_approx(lin * Float32(1.0 / self.linear_cap))
        sig = _rcp_approx(Float32(1.0) + _ex2_approx(gate * Float32(-1.4426950408889634)))
        right = alpha * _tanh_approx(gate * Float32(1.0 / self.gate_cap)) * sig
        return left * right


def _s2t_copy_and_partition(sSF: cute.Tensor, tSF: cute.Tensor):
    """tcgen05.cp (SMEM -> TMEM) partitions of a scale-factor operand."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
    tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t


# ---------------------------------------------------------------------- launch
def num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _i32_arg(t: torch.Tensor):
    return from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic()


def _fp4_ptr(t: torch.Tensor):
    return make_ptr(cutlass.Float4E2M1FN, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)


def _sf_ptr(t: torch.Tensor):
    return make_ptr(cutlass.Float8E8M0FNU, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)


@functools.cache
def _compiled(n_tile: int, inter: int, hidden: int, fc1: bool, fc2: bool, gate_cap: float, linear_cap: float,
              use_pdl: bool, early_weights: bool, max_stages: int, split1: bool, split2: bool, p2t_div: int,
              _debug: bool, early_release: bool = False):  # fmt: skip
    kernel = W4A8MoeKernel(n_tile, inter, hidden, fc1, fc2, gate_cap, linear_cap, use_pdl, early_weights,
                              max_stages, split1, split2, p2t_div, early_release)  # fmt: skip
    dev = torch.device("cuda")
    i32 = torch.zeros(64, dtype=torch.int32, device=dev)
    dummy = make_ptr(cutlass.Float4E2M1FN, 1 << 20, cute.AddressSpace.gmem, assumed_align=32)
    dummy_sf = make_ptr(cutlass.Float8E8M0FNU, 1 << 20, cute.AddressSpace.gmem, assumed_align=32)
    a = Int64(1 << 20)
    o = Int32(0)
    return cute.compile(
        kernel,
        dummy,
        dummy_sf,
        dummy,
        dummy_sf,
        Int32(1),
        a,
        a,
        a,
        a,
        a,
        _i32_arg(i32),
        _i32_arg(i32),
        _i32_arg(i32),
        _i32_arg(i32[:1]),
        a,
        o,
        o,
        o,
        o,
        a,
        Int32(1),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def workspace_bytes(
    max_tiles: int, inter: int, hidden: int, n_tile: int, grid: Optional[int] = None
) -> int:
    """Bytes of the zero-initialized workspace :func:`run_w4a8_moe` needs."""
    return workspace_layout(max_tiles, inter, hidden, n_tile, grid or num_sms())["bytes"]


def run_w4a8_moe(
    w1: Optional[torch.Tensor],
    w1_sf: Optional[torch.Tensor],
    w2: Optional[torch.Tensor],
    w2_sf: Optional[torch.Tensor],
    x: Optional[torch.Tensor],
    x_sf: Optional[torch.Tensor],
    h: torch.Tensor,
    h_sf: torch.Tensor,
    y: Optional[torch.Tensor],
    tile_expert: torch.Tensor,
    tile_limit: torch.Tensor,
    p2t: torch.Tensor,
    num_tiles: torch.Tensor,
    workspace: torch.Tensor,
    n_tile: int,
    max_tiles: int,
    gate_cap: float = 4.0,
    linear_cap: float = 25.0,
    use_pdl: bool = True,
    early_weights: bool = False,
    max_stages: int = 16,
    split1: bool = False,
    split2: bool = False,
    p2t_div: int = 1,
    dbg: Optional[torch.Tensor] = None,
    early_release: bool = False,
) -> None:
    """Launch FC1 (``w1`` given), FC2 (``w2`` given) or both fused, on the current stream (inputs not
    validated). ``early_release``: see :class:`W4A8MoeKernel`.

    ``w1`` uint8 ``[E, 2 * inter, hidden / 2]`` / ``w2`` ``[E, hidden, inter / 2]`` (packed E2M1, the
    runner's row order) with their interleaved UE8M0 scales; ``x`` E4M3 ``[tokens, hidden]`` +
    ``x_sf`` uint8 ``[tokens, hidden / 32]`` (FC1's input, gathered through ``p2t``); ``h`` E4M3
    ``[perm rows, inter]`` + ``h_sf`` ``[perm rows, inter / 32]`` (FC1's output, FC2's input); ``y``
    BF16 ``[perm rows, hidden]`` (FC2's output). ``workspace``: :func:`workspace_bytes` zeroed bytes
    (kept zero by every launch). ``max_tiles`` bounds ``num_tiles[0]``. ``p2t`` holds the token of each
    permuted row, or (``p2t_div = top_k``) its expanded index ``token * top_k + slot``."""
    fc1 = w1 is not None
    fc2 = w2 is not None
    ref = w1 if fc1 else w2
    num_experts = ref.shape[0]
    if fc1:
        inter, hidden = w1.shape[1] // 2, 2 * w1.shape[2]
    else:
        hidden, inter = w2.shape[1], 2 * w2.shape[2]
    compiled = _compiled(int(n_tile), int(inter), int(hidden), fc1, fc2, float(gate_cap), float(linear_cap),
                         bool(use_pdl), bool(early_weights), int(max_stages), bool(split1), bool(split2),
                         int(p2t_div), _DEBUG_TIMING, bool(early_release))  # fmt: skip
    grid = num_sms()
    lay = workspace_layout(int(max_tiles), int(inter), int(hidden), int(n_tile), grid)
    assert workspace.numel() * workspace.element_size() >= lay["bytes"]
    wa, wsa = (w1, w1_sf) if fc1 else (w2, w2_sf)
    wb, wsb = (w2, w2_sf) if fc2 else (w1, w1_sf)
    compiled(
        _fp4_ptr(wa),
        _sf_ptr(wsa),
        _fp4_ptr(wb),
        _sf_ptr(wsb),
        Int32(num_experts),
        Int64(x.data_ptr() if x is not None else 0),
        Int64(x_sf.data_ptr() if x_sf is not None else 0),
        Int64(h.data_ptr()),
        Int64(h_sf.data_ptr()),
        Int64(y.data_ptr() if y is not None else 0),
        _i32_arg(tile_expert),
        _i32_arg(tile_limit),
        _i32_arg(p2t),
        _i32_arg(num_tiles),
        Int64(workspace.data_ptr()),
        Int32(lay["off_f2"]),
        Int32(lay["off_ready"]),
        Int32(lay["off_done2"]),
        Int32(lay["off_parts"]),
        Int64(dbg.data_ptr() if dbg is not None else 0),
        Int32(grid),
        cuda.CUstream(torch.cuda.current_stream(ref.device).cuda_stream),
    )


__all__ = [
    "HIDDEN",
    "KBLK",
    "MMA_M",
    "SUPPORTED_N_TILES",
    "W4A8MoeKernel",
    "num_sms",
    "run_w4a8_moe",
    "workspace_bytes",
    "workspace_layout",
]
