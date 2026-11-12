# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 FP4 MLA decode query in one kernel: q_a RMSNorm, q_b projection, W_UK absorption,
RoPE and the FP4 query quantization of the packed FP4 MLA decode (SM107).

For ``T <= 16`` decode tokens and the 96 MLA heads of a Helix rank, :class:`K3MlaQFusedKernel`
turns the q_a latent ``x [T, 1536]`` (BF16, raw or already normalized) into the prequantized
query the packed FP4 MLA decode reads (``_fp4_mla_decode.run_decode``): per (token,
head) row 512 absorbed latent values and 64 RoPE values with their residual, E2M1 codes with one
E4M3 scale per 16 values. It replaces, with the same arithmetic up to the summation order of the
q_b GEMM (the MMA hardware's internal sum of a 32-value group may differ):

0. (``fuse_norm``) the q_a RMSNorm: FlashInfer's ``RMSNormKernel`` operations in its own
   per-lane layout (:meth:`K3MlaQFusedKernel.row_rstd`), bitwise its output.
1. ``q = q_b_proj(x)``: the low-M FP8 GEMM (``k3_fp8w_lowm_gemm``) with its in-kernel MXFP8
   activation quantization (E4M3 codes, one UE8M0 exponent per 32 values, bitwise the same
   helpers) and the FP8 weight's 128x128 UE8M0 block scales; the product of a 32-value group is
   scaled by ``2^(clamp(e_act + e_w - 127, 0, 254) - 127)`` as the GEMM's block-scaled MMA does,
   accumulated in FP32 in K order per K half, the halves added (the GEMM's split-K partials),
   rounded to BF16.
2. ``q_nope @ W_UK[h]^T`` with the BF16 absorption weight: ``mma.sync`` K16 steps in K order,
   FP32 accumulation, rounded to BF16 (bitwise the BF16 BMM in the tests).
3. The fused FP4 generation cache update's query half
   (``_fp4_mla_generation_fused_qk_rope_cache_update_kernel``): GPT-J RoPE of ``q_pe`` from the
   rotary table at the token's position, rounded to BF16, then the global-scaled FP4
   quantization of the 512 absorbed values and the main + residual FP4 quantization of the 64
   RoPE values, in the packed per-request row layout with swizzled scales. This part repeats
   the Triton kernel's operations (``div.full`` / ``rcp.approx`` / the contracted residual), so
   given the same BF16 values it writes the same bytes.

Work split: a cluster of two CTAs per head (192 CTAs, 12 warps each). CTA ``r`` of head ``h``
owns q_b rows ``h * 192 + 96 r .. + 95`` (CTA 0: q_nope dims 0..95; CTA 1: q_nope dims 96..127
and the 64 RoPE dims) and absorbed columns ``256 r .. 256 r + 255``:

* Before ``griddepcontrol.wait`` (nothing read there depends on the previous grid): the CTA's
  q_b rows (144 KB) are issued as six TMA boxes of 16 whole rows (contiguous in global memory)
  that land as ``[k block][row][128 bytes]`` with the 128-byte swizzle (conflict-free
  ``ldmatrix``).
* After the wait warp ``w`` loads rows ``w, w + 12, ..`` of ``x`` (32 lanes per row), normalizes
  them (``fuse_norm``) and quantizes them to MXFP8 in shared memory. Warp ``w`` then runs K half
  ``w // 6`` of the 16-row q_b tile ``w % 6`` with ``mma.sync`` m16n8k32 E4M3 (block scales
  applied per 32-value group with one FMA); the second half's sums go to the first half's warp,
  which adds them and writes q_nope (BF16) into both CTAs' shared memory (DSMEM) and q_pe
  locally. CTA 1's last warp then prefetches the positions and rotary values; the CTA's W_UK rows
  are prefetched into L2 after the activation phase, and once tiles 0..2 are done warp 0 loads
  W_UK (64 KB, one box) into their space. After a cluster barrier warps 0..7 run two 16-column
  W_UK tiles each with ``mma.sync`` m16n8k16 BF16.
* The absorbed BF16 values are staged per token and quantized 16 at a time; CTA 1 also rotates
  and quantizes q_pe. The FP4 rows and scales go straight to the decode's Q buffers.

:class:`K3MlaLatentRowKernel` is the KV half's companion: the kv_a RMSNorm and the ``k_pe`` copy
into the latent row in one launch.
"""

from __future__ import annotations

import functools
import operator
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import cuda as cuda_tma

from ..k3r_gemm.k3_fp8w_lowm_gemm import (
    _asm,
    _bf16x2_inv_scale,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _iv,
    _lds_v4_b32,
    _lds_v4_f32,
    _quant_bf16x8_to_e4m3,
    _sts_v2_b32,
    _sts_v4_b32,
    _sts_v4_f32,
    _ue8m0_of_amax,
)
from ..k3r_gemm.k3_fp8w_lowm_mxfp8 import _amax_bf16x8_words, _amax_pair
from ..k3r_helix.helix_a2a_fused import _fv, _lv, _pack_bf16x2, _unpack
from ..k3r_helix.k3_helix_tail import (
    _globaltimer,
    _mbar_expect_tx,
    _mbar_init,
    _mbar_try_wait,
    _mma_bf16,
    _shfl_bfly_b32,
    _st_u64,
)

try:
    from flashinfer.norm.utils import warp_reduce as _flashinfer_warp_reduce

    NORM_FUSION_AVAILABLE = True
except ImportError:  # pragma: no cover - FlashInfer without the CuTe DSL norm kernels
    _flashinfer_warp_reduce = None
    NORM_FUSION_AVAILABLE = False

#: MLA geometry the kernel is built for (Kimi K3).
NUM_HEADS = 96
NOPE_DIM = 128
ROPE_DIM = 64
QK_HEAD_DIM = NOPE_DIM + ROPE_DIM
KV_LORA_RANK = 512
Q_LORA_RANK = 1536
#: Layout of the FP4 query rows the packed FP4 MLA decode reads (``fp4_mla``): 512 absorbed
#: values, then the 64 RoPE values and their 64 residuals, 16 values per E4M3 scale.
FP4_GROUP = 16
Q_ROW_BYTES = (KV_LORA_RANK + 2 * ROPE_DIM) // 2
Q_SF_COLS = (KV_LORA_RANK + 2 * ROPE_DIM) // FP4_GROUP
#: Largest token count of one launch (the activation rows and their scales must fit shared
#: memory beside the resident q_b weight).
MAX_TOKENS = 16

_THREADS = 384
_WARPS = _THREADS // 32
#: Warps of the W_UK absorption (two 16-column tiles each).
_ABSORB_WARPS = 8
#: q_b weight rows per CTA (half a head) and absorbed columns per CTA.
_CTA_ROWS = QK_HEAD_DIM // 2
_CTA_COLS = KV_LORA_RANK // 2
#: Weights arrive by TMA as 3D boxes whose shared image is ``[k block][row][128 bytes]`` with the
#: 128-byte swizzle (16-byte chunk ``c`` of row ``r`` lands at chunk ``c ^ (r % 8)``), so the
#: eight rows an ``ldmatrix`` 8x8 matrix reads hit distinct banks, while each box covers whole
#: weight rows (a contiguous range of global memory). One box per 16-row q_b tile (24 KB, all of
#: K); W_UK of the CTA's 256 absorbed columns is one box of two 64-wide halves (64 KB).
_BOX_BYTES = 128
_KB = Q_LORA_RANK // _BOX_BYTES
_TILE_ROWS = 16
_NTILE = _CTA_ROWS // _TILE_ROWS
_TILE_BYTES = _TILE_ROWS * Q_LORA_RANK
_UK_HALF = _CTA_COLS * _BOX_BYTES
_UK_BYTES = 2 * _UK_HALF
#: Padded shared row strides of the buffers the kernel writes itself (16 bytes past a multiple
#: of 128 keep ``ldmatrix`` rows in distinct banks).
_A_PITCH = Q_LORA_RANK + 16
_QN_PITCH = NOPE_DIM * 2 + 16
_QPE_PITCH = ROPE_DIM * 2
_OB_PITCH = _CTA_COLS * 2 + 16
_KGROUPS = Q_LORA_RANK // 32
#: q_b is summed as this many K slices added at the end, the association of the low-M GEMM's
#: split-K partials at decode token counts (its SM107 tactic for this shape uses split 2 at
#: M <= 16).
_QB_SPLIT_K = 2
_KBLOCKS = Q_LORA_RANK // 128
#: W_UK reuses the shared memory of the first q_b chunks once every MMA warp is done with them;
#: the absorbed output is staged where the last chunks were.
_UK_TILES = -(-_UK_BYTES // _TILE_BYTES)
_OB_OFFSET = _UK_TILES * _TILE_BYTES
_SMEM_LIMIT = 227 * 1024
#: Bytes of fixed shared buffers (barriers, block exponents) ahead of the data regions.
_FIXED_BYTES = 256

#: ``2^-9``, the floor of the FP4 query scales (``_fp4_mla_quantize_q_groups``).
_Q_SCALE_FLOOR = 0x3B000000


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def smem_bytes(t_pad: int) -> int:
    """Shared memory of a launch over ``t_pad`` tokens."""
    return (
        _FIXED_BYTES
        + 2 * _KGROUPS * t_pad * 4
        + t_pad * _A_PITCH
        + _NTILE * _TILE_BYTES
        + t_pad * _QN_PITCH
        + t_pad * _QPE_PITCH
        + Q_LORA_RANK * 2
        + t_pad * (4 * 64 + 4)
        + _NTILE * 32 * t_pad // 2 * 4
        + 2 * 1024
        + 8 * 128  # region alignment
    )


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _ldsm_x4(addr, *, loc=None, ip=None):
    """``ldmatrix`` of four 8x8 b16 matrices (lane ``l`` addresses a row of matrix ``l // 8``)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ldsm_x2(addr, *, loc=None, ip=None):
    """``ldmatrix`` of two 8x8 b16 matrices (lanes 0..15 address the rows)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _mma_e4m3(a0, a1, a2, a3, b0, b1, *, loc=None, ip=None):
    """``d[16x8] = a[16x32] @ b[32x8]`` (E4M3 inputs, FP32 result, zero accumulator)."""
    zero = Float32(0.0)
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(x, loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [_fv(zero, loc=loc, ip=ip) for _ in range(4)],
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _block_scale(e_act, w_exp, *, loc=None, ip=None):
    """FP32 ``2^(clamp(e_act + w_exp - 127, 0, 254) - 127)``: the block-scaled MMA's scale of a
    32-value group with the weight exponent folded into the activation's, as the low-M GEMM
    stores it (0 means ``2^-127``)."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(e_act, loc=loc, ip=ip), _iv(w_exp, loc=loc, ip=ip)],
            "{\n.reg .b32 e, b;\n.reg .pred p;\n"
            "add.s32 e, $1, $2;\nsub.s32 e, e, 127;\nmax.s32 e, e, 0;\nmin.s32 e, e, 254;\n"
            "shl.b32 b, e, 23;\nsetp.eq.s32 p, e, 0;\n@p mov.b32 b, 0x00400000;\n"
            "mov.b32 $0, b;\n}",
            "=f,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_v2_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip) for x in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_u8(addr, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.u8 $0, [$1];", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _sts_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.shared.u8 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_bf16(addr, value, *, loc=None, ip=None):
    """Store ``value`` rounded to BF16 (nearest even) at shared ``addr``."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.shared.b16 [$0], h;\n}",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_bf16_both(addr, peer, value, *, loc=None, ip=None):
    """Store ``value`` rounded to BF16 at shared ``addr`` of this CTA and of cluster rank
    ``peer`` (same offset)."""
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(peer, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\n.reg .b32 ra;\ncvt.rn.bf16.f32 h, $2;\nst.shared.b16 [$0], h;\n"
        "mapa.shared::cluster.u32 ra, $0, $1;\nst.shared::cluster.b16 [ra], h;\n}",
        "r,r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_bf16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_u8(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ldg_s32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.s32 $0, [$1];", "=r,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ldg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _ldg_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ldg_v4_pred(addr, pred, *, loc=None, ip=None):
    """16 bytes at ``addr`` if ``pred > 0``, else zeros (no load)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip), _iv(pred, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.gt.s32 p, $5, 0;\nmov.b32 $0, 0;\nmov.b32 $1, 0;\n"
        "mov.b32 $2, 0;\nmov.b32 $3, 0;\n@p ld.global.v4.b32 {$0, $1, $2, $3}, [$4];\n}",
        "=r,=r,=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _stg_v2(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@cute.jit
def _tma_load_3d(dst: Int32, desc, y: Int32, bar: Int32):
    """One TMA box at coordinates ``(0, y, 0)`` of the 3D ``desc`` into shared ``dst``, completing
    on the shared mbarrier ``bar``."""
    inline_ptx(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [{$r0}], [{$r1}, {{$r2}, {$r3}, {$r2}}], [{$r4}];",
        read_only_args=[dst, desc, Int32(0), y, bar],
    )


@cute.jit
def _tma_prefetch_l2_3d(desc, y: Int32):
    """Prefetch the TMA box at ``(0, y, 0)`` of the 3D ``desc`` into L2."""
    inline_ptx(
        "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [{$r0}, {{$r1}, {$r2}, {$r1}}];",
        read_only_args=[desc, Int32(0), y],
    )


@dsl_user_op
def _rmsnorm_bf16x8(x0, x1, x2, x3, w0, w1, w2, w3, rstd, *, loc=None, ip=None):
    """``bf16((float(x) * rstd) * (float(w) + 0))`` of eight BF16 values (four packed words
    each): the normalization step of FlashInfer's ``RMSNormKernel`` (weight bias 0), each
    product rounded in FP32, then to BF16 (nearest even)."""
    body = ""
    for i in range(4):
        body += (
            f"shl.b32 xl, ${i + 4}, 16;\nand.b32 xh, ${i + 4}, 0xffff0000;\n"
            f"shl.b32 wl, ${i + 8}, 16;\nand.b32 wh, ${i + 8}, 0xffff0000;\n"
            "mov.b32 fxl, xl;\nmov.b32 fxh, xh;\nmov.b32 fwl, wl;\nmov.b32 fwh, wh;\n"
            "add.rn.f32 fwl, fwl, 0f00000000;\nadd.rn.f32 fwh, fwh, 0f00000000;\n"
            "mul.rn.f32 fxl, fxl, $12;\nmul.rn.f32 fxh, fxh, $12;\n"
            "mul.rn.f32 fxl, fxl, fwl;\nmul.rn.f32 fxh, fxh, fwh;\n"
            f"cvt.rn.bf16x2.f32 ${i}, fxh, fxl;\n"
        )
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(v, loc=loc, ip=ip) for v in (x0, x1, x2, x3, w0, w1, w2, w3)]
        + [_fv(rstd, loc=loc, ip=ip)],
        "{\n.reg .b32 xl, xh, wl, wh;\n.reg .f32 fxl, fxh, fwl, fwh;\n" + body + "}",
        "=r,=r,=r,=r,r,r,r,r,r,r,r,r,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


def _swz(row, chunk):
    """Byte offset of 16-byte chunk ``chunk`` of row ``row`` in a 128-byte-swizzled box."""
    return row * Int32(_BOX_BYTES) + ((chunk ^ (row & Int32(7))) << Int32(4))


@dsl_user_op
def _mbar_arrive(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bar_sync(bar_id, count: int, *, loc=None, ip=None):
    """``bar.sync bar_id, count`` (named CTA barrier)."""
    _asm(None, [_iv(bar_id, loc=loc, ip=ip)], f"bar.sync $0, {count};", "r", loc=loc, ip=ip)


@dsl_user_op
def _bar_arrive(bar_id, count: int, *, loc=None, ip=None):
    """``bar.arrive bar_id, count`` (named CTA barrier, no wait)."""
    _asm(None, [_iv(bar_id, loc=loc, ip=ip)], f"bar.arrive $0, {count};", "r", loc=loc, ip=ip)


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    """All threads of the cluster: release this CTA's shared stores, acquire the peers'."""
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cluster_ctarank(*, loc=None, ip=None):
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %cluster_ctarank;", "=r", loc=loc, ip=ip))


@dsl_user_op
def _bf16x2_to_f32x2(word, *, loc=None, ip=None):
    """The two BF16 halves of ``word`` (low half first) as FP32 (exact)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
        "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rope_pair_bf16(x, y, c, s, *, loc=None, ip=None):
    """GPT-J rotation of the pair ``(x, y)`` by ``(c, s)``: ``(x c - y s, y c + x s)``, each
    product and sum rounded separately in FP32 (``_fp4_mla_rope_fp32``), then to BF16 and back
    to FP32 (exact)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_fv(v, loc=loc, ip=ip) for v in (x, y, c, s)],
        "{\n"
        ".reg .f32 xc, ys, yc, xs, e, o;\n"
        ".reg .b16 he, ho;\n"
        "mul.rn.f32 xc, $2, $4;\n"
        "mul.rn.f32 ys, $3, $5;\n"
        "sub.rn.f32 e, xc, ys;\n"
        "mul.rn.f32 yc, $3, $4;\n"
        "mul.rn.f32 xs, $2, $5;\n"
        "add.rn.f32 o, yc, xs;\n"
        "cvt.rn.bf16.f32 he, e;\n"
        "cvt.rn.bf16.f32 ho, o;\n"
        "cvt.f32.bf16 $0, he;\n"
        "cvt.f32.bf16 $1, ho;\n"
        "}",
        "=f,=f,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _fp4_scale(amax, *, loc=None, ip=None):
    """The E4M3 scale of a 16-value group and the reciprocal the codes are scaled by:
    ``e4m3(max(amax / 6, 2^-9))`` (byte) and ``rcp.approx.ftz(float(that))``, the operations of
    ``_fp4_mla_quantize_q_groups``."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32()]),
        [_fv(amax, loc=loc, ip=ip)],
        "{\n"
        ".reg .f32 s, z;\n"
        ".reg .b16 h8, hl, hh;\n"
        ".reg .b32 h2;\n"
        "div.full.f32 s, $3, 0f40C00000;\n"
        f"max.f32 s, s, 0f{_Q_SCALE_FLOOR:08X};\n"
        "mov.f32 z, 0f00000000;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h8, z, s;\n"
        "cvt.rn.f16x2.e4m3x2 h2, h8;\n"
        "mov.b32 {hl, hh}, h2;\n"
        "cvt.f32.f16 $1, hl;\n"
        "rcp.approx.ftz.f32 $2, $1;\n"
        "cvt.u32.u16 $0, h8;\n"
        "and.b32 $0, $0, 0xff;\n"
        "}",
        "=r,=f,=f,f",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e2m1x8(v, inv, *, loc=None, ip=None):
    """E2M1 codes of the eight values ``v[i] * inv`` (``cvt.rn.satfinite``), element ``2k`` in
    the low nibble of byte ``k``."""
    body = "".join(f"mul.rn.f32 x{i}, ${i + 1}, $9;\n" for i in range(8))
    body += "".join(
        f"cvt.rn.satfinite.e2m1x2.f32 b{k}, x{2 * k + 1}, x{2 * k};\n" for k in range(4)
    )
    body += "mov.b32 $0, {b0, b1, b2, b3};\n"
    return Int32(
        _asm(
            T.i32(),
            [_fv(x, loc=loc, ip=ip) for x in v] + [_fv(inv, loc=loc, ip=ip)],
            "{\n.reg .f32 x0, x1, x2, x3, x4, x5, x6, x7;\n.reg .b8 b0, b1, b2, b3;\n" + body + "}",
            "=r," + ",".join(["f"] * 9),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e2m1x8_to_f32(word, *, loc=None, ip=None):
    """The eight E2M1 codes of ``word`` (element ``2k`` in the low nibble of byte ``k``) as FP32,
    through ``cvt.rn.f16x2.e2m1x2`` (exact), as ``_fp4_mla_unpack_e2m1x2``."""
    body = "mov.b32 {b0, b1, b2, b3}, $8;\n"
    for k in range(4):
        body += f"cvt.rn.f16x2.e2m1x2 h, b{k};\nmov.b32 {{l, u}}, h;\n"
        body += f"cvt.f32.f16 ${2 * k}, l;\ncvt.f32.f16 ${2 * k + 1}, u;\n"
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b8 b0, b1, b2, b3;\n.reg .b32 h;\n.reg .b16 l, u;\n" + body + "}",
        ",".join(["=f"] * 8) + ",r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 8, loc=loc, ip=ip)


@dsl_user_op
def _residual(v, main, scale, *, loc=None, ip=None):
    """``v - main * scale`` as ``_fp4_mla_quantize_q_residual_groups`` computes it."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip) for x in (v, main, scale)],
            "{\n.reg .f32 p;\nneg.f32 p, $2;\nfma.rn.f32 $0, p, $3, $1;\n}",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _abs_max16(v, *, loc=None, ip=None):
    """``max |v[i]|`` of 16 FP32 values (``abs.f32`` / ``max.f32``)."""
    body = "".join(f"abs.f32 a{i}, ${i + 1};\n" for i in range(16))
    body += "".join(f"max.f32 a{i}, a{i}, a{i + 8};\n" for i in range(8))
    body += "".join(f"max.f32 a{i}, a{i}, a{i + 4};\n" for i in range(4))
    body += "max.f32 a0, a0, a2;\nmax.f32 a1, a1, a3;\nmax.f32 $0, a0, a1;\n"
    regs = ", ".join(f"a{i}" for i in range(16))
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip) for x in v],
            "{\n.reg .f32 " + regs + ";\n" + body + "}",
            "=f," + ",".join(["f"] * 16),
            loc=loc,
            ip=ip,
        )
    )


def _quant_group16(v):
    """FP4 codes (two words) and the E4M3 scale byte of 16 FP32 values; also the scale as FP32."""
    sf, sf_f32, inv = _fp4_scale(_abs_max16(v))
    return _e2m1x8(v[:8], inv), _e2m1x8(v[8:], inv), sf, sf_f32


def _sf_offset(row, col: Int32):
    """Byte offset of scale column ``col`` of query row ``row`` in the swizzled 128x4 layout
    (``_fp4_mla_q_sf_offset``)."""
    padded = _ceil_div(Q_SF_COLS, 4) * 4
    return (
        Int64(col % Int32(4))
        + Int64(col // Int32(4)) * Int64(4 * 128)
        + Int64(row % Int32(32)) * Int64(16)
        + Int64((row % Int32(128)) // Int32(32)) * Int64(4)
        + Int64(row // Int32(128)) * Int64(128 * padded)
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class K3MlaQFusedKernel:
    """Fused MLA query of ``t_pad`` (a multiple of 8) tokens; see the module docstring.

    Args:
        t_pad: Tokens rounded up to a multiple of 8 (the MMA N tiles); rows past the launch's
            token count are computed on zeros and not stored.
        use_pdl: Programmatic dependent launch: weights are fetched before
            ``griddepcontrol.wait``.
        debug: Also store the BF16 q_b output and the BF16 absorbed / rotated query (tests).
        pre_tiles: q_b tiles issued before ``griddepcontrol.wait`` (the rest right after it).
        fuse_norm: ``x`` is the raw q_a latent: the kernel applies the q_a RMSNorm itself
            (FlashInfer ``RMSNormKernel`` arithmetic, see :meth:`K3MlaQFusedKernel.row_rstd`).
        timing: Thread 0 of every CTA stores ``%globaltimer`` at its phase boundaries into
            ``ts[cta * 16 + phase]`` (microbenchmarks): 0 entry, 1 after the wait, 2 activation
            quantized, 8 warp 0's q_b tile landed, 3 its MMAs done, 4 its q_b output stored, 5
            after the cluster barrier, 9 W_UK landed, 6 absorbed, 7 end.
        early_trigger: (``use_pdl``) release the dependent grid at entry instead of after the
            query stores, so a dependent that only reads the query after its own
            ``griddepcontrol.wait`` (the FP4 decode with ``pdl_wait``) runs its prologue on the
            SMs this grid's CTAs leave as they finish.
    """

    def __init__(
        self,
        t_pad: int,
        use_pdl: bool,
        debug: bool = False,
        timing: bool = False,
        fuse_norm: bool = False,
        pre_tiles: int = _NTILE,
        early_trigger: bool = False,
    ):
        if t_pad % 8 or not 8 <= t_pad <= MAX_TOKENS:
            raise ValueError(f"t_pad={t_pad}")
        self.t_pad = t_pad
        self.nt = t_pad // 8
        self.use_pdl = use_pdl
        self.debug = debug
        self.timing = timing
        self.fuse_norm = fuse_norm
        if not 0 <= pre_tiles <= _NTILE:
            raise ValueError(f"pre_tiles={pre_tiles}")
        self.pre_tiles = pre_tiles
        self.early_trigger = early_trigger and use_pdl
        assert t_pad * _OB_PITCH <= (_NTILE - _UK_TILES) * _TILE_BYTES

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        wqb_ptr: cute.Pointer,
        ew_addr: Int64,
        wuk_ptr: cute.Pointer,
        gs_addr: Int64,
        cos_sin_addr: Int64,
        pos_addr: Int64,
        qfp4_addr: Int64,
        qsf_addr: Int64,
        dbg_qb_addr: Int64,
        dbg_q_addr: Int64,
        ts_addr: Int64,
        nw_addr: Int64,
        eps: Float32,
        x_ld: Int32,
        num_tokens: Int32,
        gen_len: Int32,
        rows_per_seq: Int32,
        stream: cuda.CUstream,
    ):
        t_pad = self.t_pad

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 16], 128]
            ew: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, 128], 128]
            bsc: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 2 * _KGROUPS * t_pad], 128]
            act: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, t_pad * _A_PITCH], 128]
            ring: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, _NTILE * _TILE_BYTES], 1024]
            qn: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, t_pad * _QN_PITCH], 128]
            qpe: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, t_pad * _QPE_PITCH], 128]
            nw: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, Q_LORA_RANK * 2], 128]
            cs: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, t_pad * 4 * 64], 128]
            pos: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, t_pad], 128]
            part: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, _NTILE * 32 * t_pad // 2], 128
            ]

        self.shared_storage = SharedStorage
        # q_b weight [N, 1536] as (128 bytes, row, k block) and W_UK [H * 512, 128] as (64 values,
        # row, half): a box of whole rows lands as [k block][row][128 bytes].
        w_qb = cute.make_tensor(
            wqb_ptr,
            cute.make_layout(
                (_BOX_BYTES, NUM_HEADS * QK_HEAD_DIM, _KB), stride=(1, Q_LORA_RANK, _BOX_BYTES)
            ),
        )
        w_uk = cute.make_tensor(
            wuk_ptr,
            cute.make_layout(
                (_BOX_BYTES // 2, NUM_HEADS * KV_LORA_RANK, 2),
                stride=(1, NOPE_DIM, _BOX_BYTES // 2),
            ),
        )
        wqb_desc = cuda_tma.create_tensor_map_tiled_from_view(
            w_qb,
            box_dims=(_BOX_BYTES, _TILE_ROWS, _KB),
            stride_order=(0, 1, 2),
            swizzle=cuda_tma.TensorMapSwizzle.s128b,
        )
        wuk_desc = cuda_tma.create_tensor_map_tiled_from_view(
            w_uk,
            box_dims=(_BOX_BYTES // 2, _CTA_COLS, 2),
            stride_order=(0, 1, 2),
            swizzle=cuda_tma.TensorMapSwizzle.s128b,
        )
        self.kernel(
            x_addr, wqb_desc, ew_addr, wuk_desc, gs_addr, cos_sin_addr, pos_addr, qfp4_addr,
            qsf_addr, dbg_qb_addr, dbg_q_addr, ts_addr, nw_addr, eps, x_ld, num_tokens, gen_len,
            rows_per_seq,
        ).launch(
            grid=(2 * NUM_HEADS, 1, 1),
            block=(_THREADS, 1, 1),
            cluster=(2, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        wqb_desc: cutlass.GridConstant[cuda_tma.TensorMap],
        ew_addr: Int64,
        wuk_desc: cutlass.GridConstant[cuda_tma.TensorMap],
        gs_addr: Int64,
        cos_sin_addr: Int64,
        pos_addr: Int64,
        qfp4_addr: Int64,
        qsf_addr: Int64,
        dbg_qb_addr: Int64,
        dbg_q_addr: Int64,
        ts_addr: Int64,
        nw_addr: Int64,
        eps: Float32,
        x_ld: Int32,
        num_tokens: Int32,
        gen_len: Int32,
        rows_per_seq: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        ts = ts_addr + Int64(bidx) * 128
        self.stamp(tidx, ts, 0)
        warp = tidx // 32
        lane = tidx % 32
        head = bidx // Int32(2)
        rank = _cluster_ctarank()
        peer = rank ^ Int32(1)
        row0 = head * Int32(QK_HEAD_DIM) + rank * Int32(_CTA_ROWS)

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        mbar = storage.mbar.data_ptr().toint()
        full_bar = mbar
        uk_free_bar = mbar + Int32(8 * _NTILE)
        wuk_bar = mbar + Int32(8 * (_NTILE + 1))
        ew_s = storage.ew.data_ptr().toint()
        bsc_s = storage.bsc.data_ptr().toint()
        act_s = storage.act.data_ptr().toint()
        ring_s = storage.ring.data_ptr().toint()
        qn_s = storage.qn.data_ptr().toint()
        qpe_s = storage.qpe.data_ptr().toint()
        nw_s = storage.nw.data_ptr().toint()
        cs_s = storage.cs.data_ptr().toint()
        pos_s = storage.pos.data_ptr().toint()
        part_s = storage.part.data_ptr().toint()
        wuk_s = ring_s
        ob_s = ring_s + Int32(_OB_OFFSET)
        wqb_tmap = wqb_desc.get_ptr()
        wuk_tmap = wuk_desc.get_ptr()

        # ---- before the wait: barriers, block exponents, weights (none depends on the
        # previous grid).
        if cutlass.const_expr(self.early_trigger):
            _griddepcontrol_launch_dependents()
        if tidx == 0:
            for s in cutlass.range_constexpr(_NTILE):
                _mbar_init(full_bar + Int32(8 * s), 1)
            _mbar_init(uk_free_bar, _UK_TILES * _QB_SPLIT_K)
            _mbar_init(wuk_bar, 1)
        if tidx < Int32(_NTILE * _KBLOCKS):
            tile = tidx // Int32(_KBLOCKS)
            kb = tidx - tile * Int32(_KBLOCKS)
            block_row = (row0 + tile * Int32(_TILE_ROWS)) // Int32(128)
            _sts_u8(ew_s + tidx, _ldg_u8(ew_addr + Int64(block_row * Int32(_KBLOCKS) + kb)))
        if cutlass.const_expr(self.fuse_norm):
            if tidx < Int32(Q_LORA_RANK * 2 // 16):
                _sts_v4_b32(nw_s + tidx * Int32(16), *_ldg_v4(nw_addr + Int64(tidx) * 16))
        cute.arch.barrier()
        if tidx == 0:
            # Every 16-row q_b tile (all of K) into its own slot.
            for tile in cutlass.range_constexpr(self.pre_tiles):
                self.load_tile(tile, row0, wqb_tmap, ring_s, full_bar)

        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        self.stamp(tidx, ts, 1)
        if cutlass.const_expr(self.pre_tiles < _NTILE):
            if tidx == 0:
                for tile in cutlass.range_constexpr(self.pre_tiles, _NTILE):
                    self.load_tile(tile, row0, wqb_tmap, ring_s, full_bar)

        # ---- Activation: warp w owns rows w, w + 12, .. ; lane l holds elements 256 b + 8 l .. + 7
        # (b = 0..5) of a row, the layout of FlashInfer's RMSNormKernel for a 1536-wide row.
        rows = []
        for i in cutlass.range_constexpr(_ceil_div(self.t_pad, _WARPS)):
            t = warp + Int32(i * _WARPS)
            src = (
                x_addr
                + (
                    Int64(cutlass.max(cutlass.min(t, num_tokens - Int32(1)), Int32(0)))
                    * Int64(x_ld)
                    + Int64(lane) * 8
                )
                * 2
            )
            rows.append(
                (t, [list(_ldg_v4_pred(src + Int64(512 * b), num_tokens - t)) for b in range(6)])
            )
        for t, xw in rows:
            if t < Int32(self.t_pad):
                self.quantize_row(t, xw, lane, eps, nw_s, ew_s, act_s, bsc_s)
        cute.arch.barrier()
        self.stamp(tidx, ts, 2)
        if tidx == 0:
            # W_UK into L2 behind the q_b weight (it is read after the q_b MMAs).
            _tma_prefetch_l2_3d(wuk_tmap, head * Int32(KV_LORA_RANK) + rank * Int32(_CTA_COLS))

        # ---- q_b: warp w runs K half w // 6 of the CTA's 16-row tile w % 6.
        self.q_b_tile(warp, lane, rank, peer, head, row0, bsc_s, act_s, ring_s, qn_s, qpe_s,
                      part_s, full_bar, uk_free_bar, dbg_qb_addr, num_tokens, tidx, ts)  # fmt: skip
        self.stamp(tidx, ts, 4)
        if warp == Int32(_WARPS - 1):
            if rank == Int32(1):
                self.prefetch_rope(lane, num_tokens, pos_addr, cos_sin_addr, pos_s, cs_s)
        if warp == Int32(0):
            # W_UK into the slots of the first q_b tiles once their warps are done with them.
            self.wait(uk_free_bar, Int32(0))
            if lane == 0:
                _mbar_expect_tx(wuk_bar, _UK_BYTES)
                uk_row = head * Int32(KV_LORA_RANK) + rank * Int32(_CTA_COLS)
                _tma_load_3d(wuk_s, wuk_tmap, uk_row, wuk_bar)
        _cluster_sync()
        self.stamp(tidx, ts, 5)

        # ---- W_UK: warp w < 8 owns absorbed columns 32 w .. 32 w + 31 of the CTA's 256.
        if warp < Int32(_ABSORB_WARPS):
            self.wait(wuk_bar, Int32(0))
            self.stamp(tidx, ts, 9)
            self.absorb(warp, lane, wuk_s, qn_s, ob_s)
        cute.arch.barrier()
        self.stamp(tidx, ts, 6)

        # ---- FP4 query rows.
        # Absorbed groups (16 per token) and, on CTA 1, RoPE groups (4 per token) in one pass.
        gs = _ldg_f32(gs_addr)
        n_prefix = Int32(self.t_pad * 16)
        n_items = n_prefix + rank * Int32(self.t_pad * 4)
        for item in cutlass.range(tidx, n_items, Int32(_THREADS), unroll=1):
            if item < n_prefix:
                t = item // Int32(16)
                grp = item - t * Int32(16)
                if t < num_tokens:
                    self.store_prefix_group(t, grp, rank, head, gs, ob_s, qfp4_addr, qsf_addr,
                                            dbg_q_addr, gen_len, rows_per_seq)  # fmt: skip
            else:
                t = (item - n_prefix) // Int32(4)
                rg = item - n_prefix - t * Int32(4)
                if t < num_tokens:
                    if _lds_b32(pos_s + t * Int32(4)) >= Int32(0):
                        self.store_rope_group(t, rg, head, gs, qpe_s, cs_s, qfp4_addr, qsf_addr,
                                              dbg_q_addr, gen_len, rows_per_seq)  # fmt: skip
        if cutlass.const_expr(self.use_pdl and not self.early_trigger):
            _griddepcontrol_launch_dependents()
        self.stamp(tidx, ts, 7)

    @cute.jit
    def quantize_row(
        self,
        t: Int32,
        xw,
        lane: Int32,
        eps: Float32,
        nw_s: Int32,
        ew_s: Int32,
        act_s: Int32,
        bsc_s: Int32,
    ):
        """Activation row ``t``, held by the warp as ``xw[b]`` (four words of elements
        ``256 b + 8 l .. + 7`` per lane ``l``): normalized (``fuse_norm``: :meth:`row_rstd`, then
        the norm kernel's per-element products), then the low-M GEMM's MXFP8 quantization (the
        amax of a 32-value group, lanes ``4 m .. 4 m + 3`` of a block, by shuffles; its UE8M0
        exponent and E4M3 codes) into shared memory, with the group's MMA scales."""
        if cutlass.const_expr(self.fuse_norm):
            rstd = self.row_rstd(xw, eps)
            for b in cutlass.range_constexpr(6):
                wn = _lds_v4_b32(nw_s + (Int32(256 * b) + lane * Int32(8)) * Int32(2))
                xw[b] = list(_rmsnorm_bf16x8(*xw[b], *wn, rstd))
        # The six groups' amax shuffles are interleaved (independent chains).
        amaxes = [_amax_bf16x8_words(*xw[b]) for b in range(6)]
        amaxes = [_amax_pair(m, _shfl_bfly_b32(m, 1)) for m in amaxes]
        amaxes = [_amax_pair(m, _shfl_bfly_b32(m, 2)) for m in amaxes]
        for b in cutlass.range_constexpr(6):
            e_a = _ue8m0_of_amax(amaxes[b])
            q0, q1 = _quant_bf16x8_to_e4m3(*xw[b], _bf16x2_inv_scale(e_a))
            _sts_v2_b32(act_s + t * Int32(_A_PITCH) + Int32(256 * b) + lane * Int32(8), q0, q1)
            if lane % 4 == 0:
                # The group's MMA scale for both 128-row weight blocks the CTA's rows touch
                # (tiles 0 and 5 lie in the first and the last of them).
                kg = Int32(8 * b) + lane // Int32(4)
                kb = Int32(2 * b) + lane // Int32(16)
                for rb in cutlass.range_constexpr(2):
                    w_exp = _lds_u8(ew_s + Int32(rb * (_NTILE - 1) * _KBLOCKS) + kb)
                    _sts_f32(bsc_s + ((Int32(rb * _KGROUPS) + kg) * Int32(self.t_pad) + t) * 4,
                             _block_scale(e_a, w_exp))  # fmt: skip

    @cute.jit
    def load_tile(self, tile: int, row0: Int32, wqb_tmap, ring_s: Int32, full_bar: Int32):
        """One thread: q_b tile ``tile`` (16 whole rows) into its slot."""
        bar = full_bar + Int32(8 * tile)
        _mbar_expect_tx(bar, _TILE_BYTES)
        _tma_load_3d(
            ring_s + Int32(tile * _TILE_BYTES), wqb_tmap, row0 + Int32(tile * _TILE_ROWS), bar
        )

    @cute.jit
    def row_rstd(self, xw, eps: Float32):
        """``rsqrt(mean(x^2) + eps)`` of a 1536-wide BF16 row held as FlashInfer's
        ``RMSNormKernel`` holds it (32 lanes per row, vector width 8, six vector blocks: ``xw[b]``
        are the four words of elements ``256 b + 8 l .. + 7``), with that kernel's operations:
        the 48 values squared and summed as one ``TensorSSA`` (block-major), ``warp_reduce``
        (butterfly), the division by the width and the fast ``rsqrt``. The same DSL operations on
        the same values in the same order give the same bits."""
        xs = cute.make_rmem_tensor((48,), Float32)
        for b in cutlass.range_constexpr(6):
            for k in cutlass.range_constexpr(4):
                lo, hi = _bf16x2_to_f32x2(xw[b][k])
                xs[8 * b + 2 * k] = lo
                xs[8 * b + 2 * k + 1] = hi
        x = xs.load()
        x_sq = x * x
        local_val = x_sq.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
        sum_sq = _flashinfer_warp_reduce(local_val, operator.add, width=32)
        mean_sq = sum_sq / Float32(Q_LORA_RANK)
        return cute.math.rsqrt(mean_sq + eps, fastmath=True)

    @cute.jit
    def stamp(self, tidx: Int32, ts: Int64, phase: int):
        """``timing``: thread 0 stores ``%globaltimer`` as phase ``phase`` of this CTA."""
        if cutlass.const_expr(self.timing):
            if tidx == 0:
                _st_u64(ts + Int64(phase * 8), _globaltimer())

    @cute.jit
    def wait(self, bar: Int32, phase: Int32):
        done = Int32(0)
        while done == 0:
            done = _mbar_try_wait(bar, phase)

    @cute.jit
    def q_b_tile(
        self,
        warp: Int32,
        lane: Int32,
        rank: Int32,
        peer: Int32,
        head: Int32,
        row0: Int32,
        bsc_s: Int32,
        act_s: Int32,
        ring_s: Int32,
        qn_s: Int32,
        qpe_s: Int32,
        part_s: Int32,
        full_bar: Int32,
        uk_free_bar: Int32,
        dbg_qb_addr: Int64,
        num_tokens: Int32,
        tidx: Int32,
        ts: Int64,
    ):
        """K half ``warp // 6`` of the CTA's 16-row q_b tile ``j = warp % 6``: FP32 sums in K order;
        the second half's warp hands its sums to the first half's warp (shared memory, named
        barrier ``1 + j``), which adds them (the low-M GEMM's split-K association), rounds to BF16
        and stores q_nope (both CTAs) or q_pe."""
        nt = self.nt
        j = warp % Int32(_NTILE)
        half = warp // Int32(_NTILE)
        g = lane // 4
        q = lane % 4
        acc = cute.make_rmem_tensor((4 * nt,), Float32)
        for i in cutlass.range_constexpr(4 * nt):
            acc[i] = Float32(0.0)
        # ldmatrix row addresses: A (16 rows x 32 K bytes), B (8 tokens x 32 K bytes per n tile;
        # with .x4 two n tiles).
        a_row = (lane & 7) + ((lane >> 3) & 1) * 8
        a_hi = lane >> 4
        b_tok = (lane & 7) + (lane >> 4) * Int32(8)
        b_off = b_tok * Int32(_A_PITCH) + ((lane >> 3) & 1) * Int32(16)
        # Scales of this tile's 128-row weight block (the CTA's first or second).
        rb = ((row0 + j * Int32(_TILE_ROWS)) >> 7) - (row0 >> 7)
        s_base = bsc_s + (rb * Int32(_KGROUPS * self.t_pad) + q * Int32(2)) * Int32(4)
        self.wait(full_bar + j * Int32(8), Int32(0))
        self.stamp(tidx, ts, 8)
        tile = ring_s + j * Int32(_TILE_BYTES)
        a_off = a_row * Int32(_BOX_BYTES)
        kb_half = _KB // _QB_SPLIT_K
        # One 128-byte K block (four K32 steps) per iteration: its fragments and scales are all
        # loaded and its MMAs all issued before the scaled sums, which add the steps in K order.
        for kb in cutlass.range(half * Int32(kb_half), (half + 1) * Int32(kb_half), 1, unroll=1):
            plane = tile + kb * Int32(_TILE_ROWS * _BOX_BYTES) + a_off
            frags = []
            for kk in cutlass.range_constexpr(4):
                kg = kb * Int32(4) + Int32(kk)
                a = _ldsm_x4(plane + ((Int32(kk * 2) + a_hi) ^ (a_row & Int32(7))) * 16)
                bs = []
                for p in cutlass.range_constexpr(_ceil_div(nt, 2)):
                    addr = act_s + b_off + Int32(p * 16 * _A_PITCH) + kg * Int32(32)
                    if cutlass.const_expr(nt == 1):
                        bs.append(_ldsm_x2(addr))
                    else:
                        r4 = _ldsm_x4(addr)
                        bs.append((r4[0], r4[1]))
                        bs.append((r4[2], r4[3]))
                scales = [
                    _lds_v2_f32(s_base + (kg * Int32(self.t_pad) + Int32(n * 8)) * Int32(4))
                    for n in range(nt)
                ]
                frags.append((a, bs, scales))
            prods = [
                [_mma_e4m3(a[0], a[1], a[2], a[3], bs[n][0], bs[n][1]) for n in range(nt)]
                for a, bs, _ in frags
            ]
            for kk in cutlass.range_constexpr(4):
                for n in cutlass.range_constexpr(nt):
                    d = prods[kk][n]
                    s0, s1 = frags[kk][2][n]
                    acc[4 * n + 0] = _fma_rn(d[0], s0, acc[4 * n + 0])
                    acc[4 * n + 1] = _fma_rn(d[1], s1, acc[4 * n + 1])
                    acc[4 * n + 2] = _fma_rn(d[2], s0, acc[4 * n + 2])
                    acc[4 * n + 3] = _fma_rn(d[3], s1, acc[4 * n + 3])
        self.stamp(tidx, ts, 3)
        if j < Int32(_UK_TILES):
            cute.arch.sync_warp()
            if lane == 0:
                _mbar_arrive(uk_free_bar)
        part = part_s + (j * Int32(32) + lane) * Int32(4 * nt * 4)
        if half == Int32(1):
            for i in cutlass.range_constexpr(nt):
                _sts_v4_f32(part + Int32(16 * i), acc[4 * i], acc[4 * i + 1], acc[4 * i + 2],
                            acc[4 * i + 3])  # fmt: skip
            _bar_arrive(Int32(1) + j, 64)
        else:
            _bar_sync(Int32(1) + j, 64)
            for i in cutlass.range_constexpr(nt):
                other = _lds_v4_f32(part + Int32(16 * i))
                for c in cutlass.range_constexpr(4):
                    acc[4 * i + c] = _add_rn(acc[4 * i + c], other[c])
            # Rows of this tile within the head's 192 (q_nope dims, then the RoPE dims).
            hr = rank * Int32(_CTA_ROWS) + j * Int32(_TILE_ROWS) + g
            is_nope = rank * Int32(_CTA_ROWS) + j * Int32(_TILE_ROWS) < Int32(NOPE_DIM)
            for n in cutlass.range_constexpr(nt):
                for h in cutlass.range_constexpr(2):
                    row = hr + Int32(8 * h)
                    for c in cutlass.range_constexpr(2):
                        t = Int32(n * 8) + q * Int32(2) + Int32(c)
                        value = acc[4 * n + 2 * h + c]
                        if is_nope:
                            _sts_bf16_both(
                                qn_s + t * Int32(_QN_PITCH) + row * Int32(2), peer, value
                            )
                        else:
                            _sts_bf16(
                                qpe_s + t * Int32(_QPE_PITCH) + (row - Int32(NOPE_DIM)) * 2, value
                            )
                        if cutlass.const_expr(self.debug):
                            if t < num_tokens:
                                _stg_bf16(
                                    dbg_qb_addr
                                    + (Int64(t) * (NUM_HEADS * QK_HEAD_DIM)
                                       + Int64(head) * QK_HEAD_DIM + Int64(row)) * 2,
                                    value,
                                )  # fmt: skip

    @cute.jit
    def absorb(self, warp: Int32, lane: Int32, wuk_s: Int32, qn_s: Int32, ob_s: Int32):
        nt = self.nt
        g = lane // 4
        q = lane % 4
        a_row = (lane & 7) + ((lane >> 3) & 1) * 8
        a_hi = lane >> 4
        b_tok = (lane & 7) + (lane >> 4) * Int32(8)
        b_off = b_tok * Int32(_QN_PITCH) + ((lane >> 3) & 1) * Int32(16)
        # The two tiles' MMA chains (each in K order) are interleaved.
        accs = [[[Float32(0.0)] * 4 for _ in range(nt)] for _ in range(2)]
        for kk in cutlass.range_constexpr(NOPE_DIM // 16):
            bs = []
            for p in cutlass.range_constexpr(_ceil_div(nt, 2)):
                addr = qn_s + b_off + Int32(p * 16 * _QN_PITCH + kk * 32)
                if cutlass.const_expr(nt == 1):
                    bs.append(_ldsm_x2(addr))
                else:
                    r4 = _ldsm_x4(addr)
                    bs.append((r4[0], r4[1]))
                    bs.append((r4[2], r4[3]))
            box = wuk_s + Int32((kk // 4) * _UK_HALF)
            a_mm = [
                _ldsm_x4(box + _swz((warp * Int32(2) + Int32(mm)) * Int32(16) + a_row,
                                    Int32((kk % 4) * 2) + a_hi))
                for mm in range(2)
            ]  # fmt: skip
            for mm in cutlass.range_constexpr(2):
                a = a_mm[mm]
                for n in cutlass.range_constexpr(nt):
                    accs[mm][n] = list(
                        _mma_bf16(a[0], a[1], a[2], a[3], bs[n][0], bs[n][1], *accs[mm][n])
                    )
        for mm in cutlass.range_constexpr(2):
            m = warp * Int32(2) + Int32(mm)
            acc = accs[mm]
            col = m * Int32(16) + g
            for n in cutlass.range_constexpr(nt):
                for h in cutlass.range_constexpr(2):
                    for c in cutlass.range_constexpr(2):
                        t = Int32(n * 8) + q * Int32(2) + Int32(c)
                        _sts_bf16(
                            ob_s + t * Int32(_OB_PITCH) + (col + Int32(8 * h)) * Int32(2),
                            acc[n][2 * h + c],
                        )

    @cute.jit
    def q_row(self, t: Int32, head: Int32, gen_len: Int32, rows_per_seq: Int32):
        seq = t // gen_len
        return seq * rows_per_seq + (t - seq * gen_len) * Int32(NUM_HEADS) + head

    @cute.jit
    def store_prefix_group(
        self,
        t: Int32,
        grp: Int32,
        rank: Int32,
        head: Int32,
        gs: Float32,
        ob_s: Int32,
        qfp4_addr: Int64,
        qsf_addr: Int64,
        dbg_q_addr: Int64,
        gen_len: Int32,
        rows_per_seq: Int32,
    ):
        """Absorbed values ``16 grp .. 16 grp + 15`` of this CTA's half of token ``t``: FP4 codes
        and scale of ``float(bf16) * gs`` (``_fp4_mla_quantize_q_groups_words``)."""
        src = ob_s + t * Int32(_OB_PITCH) + grp * Int32(32)
        words = list(_lds_v4_b32(src)) + list(_lds_v4_b32(src + Int32(16)))
        v = []
        for w in words:
            lo, hi = _bf16x2_to_f32x2(w)
            v.append(_mul_rn(lo, gs))
            v.append(_mul_rn(hi, gs))
        w0, w1, sf, _ = _quant_group16(v)
        col = rank * Int32(16) + grp
        qrow = self.q_row(t, head, gen_len, rows_per_seq)
        _stg_v2(qfp4_addr + Int64(qrow) * Q_ROW_BYTES + Int64(col) * 8, w0, w1)
        _stg_u8(qsf_addr + _sf_offset(qrow, col), sf)
        if cutlass.const_expr(self.debug):
            dst = (
                dbg_q_addr
                + (
                    (Int64(t) * NUM_HEADS + Int64(head)) * (KV_LORA_RANK + ROPE_DIM)
                    + Int64(col) * 16
                )
                * 2
            )
            _stg_v4(dst, *words[:4])
            _stg_v4(dst + Int64(16), *words[4:])

    @cute.jit
    def prefetch_rope(
        self,
        lane: Int32,
        num_tokens: Int32,
        pos_addr: Int64,
        cos_sin_addr: Int64,
        pos_s: Int32,
        cs_s: Int32,
    ):
        """One warp: every token's position, and the 16 rotary ``(cos, sin)`` values of each of
        its four RoPE groups at that position, into shared memory (item ``4 t + g``, 64 bytes)."""
        for j in cutlass.range_constexpr(_ceil_div(self.t_pad * 4, 32)):
            item = lane + Int32(32 * j)
            t = item // Int32(4)
            rg = item - t * Int32(4)
            if t < num_tokens:
                pos = _ldg_s32(pos_addr + Int64(t) * 4)
                if rg == Int32(0):
                    _sts_b32(pos_s + t * Int32(4), pos)
                if pos >= Int32(0):
                    table = cos_sin_addr + (Int64(pos) * (2 * ROPE_DIM) + Int64(rg) * 16) * 4
                    words = [_ldg_v4(table + Int64(16 * i)) for i in range(4)]
                    for i in cutlass.range_constexpr(4):
                        _sts_v4_b32(cs_s + item * Int32(64) + Int32(16 * i), *words[i])

    @cute.jit
    def store_rope_group(
        self,
        t: Int32,
        rg: Int32,
        head: Int32,
        gs: Float32,
        qpe_s: Int32,
        cs_s: Int32,
        qfp4_addr: Int64,
        qsf_addr: Int64,
        dbg_q_addr: Int64,
        gen_len: Int32,
        rows_per_seq: Int32,
    ):
        """RoPE values ``16 rg .. 16 rg + 15`` of token ``t``: rotated by the prefetched table
        values of its position, rounded to BF16, then the main and residual FP4 groups
        (``_fp4_mla_quantize_q_residual_groups``)."""
        src = qpe_s + t * Int32(_QPE_PITCH) + rg * Int32(32)
        words = list(_lds_v4_b32(src)) + list(_lds_v4_b32(src + Int32(16)))
        cs = []
        for i in cutlass.range_constexpr(4):
            cs += list(_lds_v4_f32(cs_s + (t * Int32(4) + rg) * Int32(64) + Int32(16 * i)))
        rot = []
        for k in cutlass.range_constexpr(8):
            x, y = _bf16x2_to_f32x2(words[k])
            e, o = _rope_pair_bf16(x, y, cs[2 * k], cs[2 * k + 1])
            rot.append(e)
            rot.append(o)
        v = [_mul_rn(x, gs) for x in rot]
        m0, m1, sf, sf_f32 = _quant_group16(v)
        main = list(_e2m1x8_to_f32(m0)) + list(_e2m1x8_to_f32(m1))
        res = [_residual(v[i], main[i], sf_f32) for i in range(16)]
        r0, r1, rsf, _ = _quant_group16(res)
        col = Int32(KV_LORA_RANK // FP4_GROUP) + rg * Int32(2)
        qrow = self.q_row(t, head, gen_len, rows_per_seq)
        dst = qfp4_addr + Int64(qrow) * Q_ROW_BYTES + Int64(col) * 8
        _stg_v4(dst, m0, m1, r0, r1)
        _stg_u8(qsf_addr + _sf_offset(qrow, col), sf)
        _stg_u8(qsf_addr + _sf_offset(qrow, col + Int32(1)), rsf)
        if cutlass.const_expr(self.debug):
            out = (
                dbg_q_addr
                + (
                    (Int64(t) * NUM_HEADS + Int64(head)) * (KV_LORA_RANK + ROPE_DIM)
                    + KV_LORA_RANK
                    + Int64(rg) * 16
                )
                * 2
            )
            packed = [_pack_bf16x2(rot[2 * k], rot[2 * k + 1]) for k in range(8)]
            _stg_v4(out, *packed[:4])
            _stg_v4(out + Int64(16), *packed[4:])


class K3MlaLatentRowKernel:
    """The latent row of the MLA KV half: ``out[t] = [rmsnorm(c_kv[t], w, eps) | k_pe[t]]`` for a
    512-wide ``c_kv`` and 64-wide ``k_pe`` (BF16), one warp per token, with the arithmetic of
    FlashInfer's ``RMSNormKernel`` for a 512-wide row (lane ``l`` holds elements ``256 b + 8 l ..
    + 7``, b = 0, 1; the same reduction and products as :meth:`K3MlaQFusedKernel.row_rstd`), so
    ``out`` is bitwise ``torch.cat([rmsnorm(c_kv), k_pe], -1)``. One launch instead of the norm
    and the ``k_pe`` copy. PDL: the weight is read before ``griddepcontrol.wait``; dependents are
    released right after it (every earlier grid is then complete)."""

    def __init__(self, use_pdl: bool):
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        ckv_addr: Int64,
        kpe_addr: Int64,
        w_addr: Int64,
        out_addr: Int64,
        eps: Float32,
        ckv_ld: Int32,
        kpe_ld: Int32,
        out_ld: Int32,
        num_tokens: Int32,
        grid: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            ckv_addr, kpe_addr, w_addr, out_addr, eps, ckv_ld, kpe_ld, out_ld, num_tokens
        ).launch(grid=(grid, 1, 1), block=(128, 1, 1), stream=stream, use_pdl=self.use_pdl)

    @cute.kernel
    def kernel(
        self,
        ckv_addr: Int64,
        kpe_addr: Int64,
        w_addr: Int64,
        out_addr: Int64,
        eps: Float32,
        ckv_ld: Int32,
        kpe_ld: Int32,
        out_ld: Int32,
        num_tokens: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        t = bidx * Int32(4) + tidx // Int32(32)
        col = lane * Int32(8)
        ww = [_ldg_v4(w_addr + Int64(Int32(256 * b) + col) * 2) for b in range(2)]
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            _griddepcontrol_launch_dependents()
        if t < num_tokens:
            row = ckv_addr + Int64(t) * Int64(ckv_ld) * 2
            xw = [list(_ldg_v4(row + Int64(Int32(256 * b) + col) * 2)) for b in range(2)]
            kpe = _ldg_v4_pred(kpe_addr + (Int64(t) * Int64(kpe_ld) + Int64(col)) * 2,
                               Int32(8) - lane)  # fmt: skip
            xs = cute.make_rmem_tensor((16,), Float32)
            for b in cutlass.range_constexpr(2):
                for k in cutlass.range_constexpr(4):
                    lo, hi = _bf16x2_to_f32x2(xw[b][k])
                    xs[8 * b + 2 * k] = lo
                    xs[8 * b + 2 * k + 1] = hi
            x = xs.load()
            x_sq = x * x
            local_val = x_sq.reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            )
            sum_sq = _flashinfer_warp_reduce(local_val, operator.add, width=32)
            rstd = cute.math.rsqrt(sum_sq / Float32(KV_LORA_RANK) + eps, fastmath=True)
            out = out_addr + Int64(t) * Int64(out_ld) * 2
            for b in cutlass.range_constexpr(2):
                y = _rmsnorm_bf16x8(*xw[b], *ww[b], rstd)
                _stg_v4(out + Int64(Int32(256 * b) + col) * 2, *y)
            if lane < Int32(ROPE_DIM // 8):
                _stg_v4(out + Int64(Int32(KV_LORA_RANK) + col) * 2, *kpe)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_COMPILED: dict = {}


@functools.cache
def _device_ok(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 7)


def block_exponents(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """UE8M0 exponents ``uint8 [ceil(n / 128), k / 128]`` of a 128x128 block-scaled FP8 weight
    from its scale as the K3 FP8 projections hold it: power-of-two FP32 block scales, or the
    Rubin MXFP8 GEMM's R128c4 UE8M0 layout (``transform_k128_scales_to_cutedsl_mxfp8_layout``,
    whose 512-byte atoms start with the block exponent)."""
    nb, kb = _ceil_div(n, 128), k // 128
    if weight_scale.dtype == torch.uint8 and weight_scale.numel() == nb * kb * 512:
        return weight_scale.reshape(nb, kb, 512)[:, :, 0].contiguous()
    if weight_scale.dtype == torch.float32 and weight_scale.numel() == nb * kb:
        bits = weight_scale.reshape(nb, kb).contiguous().view(torch.int32)
        if bool(((bits & 0x7FFFFF) != 0).any()):
            raise ValueError("block scales must be powers of two")
        return ((bits >> 23) & 0xFF).to(torch.uint8)
    raise ValueError(
        f"unsupported q_b weight scale {tuple(weight_scale.shape)} {weight_scale.dtype}"
    )


def weights_supported(w_qb: torch.Tensor, w_uk: torch.Tensor) -> bool:
    """Whether :func:`k3_mla_q_fused` runs with these q_b / W_UK weights on their device."""
    return bool(
        w_qb.is_cuda
        and _device_ok(w_qb.device.index if w_qb.device.index is not None else 0)
        and w_qb.dtype == torch.float8_e4m3fn
        and tuple(w_qb.shape) == (NUM_HEADS * QK_HEAD_DIM, Q_LORA_RANK)
        and w_qb.is_contiguous()
        and w_qb.data_ptr() % 16 == 0
        and w_uk.dtype == torch.bfloat16
        and tuple(w_uk.shape) == (NUM_HEADS, KV_LORA_RANK, NOPE_DIM)
        and w_uk.is_contiguous()
        and w_uk.data_ptr() % 16 == 0
    )


def input_supported(x: torch.Tensor) -> bool:
    """Whether :func:`k3_mla_q_fused` reads ``x`` (BF16 ``[T, 1536]``, ``T <= MAX_TOKENS``)."""
    return bool(
        x.is_cuda
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_TOKENS
        and x.shape[1] == Q_LORA_RANK
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    )


def can_run(x: torch.Tensor, w_qb: torch.Tensor, w_uk: torch.Tensor) -> bool:
    """Whether :func:`k3_mla_q_fused` serves ``x`` with these weights."""
    return input_supported(x) and weights_supported(w_qb, w_uk)


def k3_mla_q_fused(
    x: torch.Tensor,
    w_qb: torch.Tensor,
    w_qb_exp: torch.Tensor,
    w_uk: torch.Tensor,
    q_global_scale: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    *,
    gen_len: int,
    rows_per_seq: int,
    use_pdl: bool = True,
    debug: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    timestamps: Optional[torch.Tensor] = None,
    norm: Optional[tuple[torch.Tensor, float]] = None,
    pre_tiles: int = _NTILE,
    early_trigger: bool = False,
) -> None:
    """Write the FP4 decode query of ``x`` into ``q_fp4`` / ``q_sf`` (see the module docstring).

    Args:
        x: BF16 ``[T, 1536]`` normalized q_a latent (row stride a multiple of 8 elements).
        w_qb: q_b weight, FP8 ``[96 * 192, 1536]``; ``w_qb_exp``: its block exponents
            (:func:`block_exponents`).
        w_uk: BF16 absorption weight ``[96, 512, 128]``.
        q_global_scale: FP32 ``[1]`` global scale of the FP4 query.
        cos_sin: FP32 rotary table, 128 values ``(cos, sin)`` per position.
        positions: int32 ``[>= T]`` position of every token (a negative position skips the
            token's RoPE part, as the cache update does).
        q_fp4, q_sf: The decode's Q buffers: uint8 ``[rows, 320]`` and the swizzled E4M3 scales.
            Token ``t`` of request ``t // gen_len`` owns row
            ``(t // gen_len) * rows_per_seq + (t % gen_len) * 96 + head``.
        debug: Optional BF16 ``([T, 96 * 192], [T, 96, 576])`` outputs of the q_b projection and
            of the absorbed / rotated query (tests).
        norm: Optional ``(weight, eps)`` of the q_a RMSNorm (BF16 ``[1536]``): ``x`` is then the
            raw q_a latent and the kernel normalizes it (``fuse_norm``).
        pre_tiles: q_b tiles fetched before the PDL wait (all by default; microbenchmarks).
        timestamps: Optional int64 ``[192, 16]``: ``%globaltimer`` at the phase boundaries of every
            CTA (:class:`K3MlaQFusedKernel` ``timing``).
        early_trigger: Release the dependent grid at entry (:class:`K3MlaQFusedKernel`).
    """
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return
    if num_tokens > MAX_TOKENS:
        raise ValueError(f"{num_tokens} tokens exceed the fused MLA query's {MAX_TOKENS}")
    t_pad = 8 if num_tokens <= 8 else 16
    if positions.dtype != torch.int32 or positions.numel() < num_tokens:
        raise ValueError("positions must be int32 covering every token")
    if q_fp4.dtype != torch.uint8 or q_fp4.shape[-1] != Q_ROW_BYTES or not q_fp4.is_contiguous():
        raise ValueError("q_fp4 must be a contiguous uint8 [rows, 320] tensor")
    if cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous():
        raise ValueError("cos_sin must be a contiguous FP32 table")
    dbg = (0, 0) if debug is None else (debug[0].data_ptr(), debug[1].data_ptr())
    if norm is not None and not (
        NORM_FUSION_AVAILABLE
        and norm[0].dtype == torch.bfloat16
        and norm[0].numel() == Q_LORA_RANK
        and norm[0].is_contiguous()
        and norm[0].data_ptr() % 16 == 0
    ):
        raise ValueError("the fused q_a RMSNorm needs FlashInfer and a contiguous BF16 weight")
    key = (
        t_pad,
        use_pdl,
        debug is not None,
        timestamps is not None,
        norm is not None,
        pre_tiles,
        early_trigger,
    )
    args = (
        Int64(x.data_ptr()),
        make_ptr(cutlass.Float8E4M3FN, w_qb.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int64(w_qb_exp.data_ptr()),
        make_ptr(cutlass.BFloat16, w_uk.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int64(q_global_scale.data_ptr()),
        Int64(cos_sin.data_ptr()),
        Int64(positions.data_ptr()),
        Int64(q_fp4.data_ptr()),
        Int64(q_sf.data_ptr()),
        Int64(dbg[0]),
        Int64(dbg[1]),
        Int64(0 if timestamps is None else timestamps.data_ptr()),
        Int64(0 if norm is None else norm[0].data_ptr()),
        Float32(1e-6 if norm is None else norm[1]),
        Int32(x.stride(0)),
        Int32(num_tokens),
        Int32(gen_len),
        Int32(rows_per_seq),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        kernel = K3MlaQFusedKernel(
            t_pad,
            use_pdl,
            debug=debug is not None,
            timing=timestamps is not None,
            fuse_norm=norm is not None,
            pre_tiles=pre_tiles,
            early_trigger=early_trigger,
        )
        compiled = cute.compile(kernel, *args)
        _COMPILED[key] = compiled
    compiled(*args)


def latent_row_supported(c_kv: torch.Tensor, k_pe: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`k3_mla_latent_row` serves these operands."""
    return bool(
        NORM_FUSION_AVAILABLE
        and c_kv.is_cuda
        and c_kv.dim() == 2
        and k_pe.dim() == 2
        and c_kv.shape[1] == KV_LORA_RANK
        and k_pe.shape == (c_kv.shape[0], ROPE_DIM)
        and c_kv.dtype == k_pe.dtype == weight.dtype == torch.bfloat16
        and c_kv.stride(1) == 1
        and k_pe.stride(1) == 1
        and c_kv.stride(0) % 8 == 0
        and k_pe.stride(0) % 8 == 0
        and all(t.data_ptr() % 16 == 0 for t in (c_kv, k_pe, weight))
        and weight.numel() == KV_LORA_RANK
        and weight.is_contiguous()
    )


def k3_mla_latent_row(
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    use_pdl: bool = True,
) -> torch.Tensor:
    """``torch.cat([rmsnorm(c_kv, weight, eps), k_pe], -1)`` (BF16 ``[T, 576]``) in one launch,
    bitwise FlashInfer's CuTe DSL RMSNorm followed by the copy (see :class:`K3MlaLatentRowKernel`;
    check :func:`latent_row_supported`)."""
    num_tokens = c_kv.shape[0]
    out = torch.empty((num_tokens, KV_LORA_RANK + ROPE_DIM), dtype=c_kv.dtype, device=c_kv.device)
    if num_tokens == 0:
        return out
    args = (
        Int64(c_kv.data_ptr()),
        Int64(k_pe.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(out.data_ptr()),
        Float32(eps),
        Int32(c_kv.stride(0)),
        Int32(k_pe.stride(0)),
        Int32(out.stride(0)),
        Int32(num_tokens),
        Int32(_ceil_div(num_tokens, 4)),
        cuda.CUstream(torch.cuda.current_stream(c_kv.device).cuda_stream),
    )
    key = ("latent", use_pdl)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(K3MlaLatentRowKernel(use_pdl), *args)
        _COMPILED[key] = compiled
    compiled(*args)
    return out


__all__ = [
    "MAX_TOKENS",
    "k3_mla_latent_row",
    "latent_row_supported",
    "input_supported",
    "weights_supported",
    "Q_ROW_BYTES",
    "Q_SF_COLS",
    "block_exponents",
    "can_run",
    "k3_mla_q_fused",
    "smem_bytes",
]
