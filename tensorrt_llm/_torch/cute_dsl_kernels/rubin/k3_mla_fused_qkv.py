# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The shared-memory / tensor-memory layout setup, the scale-factor SMEM -> TMEM copy and the
# block-scaled MMA issue follow dense_blockscaled_gemm_persistent.py (adapted from CUTLASS).

# ruff: noqa: E501
"""Fused MLA q path of the Kimi K3 decode (``TRTLLM_K3_MLA_FUSED_QKV``), one kernel.

For a decode batch of ``M`` (<= 32) tokens it takes the fused projection output
``fused_a = [q_a (1536) | kv_a (512) | k_pe (64)]`` (BF16) and writes, in one launch, bit for bit
what the stock chain of norm, quantize, GEMM, absorption and RoPE / KV-append kernels writes:

* the FP8 query of the absorbed decode (``quant_q_buffer [M, 12, 576]``): with
  ``q = q_b_proj(rmsnorm(q_a))`` (BF16), ``e4m3(max(q_nope @ w_kc[h].T, -448))`` in columns
  0..511 of head ``h`` and ``e4m3(clamp(q_pe, -448, 448))`` in columns 512..575. ``q`` is computed
  with the FlashInfer CuTe RMSNorm (same thread / value layout, reduction order and math), the
  1x128 FP8 activation quantization of ``fp8_quantize_1x128_r128c4`` (UE8M0 scales) and the Rubin
  block-scaled GEMM ``cute_dsl_mxfp8_gemm_rubin`` with the in-cluster K split of its tactic: the
  ``k_splits`` CTAs of a cluster each run the MMA chain of their K-tile range and the FP32
  partials are summed in K-rank order (exchanged through distributed shared memory) before the
  BF16 rounding. The absorption is the tcgen05 absorption kernel's K-ordered kind::f16 chain
  (``k3_mla_q_absorb``: tokens as M, latent columns as N, 16-wide K steps) and epilogue;
* the new latent-cache rows of the FP8 paged KV cache (``rmsnorm(kv_a)`` and the RoPE'd ``k_pe``,
  each element ``bf16 -> f32 * kv_scale -> e4m3`` with saturation) exactly as
  ``applyMLARopeAndAssignQKVKernelGeneration`` writes them;
* the FMHA scale tensors and the scheduler buffers that kernel writes (``bmm1 = [s, s * log2e]``,
  ``bmm2 = [1]``, ``cu_q``, ``cu_kv`` and the counter).

Grid: 19 clusters of ``k_splits`` CTAs (one per K rank), 256 threads each. Cluster ``h < 12``
computes the 128 q_nope rows of head ``h`` and its absorption (K rank ``r`` writes latent columns
``[r * n_r, (r + 1) * n_r)``); cluster ``12 + p`` the q_pe rows of heads ``2p`` and ``2p + 1``; the
last cluster the KV rows and the buffers. A GEMV CTA stages its weight rows, their scales and its
absorption weights before ``griddepcontrol.wait`` (none is written by a predecessor); the KV
cluster reads the step metadata (KV lengths, page table) after it.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ..blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

#: q_b_proj geometry at TP8 (12 heads of 128 NoPE + 64 RoPE dims; q LoRA rank 1536).
NUM_HEADS = 12
NOPE_DIM = 128
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
QB_ROWS = NUM_HEADS * HEAD_DIM
Q_LORA = 1536
KV_LORA = 512
#: Columns of ``fused_a`` read here: q_a, kv_a, k_pe.
FUSED_A_COLS = Q_LORA + KV_LORA + ROPE_DIM
KV_ROW = KV_LORA + ROPE_DIM
#: Bytes of one head of the FP8 query (absorbed NoPE part, then q_pe).
OUT_HEAD = KV_LORA + ROPE_DIM
PAGE_TOKENS = 64
#: GEMV tiling: 128 weight rows per CTA, K tiles of 128, MMA K 64 (kind::mxf8f6f4, 32-element scale vector).
CTA_ROWS = 128
K_TILE = 128
NUM_K_TILES = Q_LORA // K_TILE
MMA_K = 64
SF_VEC = 32
SF_TILE_BYTES = 512
THREADS = 256
WARPS = THREADS // 32
#: Absorption tiling: 128 token rows (M), the CTA's latent columns (N), K tiles of 64 NoPE dims.
ABS_K_TILE = 64
#: Cluster roles (cluster index = blockIdx.y).
NOPE_CLUSTERS = NUM_HEADS
ROPE_CLUSTERS = NUM_HEADS // 2
KV_CLUSTER = NOPE_CLUSTERS + ROPE_CLUSTERS
NUM_CLUSTERS = KV_CLUSTER + 1
#: TMEM columns: the GEMV accumulator, the weight and activation scales, the absorption accumulator.
ACC_COL = 0
SFA_COL = 128
SFB_COL = 192
ABS_COL = 256
TMEM_COLS = 512
#: TMEM_COLS is the whole tensor memory of an SM, so a CTA must have its SM alone: the dynamic SMEM
#: request is padded past half of the SM's shared memory (sm_107: 328 KiB, 1 KiB of it reserved per
#: CTA). Otherwise the scheduler may place two small CTAs on one SM and the second waits in
#: tcgen05.alloc for the first one's columns (the rest of both clusters waits with them).
SM_SMEM_BYTES = 328 * 1024
MIN_SMEM_BYTES = SM_SMEM_BYTES // 2
#: Supported token-column counts and K partitions (the in-cluster K split of the stock tactic).
SUPPORTED_N_PAD = (8, 16, 32)
SUPPORTED_K_SPLITS = (2, 3, 4)
MAX_TOKENS = SUPPORTED_N_PAD[-1]
#: FlashInfer RMSNorm thread / value layout (``RMSNormKernel`` for H <= 3072): one warp per row,
#: lane ``l`` holds columns ``8 l + 256 b + j`` (8-element vectors ``b``, values ``j``).
VEC = 8
Q_VECS = Q_LORA // (32 * VEC)
KV_VECS = KV_LORA // (32 * VEC)
LOG2E = 1.4426950408889634


def n_pad_for(m: int) -> int:
    """The MMA N (token columns, 8 / 16 / 32) for ``m`` tokens."""
    for n in SUPPORTED_N_PAD:
        if m <= n:
            return n
    raise ValueError(f"m {m} exceeds the fused window ({MAX_TOKENS})")


def k_split_ranges(k_splits: int, k_tiles: int = NUM_K_TILES):
    """The K-tile range of each split, as the stock kernel assigns them to its K ranks
    (``k_tiles // k_splits`` each, the first ``k_tiles % k_splits`` one more)."""
    each, extra = divmod(k_tiles, k_splits)
    ranges = []
    start = 0
    for rank in range(k_splits):
        count = each + (1 if rank < extra else 0)
        ranges.append((start, start + count))
        start += count
    return tuple(ranges)


def lora_cols(k_splits: int) -> int:
    """Latent columns of the absorption per K rank (a multiple of 16; the last rank's tile may
    run past column 512, which is neither loaded nor stored)."""
    return -(-KV_LORA // (16 * k_splits)) * 16


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    """``rcp.approx.ftz.f32`` (the quantizer's 1 / 448)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x2(lo, hi, *, loc=None, ip=None):
    """Two FP32 -> two E4M3 bytes (``lo`` in the low byte), round to nearest even, saturating."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rn.satfinite.e4m3x2.f32 h, $2, $1;\ncvt.u32.u16 $0, h;\n}",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None):
    """``max.f32`` (NaN-quiet)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fabs(a, *, loc=None, ip=None):
    """``abs.f32``."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_neg448(a, *, loc=None, ip=None):
    """``max.f32(a, -448)``: NaN -> -448 (the absorption epilogue's clamp before the satfinite cvt)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, 0fC3E00000;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clamp448(a, *, loc=None, ip=None):
    """``min.f32(max.f32(a, -448), 448)`` (the absorption kernel's q_pe clamp)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "{\n.reg .f32 x;\nmax.f32 x, $1, 0fC3E00000;\nmin.f32 $0, x, 0f43E00000;\n}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_bits(a, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bits_f32(a, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(a).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_v4(addr, *, loc=None, ip=None):
    """16-byte global load (four 32-bit words)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ld_global_nc_v4(addr, *, loc=None, ip=None):
    """16-byte read-only global load (weights / tables, never written by a predecessor)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ld_global_b32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_b32(addr, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_nc_v2f32(addr, *, loc=None, ip=None):
    """8-byte read-only global load of two FP32 (one cos / sin pair)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def _st_global_v2(addr, lo, hi, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(lo).ir_value(loc=loc, ip=ip),
            Int32(hi).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b32(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b16(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_v2(addr, lo, hi, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(lo).ir_value(loc=loc, ip=ip),
            Int32(hi).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_v4(addr, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_b32(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_b16(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.shared.b16 [$0], h;\n}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v4(addr, *, loc=None, ip=None):
    """16-byte shared load (four FP32)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_async_peer_v4(addr, w0, w1, w2, w3, mbar, peer, *, loc=None, ip=None):
    """16 bytes into the shared memory of cluster CTA ``peer`` at the address of the local
    ``addr``, completing their bytes on that CTA's mbarrier at the address of the local ``mbar``."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Int32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)]
        + [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(peer).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 ra, rb;\n"
        "mapa.shared::cluster.u32 ra, $0, $6;\n"
        "mapa.shared::cluster.u32 rb, $5, $6;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$1, $2, $3, $4}, [rb];\n}",
        "r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bf16x2_to_f32x2(w, *, loc=None, ip=None):
    """A little-endian BF16 pair (one 32-bit word) -> (low, high) as FP32 (exact)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 a, b;\nshl.b32 a, $2, 16;\nand.b32 b, $2, 0xFFFF0000;\nmov.b32 $0, a;\nmov.b32 $1, b;\n}",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def _f32x2_to_bf16x2(lo, hi, *, loc=None, ip=None):
    """Two FP32 -> one word of two BF16 (``lo`` in the low half), round to nearest even."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_f32(a, b, c, *, loc=None, ip=None):
    """``fma.rn.f32``: ``a * b + c`` with one rounding."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul_f32(a, b, *, loc=None, ip=None):
    """``mul.rn.f32`` (not folded or contracted by the compiler)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _load_row(addr: Int64, lane: Int32, vecs: cutlass.Constexpr, read_only: cutlass.Constexpr):
    """This lane's ``8 * vecs`` BF16 values of a row in the FlashInfer RMSNorm value layout (vector
    ``b`` = columns ``8 lane + 256 b + j``, index ``j + 8 b``) as a TensorSSA; 16-byte loads."""
    frag = cute.make_rmem_tensor(cute.make_layout((VEC, vecs)), cutlass.BFloat16)
    words = cute.make_tensor(
        cute.recast_ptr(frag.iterator, dtype=Int32), cute.make_layout((4 * vecs,))
    )
    for b in cutlass.range_constexpr(vecs):
        col = addr + Int64((lane * VEC + 256 * b) * 2)
        if cutlass.const_expr(read_only):
            v = _ld_global_nc_v4(col)
        else:
            v = _ld_global_v4(col)
        for i in cutlass.range_constexpr(4):
            words[4 * b + i] = v[i]
    return frag.load()


@cute.jit
def _rmsnorm_vals(x_bf16, w_bf16, eps: Float32, h: cutlass.Constexpr):
    """One row of the FlashInfer CuTe RMSNorm (``RMSNormKernel``, H <= 3072, BF16, weight bias 0),
    computed by one warp from this lane's values (``_load_row``): returns them normalized, as BF16.

    The arithmetic is the FlashInfer kernel's, op for op: FP32 squares reduced in value order by
    ``TensorSSA.reduce`` (initial 0), a butterfly over the 32 lanes, the mean as ``sum / H``,
    ``rsqrt(mean + eps)`` (fast-math), ``x * rstd * (w + 0)`` rounded to BF16."""
    x = x_bf16.to(Float32)
    x_sq = x * x
    local = x_sq.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
    total = local
    for i in cutlass.range_constexpr(5):
        total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << i)
    mean_sq = total / Float32(h)
    rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
    w = w_bf16.to(Float32)
    y = x * rstd * (w + Float32(0.0))
    return y.to(cutlass.BFloat16)


class K3MlaFusedQKVKernel:
    """``n_pad`` token columns, ``k_splits`` CTAs per cluster (the stock tactic's K partition);
    ``write_q`` also stores the BF16 query (``[M, 2304]``, for tests)."""

    def __init__(self, n_pad: int, k_splits: int, write_q: bool = False, use_pdl: bool = True):
        if n_pad not in SUPPORTED_N_PAD:
            raise ValueError(f"n_pad {n_pad} not in {SUPPORTED_N_PAD}")
        if k_splits not in SUPPORTED_K_SPLITS:
            raise ValueError(f"k_splits {k_splits} not in {SUPPORTED_K_SPLITS}")
        ranges = k_split_ranges(k_splits)
        if len({hi - lo for lo, hi in ranges}) != 1:
            raise ValueError(
                f"k_splits {k_splits} does not divide the {NUM_K_TILES} K tiles evenly"
            )
        self.n = n_pad
        self.k_splits = k_splits
        self.tiles = NUM_K_TILES // k_splits
        self.n_r = lora_cols(k_splits)
        self.write_q = write_q
        self.use_pdl = use_pdl
        self.cta_group = tcgen05.CtaGroup.ONE
        self.mma_inst_shape = (CTA_ROWS, n_pad, MMA_K)
        self.mma_tiler = (CTA_ROWS, n_pad, K_TILE)
        self.abs_tiler = (CTA_ROWS, self.n_r, ABS_K_TILE)
        self.ab_dtype = cutlass.Float8E4M3FN
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.acc_dtype = cutlass.Float32

    def _make_mma(self):
        return sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC,
            self.cta_group,
            self.mma_inst_shape,
            a_source=tcgen05.OperandSource.SMEM,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

    def _make_abs_mma(self):
        # The absorption kernel's MMA (k3_mla_q_absorb: Blackwell helper, kind::f16, K-major A and B).
        return sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            self.cta_group,
            self.abs_tiler[:2],
        )

    @cute.jit
    def __call__(
        self,
        wkc_ptr: cute.Pointer,
        w_addr: Int64,
        w_sf_addr: Int64,
        fa_addr: Int64,
        fa_row_bytes: Int32,
        q_ln_addr: Int64,
        kv_ln_addr: Int64,
        q_eps: Float32,
        kv_eps: Float32,
        out_addr: Int64,
        out_row_bytes: Int32,
        q_out_addr: Int64,
        kv_base_addr: Int64,
        kv_page_bytes: Int64,
        page_table_addr: Int64,
        page_table_row: Int32,
        layer_in_pool: Int32,
        kv_lens_addr: Int64,
        cos_sin_addr: Int64,
        kv_scale: Float32,
        bmm1_addr: Int64,
        bmm2_addr: Int64,
        cu_q_addr: Int64,
        cu_kv_addr: Int64,
        counter_addr: Int64,
        host_bmm1_scale: Float32,
        m: Int32,
        q_len: Int32,
        batch: Int32,
        stream: cuda.CUstream,
    ):
        tiled_mma = self._make_mma()
        tiles = self.tiles
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, tiles
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, tiles
        )
        sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, tiles
        )
        sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, tiles
        )
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfa_smem_layout, (None, None, None, 0))
        )
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfb_smem_layout, (None, None, None, 0))
        )
        num_sfa_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, tCtSFA_layout)) & 0x0000FFFF
        )
        num_sfb_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, tCtSFB_layout)) & 0x0000FFFF
        )

        abs_mma = self._make_abs_mma()
        q_smem_layout = sm100_utils.make_smem_layout_a(abs_mma, self.abs_tiler, cutlass.BFloat16, 2)
        w_smem_layout = sm100_utils.make_smem_layout_b(abs_mma, self.abs_tiler, cutlass.BFloat16, 2)
        # w_kc [heads, lora, nope] as (N = lora, K = nope, L = head), K-major.
        wkc_tensor = cute.make_tensor(
            wkc_ptr,
            cute.make_layout(
                (KV_LORA, NOPE_DIM, NUM_HEADS), stride=(NOPE_DIM, 1, KV_LORA * NOPE_DIM)
            ),
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (abs_mma.thr_id.shape,)
        )
        w_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), abs_mma.thr_id)
        tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_B(
            w_op,
            wkc_tensor,
            cute.slice_(w_smem_layout, (None, None, None, 0)),
            self.abs_tiler,
            abs_mma,
            cluster_layout_vmnk.shape,
        )
        w_stage_bytes = cute.size_in_bytes(
            cutlass.BFloat16, cute.slice_(w_smem_layout, (None, None, None, 0))
        )

        class SharedStorage:
            mma_bar: cutlass.Int64
            abs_bar: cutlass.Int64
            mail_bar: cutlass.Int64
            wkc_bar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(a_smem_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(b_smem_layout.outer)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_layout)], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_layout)], 1024
            ]
            mailbox: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.k_splits * CTA_ROWS * self.n], 128
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(q_smem_layout.outer)], 1024
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(w_smem_layout.outer)], 1024
            ]

        self.shared_storage = cute.struct(SharedStorage)
        self.kernel(
            tiled_mma,
            abs_mma,
            tma_atom_w,
            tma_tensor_w,
            a_smem_layout,
            b_smem_layout,
            sfa_smem_layout,
            sfb_smem_layout,
            tCtSFA_layout,
            tCtSFB_layout,
            q_smem_layout,
            w_smem_layout,
            num_sfa_cols,
            num_sfb_cols,
            w_stage_bytes,
            w_addr,
            w_sf_addr,
            fa_addr,
            fa_row_bytes,
            q_ln_addr,
            kv_ln_addr,
            q_eps,
            kv_eps,
            out_addr,
            out_row_bytes,
            q_out_addr,
            kv_base_addr,
            kv_page_bytes,
            page_table_addr,
            page_table_row,
            layer_in_pool,
            kv_lens_addr,
            cos_sin_addr,
            kv_scale,
            bmm1_addr,
            bmm2_addr,
            cu_q_addr,
            cu_kv_addr,
            counter_addr,
            host_bmm1_scale,
            m,
            q_len,
            batch,
        ).launch(
            grid=(self.k_splits, NUM_CLUSTERS, 1),
            block=[THREADS, 1, 1],
            cluster=(self.k_splits, 1, 1),
            smem=max(self.shared_storage.size_in_bytes(), MIN_SMEM_BYTES),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        abs_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        q_smem_layout: cute.ComposedLayout,
        w_smem_layout: cute.ComposedLayout,
        num_sfa_cols: cutlass.Constexpr,
        num_sfb_cols: cutlass.Constexpr,
        w_stage_bytes: cutlass.Constexpr,
        w_addr: Int64,
        w_sf_addr: Int64,
        fa_addr: Int64,
        fa_row_bytes: Int32,
        q_ln_addr: Int64,
        kv_ln_addr: Int64,
        q_eps: Float32,
        kv_eps: Float32,
        out_addr: Int64,
        out_row_bytes: Int32,
        q_out_addr: Int64,
        kv_base_addr: Int64,
        kv_page_bytes: Int64,
        page_table_addr: Int64,
        page_table_row: Int32,
        layer_in_pool: Int32,
        kv_lens_addr: Int64,
        cos_sin_addr: Int64,
        kv_scale: Float32,
        bmm1_addr: Int64,
        bmm2_addr: Int64,
        cu_q_addr: Int64,
        cu_kv_addr: Int64,
        counter_addr: Int64,
        host_bmm1_scale: Float32,
        m: Int32,
        q_len: Int32,
        batch: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, role, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        lane = tidx % 32
        if role < KV_CLUSTER:
            self._gemv_cta(
                tiled_mma,
                abs_mma,
                tma_atom_w,
                mW,
                a_smem_layout,
                b_smem_layout,
                sfa_smem_layout,
                sfb_smem_layout,
                tCtSFA_layout,
                tCtSFB_layout,
                q_smem_layout,
                w_smem_layout,
                num_sfa_cols,
                num_sfb_cols,
                w_stage_bytes,
                w_addr,
                w_sf_addr,
                fa_addr,
                fa_row_bytes,
                q_ln_addr,
                q_eps,
                out_addr,
                out_row_bytes,
                q_out_addr,
                m,
                tidx,
                role,
                rank,
                warp_idx,
                lane,
            )
        else:
            self._kv_cta(
                fa_addr,
                fa_row_bytes,
                kv_ln_addr,
                kv_eps,
                kv_base_addr,
                kv_page_bytes,
                page_table_addr,
                page_table_row,
                layer_in_pool,
                kv_lens_addr,
                cos_sin_addr,
                kv_scale,
                bmm1_addr,
                bmm2_addr,
                cu_q_addr,
                cu_kv_addr,
                counter_addr,
                host_bmm1_scale,
                m,
                q_len,
                batch,
                rank,
                warp_idx,
                lane,
            )

    @cute.jit
    def _gemv_cta(
        self,
        tiled_mma: cute.TiledMma,
        abs_mma: cute.TiledMma,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        q_smem_layout: cute.ComposedLayout,
        w_smem_layout: cute.ComposedLayout,
        num_sfa_cols: cutlass.Constexpr,
        num_sfb_cols: cutlass.Constexpr,
        w_stage_bytes: cutlass.Constexpr,
        w_addr: Int64,
        w_sf_addr: Int64,
        fa_addr: Int64,
        fa_row_bytes: Int32,
        q_ln_addr: Int64,
        q_eps: Float32,
        out_addr: Int64,
        out_row_bytes: Int32,
        q_out_addr: Int64,
        m: Int32,
        tidx: Int32,
        role: Int32,
        rank: Int32,
        warp_idx: Int32,
        lane: Int32,
    ):
        n = self.n
        ck = self.k_splits
        tiles = self.tiles
        chunks = n // 4
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout)
        sQ = storage.sQ.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sW = storage.sW.get_tensor(w_smem_layout.outer, swizzle=w_smem_layout.inner)
        mma_bar = storage.mma_bar.ptr
        abs_bar = storage.abs_bar.ptr
        mail_bar = storage.mail_bar.ptr
        wkc_bar = storage.wkc_bar.ptr
        is_nope = role < NOPE_CLUSTERS

        # ---- barriers; the mailbox expects the (ck - 1) partials of the peers (128 rows x n FP32 each)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mma_bar, 1)
                cute.arch.mbarrier_init(abs_bar, 1)
                cute.arch.mbarrier_init(mail_bar, 1)
                cute.arch.mbarrier_init(wkc_bar, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(mail_bar, (ck - 1) * CTA_ROWS * n * 4)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        # Peers push into this CTA's mailbox only after the cluster barrier (barriers initialized).
        cute.arch.cluster_arrive_relaxed()

        # ---- before the PDL wait: the absorption weights of this rank's latent columns (TMA)
        if is_nope:
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(wkc_bar, 2 * w_stage_bytes)
                gW = cute.local_tile(
                    mW, cute.slice_(self.abs_tiler, (0, None, None)), (None, None, None)
                )
                tCgW = abs_mma.get_slice(0).partition_B(gW)
                tWsW, tWgW = cpasync.tma_partition(
                    tma_atom_w,
                    0,
                    cute.make_layout((1,)),
                    cute.group_modes(sW, 0, 3),
                    cute.group_modes(tCgW, 0, 3),
                )
                for kt in cutlass.range_constexpr(2):
                    cute.copy(
                        tma_atom_w,
                        tWgW[(None, rank, None, role)][(None, kt)],
                        tWsW[(None, kt)],
                        tma_bar_ptr=wkc_bar,
                    )

        # ---- this cluster's 128 weight rows (two runs of 64; nope: head `role`, rope: heads 2p, 2p + 1)
        pair = role - NOPE_CLUSTERS
        row0 = cutlass.select_(is_nope, HEAD_DIM * role, 2 * HEAD_DIM * pair + NOPE_DIM)
        row1 = cutlass.select_(
            is_nope, HEAD_DIM * role + 64, 2 * HEAD_DIM * pair + HEAD_DIM + NOPE_DIM
        )
        k_lo = rank * tiles

        # ---- TMEM: GEMV accumulator, weight and activation scales, absorption accumulator
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS),
            allocator_warp_id=0,
        )
        tmem.allocate(TMEM_COLS)

        # Weight rows of this rank's K tiles -> the MMA's A operand (K-major, 128-byte swizzle), 16 bytes
        # per load; chunk c of a tile = (row c // 8, 16-byte column c % 8).
        sA_base = storage.sA.data_ptr().toint()
        per_thread = tiles * CTA_ROWS * (K_TILE // 16) // THREADS
        group = 8
        for g in cutlass.range_constexpr((per_thread + group - 1) // group):
            vals = []
            for i in cutlass.range_constexpr(g * group, min((g + 1) * group, per_thread)):
                cc = (i % 4) * THREADS + tidx
                rw = cc // 8
                grw = cutlass.select_(rw < 64, row0, row1) + rw % 64
                srw = (
                    w_addr
                    + Int64(grw) * Int64(Q_LORA)
                    + Int64((k_lo + i // 4) * K_TILE + (cc % 8) * 16)
                )
                vals.append(_ld_global_nc_v4(srw))
            for i in cutlass.range_constexpr(g * group, min((g + 1) * group, per_thread)):
                cc = (i % 4) * THREADS + tidx
                rw = cc // 8
                vw = vals[i - g * group]
                _st_shared_v4(
                    sA_base
                    + (i // 4) * (CTA_ROWS * K_TILE)
                    + rw * K_TILE
                    + (((cc % 8) ^ (rw % 8)) * 16),
                    vw[0],
                    vw[1],
                    vw[2],
                    vw[3],
                )
        # Their scales -> the R128c4 SFA tiles (4 bytes per row and K tile; the global scales are
        # 128-row x 4-column atoms, K atoms contiguous).
        sfa_base = storage.sSFA.data_ptr().toint()
        sfa_words = tiles * CTA_ROWS
        for i in cutlass.range_constexpr((sfa_words + THREADS - 1) // THREADS):
            wv = i * THREADS + tidx
            if wv < sfa_words:
                jsf = wv // CTA_ROWS
                rsf = wv % CTA_ROWS
                gsf = cutlass.select_(rsf < 64, row0, row1) + rsf % 64
                ssf = w_sf_addr + Int64(
                    ((gsf // 128) * NUM_K_TILES + k_lo + jsf) * SF_TILE_BYTES
                    + (gsf % 32) * 16
                    + ((gsf // 32) % 4) * 4
                )
                _st_shared_b32(
                    sfa_base + jsf * SF_TILE_BYTES + (rsf % 32) * 16 + (rsf // 32) * 4,
                    _ld_global_nc_b32(ssf),
                )
        # Zero the activation scale tiles (rows >= m stay zero, as the stock quantizer's padding rows).
        sfb_base = storage.sSFB.data_ptr().toint()
        for i in cutlass.range_constexpr((tiles * SF_TILE_BYTES // 4 + THREADS - 1) // THREADS):
            idx = tidx + i * THREADS
            if idx < tiles * SF_TILE_BYTES // 4:
                _st_shared_b32(sfb_base + idx * 4, Int32(0))
        # Generic-proxy stores -> the tensor core's async proxy; the barrier of wait_for_alloc orders them.
        cute.arch.fence_proxy("async.shared", space="cta")
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_layout = tiled_mma.make_fragment_C(acc_shape).layout

        # The RMSNorm weight and the quantizer constant (never written by a predecessor).
        w_ln = _load_row(q_ln_addr, lane, Q_VECS, True)
        rcp448 = _rcp_approx_ftz(Float32(448.0))
        # Weight scales SMEM -> TMEM (warp 0 issues the UTCCP copies).
        if warp_idx == 0:
            for jsa in cutlass.range_constexpr(tiles):
                self._s2t(tmem_ptr + SFA_COL + jsa * num_sfa_cols, tCtSFA_layout, sSFA, jsa)
        cute.arch.cluster_wait()

        # ---- the predecessor's output from here on
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()

        # ---- RMSNorm (whole row) + 1x128 FP8 quantization of this rank's K tiles of every token row
        # into the MMA's B operand. Warp w owns rows w, w + 8, ...: all its row loads first (a row past
        # m re-reads row m - 1 and is dropped), then the math.
        sB_base = storage.sB.data_ptr().toint()
        xs = []
        for rr in cutlass.range_constexpr(n // WARPS):
            t_ld = cutlass.min(Int32(rr * WARPS) + warp_idx, m - 1)
            xs.append(_load_row(fa_addr + Int64(t_ld) * Int64(fa_row_bytes), lane, Q_VECS, False))
        for rr in cutlass.range_constexpr(n // WARPS):
            tq = Int32(rr * WARPS) + warp_idx
            if tq < m:
                y = _rmsnorm_vals(xs[rr], w_ln, q_eps, Q_LORA)
                self._quantize_row_to_smem(y, tq, lane, rcp448, sB_base, sfb_base, k_lo)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier()

        # ---- MMAs: the chain over this rank's K tiles (in K order) into the GEMV accumulator
        if warp_idx == 0:
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kblocks = cute.size(tCrB, mode=[2])
            tCtAcc = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + ACC_COL, dtype=self.acc_dtype), acc_layout
            )
            for jm in cutlass.range_constexpr(tiles):
                # This K tile's activation scales, SMEM -> TMEM right before its MMAs (one thread
                # issues both: the tensor pipe runs them in issue order).
                self._s2t(tmem_ptr + SFB_COL + jm * num_sfb_cols, tCtSFB_layout, sSFB, jm)
                tCtSFA = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + SFA_COL + jm * num_sfa_cols, dtype=self.sf_dtype),
                    tCtSFA_layout,
                )
                tCtSFB = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + SFB_COL + jm * num_sfb_cols, dtype=self.sf_dtype),
                    tCtSFB_layout,
                )
                for kbm in cutlass.range_constexpr(num_kblocks):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, jm != 0 or kbm != 0)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        [tCrA[(None, None, kbm, jm)], tCtSFA[(None, None, kbm)]],
                        [tCrB[(None, None, kbm, jm)], tCtSFB[(None, None, kbm)]],
                        tCtAcc,
                    )
            with cute.arch.elect_one():
                tcgen05.commit(mma_bar)
        cute.arch.mbarrier_wait(mma_bar, 0)

        # ---- K-rank reduction (warps 0-3, TMEM lane = weight row): every rank pushes its FP32 partial
        # to every peer's mailbox slot of its rank, waits for the peers' partials, and sums the ck
        # partials in K-rank order (its own in its position): every rank holds the stock sum.
        mail_base = storage.mailbox.data_ptr().toint()
        mail_bar_addr = mail_bar.toint()
        if warp_idx < 4:
            row = tidx
            acc0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + ACC_COL, dtype=self.acc_dtype), acc_layout
            )
            acc_view = acc0[((None, None), 0, 0)]
            t2r = tcgen05.make_tmem_copy(
                cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(n)), self.acc_dtype),
                acc_view,
            )
            thr_t2r = t2r.get_slice(tidx)
            tTR_t = thr_t2r.partition_S(acc_view)
            # Register fragments shaped by the copy's destination view (this thread's row: n values).
            tTR_c = thr_t2r.partition_D(cute.make_identity_tensor((CTA_ROWS, n)))
            own = cute.make_rmem_tensor(tTR_c.shape, self.acc_dtype)
            cute.copy(t2r, tTR_t, own)
            # tcgen05.ld writes its registers asynchronously: none is read before this wait.
            cute.arch.fence_view_async_tmem_load()
            for d in cutlass.range_constexpr(ck):
                if rank != d:
                    for cx in cutlass.range_constexpr(chunks):
                        _st_async_peer_v4(
                            mail_base + ((rank * chunks + cx) * CTA_ROWS + row) * 16,
                            _f32_bits(own[4 * cx]),
                            _f32_bits(own[4 * cx + 1]),
                            _f32_bits(own[4 * cx + 2]),
                            _f32_bits(own[4 * cx + 3]),
                            mail_bar_addr,
                            Int32(d),
                        )
            cute.arch.mbarrier_wait(mail_bar, 0)
            total = cute.make_rmem_tensor(tTR_c.shape, self.acc_dtype)
            for sr in cutlass.range_constexpr(ck):
                for cx in cutlass.range_constexpr(chunks):
                    mv = _ld_shared_v4(mail_base + ((sr * chunks + cx) * CTA_ROWS + row) * 16)
                    for e in cutlass.range_constexpr(4):
                        val = cutlass.select_(rank == sr, own[4 * cx + e], mv[e])
                        if cutlass.const_expr(sr == 0):
                            total[4 * cx + e] = val
                        else:
                            total[4 * cx + e] = total[4 * cx + e] + val
            # BF16 query rows (two tokens per word).
            qw = []
            for tp in cutlass.range_constexpr(n // 2):
                qw.append(_f32x2_to_bf16x2(total[2 * tp], total[2 * tp + 1]))
            grq = cutlass.select_(row < 64, row0, row1) + row % 64
            if cutlass.const_expr(self.write_q):
                if rank == 0:
                    for tp in cutlass.range_constexpr(n // 2):
                        if 2 * tp < m:
                            _st_global_b16(
                                q_out_addr + Int64(2 * tp) * Int64(QB_ROWS * 2) + Int64(grq * 2),
                                qw[tp],
                            )
                        if 2 * tp + 1 < m:
                            _st_global_b16(
                                q_out_addr
                                + Int64(2 * tp + 1) * Int64(QB_ROWS * 2)
                                + Int64(grq * 2),
                                qw[tp] >> 16,
                            )
            if is_nope:
                # q_nope of head `role` -> the absorption's A operand (tokens as rows, NoPE dim `row`
                # as K; K-major, 128-byte swizzle, two K tiles of 64).
                sQ_base = storage.sQ.data_ptr().toint()
                kq = row % ABS_K_TILE
                col_bytes = (row // ABS_K_TILE) * (CTA_ROWS * ABS_K_TILE * 2) + (kq * 2) % 16
                for tok in cutlass.range_constexpr(n):
                    addr = (
                        sQ_base
                        + col_bytes
                        + (tok // 8) * 1024
                        + (tok % 8) * 128
                        + ((((kq * 2) // 16) ^ (tok % 8)) * 16)
                    )
                    _st_shared_b16(addr, qw[tok // 2] >> (16 * (tok % 2)))
                cute.arch.fence_proxy("async.shared", space="cta")
                cutlass.pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
                if warp_idx == 0:
                    cute.arch.mbarrier_wait(wkc_bar, 0)
                    tCrQ = abs_mma.make_fragment_A(sQ)
                    tCrW = abs_mma.make_fragment_B(sW)
                    abs_layout = abs_mma.make_fragment_C(
                        abs_mma.partition_shape_C(self.abs_tiler[:2])
                    ).layout
                    tAbs = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + ABS_COL, dtype=self.acc_dtype), abs_layout
                    )
                    abs_kblocks = cute.size(tCrW, mode=[2])
                    for sa in cutlass.range_constexpr(2):
                        for kba in cutlass.range_constexpr(abs_kblocks):
                            abs_mma.set(tcgen05.Field.ACCUMULATE, sa != 0 or kba != 0)
                            cute.gemm(
                                abs_mma,
                                tAbs,
                                tCrQ[(None, None, kba, sa)],
                                tCrW[(None, None, kba, sa)],
                                tAbs,
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(abs_bar)
            else:
                # q_pe of heads 2p + row // 64 (dim row % 64) -> columns 512.. of the FP8 query;
                # 4 lanes (4 consecutive dims) make one 32-bit store, rank r stores tokens tok = r (mod ck).
                head = 2 * (role - NOPE_CLUSTERS) + row // 64
                col = Int64(KV_LORA + (row % 64) - (row % 4))
                for tok in cutlass.range_constexpr(n):
                    pe = _clamp448(_bits_f32((qw[tok // 2] >> (16 * (tok % 2))) << 16))
                    byte = _e4m3x2(pe, Float32(0.0)) & 0xFF
                    b1 = cute.arch.shuffle_sync_down(byte, offset=1)
                    b2 = cute.arch.shuffle_sync_down(byte, offset=2)
                    b3 = cute.arch.shuffle_sync_down(byte, offset=3)
                    word = byte | (b1 << 8) | (b2 << 16) | (b3 << 24)
                    if lane % 4 == 0 and tok < m and tok % ck == rank:
                        _st_global_b32(
                            out_addr
                            + Int64(tok) * Int64(out_row_bytes)
                            + Int64(head * OUT_HEAD)
                            + col,
                            word,
                        )

        # ---- absorption epilogue (warps 0 and 4: TMEM lanes 0-31 = tokens): max(acc, -448) -> E4M3,
        # 16 latent columns per 16-byte store; the two warps split the 16-column chunks.
        if is_nope:
            if warp_idx == 0 or warp_idx == 4:
                cute.arch.mbarrier_wait(abs_bar, 0)
                abs_layout_e = abs_mma.make_fragment_C(
                    abs_mma.partition_shape_C(self.abs_tiler[:2])
                ).layout
                abs_view = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + ABS_COL, dtype=self.acc_dtype), abs_layout_e
                )[((None, None), 0, 0)]
                tok_e = lane
                n_chunks = self.n_r // 16
                half = (n_chunks + 1) // 2
                for ce in cutlass.range_constexpr(n_chunks):
                    owner = 0 if ce < half else 4
                    if warp_idx == owner:
                        view = cute.local_tile(abs_view, (CTA_ROWS, 16), (0, ce))
                        t2r16 = tcgen05.make_tmem_copy(
                            cute.make_copy_atom(
                                tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), self.acc_dtype
                            ),
                            view,
                        )
                        thr16 = t2r16.get_slice(lane)
                        frag_c = thr16.partition_D(cute.make_identity_tensor((CTA_ROWS, 16)))
                        accv = cute.make_rmem_tensor(frag_c.shape, self.acc_dtype)
                        cute.copy(t2r16, thr16.partition_S(view), accv)
                        cute.arch.fence_view_async_tmem_load()
                        col0 = rank * self.n_r + 16 * ce
                        if tok_e < m and col0 < KV_LORA:
                            words = []
                            for q4 in cutlass.range_constexpr(4):
                                lo = _e4m3x2(
                                    _max_neg448(accv[4 * q4]), _max_neg448(accv[4 * q4 + 1])
                                )
                                hi = _e4m3x2(
                                    _max_neg448(accv[4 * q4 + 2]), _max_neg448(accv[4 * q4 + 3])
                                )
                                words.append(lo | (hi << 16))
                            _st_global_v4(
                                out_addr
                                + Int64(tok_e) * Int64(out_row_bytes)
                                + Int64(role * OUT_HEAD + col0),
                                words[0],
                                words[1],
                                words[2],
                                words[3],
                            )
        cute.arch.barrier()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _quantize_row_to_smem(
        self, y, t: Int32, lane: Int32, rcp448: Float32, sB_base, sSFB_base, k_lo: Int32
    ):
        """``fp8_quantize_1x128_r128c4`` of the K tiles ``[k_lo, k_lo + tiles)`` of one normalized row
        (this lane's 48 BF16 values; vector ``b`` holds K tile ``2 b`` in lanes 0-15 and ``2 b + 1``
        in lanes 16-31): per 128-column block the amax (>= 1e-10) of the BF16 values, the UE8M0
        exponent of ``amax * rcp.approx.ftz(448)`` rounded up (<= 254), E4M3 codes of
        ``x * 2^(127 - e)`` (round to nearest even, saturating) into the B operand (K-major, 128-byte
        swizzle; local tile ``kt - k_lo``) and the exponent byte four times into the R128c4 scale tile."""
        n = self.n
        tiles = self.tiles
        yf = cute.make_rmem_tensor(cute.make_layout((VEC, Q_VECS)), Float32)
        yf.store(y.to(Float32))
        for b in cutlass.range_constexpr(Q_VECS):
            # Warp-uniform: does this rank own either K tile of vector b?
            if 2 * b + 1 >= k_lo and 2 * b < k_lo + tiles:
                amax = _fabs(yf[0, b])
                for j in cutlass.range_constexpr(1, VEC):
                    amax = _fmax(amax, _fabs(yf[j, b]))
                # The 16 lanes holding one 128-column block (lanes 0-15: block 2b, 16-31: 2b + 1).
                for i in cutlass.range_constexpr(4):
                    amax = _fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << i))
                amax = _fmax(amax, Float32(1e-10))
                raw = _mul_f32(amax, rcp448)
                bits = _f32_bits(raw)
                e = ((bits >> 23) & 0xFF) + cutlass.select_(
                    (bits & 0x7FFFFF) != 0, Int32(1), Int32(0)
                )
                e = cutlass.min(e, Int32(254))
                qscale = _bits_f32((Int32(254) - e) << 23)
                w0 = _e4m3x2(_mul_f32(yf[0, b], qscale), _mul_f32(yf[1, b], qscale)) | (
                    _e4m3x2(_mul_f32(yf[2, b], qscale), _mul_f32(yf[3, b], qscale)) << 16
                )
                w1 = _e4m3x2(_mul_f32(yf[4, b], qscale), _mul_f32(yf[5, b], qscale)) | (
                    _e4m3x2(_mul_f32(yf[6, b], qscale), _mul_f32(yf[7, b], qscale)) << 16
                )
                kt = 2 * b + lane // 16
                if kt >= k_lo and kt < k_lo + tiles:
                    j = kt - k_lo
                    chunk = (lane % 16) // 2
                    addr = (
                        sB_base
                        + j * (n * K_TILE)
                        + t * K_TILE
                        + ((chunk ^ (t % 8)) * 16)
                        + (lane % 2) * 8
                    )
                    _st_shared_v2(addr, w0, w1)
                    if lane % 16 == 0:
                        _st_shared_b32(
                            sSFB_base + j * SF_TILE_BYTES + (t % 32) * 16 + (t // 32) * 4,
                            e * 0x01010101,
                        )

    @cute.jit
    def _kv_cta(
        self,
        fa_addr: Int64,
        fa_row_bytes: Int32,
        kv_ln_addr: Int64,
        kv_eps: Float32,
        kv_base_addr: Int64,
        kv_page_bytes: Int64,
        page_table_addr: Int64,
        page_table_row: Int32,
        layer_in_pool: Int32,
        kv_lens_addr: Int64,
        cos_sin_addr: Int64,
        kv_scale: Float32,
        bmm1_addr: Int64,
        bmm2_addr: Int64,
        cu_q_addr: Int64,
        cu_kv_addr: Int64,
        counter_addr: Int64,
        host_bmm1_scale: Float32,
        m: Int32,
        q_len: Int32,
        batch: Int32,
        rank: Int32,
        warp_idx: Int32,
        lane: Int32,
    ):
        """The KV half of ``applyMLARopeAndAssignQKVKernelGeneration`` for this step's rows (FP8
        cache): the normalized latent and the RoPE'd ``k_pe`` of token ``t`` (request ``t // q_len``,
        position ``kv_len - q_len + t % q_len``) into row ``pos % 64`` of page ``pos // 64``; plus
        the FMHA scales, ``cu_q``, ``cu_kv`` and the scheduler counter. Warp ``w`` of K rank ``r``
        takes the rows ``t = 8 (r + ck i) + w``."""
        ck = self.k_splits
        rounds = (self.n + WARPS * ck - 1) // (WARPS * ck)
        # Before the PDL wait only the kv_a RMSNorm weight (a parameter, never written in a step).
        kv_w = _load_row(kv_ln_addr, lane, KV_VECS, True)
        # Everything else after it: the KV lengths and the page table are step metadata that an
        # earlier kernel of the step may write (the stock RoPE kernel also reads them after its wait).
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
        # Per row of this warp (a row past m re-reads row m - 1 and is dropped): its KV length, then
        # its cache row address and the (cos, sin) of the four k_pe pairs of lane % 8. The rows of the
        # predecessor's output (kv_a, k_pe) and the scan's KV length are loaded with the KV lengths.
        reqs = []
        kv_lens = []
        for rr in cutlass.range_constexpr(rounds):
            t_md = cutlass.min(Int32(rr * WARPS * ck) + rank * WARPS + warp_idx, m - 1)
            reqs.append(t_md // q_len)
            kv_lens.append(_ld_global_b32(kv_lens_addr + Int64((t_md // q_len) * 4)))
        kv_len_b = Int32(0)
        if lane < batch:
            kv_len_b = _ld_global_b32(kv_lens_addr + Int64(lane * 4))
        xs = []
        kps = []
        for rr in cutlass.range_constexpr(rounds):
            t_ld = cutlass.min(Int32(rr * WARPS * ck) + rank * WARPS + warp_idx, m - 1)
            row_addr = fa_addr + Int64(t_ld) * Int64(fa_row_bytes)
            xs.append(_load_row(row_addr + Int64(Q_LORA * 2), lane, KV_VECS, False))
            kps.append(_ld_global_v4(row_addr + Int64((Q_LORA + KV_LORA + (lane % 8) * VEC) * 2)))
        poss = []
        pages = []
        coefs = []
        for rr in cutlass.range_constexpr(rounds):
            t_md = cutlass.min(Int32(rr * WARPS * ck) + rank * WARPS + warp_idx, m - 1)
            pos = kv_lens[rr] - q_len + t_md % q_len
            poss.append(pos)
            pages.append(
                _ld_global_b32(
                    page_table_addr
                    + Int64(reqs[rr]) * Int64(page_table_row * 4)
                    + Int64((pos // PAGE_TOKENS) * 4)
                )
            )
            cs_row = []
            for i in cutlass.range_constexpr(4):
                cs_row.append(
                    _ld_global_nc_v2f32(
                        cos_sin_addr
                        + (Int64(pos) * Int64(ROPE_DIM) + Int64((lane % 8) * 4 + i)) * Int64(8)
                    )
                )
            coefs.append(cs_row)
        dsts = []
        for rr in cutlass.range_constexpr(rounds):
            page = pages[rr] + layer_in_pool
            dsts.append(
                kv_base_addr
                + Int64(page) * kv_page_bytes
                + Int64((poss[rr] % PAGE_TOKENS) * KV_ROW)
            )
        # The prefix sum of the KV lengths (lane b = request b; at most MAX_TOKENS requests).
        incl = kv_len_b
        for i in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(incl, offset=1 << i)
            if lane >= (1 << i):
                incl = incl + up
        # Scales and scheduler buffers (rank 0, warp 0): cu_kv[b] = the exclusive sum, cu_kv[batch] = the total.
        if rank == 0 and warp_idx == 0:
            if lane < batch:
                _st_global_b32(cu_kv_addr + Int64(lane * 4), incl - kv_len_b)
                _st_global_b32(cu_q_addr + Int64((lane + 1) * 4), (lane + 1) * q_len * NUM_HEADS)
            if lane == batch - 1:
                _st_global_b32(cu_kv_addr + Int64(batch * 4), incl)
            if lane == 0:
                _st_global_b32(cu_q_addr, Int32(0))
                _st_global_b32(counter_addr, Int32(0))
                _st_global_b32(bmm1_addr, _f32_bits(host_bmm1_scale))
                _st_global_b32(
                    bmm1_addr + Int64(4), _f32_bits(_mul_f32(host_bmm1_scale, Float32(LOG2E)))
                )
                _st_global_b32(bmm2_addr, _f32_bits(Float32(1.0)))
        for rr in cutlass.range_constexpr(rounds):
            t = Int32(rr * WARPS * ck) + rank * WARPS + warp_idx
            if t < m:
                dst = dsts[rr]
                # Latent: RMSNorm(kv_a) -> BF16 -> FP32 * kv_scale -> E4M3, the RoPE kernel's quantCopy.
                y = _rmsnorm_vals(xs[rr], kv_w, kv_eps, KV_LORA)
                yf = cute.make_rmem_tensor(cute.make_layout((VEC, KV_VECS)), Float32)
                yf.store(y.to(Float32))
                for b in cutlass.range_constexpr(KV_VECS):
                    w0 = _e4m3x2(_mul_f32(yf[0, b], kv_scale), _mul_f32(yf[1, b], kv_scale)) | (
                        _e4m3x2(_mul_f32(yf[2, b], kv_scale), _mul_f32(yf[3, b], kv_scale)) << 16
                    )
                    w1 = _e4m3x2(_mul_f32(yf[4, b], kv_scale), _mul_f32(yf[5, b], kv_scale)) | (
                        _e4m3x2(_mul_f32(yf[6, b], kv_scale), _mul_f32(yf[7, b], kv_scale)) << 16
                    )
                    _st_global_v2(dst + Int64(lane * VEC + 256 * b), w0, w1)
                # k_pe: GPT-J rotation of the pairs by the (cos, sin) of this position, BF16, then as above.
                if lane < ROPE_DIM // VEC:
                    out_words = []
                    for i in cutlass.range_constexpr(4):
                        x0, x1 = _bf16x2_to_f32x2(kps[rr][i])
                        cs = coefs[rr][i]
                        rot_x = _fma_f32(cs[0], x0, -_mul_f32(cs[1], x1))
                        rot_y = _fma_f32(cs[0], x1, _mul_f32(cs[1], x0))
                        rb = _f32x2_to_bf16x2(rot_x, rot_y)
                        r0, r1 = _bf16x2_to_f32x2(rb)
                        out_words.append(_e4m3x2(_mul_f32(r0, kv_scale), _mul_f32(r1, kv_scale)))
                    _st_global_v2(
                        dst + Int64(KV_LORA + lane * VEC),
                        out_words[0] | (out_words[1] << 16),
                        out_words[2] | (out_words[3] << 16),
                    )

    @cute.jit
    def _s2t(self, tmem_dst, tmem_sf_layout, sSF, kt: cutlass.Constexpr):
        """Scale factors of K tile ``kt``, SMEM -> TMEM (UTCCP 4x32x128b; call from warp 0)."""
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tSF = cute.make_tensor(cute.recast_ptr(tmem_dst, dtype=self.sf_dtype), tmem_sf_layout)
        sSF_c = cute.filter_zeros(sSF)
        tSF_c = cute.filter_zeros(tSF)
        tiled_s2t = tcgen05.make_s2t_copy(atom, tSF_c)
        thr_s2t = tiled_s2t.get_slice(0)
        mn_dim = cute.get(sSF_c.layout, mode=[0, 0])
        mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
        lay = cute.append(cute.group_modes(mn_dim, 0), cute.get(sSF_c.layout, mode=[0, 1]))
        lay = cute.append(cute.group_modes(lay, 0), cute.get(sSF_c.layout, mode=[1]))
        lay = cute.append(lay, cute.get(sSF_c.layout, mode=[2]))
        lay = cute.append(lay, cute.get(sSF_c.layout, mode=[3]))
        sSF_b = cute.make_tensor(sSF_c.iterator, lay)
        s_part = tcgen05.get_s2t_smem_desc_tensor(tiled_s2t, thr_s2t.partition_S(sSF_b))
        t_part = thr_s2t.partition_D(tSF_c)
        cute.copy(tiled_s2t, s_part[(None, None, None, None, kt)], t_part)


@functools.cache
def _compiled(n_pad: int, k_splits: int, write_q: bool, use_pdl: bool):
    kernel = K3MlaFusedQKVKernel(n_pad, k_splits, write_q, use_pdl)
    bf = cutlass.BFloat16
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    one = Int64(1 << 20)
    return cute.compile(
        kernel,
        make_ptr(bf, 1 << 20, cute.AddressSpace.gmem, assumed_align=16),
        one,
        one,
        one,
        Int32(FUSED_A_COLS * 2),
        one,
        one,
        Float32(1e-6),
        Float32(1e-6),
        one,
        Int32(NUM_HEADS * OUT_HEAD),
        one,
        one,
        Int64(PAGE_TOKENS * KV_ROW),
        one,
        Int32(1),
        Int32(0),
        one,
        one,
        Float32(1.0),
        one,
        one,
        one,
        one,
        one,
        Float32(1.0),
        Int32(1),
        Int32(1),
        Int32(1),
        stream,
    )


def compile_all(use_pdl: bool = TRTLLM_ENABLE_PDL, k_splits=SUPPORTED_K_SPLITS) -> None:
    """Compile every (n_pad, k_splits) variant ahead of CUDA graph capture."""
    for n in SUPPORTED_N_PAD:
        for ks in k_splits:
            _compiled(n, ks, False, use_pdl)


def host_bmm1_scale(q_scaling: float, qk_head_dim: int) -> float:
    """``1.f / (q_scaling * sqrt(float(qk_head_dim)))`` as the stock MLA RoPE op computes it (FP32 result)."""
    import numpy as np

    return float(np.float32(1.0 / (q_scaling * float(np.sqrt(np.float32(qk_head_dim))))))


def run_k3_mla_fused_qkv(
    fused_a: torch.Tensor,
    q_ln_weight: torch.Tensor,
    kv_ln_weight: torch.Tensor,
    q_eps: float,
    kv_eps: float,
    w_qb: torch.Tensor,
    w_qb_sf: torch.Tensor,
    w_kc: torch.Tensor,
    quant_q: torch.Tensor,
    kv_base_ptr: int,
    kv_page_bytes: int,
    page_table: torch.Tensor,
    layer_in_pool: int,
    kv_lens: torch.Tensor,
    cos_sin: torch.Tensor,
    bmm1_scale: torch.Tensor,
    bmm2_scale: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    scheduler_counter: torch.Tensor,
    host_bmm1: float,
    q_len: int,
    k_splits: int,
    *,
    q_out: torch.Tensor | None = None,
    kv_scale: float = 1.0,
    use_pdl: bool = TRTLLM_ENABLE_PDL,
) -> None:
    """Launch the fused q path (see the module docstring) for ``M = fused_a.shape[0]`` tokens of
    ``M // q_len`` requests. ``fused_a``: BF16 ``[M, >= 2112]`` (unit last stride; q_a, kv_a, k_pe);
    ``w_qb`` / ``w_qb_sf``: the q_b FP8 codes ``[2304, 1536]`` and their UE8M0 R128c4 scales;
    ``w_kc``: the BF16 absorption weight ``[12, 512, 128]``; ``quant_q``: the FP8 (or uint8) query
    ``[M, 12, 576]`` (contiguous); ``q_out``: optional BF16 ``[M, 2304]`` that also receives the
    query before the absorption; ``kv_base_ptr``: the byte address of page 0 of the layer's FP8
    pool view, ``kv_page_bytes`` the byte stride of its pages (rows of 576 bytes inside a page);
    ``page_table``: int32 ``[>= batch, pages]`` page ids (row stride ``page_table.stride(0)``) to
    which ``layer_in_pool`` is added; ``kv_lens``: int32 ``[batch]`` KV lengths including this
    step's tokens; ``cos_sin``: FP32 ``[positions, 2 * 64]`` (cos, sin) table (``64 * pos + i`` =
    the pair of rotation ``i``, as the MLA RoPE kernel reads it), long enough for every position of
    the step; the scale / scheduler tensors as the MLA RoPE op fills them; ``k_splits``: the K
    partition of the q_b GEMM tactic (2, 3 or 4)."""
    m = fused_a.shape[0]
    if m < 1 or m > MAX_TOKENS or m % q_len:
        raise ValueError(f"m {m} must be in [1, {MAX_TOKENS}] and a multiple of q_len {q_len}")
    if k_splits not in SUPPORTED_K_SPLITS:
        raise ValueError(f"k_splits {k_splits} not in {SUPPORTED_K_SPLITS}")
    batch = m // q_len
    if fused_a.dtype != torch.bfloat16 or fused_a.stride(1) != 1 or fused_a.shape[1] < FUSED_A_COLS:
        raise ValueError("fused_a must be BF16 [M, >= 2112] with unit last stride")
    if (fused_a.stride(0) * 2) % 16 or fused_a.data_ptr() % 16:
        raise ValueError("fused_a rows must be 16-byte aligned")
    if (
        w_qb.dtype != torch.float8_e4m3fn
        or tuple(w_qb.shape) != (QB_ROWS, Q_LORA)
        or not w_qb.is_contiguous()
    ):
        raise ValueError("w_qb must be contiguous FP8 [2304, 1536]")
    if (
        w_qb_sf.dtype != torch.uint8
        or w_qb_sf.numel() != QB_ROWS * Q_LORA // SF_VEC
        or not w_qb_sf.is_contiguous()
    ):
        raise ValueError("w_qb_sf must be the contiguous UE8M0 R128c4 scales of w_qb")
    if (
        w_kc.dtype != torch.bfloat16
        or tuple(w_kc.shape) != (NUM_HEADS, KV_LORA, NOPE_DIM)
        or not w_kc.is_contiguous()
        or w_kc.data_ptr() % 16
    ):
        raise ValueError("w_kc must be contiguous, 16-byte aligned BF16 [12, 512, 128]")
    if (
        quant_q.dtype not in (torch.float8_e4m3fn, torch.uint8)
        or tuple(quant_q.shape) != (m, NUM_HEADS, OUT_HEAD)
        or not quant_q.is_contiguous()
        or quant_q.data_ptr() % 16
    ):
        raise ValueError("quant_q must be a contiguous, 16-byte aligned FP8 [M, 12, 576]")
    if q_out is not None and (
        q_out.dtype != torch.bfloat16
        or tuple(q_out.shape) != (m, QB_ROWS)
        or not q_out.is_contiguous()
    ):
        raise ValueError("q_out must be contiguous BF16 [M, 2304]")
    if kv_page_bytes < PAGE_TOKENS * KV_ROW or kv_page_bytes % 16 or kv_base_ptr % 16:
        raise ValueError("KV pages must hold 64 rows of 576 bytes, 16-byte aligned")
    if page_table.dtype != torch.int32 or page_table.stride(1) != 1 or kv_lens.dtype != torch.int32:
        raise ValueError("page_table and kv_lens must be int32 (page_table rows unit-stride)")
    if (
        cos_sin.dtype != torch.float32
        or cos_sin.numel() % (2 * ROPE_DIM)
        or not cos_sin.is_contiguous()
    ):
        raise ValueError(
            "cos_sin must be a contiguous FP32 table of 128 values (64 cos / sin pairs) per position"
        )
    if q_ln_weight.numel() != Q_LORA or kv_ln_weight.numel() != KV_LORA:
        raise ValueError("layernorm weights must have 1536 and 512 elements")
    if any(
        t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16
        for t in (q_ln_weight, kv_ln_weight)
    ):
        raise ValueError("layernorm weights must be contiguous, 16-byte aligned BF16")
    compiled = _compiled(n_pad_for(m), k_splits, q_out is not None, use_pdl)
    compiled(
        make_ptr(cutlass.BFloat16, w_kc.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int64(w_qb.data_ptr()),
        Int64(w_qb_sf.data_ptr()),
        Int64(fused_a.data_ptr()),
        Int32(fused_a.stride(0) * 2),
        Int64(q_ln_weight.data_ptr()),
        Int64(kv_ln_weight.data_ptr()),
        Float32(q_eps),
        Float32(kv_eps),
        Int64(quant_q.data_ptr()),
        Int32(NUM_HEADS * OUT_HEAD),
        Int64(0 if q_out is None else q_out.data_ptr()),
        Int64(kv_base_ptr),
        Int64(kv_page_bytes),
        Int64(page_table.data_ptr()),
        Int32(page_table.stride(0)),
        Int32(layer_in_pool),
        Int64(kv_lens.data_ptr()),
        Int64(cos_sin.data_ptr()),
        Float32(kv_scale),
        Int64(bmm1_scale.data_ptr()),
        Int64(bmm2_scale.data_ptr()),
        Int64(cu_q_seqlens.data_ptr()),
        Int64(cu_kv_seqlens.data_ptr()),
        Int64(scheduler_counter.data_ptr()),
        Float32(host_bmm1),
        Int32(m),
        Int32(q_len),
        Int32(batch),
        cuda.CUstream(torch.cuda.current_stream(fused_a.device).cuda_stream),
    )


__all__ = [
    "K3MlaFusedQKVKernel",
    "MAX_TOKENS",
    "SUPPORTED_K_SPLITS",
    "compile_all",
    "host_bmm1_scale",
    "k_split_ranges",
    "lora_cols",
    "n_pad_for",
    "run_k3_mla_fused_qkv",
]
