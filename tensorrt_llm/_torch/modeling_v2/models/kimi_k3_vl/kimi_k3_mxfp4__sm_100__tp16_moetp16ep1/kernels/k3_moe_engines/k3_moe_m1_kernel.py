# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi K3 routed experts of one or two decode tokens (M = m_max <= 2) as a weight-stream kernel: ``k3_moe_m1``.

At M <= 2 every expert a token routes to is a GEMV, so the kernel spreads the experts' weight rows over all CTAs
instead of k3_moe's 128-row tiles. It reads the outputs of trtllm::k3_moe_front (or trtllm::k3_route_quant): the
tokens' top-16 global ids and bf16 weights and their MXFP8 latents. It computes this rank's routed partial [M, 3584]
bf16, the tensor k3_moe returns. Weights are read in place in the TRTLLM-Gen W4A8_MXFP4_MXFP8 layout (see
k3_moe_kernel.py), including the loader's zero padding of a rank's intermediate to a multiple of 128 (i_pad, e.g.
192 -> 256 at TP16). The kernel streams the i_tp real values only.

- Routing: one warp decodes the 16 M top-k slots into experts (the distinct local experts in lane order, each with
  its tokens). The first FC1 unit's ring fill is issued from the lanes before the CTA barrier.
- FC1 (gate_up, MXFP4 x MXFP8) in 64-row units spread over the CTAs, with k3_moe's per-stage block-scaled
  machinery. Two k-tiles go into each stage: the unit's 64 rows at k-tile 2j fill MMA rows 0-63 and at k-tile
  2j + 1 rows 64-127; B rows t and 8 + t hold token t at those k-tiles (N 16), and the scale atoms are spliced to
  match. The epilogue adds the two partial sums per token, then applies k3_moe's SiTU and MXFP8 requantization, into
  one intermediate row per (expert, token).
- Hand-off: each unit adds one release to the call's count. The count has two slots by epoch parity; CTA 0 re-arms
  the other slot, and every CTA advances its epoch. The FC2 CTAs acquire the count, then load the intermediate in
  one round.
- FC2 (down): each of 112 CTAs owns one 32-row block of the shuffled down projection. Four experts' 32-row slices
  stack in one 128-row MMA (rows 32 j .. 32 j + 31 = expert 4 i + j) against B N 16, whose row 4 t + j is token t's
  intermediate for expert 4 i + j; this is k3_moe's FC2 operand, so each expert's down projection has k3_moe's bits.
  - The A tiles load into the FC1 ring as it drains, evict-first (read once per step, like FC1's weights).
  - The weight scale atoms are spliced while FC1 runs.
- Combine: per token, the routing-weighted experts are summed in k3_moe's order (ascending local id in min(G, 5)
  slices, an expert without the token adding zero, products and sums rounded on their own), so the output matches
  k3_moe's bits except where the two FC1 partial sums round an intermediate value differently.

Push (K3_CONFIG "push" = 1): instead of writing ``out``, the combine stores this
rank's row of token t into slot [half][t][rank x copies + c] of every rank's latent exchange
(``latent_op.K3LatentExchange``, int32 [2][8][push_world][1792], bf16 pairs, -0.0 stored as +0.0) through its multicast
mapping, half = lat_flags[0] & 1 read after the grid wait; trtllm::k3_latent_reduce sums it. ``push_copies`` > 1
only emulates a larger group's receive side.

Configuration is per module instance (the shapes are trace-time constants): the loader injects K3_CONFIG =
{"i_tp": ..., "i_pad": ..., "num_local": ..., "num_ctas": ..., "m_max": ..., ["push", "push_world", "push_copies"]}
before executing the module. Launched with programmatic dependent launch: the barrier setup and the TMEM
allocation run before griddepcontrol.wait, which precedes every read of the producer's outputs and every global
write.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda.tensor_map import TensorMapDataType

_CFG = globals().get("K3_CONFIG") or {}


def _cfg(key: str, default):
    """A kernel option from the op's configuration (K3_CONFIG), else its default."""
    return type(default)(_CFG.get(key, default))


# =============================================================================
# Problem shape (trace-time constants).
# =============================================================================
H = 3584  # latent hidden = FC1 K = FC2 rows
I_TP = _cfg("i_tp", 192)  # a rank's intermediate values (the logical shard)
I_PAD = _cfg(
    "i_pad", (I_TP + 127) // 128 * 128
)  # the loader's padded intermediate (the buffers' layout)
E_LOCAL = _cfg("num_local", 896)  # this rank's experts
M_MAX = _cfg("m_max", 1)  # tokens per call (one routing warp: 16 M lanes)
NCTA = _cfg("num_ctas", 148)  # one CTA per SM
PUSH = bool(_cfg("push", 0))  # the combine pushes into the latent exchange instead of writing out
PUSH_WORLD = _cfg("push_world", 16)  # slots per (half, token) of the exchange
PUSH_COPIES = _cfg("push_copies", 1)  # slots this rank fills (rank x copies + c)
LAT_ROW_WORDS = 3584 // 2  # int32 words of a latent row in the exchange
TWO_I = 2 * I_TP
TOP_K = 16
G_CAP = TOP_K * M_MAX  # distinct local experts per call
THREADS = 256
N = 16  # MMA N: FC1 B rows 0 (k-tile 2j) and 8 (k-tile 2j + 1) are the token; FC2 B rows 0-3 are a group's experts
MMA_M, MMA_TILE_K, MMA_INST_K = 128, 128, 32
ROWS1 = 64  # FC1 rows per unit (one half of a 128-row tile)
U1 = TWO_I // ROWS1  # units per expert
K1_TILES = H // MMA_TILE_K  # 28
K1_PAIRS = K1_TILES // 2  # 14 stages per unit
NUM_KBLOCKS = MMA_TILE_K // MMA_INST_K  # 4
a_dtype = cutlass.Float4E2M1FN
b_dtype = cutlass.Float8E4M3FN
sf_dtype = cutlass.Float8E8M0FNU
a_smem_width = 8  # FP4 unpacked to 8-bit containers in shared memory
sf_vec_size = 32
num_m0_per_sf_atom = 32
num_m1_per_sf_atom = 4
num_k_per_sf_atom = 4
num_elts_atom_sf_fp16 = num_m0_per_sf_atom * num_m1_per_sf_atom * num_k_per_sf_atom // 2
num_tmem_cols_per_sf_atom = 4
NUM_BYTES_A = MMA_M * MMA_TILE_K * a_smem_width // 8  # 16384
NUM_BYTES_A_HALF = ROWS1 * MMA_TILE_K * a_smem_width // 8  # 8192
NUM_TX_A_HALF = ROWS1 * MMA_TILE_K * 4 // 8  # 4096: FP4 bytes in global memory
NUM_BYTES_B = N * MMA_TILE_K  # 2048
NUM_BYTES_SFA = 512
NUM_BYTES_SFA_RAW = 2 * NUM_BYTES_SFA  # the two k-tiles' scale atoms as loaded
NUM_BYTES_SFB = 512
SFB_GROUP_BYTES = 16
SFA_COLS = num_tmem_cols_per_sf_atom
SFB_COLS = num_tmem_cols_per_sf_atom
NUM_SF_IDS = num_k_per_sf_atom * sf_vec_size // MMA_INST_K  # 4
SITU_GATE_CAP = 4.0
SITU_LINEAR_CAP = 25.0
E4M3_MAX = 448.0
FP8_SENTINEL_I8 = -128
_LOG2E = 1.4426950408889634
# FC2
R2 = 32  # output rows per CTA: one 32-row block of the shuffled layout
FC2_CTAS = H // R2  # 112
ROW_PAD2 = I_PAD // 2  # FP4 bytes of a down row in the buffer
KB2 = I_TP // 32  # MX blocks (K 32 MMAs) of a down row's real values
KT2 = (I_TP + MMA_TILE_K - 1) // MMA_TILE_K  # 128-wide k-tiles of the real values
KA2 = (
    I_PAD // 128
)  # w2 scale atoms per 128-row block (block_scale_interleave of the padded I / 32 columns)
GROUPS2 = G_CAP // 4  # 4 experts x 32 rows per 128-row MMA
NT2 = GROUPS2 * KT2  # A tiles, B tiles and scale atoms per CTA
NUM_TX_A2 = R2 * MMA_TILE_K * 4 // 8  # 2048: FP4 bytes of one expert's 32 rows of a k-tile
NUM_BYTES_B2 = N * MMA_TILE_K  # 2048
H_ROW = (
    (I_TP + I_TP // 32 + 15) // 16 * 16
)  # an intermediate row: fp8 values + E8M0 scales, 16-byte multiple
VCH = I_TP // 16  # 16-byte value chunks of an intermediate row
SCH = (KB2 + 15) // 16  # 16-byte scale chunks
STAGE_ROUNDS = (G_CAP * M_MAX * (VCH + SCH) + 127) // 128  # load rounds of the 128 epilogue threads
# The ring: as many stages as fit next to FC2's B tiles and scale atoms (8 at TP16, 7 at TP4 x EP4).
_STAGE_BYTES = NUM_BYTES_A + NUM_BYTES_B + NUM_BYTES_SFA + NUM_BYTES_SFA_RAW + NUM_BYTES_SFB
_FIXED_BYTES = NT2 * (NUM_BYTES_B2 + NUM_BYTES_SFA + NUM_BYTES_SFB) + G_CAP * M_MAX * R2 * 4 + 4096
STAGES = min(8, (227 * 1024 - _FIXED_BYTES) // _STAGE_BYTES)
PRE_FILL = min(STAGES, K1_PAIRS)  # stages of the first unit issued before the routing barrier
# TMEM columns: FC1 accumulator, FC1 SFA / SFB per stage, FC2 accumulators (16 per group), FC2 SFA / SFB per tile.
ACC2_COL = N + 2 * STAGES * SFA_COLS
SFA2_COL = ACC2_COL + GROUPS2 * N
SFB2_COL = SFA2_COL + NT2 * SFA_COLS
TMEM_COLS = 32
while TMEM_COLS < SFB2_COL + NT2 * SFB_COLS:
    TMEM_COLS *= 2
REF_SLICES = 5  # k3_moe's FC2 slices (fc2_slices): its combine's sum tree, kept so the bits match
_REF_SLICES_FULL = [
    (s * G_CAP // REF_SLICES, (s + 1) * G_CAP // REF_SLICES) for s in range(REF_SLICES)
]
EPI_BAR_ID = 1
EPI_THREADS = 128
W_WARP = 4
X_WARP = 5
S_WARP = 6
MMA_WARP = 7
EVICT_FIRST = 0x12F0000000000000
assert TWO_I % ROWS1 == 0 and I_TP % 32 == 0 and I_PAD % 128 == 0 and I_PAD >= I_TP, (I_TP, I_PAD)
assert STAGES >= 4 and TMEM_COLS <= 512 and NCTA >= FC2_CTAS, (
    f"k3_moe_m1 does not fit i_tp {I_TP} at m_max {M_MAX}: {STAGES} ring stages, {TMEM_COLS} TMEM columns, {NCTA} CTAs"
)
assert M_MAX in (1, 2) and 4 * M_MAX <= 8, (
    M_MAX
)  # FC2 B rows 4 t + j stay in one 8-row swizzle atom


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    """a * b rounded on its own (mul.rn is never fused into an FMA)."""
    return cutlass.Float32(_llvm.inline_asm(
        _T.f32(), [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))  # fmt: skip


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    """a + b rounded on its own (add.rn is never fused into an FMA)."""
    return cutlass.Float32(_llvm.inline_asm(
        _T.f32(), [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "add.rn.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))  # fmt: skip


@dsl_user_op
def _red_release_add(addr, val, *, loc=None, ip=None):
    _llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _load_acquire(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _st_u32(addr, val, *, loc=None, ip=None):
    _llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(bf16(hi) << 16) | bf16(lo), round to nearest even."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@cute.jit
def _emit(acc, lane, row2, tq, ch, out, lat_mc, lat_flags, lat_rank):
    """Warp tq of an FC2 CTA (token tq's combine): lane l holds the value of output column row2 + 4 (l % 8) + l // 8,
    stored to ``out`` at ``ch``; with PUSH, lanes 0, 2, 4, 6 gather columns row2 + 4 l .. row2 + 4 l + 7 and store
    them as 16 bytes (bf16 pairs, -0.0 as +0.0) into this rank's slots of token tq in every rank's latent exchange
    through its multicast mapping."""
    if cutlass.const_expr(PUSH):
        bits = _pack_bf16x2(cutlass.Float32(0.0), acc) & cutlass.Int32(0xFFFF)
        bits = cutlass.select_(bits == cutlass.Int32(0x8000), cutlass.Int32(0), bits)
        b1 = cute.arch.shuffle_sync_down(bits, 8)
        b2 = cute.arch.shuffle_sync_down(bits, 16)
        b3 = cute.arch.shuffle_sync_down(bits, 24)
        b4 = cute.arch.shuffle_sync_down(bits, 1)
        b5 = cute.arch.shuffle_sync_down(bits, 9)
        b6 = cute.arch.shuffle_sync_down(bits, 17)
        b7 = cute.arch.shuffle_sync_down(bits, 25)
        half = cutlass.Int32(lat_flags.load(idx=0, is_volatile=True)) & cutlass.Int32(1)
        if (lane < 8) & (lane % 2 == 0):
            vec = (bits | (b1 << cutlass.Int32(16)), b2 | (b3 << cutlass.Int32(16)), b4 | (b5 << cutlass.Int32(16)),
                   b6 | (b7 << cutlass.Int32(16)))  # fmt: skip
            for c in cutlass.range_constexpr(PUSH_COPIES):
                slot = lat_rank * cutlass.Int32(PUSH_COPIES) + cutlass.Int32(c)
                lat_mc.store(vec, idx=((half * cutlass.Int32(8) + tq) * cutlass.Int32(PUSH_WORLD) + slot)
                             * cutlass.Int32(LAT_ROW_WORDS) + (row2 + cutlass.Int32(4) * lane) // cutlass.Int32(2),
                             alignment=16)  # fmt: skip
    else:
        out.store(cutlass.BFloat16(acc), idx=ch)


def _tanh_f32(x):
    e = cute.math.exp2(cute.math.abs(x) * cutlass.Float32(-2.0 * _LOG2E), fastmath=True)
    t = (cutlass.Float32(1.0) - e) * cute.arch.rcp_approx(cutlass.Float32(1.0) + e)
    return cutlass.select_(x < cutlass.Float32(0.0), -t, t)


def _sigmoid_f32(x):
    return cute.arch.rcp_approx(
        cutlass.Float32(1.0) + cute.math.exp2(x * cutlass.Float32(-_LOG2E), fastmath=True)
    )


def _situ(gate, up):
    g = (
        cutlass.Float32(SITU_GATE_CAP)
        * _tanh_f32(gate * cutlass.Float32(1.0 / SITU_GATE_CAP))
        * _sigmoid_f32(gate)
    )
    u = cutlass.Float32(SITU_LINEAR_CAP) * _tanh_f32(up * cutlass.Float32(1.0 / SITU_LINEAR_CAP))
    return g * u


def _block_e8m0(amax):
    """E8M0 byte of an MX block and 2^(127 - byte) as f32 (k3_moe's ceil recipe)."""
    sf = amax * cutlass.Float32(1.0 / 448.0)
    sbits = cutlass.Int32(sf.bitcast(cutlass.Int32))
    sexp = (sbits >> cutlass.Int32(23)) & cutlass.Int32(0xFF)
    mant = sbits & cutlass.Int32(0x7FFFFF)
    byte = sexp + cutlass.select_(mant != cutlass.Int32(0), cutlass.Int32(1), cutlass.Int32(0))
    byte = cutlass.select_(byte > cutlass.Int32(0xFE), cutlass.Int32(0xFE), byte)
    byte = cutlass.select_(amax > cutlass.Float32(0.0), byte, cutlass.Int32(0))
    inv = cutlass.Int32((cutlass.Int32(254) - byte) << cutlass.Int32(23)).bitcast(cutlass.Float32)
    return byte, inv


@dsl_user_op
def _match_any(value, *, loc=None, ip=None):
    """The mask of the warp's lanes holding the same 32-bit value."""
    return cutlass.Int32(_llvm.inline_asm(
        _T.i32(), [cutlass.Int32(value).ir_value(loc=loc, ip=ip)], "match.any.sync.b32 $0, $1, 0xffffffff;", "=r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))  # fmt: skip


@cute.jit
def _wait(bar: cutlass.Array, parity):
    while not cute.arch.mbarrier_try_wait(bar.data_ptr(), parity):
        pass


@cute.kernel
def k3_moe_m1_kernel(
    tma_a1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_b1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_sfa1_desc: cutlass.GridConstant[cuda.TensorMap],
    tma_a2_desc: cutlass.GridConstant[cuda.TensorMap],
    sfb1_ptr: cutlass.Int64,
    ids: cutlass.Array,  # int32 [M, 16]
    wts: cutlass.Array,  # bf16 [M, 16] routing weights (as int16 bits)
    w2s32: cutlass.Array,  # int32 words of w2_weight_scale [E, H / 128, I_PAD / 128, 512 B]
    hbuf: cutlass.Array,  # int8 [G_CAP, M, H_ROW]
    hbuf32: cutlass.Array,  # the same memory as int32 words
    counts: cutlass.Array,  # int32 [2]
    epochs: cutlass.Array,  # int32 [NCTA]
    out: cutlass.Array,  # bf16 [M, 3584] (not written with PUSH)
    offset: cutlass.Int32,
    lat_mc: cutlass.Array,  # PUSH: int32 words of the latent exchange's multicast mapping
    lat_flags: cutlass.Array,  # PUSH: int32 [4], [0] the reduce's call count
    lat_rank: cutlass.Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32

    sA = cutlass.Array(
        cutlass.Int8, NUM_BYTES_A * STAGES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sB = cutlass.Array(
        cutlass.Int8, NUM_BYTES_B * STAGES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sSFA = cutlass.Array(
        cutlass.Int8, NUM_BYTES_SFA * STAGES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sSFAraw = cutlass.Array(cutlass.Int32, NUM_BYTES_SFA_RAW * STAGES // 4, space=cutlass.AddressSpace.smem,
                            alignment=1024)  # fmt: skip
    sSFB = cutlass.Array(
        cutlass.Int8, NUM_BYTES_SFB * STAGES, space=cutlass.AddressSpace.smem, alignment=1024
    )
    b2 = cutlass.Array(
        cutlass.Int8, NT2 * NUM_BYTES_B2, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sfa2 = cutlass.Array(
        cutlass.Int8, NT2 * NUM_BYTES_SFA, space=cutlass.AddressSpace.smem, alignment=1024
    )
    sfb2 = cutlass.Array(
        cutlass.Int8, NT2 * NUM_BYTES_SFB, space=cutlass.AddressSpace.smem, alignment=1024
    )
    s_part = cutlass.Array(
        cutlass.Float32, M_MAX * ROWS1, space=cutlass.AddressSpace.smem, alignment=16
    )
    ys = cutlass.Array(
        cutlass.Float32, G_CAP * M_MAX * R2, space=cutlass.AddressSpace.smem, alignment=16
    )
    ab_full = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    ab_empty = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    fc2_full = cutlass.Array(cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8)
    scales_in_tmem = cutlass.Array(
        cutlass.Int64, STAGES, space=cutlass.AddressSpace.smem, alignment=8
    )
    acc_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc_empty = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    sfa2_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    b2_ready = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    acc2_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    localmax_smem = cutlass.Array(cutlass.Float32, 8 * M_MAX, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    s_el = cutlass.Array(cutlass.Int32, G_CAP + 1, space=cutlass.AddressSpace.smem, alignment=16)
    s_w = cutlass.Array(
        cutlass.Float32, G_CAP * M_MAX, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_tmask = cutlass.Array(cutlass.Int32, G_CAP, space=cutlass.AddressSpace.smem, alignment=16)
    s_perm = cutlass.Array(cutlass.Int32, G_CAP, space=cutlass.AddressSpace.smem, alignment=16)

    # ---- before the grid dependency: barriers, the TMEM allocation
    if warp == 0:
        if tidx < STAGES:
            prims.mbarrier_init(ab_full.subview(tidx), 3)  # A + SFA, B, SFB
            prims.mbarrier_init(ab_empty.subview(tidx), 1)
            prims.mbarrier_init(fc2_full.subview(tidx), 1)
            prims.mbarrier_init(scales_in_tmem.subview(tidx), 1)
        if tidx == 0:
            prims.mbarrier_init(acc_full.subview(0), 1)
            prims.mbarrier_init(acc_empty.subview(0), EPI_THREADS)
            prims.mbarrier_init(tmem_ready.subview(0), 32)
            prims.mbarrier_init(sfa2_ready.subview(0), EPI_THREADS)
            prims.mbarrier_init(b2_ready.subview(0), EPI_THREADS)
            prims.mbarrier_init(acc2_full.subview(0), 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp == MMA_WARP:
        prims.tcgen05_alloc(tmem_ptr_i32, TMEM_COLS)
        prims.mbarrier_arrive(tmem_ready)
        prims.tcgen05_relinquish_alloc_permit()
    if warp == W_WARP:
        prims.prefetch_tensormap(tma_a1_desc.get_ptr())
        prims.prefetch_tensormap(tma_sfa1_desc.get_ptr())
        prims.prefetch_tensormap(tma_a2_desc.get_ptr())
    if warp == X_WARP:
        prims.prefetch_tensormap(tma_b1_desc.get_ptr())

    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()

    # ---- routing: lane l = top-k slot l % 16 of token l / 16. Slots: the distinct local experts in lane order (FC1
    # units, intermediate rows, FC2 groups); the combine walks them in ascending id through s_perm (below).
    if warp == W_WARP:
        loc_e = cutlass.Int32(-1)
        wv = cutlass.Float32(0.0)
        if lane < TOP_K * M_MAX:
            loc_e = ids.load(idx=lane) - offset
            wv = cutlass.Float32(
                cutlass.Int32(cutlass.Int32(wts.load(idx=lane)) << cutlass.Int32(16)).bitcast(
                    cutlass.Float32
                )
            )
        is_local = (
            (lane < TOP_K * M_MAX) & (loc_e >= cutlass.Int32(0)) & (loc_e < cutlass.Int32(E_LOCAL))
        )
        first = is_local
        leader = lane
        if cutlass.const_expr(M_MAX > 1):
            # An expert two tokens share: the lowest of its lanes stands for it; the others add their token.
            same = _match_any(cutlass.select_(is_local, loc_e, cutlass.Int32(-1) - lane))
            leader = cute.arch.popc((same & (cutlass.Int32(0) - same)) - cutlass.Int32(1))
            first = is_local & (lane == leader)
        bal = prims.vote_sync(0xFFFFFFFF, first, prims.VoteSync.BALLOT)
        rank = cute.arch.popc(bal & cutlass.Int32(cute.arch.lanemask_lt()))
        if cutlass.const_expr(M_MAX > 1):
            for wz in cutlass.range_constexpr(G_CAP * M_MAX // 32):
                s_w.store(cutlass.Float32(0.0), idx=lane + cutlass.Int32(32 * wz))
            cute.arch.sync_warp()
            rank_l = cute.arch.shuffle_sync(rank, leader)
            if is_local:
                s_w.store(wv, idx=rank_l * cutlass.Int32(M_MAX) + lane // cutlass.Int32(TOP_K))
            if first:
                s_el.store(loc_e, idx=rank)
                s_tmask.store(cutlass.select_((same & cutlass.Int32(0xFFFF)) != cutlass.Int32(0), cutlass.Int32(1),
                                              cutlass.Int32(0))
                              | cutlass.select_((same & cutlass.Int32(-65536)) != cutlass.Int32(0), cutlass.Int32(2),
                                                cutlass.Int32(0)), idx=rank)  # fmt: skip
        else:
            if is_local:
                s_el.store(loc_e, idx=rank)
                s_w.store(wv, idx=rank)
        if lane == 0:
            s_el.store(cute.arch.popc(bal), idx=G_CAP)
        # The first unit's ring fill before the CTA barrier: its expert (rank u / U1) comes from the lanes, not
        # shared memory.
        if bx < cute.arch.popc(bal) * cutlass.Int32(U1):
            grp0 = bx // cutlass.Int32(U1)
            m_src = prims.vote_sync(0xFFFFFFFF, first & (rank == grp0), prims.VoteSync.BALLOT)
            e0 = cute.arch.shuffle_sync(
                loc_e, cute.arch.popc((m_src & (cutlass.Int32(0) - m_src)) - cutlass.Int32(1))
            )
            t128_0 = (bx % cutlass.Int32(U1)) // cutlass.Int32(2)
            coord_m0 = t128_0 * cutlass.Int32(MMA_M) + (bx % cutlass.Int32(2)) * cutlass.Int32(
                ROWS1
            )
            if prims.elect_sync():
                for kp in cutlass.range_constexpr(PRE_FILL):
                    prims.mbarrier_arrive_expect_tx(
                        ab_full.subview(kp), 2 * (NUM_TX_A_HALF + NUM_BYTES_SFA)
                    )
                    for q in cutlass.range_constexpr(2):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sA.subview(cutlass.Int32(kp * NUM_BYTES_A + q * NUM_BYTES_A_HALF)), tma_a1_desc.get_ptr(),
                            (cutlass.Int32((2 * kp + q) * MMA_TILE_K), coord_m0, e0), ab_full.subview(kp),
                            l2_cache_hint=EVICT_FIRST,
                        )  # fmt: skip
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sSFAraw.subview(cutlass.Int32((kp * NUM_BYTES_SFA_RAW + q * NUM_BYTES_SFA) // 4)),
                            tma_sfa1_desc.get_ptr(), (cutlass.Int32(0), cutlass.Int32(2 * kp + q), t128_0, e0),
                            ab_full.subview(kp), l2_cache_hint=EVICT_FIRST,
                        )  # fmt: skip
        # Ascending local id -> slot: s_perm[the distinct local experts below this one] = its slot.
        key = cutlass.select_(first, loc_e, cutlass.Int32(1 << 30))
        rank_id = cutlass.Int32(0)
        for k in range(TOP_K * M_MAX):
            other = cute.arch.shuffle_sync(key, k)
            if other < key:
                rank_id = rank_id + cutlass.Int32(1)
        if first:
            s_perm.store(rank, idx=rank_id)
    prims.barrier_cta_sync(0)
    g_n = s_el.load(idx=G_CAP)
    units = g_n * cutlass.Int32(U1)
    mine = (units - bx + cutlass.Int32(NCTA - 1)) // cutlass.Int32(NCTA)
    if bx >= units:
        mine = cutlass.Int32(0)
    ep = epochs.load(idx=bx)
    slot_addr = counts.subview(0).data_ptr().toint() + cutlass.Int64((ep & 1) * 4)
    row2 = bx * cutlass.Int32(R2)
    is_fc2 = bx < cutlass.Int32(FC2_CTAS)
    groups = (g_n + cutlass.Int32(3)) // cutlass.Int32(4)
    nt2 = cutlass.select_(is_fc2, groups * cutlass.Int32(KT2), cutlass.Int32(0))
    blk2 = row2 // cutlass.Int32(128)
    m1 = (row2 % cutlass.Int32(128)) // cutlass.Int32(32)

    # ---- weights producer (4): FC1 stages (the unit's 64 rows at k-tiles 2j, 2j + 1 and both scale atoms, raw), then
    # this CTA's FC2 A tiles into the ring as it drains (group i, k-tile t: four experts' 32 rows at 4 KB offsets).
    if warp == W_WARP:
        if bx == 0:
            if lane == 0:
                _st_u32(
                    counts.subview(0).data_ptr().toint() + cutlass.Int64(((ep + 1) & 1) * 4),
                    cutlass.Int32(0),
                )
        g = cutlass.select_(
            mine > cutlass.Int32(0), cutlass.Int32(PRE_FILL), cutlass.Int32(0)
        )  # issued pre-barrier
        ab_empty_phase = 1
        for ui in range(mine):
            u = bx + ui * cutlass.Int32(NCTA)
            grp = u // cutlass.Int32(U1)
            t128 = (u % cutlass.Int32(U1)) // cutlass.Int32(2)
            half = u % cutlass.Int32(2)
            coord_expert = s_el.load(idx=grp)
            coord_m = t128 * cutlass.Int32(MMA_M) + half * cutlass.Int32(ROWS1)
            for kp in cutlass.range(cutlass.select_(ui == cutlass.Int32(0), cutlass.Int32(PRE_FILL), cutlass.Int32(0)),
                                    K1_PAIRS, unroll=1):  # fmt: skip
                stage = g % cutlass.Int32(STAGES)
                if stage == cutlass.Int32(0) and g != cutlass.Int32(0):
                    ab_empty_phase = ab_empty_phase ^ 1
                while not cute.arch.mbarrier_try_wait(
                    ab_empty.subview(stage).data_ptr(), ab_empty_phase
                ):
                    pass
                coord_k = kp * cutlass.Int32(2 * MMA_TILE_K)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(
                        ab_full.subview(stage), 2 * (NUM_TX_A_HALF + NUM_BYTES_SFA)
                    )
                    for q in cutlass.range_constexpr(2):
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sA.subview(stage * cutlass.Int32(NUM_BYTES_A) + cutlass.Int32(q * NUM_BYTES_A_HALF)),
                            tma_a1_desc.get_ptr(), (coord_k + cutlass.Int32(q * MMA_TILE_K), coord_m, coord_expert),
                            ab_full.subview(stage), l2_cache_hint=EVICT_FIRST,
                        )  # fmt: skip
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sSFAraw.subview(
                                (stage * cutlass.Int32(NUM_BYTES_SFA_RAW) + cutlass.Int32(q * NUM_BYTES_SFA)) // 4
                            ),
                            tma_sfa1_desc.get_ptr(), (cutlass.Int32(0), kp * cutlass.Int32(2) + cutlass.Int32(q), t128,
                                                      coord_expert),
                            ab_full.subview(stage), l2_cache_hint=EVICT_FIRST,
                        )  # fmt: skip
                g = g + cutlass.Int32(1)
        for j in range(nt2):
            stage = g % cutlass.Int32(STAGES)
            if stage == cutlass.Int32(0) and g != cutlass.Int32(0):
                ab_empty_phase = ab_empty_phase ^ 1
            while not cute.arch.mbarrier_try_wait(
                ab_empty.subview(stage).data_ptr(), ab_empty_phase
            ):
                pass
            i2 = j // cutlass.Int32(KT2)
            t2 = j % cutlass.Int32(KT2)
            n_here = g_n - i2 * cutlass.Int32(4)
            if n_here > cutlass.Int32(4):
                n_here = cutlass.Int32(4)
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(
                    fc2_full.subview(stage), n_here * cutlass.Int32(NUM_TX_A2)
                )
                for jj in cutlass.range_constexpr(4):
                    if cutlass.Int32(jj) < n_here:
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sA.subview(stage * cutlass.Int32(NUM_BYTES_A) + cutlass.Int32(jj * R2 * MMA_TILE_K)),
                            tma_a2_desc.get_ptr(), (t2 * cutlass.Int32(MMA_TILE_K), row2,
                                                    s_el.load(idx=i2 * cutlass.Int32(4) + cutlass.Int32(jj))),
                            fc2_full.subview(stage), l2_cache_hint=EVICT_FIRST,
                        )  # fmt: skip
            g = g + cutlass.Int32(1)

    # ---- activations producer (5): the token's 128 K of k-tiles 2j, 2j + 1 into B rows 0 and 8, their scales.
    if warp == X_WARP:
        g = cutlass.Int32(0)
        ab_empty_phase = 1
        for ui in range(mine):
            for kp in cutlass.range(K1_PAIRS, unroll=1):
                stage = g % cutlass.Int32(STAGES)
                if stage == cutlass.Int32(0) and g != cutlass.Int32(0):
                    ab_empty_phase = ab_empty_phase ^ 1
                while not cute.arch.mbarrier_try_wait(
                    ab_empty.subview(stage).data_ptr(), ab_empty_phase
                ):
                    pass
                coord_k = kp * cutlass.Int32(2 * MMA_TILE_K)
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(ab_full.subview(stage), 2 * M_MAX * MMA_TILE_K)
                    for tq in cutlass.range_constexpr(M_MAX):
                        for q in cutlass.range_constexpr(2):
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                sB.subview(
                                    stage * cutlass.Int32(NUM_BYTES_B) + cutlass.Int32((q * 8 + tq) * MMA_TILE_K)
                                ),
                                tma_b1_desc.get_ptr(), (coord_k + cutlass.Int32(q * MMA_TILE_K), cutlass.Int32(tq)),
                                ab_full.subview(stage),
                            )  # fmt: skip
                if lane == 0:
                    for tq in cutlass.range_constexpr(M_MAX):
                        for q in cutlass.range_constexpr(2):
                            sfb_gmem = (
                                sfb1_ptr
                                + cutlass.Int64(tq * (H // sf_vec_size))
                                + cutlass.Int64(
                                    (kp * cutlass.Int32(2) + cutlass.Int32(q)) * NUM_KBLOCKS
                                )
                            )
                            prims.cp_async_shared_global(
                                sSFB.subview(stage * cutlass.Int32(NUM_BYTES_SFB)
                                             + cutlass.Int32((q * 8 + tq) * SFB_GROUP_BYTES)).data_ptr(),
                                cutlass.inttoptr(sfb_gmem, mem_space=1, dtype=sf_dtype),
                                size=4, modifier="ca", cp_size=4,
                            )  # fmt: skip
                    prims.cp_async_mbarrier_arrive(ab_full.subview(stage), noinc=True)
                g = g + cutlass.Int32(1)

    # ---- scales to TMEM (6): the unit's half of both k-tiles' atoms spliced into one MMA atom (bytes 8 h .. 8 h + 7
    # of each 16-byte row group: k-tile 2j to MMA rows 0-63, k-tile 2j + 1 to rows 64-127), then SFA and SFB to TMEM.
    if warp == S_WARP:
        _wait(tmem_ready, 0)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        sfa_col_id0 = base_col_id + N
        sfb_col_id0 = sfa_col_id0 + STAGES * SFA_COLS
        s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        sSFA32 = cutlass.Array(sSFA.data_ptr(0), shape=(NUM_BYTES_SFA * STAGES // 4,), dtype=cutlass.Int32,
                               alignment=16)  # fmt: skip
        g = cutlass.Int32(0)
        full_phase = 0
        for ui in range(mine):
            u = bx + ui * cutlass.Int32(NCTA)
            half = u % cutlass.Int32(2)
            for kp in cutlass.range(K1_PAIRS, unroll=1):
                stage = g % cutlass.Int32(STAGES)
                if stage == cutlass.Int32(0) and g != cutlass.Int32(0):
                    full_phase = full_phase ^ 1
                while not cute.arch.mbarrier_try_wait(
                    ab_full.subview(stage).data_ptr(), full_phase
                ):
                    pass
                raw = (
                    stage * cutlass.Int32(NUM_BYTES_SFA_RAW // 4)
                    + lane * cutlass.Int32(4)
                    + half * cutlass.Int32(2)
                )
                lo = sSFAraw.load(idx=raw, vector_size=2, alignment=8)
                hi = sSFAraw.load(
                    idx=raw + cutlass.Int32(NUM_BYTES_SFA // 4), vector_size=2, alignment=8
                )
                sSFA32.store((cutlass.Int32(lo[0]), cutlass.Int32(lo[1]), cutlass.Int32(hi[0]), cutlass.Int32(hi[1])),
                             idx=stage * cutlass.Int32(NUM_BYTES_SFA // 4) + lane * cutlass.Int32(4),
                             alignment=16)  # fmt: skip
                prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
                cute.arch.sync_warp()
                prims.tcgen05_fence(
                    prims.Tcgen05Fence.AFTER_THREAD_SYNC
                )  # the lanes' spliced scale atoms
                sfa_tmem_ptr = cutlass.inttoptr(
                    (base_row_id << 16) | (sfa_col_id0 + stage * SFA_COLS), 6, cutlass.Int32
                )
                sfb_tmem_ptr = cutlass.inttoptr(
                    (base_row_id << 16) | (sfb_col_id0 + stage * SFB_COLS), 6, cutlass.Int32
                )
                desc_a = prims.Tcgen05SmemDesc.build(
                    sSFA.subview(stage * NUM_BYTES_SFA), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                desc_b = prims.Tcgen05SmemDesc.build(
                    sSFB.subview(stage * NUM_BYTES_SFB), leading_byte_offset=16, stride_byte_offset=128,
                    base_offset=0, layout=0,
                )  # fmt: skip
                if prims.elect_sync():
                    prims.tcgen05_cp(s2t_shape, sfa_tmem_ptr, desc_a, multicast=s2t_multicast)
                    prims.tcgen05_cp(s2t_shape, sfb_tmem_ptr, desc_b, multicast=s2t_multicast)
                    prims.tcgen05_commit(scales_in_tmem.subview(stage))
                g = g + cutlass.Int32(1)

    # ---- MMA (7): FC1 per unit; then FC2: the spliced weight scales and (after the hand-off) the intermediate's
    # scales to TMEM, then per A tile the group's K 32 MMAs into its 16 accumulator columns.
    if warp == MMA_WARP:
        tmem_raw_addr = tmem_ptr_i32.load()
        acc_tmem_ptr = cutlass.inttoptr(tmem_raw_addr, 6, cutlass.Float32)
        idesc = prims.Tcgen05MxInstrDesc.build(
            a_dtype=a_dtype, b_dtype=b_dtype, scale_format=1, n_dim=N, m_dim=MMA_M
        )
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        sfa_col_id0 = base_col_id + N
        sfb_col_id0 = sfa_col_id0 + STAGES * SFA_COLS
        g = cutlass.Int32(0)
        st_phase = 0
        acc_empty_phase = 1
        for ui in range(mine):
            while not cute.arch.mbarrier_try_wait(acc_empty.data_ptr(), acc_empty_phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc_empty_phase = acc_empty_phase ^ 1
            scale_d = False
            for kp in cutlass.range(K1_PAIRS, unroll=1):
                stage = g % cutlass.Int32(STAGES)
                if stage == cutlass.Int32(0) and g != cutlass.Int32(0):
                    st_phase = st_phase ^ 1
                while not cute.arch.mbarrier_try_wait(
                    scales_in_tmem.subview(stage).data_ptr(), st_phase
                ):
                    pass
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                sfa_base = (base_row_id << 16) | (sfa_col_id0 + stage * SFA_COLS)
                sfb_base = (base_row_id << 16) | (sfb_col_id0 + stage * SFB_COLS)
                desc_a_base = prims.Tcgen05SmemDesc.build(
                    sA.subview(stage * NUM_BYTES_A), leading_byte_offset=16, stride_byte_offset=1024, base_offset=0,
                    layout=2,
                )  # fmt: skip
                desc_b_base = prims.Tcgen05SmemDesc.build(
                    sB.subview(stage * NUM_BYTES_B), leading_byte_offset=16, stride_byte_offset=1024, base_offset=0,
                    layout=2,
                )  # fmt: skip
                for kb in cutlass.range(NUM_KBLOCKS, unroll_full=True):
                    sf_inside = kb % NUM_SF_IDS
                    sf_col = kb // NUM_SF_IDS
                    sfa_tmem_ptr = cutlass.inttoptr(sfa_base + sf_col * SFA_COLS, 6, cutlass.Int32)
                    sfb_tmem_ptr = cutlass.inttoptr(sfb_base + sf_col * SFB_COLS, 6, cutlass.Int32)
                    idesc_u = idesc.set_sf_ids(a_sf_id=sf_inside, b_sf_id=sf_inside)
                    inc = ((MMA_INST_K * a_smem_width // 8) >> 4) * kb
                    if prims.elect_sync():
                        prims.tcgen05_mma_block_scale(
                            prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1, acc_tmem_ptr,
                            desc_a_base + inc, desc_b_base + inc, idesc_u, scale_d, sfa_tmem_ptr, sfb_tmem_ptr,
                        )  # fmt: skip
                    scale_d = True
                if prims.elect_sync():
                    prims.tcgen05_commit(ab_empty.subview(stage))
                g = g + cutlass.Int32(1)
            if prims.elect_sync():
                prims.tcgen05_commit(acc_full)
        if nt2 > cutlass.Int32(0):
            s2t_shape, s2t_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
            g0 = g
            if cutlass.const_expr(NT2 <= STAGES):
                # Every A tile has its own ring stage: wait for all of them and copy the weight scales before B is
                # ready, then one thread issues the B scale copies and every MMA back to back.
                for j in cutlass.range_constexpr(NT2):
                    if cutlass.Int32(j) < nt2:
                        _wait(fc2_full.subview((g0 + cutlass.Int32(j)) % cutlass.Int32(STAGES)), 0)
                _wait(sfa2_ready, 0)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    for j in cutlass.range_constexpr(NT2):
                        if cutlass.Int32(j) < nt2:
                            prims.tcgen05_cp(
                                s2t_shape,
                                cutlass.inttoptr((base_row_id << 16) | (base_col_id + SFA2_COL + j * SFA_COLS), 6,
                                                 cutlass.Int32),
                                prims.Tcgen05SmemDesc.build(sfa2.subview(j * NUM_BYTES_SFA), leading_byte_offset=16,
                                                            stride_byte_offset=128, base_offset=0, layout=0),
                                multicast=s2t_multicast,
                            )  # fmt: skip
                _wait(b2_ready, 0)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    for j in cutlass.range_constexpr(NT2):
                        if cutlass.Int32(j) < nt2:
                            prims.tcgen05_cp(
                                s2t_shape,
                                cutlass.inttoptr((base_row_id << 16) | (base_col_id + SFB2_COL + j * SFB_COLS), 6,
                                                 cutlass.Int32),
                                prims.Tcgen05SmemDesc.build(sfb2.subview(j * NUM_BYTES_SFB), leading_byte_offset=16,
                                                            stride_byte_offset=128, base_offset=0, layout=0),
                                multicast=s2t_multicast,
                            )  # fmt: skip
                    for j in cutlass.range_constexpr(NT2):
                        if cutlass.Int32(j) < nt2:
                            stage = (g0 + cutlass.Int32(j)) % cutlass.Int32(STAGES)
                            desc_a_base = prims.Tcgen05SmemDesc.build(
                                sA.subview(stage * NUM_BYTES_A), leading_byte_offset=16, stride_byte_offset=1024,
                                base_offset=0, layout=2,
                            )  # fmt: skip
                            desc_b_base = prims.Tcgen05SmemDesc.build(
                                b2.subview(j * NUM_BYTES_B2), leading_byte_offset=16, stride_byte_offset=1024,
                                base_offset=0, layout=2,
                            )  # fmt: skip
                            acc2_ptr = cutlass.inttoptr(
                                (base_row_id << 16) | (base_col_id + ACC2_COL + (j // KT2) * N), 6, cutlass.Float32
                            )  # fmt: skip
                            sfa_tmem_ptr = cutlass.inttoptr(
                                (base_row_id << 16) | (base_col_id + SFA2_COL + j * SFA_COLS), 6, cutlass.Int32
                            )  # fmt: skip
                            sfb_tmem_ptr = cutlass.inttoptr(
                                (base_row_id << 16) | (base_col_id + SFB2_COL + j * SFB_COLS), 6, cutlass.Int32
                            )  # fmt: skip
                            for kb2 in cutlass.range_constexpr(
                                min(NUM_KBLOCKS, KB2 - NUM_KBLOCKS * (j % KT2))
                            ):
                                prims.tcgen05_mma_block_scale(
                                    prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1, acc2_ptr,
                                    desc_a_base + ((MMA_INST_K * a_smem_width // 8) >> 4) * kb2,
                                    desc_b_base + ((MMA_INST_K * a_smem_width // 8) >> 4) * kb2,
                                    idesc.set_sf_ids(a_sf_id=kb2, b_sf_id=kb2), (j % KT2) != 0 or kb2 != 0,
                                    sfa_tmem_ptr, sfb_tmem_ptr,
                                )  # fmt: skip
                    prims.tcgen05_commit(acc2_full)
            else:
                # More A tiles than ring stages (TP4 x EP4 shapes): tiles cycle through the ring, each stage freed by
                # its MMAs' commit.
                _wait(sfa2_ready, 0)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    for j in range(nt2):
                        prims.tcgen05_cp(
                            s2t_shape,
                            cutlass.inttoptr(
                                (base_row_id << 16) | (base_col_id + SFA2_COL + j * SFA_COLS), 6, cutlass.Int32
                            ),
                            prims.Tcgen05SmemDesc.build(sfa2.subview(j * NUM_BYTES_SFA), leading_byte_offset=16,
                                                        stride_byte_offset=128, base_offset=0, layout=0),
                            multicast=s2t_multicast,
                        )  # fmt: skip
                _wait(b2_ready, 0)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                if prims.elect_sync():
                    for j in range(nt2):
                        prims.tcgen05_cp(
                            s2t_shape,
                            cutlass.inttoptr(
                                (base_row_id << 16) | (base_col_id + SFB2_COL + j * SFB_COLS), 6, cutlass.Int32
                            ),
                            prims.Tcgen05SmemDesc.build(sfb2.subview(j * NUM_BYTES_SFB), leading_byte_offset=16,
                                                        stride_byte_offset=128, base_offset=0, layout=0),
                            multicast=s2t_multicast,
                        )  # fmt: skip
                for j in range(nt2):
                    stage = g % cutlass.Int32(STAGES)
                    while not cute.arch.mbarrier_try_wait(fc2_full.subview(stage).data_ptr(),
                                                          (j // cutlass.Int32(STAGES)) % cutlass.Int32(2)):  # fmt: skip
                        pass
                    i2 = j // cutlass.Int32(KT2)
                    t2 = j % cutlass.Int32(KT2)
                    acc2_ptr = cutlass.inttoptr(
                        (base_row_id << 16) | (base_col_id + ACC2_COL + i2 * N), 6, cutlass.Float32
                    )
                    sfa_tmem_ptr = cutlass.inttoptr((base_row_id << 16) | (base_col_id + SFA2_COL + j * SFA_COLS), 6,
                                                    cutlass.Int32)  # fmt: skip
                    sfb_tmem_ptr = cutlass.inttoptr((base_row_id << 16) | (base_col_id + SFB2_COL + j * SFB_COLS), 6,
                                                    cutlass.Int32)  # fmt: skip
                    desc_a_base = prims.Tcgen05SmemDesc.build(
                        sA.subview(stage * NUM_BYTES_A), leading_byte_offset=16, stride_byte_offset=1024, base_offset=0,
                        layout=2,
                    )  # fmt: skip
                    desc_b_base = prims.Tcgen05SmemDesc.build(
                        b2.subview(j * NUM_BYTES_B2), leading_byte_offset=16, stride_byte_offset=1024, base_offset=0,
                        layout=2,
                    )  # fmt: skip
                    nkb = cutlass.Int32(KB2) - t2 * cutlass.Int32(NUM_KBLOCKS)
                    for kb2 in cutlass.range_constexpr(NUM_KBLOCKS):
                        if cutlass.Int32(kb2) < nkb:
                            idesc_u = idesc.set_sf_ids(a_sf_id=kb2, b_sf_id=kb2)
                            inc = ((MMA_INST_K * a_smem_width // 8) >> 4) * kb2
                            acc2_on = (t2 != cutlass.Int32(0)) | (
                                cutlass.Int32(kb2) != cutlass.Int32(0)
                            )
                            if prims.elect_sync():
                                prims.tcgen05_mma_block_scale(
                                    prims.MMABlockScaleKind.MXF8F6F4, prims.CTAGroup.CTA_1, acc2_ptr,
                                    desc_a_base + inc, desc_b_base + inc, idesc_u, acc2_on, sfa_tmem_ptr, sfb_tmem_ptr,
                                )  # fmt: skip
                    if prims.elect_sync():
                        prims.tcgen05_commit(ab_empty.subview(stage))
                    g = g + cutlass.Int32(1)
                if prims.elect_sync():
                    prims.tcgen05_commit(acc2_full)

    # ---- epilogue warps (0-3): FC2 weight scales spliced during FC1; the FC1 epilogue per unit; then the FC2 hand-off.
    if warp < 4:
        _wait(tmem_ready, 0)
        tmem_raw_addr = tmem_ptr_i32.load()
        base_col_id = tmem_raw_addr & 0xFFFF
        base_row_id = tmem_raw_addr >> 16
        row_id_with_warp_offset = base_row_id + warp * 32
        # FC2 weight scale atoms: MMA atom j (group i = j / KT2, k-tile t) word 4 m0 + jj = slot 4 i + jj's w2 atom
        # (its 128-row block, k-atom t) word 4 m0 + m1, m1 = this CTA's 32-row block within the 128.
        if is_fc2:
            sfa2_32 = cutlass.Array(sfa2.data_ptr(0), shape=(NT2 * NUM_BYTES_SFA // 4,), dtype=cutlass.Int32,
                                    alignment=16)  # fmt: skip
            vals = []
            for r in cutlass.range_constexpr(NT2 * (NUM_BYTES_SFA // 4) // EPI_THREADS):
                w = tidx + cutlass.Int32(r * EPI_THREADS)
                j = w // cutlass.Int32(NUM_BYTES_SFA // 4)
                rem = w % cutlass.Int32(NUM_BYTES_SFA // 4)
                slot = (j // cutlass.Int32(KT2)) * cutlass.Int32(4) + rem % cutlass.Int32(4)
                ok = (j < nt2) & (slot < g_n)
                e = cutlass.Int32(s_el.load(idx=cutlass.select_(ok, slot, cutlass.Int32(0))))
                src = (((e * cutlass.Int32(H // 128) + blk2) * cutlass.Int32(KA2) + j % cutlass.Int32(KT2))
                       * cutlass.Int32(NUM_BYTES_SFA // 4)
                       + (rem // cutlass.Int32(4)) * cutlass.Int32(4) + m1)  # fmt: skip
                v = w2s32.load(idx=cutlass.select_(ok, src, cutlass.Int32(0)))
                vals.append(cutlass.select_(ok, cutlass.Int32(v), cutlass.Int32(0)))
            for r in cutlass.range_constexpr(NT2 * (NUM_BYTES_SFA // 4) // EPI_THREADS):
                sfa2_32.store(vals[r], idx=tidx + cutlass.Int32(r * EPI_THREADS))
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            prims.mbarrier_arrive(sfa2_ready)
        # FC1 epilogue: lanes 0-63 column 0 (k-tiles 2j) + lanes 64-127 column 8 (k-tiles 2j + 1) = the unit's 64 rows;
        # warps 0-1 apply k3_moe's SiTU + MXFP8 requant to the 32 intermediate columns (one MX block).
        is_up = ((lane // 8) % 2) == 0
        up_mask = cutlass.select_(is_up, cutlass.Float32(1.0), cutlass.Float32(0.0))
        fc1_col_in_tile = warp * 16 + 2 * (lane % 8) + lane // 16
        acc_full_phase = 0
        for ui in range(mine):
            u = bx + ui * cutlass.Int32(NCTA)
            grp = u // cutlass.Int32(U1)
            t128 = (u % cutlass.Int32(U1)) // cutlass.Int32(2)
            half = u % cutlass.Int32(2)
            while not cute.arch.mbarrier_try_wait(acc_full.data_ptr(), acc_full_phase):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc_full_phase = acc_full_phase ^ 1
            tmem_ld = cutlass.inttoptr(
                (row_id_with_warp_offset << 16) | base_col_id, 6, cutlass.Float32
            )
            t2r_rmem = prims.tcgen05_ld("32x32b", tmem_ld, num=N)
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.mbarrier_arrive(acc_empty)
            if warp >= 2:
                for tq in cutlass.range_constexpr(M_MAX):
                    s_part.store(
                        cutlass.Float32(t2r_rmem[8 + tq]), idx=tq * ROWS1 + (warp - 2) * 32 + lane
                    )
            cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
            res = []
            for tq in cutlass.range_constexpr(M_MAX):
                xn = cutlass.Float32(t2r_rmem[tq]) + s_part.load(
                    idx=tq * ROWS1 + (warp % 2) * 32 + lane
                )
                partner = cute.arch.shuffle_sync_bfly(xn, 8)
                res.append(_situ(partner, xn))
                absv = cute.math.abs(res[tq]) * up_mask
                warp_amax = prims.redux_sync(absv, prims.ReductionKind.FMAX, 0xFFFFFFFF, abs=True)
                if lane == 0:
                    localmax_smem.store(warp_amax, idx=tq * 8 + warp)
            cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
            for tq in cutlass.range_constexpr(M_MAX):
                block_amax = cute.arch.fmax(
                    localmax_smem.load(idx=tq * 8), localmax_smem.load(idx=tq * 8 + 1)
                )
                byte, inv_scale = _block_e8m0(block_amax)
                qv = cute.arch.fmax(cute.arch.fmin(res[tq] * inv_scale, cutlass.Float32(E4M3_MAX)),
                                    cutlass.Float32(-E4M3_MAX))  # fmt: skip
                fp8_i8 = cutlass.Float8E4M3FN(qv).bitcast(cutlass.Int8)
                if fp8_i8 == cutlass.Int8(FP8_SENTINEL_I8):
                    fp8_i8 = cutlass.Int8(0)
                hrow = (grp * cutlass.Int32(M_MAX) + cutlass.Int32(tq)) * cutlass.Int32(H_ROW)
                if (warp < 2) & is_up:
                    hbuf.store(fp8_i8, idx=hrow + t128 * cutlass.Int32(MMA_M // 2) + half * cutlass.Int32(32)
                               + fc1_col_in_tile, alignment=1)  # fmt: skip
                if tidx == 0:
                    hbuf.store(
                        cutlass.Int8(byte & cutlass.Int32(0xFF)),
                        idx=hrow + cutlass.Int32(I_TP) + t128 * cutlass.Int32(2) + half,
                        alignment=1,
                    )
            cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
            if tidx == 0:
                _red_release_add(slot_addr, cutlass.Int32(1))
        # FC2: the call's count reaches U1 G; the intermediate rows into the B tiles (row n of tile (i, t) = slot
        # 4 i + n's K 128 t .., 128B-swizzled) and their scales into the B scale atoms (byte 16 n + k = block 4 t + k),
        # one load round; then the MMAs (warp 7), the accumulators and the combine. No local expert: zero rows.
        if is_fc2 & (g_n == cutlass.Int32(0)):
            if warp < M_MAX:
                _emit(cutlass.Float32(0.0), lane, row2, warp, warp * cutlass.Int32(H) + row2 + lane, out, lat_mc,
                      lat_flags, lat_rank)  # fmt: skip
        if is_fc2 & (g_n > cutlass.Int32(0)):
            if tidx == 0:
                while _load_acquire(slot_addr) < units:
                    pass
            cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
            b2_32 = cutlass.Array(
                b2.data_ptr(0), shape=(NT2 * NUM_BYTES_B2 // 4,), dtype=cutlass.Int32, alignment=16
            )
            sfb2_32 = cutlass.Array(sfb2.data_ptr(0), shape=(NT2 * NUM_BYTES_SFB // 4,), dtype=cutlass.Int32,
                                    alignment=16)  # fmt: skip
            got = []
            for r in cutlass.range_constexpr(STAGE_ROUNDS):
                qi = tidx + cutlass.Int32(r * EPI_THREADS)
                rr = qi // cutlass.Int32(VCH + SCH)  # intermediate row: slot rr / M, token rr % M
                ok = rr < g_n * cutlass.Int32(M_MAX)
                c = qi % cutlass.Int32(VCH + SCH)
                v4 = prims.load_ext(
                    hbuf32.subview(cutlass.select_(
                        ok, (rr * cutlass.Int32(H_ROW) + c * cutlass.Int32(16)) // cutlass.Int32(4), cutlass.Int32(0)
                    )),
                    dtype=cutlass.Int32, count=4, order="relaxed", scope="gpu",
                )  # fmt: skip
                got.append(v4)
            for r in cutlass.range_constexpr(STAGE_ROUNDS):
                qi = tidx + cutlass.Int32(r * EPI_THREADS)
                rr = qi // cutlass.Int32(VCH + SCH)
                sl = rr // cutlass.Int32(M_MAX)
                c = qi % cutlass.Int32(VCH + SCH)
                n = (rr % cutlass.Int32(M_MAX)) * cutlass.Int32(4) + sl % cutlass.Int32(
                    4
                )  # B row: 4 token + j
                v4 = got[r]
                if sl < g_n:
                    if c < cutlass.Int32(VCH):
                        t2 = c // cutlass.Int32(8)
                        cc = c % cutlass.Int32(8)
                        j = (sl // cutlass.Int32(4)) * cutlass.Int32(KT2) + t2
                        b2_32.store((cutlass.Int32(v4[0]), cutlass.Int32(v4[1]),
                                     cutlass.Int32(v4[2]), cutlass.Int32(v4[3])),
                                    idx=(j * cutlass.Int32(NUM_BYTES_B2) + n * cutlass.Int32(MMA_TILE_K)
                                         + (cc ^ n) * cutlass.Int32(16)) // cutlass.Int32(4), alignment=16)  # fmt: skip
                    else:
                        for wq in cutlass.range_constexpr(4):
                            t2 = (c - cutlass.Int32(VCH)) * cutlass.Int32(4) + cutlass.Int32(wq)
                            if t2 < cutlass.Int32(KT2):
                                j = (sl // cutlass.Int32(4)) * cutlass.Int32(KT2) + t2
                                sfb2_32.store(
                                    cutlass.Int32(v4[wq]),
                                    idx=(j * cutlass.Int32(NUM_BYTES_SFB) + n * cutlass.Int32(16))
                                    // cutlass.Int32(4),
                                )
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
            prims.mbarrier_arrive(b2_ready)
            _wait(acc2_full, 0)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            yv = []
            for gi in cutlass.range_constexpr(GROUPS2):
                for tq in cutlass.range_constexpr(M_MAX):
                    yv.append(prims.tcgen05_ld("32x32b", cutlass.inttoptr(
                        (row_id_with_warp_offset << 16) | (base_col_id + ACC2_COL + gi * N + 4 * tq + warp), 6,
                        cutlass.Float32), num=1))  # fmt: skip
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            for gi in cutlass.range_constexpr(GROUPS2):
                sl = cutlass.Int32(4 * gi) + warp
                if sl < g_n:
                    for tq in cutlass.range_constexpr(M_MAX):
                        ys.store(
                            cutlass.Float32(yv[gi * M_MAX + tq][0]),
                            idx=(sl * cutlass.Int32(M_MAX) + cutlass.Int32(tq)) * cutlass.Int32(R2)
                            + lane,
                        )
            cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=EPI_THREADS)
            if warp < M_MAX:
                tq = warp  # this warp's token
                ch = (
                    tq * cutlass.Int32(H)
                    + row2
                    + cutlass.Int32(4) * (lane % cutlass.Int32(8))
                    + lane // cutlass.Int32(8)
                )
                fast = cutlass.Boolean(False)
                if cutlass.const_expr(M_MAX == 1):
                    fast = g_n == cutlass.Int32(G_CAP)
                if fast:
                    # All G_CAP experts present (every TP16 call): k3_moe's slices at trace time; every product and sum
                    # rounded on its own, as in k3_moe (whose selects keep them apart).
                    acc = cutlass.Float32(0.0)
                    for si in cutlass.range_constexpr(REF_SLICES):
                        part = cutlass.Float32(0.0)
                        for sc in cutlass.range_constexpr(
                            _REF_SLICES_FULL[si][0], _REF_SLICES_FULL[si][1]
                        ):
                            slot = s_perm.load(idx=sc)
                            part = _add_rn(
                                part,
                                _mul_rn(
                                    ys.load(idx=slot * cutlass.Int32(R2) + lane), s_w.load(idx=slot)
                                ),
                            )
                        acc = _add_rn(acc, part)
                    _emit(acc, lane, row2, tq, ch, out, lat_mc, lat_flags, lat_rank)
                else:
                    # k3_moe's sum tree: min(G, 5) slices of consecutive experts
                    # (slice s = experts [s G / S, (s + 1) G / S)), each summed from 0 in ascending id (an expert
                    # without this token adds zero), then the slices summed in order.
                    n_sl = cutlass.select_(
                        g_n < cutlass.Int32(REF_SLICES), g_n, cutlass.Int32(REF_SLICES)
                    )
                    n_sl = cutlass.select_(n_sl < cutlass.Int32(1), cutlass.Int32(1), n_sl)
                    bnd = []
                    for b in cutlass.range_constexpr(1, REF_SLICES):
                        bnd.append(
                            cutlass.select_(
                                cutlass.Int32(b) < n_sl,
                                cutlass.Int32(b) * g_n // n_sl,
                                cutlass.Int32(-1),
                            )
                        )
                    zero = cutlass.Float32(0.0)
                    acc = zero
                    part = zero
                    for sc in cutlass.range_constexpr(G_CAP):
                        if cutlass.const_expr(sc > 0):
                            start = bnd[0] == cutlass.Int32(sc)
                            for b in cutlass.range_constexpr(1, REF_SLICES - 1):
                                start = start | (bnd[b] == cutlass.Int32(sc))
                            acc = cutlass.select_(start, acc + part, acc)
                            part = cutlass.select_(start, zero, part)
                        slot = s_perm.load(
                            idx=cutlass.select_(
                                cutlass.Int32(sc) < g_n, cutlass.Int32(sc), cutlass.Int32(0)
                            )
                        )
                        valid = cutlass.Int32(sc) < g_n
                        if cutlass.const_expr(M_MAX > 1):
                            valid = valid & (
                                ((s_tmask.load(idx=slot) >> tq) & cutlass.Int32(1))
                                != cutlass.Int32(0)
                            )
                        pq = slot * cutlass.Int32(M_MAX) + tq
                        part = part + cutlass.select_(
                            valid, ys.load(idx=pq * cutlass.Int32(R2) + lane) * s_w.load(idx=pq), zero
                        )  # fmt: skip
                    acc = acc + part
                    _emit(acc, lane, row2, tq, ch, out, lat_mc, lat_flags, lat_rank)

    # ---- teardown
    if tidx == 0:
        epochs.store(ep + cutlass.Int32(1), idx=bx)
    prims.barrier_cta_sync(0)
    if warp == MMA_WARP:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(cutlass.inttoptr(tmem_ptr_i32.load(), 6, cutlass.Float32), TMEM_COLS)


@cute.jit
def k3_moe_m1(
    a1_tensor: cute.Tensor,  # w3_w1_weight viewed (H/2, 2 I_PAD, E) FP4 bytes, K-major
    b1_tensor: cute.Tensor,  # MXFP8 activations viewed (H, M) FP8
    sfa1_tensor: cute.Tensor,  # w3_w1_weight_scale viewed (512, H/128, 2 I_PAD/128, E)
    sfb1_tensor: cute.Tensor,  # activation scales (M, H/32) E8M0
    a2_tensor: cute.Tensor,  # w2_weight viewed (I_PAD/2, H, E) FP4 bytes, K-major
    ids: cute.Tensor, wts: cute.Tensor, w2s32: cute.Tensor, hbuf: cute.Tensor, hbuf32: cute.Tensor,
    counts: cute.Tensor, epochs: cute.Tensor, out: cute.Tensor,
    lat_mc: cute.Tensor,  # PUSH: int32 words of the latent exchange's multicast mapping (else any int32 tensor)
    lat_flags: cute.Tensor,  # PUSH: int32 [4] of the exchange (else any int32 tensor)
    offset: cutlass.Int32, lat_rank: cutlass.Int32, stream: cuda_driver.CUstream,
):  # fmt: skip
    _kpp = a1_tensor.shape[0]
    _mw = a1_tensor.shape[1]
    _ew = a1_tensor.shape[2]
    tma_a1_desc = cuda.create_tensor_map_tiled(
        global_address=a1_tensor.iterator.toint(), dtype=a_dtype, global_dims=[_kpp * 2, _mw, _ew],
        global_strides=[_kpp // 16, (_mw * _kpp) // 16], box_dims=(MMA_TILE_K, ROWS1, 1),
        swizzle=cuda.TensorMapSwizzle.s128b, tma_format=TensorMapDataType.f416u4_align16b,
    )  # fmt: skip
    tma_b1_desc = cuda.create_tensor_map_tiled_from_view(
        b1_tensor,
        box_dims=(MMA_TILE_K, 1),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    sfa1_fp16 = cute.recast_tensor(sfa1_tensor, cutlass.Uint16)
    tma_sfa1_desc = cuda.create_tensor_map_tiled_from_view(
        sfa1_fp16, box_dims=(num_elts_atom_sf_fp16, 1, 1, 1), stride_order=(0, 1, 2, 3),
        swizzle=cuda.TensorMapSwizzle.none,
    )  # fmt: skip
    _kp2 = a2_tensor.shape[0]
    _h2 = a2_tensor.shape[1]
    _e2 = a2_tensor.shape[2]
    tma_a2_desc = cuda.create_tensor_map_tiled(
        global_address=a2_tensor.iterator.toint(), dtype=a_dtype, global_dims=[_kp2 * 2, _h2, _e2],
        global_strides=[_kp2 // 16, (_h2 * _kp2) // 16], box_dims=(MMA_TILE_K, R2, 1),
        swizzle=cuda.TensorMapSwizzle.s128b, tma_format=TensorMapDataType.f416u4_align16b,
    )  # fmt: skip
    k3_moe_m1_kernel(
        tma_a1_desc, tma_b1_desc, tma_sfa1_desc, tma_a2_desc, sfb1_tensor.iterator.toint(), ids, wts, w2s32, hbuf,
        hbuf32, counts, epochs, out, offset, lat_mc, lat_flags, lat_rank,
    ).launch(grid=(NCTA, 1, 1), block=(THREADS, 1, 1), cluster=(1, 1, 1), stream=stream, use_pdl=True)  # fmt: skip
