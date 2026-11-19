# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constants and device helpers of the KDA multi-token speculative verify kernels (conv-MTP).

``rubin_kda_mtp_decode_v2`` (one CTA per (head, request), or a cluster pair of CTAs while the
batch is small: a decode step can have far fewer tiles than SMs) and ``rubin_kda_mtp_decode_2cta``
(two CTAs per SM, for grids of more tiles than SMs) build on them. Both are specialized for the
verify contract: conv enabled, no bias, gated RMSNorm output, lower_bound gate, Q/K L2 norm, beta
sigmoid, W=4, K == V == 128, HV == H, ``TILE_V == V`` (one V pass), 512 threads.

Per request ``n`` a kernel processes an accepted count plus ``1 +
NUM_SPEC`` steps: it first *replays* the accepted draft tokens from the
``k/g/v/beta`` caches (raw conv inputs from the extended tail columns of
``cs_q/cs_k/cs_v``), then processes the ``1 + NUM_SPEC`` new tokens. The
recurrent state and the base conv windows are committed in place after the
first new (golden) token; the new spec tokens are cached for the next
round's replay. The pool invariant is therefore "state after the last
golden token; accepted drafts pending in the replay caches".

The recurrence runs in chunk (WY) form on the tcgen05 tensor cores, tf32 with
3xTF32 (hi/lo splits) on every product that feeds the state:

* GEMM1  ``[U0; O0] = S0 x [K~; Q~]`` -- S0 hi/lo staged into TMEM from the
  smem boxes, ``K~ = k e^{G_t}``, ``Q~ = q e^{G_t}`` (``G_t`` = cumulative
  gate of the chunk);
* GEMM3  ``L, B`` -- the T x T coupling matrices of the chunk (decays
  factored around the chunk middle and clamped at ``e^87``);
* ``V' = (I + diag(beta) L)^-1 diag(beta) (V - U0)`` by forward
  substitution (one warpgroup, one value row per thread);
* GEMM2  ``S_c = S0 e^{G_c} + V'^T Kbar`` -- issued as two column halves so
  the first half can start storing; the output combination ``O = O0 + B V'``
  rides along as 16 extra columns of the first half.

The committed state goes through swizzled smem boxes and a TMA bulk tensor
store; the gated RMSNorm (and the MXFP8 quantization of its output) runs on
warps that own no state rows while the store drains.

With ``SPLIT_V == 2`` two CTAs per (head, request) run as a cluster: both
recompute the k-side and the outputs, each seeds and stores half of the state
rows (the SM -> L2 drain of the 64 KB state is the tail of the kernel). The pair
must be co-scheduled because the state is updated in place. Two cluster
barrier phases order the pair: the first separates both CTAs' load-phase reads
of the conv windows and replay caches from their update, which only CTA 0
stores; the second separates every CTA's S0 load from the pair's state stores.
The TMA views of the state follow the tensor's own strides (the runtime's
per-layer state cache is a strided slice of a larger buffer). The state may be
fp32 or bf16; a bf16 state is exact in tf32 (the ``S0_lo`` sweep is skipped) and the committed state
is rounded to nearest on the way out.

The committed state matches an fp32 recurrence to 3xTF32 accuracy and FP8
output codes may differ from it by one step; the conv pools and replay caches
are bit-exact.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# 16 warps: the full 128-row recurrent state is processed in one pass (eight
# rows per warp) and the per-token precompute runs one (token, role) per warp.
NUM_THREADS = 512
# Two CTAs per (head, request) while HV * N <= SPLIT_MAX_TILES (up to 128 CTAs: one wave, the
# halved state store pays off); larger batches run one CTA per tile.
SPLIT_MAX_TILES = 64
TILE_K = 128
# Chunk length padding (MMA N granularity).
T_PAD = 16
# Largest draft count per verify: up to NUM_SPEC replayed drafts plus 1 + NUM_SPEC new tokens
# must fit the padded chunk (2 * 7 + 1 = 15 <= T_PAD). The executor falls back to the
# sequential verify for longer drafts.
MAX_NUM_SPEC = (T_PAD - 1) // 2
# TMEM columns: D4 (16) + D2 halves (2 x 64) + S0 hi (128) + S0 lo (128) + D1 (48) + D3 (16)
# = 464 of the 512 allocated.
TMEM_COLS = 512
# One 114 KB scratch region hosts, at different times, the TMA boxes of S0 / the committed
# state and the swizzled MMA operand tiles (floats).
SCRATCH_FLOATS = 29184
TP_STRIDE = 17  # padded row stride of the transposed (v|k, s) staging buffers


def swz64_k16(n, k):
    """Element offset in a K-major 64B-swizzled fp32 tile with 16 columns."""
    off = n * 16 + k
    return off ^ (((off >> 5) & 3) << 2)


@dsl_user_op
def cvt_bf16x2(hi: cutlass.Float32, lo: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    """Pack two fp32 values into one bf16x2 word (round to nearest even, `lo` in the low half)
    with a single cvt; returned as the fp32 bit pattern for the TMEM word stores."""
    w = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
    return w.bitcast(cutlass.Float32)


@dsl_user_op
def tcgen05_fence_before_thread_sync(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::before_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tcgen05_fence_after_thread_sync(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def round_tf32(x: cutlass.Float32) -> cutlass.Float32:
    """Round to the tf32 grid (rna on mantissa bit 13) so hi/lo splits are exact."""
    b = x.bitcast(cutlass.Int32)
    b = (b + cutlass.Int32(0x1000)) & cutlass.Int32(-8192)
    return b.bitcast(cutlass.Float32)
