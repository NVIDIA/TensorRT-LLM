# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL device kernel for KDA multi-token speculative verify (conv-MTP).

Source-integrated from the ``KDA_decode_mtp`` kernel drop ("readable KDA
conv-MTP") and restructured for the latency-bound production regime (one CTA
per (head, request), or a cluster pair of CTAs while the batch is small; TEP8
launches only ``H * N = 24`` tiles on a 212-SM Rubin). Specialized for the
benchmarked contract: conv enabled, no bias, optional BF16 gated RMSNorm
output, lower_bound gate, Q/K L2 norm, beta sigmoid, W=4, K == V == 128,
HV == H, ``TILE_V == V`` (one V pass), 512 threads.

Per request ``n`` the kernel processes an accepted count plus ``1 +
NUM_SPEC`` steps: it first *replays* the accepted draft tokens from the
``k/g/v/beta`` caches (raw conv inputs from the extended tail columns of
``cs_q/cs_k/cs_v``), then processes the ``1 + NUM_SPEC`` new tokens. The
recurrent state and the base conv windows are committed in place after the
first new (golden) token; the new spec tokens are cached for the next
round's replay. The pool invariant is therefore "state after the last
golden token; accepted drafts pending in the replay caches".

Structure (all stage boundaries are CTA barriers):

1. **Load phase.** Every warp issues its global loads up front, so the whole
   CTA waits for one DRAM round trip: the recurrent state ``h0`` through a TMA
   bulk load into swizzled smem boxes (consumed by the tensor-core stage), the
   raw conv inputs of every token (``x_q/x_k/x_v/g/beta`` for new tokens, the
   replay caches and ``cs_*`` tail columns for accepted drafts) into smem
   staging buffers, the conv weights, and the gated-RMSNorm operands for the
   epilogue.
2. **Per-token-parallel precompute.** Warp ``j`` computes the Q/K conv + SiLU +
   L2 norm of new token ``j`` and warp ``num_warps/2 + j`` its gate, beta and
   V conv, reading the width-4 windows from the staged raw sequence (rows
   ``0..W-2`` hold the committed window). Replay tokens were written to
   ``sQ/sK/sG/sBeta/sVall`` directly by the load phase. Conv-window commits
   and replay-cache writes are done by the warp that owns the token.
3. **Recurrence** in chunk (WY) form on the tcgen05 tensor cores, tf32 with
   3xTF32 (hi/lo splits) on every product that feeds the state:

   * GEMM1  ``[U0; O0] = S0 x [K~; Q~]`` -- S0 hi/lo staged into TMEM from the
     smem boxes, ``K~ = k e^{G_t}``, ``Q~ = q e^{G_t}`` (``G_t`` = cumulative
     gate of the chunk);
   * GEMM3  ``L, B`` -- the T x T coupling matrices of the chunk (decays
     factored around the chunk middle and clamped at ``e^87``);
   * ``V' = (I + diag(beta) L)^-1 diag(beta) (V - U0)`` by forward
     substitution (one warpgroup, one value row per thread);
   * GEMM2  ``S_c = S0 e^{G_c} + V'^T Kbar`` -- accumulator seeded from the
     registers holding S0, issued as two column halves so the first half can
     start storing; the output combination ``O = O0 + B V'`` rides along as 16
     extra columns of the first half.
4. **Store / epilogue.** The committed state goes through swizzled smem boxes
   and a TMA bulk tensor store; the gated RMSNorm and MXFP8 quantization read
   their operands from smem and run on the warps that own no state rows while
   the store drains.

With ``SPLIT_V == 2`` two CTAs per (head, request) run as a cluster: both
recompute the k-side and the outputs, each seeds and stores half of the state
rows (the SM -> L2 drain of the 64 KB state is the tail of the kernel). The pair
must be co-scheduled because the state is updated in place, and a cluster
barrier separates every CTA's S0 load from the pair's stores. The TMA views of
the state follow the tensor's own strides (the runtime's per-layer state cache
is a strided slice of a larger buffer). The state may be fp32 or bf16; a bf16
state is exact in tf32 (the ``S0_lo`` sweep is skipped) and the committed state
is rounded to nearest on the way out.

Measured on Rubin (nsys, state L2-resident) against the previous CUDA-core
kernel at H=12/num_spec=7 with mixed acceptance: N=2 8.61 us -> 7.07 us,
N=16 11.78 us -> 9.06 us; H=96/N=16 91.1 us -> 74.2 us. The committed state
is within 2e-6 of the fp32 kernel (3xTF32), the FP8 output codes differ in
~0.2 % of the elements by one step; the conv pools and replay caches are
bit-exact.

``BF16_MMA`` (``TRTLLM_KDA_MTP_BF16_MATH=1``) trades that precision for the
bf16-operand / fp32-accumulate scheme of the prefill kernel: S0 packed as bf16
in TMEM (64 columns), bf16 K-major smem tiles and K = 16 per instruction (18
MMAs per CTA instead of 76). Gates, cumulative gates, exponents, reductions
and the state accumulator stay fp32; the committed-state error vs fp32 grows to
~2e-2, the conv pools and replay caches stay bit-exact between the two paths.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

# 16 warps: the full 128-row recurrent state is processed in one pass (eight
# rows per warp) and the per-token precompute runs one (token, role) per warp.
NUM_THREADS = 512
# Two CTAs per (head, request) while HV * N <= SPLIT_MAX_TILES (up to 128 CTAs: one wave, the
# halved state store pays off); larger batches run one CTA per tile.
SPLIT_MAX_TILES = 64
TILE_K = 128
# Chunk length padding (MMA N granularity); NUM_SPEC <= 7 -> T_loop <= 15.
T_PAD = 16
# TMEM columns: D4 (16) + D2 halves (2 x 64) + S0 hi (128) + S0 lo (128) + D1 (48) + D3 (16)
# = 464 of the 512 allocated.
TMEM_COLS = 512
# One 114 KB scratch region hosts, at different times, the TMA boxes of S0 / the committed
# state, the swizzled MMA operand tiles and (bf16 output path) the per-warp 32x32 transposes.
# Offsets in floats, all multiples of 256 (1024 B) for the 128B-swizzle atoms.
SCRATCH_FLOATS = 29184
OFF_B4HI = 10240  # 16 x 16    Bcoef hi, immediately followed by Kbar hi -> N=80 GEMM2 half-0 tile
OFF_B4LO = 12544  # 16 x 16    Bcoef lo, immediately followed by Kbar lo
OFF_B1K = 27136  # 16 x 128   K~hi (duplicate, N=16 tile for the S0_lo sweep)
OFF_B1 = 0  # 48 x 128   [K~hi; Q~; K~lo]
OFF_A2HI = 6144  # 128 x 16   V'^T hi
OFF_A2LO = 8192  # 128 x 16   V'^T lo
OFF_B2HI = 10496  # 128 x 16   Kbar hi
OFF_B2LO = 12800  # 128 x 16   Kbar lo
OFF_A3 = 14848  # 64 x 128   [Kf; Qf hi; Qf lo; -]
OFF_B3HI = 23040  # 16 x 128   Kb hi
OFF_B3LO = 25088  # 16 x 128   Kb lo
TP_STRIDE = 17  # padded row stride of the transposed (v|k, s) staging buffers
# bf16 operand tiles (BF16_MMA): offsets in bf16 elements inside the same scratch region, all
# multiples of 512 (1024 B). [Bcoef; Kbar] is one contiguous SW32 tile (Bcoef rows 0..15, then the
# 128 Kbar rows); the GEMM2 half-1 B view starts at Kbar row 64.
OFFH_B1 = 0  # 32 x 128   [K~; Q~]
OFFH_A3 = 4096  # 64 x 128   [Kf; Qf; 0; -]
OFFH_B3 = 12288  # 16 x 128   Kb
OFFH_A2 = 14336  # 128 x 16   V'^T
OFFH_B4 = 16384  # 16 x 16    Bcoef, immediately followed by
OFFH_B2 = 16640  # 128 x 16   Kbar
XP_STRIDE = 33  # padded row stride of the 32x32 transpose blocks


def swz128(n, k, rows):
    """Element offset of (row n, col k) in a K-major 128B-swizzled fp32 tile with
    `rows` rows and 128 columns (canonical UMMA layout, see probe_swizzle.py)."""
    off = n * 32 + (k % 32) + (k // 32) * (rows * 32)
    return off ^ (((off >> 5) & 7) << 2)


def swz64_k16(n, k):
    """Element offset in a K-major 64B-swizzled fp32 tile with 16 columns."""
    off = n * 16 + k
    return off ^ (((off >> 5) & 3) << 2)


def sw128_bf16(n, k, rows):
    """Element offset of (n, k) in a K-major 128B-swizzled bf16 tile with `rows` rows and K = 128:
    two K blocks of 64 elements (rows x 128 B each), 16 B chunks XORed with n % 8."""
    kc = k % 64
    return (k // 64) * (rows * 64) + n * 64 + ((((kc >> 3) ^ (n & 7)) << 3) | (kc & 7))


def sw32_k16_bf16(n, k):
    """Element offset of (n, k) in a K-major 32B-swizzled bf16 tile with 16 columns."""
    off = n * 16 + k
    return off ^ (((off >> 6) & 1) << 3)


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


def round_bf16(x: cutlass.Float32) -> cutlass.Float32:
    """Round to the bf16 grid (rna on mantissa bit 16, two integer ops like round_tf32); the
    value stays an fp32 operand that is exact in tf32."""
    b = x.bitcast(cutlass.Int32)
    b = (b + cutlass.Int32(0x8000)) & cutlass.Int32(-65536)
    return b.bitcast(cutlass.Float32)


def round_hi(x: cutlass.Float32, BF16_MMA: bool) -> cutlass.Float32:
    """MMA operand rounding: tf32 grid for the 3xTF32 path (so the hi/lo split is exact), bf16
    grid for the bf16 operand path (the conversion into the bf16 tiles is then exact)."""
    if BF16_MMA:
        return round_bf16(x)
    return round_tf32(x)


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the SM global timer for optional in-kernel stage profiling."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.kernel
def kda_decode_mtp_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    onorm_g: cute.Tensor,
    onorm_weight: cute.Tensor,
    o: cute.Tensor,
    output_scale: cute.Tensor,
    ht: cute.Tensor,
    k_cache: cute.Tensor,
    g_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    smem_qk_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    precompute_control: cute.Tensor,
    TILE_V: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    onorm_eps: cutlass.Constexpr[float],
    scale_leading_dim: cutlass.Constexpr[int],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    USE_SETMAXREG: cutlass.Constexpr[bool],
    USE_REGULAR_METADATA: cutlass.Constexpr[bool],
    USE_PACKED_TOKEN_LAYOUT: cutlass.Constexpr[bool],
    USE_REG_Q_WEIGHTS: cutlass.Constexpr[bool],
    USE_ZERO_ACCEPTED: cutlass.Constexpr[bool],
    FUSE_PRECOMPUTE: cutlass.Constexpr[bool],
    RUNTIME_PRECOMPUTE_FLAG: cutlass.Constexpr[bool],
    FUSE_OUTPUT_NORM: cutlass.Constexpr[bool],
    QUANTIZE_OUTPUT: cutlass.Constexpr[bool],
    stage_timing: cute.Tensor,
    PROFILE_STAGES: cutlass.Constexpr[bool],
    tiled_mma1: cute.TiledMma,
    tiled_mma3: cute.TiledMma,
    tiled_mma1k: cute.TiledMma,
    smem_layout_b1: cute.ComposedLayout,
    smem_layout_a2: cute.ComposedLayout,
    smem_layout_a3: cute.ComposedLayout,
    smem_layout_b3: cute.ComposedLayout,
    smem_layout_b1k: cute.ComposedLayout,
    tiled_mma4: cute.TiledMma,
    tma_atom_t: cute.CopyAtom,
    mT_tma: cute.Tensor,
    box_layout: cute.ComposedLayout,
    tma_atom_l: cute.CopyAtom,
    mS_tma: cute.Tensor,
    tiled_mma2h: cute.TiledMma,
    smem_layout_b2h: cute.ComposedLayout,
    box_layout_t: cute.ComposedLayout,
    tiled_mma2q: cute.TiledMma,
    smem_layout_b2q: cute.ComposedLayout,
    SPLIT_V: cutlass.Constexpr[int],
    BF16_MMA: cutlass.Constexpr[bool],
    tiled_mma_f32a: cute.TiledMma,
):
    """KDA MTP verify: smem precompute + tensor-core chunk recurrence.

    With ``PROFILE_STAGES=True``, ``stage_timing`` must be an int64 tensor with at
    least ``SPLIT_V * HV * N * 64`` elements, indexed as
    ``((vh * HV + i_hv) * grid_n + i_n) * 64``. With profiling off it is never
    accessed and the host may pass any placeholder tensor.
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, vh = cute.arch.block_idx()
    # Two CTAs per (head, request): CTA vh owns the state rows [vh * V/2, (vh + 1) * V/2).
    i_h = i_hv
    if cutlass.const_expr(PROFILE_STAGES):
        t_stage0 = read_globaltimer()
        # Pre-declare so the dynamic `run_precompute` branch below only
        # reassigns (first assignment inside a dynamic branch is untraceable;
        # the name has to exist before the branch).
        t_stage1 = Int64(0)
        tp_a = Int64(0)
        tp_b = Int64(0)
        tp_c = Int64(0)
        tp_d = Int64(0)
        tp_e = Int64(0)
        tp_f = Int64(0)
        tp_g = Int64(0)
        tp_h = Int64(0)
        tp_i = Int64(0)
        tp_j = Int64(0)
        tp_k = Int64(0)
        tp_l = Int64(0)
        tp_m = Int64(0)
    if cutlass.const_expr(USE_REGULAR_METADATA):
        bos = i_n * (2 * NUM_SPEC + 1)
        eos = bos + (2 * NUM_SPEC + 1)
        slot = i_n
    elif cutlass.const_expr(USE_PACKED_TOKEN_LAYOUT):
        # Runtime verification packs exactly 1 + NUM_SPEC new rows per
        # request. Accepted drafts live in slot-indexed replay pools and do
        # not occupy rows in the projection outputs.
        bos = cutlass.Int32(0)
        eos = cutlass.Int32(0)
        slot = ssm_state_indices[i_n]
    else:
        bos = cu_seqlens[i_n]
        eos = cu_seqlens[i_n + 1]
        slot = ssm_state_indices[i_n]
    # V2 state views may place many layer pages between adjacent slots. Widen
    # the slot before indexing: slot * state stride can exceed INT32_MAX.
    slot = Int64(slot)
    h0_idx = slot * HV + i_hv
    hk_off = i_h * K
    hv_off = i_hv * V
    # Recurrent-state element type (fp32 or bf16). The TMA boxes have 128 B swizzled rows,
    # i.e. BOX_COLS = 32 fp32 or 64 bf16 columns; the state tile is NUM_SLABS such boxes.
    STATE_DTYPE = h0.element_type
    STATE_BF16 = STATE_DTYPE == cutlass.BFloat16
    # Recurrence precision. Default: 3xTF32 (hi/lo splits on every product that feeds the state).
    # BF16_MMA: bf16 tensor-core operands -- S0 packed as bf16 in TMEM (64 columns), bf16 K-major
    # smem tiles (half the bytes), K = 16 per MMA (a quarter of the instructions); the
    # bf16-operand / fp32-accumulate scheme of the prefill kernel. Gates, cumulative gates,
    # exponents, reductions and the state accumulator stay fp32 on both paths.
    # S0_LO: the 3xTF32 path needs the lo part of a fp32 state (a bf16 state is exact in tf32).
    S0_LO = (not STATE_BF16) and (not BF16_MMA)
    N1 = 2 * T_PAD if BF16_MMA else 3 * T_PAD  # GEMM1 N: [K~; Q~] (+ K~lo for the 3xTF32 sweep)
    KB_MMA = 16 if BF16_MMA else 8  # K per tcgen05 instruction
    # raw conv inputs of the committed window and of the pending drafts (bf16 in the runtime: the
    # projections are bf16, so the copy is exact; fp32 caches are accepted as well)
    CONV_DTYPE = cs_q.element_type
    # replay caches: k / v / beta may be bf16 (one bf16 ulp on the replayed drafts), the gate
    # cache stays fp32 (it feeds the cumulative-gate exponents)
    K_CACHE_DTYPE = k_cache.element_type
    V_CACHE_DTYPE = v_cache.element_type
    BETA_CACHE_DTYPE = beta_cache.element_type
    BOX_COLS = 128 // (STATE_DTYPE.width // 8)
    NUM_SLABS = K // BOX_COLS
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        commit_len = cutlass.Int32(0)
    else:
        if cutlass.const_expr(USE_PACKED_TOKEN_LAYOUT):
            commit_len = num_accepted_tokens[slot]
        else:
            commit_len = num_accepted_tokens[i_n]
        # Only NUM_SPEC drafts can be pending from the previous round. Clamp
        # so a malformed count cannot drive T_loop past the t_max-sized SMEM
        # buffers or the num_spec extents of the replay caches.
        if commit_len > NUM_SPEC:
            commit_len = cutlass.Int32(NUM_SPEC)
    if cutlass.const_expr(USE_PACKED_TOKEN_LAYOUT):
        # Existing body addresses every new row as bos + i_t, where i_t is
        # offset by commit_len after replay. Shift bos so those accesses map
        # to request-major packed rows without materialized cu_seqlens.
        bos = i_n * (1 + NUM_SPEC) - commit_len
        eos = bos + commit_len + 1 + NUM_SPEC
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        T_loop = cutlass.Int32(1 + NUM_SPEC)
        t_max = 1 + NUM_SPEC
    else:
        T_loop = commit_len + 1 + NUM_SPEC
        t_max = 2 * NUM_SPEC + 1
    vec_size = TILE_K // 32
    num_v_tiles = V // TILE_V
    NUM_V_ROWS = TILE_V // (NUM_THREADS // 32)
    # The restructured kernel assumes one V pass with eight rows per warp,
    # smem-resident conv weights and no setmaxreg register juggling.
    assert num_v_tiles == 1 and NUM_V_ROWS == 8, (
        f"kda_decode_mtp_kernel expects TILE_V == V and 8 rows per warp, got TILE_V={TILE_V}, V={V}"
    )
    assert not USE_REG_Q_WEIGHTS and not USE_SETMAXREG, (
        "kda_decode_mtp_kernel reads conv weights from smem and does not use setmaxreg"
    )
    q_weight_elems = KERNEL_WIDTH * K
    v_weight_elems = KERNEL_WIDTH * V
    k_weight_base = q_weight_elems
    v_weight_base = q_weight_elems + KERNEL_WIDTH * K
    conv_weight_elems = q_weight_elems + KERNEL_WIDTH * K + v_weight_elems
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    # Preserve the offsets and bank mapping of the shared-memory buffers from
    # the source kernel. The fused norm below reduces within one warp and does
    # not need this legacy CTA-reduction scratch.
    smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * V,)), 16)
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((conv_weight_elems,)),
        16,
    )
    if cutlass.const_expr(FUSE_OUTPUT_NORM):
        # Epilogue operands are staged in smem during the load phase.
        sOnG = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((1 + NUM_SPEC) * V,)), 16)
        sOnW = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)
    # Staged per-token inputs (raw conv sequence with the committed window
    # prepended in rows 0..W-2, raw gates, raw betas).
    sXQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXK = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * K,)), 16)
    sXV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(((t_max + 3) * V,)), 16)
    sGin = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * K,)), 16)
    sBetaRaw = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    # Per-head gate constants, staged in the load phase together with the other inputs.
    sDtBias = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    sAlog = smem.allocate_tensor(cutlass.Float32, cute.make_layout((32,)), 16)
    # ---- tensor-core chunk recurrence buffers ----
    assert NUM_SPEC <= 7 and t_max <= T_PAD, "chunk form pads the token axis to 16"
    tmem_holder = smem.allocate_array(cutlass.Int32, 1)
    mbar1 = smem.allocate_array(cutlass.Int64, 1)
    mbar2 = smem.allocate_array(cutlass.Int64, 1)
    mbar_ld = smem.allocate_array(cutlass.Int64, 1)
    mbar4 = smem.allocate_array(cutlass.Int64, 1)
    mbar2b = smem.allocate_array(cutlass.Int64, 1)
    sScratch = smem.allocate_tensor(cutlass.Float32, cute.make_layout((SCRATCH_FLOATS,)), 1024)
    # Swizzled fp32 MMA operand tiles as views into the scratch region (tf32 modes; the layouts
    # passed in are bf16 in mode 2, whose tiles are built below).
    if cutlass.const_expr(not BF16_MMA):
        sB1 = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B1, smem_layout_b1.inner, dtype=cutlass.Float32
            ),
            smem_layout_b1.outer,
        )
        sA2hi = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_A2HI, smem_layout_a2.inner, dtype=cutlass.Float32
            ),
            smem_layout_a2.outer,
        )
        sA2lo = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_A2LO, smem_layout_a2.inner, dtype=cutlass.Float32
            ),
            smem_layout_a2.outer,
        )
        sA3 = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_A3, smem_layout_a3.inner, dtype=cutlass.Float32
            ),
            smem_layout_a3.outer,
        )
        sB3hi = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B3HI, smem_layout_b3.inner, dtype=cutlass.Float32
            ),
            smem_layout_b3.outer,
        )
        sB3lo = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + OFF_B3LO, smem_layout_b3.inner, dtype=cutlass.Float32
            ),
            smem_layout_b3.outer,
        )
    sB1k = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator + OFF_B1K, smem_layout_b1k.inner, dtype=cutlass.Float32),
        smem_layout_b1k.outer,
    )
    sB1kf = cute.make_tensor(sScratch.iterator + OFF_B1K, cute.make_layout((T_PAD * 128,)))
    # Flat (unswizzled-pointer) views for chunked element writes with swz128/swz64_k16 offsets.
    sB1f = cute.make_tensor(sScratch.iterator + OFF_B1, cute.make_layout((48 * 128,)))
    sA2hif = cute.make_tensor(sScratch.iterator + OFF_A2HI, cute.make_layout((128 * 16,)))
    sA2lof = cute.make_tensor(sScratch.iterator + OFF_A2LO, cute.make_layout((128 * 16,)))
    sB2hif = cute.make_tensor(sScratch.iterator + OFF_B2HI, cute.make_layout((128 * 16,)))
    sB2lof = cute.make_tensor(sScratch.iterator + OFF_B2LO, cute.make_layout((128 * 16,)))
    sB4hif = cute.make_tensor(sScratch.iterator + OFF_B4HI, cute.make_layout((T_PAD * T_PAD,)))
    sB4lof = cute.make_tensor(sScratch.iterator + OFF_B4LO, cute.make_layout((T_PAD * T_PAD,)))
    sA3f = cute.make_tensor(sScratch.iterator + OFF_A3, cute.make_layout((64 * 128,)))
    sB3hif = cute.make_tensor(sScratch.iterator + OFF_B3HI, cute.make_layout((16 * 128,)))
    sB3lof = cute.make_tensor(sScratch.iterator + OFF_B3LO, cute.make_layout((16 * 128,)))
    # bf16 operand tiles (BF16_MMA): swizzled MMA views and flat views for the closed-form writes.
    sScratchH = cute.make_tensor(
        cute.recast_ptr(sScratch.iterator, dtype=cutlass.BFloat16),
        cute.make_layout((2 * SCRATCH_FLOATS,)),
    )
    if cutlass.const_expr(BF16_MMA):
        sB1b = cute.make_tensor(
            cute.recast_ptr(
                sScratchH.iterator + OFFH_B1, smem_layout_b1.inner, dtype=cutlass.BFloat16
            ),
            smem_layout_b1.outer,
        )
        sA3b = cute.make_tensor(
            cute.recast_ptr(
                sScratchH.iterator + OFFH_A3, smem_layout_a3.inner, dtype=cutlass.BFloat16
            ),
            smem_layout_a3.outer,
        )
        sB3b = cute.make_tensor(
            cute.recast_ptr(
                sScratchH.iterator + OFFH_B3, smem_layout_b3.inner, dtype=cutlass.BFloat16
            ),
            smem_layout_b3.outer,
        )
        sA2b = cute.make_tensor(
            cute.recast_ptr(
                sScratchH.iterator + OFFH_A2, smem_layout_a2.inner, dtype=cutlass.BFloat16
            ),
            smem_layout_a2.outer,
        )
        sB1b_f = cute.make_tensor(sScratchH.iterator + OFFH_B1, cute.make_layout((32 * 128,)))
        sA3b_f = cute.make_tensor(sScratchH.iterator + OFFH_A3, cute.make_layout((64 * 128,)))
        sB3b_f = cute.make_tensor(sScratchH.iterator + OFFH_B3, cute.make_layout((16 * 128,)))
        sA2b_f = cute.make_tensor(sScratchH.iterator + OFFH_A2, cute.make_layout((128 * 16,)))
        sB4b_f = cute.make_tensor(sScratchH.iterator + OFFH_B4, cute.make_layout((16 * 16,)))
        sB2b_f = cute.make_tensor(sScratchH.iterator + OFFH_B2, cute.make_layout((128 * 16,)))
    # Per-warp 32 x 32 transpose blocks (bf16 output path: per-warp state store).
    sXp = cute.make_tensor(sScratch.iterator, cute.make_layout((16 * 32 * XP_STRIDE,)))
    # GEMM3 result [L; B hi; B lo; -] as a plain (64 x 16) row-major matrix.
    sLB = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((4 * T_PAD, T_PAD), stride=(T_PAD, 1)), 16
    )
    sEc = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,)), 16)
    # V'^T (v, s) and Kbar^T (k, s) staged with a padded row stride (conflict-free for both
    # the row-per-thread producers and the 16-lanes-per-row tile writers).
    sVpT = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V * TP_STRIDE,)), 16)
    sKbT = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K * TP_STRIDE,)), 16)
    # Cumulative gates G_t (per k) reuse the raw-gate staging buffer (no longer needed after the precompute).
    sE = cute.make_tensor(sGin.iterator, cute.make_layout((t_max * K,)))
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar1, 1)
            cute.arch.mbarrier_init(mbar2, 1)
            cute.arch.mbarrier_init(mbar4, 1)
            cute.arch.mbarrier_init(mbar2b, 1)
            cute.arch.mbarrier_init(mbar_ld, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    # ---- TMA bulk load of S0 into the boxes; completes under the token staging /
    #      precompute and is consumed (mbar_ld) at the start of the chunk stage.
    # Issued by the whole of one warp (a TMA load issued from a single thread never
    # completes on this DSL build; the copy elects its issuing lane internally).
    # Load-phase roles besides the per-warp token rows (warp w stages token w): the last K
    # threads stage the committed q/k windows, the q/k conv weights and dt_bias, the V threads
    # below them the committed v window, and the S0 TMA is issued by the last v-window warp
    # (three window loads per lane). With 512 threads and K = V = 128 the staging lands on
    # warps 8-15, of which 11-15 own no token row at production trip counts (t_max <= 15).
    QK_STAGE_TID0 = NUM_THREADS - K
    V_STAGE_TID0 = QK_STAGE_TID0 - V
    TMA_WARP = QK_STAGE_TID0 // 32 - 1
    assert V_STAGE_TID0 >= 0 and V_STAGE_TID0 % 32 == 0 and QK_STAGE_TID0 % 32 == 0, (
        f"load-phase staging roles need whole warps for the last K + V threads, got "
        f"NUM_THREADS={NUM_THREADS}, K={K}, V={V}"
    )
    if warp_idx == TMA_WARP:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ld, V * K * (STATE_DTYPE.width // 8))
    for sl_h in cutlass.range_constexpr(NUM_SLABS):
        sTl_h = cute.make_tensor(
            cute.recast_ptr(
                sScratch.iterator + sl_h * (V * 32), box_layout.inner, dtype=STATE_DTYPE
            ),
            box_layout.outer,
        )
        gS_h_all = cute.local_tile(mS_tma, (V, BOX_COLS, 1), (0, sl_h, None))
        if cutlass.const_expr(USE_FLAT_LAYOUT):
            gS_h = gS_h_all[(None, None, 0, h0_idx)]
        else:
            gS_h = gS_h_all[(None, None, 0, (i_hv, slot))]
        tSl_h, tGl_h = cpasync.tma_partition(
            tma_atom_l,
            0,
            cute.make_layout(1),
            cute.group_modes(sTl_h, 0, 2),
            cute.group_modes(gS_h, 0, 2),
        )
        if warp_idx == TMA_WARP:
            cute.copy(tma_atom_l, tGl_h, tSl_h, tma_bar_ptr=mbar_ld)
    # TMEM allocation overlaps the in-flight loads; the pointer is retrieved after the
    # precompute barrier (first TMEM use is the S0 staging in the chunk stage).
    if warp_idx == 0:
        cute.arch.alloc_tmem(TMEM_COLS, tmem_holder)
    if cutlass.const_expr(PROFILE_STAGES):
        if tidx == 0:
            t_alloc = read_globaltimer()
            _, grid_n4, _ = cute.arch.grid_dim()
            stage_timing[((vh * HV + i_hv) * grid_n4 + i_n) * 64 + 18] = t_alloc - t_stage0
    wg = warp_idx // 4
    wg_tidx = tidx % 128
    r_q = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_decay = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_bk = cute.make_rmem_tensor(cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32)
    r_state = cute.make_rmem_tensor(
        cute.make_layout((NUM_V_ROWS * vec_size,), stride=(1,)), cutlass.Float32
    )
    r_exp_A = cutlass.Float32(0.0)
    # Issue the recurrent-state loads first so their DRAM latency overlaps the
    # whole precompute stage (one V pass: TILE_V == V).
    # S0 arrives through the TMA bulk load issued at kernel start (mbar_ld); it is read
    # from the swizzled smem boxes at the start of the chunk stage.
    # Warp-uniform request bounds: the TMA store inside this region needs a provably
    # uniform branch condition (legacy shifted layout only; the other layouts are unconditional).
    bos = cute.arch.make_warp_uniform(bos)
    eos = cute.arch.make_warp_uniform(eos)
    if cutlass.const_expr(USE_REGULAR_METADATA or USE_PACKED_TOKEN_LAYOUT) or eos > bos:
        if cutlass.const_expr(FUSE_PRECOMPUTE or RUNTIME_PRECOMPUTE_FLAG):
            if cutlass.const_expr(RUNTIME_PRECOMPUTE_FLAG):
                run_precompute = precompute_control[0] != 0
            else:
                run_precompute = True
            if run_precompute:
                # The last K threads stage the committed q/k windows, the q/k conv weights and
                # dt_bias (see the load-phase roles above).
                ct = tidx - QK_STAGE_TID0
                if ct >= 0:
                    for w in cutlass.range(KERNEL_WIDTH - 1, unroll_full=True):
                        sXQ[w * K + ct] = cutlass.Float32(cs_q[slot, hk_off + ct, w])
                        sXK[w * K + ct] = cutlass.Float32(cs_k[slot, hk_off + ct, w])
                for w in cutlass.range(KERNEL_WIDTH, unroll_full=True):
                    if ct >= 0:
                        sConvW[w * K + ct] = cutlass.Float32(w_q[hk_off + ct, w])
                        sConvW[k_weight_base + w * K + ct] = cutlass.Float32(w_k[hk_off + ct, w])
                for ld in cutlass.range(max(1, V * KERNEL_WIDTH // NUM_THREADS), unroll_full=True):
                    flat = ld * NUM_THREADS + tidx
                    if flat < V * KERNEL_WIDTH:
                        sConvW[v_weight_base + flat] = cutlass.Float32(
                            w_v[hv_off + flat % V, flat // V]
                        )
                # Load phase: one round trip for every per-token input. Warp w
                # stages tokens w, w + num_warps, ...; replay tokens go straight
                # into sQ/sK/sG/sBeta/sVall, new tokens stage their raw inputs.
                if ct >= 0:
                    sDtBias[ct] = cutlass.Float32(dt_bias[i_h * K + ct])
                if tidx == 0:
                    sAlog[0] = cutlass.Float32(A_log[i_h])
                cv = tidx - V_STAGE_TID0
                if cv >= 0 and cv < V:
                    for _w in cutlass.range(KERNEL_WIDTH - 1, unroll_full=True):
                        sXV[_w * V + cv] = cutlass.Float32(cs_v[slot, hv_off + cv, _w])
                for _rep in cutlass.range(
                    (t_max + NUM_THREADS // 32 - 1) // (NUM_THREADS // 32), unroll_full=True
                ):
                    _t = _rep * (NUM_THREADS // 32) + warp_idx
                    if _t < T_loop:
                        if _t < commit_len:
                            for i in cutlass.range(vec_size, unroll_full=True):
                                k_idx = i * 32 + in_warp_tid
                                # replayed drafts produce no output: their q rows are unused
                                sQ[_t, k_idx] = cutlass.Float32(0.0)
                                sK[_t, k_idx] = cutlass.Float32(k_cache[slot, _t, hk_off + k_idx])
                                sG[_t, k_idx] = cutlass.Float32(g_cache[slot, _t, hk_off + k_idx])
                                sXQ[(_t + 3) * K + k_idx] = cutlass.Float32(
                                    cs_q[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + _t]
                                )
                                sXK[(_t + 3) * K + k_idx] = cutlass.Float32(
                                    cs_k[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + _t]
                                )
                                sXV[(_t + 3) * V + k_idx] = cutlass.Float32(
                                    cs_v[slot, hv_off + k_idx, KERNEL_WIDTH - 1 + _t]
                                )
                                sVall[_t * V + k_idx] = cutlass.Float32(
                                    v_cache[slot, _t, hv_off + k_idx]
                                )
                            if in_warp_tid == 0:
                                sBeta[_t] = cutlass.Float32(beta_cache[slot, _t, i_hv])
                        else:
                            _tok = bos + _t
                            for i in cutlass.range(vec_size, unroll_full=True):
                                k_idx = i * 32 + in_warp_tid
                                if cutlass.const_expr(USE_FLAT_LAYOUT):
                                    sXQ[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_q[0, _tok, hk_off + k_idx]
                                    )
                                    sXK[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_k[0, _tok, hk_off + k_idx]
                                    )
                                    sXV[(_t + 3) * V + k_idx] = cutlass.Float32(
                                        x_v[0, _tok, hv_off + k_idx]
                                    )
                                else:
                                    sXQ[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_q[0, _tok, i_h, k_idx]
                                    )
                                    sXK[(_t + 3) * K + k_idx] = cutlass.Float32(
                                        x_k[0, _tok, i_h, k_idx]
                                    )
                                    sXV[(_t + 3) * V + k_idx] = cutlass.Float32(
                                        x_v[0, _tok, i_hv, k_idx]
                                    )
                                sGin[_t * K + k_idx] = cutlass.Float32(g[0, _tok, i_hv, k_idx])
                            if in_warp_tid == 0:
                                sBetaRaw[_t] = cutlass.Float32(beta[0, _tok, i_hv])
                # Padding rows [T_loop, t_max) of the gates, values and betas are zeroed (under the
                # load latency) so the chunk stage's r_Y / cumulative-gate / V' loops are branch-free.
                if tidx < K:
                    for _z in cutlass.range(T_loop, t_max, 1):
                        sG[_z, tidx] = cutlass.Float32(0.0)
                        sVall[_z * V + tidx] = cutlass.Float32(0.0)
                        if tidx == 0:
                            sBeta[_z] = cutlass.Float32(0.0)
                if cutlass.const_expr(FUSE_OUTPUT_NORM):
                    # Threads 224+ stage the gated-RMSNorm operands for the epilogue.
                    if tidx >= 224:
                        _pf_tid = tidx - 224
                        for _e in cutlass.range(
                            ((1 + NUM_SPEC) * V + (NUM_THREADS - 224) - 1) // (NUM_THREADS - 224),
                            unroll_full=True,
                        ):
                            _flat = _e * (NUM_THREADS - 224) + _pf_tid
                            if _flat < (1 + NUM_SPEC) * V:
                                _tok = _flat // V
                                _vv = _flat % V
                                sOnG[_flat] = cutlass.Float32(
                                    onorm_g[0, i_n * (1 + NUM_SPEC) + _tok, i_hv, _vv]
                                )
                        for _e in cutlass.range(
                            (V + (NUM_THREADS - 224) - 1) // (NUM_THREADS - 224), unroll_full=True
                        ):
                            _flat = _e * (NUM_THREADS - 224) + _pf_tid
                            if _flat < V:
                                sOnW[_flat] = cutlass.Float32(onorm_weight[_flat])
                cute.arch.barrier()
                # ---- per-token-parallel precompute ----
                num_warps = NUM_THREADS // 32
                half = num_warps // 2
                role = warp_idx // half
                r_exp_A = cute.math.exp(sAlog[0], fastmath=True)
                for _rep in cutlass.range((1 + NUM_SPEC + half - 1) // half, unroll_full=True):
                    j = _rep * half + warp_idx % half
                    if j < 1 + NUM_SPEC:
                        t = commit_len + j
                        if role == 0:
                            for i in cutlass.range(vec_size, unroll_full=True):
                                k_idx = i * 32 + in_warp_tid
                                r_conv_0 = cutlass.Float32(0.0)
                                for w in cutlass.range(KERNEL_WIDTH, unroll_full=True):
                                    r_conv_0 += sXQ[(t + w) * K + k_idx] * sConvW[w * K + k_idx]
                                e0 = cute.math.exp(-r_conv_0, fastmath=True)
                                r_q[i] = r_conv_0 * cute.arch.rcp_approx(cutlass.Float32(1.0) + e0)
                            sum_q = 0.0
                            for i in cutlass.range(vec_size, unroll_full=True):
                                sum_q += r_q[i] * r_q[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_q += cute.arch.shuffle_sync_bfly(
                                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            rnorm_q_scaled = cute.math.rsqrt(sum_q + 1e-06, fastmath=True) * scale
                            for i in cutlass.range(vec_size, unroll_full=True):
                                r_q[i] = r_q[i] * rnorm_q_scaled
                            for i in cutlass.range(vec_size, unroll_full=True):
                                k_idx = i * 32 + in_warp_tid
                                sQ[t, k_idx] = r_q[i]
                            for i in cutlass.range(vec_size, unroll_full=True):
                                k_idx = i * 32 + in_warp_tid
                                r_conv = sXK[t * K + k_idx] * sConvW[k_weight_base + 0 * K + k_idx]
                                for w in cutlass.range(1, KERNEL_WIDTH, unroll_full=True):
                                    r_conv += (
                                        sXK[(t + w) * K + k_idx]
                                        * sConvW[k_weight_base + w * K + k_idx]
                                    )
                                r_conv = r_conv * cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-r_conv, fastmath=True)
                                )
                                r_k[i] = r_conv
                            sum_k = 0.0
                            for i in cutlass.range(vec_size, unroll_full=True):
                                sum_k += r_k[i] * r_k[i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_k += cute.arch.shuffle_sync_bfly(
                                    sum_k, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                            for i in cutlass.range(vec_size, unroll_full=True):
                                r_k[i] = r_k[i] * rnorm_k
                            for i in cutlass.range(vec_size, unroll_full=True):
                                k_idx = i * 32 + in_warp_tid
                                sK[t, k_idx] = r_k[i]
                            if j == 0:
                                for i in cutlass.range(vec_size, unroll_full=True):
                                    k_idx = i * 32 + in_warp_tid
                                    for w in cutlass.range(KERNEL_WIDTH - 1, unroll_full=True):
                                        cs_q[slot, hk_off + k_idx, w] = sXQ[
                                            (t + 1 + w) * K + k_idx
                                        ].to(CONV_DTYPE)
                                        cs_k[slot, hk_off + k_idx, w] = sXK[
                                            (t + 1 + w) * K + k_idx
                                        ].to(CONV_DTYPE)
                            else:
                                cache_pos = j - 1
                                for i in cutlass.range(vec_size, unroll_full=True):
                                    k_idx = i * 32 + in_warp_tid
                                    k_cache[slot, cache_pos, hk_off + k_idx] = r_k[i].to(
                                        K_CACHE_DTYPE
                                    )
                                    cs_q[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = sXQ[
                                        (t + 3) * K + k_idx
                                    ].to(CONV_DTYPE)
                                    cs_k[slot, hk_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = sXK[
                                        (t + 3) * K + k_idx
                                    ].to(CONV_DTYPE)
                        else:
                            for i in cutlass.range(vec_size, unroll_full=True):
                                k_idx = i * 32 + in_warp_tid
                                r_g_raw = sGin[t * K + k_idx]
                                r_g_raw = r_g_raw + sDtBias[k_idx]
                                exp_A_x = r_exp_A * r_g_raw
                                sigmoid_val = cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-exp_A_x, fastmath=True)
                                )
                                r_gk = lower_bound * sigmoid_val
                                sG[t, k_idx] = r_gk
                                r_decay[i] = r_gk
                            r_beta_t = cute.arch.rcp_approx(
                                cutlass.Float32(1.0) + cute.math.exp(-sBetaRaw[t], fastmath=True)
                            )
                            if in_warp_tid == 0:
                                sBeta[t] = r_beta_t
                            for i in cutlass.range(vec_size, unroll_full=True):
                                v_idx = i * 32 + in_warp_tid
                                _v_conv = 0.0
                                for w in cutlass.range(KERNEL_WIDTH, unroll_full=True):
                                    _v_conv += (
                                        sXV[(t + w) * V + v_idx]
                                        * sConvW[v_weight_base + w * V + v_idx]
                                    )
                                _v_conv = _v_conv * cute.arch.rcp_approx(
                                    cutlass.Float32(1.0) + cute.math.exp(-_v_conv, fastmath=True)
                                )
                                sVall[t * V + v_idx] = _v_conv
                                r_bk[i] = _v_conv
                            if j == 0:
                                for i in cutlass.range(vec_size, unroll_full=True):
                                    v_idx = i * 32 + in_warp_tid
                                    for w in cutlass.range(KERNEL_WIDTH - 1, unroll_full=True):
                                        cs_v[slot, hv_off + v_idx, w] = sXV[
                                            (t + 1 + w) * V + v_idx
                                        ].to(CONV_DTYPE)
                            else:
                                cache_pos = j - 1
                                for i in cutlass.range(vec_size, unroll_full=True):
                                    k_idx = i * 32 + in_warp_tid
                                    g_cache[slot, cache_pos, hk_off + k_idx] = r_decay[i]
                                    v_cache[slot, cache_pos, hv_off + k_idx] = r_bk[i].to(
                                        V_CACHE_DTYPE
                                    )
                                    cs_v[slot, hv_off + k_idx, KERNEL_WIDTH - 1 + cache_pos] = sXV[
                                        (t + 3) * V + k_idx
                                    ].to(CONV_DTYPE)
                                if in_warp_tid == 0:
                                    beta_cache[slot, cache_pos, i_hv] = r_beta_t.to(
                                        BETA_CACHE_DTYPE
                                    )
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
            else:
                cute.arch.barrier()
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
        else:
            cute.arch.barrier()
            if cutlass.const_expr(PROFILE_STAGES):
                cute.arch.barrier()
                t_stage1 = read_globaltimer()
        cute.arch.barrier()
        # =================================================================
        # Chunk (WY-form) recurrence on tensor cores.
        # =================================================================
        tmem_ptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_holder)
        pool = cutlass.utils.TmemBufferPool(tmem_ptr, TMEM_COLS)
        tD4_frag = tiled_mma4.make_fragment_C(tiled_mma4.partition_shape_C((V, T_PAD)))
        tD4 = pool.allocate_tensor(tD4_frag.layout, cutlass.Float32)  # outputs (columns 0..15)
        tD2h_frag = tiled_mma2h.make_fragment_C(tiled_mma2h.partition_shape_C((V, K // 2)))
        tD2h0 = pool.allocate_tensor(
            tD2h_frag.layout, cutlass.Float32
        )  # GEMM2 half 0: k columns 0..63
        tD2h1 = pool.allocate_tensor(
            tD2h_frag.layout, cutlass.Float32
        )  # GEMM2 half 1: k columns 64..127
        tD2q_frag = tiled_mma2q.make_fragment_C(tiled_mma2q.partition_shape_C((V, T_PAD + K // 2)))
        tD2q = cute.make_tensor(tD4.iterator, tD2q_frag.layout)  # [D4 | ΔS half 0] accumulator view
        tA_frag = tiled_mma_f32a.make_fragment_A(tiled_mma_f32a.partition_shape_A((V, K)))
        if cutlass.const_expr(BF16_MMA):
            # S0 as packed bf16 (even k in the low half of each 32-bit column): 64 columns, staged
            # through an fp32 word view and consumed by the MMA through the bf16 A-fragment view.
            tAw = pool.allocate_tensor(
                cute.make_layout(((V, 8), 1, K // 16), stride=((65536, 1), 0, 8)), cutlass.Float32
            )
            tAb = cute.make_tensor(
                cute.recast_ptr(tAw.iterator, dtype=cutlass.BFloat16),
                tiled_mma1.make_fragment_A(tiled_mma1.partition_shape_A((V, K))).layout,
            )
            tAhi = tAw
            tAlo = tAw
        else:
            tAhi = pool.allocate_tensor(tA_frag.layout, cutlass.Float32)
            tAlo = pool.allocate_tensor(tA_frag.layout, cutlass.Float32)
        tD1_frag = tiled_mma1.make_fragment_C(tiled_mma1.partition_shape_C((V, N1)))
        tD1 = pool.allocate_tensor(tD1_frag.layout, cutlass.Float32)
        tD3_frag = tiled_mma3.make_fragment_C(tiled_mma3.partition_shape_C((4 * T_PAD, T_PAD)))
        tD3 = pool.allocate_tensor(tD3_frag.layout, cutlass.Float32)
        tD2 = cute.make_tensor(
            tD2h0.iterator, tA_frag.layout
        )  # A-view layout for st/ld (all 128 columns)

        st32 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
        ld16 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
        ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
        tiled_stA = tcgen05.make_tmem_copy(st32, tD2)
        thr_stA = tiled_stA.get_slice(wg_tidx)
        tD2_st = thr_stA.partition_D(tD2)
        rA_shape = thr_stA.partition_S(tD2).shape
        if cutlass.const_expr(BF16_MMA):
            st16w = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
            thr_stAw = tcgen05.make_tmem_copy(st16w, tAw).get_slice(wg_tidx)
            tAw_st = thr_stAw.partition_D(tAw)
            r_pack = cute.make_rmem_tensor(thr_stAw.partition_S(tAw).shape[:-1], cutlass.Float32)
            r_pack_flat = cute.make_tensor(r_pack.iterator, cute.make_layout((16,)))
            tAhi_st = tD2_st
            tAlo_st = tD2_st
        else:
            tAhi_st = thr_stA.partition_D(tAhi)
            tAlo_st = thr_stA.partition_D(tAlo)
        tiled_ldD2 = tcgen05.make_tmem_copy(ld32, tD2)
        thr_ldD2 = tiled_ldD2.get_slice(wg_tidx)
        tD2_ld = thr_ldD2.partition_S(tD2)
        rS = cute.make_rmem_tensor(thr_ldD2.partition_D(tD2).shape[:-1], cutlass.Float32)
        rS_flat = cute.make_tensor(rS.iterator, cute.make_layout((32,)))
        # TMA store partitions for the NUM_SLABS boxes of this CTA's committed-state tile.
        # Created here, outside the dynamic request-bounds region (the exec-TMA lowering is
        # only legal at this level); the copies are issued inside.
        sBoxf = cute.make_tensor(sScratch.iterator, cute.make_layout((V * K,)))
        sBoxh = cute.make_tensor(
            cute.recast_ptr(sScratch.iterator, dtype=cutlass.BFloat16), cute.make_layout((V * K,))
        )
        tS_list = []
        tG_list = []
        for sl_h in cutlass.range_constexpr(NUM_SLABS):
            sT_h = cute.make_tensor(
                cute.recast_ptr(
                    sScratch.iterator + sl_h * ((V // SPLIT_V) * 32),
                    box_layout_t.inner,
                    dtype=STATE_DTYPE,
                ),
                box_layout_t.outer,
            )
            gT_h_all = cute.local_tile(mT_tma, (V // SPLIT_V, BOX_COLS, 1), (vh, sl_h, None))
            if cutlass.const_expr(USE_FLAT_LAYOUT):
                gT_h = gT_h_all[(None, None, 0, h0_idx)]
            else:
                gT_h = gT_h_all[(None, None, 0, (i_hv, slot))]
            tS_h, tG_h = cpasync.tma_partition(
                tma_atom_t,
                0,
                cute.make_layout(1),
                cute.group_modes(sT_h, 0, 2),
                cute.group_modes(gT_h, 0, 2),
            )
            tS_list.append(tS_h)
            tG_list.append(tG_h)
        r_hi = cute.make_rmem_tensor(rA_shape[:-1], cutlass.Float32)
        r_lo = cute.make_rmem_tensor(rA_shape[:-1], cutlass.Float32)
        r_hi_flat = cute.make_tensor(r_hi.iterator, cute.make_layout((32,)))
        r_lo_flat = cute.make_tensor(r_lo.iterator, cute.make_layout((32,)))

        # ---- S0 row v = wg_tidx, columns [32 wg, +32), from the TMA-loaded swizzled box
        #      (16 B chunks: conflict-free LDS.128 under the 128B swizzle)
        cute.arch.mbarrier_wait(mbar_ld, 0)
        # in-place state update: the peer CTA must not store its half before our S0 load
        # completed -> arrive here, wait right before the state store.
        if cutlass.const_expr(SPLIT_V == 2):
            cute.arch.cluster_arrive_relaxed()
        if cutlass.const_expr(PROFILE_STAGES):
            if tidx == 0:
                _tq28 = read_globaltimer()
                _, _gq28, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq28 + i_n) * 64 + 28] = _tq28 - t_stage0
        vx = 4 * (wg_tidx % 8)
        if cutlass.const_expr(STATE_BF16):
            # bf16 box: 64 columns per 128 B row, so this warpgroup's 32 columns are the 16 B
            # chunks [4 (wg % 2), +4) of box wg // 2. A bf16 value is exact in tf32: no lo part,
            # and the S0_lo sweep of GEMM1 is skipped.
            for c in cutlass.range(4, unroll_full=True):
                chunk = (4 * (wg % 2) + c) ^ (wg_tidx % 8)
                off_l = (wg // 2) * (V * 64) + 64 * wg_tidx + 8 * chunk
                for i in cutlass.range(8, unroll_full=True):
                    r_state[8 * c + i] = cutlass.Float32(sBoxh[off_l + i])
            if cutlass.const_expr(not BF16_MMA):
                for i in cutlass.range(32, unroll_full=True):
                    r_hi_flat[i] = r_state[i]
                for j in cutlass.range_constexpr(4):
                    if wg == j:
                        cute.copy(st32, r_hi, tAhi_st[(None, None, None, j)])
        else:
            for c in cutlass.range(8, unroll_full=True):
                off_l = wg * (V * 32) + 32 * wg_tidx + ((4 * c) ^ vx)
                for i in cutlass.range(4, unroll_full=True):
                    r_state[4 * c + i] = sBoxf[off_l + i]
            if cutlass.const_expr(not BF16_MMA):
                for i in cutlass.range(32, unroll_full=True):
                    s_hi = round_hi(r_state[i], BF16_MMA)
                    r_hi_flat[i] = s_hi
                    if cutlass.const_expr(S0_LO):
                        r_lo_flat[i] = r_state[i] - s_hi
                for j in cutlass.range_constexpr(4):
                    if wg == j:
                        cute.copy(st32, r_hi, tAhi_st[(None, None, None, j)])
                        if cutlass.const_expr(S0_LO):
                            cute.copy(st32, r_lo, tAlo_st[(None, None, None, j)])
        if cutlass.const_expr(BF16_MMA):
            # this warpgroup's 32 state columns -> 16 packed bf16x2 words -> TMEM word slab wg
            for i in cutlass.range(16, unroll_full=True):
                r_pack_flat[i] = cvt_bf16x2(r_state[2 * i + 1], r_state[2 * i])
            for j in cutlass.range_constexpr(4):
                if wg == j:
                    cute.copy(st16w, r_pack, tAw_st[(None, None, None, j)])
        cute.arch.fence_view_async_tmem_store()
        if cutlass.const_expr(PROFILE_STAGES):
            if tidx == 0:
                _tq29 = read_globaltimer()
                _, _gq29, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq29 + i_n) * 64 + 29] = _tq29 - t_stage0

        # ---- cumulative gates per k column (thread kq, quarter q owns tokens t = q mod 4):
        #      G_t -> sE, the T x T coupling factors, and the GEMM1 / GEMM3 operand rows written
        #      with closed-form 128B-swizzle offsets (lanes = consecutive k: conflict-free).
        kq = tidx % K
        quarter = tidx // K
        t_mid = T_loop // 2
        r_G = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
        g_run = cutlass.Float32(0.0)
        for t in cutlass.range(T_PAD, unroll_full=True):
            if t < t_max:
                g_run = g_run + sG[t, kq]
            r_G[t] = g_run
        g_mid = cutlass.Float32(0.0)
        for t in cutlass.range(T_PAD, unroll_full=True):
            if t == t_mid:
                g_mid = r_G[t]
        # Factors of the T x T coupling: e^{G_t - G_mid} (t side) and e^{G_mid - G_s} (s side),
        # clamped at e^87 so a chunk whose decay exceeds the fp32 range yields ~0 instead of
        # inf/NaN (the affected products are < e^{-|g|} and negligible either way).
        exp_clamp = cutlass.Float32(87.0)
        # the scratch region (S0 boxes) is reused for the MMA operand tiles below
        cute.arch.barrier()
        if cutlass.const_expr(PROFILE_STAGES):
            if tidx == 0:
                _tq30 = read_globaltimer()
                _, _gq30, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq30 + i_n) * 64 + 30] = _tq30 - t_stage0
        ka = kq // 32
        kc = kq % 32
        b1_base = ka * (3 * T_PAD * 32)
        a3_base = ka * (4 * T_PAD * 32)
        b3_base = ka * (T_PAD * 32)
        b1k_base = ka * (T_PAD * 32)
        for t in cutlass.range(T_PAD, unroll_full=True):
            if quarter == t % 4:
                kt_hi = cutlass.Float32(0.0)
                kt_lo = cutlass.Float32(0.0)
                qt_r = cutlass.Float32(0.0)
                kf = cutlass.Float32(0.0)
                qf_hi = cutlass.Float32(0.0)
                qf_lo = cutlass.Float32(0.0)
                kb_hi = cutlass.Float32(0.0)
                kb_lo = cutlass.Float32(0.0)
                if t < T_loop:
                    d_t = r_G[t] - g_mid
                    e_t = cute.math.exp(-cute.arch.fmax(-d_t, -exp_clamp), fastmath=True)
                    e_bt = cute.math.exp(-cute.arch.fmax(d_t, -exp_clamp), fastmath=True)
                    eg_t = cute.math.exp(r_G[t], fastmath=True)
                    sE[t * K + kq] = r_G[t]
                    if t == commit_len:
                        sEc[kq] = eg_t
                    k_t = sK[t, kq]
                    q_t = sQ[t, kq]
                    kt = k_t * eg_t
                    kt_hi = round_hi(kt, BF16_MMA)
                    qt_r = round_hi(q_t * eg_t, BF16_MMA)
                    kf = round_hi(k_t * e_t, BF16_MMA)
                    qf = q_t * e_t
                    qf_hi = round_hi(qf, BF16_MMA)
                    kb = k_t * e_bt
                    kb_hi = round_hi(kb, BF16_MMA)
                    if cutlass.const_expr(not BF16_MMA):
                        kt_lo = kt - kt_hi
                        qf_lo = qf - qf_hi
                        kb_lo = kb - kb_hi
                if cutlass.const_expr(BF16_MMA):
                    sB1b_f[sw128_bf16(t, kq, 2 * T_PAD)] = cutlass.BFloat16(kt_hi)
                    sB1b_f[sw128_bf16(T_PAD + t, kq, 2 * T_PAD)] = cutlass.BFloat16(qt_r)
                    sA3b_f[sw128_bf16(t, kq, 4 * T_PAD)] = cutlass.BFloat16(kf)
                    sA3b_f[sw128_bf16(T_PAD + t, kq, 4 * T_PAD)] = cutlass.BFloat16(qf_hi)
                    # rows 32..47 feed the (unused, must-be-zero) B_lo block of D3
                    sA3b_f[sw128_bf16(2 * T_PAD + t, kq, 4 * T_PAD)] = cutlass.BFloat16(0.0)
                    sB3b_f[sw128_bf16(t, kq, T_PAD)] = cutlass.BFloat16(kb_hi)
                else:
                    # rows t, 16+t, 32+t share t % 8 -> the same 16-byte-chunk XOR
                    xc = kc ^ (4 * (t % 8))
                    sB1f[b1_base + 32 * t + xc] = kt_hi
                    if cutlass.const_expr(S0_LO):
                        sB1kf[b1k_base + 32 * t + xc] = kt_hi
                    sB1f[b1_base + 32 * (T_PAD + t) + xc] = qt_r
                    sB1f[b1_base + 32 * (2 * T_PAD + t) + xc] = kt_lo
                    sA3f[a3_base + 32 * t + xc] = kf
                    sA3f[a3_base + 32 * (T_PAD + t) + xc] = qf_hi
                    sA3f[a3_base + 32 * (2 * T_PAD + t) + xc] = qf_lo
                    sB3hif[b3_base + 32 * t + xc] = kb_hi
                    sB3lof[b3_base + 32 * t + xc] = kb_lo
        cute.arch.fence_view_async_shared()
        tcgen05_fence_before_thread_sync()
        cute.arch.barrier()
        tcgen05_fence_after_thread_sync()
        if cutlass.const_expr(PROFILE_STAGES):
            tp_a = read_globaltimer()

        # ---- operand fragments of GEMM1 (D1 = S0_hi B1^T + S0_lo K~hi^T) and GEMM3
        #      (D3 = A3 Kb_hi^T + A3 Kb_lo^T); the MMAs are issued by warp 0 after the Kbar staging
        if cutlass.const_expr(BF16_MMA):
            tCrB1 = tiled_mma1.make_fragment_B(sB1b)
            tCrA3 = tiled_mma3.make_fragment_A(sA3b)
            tCrB3hi = tiled_mma3.make_fragment_B(sB3b)
            tA1 = tAb
        else:
            sB1_tf = cute.make_tensor(
                cute.recast_ptr(sB1.iterator, smem_layout_b1.inner, dtype=cutlass.TFloat32),
                sB1.layout,
            )
            tCrB1 = tiled_mma1.make_fragment_B(sB1_tf)
            sB1k_tf = cute.make_tensor(
                cute.recast_ptr(sB1k.iterator, smem_layout_b1k.inner, dtype=cutlass.TFloat32),
                sB1k.layout,
            )
            tCrB1k = tiled_mma1k.make_fragment_B(sB1k_tf)
            tD1k = cute.make_tensor(
                tD1.iterator,
                tiled_mma1k.make_fragment_C(tiled_mma1k.partition_shape_C((V, T_PAD))).layout,
            )
            tAhi_tf = cute.make_tensor(
                cute.recast_ptr(tAhi.iterator, dtype=cutlass.TFloat32), tAhi.layout
            )
            tAlo_tf = cute.make_tensor(
                cute.recast_ptr(tAlo.iterator, dtype=cutlass.TFloat32), tAlo.layout
            )
            sA3_tf = cute.make_tensor(
                cute.recast_ptr(sA3.iterator, smem_layout_a3.inner, dtype=cutlass.TFloat32),
                sA3.layout,
            )
            sB3hi_tf = cute.make_tensor(
                cute.recast_ptr(sB3hi.iterator, smem_layout_b3.inner, dtype=cutlass.TFloat32),
                sB3hi.layout,
            )
            sB3lo_tf = cute.make_tensor(
                cute.recast_ptr(sB3lo.iterator, smem_layout_b3.inner, dtype=cutlass.TFloat32),
                sB3lo.layout,
            )
            tCrA3 = tiled_mma3.make_fragment_A(sA3_tf)
            tCrB3hi = tiled_mma3.make_fragment_B(sB3hi_tf)
            tCrB3lo = tiled_mma3.make_fragment_B(sB3lo_tf)
            tA1 = tAhi_tf
        # ---- GEMM2 accumulator seed (D2 = S0 e^{G_c}) and the Kbar tiles first; warp 0 then
        #      issues GEMM1/GEMM3 while the other warps build the B2 tiles.
        # Kbar^T staging: thread (kq, quarter) -> sKbT[kq, s] for its 4 tokens (lanes = k: conflict-free)
        for i in cutlass.range(4, unroll_full=True):
            s_idx = quarter * 4 + i
            b_val = cutlass.Float32(0.0)
            if s_idx <= commit_len:
                b_val = sK[s_idx, kq] * cute.math.exp(
                    sE[commit_len * K + kq] - sE[s_idx * K + kq], fastmath=True
                )
            sKbT[kq * TP_STRIDE + s_idx] = b_val
        if warp_idx == 0:
            cute.arch.barrier_arrive(barrier_id=5, number_of_threads=NUM_THREADS)
            tiled_mma3.set(tcgen05.Field.ACCUMULATE, False)
            for kb_i in cutlass.range_constexpr(K // KB_MMA):
                cute.gemm(
                    tiled_mma3,
                    tD3,
                    tCrA3[(None, None, kb_i, 0)],
                    tCrB3hi[(None, None, kb_i, 0)],
                    tD3,
                )
                tiled_mma3.set(tcgen05.Field.ACCUMULATE, True)
            if cutlass.const_expr(not BF16_MMA):
                for kb_i in cutlass.range_constexpr(K // 8):
                    cute.gemm(
                        tiled_mma3,
                        tD3,
                        tCrA3[(None, None, kb_i, 0)],
                        tCrB3lo[(None, None, kb_i, 0)],
                        tD3,
                    )
            tiled_mma1.set(tcgen05.Field.ACCUMULATE, False)
            for kb_i in cutlass.range_constexpr(K // KB_MMA):
                cute.gemm(
                    tiled_mma1, tD1, tA1[(None, None, kb_i)], tCrB1[(None, None, kb_i, 0)], tD1
                )
                tiled_mma1.set(tcgen05.Field.ACCUMULATE, True)
            if cutlass.const_expr(S0_LO):
                tiled_mma1k.set(tcgen05.Field.ACCUMULATE, True)
                for kb_i in cutlass.range_constexpr(K // 8):
                    cute.gemm(
                        tiled_mma1k,
                        tD1k,
                        tAlo_tf[(None, None, kb_i)],
                        tCrB1k[(None, None, kb_i, 0)],
                        tD1k,
                    )
            with cute.arch.elect_one():
                tcgen05.commit(mbar1)
        else:
            cute.arch.barrier(barrier_id=5, number_of_threads=NUM_THREADS)
        if cutlass.const_expr(PROFILE_STAGES):
            tp_j = read_globaltimer()
        if cutlass.const_expr(PROFILE_STAGES):
            if tidx == 32:
                _tq31 = read_globaltimer()
                _, _gq31, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq31 + i_n) * 64 + 31] = _tq31 - t_stage0
        # ---- GEMM2 accumulator seed (overlaps the GEMM1/GEMM3 execution)
        # D4 (output-combination) columns start from zero: GEMM2 half 0 accumulates into [D4 | ΔS]
        st16 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
        tiled_stD4 = tcgen05.make_tmem_copy(st16, tD4)
        tD4_st = tiled_stD4.get_slice(wg_tidx).partition_D(tD4)
        r_zero = cute.make_rmem_tensor(
            tiled_stD4.get_slice(wg_tidx).partition_S(tD4).shape, cutlass.Float32
        )
        for i in cutlass.range(cute.size(r_zero), unroll_full=True):
            r_zero[i] = cutlass.Float32(0.0)
        if wg == 0:
            cute.copy(st16, r_zero, tD4_st)
        # only this CTA's state rows (TMEM lanes [vh * V/2, +V/2)) are seeded and stored
        if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
            for i in cutlass.range(32, unroll_full=True):
                kcol = wg * 32 + i
                r_hi_flat[i] = r_state[i] * sEc[kcol]
            for j in cutlass.range_constexpr(4):
                if wg == j:
                    cute.copy(st32, r_hi, tD2_st[(None, None, None, j)])
        cute.arch.fence_view_async_tmem_store()
        # ---- B2 tiles (k rows x s cols, K-major SW64) from sKbT while the MMAs run:
        #      lanes 0..15 -> row 2w, lanes 16..31 -> row 2w+1, four 32-row rounds
        s_col = in_warp_tid % T_PAD
        if warp_idx > 0:
            for r in cutlass.range(4, unroll_full=True):
                k_row = 32 * r + 2 * warp_idx + in_warp_tid // T_PAD
                b_val2 = sKbT[k_row * TP_STRIDE + s_col]
                b_hi = round_hi(b_val2, BF16_MMA)
                if cutlass.const_expr(BF16_MMA):
                    sB2b_f[sw32_k16_bf16(k_row, s_col)] = cutlass.BFloat16(b_hi)
                else:
                    o_b2 = swz64_k16(k_row, s_col)
                    sB2hif[o_b2] = b_hi
                    sB2lof[o_b2] = b_val2 - b_hi
            # warp 0 is issuing the MMAs: warps 1..4 cover its rows 32 r + {0, 1}
            if warp_idx <= 4:
                k_row0 = 32 * (warp_idx - 1) + in_warp_tid // T_PAD
                b_val0 = sKbT[k_row0 * TP_STRIDE + s_col]
                b_hi0 = round_hi(b_val0, BF16_MMA)
                if cutlass.const_expr(BF16_MMA):
                    sB2b_f[sw32_k16_bf16(k_row0, s_col)] = cutlass.BFloat16(b_hi0)
                else:
                    o_b20 = swz64_k16(k_row0, s_col)
                    sB2hif[o_b20] = b_hi0
                    sB2lof[o_b20] = b_val0 - b_hi0

        if cutlass.const_expr(PROFILE_STAGES):
            tp_k = read_globaltimer()
        # ---- wait for GEMM1/GEMM3
        cute.arch.mbarrier_wait(mbar1, 0)
        tcgen05_fence_after_thread_sync()
        if cutlass.const_expr(PROFILE_STAGES):
            tp_b = read_globaltimer()
        tiled_ld1 = tcgen05.make_tmem_copy(ld16, tD1)
        thr_ld1 = tiled_ld1.get_slice(wg_tidx)
        tD1_ld = thr_ld1.partition_S(tD1)
        rD1 = cute.make_rmem_tensor(thr_ld1.partition_D(tD1).shape, cutlass.Float32)
        cute.copy(ld16, tD1_ld, rD1)
        cute.arch.fence_view_async_tmem_load()
        rD1_flat = cute.make_tensor(rD1.iterator, cute.make_layout((N1,)))
        v_row = wg_tidx
        # warpgroup 0: D3 (64 x 16) -> smem [L; B_hi; B_lo; -]
        ld3 = cute.make_copy_atom(
            tcgen05.Ld16x128bOp(tcgen05.Repetition(T_PAD * 32 // 128)), cutlass.Float32
        )
        tiled_ld3 = tcgen05.make_tmem_copy(ld3, tD3)
        thr_ld3 = tiled_ld3.get_slice(wg_tidx)
        tD3_ld = thr_ld3.partition_S(tD3)
        sLB3 = cute.make_tensor(
            sLB.iterator,
            cute.make_layout(((4 * T_PAD, T_PAD), 1, 1), stride=((T_PAD, 1), 0, 0)),
        )
        tD3_sm = thr_ld3.partition_D(sLB3)
        rD3 = cute.make_rmem_tensor(tD3_sm.shape, cutlass.Float32)
        if wg == 0:
            cute.copy(ld3, tD3_ld, rD3)
            cute.arch.fence_view_async_tmem_load()
            cute.autovec_copy(rD3, tD3_sm)
        cute.arch.barrier()
        if cutlass.const_expr(PROFILE_STAGES):
            tp_c = read_globaltimer()
        if cutlass.const_expr(PROFILE_STAGES):
            if tidx == 0:
                _, _gq32, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq32 + i_n) * 64 + 32] = tp_c - t_stage0
        # ---- warpgroup 0: V' by forward substitution (row v per thread, four accumulators);
        #      warpgroups 1-2: output-combination operand Bcoef[t, s] = B_hi + B_lo (rows t < T_loop,
        #      s <= t; zero elsewhere), one element per thread, K-major SW64 16 x 16 tiles (hi / lo)
        #      that head the GEMM2 half-0 B tile
        if wg == 0:
            r_Y = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
            for s in cutlass.range(T_PAD, unroll_full=True):
                y = cutlass.Float32(0.0)
                if s < t_max:
                    if cutlass.const_expr(BF16_MMA):
                        u0 = rD1_flat[s]
                    else:
                        u0 = rD1_flat[s] + rD1_flat[2 * T_PAD + s]
                    y = sBeta[s] * (sVall[s * V + v_row] - u0)
                r_Y[s] = y
            r_V = cute.make_rmem_tensor(cute.make_layout((T_PAD,)), cutlass.Float32)
            for t in cutlass.range(T_PAD, unroll_full=True):
                vp = cutlass.Float32(0.0)
                if t < t_max:
                    acc0 = cutlass.Float32(0.0)
                    acc1 = cutlass.Float32(0.0)
                    acc2 = cutlass.Float32(0.0)
                    acc3 = cutlass.Float32(0.0)
                    for s in range(t):
                        if s % 4 == 0:
                            acc0 = acc0 + sLB[t, s] * r_V[s]
                        elif s % 4 == 1:
                            acc1 = acc1 + sLB[t, s] * r_V[s]
                        elif s % 4 == 2:
                            acc2 = acc2 + sLB[t, s] * r_V[s]
                        else:
                            acc3 = acc3 + sLB[t, s] * r_V[s]
                    vp = r_Y[t] - sBeta[t] * ((acc0 + acc1) + (acc2 + acc3))
                r_V[t] = vp
            for t in cutlass.range(T_PAD, unroll_full=True):
                sVpT[v_row * TP_STRIDE + t] = r_V[t]
            if cutlass.const_expr(PROFILE_STAGES):
                if tidx == 0:
                    _tq36 = read_globaltimer()
                    _, _gq36, _ = cute.arch.grid_dim()
                    stage_timing[((vh * HV + i_hv) * _gq36 + i_n) * 64 + 36] = _tq36 - t_stage0
        else:
            if tidx < 128 + T_PAD * T_PAD:
                t4 = (tidx - 128) // T_PAD
                s4 = (tidx - 128) % T_PAD
                bc = cutlass.Float32(0.0)
                if (t4 < T_loop) & (s4 <= t4):
                    bc = sLB[T_PAD + t4, s4] + sLB[2 * T_PAD + t4, s4]
                bc_hi = round_hi(bc, BF16_MMA)
                if cutlass.const_expr(BF16_MMA):
                    sB4b_f[sw32_k16_bf16(t4, s4)] = cutlass.BFloat16(bc_hi)
                else:
                    o_b4 = swz64_k16(t4, s4)
                    sB4hif[o_b4] = bc_hi
                    sB4lof[o_b4] = bc - bc_hi
        cute.arch.barrier()
        if cutlass.const_expr(PROFILE_STAGES):
            if tidx == 0:
                _tq37 = read_globaltimer()
                _, _gq37, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq37 + i_n) * 64 + 37] = _tq37 - t_stage0
        # ---- A2 tiles (v rows x s cols) from sVpT, same 16-lanes-per-row mapping; s > c -> 0
        for r in cutlass.range(4, unroll_full=True):
            v_row2 = 32 * r + 2 * warp_idx + in_warp_tid // T_PAD
            a_val = sVpT[v_row2 * TP_STRIDE + s_col]
            a_hi = round_hi(a_val, BF16_MMA)
            if cutlass.const_expr(BF16_MMA):
                sA2b_f[sw32_k16_bf16(v_row2, s_col)] = cutlass.BFloat16(a_hi)
            else:
                o_a2 = swz64_k16(v_row2, s_col)
                sA2hif[o_a2] = a_hi
                sA2lof[o_a2] = a_val - a_hi
        cute.arch.fence_view_async_shared()
        tcgen05_fence_before_thread_sync()
        cute.arch.barrier()
        tcgen05_fence_after_thread_sync()
        if cutlass.const_expr(PROFILE_STAGES):
            tp_d = read_globaltimer()
        if cutlass.const_expr(BF16_MMA):
            tCrA2hi = tiled_mma2h.make_fragment_A(sA2b)
            tCrA2hi_q = tiled_mma2q.make_fragment_A(sA2b)
            sB2b_h1 = cute.make_tensor(
                cute.recast_ptr(
                    sScratchH.iterator + OFFH_B2 + 64 * T_PAD,
                    smem_layout_b2h.inner,
                    dtype=cutlass.BFloat16,
                ),
                smem_layout_b2h.outer,
            )
            tCrB2hi_h1 = tiled_mma2h.make_fragment_B(sB2b_h1)
            sB2qb = cute.make_tensor(
                cute.recast_ptr(
                    sScratchH.iterator + OFFH_B4, smem_layout_b2q.inner, dtype=cutlass.BFloat16
                ),
                smem_layout_b2q.outer,
            )
            tCrB2q_hi = tiled_mma2q.make_fragment_B(sB2qb)
        else:
            sA2hi_tf = cute.make_tensor(
                cute.recast_ptr(sA2hi.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32),
                sA2hi.layout,
            )
            sA2lo_tf = cute.make_tensor(
                cute.recast_ptr(sA2lo.iterator, smem_layout_a2.inner, dtype=cutlass.TFloat32),
                sA2lo.layout,
            )
            tCrA2hi = tiled_mma2h.make_fragment_A(sA2hi_tf)
            tCrA2lo = tiled_mma2h.make_fragment_A(sA2lo_tf)
            # B2 halves: k rows 0..63 / 64..127 of the SW64 K-major tiles (a 4 KB shift keeps the swizzle)
            sB2hi_h1 = cute.make_tensor(
                cute.recast_ptr(
                    sScratch.iterator + OFF_B2HI + 64 * T_PAD,
                    smem_layout_b2h.inner,
                    dtype=cutlass.Float32,
                ),
                smem_layout_b2h.outer,
            )
            sB2lo_h1 = cute.make_tensor(
                cute.recast_ptr(
                    sScratch.iterator + OFF_B2LO + 64 * T_PAD,
                    smem_layout_b2h.inner,
                    dtype=cutlass.Float32,
                ),
                smem_layout_b2h.outer,
            )
            tCrB2hi_h1 = tiled_mma2h.make_fragment_B(
                cute.make_tensor(
                    cute.recast_ptr(
                        sB2hi_h1.iterator, smem_layout_b2h.inner, dtype=cutlass.TFloat32
                    ),
                    sB2hi_h1.layout,
                )
            )
            tCrB2lo_h1 = tiled_mma2h.make_fragment_B(
                cute.make_tensor(
                    cute.recast_ptr(
                        sB2lo_h1.iterator, smem_layout_b2h.inner, dtype=cutlass.TFloat32
                    ),
                    sB2lo_h1.layout,
                )
            )
            sB2q_hi = cute.make_tensor(
                cute.recast_ptr(
                    sScratch.iterator + OFF_B4HI, smem_layout_b2q.inner, dtype=cutlass.Float32
                ),
                smem_layout_b2q.outer,
            )
            sB2q_lo = cute.make_tensor(
                cute.recast_ptr(
                    sScratch.iterator + OFF_B4LO, smem_layout_b2q.inner, dtype=cutlass.Float32
                ),
                smem_layout_b2q.outer,
            )
            tCrA2hi_q = tiled_mma2q.make_fragment_A(sA2hi_tf)
            tCrA2lo_q = tiled_mma2q.make_fragment_A(sA2lo_tf)
            tCrB2q_hi = tiled_mma2q.make_fragment_B(
                cute.make_tensor(
                    cute.recast_ptr(
                        sB2q_hi.iterator, smem_layout_b2q.inner, dtype=cutlass.TFloat32
                    ),
                    sB2q_hi.layout,
                )
            )
            tCrB2q_lo = tiled_mma2q.make_fragment_B(
                cute.make_tensor(
                    cute.recast_ptr(
                        sB2q_lo.iterator, smem_layout_b2q.inner, dtype=cutlass.TFloat32
                    ),
                    sB2q_lo.layout,
                )
            )
        if warp_idx == 0:
            # GEMM2 half 0 fused with the output combination: [D4 | ΔS_0..63] += A2 x [Bcoef; Kbar_0..63]
            tiled_mma2q.set(tcgen05.Field.ACCUMULATE, True)
            for kb_i in cutlass.range_constexpr(T_PAD // KB_MMA):
                cute.gemm(
                    tiled_mma2q,
                    tD2q,
                    tCrA2hi_q[(None, None, kb_i, 0)],
                    tCrB2q_hi[(None, None, kb_i, 0)],
                    tD2q,
                )
            if cutlass.const_expr(not BF16_MMA):
                for kb_i in cutlass.range_constexpr(T_PAD // 8):
                    cute.gemm(
                        tiled_mma2q,
                        tD2q,
                        tCrA2hi_q[(None, None, kb_i, 0)],
                        tCrB2q_lo[(None, None, kb_i, 0)],
                        tD2q,
                    )
                for kb_i in cutlass.range_constexpr(T_PAD // 8):
                    cute.gemm(
                        tiled_mma2q,
                        tD2q,
                        tCrA2lo_q[(None, None, kb_i, 0)],
                        tCrB2q_hi[(None, None, kb_i, 0)],
                        tD2q,
                    )
            with cute.arch.elect_one():
                tcgen05.commit(mbar2)
            tiled_mma2h.set(tcgen05.Field.ACCUMULATE, True)
            for kb_i in cutlass.range_constexpr(T_PAD // KB_MMA):
                cute.gemm(
                    tiled_mma2h,
                    tD2h1,
                    tCrA2hi[(None, None, kb_i, 0)],
                    tCrB2hi_h1[(None, None, kb_i, 0)],
                    tD2h1,
                )
            if cutlass.const_expr(not BF16_MMA):
                for kb_i in cutlass.range_constexpr(T_PAD // 8):
                    cute.gemm(
                        tiled_mma2h,
                        tD2h1,
                        tCrA2hi[(None, None, kb_i, 0)],
                        tCrB2lo_h1[(None, None, kb_i, 0)],
                        tD2h1,
                    )
                for kb_i in cutlass.range_constexpr(T_PAD // 8):
                    cute.gemm(
                        tiled_mma2h,
                        tD2h1,
                        tCrA2lo[(None, None, kb_i, 0)],
                        tCrB2hi_h1[(None, None, kb_i, 0)],
                        tD2h1,
                    )
            with cute.arch.elect_one():
                tcgen05.commit(mbar2b)
        if cutlass.const_expr(PROFILE_STAGES):
            tp_e = read_globaltimer()

        # ---- outputs of the new tokens: o_t = O0_t + D4[v, t] (columns 0..15 of GEMM2 half 0),
        #      warpgroup w writes
        #      new tokens j = w mod 4
        cute.arch.mbarrier_wait(mbar2, 0)
        tcgen05_fence_after_thread_sync()
        if cutlass.const_expr(PROFILE_STAGES):
            if tidx == 0:
                _tq38 = read_globaltimer()
                _, _gq38, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq38 + i_n) * 64 + 38] = _tq38 - t_stage0
        tiled_ld4 = tcgen05.make_tmem_copy(ld16, tD4)
        thr_ld4 = tiled_ld4.get_slice(wg_tidx)
        rD4 = cute.make_rmem_tensor(thr_ld4.partition_D(tD4).shape, cutlass.Float32)
        cute.copy(ld16, thr_ld4.partition_S(tD4), rD4)
        cute.arch.fence_view_async_tmem_load()
        rD4_flat = cute.make_tensor(rD4.iterator, cute.make_layout((T_PAD,)))
        for j_new in cutlass.range(1 + NUM_SPEC, unroll_full=True):
            if wg == j_new % 4:
                t_new = commit_len + j_new
                o_val = cutlass.Float32(0.0)
                for tt in cutlass.range(T_PAD, unroll_full=True):
                    if tt == t_new:
                        o_val = rD1_flat[T_PAD + tt] + rD4_flat[tt]
                if cutlass.const_expr(FUSE_OUTPUT_NORM):
                    sVall[j_new * V + v_row] = cutlass.Float32(cutlass.BFloat16(o_val))
                else:
                    o[0, bos + t_new, i_hv, v_row] = cutlass.BFloat16(o_val)
        if cutlass.const_expr(PROFILE_STAGES):
            tp_f = read_globaltimer()
        # ---- committed-state store through TMA: D2 slab -> swizzled smem box (row per thread,
        #      16 B chunks land in distinct banks under the 128B swizzle) -> one bulk tensor store
        if cutlass.const_expr(SPLIT_V == 2):
            cute.arch.cluster_wait()
        if cutlass.const_expr(not FUSE_OUTPUT_NORM):
            # bf16 path has no epilogue to overlap with: every warp stores its own slab
            # (per-warp 32x32 smem transpose; the TMA store is used on the fused-norm path).
            xp_base = warp_idx * 32 * XP_STRIDE
            cute.arch.mbarrier_wait(mbar2, 0)
            cute.arch.mbarrier_wait(mbar2b, 0)
            tcgen05_fence_after_thread_sync()
            for j in cutlass.range_constexpr(4):
                if wg == j:
                    cute.copy(ld32, tD2_ld[(None, None, None, j)], rS)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.barrier()
            for i in cutlass.range(32, unroll_full=True):
                sXp[xp_base + in_warp_tid * XP_STRIDE + i] = rS_flat[i]
            cute.arch.sync_warp()
            if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
                for j in cutlass.range(32, unroll_full=True):
                    val = sXp[xp_base + j * XP_STRIDE + in_warp_tid]
                    if cutlass.const_expr(USE_FLAT_LAYOUT):
                        ht[h0_idx, (warp_idx % 4) * 32 + j, wg * 32 + in_warp_tid] = val.to(
                            STATE_DTYPE
                        )
                    else:
                        ht[slot, i_hv, (warp_idx % 4) * 32 + j, wg * 32 + in_warp_tid] = val.to(
                            STATE_DTYPE
                        )
        if cutlass.const_expr(FUSE_OUTPUT_NORM):
            # All 16 warps produced disjoint value rows above. One CTA
            # barrier makes the BF16-rounded rows visible before assigning
            # one complete token to each warp.
            if cutlass.const_expr(PROFILE_STAGES):
                if tidx == 0:
                    _tq39 = read_globaltimer()
                    _, _gq39, _ = cute.arch.grid_dim()
                    stage_timing[((vh * HV + i_hv) * _gq39 + i_n) * 64 + 39] = _tq39 - t_stage0
            cute.arch.barrier()
            if cutlass.const_expr(PROFILE_STAGES):
                tp_g = read_globaltimer()
            # ---- committed state: the state-owning warps of warpgroups 0/1 stage k columns 0..63
            #      (GEMM2 half 0, mbar2), those of warpgroups 2/3 columns 64..127 (half 1, mbar2b)
            #      into the swizzled smem boxes; the first staging warp of each pair issues the TMA
            #      stores once its half is visible (named barriers 6 / 7) and waits for the bulk
            #      groups at the very end. The warps without state rows run the norm epilogue.
            if wg < 2:
                cute.arch.mbarrier_wait(mbar2, 0)
            else:
                cute.arch.mbarrier_wait(mbar2b, 0)
            tcgen05_fence_after_thread_sync()
            if cutlass.const_expr(PROFILE_STAGES):
                tp_l = read_globaltimer()
            for sl in cutlass.range_constexpr(4):
                if wg == sl:
                    if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
                        cute.copy(ld32, tD2_ld[(None, None, None, sl)], rS)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(STATE_BF16):
                            # bf16 box: chunks [4 (sl % 2), +4) of box sl // 2, round to nearest
                            for c in cutlass.range(4, unroll_full=True):
                                chunk = (4 * (sl % 2) + c) ^ (wg_tidx % 8)
                                off_b = (
                                    (sl // 2) * ((V // SPLIT_V) * 64)
                                    + 64 * (wg_tidx - (V // SPLIT_V) * vh)
                                    + 8 * chunk
                                )
                                for i in cutlass.range(8, unroll_full=True):
                                    sBoxh[off_b + i] = cutlass.BFloat16(rS_flat[8 * c + i])
                        else:
                            for c in cutlass.range(8, unroll_full=True):
                                off_b = (
                                    sl * ((V // SPLIT_V) * 32)
                                    + 32 * (wg_tidx - (V // SPLIT_V) * vh)
                                    + ((4 * c) ^ vx)
                                )
                                for i in cutlass.range(4, unroll_full=True):
                                    sBoxf[off_b + i] = rS_flat[4 * c + i]
            # last TMEM reads of the kernel: warps 1..15 signal barrier 8, warp 0 frees TMEM below
            tcgen05_fence_before_thread_sync()
            if warp_idx > 0:
                cute.arch.barrier_arrive(barrier_id=8, number_of_threads=NUM_THREADS)
            if cutlass.const_expr(PROFILE_STAGES):
                if tidx == 0:
                    _tq23 = read_globaltimer()
                    _, _gq23, _ = cute.arch.grid_dim()
                    stage_timing[((vh * HV + i_hv) * _gq23 + i_n) * 64 + 23] = _tq23 - t_stage0
            if cutlass.const_expr(PROFILE_STAGES):
                if tidx == 256:
                    _tq33 = read_globaltimer()
                    _, _gq33, _ = cute.arch.grid_dim()
                    stage_timing[((vh * HV + i_hv) * _gq33 + i_n) * 64 + 33] = _tq33 - t_stage0
            cute.arch.fence_view_async_shared()
            if cutlass.const_expr(SPLIT_V == 1) or ((warp_idx % 4) // 2 == vh):
                if wg < 2:
                    cute.arch.barrier(barrier_id=6, number_of_threads=256 // SPLIT_V)
                    if cutlass.const_expr(PROFILE_STAGES):
                        if tidx == 0:
                            _tq24 = read_globaltimer()
                            _, _gq24, _ = cute.arch.grid_dim()
                            stage_timing[((vh * HV + i_hv) * _gq24 + i_n) * 64 + 24] = (
                                _tq24 - t_stage0
                            )
                    if tidx == (V // 2) * vh:
                        for s in cutlass.range_constexpr(NUM_SLABS // 2):
                            cute.copy(tma_atom_t, tS_list[s], tG_list[s])
                        cute.arch.cp_async_bulk_commit_group()
                    if cutlass.const_expr(PROFILE_STAGES):
                        if tidx == 0:
                            _tq25 = read_globaltimer()
                            _, _gq25, _ = cute.arch.grid_dim()
                            stage_timing[((vh * HV + i_hv) * _gq25 + i_n) * 64 + 25] = (
                                _tq25 - t_stage0
                            )
                else:
                    cute.arch.barrier(barrier_id=7, number_of_threads=256 // SPLIT_V)
                    if tidx == 256 + (V // 2) * vh:
                        for s in cutlass.range_constexpr(NUM_SLABS // 2, NUM_SLABS):
                            cute.copy(tma_atom_t, tS_list[s], tG_list[s])
                        cute.arch.cp_async_bulk_commit_group()
                    if cutlass.const_expr(PROFILE_STAGES):
                        if tidx == 256:
                            _tq34 = read_globaltimer()
                            _, _gq34, _ = cute.arch.grid_dim()
                            stage_timing[((vh * HV + i_hv) * _gq34 + i_n) * 64 + 34] = (
                                _tq34 - t_stage0
                            )
            if warp_idx == 0:
                # all TMEM reads are done (barrier 8): free the columns while the store drains
                cute.arch.barrier(barrier_id=8, number_of_threads=NUM_THREADS)
                tcgen05_fence_after_thread_sync()
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS)
            if cutlass.const_expr(PROFILE_STAGES):
                if tidx == 64:
                    _tq35 = read_globaltimer()
                    _, _gq35, _ = cute.arch.grid_dim()
                    stage_timing[((vh * HV + i_hv) * _gq35 + i_n) * 64 + 35] = _tq35 - t_stage0
            # norm epilogue (CTA 0) on the warps that own no state rows: (warp % 4) in {2, 3}
            token_idx = (warp_idx // 4) * 2 + (warp_idx % 4 - 2)
            if (vh == 0) & ((warp_idx % 4) // 2 == 1):
                while token_idx < 1 + NUM_SPEC:
                    # gated values p_v = o_v * w_v * sigmoid(g_v); the two warp reductions (sum of
                    # squares, max |p_v|) are interleaved. bf16 rounding is monotonic, so
                    # amax(bf16(p_v * inv_rms)) = bf16(max|p_v| * inv_rms).
                    pre_values = cute.make_rmem_tensor(
                        cute.make_layout((V // 32,), stride=(1,)), cutlass.Float32
                    )
                    rms_partial = cutlass.Float32(0.0)
                    pre_max = cutlass.Float32(0.0)
                    for i in cutlass.range(V // 32, unroll_full=True):
                        v_idx = i * 32 + in_warp_tid
                        value = sVall[token_idx * V + v_idx]
                        gate = sOnG[token_idx * V + v_idx]
                        sigmoid_gate = cute.arch.rcp_approx(
                            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
                        )
                        p_v = value * sOnW[v_idx] * sigmoid_gate
                        pre_values[i] = p_v
                        rms_partial += value * value
                        pre_max = cute.arch.fmax(pre_max, cute.arch.fmax(p_v, -p_v))
                    for offset in [16, 8, 4, 2, 1]:
                        rms_partial += cute.arch.shuffle_sync_bfly(
                            rms_partial, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        pre_max = cute.arch.fmax(
                            pre_max,
                            cute.arch.shuffle_sync_bfly(
                                pre_max, offset=offset, mask=-1, mask_and_clamp=31
                            ),
                        )
                    inv_rms = cute.math.rsqrt(
                        rms_partial / cutlass.Float32(V) + cutlass.Float32(onorm_eps),
                        fastmath=True,
                    )
                    if cutlass.const_expr(PROFILE_STAGES):
                        if tidx == 64:
                            _tq19 = read_globaltimer()
                            _, _gq19, _ = cute.arch.grid_dim()
                            stage_timing[((vh * HV + i_hv) * _gq19 + i_n) * 64 + 19] = (
                                _tq19 - t_stage0
                            )
                    output_row = i_n * (1 + NUM_SPEC) + token_idx
                    normalized_values = cute.make_rmem_tensor(
                        cute.make_layout((V // 32,), stride=(1,)), cutlass.Float32
                    )
                    for i in cutlass.range(V // 32, unroll_full=True):
                        normalized = pre_values[i] * inv_rms
                        if cutlass.const_expr(QUANTIZE_OUTPUT):
                            # Preserve the old standalone RMSNorm's BF16 output
                            # boundary in registers. The value goes straight to
                            # MXFP8 below; it is never written to or reloaded from
                            # HBM as BF16.
                            normalized = cutlass.Float32(cutlass.BFloat16(normalized))
                        normalized_values[i] = normalized
                    if cutlass.const_expr(QUANTIZE_OUTPUT):
                        output_amax = cutlass.Float32(cutlass.BFloat16(pre_max * inv_rms))
                        output_amax = cute.arch.fmax(output_amax, cutlass.Float32(1.0e-10))
                        output_sf = (output_amax * cutlass.Float32(1.0 / 448.0)).to(
                            cutlass.Float8E8M0FNU
                        )
                        quant_scale = cute.arch.rcp_approx(cutlass.Float32(output_sf))
                        for i in cutlass.range(V // 32, unroll_full=True):
                            v_idx = i * 32 + in_warp_tid
                            o[output_row, i_hv * V + v_idx] = cutlass.Float8E4M3FN(
                                normalized_values[i] * quant_scale
                            )
                        if in_warp_tid == 0:
                            scale_bytes = cute.make_tensor(
                                cute.recast_ptr(
                                    output_scale.iterator,
                                    dtype=cutlass.Float8E8M0FNU,
                                ),
                                cute.make_layout(output_scale.shape),
                            )
                            scale_offset = (
                                (output_row // 128) * HV * 512
                                + i_hv * 512
                                + (output_row % 32) * 16
                                + ((output_row % 128) // 32) * 4
                            )
                            for replica in cutlass.range_constexpr(4):
                                scale_bytes[scale_offset + replica] = output_sf
                            if cutlass.const_expr(PROFILE_STAGES):
                                if tidx == 64:
                                    _tq20 = read_globaltimer()
                                    _, _gq20, _ = cute.arch.grid_dim()
                                    stage_timing[((vh * HV + i_hv) * _gq20 + i_n) * 64 + 20] = (
                                        _tq20 - t_stage0
                                    )
                    else:
                        for i in cutlass.range(V // 32, unroll_full=True):
                            v_idx = i * 32 + in_warp_tid
                            o[0, output_row, i_hv, v_idx] = cutlass.BFloat16(normalized_values[i])
                    token_idx = token_idx + NUM_THREADS // 32
            if (tidx == (V // 2) * vh) | (tidx == 256 + (V // 2) * vh):
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            if cutlass.const_expr(PROFILE_STAGES):
                tp_m = read_globaltimer()
                if tidx == 0:
                    _, grid_n2, _ = cute.arch.grid_dim()
                    tb2 = ((vh * HV + i_hv) * grid_n2 + i_n) * 64
                    stage_timing[tb2 + 15] = tp_l - t_stage0
                    stage_timing[tb2 + 16] = tp_m - t_stage0
                if tidx == 256:
                    _, _gq26, _ = cute.arch.grid_dim()
                    stage_timing[((vh * HV + i_hv) * _gq26 + i_n) * 64 + 26] = tp_l - t_stage0
        if cutlass.const_expr(PROFILE_STAGES):
            tp_h = read_globaltimer()
            tp_i = tp_h
            if tidx == 32:
                _, _gq21, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq21 + i_n) * 64 + 21] = tp_h - t_stage0
            if tidx == 224:
                _, _gq22, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq22 + i_n) * 64 + 22] = tp_h - t_stage0
        if cutlass.const_expr(PROFILE_STAGES):
            cute.arch.barrier()
            t_stage2 = read_globaltimer()
            if tidx == 256:
                _, _gq27, _ = cute.arch.grid_dim()
                stage_timing[((vh * HV + i_hv) * _gq27 + i_n) * 64 + 27] = t_stage2 - t_stage0
            if tidx == 0:
                _, grid_n, _ = cute.arch.grid_dim()
                timing_base = ((vh * HV + i_hv) * grid_n + i_n) * 64
                stage_timing[timing_base + 0] = t_stage1 - t_stage0
                stage_timing[timing_base + 1] = t_stage2 - t_stage1
                stage_timing[timing_base + 2] = t_stage2 - t_stage0
                stage_timing[timing_base + 3] = t_stage0
                stage_timing[timing_base + 4] = tp_a - t_stage0
                stage_timing[timing_base + 5] = tp_b - t_stage0
                stage_timing[timing_base + 6] = tp_c - t_stage0
                stage_timing[timing_base + 7] = tp_d - t_stage0
                stage_timing[timing_base + 8] = tp_e - t_stage0
                stage_timing[timing_base + 9] = tp_f - t_stage0
                stage_timing[timing_base + 10] = tp_g - t_stage0
                stage_timing[timing_base + 11] = tp_h - t_stage0
                stage_timing[timing_base + 12] = tp_i - t_stage0
                stage_timing[timing_base + 13] = tp_j - t_stage0
                stage_timing[timing_base + 14] = tp_k - t_stage0
    tcgen05_fence_before_thread_sync()
    cute.arch.barrier()
    if cutlass.const_expr(not FUSE_OUTPUT_NORM):
        # (the fused-norm path released TMEM right after the state staging)
        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_ptr_d = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_holder)
            cute.arch.dealloc_tmem(tmem_ptr_d, TMEM_COLS)
    if cutlass.const_expr(PROFILE_STAGES):
        if tidx == 0:
            t_exit = read_globaltimer()
            _, grid_n3, _ = cute.arch.grid_dim()
            tb3 = ((vh * HV + i_hv) * grid_n3 + i_n) * 64
            stage_timing[tb3 + 17] = t_exit - t_stage0
