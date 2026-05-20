/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_8BIT_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_8BIT_CUH_

// 8-bit (int8, future e4m3) kernel path for the incremental SSU kernel:
// storage, replay, encode, output, and __global__ kernel.

#include "kernel_checkpointing_ssu_common.cuh"

namespace flashinfer::mamba::checkpointing {

// =============================================================================
// 8-bit chain-rewrite storage (sibling of CheckpointingSsuStorage)
// =============================================================================
// Used by `checkpointing_ssu_kernel_8bit` for int8 and fp8 (e4m3) state.
// Differs from the generic `CheckpointingSsuStorage<input_t, int8_t, ...>`:
//   1. No `new_state` staging buffer — matmul-3 chains state's fp32 C-frag
//      directly into the next mma's A-operand in registers (à la
//      `convert_layout_acc_Aregs`), so no smem round-trip is needed.
//   2. Adds an `output_transpose` buffer used to flip the (D, T) output frag
//      back to (T, D) before the gmem STG.  Matmul-3/4 in the chain path
//      compute init_out^T[D, T] (M=D), so the per-warp M-shard frags must be
//      transposed via smem before storing into the (T, D) gmem layout.
//
// All shared Phase 0/1 buffers (CB_scaled, B, C, x, z, old_x, old_B, scalars,
// state) are byte-for-byte identical to the generic struct — Phase 0/1
// helpers (`compute_CB_scaled_2warp`, B/C/x/z loaders, etc.) are templated on
// `SmemT` and read these by name, so they work unchanged.
template <typename input_t, typename state_t_, int NPREDICTED_, int MAX_WINDOW_, int D_PER_CTA,
          int DSTATE>
struct CheckpointingSsuStorage8bit {
  using state_t = state_t_;
  static_assert(sizeof(state_t) == 1, "CheckpointingSsuStorage8bit requires a 1-byte state_t");

  static constexpr int NPREDICTED = NPREDICTED_;
  static constexpr int MAX_WINDOW = MAX_WINDOW_;
  static constexpr int D_SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(D_PER_CTA);
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);
  static constexpr int CB_ROW_STRIDE = SmemSwizzle<input_t>::ATOM_COLS;

  // Shared Phase 0/1 buffers (same shape/swizzle as `CheckpointingSsuStorage`).
  alignas(16) input_t CB_scaled[NPREDICTED_PAD_MMA_M * CB_ROW_STRIDE];
  alignas(16) input_t B[NPREDICTED_PAD_MMA_N * DSTATE];
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];
  alignas(16) input_t x[NPREDICTED_PAD_MMA_M * D_SMEM_COLS];
  alignas(16) input_t z[NPREDICTED_SWIZZLE_R * D_SMEM_COLS];
  alignas(16) input_t old_x[MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS];
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];

  float old_dt[MAX_WINDOW];
  float old_cumAdt[MAX_WINDOW];
  float dt_proc[NPREDICTED];
  float cumAdt[NPREDICTED];
  // decay[t] = exp(cumAdt[t]) — precomputed at Phase 0 alongside cumAdt so the
  // output decay broadcast in `compute_output_8bit` is a plain LDS instead of a
  // per-element __expf.  Each of the 4 warps redundantly writes the same values
  // (same pattern as cumAdt); cross-warp visibility comes via the kernel's
  // existing __syncthreads before compute_output_8bit.
  float decay[NPREDICTED];

  // state — int8 input, only LDS'd in the single replay pass.  After replay
  // completes its dequant + matmul into the C-frag, smem.state is dead — so
  // `output_transpose` could in principle alias it (8 KB int8 vs 2 KB bf16
  // overlap easily), but for clarity we keep them separate; alias is a
  // Phase-4 micro-optimization.
  alignas(16) state_t state[D_PER_CTA * DSTATE];

  // output_transpose — physical (NPREDICTED_PAD_MMA_M, OUTPUT_TRANSPOSE_ROW_STRIDE)
  // input_t scratch buffer with PADDED row stride for bank-conflict-free per-thread
  // STS + 16-byte-aligned cooperative LDS.128.  Used by `compute_output_int8` to
  // flip the per-warp `frag_y_DxT[D, T]` register layout into `(T, D)` gmem order.
  //
  // Row stride: D_PER_CTA + 8 = 72 bf16 elts = 144 bytes.  The 8-elt (16-byte) pad
  // gives:
  //   - 144 % 16 == 0 → LDS.128 / STG.128 stays 16-byte aligned across all rows.
  //   - 144 / 4 % 32 == 4 → adjacent t-rows shift bank assignment by 4 banks.
  // For the m16n8 partition_C STS pattern (per-elt: 4 lanes write at fixed d,
  // t ∈ {0, 2, 4, 6} → banks {0, 4, 8, 12} on the padded layout — all distinct,
  // no conflicts), the padded layout cuts STS bank conflicts from ~63% of
  // wavefronts (NCU v16.0) down to 0%.
  // Volume: 16 × 72 × 2 B = 2.25 KB (vs unswizzled 2 KB; +256 B).
  static constexpr int OUTPUT_TRANSPOSE_ROW_STRIDE = D_PER_CTA + 8;
  alignas(16) input_t output_transpose[NPREDICTED_PAD_MMA_M * OUTPUT_TRANSPOSE_ROW_STRIDE];
};

// =============================================================================
// State-dtype dispatch helpers
// =============================================================================
// `state_t` is one of: `int8_t` (symmetric int8, ±127) or `__nv_fp8_e4m3` (fp8
// e4m3, ±448).  Both are 1-byte storage; the kernel's smem layout is identical.
// Differences live in: (a) the RN encode primitive, (b) the QUANT_MAX clip
// bound, and (c) packing/unpacking the byte from a u16 `Pair`.

template <typename state_t>
__device__ __forceinline__ uint8_t state_byte_of(state_t v) {
  if constexpr (std::is_same_v<state_t, int8_t>) {
    return static_cast<uint8_t>(static_cast<int8_t>(v));
  } else {
    static_assert(std::is_same_v<state_t, __nv_fp8_e4m3>,
                  "8-bit state_t must be int8_t or __nv_fp8_e4m3");
    return reinterpret_cast<__nv_fp8_storage_t const&>(v);
  }
}

// fp32 → state_t with RN + saturate.  Single-element scalar — the kernel's
// smem layout writes pairs as u16, so per-element conversion fits the
// per-thread fragment topology directly.
template <typename state_t>
__device__ __forceinline__ state_t encode_rn_8bit(float x) {
  if constexpr (std::is_same_v<state_t, int8_t>) {
    return conversion::cvt_rni_sat_s8(x);
  } else {
    static_assert(std::is_same_v<state_t, __nv_fp8_e4m3>,
                  "8-bit state_t must be int8_t or __nv_fp8_e4m3");
    // cuda_fp8 ctor compiles to `cvt.rn.satfinite.e4m3.f32` on sm_89+.
    return __nv_fp8_e4m3(x);
  }
}

// Per-state-dtype symmetric clip / encode-scale denominator.
//   int8:           ±127 (matches Triton reference, leaves -128 unused)
//   fp8_e4m3fn:     ±448 (max finite e4m3 value)
template <typename state_t>
__device__ __forceinline__ constexpr float quant_max_8bit() {
  if constexpr (std::is_same_v<state_t, int8_t>) {
    return 127.0f;
  } else {
    static_assert(std::is_same_v<state_t, __nv_fp8_e4m3>,
                  "8-bit state_t must be int8_t or __nv_fp8_e4m3");
    return 448.0f;
  }
}

// SM80 m16n8k16 C-frag → A-frag layout reshape for chained mma (state →
// matmul-3 in the int8 chain rewrite).
//
// Pattern mirrors the SM90 helper at attention/hopper/utils.cuh:103, but:
//   - SM80 m16n8 C-frag inner per-thread layout is rank-2 ((col_pair=2,
//     row_pair=2)) — there's no inner "N/8" stride mode like SM90.
//   - We instead `logical_divide` the *outer* MMA_N axis by 2: each pair of
//     m16n8 N-atoms (= 16 cols of the producing mma's N) becomes one K=16
//     atom of the chained m16n8k16 mma's A operand.
//
// Lane-element mapping (verified by hand on the m16n8k16 PTX layout):
//   C-frag at (cp, rp, mma_n=2k+kh) maps to:    row=tid/4+rp*8,  col=4*(tid%2)+cp+(2k+kh)*8
//   A-frag at (cp, rp, kh, mma_k=k)   maps to:    row=tid/4+rp*8,  col=4*(tid%2)+cp+8*kh + 16k
//   Same element: (2k+kh)*8 + cp == 8*kh + cp + 16k. ✓
//
// Input layout:  ((2, 2),    MMA_M, MMA_N)            — m16n8 C-frag
// Output layout: ((2, 2, 2), MMA_M, MMA_N / 2)        — m16n8k16 A-frag,
//                                                       MMA_K = MMA_N / 2
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs_sm80(Layout acc_layout) {
  using namespace cute;
  using X = Underscore;
  static_assert(decltype(size<0, 0>(acc_layout))::value == 2,
                "C-frag inner mode must be (col_pair=2, row_pair=2)");
  static_assert(decltype(size<0, 1>(acc_layout))::value == 2,
                "C-frag inner mode must be (col_pair=2, row_pair=2)");
  static_assert(decltype(rank(acc_layout))::value == 3,
                "C-frag must be rank-3 ((C0,C1), MMA_M, MMA_N)");
  static_assert(decltype(rank(get<0>(acc_layout)))::value == 2,
                "SM80 m16n8 C-frag inner is rank-2 (no inner stride mode like SM90)");
  // logical_divide the outer MMA_N axis by 2 → ((2, 2), MMA_M, (2, MMA_N/2))
  auto l = logical_divide(acc_layout, Shape<X, X, _2>{});
  return make_layout(
      make_layout(get<0, 0>(l), get<0, 1>(l), get<2, 0>(l)),  // ((col_pair, row_pair, k_half))
      get<1>(l),                                              // MMA_M
      get<2, 1>(l));                                          // MMA_K = MMA_N / 2
}

// =============================================================================
// Phase 1b: Replay for QUANTIZED state (int8) with RN encoding.
// =============================================================================
// state[D, dstate] = dequant(state_q, decode_scale) * total_decay
//                   + old_x^T @ (coeff * old_B)
//
// Layout: per-warp M-shard via TiledMma `Layout<_4, _1>`.  Each warp owns
// D_PER_CTA / 4 D-rows × full DSTATE.  This makes amax-over-dstate fully
// warp-local (no atomic, no cross-warp __syncthreads), at the cost of
// loading full B (old_B) from smem in every warp (vs partitioning N
// across warps in the bf16/fp16 path).  Constraint: per-warp M must equal
// the m16n8 atom M (=16), so D_PER_CTA must be 64 — the wrapper enforces
// d_split == 1 for int8.
//
// Pipeline:
//   1. Replay n-loop: m16n8 matmul, write fp32 frag → smem.new_state.
//   2. STG redistribution + amax + encode pass (warp-local):
//        Each warp covers M_PER_WARP = 16 D-rows × 128 cols of new_state.
//        Re-tile 32 lanes as 4 row-groups × 8 col-segments.  Per round
//        r ∈ [0, 4): each lane reads 16 fp32 (4× LDS.128) for one D-row,
//        computes a lane-amax (16 fmaxf), `__shfl_xor` over the 8
//        col-lanes (mask 1, 2, 4) for the full-row amax, encodes 16 int8,
//        and STG.128's them to gmem.  One writer per row stores
//        decode_scale = amax/QUANT_MAX to params.state_scale.
//
// matmul-3 reads new_state (fp32) on the same M-shard partition, so no
// cross-warp visibility is needed.  The `__syncthreads` after this
// function returns is for dt_proc / cumAdt visibility (Phase 2), not for
// state.

// ─────────────────────────────────────────────────────────────────────────
// replay_state_mma_8bit_chain: int8-state chain rewrite — PASS 1 only.
//
// Drops bf16 new_state smem buffer entirely; matmul-3 is fused inline with
// replay HMMA on a per-K-pair cadence (1 K-atom of A in flight at a time).
//
// Pipeline (single fused loop over K-pairs):
//   - For kpair ∈ [0, NUM_K_PAIRS=8):
//     - Replay 2 m16n8 N-atoms → fp32 frag_h × 2 (16 dstate cols of state).
//     - Update per-thread amax (fp32, bit-exact).
//     - Cast fp32 → bf16, pack into K-atom-sized A frag (`a_kpair`,
//       8 bf16/thread = 4 32-bit regs).  Layout matches the m16n8k16 A
//       operand directly (`partition_fragment_A` of the chain TiledMma).
//     - LDS one K-atom of B from `smem.C[T_pad, kpair*16..+16]`.
//     - `cute::gemm` accumulates one K-atom into `frag_y_DxT`:
//          `frag_y_DxT[D, T] += new_state[D, kpair*16..+16]
//                              @ smem.C[T_pad, kpair*16..+16]^T`.
//     - Both `a_kpair` and `b_kpair` go out of scope at iter end.
// Post-loop:
//   - Warp-local amax reduce (`__shfl_xor` over 4 col-lanes per row pair).
//   - Compute `decode_scale = amax/127`, `encode_scale = 127/amax` per row.
//   - STG `decode_scale` to gmem (one writer per (cache, head, d_row)).
//   - Return `encode_scale_per_row[2]` to the caller — needed by the
//     PASS 2 helper (`encode_state_replay_8bit`) which runs *after*
//     `compute_output_8bit` so that `frag_y_DxT`'s 8 fp32 regs are dead by
//     the time PASS 2's replay-again runs.
//
// Math identity for the chain (why writing `frag_h(j)` to `a_kpair(local_n*4+j)`
// places the bytes in the m16n8k16 A operand's expected position):
//   linear(cp, rp, kh, _, mma_k) = cp + 2*rp + 4*kh + 8*mma_k
//                                = cp + 2*rp + 4*(mma_n%2) + 8*(mma_n/2)
//                                = cp + 2*rp + 4*mma_n
// = same linear index as the C-frag for the 2 m16n8 N-atoms making up this
// K-pair.  No layout helper needed.
//
// No internal __syncthreads — smem.C is redundantly loaded by all 4 warps
// so chain matmul-3 sees each warp's own data without cross-warp sync.
// The caller's single __syncthreads between all replay passes and
// compute_output_8bit provides smem.CB_scaled / smem.x / smem.z visibility.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, typename SmemT,
          typename FragYDxT>
__device__ __forceinline__ void replay_state_mma_8bit_chain(
    SmemT& smem, CheckpointingSsuParams const& params, int warp, int lane, int prev_k, int d_tile,
    int64_t cache_slot, int head, bool must_checkpoint, FragYDxT& frag_y_DxT,
    float (&encode_scale_per_row_out)[2], float (&total_scale_out)[2]) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "replay_state_mma_8bit_chain requires 2-byte input_t");
  static_assert(sizeof(state_t) == 1,
                "replay_state_mma_8bit_chain is for 1-byte state_t (int8/fp8) only");
  static_assert(D_PER_CTA == 64,
                "replay_state_mma_8bit_chain requires D_PER_CTA == 64 (M-shard, per-warp M=16).");

  constexpr int NUM_WARPS = 4;
  constexpr int M_PER_WARP = D_PER_CTA / NUM_WARPS;  // 16
  static_assert(M_PER_WARP == MMA_prop::M, "Per-warp M must equal m16n8 atom M (=16)");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  int const tid = warp * warpSize + lane;

  // Atom-K dispatch: K_BIG=16 (default), K_SMALL=8 if MAX_WINDOW ≤ 8.
  using MmaAtomReplayType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                               MMA_prop::AtomK16, MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  // Replay TiledMma: M-shard, 4 warps along M, 1 along N.  Output is
  // ((2,2), 1, NUM_N_PASSES) per thread of fp32 (or bf16 view for new_state).
  auto tiled_mma_replay =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomReplayType>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_replay = tiled_mma_replay.get_slice(tid);

  // Chain TiledMma: m16n8k16 (always K_BIG=16 since K=DSTATE/16 atoms ≥ 1),
  // same M-shard layout as replay.  M_per_warp=16 (1 m-atom),
  // N=NPREDICTED_PAD_MMA_M (T_pad, ≤ 16 = up to 2 n-atoms per warp), K=DSTATE.
  auto tiled_mma_chain =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_chain = tiled_mma_chain.get_slice(tid);

  constexpr int N_PER_PASS = MMA_prop::N;            // 8
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;  // 16
  constexpr int FRAG_SIZE = 4;
  constexpr int D_ROWS_PER_THREAD = 2;
  constexpr float QUANT_MAX = quant_max_8bit<state_t>();

  float const total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float const total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  int const lane_d = lane / 4;
  int const warp_d_base = warp * M_PER_WARP;

  // ── Per-row decode_scale for state init.
  auto const* __restrict__ state_scale_ptr = reinterpret_cast<float const*>(params.state_scale);
  int64_t const state_scale_base = cache_slot * params.state_scale_stride_seq +
                                   (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;
  float decode_scale_in[D_ROWS_PER_THREAD];
  decode_scale_in[0] = state_scale_ptr[state_scale_base + warp_d_base + lane_d];
  decode_scale_in[1] = state_scale_ptr[state_scale_base + warp_d_base + lane_d + 8];
  float total_scale[D_ROWS_PER_THREAD];
  total_scale[0] = decode_scale_in[0] * total_decay;
  total_scale[1] = decode_scale_in[1] * total_decay;

  // ── A operand (replay): old_x [MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS] → LDSM_T.
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A_replay = thr_mma_replay.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_replay_view = s2r_thr_A.retile_D(frag_A_replay);
  cute::copy(s2r_A, smem_A_s2r, frag_A_replay_view);

  // ── Bake dB coefficients into frag_A once (8 scale ops), replacing 16×
  // per-N-pass compute_dB_scaling on frag_B (64 scale ops).
  // dB coefficients c[k] baked into frag_A once, replacing per-N-pass B scaling.
  apply_dA_coeff<MAX_WINDOW_PAD_MMA_K>(frag_A_replay, smem, total_cumAdt, prev_k, lane);

  // ── B operand (replay): old_B per-pass.
  auto layout_B_replay = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_B)), layout_B_replay);
  auto s2r_B_replay = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_B_replay = s2r_B_replay.get_slice(tid);

  // ── State: 1-byte input pointer + manual swizzle offsets (read in BOTH passes).
  // Drop bf16 new_state staging — replay's fp32 frag flows directly into the
  // register-resident `new_state` tensor below.
  state_t* state_base = reinterpret_cast<state_t*>(smem.state);

  // Manual swizzle offsets for m16n8 C-fragment layout (1-byte Swizzle<3,4,3>).
  // off = row * 128 + (col ^ ((row & 7) << 4)).
  // row_hi = row_lo + 8;  (row+8)&7 == row&7  ⇒  off_hi = off_lo + 1024.
  // Fragment col within each N_PER_PASS=8 tile: (lane % 4) * 2.
  int const row_lo = warp_d_base + lane_d;
  int const frag_col_base = (lane & 3) << 1;
  int const state_base_lo = row_lo << 7;  // row_lo * DSTATE
  int const state_xor = (row_lo & 7) << 4;

  float per_thread_amax[D_ROWS_PER_THREAD] = {0.f, 0.f};

  // No __syncthreads here — smem.C is redundantly loaded by all 4 warps
  // (each warp sees its own cp.async via __syncwarp in load_data).  Cross-warp
  // visibility for smem.CB_scaled / smem.x / smem.z is established by the
  // caller's __syncthreads between this function and compute_output_8bit.

  // ── smem.C view + B-operand TiledCopy for chain matmul-3 (hoisted before
  // the loop; same view per K-pair, B sliced per K-atom inside the loop).
  auto layout_C_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.C)),
                              layout_C_swz);
  auto s2r_B_chain =
      make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, MMA_prop::operand_t>{}, tiled_mma_chain);
  auto s2r_thr_B_chain = s2r_B_chain.get_slice(tid);

  constexpr int NUM_K_PAIRS = NUM_N_PASSES / 2;  // 8 for DSTATE=128
  static_assert(NUM_N_PASSES % 2 == 0, "Per-K-pair fusion requires NUM_N_PASSES to be even");
  static_assert(MMA_prop::K_BIG == 16, "Chain mma assumes m16n8k16 K-atom = 16");

  // ════════════════════════════════════════════════════════════════════════
  // PASS 1 — fused replay + chain matmul-3 (per-K-pair):
  //   For each kpair ∈ [0, NUM_K_PAIRS):
  //     - Run 2 replay HMMAs (N-passes 2*kpair, 2*kpair+1) → fp32 frag_h × 2.
  //     - Update per-thread amax (bit-exact fp32).
  //     - Pack each pair's 4 fp32 → 4 bf16 into a tiny K-atom-sized A frag
  //       (`a_kpair` shape ((2,2,2), 1, 1) of bf16 = 8 elts/thread = 4
  //       32-bit regs).  Linear positions [local_n*4 .. local_n*4+3]
  //       within `a_kpair` map to the m16n8k16 A operand's (kh=local_n)
  //       slice — proven by the linear-index identity in the deleted
  //       `new_state`-tensor comment above.
  //     - LDS one K-atom of B (smem.C[T_pad, kpair*16..+16]) into a
  //       similarly small `b_kpair` frag (4 32-bit regs / thread).
  //     - `cute::gemm` accumulates one K-atom into `frag_y_DxT`.
  //     - Both `a_kpair` and `b_kpair` go out of scope at iter end → the
  //       compiler frees those ~8 32-bit regs/thread for the next iter.
  // Net: register footprint drops from the 32 regs of the old register-
  // resident `new_state` array (held across the whole loop) to ~8 regs in
  // flight.  Frees ~24 regs/thread → potentially +1-2 blocks/SM occupancy.
  // ════════════════════════════════════════════════════════════════════════
#pragma unroll
  for (int kpair = 0; kpair < NUM_K_PAIRS; ++kpair) {
    // K-atom-sized A frag for chain matmul-3 (filled across the 2 N-passes).
    Tensor a_kpair = thr_mma_chain.partition_fragment_A(make_tensor(
        (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MMA_prop::K_BIG>{})));
    static_assert(decltype(size(a_kpair))::value == 8,
                  "a_kpair must hold 1 m16n8k16 K-atom of A = 8 bf16/thread");

#pragma unroll
    for (int local_n = 0; local_n < 2; ++local_n) {
      int const n = kpair * 2 + local_n;
      int const n_base = n * N_PER_PASS;

      Tensor frag_h = thr_mma_replay.partition_fragment_C(
          make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));
      static_assert(decltype(size(frag_h))::value == FRAG_SIZE,
                    "FRAG_SIZE must match the partitioned C-fragment size");

      // Zero-init accumulator — MMA from scratch, state added after.
      clear(frag_h);

      // Replay B operand load.
      Tensor smem_B_n =
          local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                     make_coord(n, _0{}));
      auto smem_B_s2r_n = s2r_thr_B_replay.partition_S(smem_B_n);
      Tensor frag_B_replay = thr_mma_replay.partition_fragment_B(make_tensor(
          (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      auto frag_B_replay_view = s2r_thr_B_replay.retile_D(frag_B_replay);
      cute::copy(s2r_B_replay, smem_B_s2r_n, frag_B_replay_view);

      // Replay HMMA: frag_h = frag_A_scaled @ frag_B (c[k] baked into A).
      cute::gemm(tiled_mma_replay, frag_h, frag_A_replay, frag_B_replay, frag_h);

      {
        int const off_lo = state_base_lo + ((frag_col_base + n_base) ^ state_xor);
        Pair<state_t> const p0 = *reinterpret_cast<Pair<state_t> const*>(&state_base[off_lo]);
        frag_h(0) += toFloat(p0[Int<0>{}]) * total_scale[0];
        frag_h(1) += toFloat(p0[Int<1>{}]) * total_scale[0];
        Pair<state_t> const p1 =
            *reinterpret_cast<Pair<state_t> const*>(&state_base[off_lo + 1024]);
        frag_h(2) += toFloat(p1[Int<0>{}]) * total_scale[1];
        frag_h(3) += toFloat(p1[Int<1>{}]) * total_scale[1];
      }

      // Update amax (fp32, bit-exact) AND pack 4 fp32 → 4 bf16 into a_kpair
      // at offset local_n*4 (matches A-frag's (kh=local_n) slice).
#pragma unroll
      for (int i = 0; i < FRAG_SIZE; i += 2) {
        int const d_idx = i / 2;
        float const a0 = fabsf(frag_h(i));
        float const a1 = fabsf(frag_h(i + 1));
        per_thread_amax[d_idx] = fmaxf(per_thread_amax[d_idx], fmaxf(a0, a1));

        Pair<MMA_prop::operand_t> const q =
            pack_float2<MMA_prop::operand_t>(make_float2(frag_h(i), frag_h(i + 1)));
        *reinterpret_cast<Pair<MMA_prop::operand_t>*>(&a_kpair(local_n * FRAG_SIZE + i)) = q;
      }
    }

    // ── B operand for chain matmul-3 K-atom: smem.C[T_pad, kpair*16..+16] ──
    Tensor smem_C_k =
        local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<MMA_prop::K_BIG>{}),
                   make_coord(_0{}, kpair));
    auto smem_C_k_s2r = s2r_thr_B_chain.partition_S(smem_C_k);
    Tensor b_kpair = thr_mma_chain.partition_fragment_B(
        make_tensor((MMA_prop::operand_t*)0x0,
                    make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<MMA_prop::K_BIG>{})));
    auto b_kpair_view = s2r_thr_B_chain.retile_D(b_kpair);
    cute::copy(s2r_B_chain, smem_C_k_s2r, b_kpair_view);

    // Single K-atom chain matmul-3: frag_y_DxT += a_kpair @ b_kpair
    // frag_y_DxT (pre-zeroed by caller) accumulates across all 8 K-atoms.
    cute::gemm(tiled_mma_chain, frag_y_DxT, a_kpair, b_kpair, frag_y_DxT);
  }

  // ── Warp-local amax reduce (Layout<_4,_1> → fully warp-local; no atomics).
#pragma unroll
  for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
    per_thread_amax[i] = fmaxf(per_thread_amax[i],
                               __shfl_xor_sync(constants::MASK_ALL_LANES, per_thread_amax[i], 1));
    per_thread_amax[i] = fmaxf(per_thread_amax[i],
                               __shfl_xor_sync(constants::MASK_ALL_LANES, per_thread_amax[i], 2));
  }

  // ── encode scale (Triton fall-through for amax==0).
  // decode_scale = 1 / encode_scale (mathematically: decode = amax/QUANT_MAX,
  // encode = QUANT_MAX/amax, so decode = 1/encode; and when amax==0 both fall
  // through to 1.f → 1/1 == 1).  Computed inline at the STG below — keeping
  // only `encode_scale_per_row` in regs saves 2 fp32 regs across PASS 2.
  float encode_scale_per_row[D_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
    float const a = per_thread_amax[i];
    encode_scale_per_row[i] = (a == 0.f) ? 1.f : (QUANT_MAX / a);
  }

  // ── STG decode_scale (one writer per (cache, head, d_row)).
  if (must_checkpoint && (lane & 3) == 0) {
    auto* __restrict__ state_scale_w = reinterpret_cast<float*>(params.state_scale);
#pragma unroll
    for (int i = 0; i < D_ROWS_PER_THREAD; ++i) {
      int const d_row_in_atom = lane_d + (i & 1) * 8;
      int const d_row = warp_d_base + d_row_in_atom;
      state_scale_w[state_scale_base + d_row] = 1.f / encode_scale_per_row[i];
    }
  }

  // Hand `encode_scale_per_row` AND `total_scale` (= OLD decode_scale_in ×
  // total_decay) to the caller so PASS 2 (encode replay-again) can:
  //   - dequantize the OLD int8 state with the OLD decode_scale (NOT the NEW
  //     one we just STG'd above — re-reading params.state_scale in PASS 2
  //     would pick up the new value and corrupt the encode), and
  //   - encode the NEW state with the right encode_scale = 127 / amax.
  encode_scale_per_row_out[0] = encode_scale_per_row[0];
  encode_scale_per_row_out[1] = encode_scale_per_row[1];
  total_scale_out[0] = total_scale[0];
  total_scale_out[1] = total_scale[1];
}

// ─────────────────────────────────────────────────────────────────────────
// encode_state_replay_8bit: PASS 2 of the int8 chain rewrite.
//
// Re-runs the replay matmul fresh (replay-again), encodes the post-replay
// state fp32 → int8 using `encode_scale_per_row[]` from PASS 1, and STG.16's
// the int8 pairs to gmem.  Bit-exact with Triton's fp32-encode path.
//
// Called *after* `compute_output_8bit` so that:
//   - `frag_y_DxT`'s 8 fp32 regs are dead (chain matmul-3's accumulator
//     was consumed by the output STG).
//   - PASS 2's gmem STGs fire alongside `store_old_x` / dt_proc / cumAdt
//     writes — all gmem traffic at the kernel tail where there's nothing
//     else to do.
//
// The setup (TiledMma, frag_A_replay, smem layouts) is duplicated
// from `replay_state_mma_8bit_chain` — separate stack frame keeps register
// allocation simple and avoids cross-function lifetime tracking.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS,
          typename SmemT>
__device__ __forceinline__ void encode_state_replay_8bit(
    SmemT& smem, CheckpointingSsuParams const& params, int warp, int lane, int prev_k, int d_tile,
    int64_t cache_slot, int head, float const (&encode_scale_per_row)[2],
    float const (&total_scale)[2], int64_t rand_seed, int64_t state_ptr_offset) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "encode_state_replay_8bit requires 2-byte input_t");
  static_assert(sizeof(state_t) == 1,
                "encode_state_replay_8bit is for 1-byte state_t (int8/fp8) only");
  static_assert(D_PER_CTA == 64,
                "encode_state_replay_8bit requires D_PER_CTA == 64 (M-shard, per-warp M=16).");

  constexpr int NUM_WARPS = 4;
  constexpr int M_PER_WARP = D_PER_CTA / NUM_WARPS;
  static_assert(M_PER_WARP == MMA_prop::M, "Per-warp M must equal m16n8 atom M (=16)");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  int const tid = warp * warpSize + lane;

  using MmaAtomReplayType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG,
                                               MMA_prop::AtomK16, MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  auto tiled_mma_replay =
      make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomReplayType>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_replay = tiled_mma_replay.get_slice(tid);

  constexpr int N_PER_PASS = MMA_prop::N;
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;
  constexpr int FRAG_SIZE = 4;
  constexpr int D_ROWS_PER_THREAD = 2;

  float const total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;

  // total_scale (= OLD decode_scale_in × total_decay) was computed in PASS 1
  // and is passed in by reference.  We MUST NOT re-load decode_scale_in from
  // params.state_scale here — by the time PASS 2 runs, PASS 1 has already
  // STG'd the NEW decode_scale to that same gmem location.

  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A_replay = thr_mma_replay.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_replay_view = s2r_thr_A.retile_D(frag_A_replay);
  cute::copy(s2r_A, smem_A_s2r, frag_A_replay_view);

  // dB coefficients baked into frag_A (same identity as PASS 1).
  apply_dA_coeff<MAX_WINDOW_PAD_MMA_K>(frag_A_replay, smem, total_cumAdt, prev_k, lane);

  auto layout_B_replay = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_B)), layout_B_replay);
  auto s2r_B_replay = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma_replay);
  auto s2r_thr_B_replay = s2r_B_replay.get_slice(tid);

  state_t* state_base = reinterpret_cast<state_t*>(smem.state);

  // Manual swizzle offsets (same derivation as replay_state_mma_8bit_chain).
  int const lane_d = lane / 4;
  int const warp_d_base = warp * M_PER_WARP;
  int const row_lo = warp_d_base + lane_d;
  int const frag_col_base = (lane & 3) << 1;
  int const state_base_lo = row_lo << 7;
  int const state_xor = (row_lo & 7) << 4;

  // Philox state for SR — one refresh every 4 n-passes (cvt_rs_sat_s8x4_f32
  // packs 4 int8s per u32 of randomness, so 1 Philox call covers 16 int8s).
  [[maybe_unused]] uint32_t rand_idx[4];

#pragma unroll
  for (int n = 0; n < NUM_N_PASSES; ++n) {
    int const n_base = n * N_PER_PASS;

    Tensor frag_h = thr_mma_replay.partition_fragment_C(
        make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));

    // Zero-init accumulator — MMA from scratch, state added after.
    clear(frag_h);

    Tensor smem_B_n =
        local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                   make_coord(n, _0{}));
    auto smem_B_s2r_n = s2r_thr_B_replay.partition_S(smem_B_n);
    Tensor frag_B_replay = thr_mma_replay.partition_fragment_B(make_tensor(
        (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
    auto frag_B_replay_view = s2r_thr_B_replay.retile_D(frag_B_replay);
    cute::copy(s2r_B_replay, smem_B_s2r_n, frag_B_replay_view);

    // HMMA: frag_h = frag_A_scaled @ frag_B (c[k] baked into A).
    cute::gemm(tiled_mma_replay, frag_h, frag_A_replay, frag_B_replay, frag_h);

    {
      int const off_lo = state_base_lo + ((frag_col_base + n_base) ^ state_xor);
      Pair<state_t> const p0 = *reinterpret_cast<Pair<state_t> const*>(&state_base[off_lo]);
      frag_h(0) += toFloat(p0[Int<0>{}]) * total_scale[0];
      frag_h(1) += toFloat(p0[Int<1>{}]) * total_scale[0];
      Pair<state_t> const p1 = *reinterpret_cast<Pair<state_t> const*>(&state_base[off_lo + 1024]);
      frag_h(2) += toFloat(p1[Int<0>{}]) * total_scale[1];
      frag_h(3) += toFloat(p1[Int<1>{}]) * total_scale[1];
    }

    // ── Encode + in-place STS to smem.state at cols [n*8, n*8+8) ──
    // Overwrites the OLD 8-bit input *for this n-pass's cols only*.  The next
    // n-pass dequants from a DIFFERENT col band [(n+1)*8, +8) — still OLD —
    // so no read-after-write hazard.  Each warp writes only to its own
    // M-shard rows; cross-warp visibility is established by the
    // caller's __syncthreads before the cooperative store_state call.
    {
      int const off_lo = state_base_lo + ((frag_col_base + n_base) ^ state_xor);
      float const e0 = encode_scale_per_row[0];
      float const e1 = encode_scale_per_row[1];

      if constexpr (PHILOX_ROUNDS > 0) {
        // One Philox4x call yields 4 independent u32s — enough for 4 n-passes
        // when each pass packs all 4 of its 8-bit outputs into one u32 via the
        // dtype-specific x4 cvt_rs (int8: `cvt_rs_sat_s8x4_f32`, 16-bit
        // randomness/elt via bitrev16 trick; fp8 e4m3: `cvt_rs_e4m3x4_f32`,
        // native PTX `cvt.rs.satfinite.e4m3x4.f32` on sm_100a+ with SW fallback).
        int const rand_pos = n & 3;
        if (rand_pos == 0) {
          int64_t const philox_off =
              state_ptr_offset + (int64_t)row_lo * DSTATE + (frag_col_base + n_base);
          conversion::philox_randint4x<PHILOX_ROUNDS>(rand_seed, philox_off, rand_idx[0],
                                                      rand_idx[1], rand_idx[2], rand_idx[3]);
        }
        // Packed layout: byte 0 = q0_lo, byte 1 = q1_lo (→ row_lo store at off_lo)
        //                byte 2 = q0_hi, byte 3 = q1_hi (→ row_hi store at off_lo + 1024)
        uint32_t packed;
        if constexpr (std::is_same_v<state_t, int8_t>) {
          packed = conversion::cvt_rs_sat_s8x4_f32(frag_h(0) * e0, frag_h(1) * e0, frag_h(2) * e1,
                                                   frag_h(3) * e1, rand_idx[rand_pos]);
        } else {
          static_assert(std::is_same_v<state_t, __nv_fp8_e4m3>,
                        "8-bit SR supports state_t in {int8_t, __nv_fp8_e4m3}");
          packed = conversion::cvt_rs_e4m3x4_f32(frag_h(0) * e0, frag_h(1) * e0, frag_h(2) * e1,
                                                 frag_h(3) * e1, rand_idx[rand_pos]);
        }
        Pair<state_t> q_lo, q_hi;
        q_lo.raw = static_cast<uint16_t>(packed & 0xFFFFu);
        q_hi.raw = static_cast<uint16_t>(packed >> 16);
        *reinterpret_cast<Pair<state_t>*>(&state_base[off_lo]) = q_lo;
        *reinterpret_cast<Pair<state_t>*>(&state_base[off_lo + 1024]) = q_hi;
      } else {
        // d_idx=0: row_lo
        state_t const q0_lo = encode_rn_8bit<state_t>(frag_h(0) * e0);
        state_t const q1_lo = encode_rn_8bit<state_t>(frag_h(1) * e0);
        Pair<state_t> q_lo;
        q_lo.raw = static_cast<uint16_t>(state_byte_of(q0_lo)) |
                   (static_cast<uint16_t>(state_byte_of(q1_lo)) << 8);
        *reinterpret_cast<Pair<state_t>*>(&state_base[off_lo]) = q_lo;
        // d_idx=1: row_hi = row_lo + 8, off_hi = off_lo + 1024
        state_t const q0_hi = encode_rn_8bit<state_t>(frag_h(2) * e1);
        state_t const q1_hi = encode_rn_8bit<state_t>(frag_h(3) * e1);
        Pair<state_t> q_hi;
        q_hi.raw = static_cast<uint16_t>(state_byte_of(q0_hi)) |
                   (static_cast<uint16_t>(state_byte_of(q1_hi)) << 8);
        *reinterpret_cast<Pair<state_t>*>(&state_base[off_lo + 1024]) = q_hi;
      }
    }
  }

  // No __syncthreads or cooperative STG here — the caller's single sync
  // provides cross-warp smem.state visibility, then calls store_state.
}

// ────────────────────────────────────────────────────────────────────────
// compute_output_8bit: transposed matmul-4 + epilogue + smem-transpose STG
// ────────────────────────────────────────────────────────────────────────
// Companion to `replay_state_mma_int8_chain`.  Consumes the per-warp
// `frag_y_DxT` (shape ((2,2), 1, T_pad/8) of fp32 per thread; M=D-shard,
// N=T_pad) — pre-loaded with init_out^T from chain matmul-3 — and:
//   1. Decay broadcast: frag_y_DxT *= exp(cumAdt[t])  (per T-col, scalar LDS).
//   2. Chain matmul-4 transposed: frag_y_DxT += x^T[D, T] @ CB_scaled^T[T, T]
//        A operand: smem.x viewed via x_trans (D, T) → LDSM_N feeds A(M=D, K=T).
//        B operand: smem.CB_scaled (T, T) → LDSM_T feeds B(K=T, N=T).
//   3. D*x skip: frag_y_DxT(d, t) += D_val * x[t, d]  (scalar LDS per element;
//        consecutive frag elts at fixed D, varying T → not pair-loadable).
//   4. z-gate: frag_y_DxT *= z * sigmoid(z)            (scalar LDS per element).
//   5. fp32 → input_t pack (in-place register cvt via pack_float2).
//   6. Per-thread STS to smem.output_transpose at (T, D) layout.
//   7. __syncthreads.
//   8. Cooperative STG.128 from smem.output_transpose (T, D) to gmem (T, D).
//
// Cross-warp dependencies (smem.x, smem.z, smem.CB_scaled) are already
// visible because the caller's __syncthreads fires between all replay
// passes and this function.
template <typename input_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE, int NUM_WARPS,
          typename SmemT, typename FragYDxT>
__device__ __forceinline__ void compute_output_8bit(SmemT& smem,
                                                    CheckpointingSsuParams const& params, int warp,
                                                    int lane, int d_tile, int64_t out_seq_base,
                                                    int head, int64_t cache_slot, float D_val,
                                                    int seq_len, FragYDxT& frag_y_DxT) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "compute_output_8bit requires 2-byte input_t");
  static_assert(D_PER_CTA == 64, "compute_output_8bit requires D_PER_CTA == 64");
  static_assert(NUM_WARPS == 4, "compute_output_8bit requires 4 warps");

  int const tid = warp * warpSize + lane;

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  constexpr int M_PER_WARP = D_PER_CTA / NUM_WARPS;  // 16

  // Same TiledMma as replay_state_mma_int8_chain (M-shard, m16n8k16).
  auto tiled_mma_chain =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_chain = tiled_mma_chain.get_slice(tid);

  // ── Smem views ──
  // x_trans: x physically stored at (T, D); transposed view at (D, T).
  // Used as the A operand of the chain matmul-4.
  auto layout_x_trans =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)), layout_x_trans);
  Tensor smem_x_trans_tile =
      local_tile(smem_x_trans, make_shape(Int<D_PER_CTA>{}, Int<NPREDICTED_PAD_MMA_M>{}),
                 make_coord(_0{}, _0{}));

  // x natural (T, D) view — for D-skip + z-gate per-element scalar LDS.
  auto layout_x = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();

  // z natural (T, D) view (aliased so padded rows alias valid rows).
  auto layout_z = make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS,
                                                  SmemT::NPREDICTED>();

  // CB_scaled (T, T_pad) within (NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE).
  auto layout_cb =
      make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.CB_scaled)), layout_cb);

  // ── Per-thread (d, t) coord lookup for epilogue scalar reads + smem-transpose write ──
  auto id_tile = make_identity_tensor(make_shape(Int<D_PER_CTA>{}, Int<NPREDICTED_PAD_MMA_M>{}));
  auto id_part = thr_mma_chain.partition_C(id_tile);

  // ── 1. Decay broadcast: frag_y(i) *= exp(cumAdt[t]) ──
  // Reads `smem.decay[t]` (= exp(cumAdt[t])) precomputed in Phase 0 by
  // `compute_cumAdt` — fused EX2 with the cumsum write, replacing ~512 per-CTA
  // __expf calls in this inner loop with a single LDS per element.
  // For padded T-cols (t >= NPREDICTED), the read returns garbage but the STG
  // at the end is predicated on t < NPREDICTED, so the garbage never reaches gmem.
#pragma unroll
  for (int i = 0; i < size(frag_y_DxT); ++i) {
    int const t = get<1>(id_part(i));
    if (t < seq_len) {
      frag_y_DxT(i) *= smem.decay[t];
    }
  }

  // ── 2. Chain matmul-4: frag_y_DxT += x^T @ CB^T ──
  // A operand: smem.x physically (T, D); transposed view (D, T) used as
  // A(M=D, K=T).  The transposed view has D-stride=1, T-stride=D — same
  // pattern as replay's A from old_x — so use LDSM_T to produce row-major
  // A from this column-wise smem source.
  auto s2r_A_x =
      make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, MMA_prop::operand_t>{}, tiled_mma_chain);
  auto s2r_thr_A_x = s2r_A_x.get_slice(tid);
  auto smem_x_s2r = s2r_thr_A_x.partition_S(smem_x_trans_tile);
  Tensor frag_A_x = thr_mma_chain.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<NPREDICTED_PAD_MMA_M>{})));
  auto frag_A_x_view = s2r_thr_A_x.retile_D(frag_A_x);
  cute::copy(s2r_A_x, smem_x_s2r, frag_A_x_view);

  // B operand for chain matmul-4 = CB^T.  smem.CB natural view shape (T, T)
  // already has T_inner stride 1 = K-major.  Use LDSM_N (no transpose).
  auto s2r_B_CB =
      make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, MMA_prop::operand_t>{}, tiled_mma_chain);
  auto s2r_thr_B_CB = s2r_B_CB.get_slice(tid);
  auto smem_CB_s2r = s2r_thr_B_CB.partition_S(smem_CB);
  Tensor frag_B_CB = thr_mma_chain.partition_fragment_B(
      make_tensor((MMA_prop::operand_t*)0x0,
                  make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<NPREDICTED_PAD_MMA_M>{})));
  auto frag_B_CB_view = s2r_thr_B_CB.retile_D(frag_B_CB);
  cute::copy(s2r_B_CB, smem_CB_s2r, frag_B_CB_view);

  cute::gemm(tiled_mma_chain, frag_y_DxT, frag_A_x, frag_B_CB, frag_y_DxT);

  // ── 3. D*x skip: frag_y(d, t) += D_val * x[t, d] (scalar LDS per element) ──
  if (D_val != 0.f) {
    auto* __restrict__ smem_x_base = reinterpret_cast<input_t const*>(smem.x);
#pragma unroll
    for (int i = 0; i < size(frag_y_DxT); ++i) {
      int const d = get<0>(id_part(i));
      int const t = get<1>(id_part(i));
      if (t < seq_len) {
        int const off = layout_x(t, d);
        frag_y_DxT(i) += D_val * toFloat(smem_x_base[off]);
      }
    }
  }

  // ── 4. z-gate: frag_y *= z * sigmoid(z) (scalar LDS per element) ──
  if (params.z != nullptr) {
    auto* __restrict__ smem_z_base = reinterpret_cast<input_t const*>(smem.z);
#pragma unroll
    for (int i = 0; i < size(frag_y_DxT); ++i) {
      int const d = get<0>(id_part(i));
      int const t = get<1>(id_part(i));
      if (t < seq_len) {
        int const off = layout_z(t, d);
        float const z = toFloat(smem_z_base[off]);
        frag_y_DxT(i) *= z * __fdividef(1.f, (1.f + __expf(-z)));
      }
    }
  }

  // ── 5. Pack fp32 → input_t per element + 6. STS to smem.output_transpose (T, D) ──
  // Padded row stride (D_PER_CTA + 8 = 72 bf16 = 144 bytes) gives:
  //   - 16-byte-aligned LDS.128 / STG.128 across all rows.
  //   - 4-bank shift per row → m16n8 STS pattern hits {bank 0, 4, 8, 12} for the
  //     4 t-rows of an elt → bank-conflict-free (vs 4-way conflict at stride 64).
  // See CheckpointingSsuStorage8bit::OUTPUT_TRANSPOSE_ROW_STRIDE for derivation.
  constexpr int kSmemRowStride = SmemT::OUTPUT_TRANSPOSE_ROW_STRIDE;  // 72 bf16 elts
  auto* __restrict__ smem_out_base = reinterpret_cast<input_t*>(smem.output_transpose);
#pragma unroll
  for (int i = 0; i < size(frag_y_DxT); ++i) {
    int const d = get<0>(id_part(i));
    int const t = get<1>(id_part(i));
    if (t < seq_len) {
      // Pack via pack_float2(f, 0.f) and take low elt — emits a single cvt
      // (compiler folds the dummy into a no-op for the discarded high half).
      smem_out_base[t * kSmemRowStride + d] =
          pack_float2<input_t>(make_float2(frag_y_DxT(i), 0.f))[Int<0>{}];
    }
  }

  // ── 7. Warp sync for cross-lane STS→LDS ordering ──
  __syncwarp();

  // ── 8. Warp-local cooperative STG.128: 32 lanes → one warp's 16 D-rows ──
  // Each warp's data: 16 D-rows × T_pad=16 cols × 2 B = 512 B.
  // Re-tile 32 lanes: (t = lane%16, d_group = lane/16 ∈ {0, 1}) → covers
  // T_pad × 2 D-groups = 32 slots, each STG.128 = 8 D-cols × 2 B = 16 B.
  // No cross-warp coordination → no __syncthreads.
  constexpr int kElsPerSTG = 16 / sizeof(input_t);          // 8 bf16 elts per STG.128
  constexpr int kDGroupsPerWarp = M_PER_WARP / kElsPerSTG;  // = 16 / 8 = 2
  static_assert(NPREDICTED_PAD_MMA_M * kDGroupsPerWarp == 32,
                "warp-local STG re-tile: T_pad × dGroupsPerWarp must equal warpSize");

  int const stg_t = lane % NPREDICTED_PAD_MMA_M;
  int const stg_d_group = lane / NPREDICTED_PAD_MMA_M;
  int const warp_d_base = warp * M_PER_WARP;
  int const stg_d = warp_d_base + stg_d_group * kElsPerSTG;

  if (stg_t < seq_len) {
    int const smem_off = stg_t * kSmemRowStride + stg_d;

    auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
    int64_t const out_base = out_seq_base + (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;
    int64_t const gmem_off = out_base + (int64_t)stg_t * params.out_stride_token + stg_d;

    // 128-bit copy.  smem_off * 2 B = (t * 144 + d * 2) is 16-byte aligned
    // for any t when d % 8 == 0 (here d_offset_within_warp = 0 or 8).
    using Vec = uint4;
    *reinterpret_cast<Vec*>(&output_ptr[gmem_off]) =
        *reinterpret_cast<Vec const*>(&smem_out_base[smem_off]);
  }
}

// =============================================================================
// add_init_out_8bit: matmul-3 for the no-checkpoint path (N-shard).
// =============================================================================
// Computes `frag_y[T, D] = C @ dequant(smem.state)^T` in N-shard `Layout<_1,_4>`,
// reading int8/fp8 state directly from smem and dequanting per-element to bf16
// in registers before the HMMA.  Mirrors the bf16 path's `add_init_out` but with
// a custom B-operand loader since the 1-byte state can't use LDSM directly.
//
// Per-thread B-frag layout (m16n8k16, PTX ISA):
//   Per-lane 4 bf16 elts at (K, N) = {(2t, gID), (2t+1, gID), (2t+8, gID), (2t+9, gID)}
//   where t = lane%4, gID = lane/4.
// In our matmul-3: B = state^T = (DSTATE = K-axis, D_PER_CTA = N-axis).  N-shard
// gives each warp `MMA::N = 8` D-cols per N-tile, so this lane's d_row =
// n_tile*N_TILE + warp*8 + lane/4.
//
// Per K-tile: load 4 int8 bytes per lane (2 byte-pair LDS into Pair<state_t>),
// CAST to bf16 (no per-row scale), pack into B-frag, HMMA into the n-th frag_y.
// `decode_scale[d]` is per-output-col (constant across the K reduction), so
// the caller pulls it OUT of the inner product and applies it post-matmul in
// the β-scale loop:
//   y[t, d] = decode_scale[d] · Σ_n C[t, n] · state_byte[d, n]
// This eliminates the per-cell `... * scale` FMUL chain (was the
// long_scoreboard hotspot at line 1066) at the cost of 2 extra FMUL/elt in
// the post-matmul C-frag scale (net 1792× fewer FMUL per warp).
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, typename SmemT,
          typename TiledMma, typename ThrMma, typename... FragY>
__device__ __forceinline__ void add_init_out_8bit(SmemT const& smem, int warp, int lane,
                                                  TiledMma const& tiled_mma, ThrMma const& thr_mma,
                                                  int tid, FragY&... frag_y) {
  using namespace cute;
  static_assert(sizeof(state_t) == 1, "add_init_out_8bit requires 1-byte state");
  static_assert(D_PER_CTA == 64, "add_init_out_8bit requires D_PER_CTA == 64 (8-bit D_SPLIT=1)");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int K_TILE = MMA_prop::K_BIG;                 // 16
  constexpr int NUM_K_TILES = DSTATE / K_TILE;            // 8
  constexpr int N_TILE = cute::tile_size<1>(TiledMma{});  // 32 (4 warps × MMA::N=8)
  constexpr int NUM_N_TILES = sizeof...(FragY);           // 2 (D_PER_CTA / N_TILE)
  static_assert(NUM_N_TILES * N_TILE == D_PER_CTA, "FragY count must match D_PER_CTA / N_TILE");

  // ── Per-thread coords ──
  int const t = lane & 3;                      // K-pair index within K-atom
  int const lane_d = lane >> 2;                // gID = lane/4; selects N-col within atom
  int const warp_d_base = warp * MMA_prop::N;  // warp's 8-col offset within an N-tile

  // ── A operand (C): swizzled (T_pad, DSTATE), K-tiled per K-loop iter ──
  auto layout_C_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, DSTATE, SmemT::NPREDICTED>();
  Tensor smem_C = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.C)),
                              layout_C_swz);
  Tensor smem_C_ktiled = local_tile(smem_C, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<K_TILE>{}),
                                    make_coord(_0{}, _));
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto smem_A_s2r = s2r_thr_A.partition_S(smem_C_ktiled);
  Tensor frag_A = thr_mma.partition_fragment_A(smem_C_ktiled(_, _, _0{}));
  auto frag_A_view = s2r_thr_A.retile_D(frag_A);

  // ── B-frag (one m16n8k16 atom of B; 4 bf16 elts/lane) ──
  Tensor frag_B = thr_mma.partition_fragment_B(
      make_tensor((MMA_prop::operand_t*)0x0, make_shape(Int<K_TILE>{}, Int<MMA_prop::N>{})));
  static_assert(decltype(size(frag_B))::value == 4, "B-frag must be 4 elts/lane for m16n8k16");

  // ── Smem state base (1-byte) + Swizzle<3,4,3> XOR formula:
  //   off = d_row * DSTATE + (K XOR ((d_row & 7) << 4))
  // K within the same swizzle row-group {0..7 mod 8} shares the same XOR mask. ──
  state_t const* state_base = reinterpret_cast<state_t const*>(smem.state);

  // Pre-clear accumulators (caller doesn't pre-zero — matches bf16 add_init_out).
  (clear(frag_y), ...);

  // Parameter-pack indexing via pointer array (same pattern as pipelined_kloop_gemm).
  using FragY0 = std::tuple_element_t<0, std::tuple<FragY...>>;
  FragY0* frag_y_p[NUM_N_TILES] = {(&frag_y)...};

  // ── K-loop ──
#pragma unroll
  for (int k = 0; k < NUM_K_TILES; ++k) {
    int const K_base = k * K_TILE;

    // Load A K-tile via LDSM (shared across all N-tiles within this K-tile).
    cute::copy(s2r_A, smem_A_s2r(_, _, _, k), frag_A_view);

    CUTE_UNROLL
    for (int n = 0; n < NUM_N_TILES; ++n) {
      int const d_row = n * N_TILE + warp_d_base + lane_d;
      int const state_base_lo = d_row << 7;    // d_row * DSTATE (DSTATE=128 → <<7)
      int const state_xor = (d_row & 7) << 4;  // Swizzle<3,4,3>
      int const off_lo = state_base_lo + ((K_base + (t << 1)) ^ state_xor);
      int const off_hi = state_base_lo + ((K_base + (t << 1) + 8) ^ state_xor);

      Pair<state_t> const p_lo = *reinterpret_cast<Pair<state_t> const*>(&state_base[off_lo]);
      Pair<state_t> const p_hi = *reinterpret_cast<Pair<state_t> const*>(&state_base[off_hi]);

      // Pure int8/fp8 → bf16 cast.  decode_scale is applied post-matmul in
      // the caller's β-scale loop.
      Pair<MMA_prop::operand_t> const b_lo = pack_float2<MMA_prop::operand_t>(
          make_float2(toFloat(p_lo[Int<0>{}]), toFloat(p_lo[Int<1>{}])));
      Pair<MMA_prop::operand_t> const b_hi = pack_float2<MMA_prop::operand_t>(
          make_float2(toFloat(p_hi[Int<0>{}]), toFloat(p_hi[Int<1>{}])));

      // frag_B(0,1) = K-pair at {K_base+2t, K_base+2t+1}; (2,3) = at {+8, +9}.
      *reinterpret_cast<Pair<MMA_prop::operand_t>*>(&frag_B(0)) = b_lo;
      *reinterpret_cast<Pair<MMA_prop::operand_t>*>(&frag_B(2)) = b_hi;

      cute::gemm(tiled_mma, *frag_y_p[n], frag_A, frag_B, *frag_y_p[n]);
    }
  }
}

// =============================================================================
// compute_no_write_output_8bit — N-shard output for the no-checkpoint path.
// =============================================================================
// Mirror of the bf16 path's `compute_no_write_output`, but with int8/fp8 state.
// Uses N-shard `Layout<_1,_4>` (best smem traffic) instead of the M-shard chain
// — the M-shard exists only for amax reduction in the checkpoint path, which
// doesn't run here.
//
//   y[t, d] = β(t) · u[t, d]                                              (matmul-3 via
//   add_init_out_8bit)
//           + Σ_{j<NPREDICTED}  CB_scaled[t, j] · x[j, d]                 (add_cb_x — existing)
//           + Σ_{i<prev_k}      CB_old   [t, i] · old_x[i, d]             (add_cb_old_x — existing)
//           + D[d] · x[t, d]    (+ z gating)
//
//   β(t) = exp(total_old_cumAdt + cumAdt[t]) where total_old_cumAdt =
//          smem.old_cumAdt[prev_k − 1] (= 0 when prev_k == 0).
template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void compute_no_write_output_8bit(
    SmemT& smem, CheckpointingSsuParams const& params, int warp, int lane, int prev_k, int d_tile,
    int64_t out_seq_base, int head, int64_t cache_slot, float D_val, int seq_len) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "compute_no_write_output_8bit requires 2-byte input_t");
  static_assert(sizeof(state_t) == 1, "compute_no_write_output_8bit is for 1-byte state");
  static_assert(D_PER_CTA == 64, "compute_no_write_output_8bit requires D_PER_CTA == 64");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA for matmul-3 + matmul-4-new (m16n8k16) ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── TiledMMA for matmul-4-old: K = MAX_WINDOW_PAD_MMA_K ∈ {8, 16} → atom dispatch ──
  using MmaAtomOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, MMA_prop::AtomK16,
                                        MMA_prop::AtomK8>;
  using LdsmAOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U32x4_LDSM_N,
                                      SM75_U32x2_LDSM_N>;
  using LdsmBOld = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                      SM75_U16x2_LDSM_T>;
  auto tiled_mma_old = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomOld>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma_old = tiled_mma_old.get_slice(tid);

  // ── Swizzled smem views ──
  auto layout_x_swz = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)),
                              layout_x_swz);
  auto layout_x_trans_swz =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)), layout_x_trans_swz);

  auto layout_old_x_trans_swz =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_old_x_trans =
      make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)),
                  layout_old_x_trans_swz);

  auto layout_z_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS, NPREDICTED>();
  Tensor smem_z =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.z)), layout_z_swz);

  // ── S2R copies (matmul-4-new) ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B_trans =
      make_tiled_copy_B(Copy_Atom<SM75_U16x2_LDSM_T, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B_trans = s2r_B_trans.get_slice(tid);

  // ── S2R copies (matmul-4-old) ──
  auto s2r_A_old = make_tiled_copy_A(Copy_Atom<LdsmAOld, MMA_prop::operand_t>{}, tiled_mma_old);
  auto s2r_thr_A_old = s2r_A_old.get_slice(tid);
  auto s2r_B_old_trans =
      make_tiled_copy_B(Copy_Atom<LdsmBOld, MMA_prop::operand_t>{}, tiled_mma_old);
  auto s2r_thr_B_old_trans = s2r_B_old_trans.get_slice(tid);

  // ── Load CB_scaled A operand (cols [0, NPREDICTED_PAD_MMA_M)) ──
  auto layout_cb_swz =
      make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.CB_scaled)), layout_cb_swz);
  auto smem_CB_s2r = s2r_thr_A.partition_S(smem_CB);
  Tensor frag_CB_A = thr_mma.partition_fragment_A(smem_CB);
  auto frag_CB_A_view = s2r_thr_A.retile_D(frag_CB_A);
  cute::copy(s2r_A, smem_CB_s2r, frag_CB_A_view);

  // ── Load CB_old A operand (cols [NPREDICTED_PAD_MMA_M, +MAX_WINDOW_PAD_MMA_K)) ──
  auto layout_cb_full = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.CB_scaled)), layout_cb_full);
  Tensor smem_CB_old =
      local_tile(smem_CB_full, make_tile(Int<NPREDICTED_PAD_MMA_M>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                 make_coord(_0{}, NPREDICTED_PAD_MMA_M / MAX_WINDOW_PAD_MMA_K));
  auto smem_CB_old_s2r = s2r_thr_A_old.partition_S(smem_CB_old);
  Tensor frag_CB_old_A = thr_mma_old.partition_fragment_A(smem_CB_old);
  auto frag_CB_old_A_view = s2r_thr_A_old.retile_D(frag_CB_old_A);
  cute::copy(s2r_A_old, smem_CB_old_s2r, frag_CB_old_A_view);

  // ── Decay broadcast (per-T scalar, stride-0 on N) ──
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(smem.cumAdt),
      make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  float const total_old_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float const beta_extra = __expf(total_old_cumAdt);

  // ── Post-matmul-3 state decode_scale (factored OUT of the inner K-product). ──
  // y[t, d] = decode_scale[d] · raw_y[t, d] where raw_y = C @ (bf16)(state_byte)^T.
  // Per lane, the m16n8 C-frag holds 4 elts spanning 2 unique d-cols (d_lo = 2t,
  // d_hi = 2t+1) per N-tile.  Indexed by `i & 1` in the epilogue scale loop.
  auto const* __restrict__ state_scale_ptr = reinterpret_cast<float const*>(params.state_scale);
  int64_t const state_scale_base = cache_slot * params.state_scale_stride_seq +
                                   (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;
  constexpr int NUM_N_TILES = D_PER_CTA / N_TILE;
  static_assert(NUM_N_TILES == 2,
                "compute_no_write_output_8bit assumes NUM_N_TILES == 2 (D_PER_CTA=64, N_TILE=32)");
  int const t_col = lane & 3;  // 2t and 2t+1 are this lane's two C-frag d-cols
  float decode_scale[NUM_N_TILES][2];
  CUTE_UNROLL
  for (int n = 0; n < NUM_N_TILES; ++n) {
    int const d_lo = n * N_TILE + warp * MMA_prop::N + (t_col << 1);
    decode_scale[n][0] = state_scale_ptr[state_scale_base + d_lo];
    decode_scale[n][1] = state_scale_ptr[state_scale_base + d_lo + 1];
  }

  // ── Gmem output base ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  int64_t const out_base = out_seq_base + (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;

  // ── Row predicate ──
  auto id_tile = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  bool const pred_row_lo = get<0>(id_part(0)) < seq_len;
  bool const pred_row_hi = get<0>(id_part(2)) < seq_len;

  auto epilogue = [&](auto& frag_y, int n) {
  // β-scale + state decode_scale fused: frag_y(i) *= β · exp(cumAdt[t]) · decode_scale[d].
  // The decode_scale[d] absorbs the per-row state quant factor (was previously
  // multiplied into each B-element during dequant).
#pragma unroll
    for (int i = 0; i < size(frag_y); ++i) {
      int const d_idx = i & 1;  // i=0,2 → d_lo; i=1,3 → d_hi
      frag_y(i) *= beta_extra * __expf(decay_part(i)) * decode_scale[n][d_idx];
    }

    // matmul-4-new (CB_scaled @ x).
    add_cb_x<input_t, MMA_prop::operand_t, N_TILE, NPREDICTED_PAD_MMA_M>(
        frag_y, frag_CB_A, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);

    // matmul-4-old (CB_old @ old_x).
    add_cb_old_x<input_t, MMA_prop::operand_t, N_TILE, MAX_WINDOW_PAD_MMA_K>(
        frag_y, frag_CB_old_A, smem_old_x_trans, s2r_B_old_trans, s2r_thr_B_old_trans, thr_mma_old,
        tiled_mma_old, n);

    // D·x.
    add_D_skip<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);

    // z-gate.
    compute_z_gating<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);

    // Direct partition_C STG.
    auto gOut_tile = make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                                 make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}),
                                             make_stride(params.out_stride_token, _1{})));
    auto gOut_part = thr_mma.partition_C(gOut_tile);
#pragma unroll
    for (int i = 0; i < size(frag_y); i += 2) {
      bool const pred_i = (i & 2) ? pred_row_hi : pred_row_lo;
      if (pred_i) {
        *reinterpret_cast<Pair<input_t>*>(&gOut_part(i)) =
            pack_float2<input_t>(make_float2(frag_y(i), frag_y(i + 1)));
      }
    }
  };

  // ── Matmul-3: frag_y = C @ (bf16)(state_byte)^T  (smem.state retains s_0 since
  //   replay skipped; decode_scale[d] applied post-matmul in the epilogue). ──
  Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
  Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
  add_init_out_8bit<input_t, state_t, DIM, D_PER_CTA, DSTATE>(smem, warp, lane, tiled_mma, thr_mma,
                                                              tid, frag_y_0, frag_y_1);
  epilogue(frag_y_0, 0);
  epilogue(frag_y_1, 1);
}

// =============================================================================
// Per-path dispatcher: no-checkpoint branch (must_checkpoint == false).
// =============================================================================
// Sync makes warps 0,1's CB_scaled writes AND warps 2,3's CB_old writes
// visible to all warps before matmul-3 and matmul-4 read smem.{CB_scaled,
// CB_old, x, z}.  Matches the bf16 path's `ssu_nocheckpoint`.
template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void ssu_nocheckpoint_8bit(
    SmemT& smem, CheckpointingSsuParams const& params, int warp, int lane, int prev_k, int d_tile,
    int64_t out_seq_base, int head, int64_t cache_slot, float D_val, int seq_len) {
  __syncthreads();
  compute_no_write_output_8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                               NUM_WARPS>(smem, params, warp, lane, prev_k, d_tile, out_seq_base,
                                          head, cache_slot, D_val, seq_len);
}

// =============================================================================
// Per-path dispatcher: checkpoint branch (must_checkpoint == true).
// =============================================================================
// Encapsulates the existing M-shard chain: PASS 1 replay+matmul-3 → frag_y_DxT,
// PASS 2 re-replay+encode (always runs since must_checkpoint==true here), the
// single __syncthreads, cooperative state STG, and the transposed matmul-4 +
// transpose-STG output.
//
// Pulled out of `checkpointing_ssu_kernel_8bit` to mirror the bf16 path's
// `ssu_checkpoint` and make the kernel-body dispatch on
// must_checkpoint readable.
template <typename input_t, typename state_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE,
          int NUM_WARPS, int PHILOX_ROUNDS, typename SmemT>
__device__ __forceinline__ void ssu_checkpoint_8bit(SmemT& smem,
                                                    CheckpointingSsuParams const& params, int warp,
                                                    int lane, int prev_k, int d_tile,
                                                    int64_t out_seq_base, int head,
                                                    int64_t cache_slot, float D_val, int seq_len) {
  using namespace cute;
  int const tid = warp * warpSize + lane;

  // ── Allocate per-warp frag_y_DxT (chain mma C-frag, fp32) ──
  // Layout ((2, 2), MMA_M=1, MMA_N=NPREDICTED_PAD_MMA_M/8) per thread.
  // Caller must zero before chain matmul-3 accumulates.
  auto tiled_mma_chain =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_4, _1>>{});
  auto thr_mma_chain = tiled_mma_chain.get_slice(tid);
  auto id_DxT =
      make_identity_tensor(make_shape(Int<D_PER_CTA>{}, Int<SmemT::NPREDICTED_PAD_MMA_M>{}));
  Tensor frag_y_DxT = thr_mma_chain.partition_fragment_C(id_DxT);
  cute::clear(frag_y_DxT);

  // ── Phase 1b: replay + amax + chain matmul-3 → frag_y_DxT (init_out^T).
  // `encode_scale_per_row[]` is computed at the end of PASS 1 (from the warp-
  // reduced amax) and consumed by `encode_state_replay_8bit` further below —
  // *after* `compute_output_8bit` consumes `frag_y_DxT` and STGs the output.
  float encode_scale_per_row[2];
  float total_scale[2];
  replay_state_mma_8bit_chain<input_t, state_t, DIM, D_PER_CTA, DSTATE>(
      smem, params, warp, lane, prev_k, d_tile, cache_slot, head, /*must_checkpoint=*/true,
      frag_y_DxT, encode_scale_per_row, total_scale);

  // ── Philox seed for stochastic rounding (deferred to reduce register pressure) ──
  [[maybe_unused]] int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
  // `state_ptr_offset` is int64 — matches Triton's `base_rand =
  // cache_batch_idx * stride_state_batch + ...` (cache_batch_idx is .to(int64)).
  // Full 64 bits flow through `philox_randint4x`, which splits low/high
  // across Philox c0/c1.  No collision risk at large serving cache sizes.
  int64_t const state_ptr_offset =
      cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE;

  // ── PASS 2 (replay-again): re-run replay HMMA, encode fp32 → int8 to
  // smem.state.  Runs BEFORE the sync so both replay passes overlap with
  // warps 0,1's CB precompute — one fewer __syncthreads in the kernel.
  // frag_y_DxT stays live through PASS 2 (extra register pressure accepted). ──
  encode_state_replay_8bit<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS>(
      smem, params, warp, lane, prev_k, d_tile, cache_slot, head, encode_scale_per_row, total_scale,
      rand_seed, state_ptr_offset);

  // ── Single sync: cross-warp visibility for smem.CB_scaled (warps 0,1) /
  // smem.x (warp 2) / smem.z (warp 3) / smem.state (all warps' M-shards). ──
  __syncthreads();

  // ── Cooperative STG.128 for encoded state (after sync for cross-warp
  // smem.state visibility).  Fire-and-forget before compute_output_8bit. ──
  store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile, head,
                                                          cache_slot);

  // ── Phase 2: transposed matmul-4 + epilogue + smem-transpose STG ──
  compute_output_8bit<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
      smem, params, warp, lane, d_tile, out_seq_base, head, cache_slot, D_val, seq_len, frag_y_DxT);
}

// =============================================================================
// Kernel — int8 chain rewrite (separate kernel from the generic path)
// =============================================================================
// The int8 path uses a fundamentally different output computation:
//   1. M-shard replay (Layout<_4, _1>) — same as v15.4.
//   2. Chained matmul-3: replay's fp32 C-frag → bf16 A-frag in registers via
//      `convert_layout_acc_Aregs_sm80` (no smem.new_state staging).
//   3. Transposed matmul-4: x as A (M=D), CB^T as B → output^T(D, T) in regs.
//   4. Smem-transpose + cooperative STG.128 to (T, D) gmem.
// To keep the generic kernel uncluttered (no `if constexpr (sizeof(state_t) == 1)`
// branches), the int8 kernel is a standalone function that calls the new
// helpers (`replay_state_mma_8bit_chain`, `compute_output_8bit`) and uses
// `CheckpointingSsuStorage8bit` for smem.  Phase 0/1 helpers (`load_data`,
// `store_old_B`, `compute_CB_scaled_2warp`) are reused verbatim — they only
// touch shared smem fields that both storage structs expose by name.
//
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, bool VARLEN = false>
__global__ void checkpointing_ssu_kernel_8bit(CheckpointingSsuParams params) {
  using namespace cute;
  static_assert(sizeof(state_t) == 1,
                "checkpointing_ssu_kernel_8bit requires 1-byte state_t (int8 or fp8 e4m3)");
  static_assert(NPREDICTED <= MAX_WINDOW);
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG);
  // int8 path uses M-shard layout (Layout<_4,_1>): per-warp M = 16 = m16n8
  // atom M.  D_PER_CTA must equal DIM (D_SPLIT=1) to give 4×16=64 D-rows/CTA.
  // The wrapper enforces d_split == 1 for int8.
  constexpr int D_PER_CTA = DIM;
  static_assert(D_PER_CTA == 64, "int8 chain kernel requires DIM == 64");
  assert(params.d_split == 1);

  using SmemT =
      CheckpointingSsuStorage8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // Grid: (1, batch, nheads).  D-tile is always 0 for int8 (D_SPLIT=1).
  int const d_tile = blockIdx.x;
  int const seq = blockIdx.y;
  int const head = blockIdx.z;
  int const lane = threadIdx.x;
  int const warp = threadIdx.y;
  int const group_idx = head / HEADS_PER_GROUP;

  // ── Resolve cache slot ──
  auto const* __restrict__ sbi = reinterpret_cast<stateIndex_t const*>(params.state_batch_indices);
  int64_t const cache_slot = sbi ? static_cast<int64_t>(sbi[seq]) : seq;
  if (cache_slot == params.pad_slot_id) return;

  auto const* __restrict__ buf_idx_ptr = reinterpret_cast<int32_t const*>(params.cache_buf_idx);
  int const buf_read = __ldg(&buf_idx_ptr[cache_slot]);

  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = prev_ptr[cache_slot];

  // ── Varlen vs non-varlen prologue.  The kernel branches once on the
  // VARLEN template; downstream helpers receive `seq_len` (constexpr-foldable
  // NPREDICTED in non-varlen, runtime in varlen) and pre-computed per-sequence
  // gmem base offsets (`x_seq_base` etc.) — they're varlen-agnostic.
  //
  // Uniform gmem-base formula: `outer * *_stride_seq` where
  //   non-varlen: outer = seq (= blockIdx.y), stride_seq = x.stride(0).
  //   varlen   : outer = cu_seqlens[seq], stride_seq = x.stride(1).
  // The wrapper picks the right stride_seq value; the kernel only branches
  // on whether to load cu_seqlens.
  int seq_len;
  int64_t outer;
  if constexpr (VARLEN) {
    auto const* __restrict__ cu_seqlens = reinterpret_cast<int32_t const*>(params.cu_seqlens);
    // Two LDG.E.32 (not one LDG.E.64): cu_seqlens is only 4-byte aligned
    // at `&cu_seqlens[seq]` when seq is odd, and PTX
    // `ld.global.v2.b32` faults on a 4-byte-aligned address.  ptxas emits
    // the two scalar loads back-to-back; latency is hidden against the
    // following ALU work.
    int const bos = __ldg(&cu_seqlens[seq]);
    int const eos = __ldg(&cu_seqlens[seq + 1]);
    seq_len = eos - bos;
    if (seq_len <= 0) return;
    outer = (int64_t)bos;
  } else {
    seq_len = NPREDICTED;
    outer = (int64_t)seq;
  }
  // x/B/C bases computed inside `load_post_pdl_wait_data` from `outer` —
  // see generic kernel for rationale (avoid pinning 6 regs across gdc_wait).
  int64_t const dt_seq_base = outer * params.dt_stride_seq + head;
  int64_t const z_seq_base = outer * params.z_stride_seq;
  int64_t const out_seq_base = outer * params.out_stride_seq;

  bool const must_checkpoint = (prev_k + seq_len > MAX_WINDOW);
  int const buf_write = must_checkpoint ? (1 - buf_read) : buf_read;
  int const write_offset = must_checkpoint ? 0 : prev_k;

  // ── Load scalars (A, dt_bias, D) ──
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  float const A_val = toFloat(A_ptr[head]);
  float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  // ── Phase 0: two-phase load around the PDL barrier (see generic kernel
  // for the full rationale).  Pre-wait: state + old_* cache + in_proj
  // outputs (dt, z) + scalar scans.  Post-wait: x/B/C from conv1d. ──
  // ENABLE_PDL is JIT-stamped; `if constexpr` keeps only one load path in
  // the binary (no register pressure leak from the unused path).
  if constexpr (ENABLE_PDL) {
    load_pre_pdl_wait_data<input_t, dt_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                           NUM_WARPS>(smem, params, lane, warp, d_tile, head, group_idx, cache_slot,
                                      buf_read, A_val, dt_bias_val, dt_seq_base, z_seq_base,
                                      seq_len);
    gdc_wait();
    load_post_pdl_wait_data<input_t, NPREDICTED, DIM, D_PER_CTA, DSTATE>(
        smem, params, lane, warp, d_tile, head, group_idx, outer, seq_len);
  } else {
    load_data<input_t, dt_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
        smem, params, lane, warp, d_tile, head, group_idx, cache_slot, buf_read, A_val, dt_bias_val,
        outer, seq_len);
  }

  // ── store_old_B hoist (warps 0,1 only, d_tile == 0) ──
  if (d_tile == 0 && warp < 2) {
    store_old_B<input_t, NPREDICTED, DSTATE, HEADS_PER_GROUP>(
        smem, params, warp, lane, head, group_idx, cache_slot, buf_write, write_offset, seq_len);
  }

  // ── CB precompute (4-warp split): warps 0,1 compute CB_scaled (new tokens);
  // warps 2,3 compute CB_old (old tokens) in the no-write path only.  Mirrors
  // the bf16 path's dispatch — warps 2,3 stay idle in checkpoint mode and
  // pick up work below inside `ssu_checkpoint_8bit`'s replay. ──
  if (warp < 2) {
    compute_CB_scaled_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane, seq_len);
  } else if (!must_checkpoint) {
    compute_CB_old_2warp<input_t, NPREDICTED, MAX_WINDOW, DSTATE>(smem, warp, lane, prev_k,
                                                                  seq_len);
  }

  // ── Phase 1b + 2: per-path dispatch ──
  // Checkpoint: M-shard chain (PASS 1 + PASS 2 + sync + state STG + transposed
  //             matmul-4 with smem-transpose STG).
  // No-write : N-shard matmul-3 from int8/fp8 state + matmul-4-new + matmul-4-old
  //             + direct partition_C STG (mirrors the bf16 no-write path).
  // must_checkpoint is uniform across the CTA — both branches contain a
  // __syncthreads so divergence is balanced.
  if (must_checkpoint) {
    ssu_checkpoint_8bit<input_t, state_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                        PHILOX_ROUNDS>(smem, params, warp, lane, prev_k, d_tile, out_seq_base, head,
                                       cache_slot, D_val, seq_len);
  } else {
    ssu_nocheckpoint_8bit<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                          NUM_WARPS>(smem, params, warp, lane, prev_k, d_tile, out_seq_base, head,
                                     cache_slot, D_val, seq_len);
  }

  // ── PDL: signal downstream that `output` is written.  Cache writes below
  // target tensors only the next SSU step reads, not the immediate
  // downstream kernel — safe to signal first. ──
  if constexpr (ENABLE_PDL) {
    gdc_launch_dependents();
  }

  // ── Phase 3: cache writes (old_x, dt_proc, cumAdt) ──
  store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                   cache_slot, write_offset, seq_len);
  if (d_tile == 0 && warp == 0 && lane < seq_len) {
    auto* __restrict__ old_dt_w = reinterpret_cast<float*>(params.old_dt);
    int64_t const dt_w_base = cache_slot * params.old_dt_stride_seq +
                              buf_write * params.old_dt_stride_dbuf +
                              head * params.old_dt_stride_head;
    old_dt_w[dt_w_base + write_offset + lane] = smem.dt_proc[lane];
  }
  if (d_tile == 0 && warp == 1 && lane < seq_len) {
    auto* __restrict__ old_cumAdt_w = reinterpret_cast<float*>(params.old_cumAdt);
    int64_t const ca_w_base = cache_slot * params.old_cumAdt_stride_seq +
                              buf_write * params.old_cumAdt_stride_dbuf +
                              head * params.old_cumAdt_stride_head;
    old_cumAdt_w[ca_w_base + write_offset + lane] = smem.cumAdt[lane];
  }
}

}  // namespace flashinfer::mamba::checkpointing

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_8BIT_CUH_
