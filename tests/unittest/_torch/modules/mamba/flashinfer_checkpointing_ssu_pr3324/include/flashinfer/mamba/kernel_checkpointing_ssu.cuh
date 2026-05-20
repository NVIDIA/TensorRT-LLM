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
#ifndef FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_CUH_
#define FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_CUH_

// Incremental SSU kernel — matmul-based, tensor-core MMA (single path).
// Single CTA per (batch, head). Grid: (batch, nheads).
// 4 warps per CTA, 128 threads total.
//
// Single __syncthreads() via per-warp data ownership.  Every smem
// read before the final barrier is served by data the same warp loaded.
// No mbarriers, no cross-warp visibility for the first half of the kernel.
//
// Phase 0 (per-warp cp.async, no cross-warp sync):
//   State: each warp loads own DIM slice (rows [16W : 16W+16]).
//   B, C:  redundant on W0, W1 (both do 2-warp CB).
//   old_B: redundant on all 4 warps (each warp's replay needs full DSTATE).
//   old_x: redundant on all 4 warps.
//   x:     W2 only  (Phase-2 read — covered by the single syncthreads).
//   z:     W3 only  (Phase-2 read — covered by the single syncthreads).
//   Scalars + cumAdt: redundant on each warp's first NPREDICTED/MAX_WINDOW lanes.
//   Each warp: __pipeline_commit → __pipeline_wait_prior(0) → __syncwarp.
//
// Phase 1 (runs with *no* barrier; CB ‖ replay parallelism preserved):
//   - store_old_B hoisted here — W0,W1 only (they hold valid smem.B).
//   - Warps 0,1: compute_CB_scaled_2warp (bf16 HMMA → swizzled smem.CB_scaled).
//   - All warps: replay_state_mma (HMMA; state in smem updated in-place,
//                each warp touches only its own DIM rows).
//
// __syncthreads()   ← THE ONE.  Provides cross-warp visibility of:
//                      CB_scaled (W0,W1→all), x (W2→all), z (W3→all).
//
// Phase 2: compute_and_store_output
//   = (C @ state^T) * decay + CB_scaled @ x + D*x, z-gate → direct gmem STG.
//   State writeback hoisted inside the orchestrator once matmul 3 has
//   finished consuming smem.state.
//
// Phase 3: old_x / old_dt / old_cumAdt cache writes.

#include "kernel_checkpointing_ssu_common.cuh"

namespace flashinfer::mamba::checkpointing {

// =============================================================================
// Shared memory layout
// =============================================================================
// smem holds the state in its native dtype (`state_t`).  cp.async pulls the
// native dtype straight into smem (with the matching `SmemSwizzle<state_t>`),
// and the conversion to `MMA_prop::operand_t` happens on the register read inside
// add_init_out / replay_state_mma.
// `D_PER_CTA` is the per-CTA D dimension after D-split.
// For D_SPLIT = 1 (default), D_PER_CTA == DIM (per-head DIM).  At D_SPLIT > 1
// the storage is sliced: each CTA owns a contiguous D_PER_CTA-row slice of
// the head's D axis.  Buffers that aren't D-owned (B, C, old_B, scalars) are
// unaffected.
//
// Note on D_SMEM_COLS: the Swizzle<3,3,3> atom for bf16 is (8, 64), so
// `make_swizzled_layout_rc` requires col counts to be multiples of 64.  When
// D_PER_CTA < 64 (e.g. D_SPLIT=2 → D_PER_CTA=32) we pad the D-owned buffer
// cols up to the swizzle atom width.  The cp.async only fills the first
// D_PER_CTA cols; the padded tail is unused but keeps the swizzle layout
// well-formed.  Cost: 1 KB per [NPREDICTED_PAD_MMA_M, 64] buffer at D_PER_CTA=32.
template <typename input_t, typename state_t, int NPREDICTED_, int MAX_WINDOW_, int D_PER_CTA,
          int DSTATE>
struct CheckpointingSsuStorage {
  // Re-export the two T/W axis sizes so helpers that only see SmemT can
  // recover them.
  static constexpr int NPREDICTED = NPREDICTED_;
  static constexpr int MAX_WINDOW = MAX_WINDOW_;
  // Swizzle atom width for input_t (= 64 cols for 2-byte types).
  static constexpr int D_SMEM_COLS = next_multiple_of<SmemSwizzle<input_t>::ATOM_COLS>(D_PER_CTA);
  // M-dim of the output MMAs (C, x, z, CB_scaled): always m16-tiled (keyed
  // off NPREDICTED, the new-tokens count).
  static constexpr int NPREDICTED_PAD_MMA_M = next_multiple_of<MMA_prop::M>(NPREDICTED);
  // N-dim of the precompute-CB MMA (matmul-1: C @ B^T).  B's row count is
  // the matmul N-axis → padded to MMA::N=8.  When NPREDICTED ≤ 8 only warp
  // 0 has valid B rows; warp 1 zero-fills its CB slice.
  static constexpr int NPREDICTED_PAD_MMA_N = next_multiple_of<MMA_prop::N>(NPREDICTED);
  // K-dim of the replay MMA (matmul-2: old_x^T @ dB_scaled).  Padded to
  // the small atom's K (== the LDSM unit for 2-byte elements).  When
  // MAX_WINDOW ≤ MMA::K_SMALL=8, replay picks the small atom (1 K-tile,
  // smaller smem, +1 CTA/SM occupancy); otherwise the big atom.  Assumes
  // MAX_WINDOW ≤ MMA::K_BIG (asserted in the wrapper).
  static constexpr int MAX_WINDOW_PAD_MMA_K = next_multiple_of<MMA_prop::K_SMALL>(MAX_WINDOW);
  // Row count for buffers padded only to the input-type swizzle atom's row
  // extent (8 for 2-byte, 4 for 4-byte) — used by C and z, which alias the
  // second m-tile back onto the first via `make_aliased_swizzled_layout_rc`.
  // Keyed off NPREDICTED.
  static constexpr int NPREDICTED_SWIZZLE_R =
      next_multiple_of<SmemSwizzle<input_t>::ATOM_ROWS>(NPREDICTED);

  // All 2D smem buffers below are stored as flat 1D arrays — the actual
  // physical layout is determined by `make_swizzled_layout_rc<...>` at each
  // access site, which scrambles (row, col) → physical offset via the
  // Swizzle XOR.  Declaring them as `T[ROWS][COLS]` would falsely suggest a
  // row-major C-array layout that nobody ever uses; the only thing that
  // matters here is total byte count and 16-byte alignment.

  // CB_scaled — logical (NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE) Swizzle<3,3,3>.
  // CB_ROW_STRIDE pads each row to one bank cycle (128 B = 32 banks × 4 B)
  // worth of `input_t` so LDSM reads in matmul-4's A operand are
  // conflict-free.  Equals the swizzle atom's col extent for `input_t`
  // (64 for 2-byte, 32 for 4-byte).  Logical CB matrix is
  // (NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M); trailing cols are padding.
  static constexpr int CB_ROW_STRIDE = SmemSwizzle<input_t>::ATOM_COLS;
  alignas(16) input_t CB_scaled[NPREDICTED_PAD_MMA_M * CB_ROW_STRIDE];

  // B — logical (NPREDICTED_PAD_MMA_N, DSTATE).  Row count is matmul-1's
  // N-axis (since matmul-1 = C @ B^T).  Padding rows inside [NPREDICTED,
  // NPREDICTED_PAD_MMA_N) contain garbage — valid output uses only
  // [0, NPREDICTED).  Warp-1 of compute_CB_scaled_2warp reads rows ≥ 8 of a
  // 16-row view; those reads spill into C/old_B smem but are masked to 0 by
  // the (j < NPREDICTED) CB-store predicate since j ≥ 8 ≥ NPREDICTED when
  // NPREDICTED_PAD_MMA_N == 8.
  alignas(16) input_t B[NPREDICTED_PAD_MMA_N * DSTATE];

  // C — physical (next_multiple_of<ATOM_ROWS>(NPREDICTED), DSTATE).  Padded
  // only to the swizzle atom's row extent (8 for 2-byte, 4 for 4-byte), not
  // to MMA_prop::M=16.  cp.async writes to this exact extent (CShape's first
  // dim shrunk to match — see load_data).  The MMA still views it as
  // NPREDICTED_PAD_MMA_M=16 rows via `make_aliased_swizzled_layout_rc`,
  // which aliases the second m-tile back onto the first via stride-0
  // row-tile mode.  Garbage feeds output rows ≥ NPREDICTED — predicated
  // out at gmem store.  Saves up to 2 KB of smem at NPREDICTED ≤ ATOM_ROWS,
  // no-op when NPREDICTED > ATOM_ROWS.
  alignas(16) input_t C[NPREDICTED_SWIZZLE_R * DSTATE];

  // x — logical (NPREDICTED_PAD_MMA_M, D_SMEM_COLS).  Cols padded to
  // D_SMEM_COLS for swizzle atom alignment; cp.async only fills cols
  // [0, D_PER_CTA), the tail is unused.
  alignas(16) input_t x[NPREDICTED_PAD_MMA_M * D_SMEM_COLS];

  // z — physical (next_multiple_of<ATOM_ROWS>(NPREDICTED), D_SMEM_COLS).
  // Padded only to the swizzle atom's row extent (8 for 2-byte, 4 for
  // 4-byte), not to MMA_prop::M=16.  z is never an MMA operand — the
  // z-gating epilogue reads it via `partition_C` of the m16n8 c-frag, so
  // the MMA still views it as NPREDICTED_PAD_MMA_M=16 rows via
  // `make_aliased_swizzled_layout_rc`, which aliases the second m-tile back
  // onto the first via stride-0 row-tile mode.  Garbage feeds output rows
  // ≥ NPREDICTED — predicated out at gmem store.  Saves up to 1 KB of smem
  // at NPREDICTED ≤ ATOM_ROWS, no-op when NPREDICTED > ATOM_ROWS.
  alignas(16) input_t z[NPREDICTED_SWIZZLE_R * D_SMEM_COLS];

  // Old cache data loaded in Phase 0 (consumed in Phase 1 replay).
  // old_x — logical (MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS); ldmatrix.trans
  // feeds replay MMA A-operand (only the first D_PER_CTA cols are valid
  // data).
  alignas(16) input_t old_x[MAX_WINDOW_PAD_MMA_K * D_SMEM_COLS];

  // old_B — logical (MAX_WINDOW_PAD_MMA_K, DSTATE) Swizzle<3,3,3>.  Replay
  // MMA reads via ldmatrix.trans (LDSM_T) + register scaling.  Padding
  // rows zero-filled via cp.async ZFILL.
  alignas(16) input_t old_B[MAX_WINDOW_PAD_MMA_K * DSTATE];

  float old_dt[MAX_WINDOW];
  float old_cumAdt[MAX_WINDOW];

  // Processed dt for new tokens (Phase 1a uses this for CB_scaled + cumAdt)
  float dt_proc[NPREDICTED];

  // Cumulative A*dt — computed once by warp 0, read by all warps after sync
  float cumAdt[NPREDICTED];

  // state — logical (D_PER_CTA, DSTATE) in `state_t` (native dtype).  The
  // MMA path reinterprets 2-byte state as bf16 for LDSM; f32 state is loaded
  // via UniversalCopy and converted to bf16 in registers inside
  // add_init_out.
  alignas(16) state_t state[D_PER_CTA * DSTATE];
};

// =============================================================================
// Stochastic-round one fp32 pair to a packed f16x2 u32 with amortized philox
// refresh.  rand_idx[4] is mutated in place every 4th call (when pair_idx & 3
// == 0): a single philox_randint4x feeds 4 consecutive cvt_rs calls, then
// gets refreshed.  Each refresh uses a per-lane unique `philox_off` so the
// generated randints don't collide across threads.  Triton bit-equality is
// intentionally given up here; unbiasedness still holds since each pair's
// cvt_rs gets its own dedicated 32-bit randint.
// =============================================================================
template <int PHILOX_ROUNDS>
__device__ __forceinline__ uint32_t
stochastic_round_pair_with_philox_refresh(float a, float b, int pair_idx, int64_t rand_seed,
                                          int64_t philox_off, uint32_t (&rand_idx)[4]) {
  int const rand_pos = pair_idx & 3;
  if (rand_pos == 0) {
    conversion::philox_randint4x<PHILOX_ROUNDS>(rand_seed, philox_off, rand_idx[0], rand_idx[1],
                                                rand_idx[2], rand_idx[3]);
  }
  return conversion::cvt_rs_f16x2_f32(a, b, rand_idx[rand_pos]);
}

// =============================================================================
// Cross-pass shfl_xor + STG.64 state writeback.
//
// Given two passes' worth of post-cvt_rs packed u32s buffered in `my_packed`
// (pass-0 in [0][:], pass-1 in [1][:]), exchange via shfl_xor across lane^1
// neighbors so that all 32 lanes can issue ONE STG.64 each per pair iter:
//   - even lane k stores PASS n0 (cols (k%4)*2..(k%4)*2+3 of warp's n0 slice)
//   - odd  lane k stores PASS n1 (cols (k%4)*2-2..(k%4)*2+1 of warp's n1 slice)
//
// Halves the STG instruction count vs per-pass writeback: 1 STG.64 per pair
// iter covers BOTH passes' data via cross-lane participation.
// =============================================================================
template <int PAIRS_PER_PASS, int N_PER_PASS, int DSTATE, typename state_t, typename IdPart>
__device__ __forceinline__ void exchange_ntile_state_store_global(
    state_t* __restrict__ state_w_base, int np, int lane,
    uint32_t const (&my_packed)[2][PAIRS_PER_PASS], IdPart const& id_part) {
  using namespace cute;
  static_assert(sizeof(state_t) == 2,
                "exchange_ntile_state_store_global requires 2-byte state_t for STG.64 alignment");
  int const n_base_p0 = np * N_PER_PASS;
  int const n_base_p1 = (np + 1) * N_PER_PASS;
#pragma unroll
  for (int p = 0; p < PAIRS_PER_PASS; ++p) {
    int const i = p * 2;
    // xor mask = 1 swaps neighbor lanes: lane 0 <-> lane 1, lane 2 <-> lane 3, ...
    uint32_t const peer_p0 = __shfl_xor_sync(constants::MASK_ALL_LANES, my_packed[0][p], 1);
    uint32_t const peer_p1 = __shfl_xor_sync(constants::MASK_ALL_LANES, my_packed[1][p], 1);

    int const row = get<0>(id_part(i));
    int const col_p0 = get<1>(id_part(i)) + n_base_p0;
    int const col_p1 = get<1>(id_part(i)) + n_base_p1;

    uint64_t combined;
    int32_t gmem_off;
    if ((lane & 1) == 0) {
      // Even lane: store PASS n0 — my (lower col) in low, peer in high.
      combined = static_cast<uint64_t>(my_packed[0][p]) |
                 (static_cast<uint64_t>(peer_p0) << constants::num_bits_uint32);
      gmem_off = row * DSTATE + col_p0;
    } else {
      // Odd lane: store PASS n1 — peer (lower col) in low, my in high.
      // STG addr = gmem[row*DSTATE + (peer's col base)] = col_p1 - 2.
      combined = static_cast<uint64_t>(peer_p1) |
                 (static_cast<uint64_t>(my_packed[1][p]) << constants::num_bits_uint32);
      gmem_off = row * DSTATE + (col_p1 - 2);
    }
    *reinterpret_cast<uint64_t*>(&state_w_base[gmem_off]) = combined;
  }
}

// =============================================================================
// Phase 1b: Replay — tensor-core MMA path (matmul 2: state recurrence).
// state[D, dstate] = state * total_decay + old_x^T @ (coeff * old_B)
// All 128 threads cooperate.
//
// Warps along N=DSTATE:
//   TiledMMA uses Layout<_1, _4> — per pass covers (M=DIM, N=4×MMA_prop::N=32).
//   Each warp owns: full M (DIM/16 m-atoms) and one n-atom of 8 cols.
//   Why: A is small (DIM × K), B is bigger (DSTATE × K).  M-split (`_4×1`)
//   redundantly loaded full B from each warp (4× × 4 KB = 16 KB).  N-split
//   (`_1×4`) instead redundantly loads full A (4× × 2 KB = 8 KB) and reads
//   B disjointly across warps — net smem read drops 18 KB → 12 KB per replay
//   (~33%) at K_BIG.  Also unlocks D-split D_PER_CTA < 64.
// =============================================================================
// state_w_base (f16+philox path): pre-offset gmem pointer to this CTA's owned
// [D_PER_CTA, DSTATE] state slice (params.state + cache_slot *
// state_stride_seq + head * DIM*DSTATE + d_tile * D_PER_CTA*DSTATE).
// Computed in the kernel preamble.  Combining base + offset into one i64
// pointer drops the cross-iter live-range cost from 4 regs (state_w ptr +
// state_gmem_off) to 2 regs (just the base), and the per-pair STG.32 uses an
// i32 element offset inside the chunk.  Use this instead of separately
// holding params.state-ptr and state_gmem_off.
template <typename input_t, typename state_t, int DIM, int D_PER_CTA, int DSTATE, int PHILOX_ROUNDS,
          typename SmemT>
__device__ __forceinline__ void replay_state_mma(SmemT& smem, CheckpointingSsuParams const& params,
                                                 int warp, int lane, int prev_k, int d_tile,
                                                 int64_t state_ptr_offset, state_t* state_w_base,
                                                 int64_t rand_seed, bool must_checkpoint) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "replay_state_mma requires 2-byte input type");
  static_assert(D_PER_CTA % 16 == 0, "D_PER_CTA must be divisible by 16 (m16n8 atom)");
  static_assert(D_PER_CTA >= 16, "D_PER_CTA must be at least 16");

  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;  // 8 or 16
  int const tid = warp * warpSize + lane;

  // Atom K matches the cache-window tile (MAX_WINDOW_PAD_MMA_K).
  //   K == MMA_prop::K_BIG   (16) → m16n8k16 + x4/x2 ldmatrix.trans
  //   K == MMA_prop::K_SMALL (8)  → m16n8k8  + x2/x1 ldmatrix.trans
  using MmaAtomType = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, MMA_prop::AtomK16,
                                         MMA_prop::AtomK8>;
  using LdsmA = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x8_LDSM_T,
                                   SM75_U16x4_LDSM_T>;
  using LdsmB = std::conditional_t<MAX_WINDOW_PAD_MMA_K == MMA_prop::K_BIG, SM75_U16x4_LDSM_T,
                                   SM75_U16x2_LDSM_T>;

  // 4 warps along N=DSTATE; each warp covers full M (D_PER_CTA/16 m-atoms).
  auto tiled_mma = make_tiled_mma(MMA_Atom<MMA_Traits<MmaAtomType>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // Per-pass output tile is (D_PER_CTA, N_PER_PASS).  N_PER_PASS = 4 warps × n8 = 32 cols.
  constexpr int N_PER_PASS = 4 * MMA_prop::N;
  static_assert(DSTATE % N_PER_PASS == 0,
                "DSTATE must be divisible by 4 * MMA_prop::N for _1x4 warp layout");
  constexpr int NUM_N_PASSES = DSTATE / N_PER_PASS;

  float total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float total_decay = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;

  // ── A operand: old_x [MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS] Swizzle<3,3,3>, transposed
  // view [M=D_SMEM_COLS, K=MAX_WINDOW_PAD_MMA_K].  D_SMEM_COLS may be padded above
  // D_PER_CTA when D_PER_CTA < swizzle atom; local_tile to D_PER_CTA
  // restricts the LDSM to the valid sub-tile.  Each warp loads the FULL M (4×
  // redundant across warps).  See header comment for traffic accounting. ──
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  auto layout_A_full =
      make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, D_SMEM_COLS>();
  Tensor smem_A_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_x)), layout_A_full);
  Tensor smem_A = local_tile(smem_A_full, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                             make_coord(_0{}, _0{}));

  auto s2r_A = make_tiled_copy_A(Copy_Atom<LdsmA, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  Tensor smem_A_s2r = s2r_thr_A.partition_S(smem_A);
  Tensor frag_A = thr_mma.partition_fragment_A(make_tensor(
      (MMA_prop::operand_t*)0x0, make_shape(Int<D_PER_CTA>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
  Tensor frag_A_view = s2r_thr_A.retile_D(frag_A);

  cute::copy(s2r_A, smem_A_s2r, frag_A_view);
  // old_x is input_t == MMA_prop::operand_t (bf16) — no conversion needed.

  // ── B operand: old_B [MAX_WINDOW_PAD_MMA_K, DSTATE] swizzled, transposed view
  // [N=DSTATE, K=MAX_WINDOW_PAD_MMA_K].  Per pass loads N_PER_PASS=32 cols across
  // 4 warps; partition_S splits — each warp gets its disjoint 8-col slice. ──
  auto layout_B = make_swizzled_layout_rc_transpose<input_t, MAX_WINDOW_PAD_MMA_K, DSTATE>();
  Tensor smem_B_full = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.old_B)), layout_B);

  auto s2r_B = make_tiled_copy_B(Copy_Atom<LdsmB, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);

  // ── State: per-CTA swizzle layout [D_PER_CTA, DSTATE]. ──
  auto layout_state_swz = make_swizzled_layout_rc<state_t, D_PER_CTA, DSTATE>();
  state_t* state_base = reinterpret_cast<state_t*>(smem.state);

  // ── Per-pass identity for (row, col) coords ──
  // partition_C of an identity tensor of the per-pass output shape gives this
  // thread's (row, col) at every C-frag position, including warp-N offset.
  // Frag size per thread = (M_atoms=D_PER_CTA/16) × (N_atoms_per_warp=1) × 4 elts.
  auto id_tile = make_identity_tensor(make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  // Linear order from CuTe's column-major partition_C with m16n8 atom:
  //   i=0,1: same row (= row_lo of M-atom 0), adjacent cols (col_off, col_off+1)
  //   i=2,3: same row (= row_hi of M-atom 0), adjacent cols
  //   i=4,5: same row (= row_lo of M-atom 1)
  //   ... (V index 0..3 inside each m16n8, then M-atoms in M-major order)
  // Pair load at (i, i+1) covers two consecutive bf16 elts → one 32-bit LDS.

  // Precompute dB coefficients once — depend only on K (lane), not on N.
  constexpr int LANES_PER_N_COL = warpSize / MMA_prop::N;  // = 4 for m16n8k_
  constexpr int DB_COEFFS_PER_LANE = MAX_WINDOW_PAD_MMA_K / LANES_PER_N_COL;
  float dB_coeff[DB_COEFFS_PER_LANE];
  precompute_dB_coeff<DB_COEFFS_PER_LANE>(dB_coeff, smem, total_cumAdt, prev_k, lane);

  using pair_t = Pair<state_t>;

  // Philox state amortized across 4 consecutive pair conversions: each call
  // returns 4 randints, all 4 get consumed before the next refresh (vs. 1-of-4
  // in the Triton-bit-equal layout — see writeback loop below).  Compile-time
  // pair_idx (n-loop and i-loop both unrolled) keeps `rand_idx[pair_idx & 3]`
  // as a known register access — no local-memory spill.
  constexpr bool kPhiloxF16 = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;
  [[maybe_unused]] uint32_t rand_idx[4];
  // state_w_base is the pre-combined (params.state + state_gmem_off) base
  // pointer — see the function header.  No separate state_w / state_gmem_off
  // alive in this scope.

  // ── Vectorized state writeback (cross-pass STG.64 fusion) ──────────
  // smem always gets nearest-even f32→state_t (consumed by matmul 3 — must
  // match Triton's f32→bf16 path as closely as possible).  Gmem cache, when
  // PHILOX_ROUNDS > 0 and state_t == __half, gets PTX cvt.rs.f16x2.f32
  // stochastic rounding direct from registers via cross-pass STG.64; the
  // smem→gmem `store_state` is gated off in compute_and_store_output.
  //
  // Cross-pass STG fusion: do PASS n0 and PASS n1 back-to-back, buffering
  // the post-cvt_rs packed u32s of n0 across n1's HMMA + cvt_rs.  Then issue
  // ONE STG.64 instruction per pair iter, all 32 lanes active:
  //   - even lane stores PASS n0 data at the warp's n0 column slice
  //   - odd  lane stores PASS n1 data at the warp's n1 column slice
  // Halves the STG instruction count vs per-pass writeback (16 STG.64/thread
  // per 2 passes vs 16 + 16 = 32 STG.64/thread previously — same byte volume).
  //
  // Randint amortization: rand_idx[4] refreshed every 4 pairs; each pair's
  // cvt_rs uses one of the 4 randints.  Triton bit-equality is intentionally
  // given up; unbiasedness still holds.
  constexpr int PAIRS_PER_PASS = D_PER_CTA / 8;  // = (D_PER_CTA/16) × 2 row-pair iters
  static_assert(NUM_N_PASSES % 2 == 0, "Cross-pass STG fusion requires even NUM_N_PASSES");

#pragma unroll
  for (int np = 0; np < NUM_N_PASSES; np += 2) {
    // Buffer of post-cvt_rs packed u32s for both passes (philox path only).
    [[maybe_unused]] uint32_t my_packed[2][PAIRS_PER_PASS];

#pragma unroll
    for (int local_n = 0; local_n < 2; ++local_n) {
      int const n = np + local_n;
      int const n_base = n * N_PER_PASS;

      // ── Allocate per-pass C-frag (4 × M_atoms fp32 elts/thread) ──
      Tensor frag_h = thr_mma.partition_fragment_C(
          make_tensor((float*)0x0, make_shape(Int<D_PER_CTA>{}, Int<N_PER_PASS>{})));

      // ── Load state × total_decay into frag_h. ──
#pragma unroll
      for (int i = 0; i < size(frag_h); i += 2) {
        int const row = get<0>(id_part(i));
        int const col = get<1>(id_part(i)) + n_base;
        int const off = layout_state_swz(row, col);
        pair_t const p = *reinterpret_cast<pair_t const*>(&state_base[off]);
        frag_h(i) = toFloat(p[cute::Int<0>{}]) * total_decay;
        frag_h(i + 1) = toFloat(p[cute::Int<1>{}]) * total_decay;
      }

      // ── LDSM.T per-pass B (per warp = 1 atom of 8 cols of N) ──
      Tensor smem_B_n =
          local_tile(smem_B_full, make_tile(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{}),
                     make_coord(n, _0{}));
      auto smem_B_s2r_n = s2r_thr_B.partition_S(smem_B_n);

      Tensor frag_B = thr_mma.partition_fragment_B(make_tensor(
          (MMA_prop::operand_t*)0x0, make_shape(Int<N_PER_PASS>{}, Int<MAX_WINDOW_PAD_MMA_K>{})));
      auto frag_B_view = s2r_thr_B.retile_D(frag_B);

      cute::copy(s2r_B, smem_B_s2r_n, frag_B_view);

      compute_dB_scaling<DB_COEFFS_PER_LANE>(frag_B, dB_coeff);

      // ── HMMA: frag_h += frag_A @ frag_B ──
      cute::gemm(tiled_mma, frag_h, frag_A, frag_B, frag_h);

      // ── Smem write (always) + cvt_rs into my_packed (philox path) ──
#pragma unroll
      for (int i = 0; i < size(frag_h); i += 2) {
        int const row = get<0>(id_part(i));
        int const col = get<1>(id_part(i)) + n_base;
        int const off = layout_state_swz(row, col);

        // Smem write — always nearest-even (output's matmul 3 reads this).
        pair_t const q = pack_float2<state_t>(make_float2(frag_h(i), frag_h(i + 1)));
        *reinterpret_cast<pair_t*>(&state_base[off]) = q;

        if constexpr (kPhiloxF16) {
          static_assert(sizeof(state_t) == 2, "STG.64 cooperative path requires 2-byte state_t");
          int const pair_idx = n * PAIRS_PER_PASS + i / 2;
          // Per-lane philox_off is unique per (thread, refresh group) — each
          // pair gets its own randint bits.  Always computed; only consumed
          // by the refresh branch inside the helper.
          int64_t const philox_off =
              state_ptr_offset + (int64_t)(d_tile * D_PER_CTA + row) * DSTATE + col;
          // Buffer the SR'd packed u32 — store happens after BOTH passes.
          my_packed[local_n][i / 2] = stochastic_round_pair_with_philox_refresh<PHILOX_ROUNDS>(
              frag_h(i), frag_h(i + 1), pair_idx, rand_seed, philox_off, rand_idx);
        }
      }
    }

    // ── Cross-pass STG.64: all 32 lanes active. ─────────────────────────
    // m16n8 lane layout: lane k → row k/4, cols (k%4)*2..(k%4)*2+1.  Lanes
    // (2k, 2k+1) hold adjacent col-pairs of the same row.  After shfl_xor,
    // the even/odd lane each has a 4-col contiguous block (in different
    // bit-orders).  Even lane STG.64s the n0-pass block at its own col
    // base; odd lane STG.64s the n1-pass block at the peer's (lower) col
    // — both 8-byte aligned for state_t = f16.
    // Runtime-gated on must_checkpoint: non-checkpoint steps skip the gmem
    // STGs entirely (state HBM remains the prior checkpoint).  The cvt_rs
    // SR + philox refresh above still ran — only the STGs are elided —
    // because skipping them would require routing must_checkpoint into the
    // pair_idx amortization logic, which lives across the n-loop.
    if constexpr (kPhiloxF16) {
      if (must_checkpoint) {
        exchange_ntile_state_store_global<PAIRS_PER_PASS, N_PER_PASS, DSTATE>(
            state_w_base, np, lane, my_packed, id_part);
      }
    }
  }
}

// ── Orchestrator: compute_and_store_output ─────────────────────────────
//     out = (C @ state^T) * decay + CB_scaled @ x + D*x, then z-gate.
//     All operations on register-resident frag_y — no smem round-trip.
//     Result converted f32 → input_t in registers and stored directly to gmem
//     via partition_C of the global output tensor (like CUTLASS sgemm_sm80 epilogue).
template <typename input_t, typename state_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE,
          int NUM_WARPS, int PHILOX_ROUNDS, typename SmemT>
__device__ __forceinline__ void compute_and_store_output(SmemT& smem,
                                                         CheckpointingSsuParams const& params,
                                                         int warp, int lane, int d_tile,
                                                         int64_t out_seq_base, int head,
                                                         int64_t cache_slot, float D_val,
                                                         bool must_checkpoint, int seq_len) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "compute_and_store_output requires 2-byte input type");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA: 128 threads, covers [16, 32] output per step ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── Swizzled smem views ──
  // When D_PER_CTA < swizzle atom (= 64 for bf16), the underlying
  // smem buffer is padded to D_SMEM_COLS so the swizzle layout is well-formed.
  // Per-pass MMA loops only iterate D_PER_CTA / N_TILE tiles → never touch
  // the padded tail.
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;

  // x: swizzled [NPREDICTED_PAD_MMA_M, D_SMEM_COLS]
  auto layout_x_swz = make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x = make_tensor(make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)),
                              layout_x_swz);
  auto layout_x_trans_swz =
      make_swizzled_layout_rc_transpose<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS>();
  Tensor smem_x_trans = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.x)), layout_x_trans_swz);

  // z: aliased swizzled [NPREDICTED_PAD_MMA_M, D_SMEM_COLS] — physical buffer
  // is only next_multiple_of<ATOM_ROWS>(NPREDICTED) rows tall; second m-tile
  // aliases first.  Ghost rows feed predicated-out output rows.
  auto layout_z_swz =
      make_aliased_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, D_SMEM_COLS, NPREDICTED>();
  Tensor smem_z =
      make_tensor(make_smem_ptr(reinterpret_cast<input_t const*>(smem.z)), layout_z_swz);

  // ── S2R copies ──
  auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_A = s2r_A.get_slice(tid);
  auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B = s2r_B.get_slice(tid);
  auto s2r_B_trans =
      make_tiled_copy_B(Copy_Atom<SM75_U16x2_LDSM_T, MMA_prop::operand_t>{}, tiled_mma);
  auto s2r_thr_B_trans = s2r_B_trans.get_slice(tid);

  // ── Load CB_scaled A operand from smem (precomputed by warps 0,1 between syncs) ──
  // Row stride matches the buffer's padded width (one swizzle atom of `input_t`).
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  auto layout_cb_swz =
      make_swizzled_layout_rc<input_t, NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M, CB_ROW_STRIDE>();
  Tensor smem_CB = make_tensor(
      make_smem_ptr(reinterpret_cast<MMA_prop::operand_t const*>(smem.CB_scaled)), layout_cb_swz);
  auto smem_CB_s2r = s2r_thr_A.partition_S(smem_CB);
  Tensor frag_CB_A = thr_mma.partition_fragment_A(smem_CB);
  auto frag_CB_A_view = s2r_thr_A.retile_D(frag_CB_A);
  cute::copy(s2r_A, smem_CB_s2r, frag_CB_A_view);

  // Decay broadcast: cumAdt[t] → [NPREDICTED_PAD_MMA_M, N_TILE] with stride-0 on N.
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(smem.cumAdt),
      make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  // ── Gmem output: partition_C for direct register → gmem store ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  // out_base lands on this CTA's D-slice within the head.
  int64_t const out_base = out_seq_base + (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;

  // Row predicate for padding.  The epilogue store loop iterates i in steps
  // of 2 and only consults pred(0) and pred(2) — m16n8k16 C-frag per thread
  // has 4 elts at rows {t/4, t/4, t/4+8, t/4+8}, so there are only 2 unique
  // row predicates.  Compute them once and skip the 4-wide pred tensor.
  auto id_tile = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  bool const pred_row_lo = get<0>(id_part(0)) < seq_len;
  bool const pred_row_hi = get<0>(id_part(2)) < seq_len;

  // Number of output N-tiles per pass = D_PER_CTA / N_TILE.
  // D_SPLIT=1, D_PER_CTA=64, N_TILE=32 → NUM_N_TILES = 2 (current behavior).
  // D_SPLIT=2, D_PER_CTA=32                → NUM_N_TILES = 1 (uses _n1 variant).
  constexpr int NUM_N_TILES = D_PER_CTA / N_TILE;
  static_assert(NUM_N_TILES == 1 || NUM_N_TILES == 2,
                "Output epilogue supports NUM_N_TILES = D_PER_CTA / N_TILE in {1, 2}");

  // ── Epilogue lambda (defined once; called per N-tile from each branch) ──
  auto epilogue = [&](auto& frag_y, int n) {
  // Decay: frag_y *= exp(cumAdt[t])
#pragma unroll
    for (int i = 0; i < size(frag_y); ++i) {
      frag_y(i) *= __expf(decay_part(i));
    }

    // frag_y += CB_scaled @ x (CB from smem LDSM, x from smem via ldmatrix.trans)
    add_cb_x<input_t, MMA_prop::operand_t, N_TILE, NPREDICTED_PAD_MMA_M>(
        frag_y, frag_CB_A, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);

    // frag_y += D * x[t, d]
    add_D_skip<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);

    // frag_y *= z * sigmoid(z)
    compute_z_gating<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);

    // Store frag_y directly to gmem (register → gmem, no smem round-trip).
    auto gOut_tile = make_tensor(make_gmem_ptr(output_ptr + out_base + n * N_TILE),
                                 make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}),
                                             make_stride(params.out_stride_token, _1{})));
    auto gOut_part = thr_mma.partition_C(gOut_tile);
    // Vectorized pair store: elements i and i+1 are same-row, consecutive columns
    // in the m16n8k16 partition_C layout, so &gOut_part(i+1) == &gOut_part(i) + 1.
    // Address is naturally aligned to sizeof(Pair<input_t>) since MMA column
    // index = (lane%4)*2 → even.  pack_float2 dispatches to the native packed
    // cvt (e.g. cvt.rn.bf16x2.f32 for bf16) — one instruction for the pair.
#pragma unroll
    for (int i = 0; i < size(frag_y); i += 2) {
      // Bit 1 of i toggles between the two row groups of the m16n8k16
      // C-frag: i∈{0,1} → row t/4, i∈{2,3} → row t/4+8 (repeats per M-atom).
      bool const pred_i = (i & 2) ? pred_row_hi : pred_row_lo;
      if (pred_i) {
        *reinterpret_cast<Pair<input_t>*>(&gOut_part(i)) =
            pack_float2<input_t>(make_float2(frag_y(i), frag_y(i + 1)));
      }
    }
  };

  // Skip the smem→gmem state copy when philox+f16: `replay_state_mma`
  // already did the gmem store with stochastic rounding direct from registers.
  constexpr bool kSkipSmemToGmemState = (PHILOX_ROUNDS > 0) && std::is_same_v<state_t, __half>;

  // ── Matmul 3 + store_state + epilogue, dispatching on NUM_N_TILES ──
  // (NumNTiles is deduced from the variadic frag_y... pack in `add_init_out`.)
  if constexpr (NUM_N_TILES == 2) {
    Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
    Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
    add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0,
                                                      frag_y_1);
    // State writeback hoisted here — after matmul 3 has finished consuming
    // smem.state, before matmul 4 which reads only smem.x / smem.CB_scaled
    // / smem.z.  STGs fire-and-forget alongside the epilogue (matmul 4 +
    // D*x + z-gate + output STG).  Runtime-gated on must_checkpoint:
    // non-checkpoint steps leave the prior state HBM intact (saving
    // bandwidth — that's the perf win of the checkpointing design).
    if constexpr (!kSkipSmemToGmemState) {
      if (must_checkpoint) {
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot);
      }
    }
    epilogue(frag_y_0, 0);
    epilogue(frag_y_1, 1);
  } else {  // NUM_N_TILES == 1 (D_SPLIT = 2 path)
    Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
    add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0);
    // No sync needed before store_state: the post-replay __syncthreads()
    // in the kernel already established cross-warp visibility of replay's
    // writes to smem.state, and nothing after that point writes to it
    // (add_init_out is read-only on smem.state).
    if constexpr (!kSkipSmemToGmemState) {
      if (must_checkpoint) {
        store_state<state_t, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(smem, params, warp, lane, d_tile,
                                                                head, cache_slot);
      }
    }
    epilogue(frag_y_0, 0);
  }
}

// ── Orchestrator: compute_no_write_output (must_checkpoint == false path) ──
//   Skips the replay matmul entirely.  smem.state still holds s_0 after Phase 0,
//   so matmul-3 via add_init_out computes u^T = C @ s_0^T directly.
//
//   y[t, d] = β(t) · u[t, d]
//           + Σ_{j<NPREDICTED}  CB_scaled[t, j] · x[j, d]            (matmul-4-new)
//           + Σ_{i<prev_k}      CB_old   [t, i] · old_x[i, d]        (matmul-4-old, NEW)
//           + D[d] · x[t, d]   (+ z gating)
//
//   β(t) = exp(total_old_cumAdt + cumAdt[t]) where total_old_cumAdt =
//          smem.old_cumAdt[prev_k − 1]  (= 0 when prev_k == 0).
//
//   CB_old is populated by `compute_CB_old_2warp` on warps 2,3 before the
//   `__syncthreads()` in the no-write dispatcher.  It lives in the CB_scaled
//   buffer at cols [NPREDICTED_PAD_MMA_M, NPREDICTED_PAD_MMA_M + MAX_WINDOW_PAD_MMA_K).
template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void compute_no_write_output(
    SmemT& smem, CheckpointingSsuParams const& params, int warp, int lane, int prev_k, int d_tile,
    int64_t out_seq_base, int head, int64_t cache_slot, float D_val, int seq_len) {
  using namespace cute;
  static_assert(sizeof(input_t) == 2, "compute_no_write_output requires 2-byte input type");

  constexpr int NPREDICTED_PAD_MMA_M = SmemT::NPREDICTED_PAD_MMA_M;
  constexpr int MAX_WINDOW_PAD_MMA_K = SmemT::MAX_WINDOW_PAD_MMA_K;
  constexpr int D_SMEM_COLS = SmemT::D_SMEM_COLS;
  constexpr int CB_ROW_STRIDE = SmemT::CB_ROW_STRIDE;
  int const tid = warp * warpSize + lane;

  // ── TiledMMA for matmul-3 + matmul-4-new (K=NPREDICTED_PAD_MMA_M=16 fits K_BIG). ──
  auto tiled_mma =
      make_tiled_mma(MMA_Atom<MMA_Traits<MMA_prop::AtomK16>>{}, Layout<Shape<_1, _4>>{});
  auto thr_mma = tiled_mma.get_slice(tid);

  // ── TiledMMA for matmul-4-old: K = MAX_WINDOW_PAD_MMA_K ∈ {8, 16} → atom dispatch. ──
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

  // ── S2R copies (matmul-4-old, K-dispatched atoms) ──
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
  // Use the full physical (T_pad, CB_ROW_STRIDE) padded swizzle view — byte-
  // compatible with both the CB_scaled (T_pad, T_pad, CB_ROW_STRIDE) write
  // layout and compute_CB_old_2warp's wide write layout (inner offset
  // r*CB_ROW_STRIDE + c is identical across the three views).
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

  // ── Decay broadcast: cumAdt[t] (per-T scalar) with stride-0 on N. ──
  constexpr int N_TILE = cute::tile_size<1>(decltype(tiled_mma){});
  Tensor decay_bcast = make_tensor(
      make_smem_ptr(smem.cumAdt),
      make_layout(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}), make_stride(_1{}, _0{})));
  Tensor decay_part = thr_mma.partition_C(decay_bcast);

  // ── β extra factor: exp(total_old_cumAdt) — uniform constant across (t, d) ──
  float const total_old_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
  float const beta_extra = __expf(total_old_cumAdt);

  // ── Gmem output base ──
  auto* __restrict__ output_ptr = reinterpret_cast<input_t*>(params.output);
  int64_t const out_base = out_seq_base + (int64_t)head * DIM + (int64_t)d_tile * D_PER_CTA;

  // ── Row predicate (same pattern as compute_and_store_output) ──
  auto id_tile = make_identity_tensor(make_shape(Int<NPREDICTED_PAD_MMA_M>{}, Int<N_TILE>{}));
  auto id_part = thr_mma.partition_C(id_tile);
  bool const pred_row_lo = get<0>(id_part(0)) < seq_len;
  bool const pred_row_hi = get<0>(id_part(2)) < seq_len;

  constexpr int NUM_N_TILES = D_PER_CTA / N_TILE;
  static_assert(NUM_N_TILES == 1 || NUM_N_TILES == 2,
                "compute_no_write_output supports NUM_N_TILES in {1, 2}");

  // ── Epilogue per N-tile ──
  auto epilogue = [&](auto& frag_y, int n) {
  // 1. β-scale: frag_y(t, d) *= exp(total_old_cumAdt + cumAdt[t]).
  //    Matmul-3 produced u^T = C @ s_0^T; this scales the u term by β
  //    BEFORE matmul-4 adds the CB·x and CB_old·old_x contributions.
#pragma unroll
    for (int i = 0; i < size(frag_y); ++i) {
      frag_y(i) *= beta_extra * __expf(decay_part(i));
    }

    // 2. frag_y += CB_scaled @ x  (matmul-4 over new tokens).
    add_cb_x<input_t, MMA_prop::operand_t, N_TILE, NPREDICTED_PAD_MMA_M>(
        frag_y, frag_CB_A, smem_x_trans, s2r_B_trans, s2r_thr_B_trans, thr_mma, tiled_mma, n);

    // 3. frag_y += CB_old @ old_x  (matmul-4 over old tokens — NEW).
    add_cb_old_x<input_t, MMA_prop::operand_t, N_TILE, MAX_WINDOW_PAD_MMA_K>(
        frag_y, frag_CB_old_A, smem_old_x_trans, s2r_B_old_trans, s2r_thr_B_old_trans, thr_mma_old,
        tiled_mma_old, n);

    // 4. frag_y += D · x[t, d].
    add_D_skip<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_x, thr_mma, D_val, n);

    // 5. frag_y *= z · sigmoid(z).
    compute_z_gating<input_t, NPREDICTED_PAD_MMA_M, N_TILE>(frag_y, smem_z, thr_mma, params.z, n);

    // 6. Store frag_y → gmem via partition_C (same pattern as compute_and_store_output).
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

  // ── Matmul-3: frag_y = C @ s_0^T (smem.state retains s_0 since replay skipped) ──
  if constexpr (NUM_N_TILES == 2) {
    Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
    Tensor frag_y_1 = thr_mma.partition_fragment_C(id_tile);
    add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0,
                                                      frag_y_1);
    epilogue(frag_y_0, 0);
    epilogue(frag_y_1, 1);
  } else {  // NUM_N_TILES == 1 (D_SPLIT = 2 path)
    Tensor frag_y_0 = thr_mma.partition_fragment_C(id_tile);
    add_init_out<input_t, state_t, D_PER_CTA, DSTATE>(smem, tiled_mma, thr_mma, tid, frag_y_0);
    epilogue(frag_y_0, 0);
  }
}

// ── Per-path dispatchers (called from checkpointing_ssu_kernel) ──
// ssu_checkpoint: replay → sync → output (today's body).
// ssu_nocheckpoint: sync → no-write output (skips replay).
template <typename input_t, typename state_t, int NPREDICTED, int DIM, int D_PER_CTA, int DSTATE,
          int NUM_WARPS, int PHILOX_ROUNDS, typename SmemT>
__device__ __forceinline__ void ssu_checkpoint(SmemT& smem, CheckpointingSsuParams const& params,
                                               int warp, int lane, int prev_k, int d_tile,
                                               int64_t out_seq_base, int head, int64_t cache_slot,
                                               float D_val, int seq_len) {
  // ── DO NOT HOIST `rand_seed` ── see kernel preamble for the perf rationale.
  int64_t const rand_seed = (PHILOX_ROUNDS > 0) ? *params.rand_seed : 0;
  // `state_ptr_offset` is int64 — matches Triton's `base_rand =
  // cache_batch_idx * stride_state_batch + ...` (cache_batch_idx is .to(int64)).
  // Full 64 bits flow through `philox_randint4x`, which splits low/high
  // across Philox c0/c1.  No collision risk at large serving cache sizes.
  int64_t const state_ptr_offset =
      cache_slot * params.state_stride_seq + (int64_t)head * DIM * DSTATE;
  state_t* const state_w_base = reinterpret_cast<state_t*>(params.state) +
                                cache_slot * params.state_stride_seq +
                                (int64_t)head * DIM * DSTATE + (int64_t)d_tile * D_PER_CTA * DSTATE;
  replay_state_mma<input_t, state_t, DIM, D_PER_CTA, DSTATE, PHILOX_ROUNDS>(
      smem, params, warp, lane, prev_k, d_tile, state_ptr_offset, state_w_base, rand_seed,
      /*must_checkpoint=*/true);

  __syncthreads();

  compute_and_store_output<input_t, state_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS,
                           PHILOX_ROUNDS>(smem, params, warp, lane, d_tile, out_seq_base, head,
                                          cache_slot, D_val, /*must_checkpoint=*/true, seq_len);
}

template <typename input_t, typename state_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int D_PER_CTA, int DSTATE, int NUM_WARPS, typename SmemT>
__device__ __forceinline__ void ssu_nocheckpoint(SmemT& smem, CheckpointingSsuParams const& params,
                                                 int warp, int lane, int prev_k, int d_tile,
                                                 int64_t out_seq_base, int head, int64_t cache_slot,
                                                 float D_val, int seq_len) {
  // Sync makes warps 0,1's CB_scaled writes and warps 2,3's CB_old writes
  // visible to all warps before matmul-4 reads CB_scaled + CB_old.  Also
  // covers smem.x (warp 2-loaded) and smem.z (warp 3-loaded) for Phase 2.
  __syncthreads();

  compute_no_write_output<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE,
                          NUM_WARPS>(smem, params, warp, lane, prev_k, d_tile, out_seq_base, head,
                                     cache_slot, D_val, seq_len);
}

// =============================================================================
// Kernel
// =============================================================================
template <typename input_t, typename dt_t, typename weight_t, typename matrixA_t, typename state_t,
          typename stateIndex_t, typename state_scale_t, int NPREDICTED, int MAX_WINDOW, int DIM,
          int DSTATE, int HEADS_PER_GROUP, int PHILOX_ROUNDS, int NUM_WARPS, int D_SPLIT = 1,
          bool VARLEN = false>
__global__ void checkpointing_ssu_kernel(CheckpointingSsuParams params) {
  // Per-head DIM is sharded across `D_SPLIT` CTAs (D_PER_CTA each).
  static_assert(DIM % D_SPLIT == 0, "DIM must be divisible by D_SPLIT");
  constexpr int D_PER_CTA = DIM / D_SPLIT;
  static_assert(D_PER_CTA >= 32,
                "D_PER_CTA must be >= 32 (output MMA m16n8 with _1×4 warp layout). "
                "D_SPLIT=4 (D_PER_CTA=16) needs warp-count restructure.");
  static_assert(NPREDICTED <= MAX_WINDOW,
                "NPREDICTED must be <= MAX_WINDOW (new tokens must fit in cache)");
  static_assert(MAX_WINDOW <= MMA_prop::K_BIG,
                "MAX_WINDOW must be <= MMA::K_BIG=16 (single replay K-tile assumption)");
  // Cross-check: host launcher must dispatch the template specialization
  // matching the runtime params.d_split it stamped into the struct.
  assert(params.d_split == D_SPLIT);
  using SmemT =
      CheckpointingSsuStorage<input_t, state_t, NPREDICTED, MAX_WINDOW, D_PER_CTA, DSTATE>;
  extern __shared__ __align__(128) char smem_buf[];
  auto& smem = *reinterpret_cast<SmemT*>(smem_buf);

  // Grid layout (D_SPLIT, batch, nheads).
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

  // ── Double-buffer index ──
  auto const* __restrict__ buf_idx_ptr = reinterpret_cast<int32_t const*>(params.cache_buf_idx);
  int const buf_read = __ldg(&buf_idx_ptr[cache_slot]);

  // ── prev_num_accepted_tokens ──
  auto const* __restrict__ prev_ptr = reinterpret_cast<int32_t const*>(params.prev_num_accepted);
  int const prev_k = prev_ptr[cache_slot];

  // ── Varlen vs non-varlen prologue.  See checkpointing_ssu_kernel_8bit for
  // the rationale: `seq_len` flows downstream as a constexpr-foldable
  // NPREDICTED in non-varlen, runtime int in varlen.
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
  // x/B/C bases are computed inside `load_post_pdl_wait_data` from `outer`
  // so the products don't get pinned in registers across `gdc_wait` (asm
  // volatile blocks rematerialization; cost was ~6 extra regs).  dt/z bases
  // are only consumed pre-wait, and out_base only post-replay — fine to
  // precompute.
  int64_t const dt_seq_base = outer * params.dt_stride_seq + head;
  int64_t const z_seq_base = outer * params.z_stride_seq;
  int64_t const out_seq_base = outer * params.out_stride_seq;

  // ── Per-CTA implicit checkpoint criterion ──
  // When the new tokens would overflow the cache buffer, we must checkpoint:
  // replay [0, prev_k) into state, write state to HBM, write the new tokens
  // to the **staging** buffer (1 - buf_read) at offset 0.  Otherwise, we
  // append the new tokens to the **active** buffer (buf_read) at offset
  // prev_k and skip the state HBM write entirely.  Cache writes always
  // happen — only their target buffer + offset depends on must_checkpoint.
  bool const must_checkpoint = (prev_k + seq_len > MAX_WINDOW);
  int const buf_write = must_checkpoint ? (1 - buf_read) : buf_read;
  int const write_offset = must_checkpoint ? 0 : prev_k;

  // ── Load A (scalar, tie_hdim), dt_bias, and D (hoisted to hide gmem latency) ──
  auto const* __restrict__ A_ptr = reinterpret_cast<matrixA_t const*>(params.A);
  auto const* __restrict__ dt_bias_ptr = reinterpret_cast<weight_t const*>(params.dt_bias);
  auto const* __restrict__ D_ptr = reinterpret_cast<weight_t const*>(params.D);
  float const A_val = toFloat(A_ptr[head]);
  float const dt_bias_val = dt_bias_ptr ? toFloat(dt_bias_ptr[head]) : 0.f;
  float const D_val = D_ptr ? toFloat(D_ptr[head]) : 0.f;

  // ════════════════════════════════════════════════════════════════════════
  // Phase 0: Load all data into smem (per-warp ownership)
  // ════════════════════════════════════════════════════════════════════════
  // Two-phase load around the PDL barrier:
  //   1. Issue cp.async for cache (state, old_B, old_x) and in_proj-derived
  //      data (z); run scalar LDGs (old_dt, old_cumAdt, dt → dt_proc) and the
  //      cumAdt warp scan.  None of these depend on conv1d, so they overlap
  //      with the upstream's tail.
  //   2. `gdc_wait()` — wait for the upstream conv1d to signal (no-op when
  //      the kernel isn't launched with the PDL attribute).
  //   3. Issue cp.async for conv1d outputs (x, B, C), then __pipeline_commit
  //      + __pipeline_wait_prior(0) + __syncwarp drains BOTH halves' cp.async
  //      (they share the per-thread async group).
  //
  // Each warp sees its own cp.async via __syncwarp.  Cross-warp visibility
  // is established by the post-replay __syncthreads below — replay reads of
  // state are safe because (a) replay's frag_h initial load sees only the
  // current warp's lane positions, and (b) the actual _1×4 cross-warp
  // dependency is on writes that haven't happened yet at this point.
  // ENABLE_PDL is JIT-stamped (see checkpointing_ssu_customize_config.jinja).
  // `if constexpr` keeps only the chosen branch in the binary — no register
  // pressure leak from the unused path.
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

  // old_B writeback hoisted ahead of Phase 1.  Source (smem.B) is consumed
  // only by Phase 1a CB; the STGs fire-and-forget onto the memory subsystem
  // and complete in parallel with all subsequent compute.  Only W0, W1 hold
  // valid smem.B at this point (they're the ones that cp.async'd B).  Gate
  // accordingly — store halves its thread count but B is small (4 KB) so
  // still cheap.  old_B is D-independent (per-group, full DSTATE) — only
  // d_tile == 0 writes; other d_tiles would emit identical payloads.
  if (d_tile == 0 && warp < 2) {
    store_old_B<input_t, NPREDICTED, DSTATE, HEADS_PER_GROUP>(
        smem, params, warp, lane, head, group_idx, cache_slot, buf_write, write_offset, seq_len);
  }

  // CB precompute (4-warp split): warps 0,1 compute CB_scaled (new tokens);
  // warps 2,3 compute CB_old (old tokens) in the no-write path only.  Both
  // halves write to disjoint col ranges of the same swizzled smem.CB_scaled
  // buffer.  In the checkpoint path, warps 2,3 stay idle here and pick up
  // work below in `ssu_checkpoint`'s replay matmul.
  if (warp < 2) {
    compute_CB_scaled_2warp<input_t, NPREDICTED, DSTATE>(smem, warp, lane, seq_len);
  } else if (!must_checkpoint) {
    compute_CB_old_2warp<input_t, NPREDICTED, MAX_WINDOW, DSTATE>(smem, warp, lane, prev_k,
                                                                  seq_len);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Phase 1b + 2: Per-path dispatch
  // ════════════════════════════════════════════════════════════════════════
  // Checkpoint path: replay + sync + compute_and_store_output (today's body).
  // No-write path : sync + compute_no_write_output (skips replay; matmul-3
  //                 reads s_0 directly from smem.state, matmul-4 extends with
  //                 the CB_old @ old_x contribution over [0, prev_k)).
  // must_checkpoint is uniform across the CTA (derived from broadcast prev_k
  // + compile-time NPREDICTED + MAX_WINDOW), so both branches contain a
  // __syncthreads and divergence is balanced.
  if (must_checkpoint) {
    ssu_checkpoint<input_t, state_t, NPREDICTED, DIM, D_PER_CTA, DSTATE, NUM_WARPS, PHILOX_ROUNDS>(
        smem, params, warp, lane, prev_k, d_tile, out_seq_base, head, cache_slot, D_val, seq_len);
  } else {
    ssu_nocheckpoint<input_t, state_t, NPREDICTED, MAX_WINDOW, DIM, D_PER_CTA, DSTATE, NUM_WARPS>(
        smem, params, warp, lane, prev_k, d_tile, out_seq_base, head, cache_slot, D_val, seq_len);
  }

  // ── PDL: signal downstream that `output` is written.  The cache writes
  // below target tensors that only the next SSU step reads, not the
  // immediate downstream kernel, so we can signal before issuing them.
  if constexpr (ENABLE_PDL) {
    gdc_launch_dependents();
  }

  // ── Phase 3: Store to global memory ──
  // (old_B hoisted to pre-Phase-1; state hoisted into compute_and_store_output.)

  // Cache writes — old_x uses all warps (vectorized), dt/cumAdt one warp each.
  // Each writes the new NPREDICTED tokens at gmem offset `write_offset` into
  // buffer `buf_write` (computed above from must_checkpoint).
  store_old_x<input_t, NPREDICTED, DIM, D_PER_CTA>(smem, params, warp, lane, d_tile, head,
                                                   cache_slot, write_offset, seq_len);
  // dt_proc / cumAdt are D-independent — only d_tile == 0 writes.
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

#endif  // FLASHINFER_MAMBA_KERNEL_CHECKPOINTING_SSU_CUH_
