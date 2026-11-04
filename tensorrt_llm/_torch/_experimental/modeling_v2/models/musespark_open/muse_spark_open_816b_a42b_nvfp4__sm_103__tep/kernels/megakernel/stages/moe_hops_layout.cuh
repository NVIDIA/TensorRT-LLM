/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The MoE-side hops of a decoder layer (stages/moe_hops.cuh): shapes, the per-copy layout of the multicast workspace
// and of the stages' local arena, the clear-unit geometry. Included inside namespace mso_mk by msoMegakernel.cu, in
// both compilation passes (the C ABI reports these values; moe_hops.py mirrors and checks them).
//
// Multicast workspace (MoeHopWorkspace, one McastGPUBuffer per tensor-parallel group of 8), per copy (launch n uses
// copy n % 3), every 32-bit word kEmpty until its producer stores it:
//   hop1    [8 source ranks][8 rows][8192]  BF16  o_proj partials (rank r's producer writes slot r of every rank)
//   pool    [2 K halves][8 rows][4096]      FP32  pool GEMV partials: rank r's columns [512 r, +512) of both halves
//   logits  [256 experts][8 rows]           FP32  router logits: rank r's experts [32 r, +32)
//   fin     [8 source ranks][8 rows][4096]  BF16  the experts' weighted-sum partials (finalize all-reduce input)
//   f1      [8 rows][8192]                  BF16  unpool output: rank r's columns [1024 r, +1024)
// Local arena (per copy, this rank only): x_ffn [8][8192] BF16 (hop1 consumer -> router / pool), m [8][4096] BF16
// (finalize -> unpool), and the four row-norm exchanges [4 phases][8 rows][32 slots] FP32; with MSO_MK_NB also the
// one-exchange slots [2 boundaries][8 rows][32 slots] x 16 B (kMhArAbc).

#pragma once

// The FMA contraction of the boundary arithmetic (stages/moe_hops.cuh: mhSq, mhGated).
#ifndef MSO_MK_MH_FMA
#define MSO_MK_MH_FMA 1
#endif
#if MSO_MK_MH_FMA < 0 || MSO_MK_MH_FMA > 2
#error MSO_MK_MH_FMA must be 0 (no contraction), 1 (fma(a, r, b v)) or 2 (fma(b, v, a r))
#endif

constexpr int kMhRanks = 8;
constexpr int kMhExperts = 256;
constexpr int kMhLocalExperts = kMhExperts / kMhRanks;
constexpr int kMhPoolWidth = 4096;                    // pooled width (expert hidden)
constexpr int kMhPoolLocal = kMhPoolWidth / kMhRanks; // a rank's pool rows (512)
constexpr int kMhUnpoolLocal = 8192 / kMhRanks;       // a rank's unpool rows (1024)
// Row units of the two boundaries (MhH1, MhF1): U per row, each owning kMhRowW columns; thread t's vector j (of
// kMhRowV) covers columns q kMhRowW + 8 (t + 256 j). U = 1: a row's norm reductions stay in one CTA.
#ifndef MSO_MK_MH_ROW_UNITS
#define MSO_MK_MH_ROW_UNITS 4
#endif
#if MSO_MK_MH_ROW_UNITS != 1 && MSO_MK_MH_ROW_UNITS != 2 && MSO_MK_MH_ROW_UNITS != 4
#error MSO_MK_MH_ROW_UNITS must be 1, 2 or 4
#endif
constexpr int kMhRowU = MSO_MK_MH_ROW_UNITS;
// The row-norm exchange between the U > 1 units of a row (stages/moe_hops_xchg.cuh): 0 = every warp polls the row's
// 32 slots in the arena (global memory); 1 (default) = the same slots, warp 0 polls them and passes them on in shared
// memory; 2 = the U units of a row form a thread-block cluster and every unit stores its slots into the shared memory
// of each unit of the row (DSMEM), where they are polled (the launch then has cluster dimension U). At 212 CTAs, one
// per SM, clusters of 2 can all be co-resident on Rubin but clusters of 4 cannot (msoMkLaunch refuses those launches).
#ifndef MSO_MK_MH_XCHG
#define MSO_MK_MH_XCHG 1
#endif
#if MSO_MK_MH_XCHG < 0 || MSO_MK_MH_XCHG > 2
#error MSO_MK_MH_XCHG must be 0 (arena, every warp polls), 1 (arena, one polling warp) or 2 (cluster DSMEM)
#endif
constexpr int kMhXchg = MSO_MK_MH_XCHG;
// The launch's cluster dimension: the row units of one row are cluster ranks 0 .. U - 1 (the plan puts row unit u on
// CTA 160 + u, so a row starts on a multiple of U).
constexpr int kMhClusterDim = (kMhXchg == 2 && kMhRowU > 1) ? kMhRowU : 1;
constexpr int kMhRowW = 8192 / kMhRowU;
constexpr int kMhRowV = kMhRowW / (8 * 256);
constexpr int kMhBoundaryUnits = 8 * kMhRowU; // MhH1 / MhF1 units
constexpr int kMhCoefH1Bytes = kMhRowW * 10;  // per unit: alpha1, beta1 (FP32), w_preffn (BF16)
constexpr int kMhCoefF1Bytes = kMhRowW * 12;  // per unit: alpha2, beta2 (FP32), gain_pf, w_next (BF16)
// The F1 coefficient blob starts with the whole row's gain_pf (BF16 [8192]), then the U unit parts.
constexpr int kMhCoefF1GainBytes = 8192 * 2;
// F1's first norm (MSO_MK_MH_F1FULL = 1, U > 1): every F1 unit polls the whole row of the gathered branch and sums the
// row's 32 stock slots itself (in shared memory), so only the second norm crosses CTAs; its unit weights then add the
// whole row's gain (kMhCoefF1GainBytes after its part).
#ifndef MSO_MK_MH_F1FULL
#define MSO_MK_MH_F1FULL 0
#endif
#if MSO_MK_MH_F1FULL != 0 && MSO_MK_MH_F1FULL != 1
#error MSO_MK_MH_F1FULL must be 0 or 1
#endif
constexpr bool kMhF1Full = MSO_MK_MH_F1FULL == 1 && kMhRowU > 1;
constexpr int kMhF1UnitBytes = kMhCoefF1Bytes + (kMhF1Full ? kMhCoefF1GainBytes : 0);
constexpr int kMhRowUnits = 32;                     // stub / finalize units: 8 rows x 4 column quarters
constexpr int kMhPoolUnits = 2 * kMhPoolLocal / 16; // 32 tiles x 2 K halves
constexpr int kMhUnpoolUnits = kMhUnpoolLocal / 16; // 64 tiles
constexpr int kMhRouterUnits = kMhLocalExperts;     // one expert row each
constexpr int kMhPoolUnitBytes = 64 * 2048;         // 64 chunks of a tile (half of K = 8192)
constexpr int kMhUnpoolUnitBytes = 64 * 2048;       // 64 chunks (all of K = 4096)
constexpr int kMhRouterUnitBytes = 8192 * 4;        // one FP32 row

// Multicast workspace, per copy (bytes).
constexpr int64_t kMhH1Off = 0;
constexpr int64_t kMhH1Bytes = static_cast<int64_t>(kMhRanks) * 8 * 8192 * 2;
constexpr int64_t kMhPoolOff = kMhH1Off + kMhH1Bytes;
constexpr int64_t kMhPoolBytes = 2LL * 8 * kMhPoolWidth * 4;
constexpr int64_t kMhLogitOff = kMhPoolOff + kMhPoolBytes;
constexpr int64_t kMhLogitBytes = static_cast<int64_t>(kMhExperts) * 8 * 4;
constexpr int64_t kMhFinOff = kMhLogitOff + kMhLogitBytes;
constexpr int64_t kMhFinBytes = static_cast<int64_t>(kMhRanks) * 8 * kMhPoolWidth * 2;
constexpr int64_t kMhF1Off = kMhFinOff + kMhFinBytes;
constexpr int64_t kMhF1Bytes = 8LL * 8192 * 2;
constexpr int64_t kMhCopyBytes = kMhF1Off + kMhF1Bytes;
// ---- MSO_MK_FZ: zero-exchange F1 (0 or 1; stages/moe_hops.cuh). Each MhUnpool thread also publishes the stock
// boundary thread partials of its 8 columns (the leaves of the F1 row statistics S1, A, B, C of nb's one exchange) into
// every rank's leaf area, once per reading quarter: [4 quarters][8 rows][1024 leaves] x 16 B (leaf = column / 8), so
// each leaf word has one reader (MhF1 unit (row, quarter) of that rank), which resets it after use, as the F1 area's
// words. MhF1 sums its row's leaves in nb's tree: inv1 / inv2 are nb's bits, with no row exchange. r' of this rank's
// unpool columns reaches the unpool units through the rank-local r' area [8 rows][1024] FP32 (written by MhH1 through
// the unicast mapping, data as its own flag, reset by its one reader). Both lie in a per-copy fz block after the room
// of kMhFzMaxCopies hop copies (at kMhFzMaxCopies * kMhCopyBytes + copy * kMhFzBytes; the workspace holds at most that
// many copies), so the copies keep their layout. ----
// Numerics contract of MSO_MK_FZ (MUSESPARK_OPEN_MEGAKERNEL_FZ, kernels/megakernel/serving.py; unset = the served
// build's level): no numerics of its own on top of MSO_MK_NB bit 2. MhF1 rebuilds nb's four row statistics S1, A, B,
// C from the same stock per-thread leaves in nb's summation tree, so inv1, inv2, r'' and the captures are nb's bits:
// the 8-rank chain test (test_fz_chain) checks every output of check_nb3[_bh3]_fz1 bitwise against check_nb3[_bh3].
// Not in the served build: it is slower.
// The switch is kept because the build is not bitwise the stock layers.
#ifndef MSO_MK_FZ
#define MSO_MK_FZ 0
#endif
#if MSO_MK_FZ != 0 && MSO_MK_FZ != 1
#error MSO_MK_FZ must be 0 or 1
#endif
#if MSO_MK_FZ
constexpr int64_t kMhFzLeafOff = 0;
constexpr int64_t kMhFzLeafBytes = 4LL * 8 * 1024 * 16;
constexpr int64_t kMhFzR1Off = kMhFzLeafOff + kMhFzLeafBytes;
constexpr int64_t kMhFzR1Bytes = 8LL * kMhUnpoolLocal * 4;
constexpr int64_t kMhFzBytes = kMhFzR1Off + kMhFzR1Bytes;
constexpr int kMhFzMaxCopies = 8;
static_assert(kMhFzBytes % 4096 == 0, "fz blocks stay page aligned");
#endif
// ---- end MSO_MK_FZ ----
static_assert(kMhCopyBytes % 4096 == 0, "copies stay page aligned");

// ---- MSO_MK_NB: one-exchange boundaries (a bit mask; 1 = MhH1, 2 = MhF1): the boundary's first-norm exchange also
// carries three more row sums, from which the second norm's variance follows, so its second exchange goes away
// (stages/moe_hops.cuh nbRowSum4). The first norm stays bitwise; the second norm's statistic is taken on the FP32
// gated residual instead of its BF16 rounding (not bitwise). Only the arena grows (kMhArAbc); 0 builds the
// two-exchange boundaries. ----
// Numerics contract of MSO_MK_NB (MUSESPARK_OPEN_MEGAKERNEL_NB, kernels/megakernel/serving.py; unset = the served
// build's level). What changes: the second norm's statistic of each boundary. Stock (and NB = 0) computes inv2 from
// the mean square of the BF16-rounded gated residual in a second row exchange. NB takes it from the first exchange's
// extra row sums as the expanded square mean(u^2) = A + 2 inv1 B + inv1^2 C (A = sum (alpha r)^2,
// B = sum (alpha r)(beta x), C = sum (beta x)^2), i.e. over the unrounded FP32 u = alpha r + beta (x inv1) (x inv1
// before its BF16 rounding). Why it is not a precision change: every stored value keeps its type (BF16 activations,
// FP32 residual and sums); the statistic is summed in FP32 as in stock, only the point where u is rounded to BF16
// moves after the statistic, and inv1, the residual stream r' / r'' and the captures stay bitwise. Every boundary
// output is the correctly rounded float64 evaluation of the expansion (max 0.500027 ulp), so about 1 ulp moves on a
// few % of each boundary's BF16 outputs (the 8-rank chain test, test_nb_chain).
// The switch is kept because the build is not bitwise the stock layers.
#ifndef MSO_MK_NB
#define MSO_MK_NB 0
#endif
#if MSO_MK_NB < 0 || MSO_MK_NB > 3
#error MSO_MK_NB must be a mask of 1 (MhH1) and 2 (MhF1)
#endif
// ---- end MSO_MK_NB ----
// ---- MSO_MK_BH: two-term MoE input consumers at H1 (a bit mask; stages/moe_hops.cuh). The pool / router input
// x_ffn = (w + 1) inv2 (alpha r + beta inv1 x) is linear in its two terms, so MhPool / MhRouter compute
// inv2 (L(rA) + inv1 L(xB)) with rA = (alpha r)(w + 1) and xB = (beta x)(w + 1), which MhH1 publishes into the arena
// (rA before its hop-1 poll, xB right after its rank sum), and take inv1 / inv2 from MhH1's one-exchange slots. Bits:
// 1 xB and the consumers' combine (alone: a timing build without the residual term, attribution only); 2 the residual
// term rA (the two-term input proper); 4 lo planes bf16(t - bf16(t)) of both terms, read by the pool only. Only the
// arena grows. ----
// Numerics contract of MSO_MK_BH (MUSESPARK_OPEN_MEGAKERNEL_BH, kernels/megakernel/serving.py; unset = the served
// build's level). What changes: the MoE input of the pool and router GEMVs at H1. With NB alone they read x_ffn =
// bf16((w + 1) inv2 u), u = alpha r + beta inv1 x, one BF16 rounding of the whole input. BH splits the input at its
// linear point into two BF16 terms rA = bf16((alpha r)(w + 1)) and xB = bf16((beta x)(w + 1)), runs the GEMV on each
// and combines inv2 (L(rA) + inv1 L(xB)) in FP32 (bit 4 adds the lo planes bf16(t - bf16(t)) of both terms on the
// pool: hi + lo, about 2^-17 relative per term). So the BF16 input rounding points (each term instead of x_ffn) and
// the sum order (two partial GEMVs, then one FP32 combine) change. Why it is not a precision change: every GEMV
// operand is still BF16 x BF16 with FP32 accumulation, the pool keeps its single BF16 rounding of the two K halves'
// FP32 sum, inv1 / inv2 are nb's bits (recomputed from the same one-exchange slot words), and the residual stream,
// the captures and x_ffn's check output stay bitwise. Against the unrounded two-term value the pool and the logits are
// closer than nb's x_ffn path; the stock reference itself rounds x_ffn, so a check-mode distance to stock includes
// that rounding. The lo planes (bit 4) are not in the served build.
// The switch is kept because the build is not bitwise the stock layers.
// Bit 8 (pool-only consumers, with MSO_MK_RL bit 1): the router has no two-term path and no x_ffn path at all
// (stageMhRouter traps): every router slot of a launch must run rl's units (kIFlagRlOn on every layer, served with
// MUSESPARK_OPEN_MEGAKERNEL_RL=all, which serving.py enforces). MhH1 and the pool run bh's code unchanged, and with
// kIFlagRlOn bh's router code never runs, so a bit-8 build computes what the build without bit 8 computes at
// RL=all; only the dead router code leaves the binary. Numerics contract: none beyond bh's (above) and rl's
// (msoMegakernel.cu MSO_MK_RL).
#ifndef MSO_MK_BH
#define MSO_MK_BH 0
#endif
#if MSO_MK_BH < 0 || MSO_MK_BH > 15
#error MSO_MK_BH must be a mask of 1 (branch term), 2 (residual term), 4 (lo planes) and 8 (pool-only consumers)
#endif
#if (MSO_MK_BH & 8) != 0 && (MSO_MK_RL & 1) == 0
#error MSO_MK_BH: bit 8 (pool-only consumers) needs the router logits from rl (MSO_MK_RL bit 1)
#endif
#if MSO_MK_BH != 0 && (MSO_MK_BH & 1) == 0
#error MSO_MK_BH: bit 1 (the early branch term and the consumers' combine) is required
#endif
#if (MSO_MK_BH & 4) != 0 && (MSO_MK_BH & 2) == 0
#error MSO_MK_BH: bit 4 (lo planes) requires bit 2 (the residual term)
#endif
#if MSO_MK_BH != 0 && (MSO_MK_NB & 1) == 0
#error MSO_MK_BH: the consumers read MhH1's one-exchange slots (MSO_MK_NB bit 1)
#endif
#if MSO_MK_FZ && (MSO_MK_NB & 2) == 0
#error MSO_MK_FZ: MhF1 rebuilds the statistics of its one-exchange boundary (MSO_MK_NB bit 2)
#endif
#if MSO_MK_BH
constexpr int kMhBh = MSO_MK_BH;
#endif
// ---- end MSO_MK_BH ----

// Local arena, per copy (bytes).
constexpr int kMhArX = 0;
constexpr int kMhArM = kMhArX + 8 * 8192 * 2;
constexpr int kMhArSsq = kMhArM + 8 * kMhPoolWidth * 2;
#if MSO_MK_NB // nb: [2 boundaries][8 rows][32 slots] x 4 FP32 (S1, A, B, C) after the four norm phases, then 128 pad
// bytes so that the clear span stays a whole number of 16-byte vectors per clear unit
constexpr int kMhArAbc = kMhArSsq + 4 * 8 * 32 * 4;
constexpr int kMhArAbcBytes = 2 * 8 * 32 * 16;
#if MSO_MK_BH // bh: the term planes [8][8192] BF16 after the one-exchange slots: rA (bit 2), rA lo, xB lo (bit 4); xB
// itself takes x_ffn's words (kMhArX). The copy is padded to 128 B with a clear span of whole vectors per clear unit.
constexpr int kMhPlaneBytes = 8 * 8192 * 2;
constexpr int kMhArRA = kMhArAbc + kMhArAbcBytes;
constexpr int kMhArRALo = kMhArRA + kMhPlaneBytes;
constexpr int kMhArXLo = kMhArRALo + kMhPlaneBytes;
constexpr int kMhBhPlanes = (MSO_MK_BH & 4) ? 3 : ((MSO_MK_BH & 2) ? 1 : 0);

constexpr int mhBhArenaBytes(int end)
{
    int a = (end + 127) / 128 * 128;
    while ((a + kMhPoolBytes + kMhLogitBytes) % (20 * 16) != 0)
    {
        a += 128;
    }
    return a;
}

constexpr int kMhArenaBytes = mhBhArenaBytes(kMhArRA + kMhBhPlanes * kMhPlaneBytes);
static_assert(kMhArRA % 16 == 0, "bh: 16-byte plane vectors");
#else
constexpr int kMhArenaBytes = kMhArAbc + kMhArAbcBytes + 128;
#endif
#if MSO_MK_FZ
static_assert(kMhRowV == 1 && !kMhF1Full && kMhRowW == 2048, "fz: one stock thread per vector, 4 quarters per row");
#endif
static_assert(kMhArAbc % 16 == 0 && kMhArenaBytes % 128 == 0, "nb: 16-byte slots, 128-byte aligned arena copies");
static_assert(kMhRowU > 1 && kMhXchg == 1, "nb: the one-exchange boundary replaces the one-poller arena exchange");
static_assert(!(MSO_MK_NB & 2) || !kMhF1Full, "nb: MhF1's first norm goes through the exchange (MSO_MK_MH_F1FULL = 0)");
#else
constexpr int kMhArenaBytes = kMhArSsq + 4 * 8 * 32 * 4;
#endif

// Clear units: the previous launch's copy of the local arena, then of the F2' areas (pool + logits), as one byte range.
constexpr int kMhClearUnits = 20;
constexpr int64_t kMhClearSpan = kMhArenaBytes + kMhPoolBytes + kMhLogitBytes;
constexpr int kMhClearUnitBytes = static_cast<int>(kMhClearSpan / kMhClearUnits);
static_assert(kMhClearSpan % kMhClearUnits == 0 && kMhClearUnitBytes % 16 == 0, "clear units of whole vectors");

// ---- rl (router logits by linearity, kIFlagRlOn): the rl units and their multicast workspace ----
// MhRouter's 64 row-split units become rl units i = e + 32 h (experts [4 i, +4), all 8 rows), each with a 32 KiB weight
// slab (MhRouter's region): the A part W_R diag(gamma_a) over the rank's residual columns [1024 rank, +1024), then the
// D part D_r = W_R diag(gamma_b) W_o,r over the rank's o_proj input, both FP32. The workspace (per copy): the A and the
// B partials [8 source ranks][8 rows][256 experts] FP32, every word kEmpty until its producer stores it.
constexpr int kRlUnits = 2 * kMhLocalExperts;                  // 64
constexpr int kRlExpertsPerUnit = kMhExperts / kRlUnits;       // 4
constexpr int kRlOCols = 1024;                                 // a rank's o_proj input width and residual K slice
constexpr int kRlPartBytes = kRlOCols * kRlExpertsPerUnit * 4; // 16 KiB
constexpr int kRlUnitBytes = 2 * kRlPartBytes;                 // 32 KiB
static_assert(kRlUnitBytes == kMhRouterUnitBytes && kRlOCols * kMhRanks == 8192, "an rl unit takes MhRouter's region");
constexpr int64_t kRlTermBytes = static_cast<int64_t>(kMhRanks) * 8 * kMhExperts * 4;
constexpr int64_t kRlCopyBytes = 2 * kRlTermBytes;
static_assert(kRlCopyBytes % 4096 == 0, "copies stay page aligned");
// The oracle / record of one layer (kPRlOracle): the F2' areas (pool [2][8][4096] FP32, logits [256][8] FP32), then the
// residual r FP32 [8][8192] and this rank's gated o_proj input BF16 [8][1024]. The boundary slices assume U = 4 units
// per row and 256 threads; other builds ignore the rl bits.
constexpr int64_t kRlOracleBytes = kMhPoolBytes + kMhLogitBytes;
constexpr int64_t kRlRecResOff = kRlOracleBytes;
constexpr int64_t kRlRecOOff = kRlRecResOff + 8LL * 8192 * 4;
constexpr int64_t kRlRecBytes = kRlRecOOff + 8LL * kRlOCols * 2;
constexpr bool kRlOk = kMhRowU == 4 && kThreads == 256;
// ---- end rl ----
