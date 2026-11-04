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

// NVFP4 MoE experts stage of the megakernel: shapes, the layer descriptor, the workspace and shared-memory maps.
// Included inside namespace mso_mk, outside the device-code guard (the C ABI reports these constants).
#pragma once

// ---- geometry ----------------------------------------------------------------------------------------------------
constexpr int kMoeExperts = 256;                    // router experts
constexpr int kMoeTopK = 8;
constexpr int kMoeRanks = 8;                        // expert-parallel ranks
constexpr int kMoeBlock = kMoeExperts / kMoeRanks;  // 32 experts per ring block
constexpr int kMoeWindow = 3 * kMoeBlock;           // local stack [own | r-1 | r-2] = 96 experts
constexpr int kMoeMaxSlots = kTok * kMoeTopK;       // distinct local experts of a step, at most 64
constexpr int kMoeHidden = 4096;                    // pooled width: FC1 K, FC2 rows
constexpr int kMoeInter = 4096;                     // FC2 K; FC1 has 2 x 4096 gate/up rows
constexpr int kMoeKb = 16;                          // k-blocks of 256 per 4096-wide K
constexpr int kMoeFc1Tiles = 2 * kMoeInter / 128;   // 64
constexpr int kMoeFc2Tiles = kMoeHidden / 128;      // 32
constexpr int kMoeFc1Units = kMoeFc1Tiles * kMoeKb; // per slot: 1024 (tile, k-block) units
constexpr int kMoeFc2Units = kMoeFc2Tiles * kMoeKb; // per slot: 512
constexpr int kMoeUnitA = 128 * 128; // A bytes of a unit: 128 rows x 128 B (256 FP4), TMA box, SWIZZLE_128B
constexpr int kMoeUnitSf = 128 * 16; // its E4M3 scales: 4 atoms of 128 rows x 4 (128x4 layout), contiguous
constexpr int kMoeUnitBytes = kMoeUnitA + kMoeUnitSf; // 18 KiB
constexpr int kMoeStages = 9;                         // weight ring
// ---- the deepest ring (kMoeFlagE3Ring10); barriers and TMEM columns are laid out for it ----
constexpr int kMoeStagesMax = 10;
// ---- end deepest ring ----
constexpr int kMoeAccs = 4;              // TMEM accumulators
constexpr int kMoeFinalUnits = kTok * 8; // (token, 512-column chunk) units of the weighted sum

// ---- the layer descriptor (int64 words, global memory; kPMoe points at it)
// -------------------------------------------
enum MoeDesc : int
{
    kMdMagic = 0,
    kMdLogits,    // FP32 [8][256] router logits, one Lamport word each (kEmpty = not arrived)
    kMdPool,      // BF16 [8][4096] gathered pool output (pre-norm), Lamport words of two BF16
    kMdBias,      // FP32 [256] expert selection bias
    kMdGainPre,   // BF16 [4096] pre-expert Gemma gain, already bf16(w + 1)
    kMdGs,        // FP32 [1] the quant's global scale (fc31_input_scale)
    kMdTmapW13,   // CUtensorMap (128 B, 64-B aligned, global): w3_w1 [96 * 8192 rows][2048 B]
    kMdSfW13,     // E4M3 128x4-interleaved scales of w3_w1, [96][8192 * 256]
    kMdTmapW2,    // CUtensorMap of w2 [96 * 4096 rows][2048 B]
    kMdSfW2,      // scales of w2, [96][4096 * 256]
    kMdAlpha1,    // FP32 [96] fc31_alpha (output1_scale_gate_scalar)
    kMdScaleC,    // FP32 [96] fc31_scale_c (output1_scale_scalar)
    kMdAlpha2,    // FP32 [96] fc2_alpha (output2_scale_scalar)
    kMdGainPost,  // BF16 [4096] post_expert_norm_weight w (gain = w + 1, FP32)
    kMdWs,        // the stage's workspace: [copies][kMoeWsBytes], every word kEmpty between uses
    kMdPart,      // FP32 [kCtas][2][128][8] single-reader split-tile partials, kEmpty between launches
    kMdRank,      // this rank (placement)
    kMdOut,       // BF16 [8][4096] this rank's partial (plain stores)
    kMdDbg,       // check mode: the CTA-0 debug block (routing, quantized input), or 0
    kMdEps,       // FP32 bits: pre-expert norm eps (low word) | post-expert norm eps << 32
    kMdRenormEps, // FP32 bits: routing renormalization eps
    kMdFlags,     // MoeFlag bits
    kMdRows,      // check mode: BF16 [64][4096] stock-format FC2 rows (row t * 8 + k), or 0
    kMdNumWords = 24,
};

enum MoeFlag : int64_t
{
    kMoeFlagTileCut = 1,          // whole 128-row tiles per CTA (no split-tile partials): bitwise the stock rows
    kMoeFlagPiece0First = 2,      // request the first ring stage, quantize, then the rest
    kMoeFlagFc2TailL2 = 4,        // the producer prefetches the FC2 units past the ring into L2 before it waits for h
    kMoeFlagWarmTwice = 8,        // attribution: routing ... quantization run once more first (warm code), same outputs
    kMoeFlagNoWarmTma = 16,       // no dummy tensor copy per stack at stage entry (that warm-up switched off)
    kMoeFlagDbgNoMma = 32,        // attribution only (wrong results): the MMA warp commits without copies or MMAs
    kMoeFlagIssueFirst = 64,      // the early requests before the warps' row loads (else under them)
    kMoeFlagLoaderEarly = 128,    // the h loader polls from the stage's start (else from the CTA's last FC1 MMA issue)
    kMoeFlagSpinSleep = 256,      // the epilogue's accumulator waits and the loader's start wait sleep between tries
    kMoeFlagDbgNoSfa = 512,       // attribution only (wrong results): no weight-scale copies to TMEM
    kMoeFlagDbgCopiesOnly = 1024, // attribution only (wrong results): the scale copies but no MMAs
    kMoeFlagNoPairs = 2048,       // the MMA warp issues one FC1 unit per elected block (else two of the same tile)
    kMoeFlagNoPoolCopy = 4096,    // the quantization polls the rows from global memory (else: one bulk copy at entry)
    kMoeFlagLocalQuant = 16384,   // every CTA quantizes all 8 rows itself (else each CTA its share, then a gather)
    // ---- integration ----
    kMoeFlagMhF2
    = 32768, // inputs from the MoE hop workspace's F2' areas, output into its finalize area (integration.cuh)
    // ---- end integration ----
    // ---- MoE front ----
    // With kMoeFlagMhF2: route() polls the [256 experts][8 rows] logit area CTA-wide (thread i: vectors i and i + 256,
    // coalesced) and passes it to the token warps through shared memory, instead of 8 scalar words per lane per warp.
    kMoeFlagLogitsCoop = 1LL << 20,
    // The spread quantization (shareQuant) runs before the routing: the pooled input is quantized while the logits
    // are in flight (it does not depend on the routing).
    kMoeFlagQuantFirst = 1LL << 21,
    // The FC2 units past the ring are prefetched into L2 when the producer starts (else, with kMoeFlagFc2TailL2, when
    // it reaches them); they are read later in this launch by this CTA's ring.
    kMoeFlagFc2TailEarly = 1LL << 22,
    // Spread quantization: all kMoeStages ring stages are requested in gatherX (else kTok), the ninth by warp 0.
    kMoeFlagEarly9 = 1LL << 23,
    // Spread quantization without kMoeFlagQuantFirst: the early ring requests right after the cut (lane 0 of each
    // warp), then the quantization (shareQuant), then the gather (which then requests nothing).
    kMoeFlagIssueAtCut = 1LL << 24,
    // Spread quantization without kMoeFlagQuantFirst / kMoeFlagIssueAtCut: routing, placement and cut first, then the
    // quantization, then the gather (which requests the early stages as usual).
    kMoeFlagQuantAtCut = 1LL << 25,
    // ---- end MoE front ----
    // ---- kMoeFlagE3* (FC1 -> h -> FC2 hand-off, FC2 tail) ----
    // The h loader polls every k-block its slot needs at once (16 in flight; per warp its 4 tokens' code chunks and
    // the scale vectors of 8 k-blocks), and stores / releases each one as soon as the warp has it, in first-use order;
    // else four k-blocks per round, waited in order.
    kMoeFlagE3HWide = 1LL << 26,
    // The MMA warp issues FC2 units in pairs (the next k-block of the same tile in the same elected block, as FC1).
    kMoeFlagE3Fc2Pairs = 1LL << 27,
    // A 10-stage weight ring: FC1's B operand (X) and FC2's (h) share one 16 KiB area (the h loader stores only once
    // the CTA's FC1 MMAs are complete), the pre-expert gain sits in ring stage 9 until the quantization has read it.
    // Spread quantization only (not with kMoeFlagLocalQuant / kMoeFlagWarmTwice).
    kMoeFlagE3Ring10 = 1LL << 28,
    // The MMA warp issues FC1 units in groups of up to 4 (else pairs) of the same tile, in one elected block.
    kMoeFlagE3Fc1Quads = 1LL << 29,
    // The MMA warp issues FC2 units in groups of up to 4 of the same tile (overrides kMoeFlagE3Fc2Pairs).
    kMoeFlagE3Fc2Quads = 1LL << 30,
    // The epilogue requests a finishing tile's first earlier partial and (FC2) the post-expert gain before it waits for
    // the tile's accumulator (same values, same summation order).
    kMoeFlagE3EpiPre = 1LL << 31,
    // The MMA warp copies all 16 k-blocks' FC1 token scales to TMEM at its start (while the first weights are in
    // flight), not each with its k-block's first unit.
    kMoeFlagE3XsfPre = 1LL << 32,
    // ---- end kMoeFlagE3* ----
};

// ---- bytes in flight (MoeFlag bits 52-59) ----
// Bits 52-55: with the spread quantization, gatherX requests only this many early ring units (0: kTok), so the
// cross-CTA X gather and the run-once staging code are not queued behind a GPU-wide burst of early requests.
constexpr int kMoeCapEarlyShift = 52;
// Bits 56-59: the producer keeps at most this many of its weight units requested and not yet landed (0: the ring
// depth): it requests unit j once unit j - n is in shared memory.
constexpr int kMoeCapInflightShift = 56;
// ---- end bytes in flight ----
// ---- kMoeW2CapShift / kMoeFlagW2* (MoE front + expert stream; MoeFlag bits 36-43) ----
// Bits 36-39 of the flags: the weight ring keeps at most N units requested and not yet consumed (0 = the ring depth).
// The producer requests unit j once unit j - N's MMAs are complete, and the early burst is min(N, 8) units, so the
// stage has at most N x 18 KiB of weights in flight per CTA (for a single request, the bandwidth peaks before every
// CTA's whole ring is in flight).
constexpr int kMoeW2CapShift = 36;
constexpr int64_t kMoeW2CapMask = 0xfLL << kMoeW2CapShift;
// Bit 41: the placement in warp 0 alone, warp-synchronous (p4a::placeWarp; same slots, same order).
constexpr int64_t kMoeFlagW2PlaceWarp = 1LL << 41;
// Bit 42: the MMA warp copies a new FC2 k-block's h scales to TMEM inside the unit's (or pair's) MMA block, not in an
// elected block of their own (same tensor operations in the same order).
constexpr int64_t kMoeFlagW2Fc2OneElect = 1LL << 42;
// Bit 43: the in-flight cap applies to the FC1 units only (the FC2 units fill the whole ring before h arrives).
constexpr int64_t kMoeFlagW2CapFc1Only = 1LL << 43;
// Bit 40 (with kMoeFlagMhF2 and kMoeFlagLogitsCoop, used by the kMoeFlagQuantFirst order): shareQuant requests the
// routing's logit vectors once its pool rows are in, and a warp whose vectors all came back full stages them for
// route() (which then polls nothing for that warp); the other warps poll as without the bit. Same values, same staging.
constexpr int64_t kMoeFlagW2LogitsPre = 1LL << 40;
// ---- end kMoeW2CapShift / kMoeFlagW2* ----

// ---- MSO_MK_IC_MOE_FIX: the experts flags as compile-time constants (build modes *_icf*_s4) ----
// The bits of MSO_MK_IC_MOE_MASK take the values of MSO_MK_IC_MOE_VAL in every read of the descriptor's flags word, so
// the branches of the other flag settings fold away at compile time (a smaller kernel; the same operations in the same
// order for the settings the build fixes). The stage traps if a launch's flags differ from the fixed ones on those
// bits.
#ifndef MSO_MK_IC_MOE_FIX
#define MSO_MK_IC_MOE_FIX 0
#endif
#if MSO_MK_IC_MOE_FIX
#if (MSO_MK_LANES & 16)
#error "MSO_MK_IC_MOE_FIX fixes the NVFP4 experts' flags; the BF16 experts (MSO_MK_LANES bit 4) read the same word"
#endif
constexpr int64_t kIcMoeMask = static_cast<int64_t>(MSO_MK_IC_MOE_MASK);
constexpr int64_t kIcMoeVal = static_cast<int64_t>(MSO_MK_IC_MOE_VAL);
static_assert((kIcMoeVal & ~kIcMoeMask) == 0, "the fixed flag values lie inside the mask");
#else
constexpr int64_t kIcMoeMask = 0;
constexpr int64_t kIcMoeVal = 0;
#endif

__device__ __forceinline__ int64_t icMoeFlags(int64_t raw)
{
    return (raw & ~kIcMoeMask) | kIcMoeVal;
}

// ---- end MSO_MK_IC_MOE_FIX ----
// ---- MSO_MK_RT: the experts front's own code (build modes *_rt<m>*: a bit mask; 0 = the code as is) ----
// Bit 1: route()'s top 8 from the lane's 8 keys sorted once, then 8 rounds of one redux over the lanes' heads and a
//        ballot for the lowest lane among the maxima (same selections, weights and order).
// Bit 2: the placement without place()'s four CTA barriers: route() publishes each token's presence words, every warp
//        computes the ring flow and the three local presence words itself (p4a::rtPlace; same window ids, slots, E).
// Bit 4 (with bit 2): no barrier after the placement with the stock queue: lane 0 of each early warp requests its
//        early ring unit from the closed-form queue position right after the placement, the slot scalars read the
//        slot's window id from the presence words (the slot tables are read only after the stage's role barrier).
// Bit 8: with QuantFirst the X gather's loads go out inside route() once the logits are in, and are completed and
//        stored after the picks (gatherX's addresses and stores); the stage then runs no gatherX, and its early
//        requests follow the cut (or the placement, bit 4) with no X loads behind them.
// Bit 16: in the lean builds the debug-block code (kMdDbg, kMdRows: check-mode copies of the routing, the quantized
//        input and the FC2 rows) is compiled out; no lean launch passes a debug block, and the stage traps if one does.
// Bit 32: the window's per-expert scalars (alpha1, scale_c, alpha2: FP32 [96] each) are copied into shared memory at
//        stage entry (cp.async, threads 0-95), so the slot scalars after the cut are shared-memory reads, not global
//        loads whose latency gatherX hides when it runs (bit 8 runs no gatherX).
// Bit 64 (with bit 8): route()'s X gather by cp.async straight into shared memory (the codes into the SW128 X tile,
//        the scale vectors into a temp that is reshuffled at the end), polled there; no X values held in registers
//        across the picks.
#ifndef MSO_MK_RT
#define MSO_MK_RT 0
#endif
#if (MSO_MK_RT & ~127) || ((MSO_MK_RT & 4) && !(MSO_MK_RT & 2)) || ((MSO_MK_RT & 64) && !(MSO_MK_RT & 8))
#error "MSO_MK_RT: bits 1, 2, 4, 8, 16, 32, 64; bit 4 needs bit 2, bit 64 needs bit 8"
#endif
// ---- end MSO_MK_RT ----
// ---- kMoeFlagLx* (per-unit issue loops; MoeFlag bits 33-34; same requests, same tensor operations, same order) ----
// Bit 33: the weight stream after the early requests runs on the whole of warp 0 (produceLoopLx): the CTA's units in
// lane-parallel windows of 32, one shuffle per unit, running stage / parity counters, one elected lane issuing with
// warp-uniform operands. Taken only without a kMoeW2CapMask cap below the ring depth and without the FC2 tail
// prefetches (else produceLoop).
constexpr int64_t kMoeFlagLxProducer = 1LL << 33;
// Bit 34: the MMA warp walks a per-unit table (mmaLoopLx): new tile, last of tile, pair, first FC1 use of a k-block,
// first use of an h k-block within its FC2 slot and the slot changes, in closed form per lane for a window of 64
// units; running counters for the stage, the tile, the h slot and their barrier parities. Taken only with
// kMoeFlagW2Fc2OneElect and without the kMoeFlagE3 unit groups (else mmaLoop).
constexpr int64_t kMoeFlagLxMma = 1LL << 34;
// Attribution hooks of the Lx loops (every hook is off in this build).
constexpr int64_t kLxAttrNoStream = 0;
constexpr int64_t kLxAttrProdDelay = 0;
constexpr int64_t kLxAttrMmaDelay = 0;
constexpr int64_t kLxAttrMmaArrive = 0;
constexpr int64_t kLxAttrRequestPath = 0;
constexpr uint32_t kLxAttrDelayNs = 100;
// ---- end kMoeFlagLx* ----
// ---- kMoeFlagPx* (the stock persistent_experts tile queue; MoeFlag bit 35) ----
// Bit 35: the work of the launch is persistent_experts' static queue of whole 128-row tiles (16 k-blocks each, one
// accumulator, K ascending: its rows bit for bit) instead of the byte cut: E slots give 96 E tile positions, FC1 tiles
// first (rows major across the slots, its order 1 with its early slots), then FC2 tiles (slot major); the positions run
// on G = ceil(96 E / R) CTAs, R = ceil(96 E / kCtas) rounds (its balance 1), CTA c < G taking c, c + G, c + 2 G, ...
// No tile is split, so there are no partials; the last round's FC1 tiles (the highest rows, which FC2 reads in its last
// k-blocks) run beside FC2 tiles on other CTAs. CTAs >= G stream nothing (they still route, quantize and run their
// final unit).
constexpr int64_t kMoeFlagPxQueue = 1LL << 35;
// Bit 17 (with bit 35): the MMA warp in the stock kernel's loop shape (p4a::mmaLoopPx: per tile, k-block pairs from
// running ring counters, no table; the same tensor operations and commits per unit as mmaLoop).
constexpr int64_t kMoeFlagPxMma = 1LL << 17;
// Bit 18 (with bit 35, not with kMoeFlagE3Ring10): an 8-stage weight ring (the stock kernel's depth on sm_107), the
// producer, the MMA warp and the h loader on the 8-stage layout (the 9-stage layout's X / h areas, stage 8 unused).
constexpr int64_t kMoeFlagPxRing8 = 1LL << 18;
constexpr int kMoeStagesPx8 = 8;
// ---- end kMoeFlagPx* ----
// ---- MSO_MK_UE: the stock queue's per-unit pace (build modes *_ue<m>*: a bit mask; 0 = the code as is). Each bit
// changes only the px instantiations (the stock queue, kMoeFlagPxQueue) and keeps every tile's tensor operations, their
// operands and their order (whole tiles, K ascending): the same rows bit for bit.
// Bit 1: the h loader polls its FC2 slots' h from the stage's start (kMoeFlagLoaderEarly), not from the CTA's last FC1
//        MMA issue: with three or more rounds an FC2 tile's early k-blocks are published rounds before its CTA's FC1
//        ends.
// Bit 2: the h loader of a CTA with two or more FC1 tiles starts when the MMA warp begins the CTA's last FC1 tile (the
//        k-blocks of the earlier rounds are published by then; a CTA with one FC1 tile keeps the last-issue start).
// Bit 4: the producer keeps the whole ring in flight (no in-flight cap; the cap field's early-request count stays).
// Bit 8: the MMA warp issues one k-block per elected block (each unit waits only for its own stage and h k-block), not
//        k-block pairs.
#ifndef MSO_MK_UE
#define MSO_MK_UE 0
#endif
#if (MSO_MK_UE & ~15) || ((MSO_MK_UE & 3) == 3)
#error "MSO_MK_UE: bits 1, 2, 4, 8; bits 1 and 2 exclude each other"
#endif
constexpr bool kUeLoaderEarly = (MSO_MK_UE & 1) != 0;
constexpr bool kUeLastTile = (MSO_MK_UE & 2) != 0;
constexpr bool kUeNoCap = (MSO_MK_UE & 4) != 0;
constexpr bool kUeSingle = (MSO_MK_UE & 8) != 0;
// ---- end MSO_MK_UE ----
// ---- MSO_MK_ET: the px path's tail (build modes *_et<m>*: a bit mask; 0 = the code as is, token for token).
// Each bit changes only the px path (kMoeFlagPxQueue) and keeps every value, word and writer: per token and per pick
// the same operations on the same operands in the same order, so the rows, the sums and the outputs keep their bits.
// Bit 1: the FC2 epilogue in phases (fc2EpiloguePx): the slot's 8 k bytes in one load, the post gain loaded before
//        the tile's accumulator wait, then across the 8 tokens: the BF16 rows and their gather shuffles, the 8 row
//        stores (predicated, no branch per token), the 5 butterfly rounds each over the 8 tokens, lane 0's sums.
// Bit 2: the final unit (finalUnitPx) waits for its first local pick's sums as finalUnit does, then requests every
//        other load of its window of up to 4 local picks at once (lanes 8 j .. 8 j + 7 hold pick j's sums) and
//        re-polls only the words still empty; the sums, scales and the accumulation in k order are finalUnit's.
// Bit 4: as bit 2 without the first wait: every load of the window from the start (excludes bit 2).
#ifndef MSO_MK_ET
#define MSO_MK_ET 0
#endif
#if (MSO_MK_ET & ~7) || ((MSO_MK_ET & 6) == 6)
#error "MSO_MK_ET: bits 1, 2, 4; bits 2 and 4 exclude each other"
#endif
// ---- end MSO_MK_ET ----
// ---- MSO_MK_CX: an attribution-only oracle of the experts front (build modes *_cx<m>*: 0 = the code as is,
// token for token). Never served. Bit 1: a non-zero descriptor word kMdDbg is this layer's oracle slot (kCxSlotBytes,
// below), which the lean rt16 builds otherwise leave unused (the stage traps on it). A launch whose slot is valid
// (have == want, level != 0) loads the routing state (route()'s and place()'s shared-memory tables and E) from the slot
// at stage entry and skips route() and place(); level 1 also skips QuantFirst and requests the early weight units right
// after the cut (the kMoeFlagIssueAtCut path: cut -> requests -> shareQuant -> gatherX), level 2 keeps the served
// order.
// A launch whose slot is stale records: CTA 0 stores its tables after place() and, after its warp roles (when every
// CTA has passed its own stage entry, since CTA 0's gatherX has seen every CTA's quantization share), have = want.
// The harness bumps want before every timing call; within one call the inputs and the routing are fixed.
#ifndef MSO_MK_CX
#define MSO_MK_CX 0
#endif
#if MSO_MK_CX
#if (MSO_MK_CX & ~1)
#error "MSO_MK_CX: bit 1 only"
#endif
#if !((MSO_MK_RT & 16) && MSO_MK_LEAN) || (MSO_MK_RT & 6)
#error "MSO_MK_CX: lean rt16 builds only (the slot rides in kMdDbg), without rt bits 2 / 4"
#endif
// Slot layout (bytes): [0] want (u32, host), [4] have (u32, kernel), [8] level (u32, host: 0 record only, 1 early
// order, 2 served order), [12] early units (u32, host: 0 = the served count), [16] E (u32, kernel), [64, 64 + 16
// kCxImgVecs) the image of the reduction area [kMoeRdSlotExpert, kMoeRdRstd).
constexpr int kCxSlotBytes = 2048;
constexpr int kCxImgOff = 64;
#endif
// ---- end MSO_MK_CX ----
// ---- MSO_MK_RP: the routing state published once per rank (build modes *_rp<m>*: 0 = the code as is, token for
// token). Bit 1: CTA kCtas - 1 (the publisher, which quantizes no share: the groups are split over the other CTAs) runs
// route() and place() as every CTA does without rp and publishes the resulting tables and E into the workspace's
// routing area (kMoeWsRp); every other CTA polls that area after its quantization instead of running route() and
// place(). The same code computes the same tables on one CTA instead of all, and each quantization group's arithmetic
// does not depend on the CTA that runs it, so the outputs keep their bits. Area: 240 vectors of two image words each,
// every 32-bit word split into its 16-bit halves with bit 16 set (never kEmpty), then E the same way; the loaders'
// clear of the next copy (loaderLoop) empties it with the rest of the workspace.
// Numerics contract (rp 1 in the served build, kernels/megakernel/serving.py DEFAULT_MODE): value-identical by
// construction (the same route() / place() code on the same logits, a bit copy of its tables, the quantization groups'
// arithmetic independent of the CTA); in check mode every layer's check digits equal the build without rp. Alone it is
// slower; it pays together with rl, whose early logits let the publisher's tables land before the other CTAs'
// quantization ends. ----
#ifndef MSO_MK_RP
#define MSO_MK_RP 0
#endif
#if (MSO_MK_RP & ~1)
#error "MSO_MK_RP: bit 1 only"
#endif
#if MSO_MK_RP
constexpr int kRpImgVecs = 240;         // (kMoeRdRstd - kMoeRdSlotExpert) / 8 image word pairs
constexpr int kRpVecs = kRpImgVecs + 1; // + E
constexpr int kRpBytes = 256 * 16;      // the area per copy (whole 16-byte granules)
#endif
// ---- end MSO_MK_RP ----
// ---- MSO_MK_XP: the experts front's X path (build modes *_xp<m>*: 0 = the code as is, token for token).
// Bit 1: row owners. The 512 quantization groups (kTok rows x 64 groups of 4 blocks) go to CTAs [kXpOwner0,
// kXpOwner0 + kTok): CTA kXpOwner0 + r polls pooled row r and quantizes all of it; every other CTA quantizes nothing,
// so it polls no pooled row (one poll round of the rank reads kTok rows instead of about one row per CTA) and its
// routing-state poll and weight requests do not wait behind a share of its own. Bit 2: two owners per row (CTAs
// [kXpOwner0, kXpOwner0 + 2 kTok)), each polls and sums the whole row and quantizes half of its groups. Bit 4 (needs rp
// bit 1): a CTA other than rp's publisher that quantizes no group polls the routing state first and requests its early
// weight units at the cut (the kMoeFlagIssueAtCut path of stageCut), before its quantization call and the X gather;
// without bit 1 / 2 every such CTA (rp: all but the publisher) does so, after the state, before its share. Bit 8 (needs
// rp bit 1): rp's publisher runs route() / place() once on zero logits at its stage entry, before its logit poll
// (xpWarmPublisher), so that its real pass, on the routing chain, runs code it has just fetched.
// Bit 32 (cut table, px queue only): at the stage entry, threads [128, 128 + kMoeMaxSlots] each compute the CTA's stock
// queue cut for one slot count E = tid - 128 (makeCutPxBody: the schedule's integer divisions) and its first tile
// (pxTile, tile 0), and store them as one vector of kMoeRdXpCut; after the state, stageCut<true> reads the vector of
// the launch's E instead of running the divisions on the request chain. Bit 64 (needs bit 32): gatherX's early
// requests (units j < kMoeKb: k-block j of tile 0) take tile 0's (fc, first list unit) from the same vector instead of
// pxTile's decode. Bit 128 (MhClear h1): MhH1 does not empty its hop-1 slots after reading them; the MhClear units
// [0, kXpH1ClearUnits) of the next launch empty the previous copy's whole hop-1 area (the pool / logit area rule of the
// moe_hops.cuh ownership note), so that MhH1's rlFinalize follows its boundary without the 32 KiB of empty stores.
// Bit 256 (MhClear h1, merged form; not with 128): the same, with the hop-1 area inside MhClear's one byte range (the
// arena copy, then the multicast copy's hop-1 + pool + logit areas) over all its units. Bit 512 (row-aligned shares,
// needs rp bit 1, not with bits 1 / 2 / 4): the quantizing CTAs are split over the rows in order and each row's 64
// groups over its CTAs, so no CTA polls and sums two pooled rows; bit 1024 (needs 512): the MhPool CTAs [0, 64), which
// enter the stage last, quantize none (their groups go to CTAs [64, kCtas - 1)).
// Numerics contract (xp 40 in the served build, kernels/megakernel/serving.py DEFAULT_MODE): value-identical by
// construction: a group's codes and scales are shareQuant's arithmetic on the whole row (the row's sum of squares in
// the block-per-thread order, which no CTA index enters), so moving groups between CTAs, and the order of the requests,
// the state poll and the quantization within a CTA, changes no bit of X or of any output; the warm-up pass's words are
// written again (or cleared) before any reader; bits 32 / 64 compute the same integers (makeCutPxBody, pxTile) earlier
// and read them back, so the cut, the unit order and every request are the same; bits 128 / 256 move only when (not
// whether) a hop-1 slot is emptied before its next writer; bits 512 / 1024 move groups between CTAs (the argument
// above); no arithmetic changes. The check build's digits must equal the build without xp. ----
#ifndef MSO_MK_XP
#define MSO_MK_XP 0
#endif
#if (MSO_MK_XP & ~(15 | 32 | 64 | 128 | 256 | 512 | 1024))
#error "MSO_MK_XP: bits 1, 2, 4, 8, 32, 64, 128, 256, 512, 1024 only (bit 16 is reserved and unused)"
#endif
#if (MSO_MK_XP & 128) && (MSO_MK_XP & 256)
#error "MSO_MK_XP: bits 128 and 256 are two forms of one lever (exclusive)"
#endif
#if (MSO_MK_XP & 256) && !((MSO_MK_RL & 1) && !(MSO_MK_RL & 8))
#error "MSO_MK_XP bit 256 (hop-1 slots emptied by MhClear) is built for rl's MhH1 logits (rl bit 1 without bit 8)"
#endif
#if (MSO_MK_XP & 512) && (!(MSO_MK_RP & 1) || (MSO_MK_XP & 7))
#error "MSO_MK_XP bit 512 (row-aligned shares) needs rp bit 1 and excludes bits 1 / 2 / 4"
#endif
#if (MSO_MK_XP & 1024) && !(MSO_MK_XP & 512)
#error "MSO_MK_XP bit 1024 (no share on the MhPool CTAs) needs bit 512"
#endif
#if (MSO_MK_XP & 64) && !(MSO_MK_XP & 32)
#error "MSO_MK_XP bit 64 (the early requests from the cut table) needs bit 32 (the cut table)"
#endif
#if (MSO_MK_XP & 32) && (MSO_MK_RT & (2 | 4 | 32))
#error                                                                                                                 \
    "MSO_MK_XP bit 32: the cut table takes the reduction area's free end, which rt bits 2 / 32 use (and rt bit 4 cuts in rtPlace)"
#endif
#if (MSO_MK_XP & 128) && !((MSO_MK_RL & 1) && !(MSO_MK_RL & 8))
#error "MSO_MK_XP bit 128 (hop-1 slots emptied by MhClear) is built for rl's MhH1 logits (rl bit 1 without bit 8)"
#endif
#if (MSO_MK_XP & 8) && (!(MSO_MK_RP & 1) || (MSO_MK_RL & 8) || MSO_MK_CX)
#error "MSO_MK_XP bit 8 (the publisher's warm-up pass) needs rp bit 1, without rl bit 8 and the cx oracle"
#endif
#if (MSO_MK_XP & 3) == 3
#error "MSO_MK_XP: bits 1 and 2 are exclusive (one or two owners per row)"
#endif
#if (MSO_MK_XP & 4) && !(MSO_MK_RP & 1)
#error "MSO_MK_XP bit 4 (state first) polls rp's published state: it needs MSO_MK_RP bit 1"
#endif
#if (MSO_MK_XP & 4) && (MSO_MK_CX || (MSO_MK_RT & 6))
#error "MSO_MK_XP bit 4: not with the cx oracle or rt bits 2 / 4 (their own stage-front orders)"
#endif
#if MSO_MK_XP & 3
constexpr int kXpPerRow = (MSO_MK_XP & 2) ? 2 : 1; // owners per pooled row
constexpr int kXpOwner0 = 96; // CTAs [96, 128) run no MoE-half unit before the experts stage (they enter it first)
#endif
// ---- end MSO_MK_XP ----
// ---- MSO_MK_CF: the h loader's code (build modes *_cf<m>*: a bit mask; 0 = the code as is, token for token).
// Each bit changes only loaderLoop (warps 2-3 of the experts stage) and keeps every load, store, barrier arrival and
// their order: the same h words land at the same shared-memory addresses before the same HReady arrivals.
// Bit 1: the narrow h path's four in-flight k-blocks are stored and released by one rolled body (k-block p + i of
//        the group takes its two load registers by a select over the four, then polls and stores as the unrolled
//        copies do) instead of four unrolled copies; the four loads of a group are issued as in those copies, all
//        before the first store.
// Bit 2: the clear of the next copy's workspace slice (one 16 B store per 64 granules) as a rolled loop.
// Bit 4: with the stock queue (px) the slot scan in closed form: every px FC2 tile is one slot (a CTA's FC2 positions
//        are pxG >= 96 apart and a slot has 32 tiles, the same fact mmaLoopPx's HFree commits use) and its units are
//        k-blocks 0 .. 15 in order, so need = 0xffff, the first-use order is 0, 1, .., 15 and the slot ends 16 units
//        later; the scan's sixteen unit decodes do not run. The check builds (MSO_MK_LEAN 0) still run the scan and
//        trap when it differs from the closed form.
// Value-identical: the h words are polled from the same addresses until full and stored unchanged; only the code
// form (bits 1, 2) and which integers are recomputed (bit 4) change.
#ifndef MSO_MK_CF
#define MSO_MK_CF 0
#endif
#if (MSO_MK_CF & ~7)
#error "MSO_MK_CF: bits 1 (rolled h store / release body), 2 (rolled workspace clear), 4 (px closed-form slot scan)"
#endif
// ---- end MSO_MK_CF ----

constexpr int64_t kMoeMagic = 0x4d4f454d4b503441LL; // "MOEMKP4A"

// ---- the stage workspace (per copy) --------------------------------------------------------------------------------
// h (the FC2 B operand): per slot, per k-block, per token 128 B of E2M1 (256 intermediate columns) and 16 B of scales.
constexpr int kMoeWsH = 0;
constexpr int kMoeWsHBytes = kMoeMaxSlots * kMoeKb * kTok * 128;  // 1 MiB
constexpr int kMoeWsHsf = kMoeWsH + kMoeWsHBytes;
constexpr int kMoeWsHsfBytes = kMoeMaxSlots * kMoeKb * kTok * 16; // 128 KiB
// FC2 rows of the local (token, k) pairs: BF16 [64][4096].
constexpr int kMoeWsY = kMoeWsHsf + kMoeWsHsfBytes;
constexpr int kMoeWsYBytes = kTok * kMoeTopK * kMoeHidden * 2; // 512 KiB
// sum of (y (1 + w_post))^2 per (token, k, FC2 tile): FP32 [64][32].
constexpr int kMoeWsSsq = kMoeWsY + kMoeWsYBytes;
constexpr int kMoeWsSsqBytes = kTok * kMoeTopK * kMoeFc2Tiles * 4;
// The quantized expert input, published by the CTA that quantizes each 4-block group (spread quantization, the
// default): E2M1 [8][2048 B] (Lamport words of 8 codes) and E4M3 [8][256 B] (words of 4 scales).
constexpr int kMoeWsXq = kMoeWsSsq + kMoeWsSsqBytes;
constexpr int kMoeWsXqBytes = kTok * kMoeHidden / 2;
constexpr int kMoeWsXsf = kMoeWsXq + kMoeWsXqBytes;
constexpr int kMoeWsXsfBytes = kTok * kMoeHidden / 16;
#if MSO_MK_RP
// ---- MSO_MK_RP: the published routing state (kRpVecs Lamport vectors), after the quantized input ----
constexpr int kMoeWsRp = kMoeWsXsf + kMoeWsXsfBytes;
constexpr int kMoeWsBytes = kMoeWsRp + kRpBytes;
#else
constexpr int kMoeWsBytes = kMoeWsXsf + kMoeWsXsfBytes;
#endif
static_assert(kMoeWsBytes % 16 == 0, "the clear cut is in 16-byte granules");

constexpr int kMoePartSlotBytes = 128 * kTok * 4; // FP32 [128 rows][8 tokens]
constexpr int kMoePartBytes = kCtas * 2 * kMoePartSlotBytes;

// ---- the check-mode debug block (CTA 0 writes it) -------------------------------------------------------------------
constexpr int kMoeDbgSel = 0;     // int32 [8][8] selected expert ids
constexpr int kMoeDbgW = 256;     // FP32 [8][8] routing weights
constexpr int kMoeDbgLocal = 512; // int32 [8][8] window ids (local < 96, else 96 + e % 160)
constexpr int kMoeDbgLoads = 768; // int32 [16]: per-block loads before, then after placement
constexpr int kMoeDbgSlots = 832; // int32 [1 + 64]: E, then the slots' local ids
constexpr int kMoeDbgXq = 1280;   // u8 [8][2048] the quantized expert input (E2M1, element 2i in the low nibble)
constexpr int kMoeDbgXsf = kMoeDbgXq + kTok * kMoeHidden / 2; // u8 [8][256] linear E4M3 block scales
constexpr int kMoeDbgBytes = kMoeDbgXsf + kTok * kMoeHidden / 16;

// ---- shared memory of the stage (offsets in the unit region [0, kSmemMain) and in the reduction area)
// ----------------
constexpr int kMoeSmRing = 0;                                    // kMoeStages x 18 KiB
constexpr int kMoeSmX = kMoeSmRing + kMoeStages * kMoeUnitBytes; // FC1 B: [16 k-blocks][8 rows][128 B], SW128
constexpr int kMoeSmH = kMoeSmX + kMoeKb * kTok * 128;           // FC2 B of the current slot, same layout
constexpr int kMoeSmEnd = kMoeSmH + kMoeKb * kTok * 128;
static_assert(kMoeSmX % 1024 == 0 && kMoeSmH % 1024 == 0, "SW128 atoms are 1 KiB aligned");
// ---- kMoeFlagE3Ring10 layout: 10 ring stages, then one 16 KiB B area for X (FC1) and h (FC2) ----
constexpr int kMoeSmXH10 = kMoeSmRing + kMoeStagesMax * kMoeUnitBytes;
constexpr int kMoeSmEnd10 = kMoeSmXH10 + kMoeKb * kTok * 128;
constexpr int kMoeSmGain10 = kMoeSmRing + (kMoeStagesMax - 1) * kMoeUnitBytes;
static_assert(kMoeSmXH10 % 1024 == 0, "SW128 atoms are 1 KiB aligned");
static_assert(kMoeSmGain10 + kMoeHidden * 2 <= kMoeSmXH10, "the gain fits ring stage 9");
// ---- end kMoeFlagE3Ring10 layout ----
// Ring stages requested before the quantization; the later stages hold the 8 pre-norm rows meanwhile
// ([8][4096] BF16) and are requested once the quantization has read them.
constexpr int kMoeStagesEarly = 5;
constexpr int kMoeSmQStage = kMoeSmRing + kMoeStagesEarly * kMoeUnitBytes;
static_assert(kMoeSmQStage + kTok * kMoeHidden * 2 <= kMoeSmX, "quantization staging inside the late ring stages");
static_assert(kMoeStagesEarly <= kTok, "one early request per quantizing warp");
// The pre-expert gain (BF16 [4096], one bulk copy at stage entry) sits in the FC2 B area until the quantization has
// read it (the h loader writes that area only after the quantization's barrier).
constexpr int kMoeSmGain = kMoeSmH;
static_assert(kMoeHidden * 2 <= kMoeKb * kTok * 128, "the gain fits the FC2 B area");
// Reduction area (kSmemRed, 16 KiB):
constexpr int kMoeRdBars = 0;          // mbarriers
constexpr int kMoeRdDesc = 512;        // the descriptor (kMdNumWords int64)
constexpr int kMoeRdScal = 768;        // scalars (int32): see kMoeSc*
constexpr int kMoeRdSlotExpert = 1024; // int32 [64]
constexpr int kMoeRdSlotMask = 1280;   // u32 [64]: tokens routed to the slot
constexpr int kMoeRdSlotK = 1536;      // int8 [64][8]: k of (slot, token)
constexpr int kMoeRdSel = 2048;        // int32 [8][8]
constexpr int kMoeRdW = 2304;          // FP32 [8][8]
constexpr int kMoeRdLocal = 2560;      // int32 [8][8]
constexpr int kMoeRdTokSlot = 2816;    // int8 [8][8]: slot of (token, k) or -1
constexpr int kMoeRdPres = 2880;       // u32 [8] block presence + u32 [3] local presence
constexpr int kMoeRdRstd = 2944;       // FP32 [8]
constexpr int kMoeRdMoved = 3008;      // int32 [16]: moved[b][1] at b, moved[b][2] at 8 + b
constexpr int kMoeRdRed = 3072;        // FP32 [8 warps][8 tokens] quant sums; epilogue cross-warp sums
constexpr int kMoeRdStgX = 4096;       // FC1 token scales staged for tcgen05.cp: 32 pairs x 128 B
constexpr int kMoeRdStgH = 8192;       // FC2 token scales of the current slot: 32 pairs x 128 B
constexpr int kMoeRdEpi = 12288;       // FC1 epilogue staging: 8 x 32 B codes + 8 x 4 B scales
constexpr int kMoeRdEpiSq = 12800;     // FC2 epilogue: FP32 [4 warps][8 tokens]
constexpr int kMoeRdSlotScal = 13312;  // FP32 [3][64]: per slot alpha1, scale_c, alpha2
#if MSO_MK_RT & 2
// MSO_MK_RT bit 2: u32 [8 blocks][8 tokens]: bit i of word [b][t] = token t chose expert 32 b + i (route() writes,
// rtPlace reads).
constexpr int kMoeRdRtPres = 14336;
static_assert(kMoeRdRtPres >= kMoeRdSlotScal + 3 * 256 && kMoeRdRtPres + 4 * kMoeRanks * kTok <= 16384,
    "rt presence words in the reduction area's free end");
#endif
#if MSO_MK_RT & 32
// MSO_MK_RT bit 32: FP32 [3][96] alpha1, scale_c, alpha2 of the window (cp.async at stage entry, read by the slot
// scalars).
constexpr int kMoeRdRtScal = 15104;
static_assert(kMoeRdRtScal >= 14336 + 4 * kMoeRanks * kTok && kMoeRdRtScal + 3 * 4 * kMoeWindow <= 16384,
    "rt window scalars in the reduction area's free end");
#endif
#if MSO_MK_XP & 32
// MSO_MK_XP bit 32: u32x4 [kMoeMaxSlots + 1], entry E = the CTA's stock queue cut for E slots: (pxG, pxEarly,
// nT1 | nT << 16, tile 0's first list unit | fc << 20); written at the stage entry before its barrier, read after the
// state.
constexpr int kMoeRdXpCut = 14336;
static_assert(kMoeRdXpCut >= kMoeRdSlotScal + 3 * 256 && kMoeRdXpCut + 16 * (kMoeMaxSlots + 1) <= 16384,
    "xp cut table in the reduction area's free end");
#endif
// Scalars (int32 index into kMoeRdScal).
constexpr int kMoeScTmem = 0;
constexpr int kMoeScE = 1;
// Barrier indices.
constexpr int kMoeBarFull = 0;
constexpr int kMoeBarEmpty = kMoeBarFull + kMoeStagesMax;    // room for kMoeFlagE3Ring10
constexpr int kMoeBarAccFull = kMoeBarEmpty + kMoeStagesMax; // room for kMoeFlagE3Ring10
constexpr int kMoeBarAccEmpty = kMoeBarAccFull + kMoeAccs;
constexpr int kMoeBarHReady = kMoeBarAccEmpty + kMoeAccs;
constexpr int kMoeBarHFree = kMoeBarHReady + kMoeKb;
constexpr int kMoeBarGain = kMoeBarHFree + 1;
constexpr int kMoeBarFc1Done = kMoeBarGain + 1; // the MMA warp has issued its last FC1 unit (the h loader's start)
constexpr int kMoeBarPool = kMoeBarFc1Done + 1; // the stage-entry copy of the pool rows into the quantization staging
// ---- the CTA's FC1 MMAs are complete (tcgen05.commit after its last FC1 unit; kMoeFlagE3Ring10's h loader) ----
constexpr int kMoeBarFc1Cmpl = kMoeBarPool + 1;
constexpr int kMoeNumBars = kMoeBarFc1Cmpl + 1;
// ---- end kMoeBarFc1Cmpl ----
static_assert(kMoeNumBars * 8 <= kMoeRdDesc, "barriers");

// TMEM columns.
constexpr uint32_t kMoeTmemCols = 512;
constexpr uint32_t kMoeTmD = 0;      // accumulator a: [8 a, 8 a + 8)
constexpr uint32_t kMoeTmSfa = 32;   // ring stage s: [32 + 16 s, +16), 4 columns per K step
constexpr uint32_t kMoeTmSfbX = 256; // FC1 token scales: 64 K steps x 2 columns
constexpr uint32_t kMoeTmSfbH = 384; // FC2 token scales
static_assert(kMoeTmSfa + 16 * kMoeStagesMax <= kMoeTmSfbX, "TMEM map"); // room for kMoeFlagE3Ring10

// Watchdog wait codes of the stage.
enum MoeWaitCode : int
{
    kCodeMoeLogits = 160,
    kCodeMoePool = 161,
    kCodeMoeBar = 162,
    kCodeMoePart = 163,
    kCodeMoeH = 164,
    kCodeMoeSsq = 165,
    kCodeMoeY = 166,
};

// Sub-stamps (kStampSub + k) the stage writes.
constexpr int kMoeStRouted = 8;
constexpr int kMoeStQuant = 9;
constexpr int kMoeStFirstFull = 10;
constexpr int kMoeStFc1Issued = 11;
constexpr int kMoeStFc1EpiDone = 12;
constexpr int kMoeStFirstH = 13;
constexpr int kMoeStMmaDone = 14;
constexpr int kMoeStEpiDone = 15;
constexpr int kMoeStFinalIn = 16;
constexpr int kMoeStFinalDone = 17;
constexpr int kMoeStDesc = 18;        // the descriptor is in shared memory, TMEM allocated
constexpr int kMoeStLogits = 19;      // warp 0 has its token's logits
constexpr int kMoeStTop8 = 20;        // every token's top 8 and weights
constexpr int kMoeStEarly = 21;       // thread 0: the early ring stages are requested
constexpr int kMoeStPool = 22;        // thread 0: its pre-norm vectors are in shared memory
constexpr int kMoeStRstd = 23;        // the 8 rows' RMS factors
constexpr int kMoeStCut = 24;         // thread 0: the cut is known, the slot scalars requested
constexpr int kMoeStHAll = 25;        // the h loader (thread 64): every h k-block of the CTA's FC2 slots is in
constexpr int kMoeStLastFull = 26;    // the MMA warp: the CTA's last unit's weights are in
constexpr int kMoeStFc2EpiIn = 27;    // the epilogue (thread 128): the first FC2 tile's accumulator is complete
constexpr int kMoeStFc1LastFull = 28; // the MMA warp: the CTA's last FC1 unit's weights are in
constexpr int kMoeStWarm = 29;        // kMoeFlagWarmTwice: the first (warm-up) pass is done
constexpr int kMoeStPreIssue = 29;    // thread 0 in the quantization, right before its early request (shares 29)
constexpr int kMoeStTailIssued = 30;  // the producer: the first unit past the ring after FC1 is requested
constexpr int kMoeStFc2First = 31;    // the MMA warp: the first FC2 unit's MMAs are issued
// ---- kE3Tr*: per-unit stamp words (stamped builds), kStampE3 + offset, %globaltimer ----
constexpr int kE3TrUnits = 16;                   // FC2 units 0..15 of the CTA's FC2 part
constexpr int kE3TrReq = 0;                      // [i] the producer requested FC2 unit i
constexpr int kE3TrHok = kE3TrReq + kE3TrUnits;  // [i] the MMA warp: unit i's h k-block is in (after its wait)
constexpr int kE3TrFull = kE3TrHok + kE3TrUnits; // [i] the MMA warp: unit i's weights are in (after its wait)
constexpr int kE3TrIss = kE3TrFull + kE3TrUnits; // [i] the MMA warp: unit i's MMAs and commits issued
constexpr int kE3TrHkb = kE3TrIss + kE3TrUnits;  // [kb] the h loader (warp 2): k-block kb of the first slot stored
constexpr int kE3TrTile = kE3TrHkb + kMoeKb;     // [ti] the epilogue: tile ti's outputs stored (ti < 8)
constexpr int kE3TrFc1Req = kE3TrTile + 8;       // [k] the producer requested the CTA's FC1 unit n1 - 8 + k
constexpr int kE3TrFc1Full = kE3TrFc1Req + 8;    // [k] the MMA warp: FC1 unit n1 - 8 + k's weights are in
constexpr int kE3TrWords = kE3TrFc1Full + 8;     // 104
// ---- end kE3Tr* ----
