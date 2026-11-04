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

// Shapes and shared-memory layout of the full-attention stages (fullAttention.cuh); host and device. Included by
// msoMegakernel.cu inside namespace mso_mk, after kSmemMain.

namespace fa
{
constexpr int kD = 64;
constexpr int kPage = 32;
constexpr int kHkv = 2;
constexpr int kG = 8;
constexpr int kHq = kHkv * kG;
constexpr int kRows = 128;                      // TMEM lanes = query rows of both KV heads
constexpr int kQkvCols = 2 * (kHq + kHkv) * kD; // 2304: [q_0 | g_0 | ... | q_15 | g_15 | k_0 | k_1 | v_0 | v_1]
constexpr int kKCol = 2 * kHq * kD;             // 2048
constexpr int kVCol = kKCol + kHkv * kD;        // 2176
constexpr int kOutCols = kHq * kD;              // 1024
#ifndef MSO_MK_FA_RING
#define MSO_MK_FA_RING 16
#endif
#ifndef MSO_MK_FA_EXP16
#define MSO_MK_FA_EXP16 0
#endif
#if MSO_MK_FA_EXP16
#error "MSO_MK_FA_EXP16: the FP16 exps are not implemented by the two-warpgroup FaAttn"
#endif
#ifndef MSO_MK_FA_DBG
#define MSO_MK_FA_DBG                                                                                                  \
    0 // attribution builds (timing only, wrong results): bit 0 no MMAs, bit 1 no softmax arithmetic,
      // bit 2 the softmax exps replaced by one FMUL each, bit 3 QK without its four Q_lo MMAs, bit 4
      // no PV MMAs, bit 5 no S tcgen05.ld (row-dependent stand-in scores), bit 6 no P tcgen05.st /
      // wait::st, bit 7 the refills arm their slot without the TMA (stale keys), bit 8 (with bit 7)
      // no empty wait by the refill warps, bit 9 no tcgen05.fence::after_thread_sync before the PV /
      // QK MMAs
#endif
#ifndef MSO_MK_FA_RC
#define MSO_MK_FA_RC 32 // columns per tcgen05.ld / st chunk of the O rescale (16 or 32; same multiplies)
#endif
static_assert(MSO_MK_FA_RC == 16 || MSO_MK_FA_RC == 32, "MSO_MK_FA_RC: 16 or 32");
// ---- MSO_MK_FA_XQ = 1: P in shared memory and each warpgroup's QK(t + 2) issued by the other warpgroup's issuer
// (see stageFaAttn) ----
#ifndef MSO_MK_FA_XQ
#define MSO_MK_FA_XQ 0
#endif
// ---- end MSO_MK_FA_XQ ----
// ---- MSO_MK_KM: FaAttn's MMA issue (a bit mask; 0 = the per-MMA statements of stageFaAttn) ----
// bit 1: PV(t) of a full tile (4 pages) and its two commits (empty[slot], odone[wg]) as one warp-collective statement
//        with one elect.sync; the tile slot is a template argument (kmForSlot), so every B descriptor and A TMEM column
//        is a warp-uniform base plus a compile-time offset and the statement has no per-MMA branch. A tail tile (fewer
//        than 4 pages) keeps the per-MMA path.
// bit 2: QK(t) (8 MMAs) and its s_full commit likewise.
// bit 4: the issuer warp issues QK(t + 2) only after PV(t) completed (it waits on odone[wg], phase u, which the
//        epilogue drain also waits on; a wait does not change the barrier), so its QK MMAs go into a drained queue.
// bit 8: PV(t) of a full tile and its two commits as one statement with ONE code copy for every slot: the slot's first
// V
//        page descriptor is formed at run time and the other three pages' (and A's TMEM columns) are added inside the
//        statement, so the issuer's per-tile path has no slot dispatch and no per-MMA branch. Not with bit 1.
// bit 16: QK(t) (8 MMAs) and its s_full commit as one statement with one code copy likewise (Q's descriptors and the
//        slot's K descriptor offsets added inside the statement), and qkTile's step-page patch check is one compare per
//        tile (the pages of a tile are consecutive, so "some page is at or past the first new page" is "its last page
//        is"); the per-page patch loop runs only on a tile that has such a page. Not with bit 2.
// bit 32: with 8 / 16, those statements form each MMA's operands right before it in one temporary (pair) instead of
//        all of them up front: fewer registers live across the statement, the same MMAs in the same order.
// bit 64: with 8, one PV statement for every tile, a tail tile's included: MMA q is predicated on q < np (the tile's
//        pages), as the per-MMA path issues it, so the issuer's path has no branch over a tail path.
// Numerics contract: value-identical by construction. The same MMAs (same A / B operands, instruction descriptors and
// accumulate flags) are issued by the same lane (lane 0 of the converged issuer warp, as elect.sync picks it for the
// per-MMA statements) in the same order with the same commits after them, and the same pages are patched; only the
// instruction sequence that forms the operands and issues them (and, bit 4, when QK is issued) changes, so the tensor
// pipe computes the same accumulations (check builds bitwise against `check`, test_megakernel_full_attention.py).
#ifndef MSO_MK_KM
#define MSO_MK_KM 0
#endif
static_assert((MSO_MK_KM & ~127) == 0, "MSO_MK_KM: bits 1, 2, 4, 8, 16, 32 and 64");
static_assert((MSO_MK_KM & 64) == 0 || (MSO_MK_KM & 8) != 0, "MSO_MK_KM: bit 64 extends the bit 8 statement");
static_assert((MSO_MK_KM & 32) == 0 || (MSO_MK_KM & 24) != 0, "MSO_MK_KM: bit 32 shapes the bit 8 / 16 statements");
static_assert((MSO_MK_KM & 9) != 9 && (MSO_MK_KM & 18) != 18, "MSO_MK_KM: bit 8 replaces bit 1, bit 16 replaces bit 2");
// MSO_MK_KM_TL = 1 (attribution only, stamped stage-test builds): the issuer lane of each warpgroup writes clock64 at
// eight points of its iterations kKmTlU0 .. kKmTlU0 + kKmTlIters - 1 into the experts stamp words (kStampE3, unused by
// the FaAttn stage test): S observed, softmax start, P stored, wait::st, barrier, PV issued, QK(t + 2) start / end.
// MSO_MK_KM_TL = 2: iterations 3 and 4 only, 20 points each (the five softmax points, then after each PV MMA, the PV
// commits, the QK start, after each QK MMA and the QK commit; per-MMA statements, so with MSO_MK_KM = 0).
#ifndef MSO_MK_KM_TL
#define MSO_MK_KM_TL 0
#endif
static_assert(MSO_MK_KM_TL != 2 || MSO_MK_KM == 0, "MSO_MK_KM_TL = 2 stamps the per-MMA statements");
static_assert(MSO_MK_KM == 0 || (MSO_MK_FA_DBG == 0 && MSO_MK_FA_XQ == 0),
    "MSO_MK_KM: not with the FaAttn attribution builds or MSO_MK_FA_XQ");
// ---- end MSO_MK_KM ----
// ---- MSO_MK_FA_ST = 1 (attribution only, per-unit stamped builds: MSO_MK_SY_LITE = 1, where no stage writes the
// experts stamp words kStampE3): each FaAttn unit writes a clock64 record of its KV loop into those words (fa::kSt*:
// per warpgroup the S-seen, barrier-passed and slot-full times of its first fa::kStU iterations, its loop end, a
// %globaltimer anchor at Q ready, the epilogue's drain and partials-stored times, the refill warps' empty-wait cycles
// and the unit's geometry), then the epilogue's barrier and TMEM free, and the FaMerge unit that follows FaAttn on the
// same CTA writes its start, gate, per-warp poll completion (with the poll rounds), barrier and end times (fa::kStFa* /
// kStFm*). No other code changes; the timed builds never carry it. ----
#ifndef MSO_MK_FA_ST
#define MSO_MK_FA_ST 0
#endif
static_assert(MSO_MK_FA_ST == 0 || MSO_MK_FA_ST == 1, "MSO_MK_FA_ST: 0 or 1");
static_assert(MSO_MK_FA_ST == 0 || (MSO_MK_SY_LITE == 1 && MSO_MK_KM_TL == 0 && MSO_MK_FA_XQ == 0),
    "MSO_MK_FA_ST: a per-unit stamped (MSO_MK_SY_LITE) build without the km timeline or MSO_MK_FA_XQ");
// ---- end MSO_MK_FA_ST ----
// ---- MSO_MK_LM: the full layer's merge hand-off (a bit mask; 0 = fixed FaMerge rows, the gate polled first) ----
// bit 1 (T, arrival ticket): FaMerge's row is not the unit's but a ticket. At entry thread 0 takes t = an atomic add on
//        counter kCLmTicket of the launch's copy (ctx.counters, zeroed by the launch before at its dependency wait,
//        as kCClear); the first kTok * kHq arrivals merge row t, the others return at once. With the host plan of
//        layer_mk.lm_ticket_plan (a FaMerge candidate right after every CTA's FaAttn) the merges run on the CTAs whose
//        FaAttn ends first instead of on fixed hosts. Any plan with at least kTok * kHq FaMerge units is correct.
// bit 2 (S, the gate spread): with MSO_MK_FT_GATE and kIFlagQkvArena, warp w polls gate-run slot w of its row (the
//        float4 ftMergeGate's lane loads for slot w) in the same loop as its partials, leaves its two FP32 values in
//        shared memory and its even lane empties the float4 once the warp's poll exits; after the polls' barrier warp 0
//        sums slots 0 .. 7 in ascending order from +0 and rounds once to BF16, so warp 0 does not poll the gate before
//        its partials.
// Numerics contract: value-identical by construction. T: the same merge code on the same row's slots in the same order;
// only the CTA that runs it changes. S: the same words loaded from the same addresses and the same FP32 additions in
// the same order (an absent slot adds +0.0f, which is exact: the running sum starts at +0 and round-to-nearest never
// makes it -0, so s + 0 = s), the same BF16 rounding, the same emptying stores; check builds digit-identical to the lm
// = 0 build.
#ifndef MSO_MK_LM
#define MSO_MK_LM 0
#endif
static_assert((MSO_MK_LM & ~3) == 0, "MSO_MK_LM: bits 1 and 2");
static_assert((MSO_MK_LM & 2) == 0 || MSO_MK_FT_GATE == 1, "MSO_MK_LM bit 2 spreads ftMergeGate (MSO_MK_FT_GATE)");
static_assert((MSO_MK_LM & 3) == 0 || MSO_MK_FT_DRY == 0, "MSO_MK_LM: not with FaMerge's dry pass (MSO_MK_FT_DRY)");
// ---- end MSO_MK_LM ----
#ifndef MSO_MK_FA_L2PROMO
#define MSO_MK_FA_L2PROMO 256              // the KV tensor map's L2 promotion (bytes: 0, 64, 128 or 256)
#endif
constexpr int kTilePages = 4;              // a QK tile: 128 keys = 4 pages
constexpr int kTileKeys = kTilePages * kPage;
constexpr int kRing = MSO_MK_FA_RING;      // ring pages
static_assert(kRing % kTilePages == 0 && kRing / kTilePages >= 3 && kRing / kTilePages <= 8, "3 .. 8 tile slots");
constexpr int kSlots = kRing / kTilePages; // tile slots
constexpr int kMaxUnitPages = 256;         // the stock split kernel's MAX_CTA_PAGES
#ifndef MSO_MK_FA_VBOX
#define MSO_MK_FA_VBOX 1                   // 1: a page's V (both heads) as one 4 KiB TMA box; 0: two 2 KiB boxes
#endif
constexpr bool kVBox = MSO_MK_FA_VBOX != 0;
constexpr int kBox = 2048;                      // a K TMA box: one head's 32 rows of a K block
constexpr int kStageBytes = 4 * kBox;           // a page: K head 0, K head 1, V head 0, V head 1
constexpr int kBoxesPerPage = kVBox ? 3 : 4;
constexpr int kVPage = kVBox ? 2 * kBox : kBox; // ring page stride of the V part
// Region layout (offsets from the unit's shared-memory offset, which is 1 KiB aligned). The Q-norm and (m, l)
// exchanges use the kernel's reduction area (kSmemRed), free while the unit runs.
constexpr int kOffFull = 0;    // u64 [kSlots]: a tile slot's TMA bytes
constexpr int kOffEmpty = 64;  // u64 [kSlots]: a tile slot's PVs complete
constexpr int kOffSFull = 128; // u64 [2]: warpgroup wg's S buffer holds its next tile's QK
constexpr int kOffODone = 160; // u64 [2]: warpgroup wg's PVs of its latest tile complete
constexpr int kOffTmem = 256;  // u32: the TMEM base address
constexpr int kOffScal = 320;  // int32 [8]: L, p0, p1, eff, P, the KV scale (FP32 bits)
constexpr int kOffIds = 1024;  // int32 [2][kMaxUnitPages]: K and V block ids of the unit's pages
constexpr int kOffQ = kOffIds + 2 * 4 * kMaxUnitPages; // E4M3 Q_hi, Q_lo: [2 K atoms][128 rows][64 B] each (16 KiB)
// Ring: [K head 0 | K head 1] x [kRing pages][2 KiB], then V: with kVBox [kRing pages][head 0 | head 1][2 KiB], else
// [V head 0 | V head 1] x [kRing pages][2 KiB].
constexpr int kOffRing = kOffQ + 2 * 16384;
constexpr int kRingPart = kRing * kBox; // bytes of one head's K (or V) part
constexpr int kOffV = 2 * kRingPart;    // the V part, from the ring's start
// ---- with MSO_MK_FA_XQ, P of warpgroup wg (E4M3, K-major SW64: [2 key atoms][128 rows][64 B]) after the ring ----
constexpr int kOffP = kOffRing + kRing * kStageBytes;
constexpr int kPBytes = 2 * 8192;
constexpr int kRegionBytes = kOffP + (MSO_MK_FA_XQ ? 2 * kPBytes : 0);
// ---- end MSO_MK_FA_XQ's P ----
static_assert(kOffQ % 1024 == 0 && kOffRing % 1024 == 0 && kOffP % 1024 == 0, "swizzled tiles 1 KiB aligned");
static_assert(kRegionBytes <= kSmemMain, "full-attention region exceeds the unit area");
constexpr float kSmScaleLog2 = 1.4426950408889634f / 8.0f; // log2(e) / sqrt(64)
constexpr float kPre = 4.0f; // P is formed against m - 2^4 (log2 units): P <= 2^8 before the E4M3 rounding
constexpr float kTau = 4.0f; // m is raised only when the tile max exceeds it by more than 2^4 (log2 units)
constexpr int kMergeUnits = kRows;
// Instruction descriptors (kind::f8f6f4, E4M3 x E4M3 -> F32, M = 128, N = 128): QK with A and B K-major; PV with B
// MN-major.
constexpr uint32_t kIdescQk = (1u << 4) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
constexpr uint32_t kIdescPv = (1u << 4) | (1u << 16) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
} // namespace fa

// ---- kPIFlags bits 28 .. 33 (kIFlagF3*: FaAttn stage internals, runtime-selected, one binary for every setting) ----
// kIFlagF3FaPingPong: the two warpgroups' softmaxes are serialized in tile order (named barriers 3 and 4), so one
// warpgroup's exps never share the SM's issue slots / MUFU with the other's and its MMAs run under the other's softmax.
constexpr int64_t kIFlagF3FaPingPong = int64_t(1) << 28;
// kIFlagF3FaRefillSib: a slot's TMA refill is armed and issued by the warpgroup's three non-issuer warps (pages q, q +
// 3), so the issuer warp goes from its QK issue straight to its next S wait.
constexpr int64_t kIFlagF3FaRefillSib = int64_t(1) << 29;
// kIFlagFaLateLight's late-slot weight (of 16) is kFaLateWeight - ((kPIFlags >> 30) & 3): bits 30 / 31 tune it per
// layer.
constexpr int kF3FaLateWShift = 30;
// kIFlagF3FaPpLate (with kIFlagF3FaPingPong): the ordered region is only the exp / pack / P-store part of the softmax;
// the TMEM load, the max and the O rescale of one warpgroup overlap the other warpgroup's exps.
constexpr int64_t kIFlagF3FaPpLate = int64_t(1) << 32;
// kIFlagF3FaQHalf: each Q-prep thread loads (polls) only the half row it splits and the two halves' norm sums are
// exchanged with the partner thread of the other warpgroup (warps w and w + 4, named barrier 5 + w): half the Q reads,
// which every CTA of the layer makes on the same 32 KiB. The same sum order, so the same bits.
constexpr int64_t kIFlagF3FaQHalf = int64_t(1) << 33;
// The ping-pong order serializes the two softmaxes; with few tiles per warpgroup there is nothing to overlap, so it
// applies from this many tiles per unit on (100k context: 4 tiles, 450k: 17).
constexpr int kF3FaPpMinTiles = 8;
// ---- end kPIFlags bits 28 .. 33 ----

// ---- kPIFlags bits 34 .. 39 (kIFlagFa*: FaAttn stage internals, runtime-selected) ----
// kIFlagFaIssueOrder: the two issuer warps issue their MMA chains in tile order (chain 0 = QK(0), chain 1 = QK(1),
// chain t + 2 = PV(t) and QK(t + 2), chain k by warpgroup k % 2), named barriers 9 and 10 between the two issuer warps
// only. The tensor pipe then runs one warpgroup's chain at a time, so the two warpgroups' S buffers become ready one
// chain apart instead of together; the softmaxes are not ordered (unless kIFlagF3FaPingPong also is). Units of >= 2
// tiles.
constexpr int64_t kIFlagFaIssueOrder = int64_t(1) << 34;
// kIFlagFaEntryAtQ: the ring tiles the prologue does not request (FaEntryPages < the ring: fewer KV bytes in flight
// GPU-wide while every CTA's Q words are loaded) are requested by warp 0 right after its Q words landed instead of
// after Q is ready in shared memory. Request order only.
constexpr int64_t kIFlagFaEntryAtQ = int64_t(1) << 35;
// kIFlagFaQRep (with kIFlagQkvArena): QkvSum also writes the q columns into a second arena copy (kArAhQ, unused by a
// full layer's launch: BF16 [8 tokens][16 query heads][64]) and the FaAttn units on odd CTAs read it, so each Q word
// has half the readers. The same BF16 words, so the same bits.
constexpr int64_t kIFlagFaQRep = int64_t(1) << 36;
// kIFlagFaEarlyMeta: a FaAttn unit that follows the CTA's QKV run gets its geometry, the KV scale and its pages' block
// ids at kernel entry (before the dependency wait, from the immutable kv_lens / block table) into its own meta area
// (kSmemFaMeta, outside the unit regions), so the deferred prologue after the run copies them from shared memory
// instead of two dependent global round trips. Units of more than fa::kMetaIdsCap pages keep the global path.
constexpr int64_t kIFlagFaEarlyMeta = int64_t(1) << 37;

namespace fa
{
constexpr int kMetaIds = 64; // byte offset of the K ids, then the V ids (scal: int32 [8] at 0, scal[7] = the valid tag)
constexpr int kMetaIdsCap = (kSmemFaMetaBytes - kMetaIds) / 8; // 248 pages
constexpr int kMetaValid = 0x4641454d;                         // scal[7] when the meta area holds this launch's unit
} // namespace fa

// kIFlagFaQPrep (with kIFlagQkvArena): the QkvSum unit of each q block also runs the FaAttn Q prep of its head (the
// stock fold kernel's weightless RMSNorm in its sum order, the q scale, the E4M3 hi / lo split) and publishes Q_hi and
// Q_lo as Lamport words at kArAhQ ([128 FaAttn rows][64 B] each); every FaAttn unit then polls its rows' prepared bytes
// instead of the raw q and preparing them itself. The same statements on the same BF16 words, so the same bits.
// Excludes kIFlagFaQRep (same arena area).
constexpr int64_t kIFlagFaQPrep = int64_t(1) << 38;
// kIFlagFaRunL2: at kernel entry thread 0 also prefetches into L2 the weights of a QkvRun unit whose copy waits for an
// earlier unit (a second QKV run, layer_mk.plan_full_one(late_first=True)), so its copy reads L2 when its slot frees.
constexpr int64_t kIFlagFaRunL2 = int64_t(1) << 39;
// ---- end kPIFlags bits 34 .. 39 ----
