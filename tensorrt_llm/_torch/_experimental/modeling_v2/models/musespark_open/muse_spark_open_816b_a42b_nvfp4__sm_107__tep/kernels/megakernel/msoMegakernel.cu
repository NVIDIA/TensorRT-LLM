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

// The MuseSparkOpen (816B-A42B NVFP4, TEP8) decode-layer megakernel library for Rubin (sm_107a, 212 SMs), and its first
// stage: the QKV projection of a decoder layer for the 8 verify rows of a K=7 DFlash2 step.
//
// The launch is one CTA of 256 threads per SM (kCtas). Every CTA runs a list of work units fixed on the host
// (schedule.py): per unit its stage, its index, its shared-memory offset and the unit after which its weights are
// requested (-1: at entry; -2: the unit has no weight copy). The weights of a unit whose slot is free at entry go out
// before anything else the CTA does, in particular before griddepcontrol.wait, so they stream while the previous kernel
// still runs. The weight table (per CTA and unit: source, bytes) is computed on the device from unitWeights() once,
// before capture, and stored after the role tables in the parameter block.
//
// Data is its own flag. Every 32-bit word of a buffer polled by a consumer starts as kEmpty = 0x80000000 (FP32 -0.0);
// producers store lamport(w) (a payload equal to kEmpty is stored as +0) and consumers poll exactly the words they
// read, all loads in flight together, leaving each poll loop with __all_sync (a lane that leaves a spin alone splits
// the warp for the rest of the kernel) and never backing off. A buffer with one reader is emptied by that reader after
// its output (single-reader self-clear). A buffer with several readers lives in an arena of C >= 3 copies: launch n
// uses copy n % C and clear units empty copy (n + 1) % C for launch n + 1 (whose previous user, launch n + 1 - C <= n -
// 2, completed before launch n passed its griddepcontrol.wait). Counters come in C sets the same way; each launch
// zeroes the next set in its prologue.
//
// The GEMV is mma.sync m16n8k16 BF16 with M = 16 weight rows and N = the 8 rows: the weights are repacked on the host
// (pack_bf16) into the MMA's fragment order, [M/16 tiles][K/64 chunks][2048 B], with a K permutation inside each chunk
// that lets every thread read its activation fragment as 32 contiguous bytes.
//
// Compile modes: MSO_MK_LEAN = 1 compiles out every watchdog (timed builds); MSO_MK_STAMPS_MODE = 1 compiles out every
// stamp site (timed builds), 0 reads the stamp pointer at run time (check and stamped builds).

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#ifndef MSO_MK_LEAN
#define MSO_MK_LEAN 0
#endif
#if MSO_MK_LEAN != 0 && MSO_MK_LEAN != 1
#error MSO_MK_LEAN must be 0 or 1
#endif
#ifndef MSO_MK_STAMPS_MODE
#define MSO_MK_STAMPS_MODE 0
#endif
#if MSO_MK_STAMPS_MODE != 0 && MSO_MK_STAMPS_MODE != 1
#error MSO_MK_STAMPS_MODE must be 0 (runtime stamps) or 1 (no stamps)
#endif
#ifndef MSO_MK_CTAS
#define MSO_MK_CTAS 212
#endif
// Attribution builds of the QKV run's inner loop (timing only, wrong results): 1 = no MMA (the shared-memory weight
// loads only), 2 = no shared-memory loads (the MMAs on register operands only).
#ifndef MSO_MK_DBG
#define MSO_MK_DBG 0
#endif
// The QKV weight's packed order: 1 = [tile][chunk] (a CTA's run covers <= 2 tiles and up to 87 distinct x chunks);
// 4 = [K slice][tile][chunk in slice] (4 slices of 2048 columns; a run stays in one slice, so a CTA reads 32 x chunks,
// each warp 4 of them, and a tile sums the partials of its 4 slices).
#ifndef MSO_MK_QKV_SLICES
#define MSO_MK_QKV_SLICES 1
#endif
// ---- MSO_MK_LANES: the device stages compiled in (bit 0: the attention half, 1: full attention, 2: the MoE experts,
// 3: the MoE-side hops + the integration hand-offs, 4: the BF16 experts of layers 0 / 61, off by default). Host ABI
// functions and parameter fields are always present. ----
// MSO_MK_I2B_PRO: the prologue's parameter copy without thread 0 (1, the default) or with it (0).
#ifndef MSO_MK_I2B_PRO
#define MSO_MK_I2B_PRO 1
#endif
#ifndef MSO_MK_LANES
#define MSO_MK_LANES 15
#endif
// ---- end MSO_MK_LANES ----
#if MSO_MK_STAMPS_MODE == 0
#define MSO_MK_STAMPS(c) (c).stamps
#else
#define MSO_MK_STAMPS(c) static_cast<uint64_t*>(nullptr)
#endif
// ---- MSO_MK_SY_*: stamped-build options. MSO_MK_SY_CLK, the stamps' clock: 0: %globaltimer (ns, device-wide); 1:
// %clock64 (SM cycles) for every stamp except the CTA's anchors (entry, dependency wait, exit), each anchor read
// between two clock64 reads, so the host maps a CTA's cycle stamps onto the device clock (library.dump_timeline). A
// clock64 read costs a few cycles, a %globaltimer read several hundred, so the cycle build perturbs the timed chain
// far less. ----
#ifndef MSO_MK_SY_CLK
#define MSO_MK_SY_CLK 0
#endif
#if MSO_MK_SY_CLK != 0 && MSO_MK_SY_CLK != 1
#error MSO_MK_SY_CLK must be 0 or 1
#endif
// The vector poll (pollVecs) as one loop whose first pass issues the loads (1), or the first loads unrolled before the
// test-and-reissue loop (0, the default).
#ifndef MSO_MK_SY_POLL
#define MSO_MK_SY_POLL 0
#endif
#if MSO_MK_SY_POLL != 0 && MSO_MK_SY_POLL != 1
#error MSO_MK_SY_POLL must be 0 or 1
#endif
// Stamped builds: thread 0 records the unit it enters in host memory (1, the default: after a trap the record shows
// where each CTA was) or not (0: the stamped build without host-memory stores, for attribution).
#ifndef MSO_MK_SY_PROG
#define MSO_MK_SY_PROG 1
#endif
#if MSO_MK_SY_PROG != 0 && MSO_MK_SY_PROG != 1
#error MSO_MK_SY_PROG must be 0 or 1
#endif
// Stamped builds: every stamp site (0, the default) or only the per-unit stamps, the prologue and the anchors (1: no
// sub-stamps, no poll probe, and none of the stamped builds' extra barriers / barrier probes inside the stages).
#ifndef MSO_MK_SY_LITE
#define MSO_MK_SY_LITE 0
#endif
#if MSO_MK_SY_LITE != 0 && MSO_MK_SY_LITE != 1
#error MSO_MK_SY_LITE must be 0 or 1
#endif
#if MSO_MK_SY_LITE
#define MSO_MK_STAMPS_DETAIL(c) static_cast<uint64_t*>(nullptr)
#else
#define MSO_MK_STAMPS_DETAIL(c) MSO_MK_STAMPS(c)
#endif
// ---- end MSO_MK_SY_* ----
// ---- MSO_MK_PB: B-fragment input polls with one 256-bit load per fragment (pollVecs256), a bit mask per poll site:
// 1 MhUnpool's m, 2 MhPool's x_ffn, 4 QkvRun's early x (stageQkvRunSliced), 8 AhOproj's input. 0: every site with two
// 16-byte loads per fragment (pollVecs). ----
// Numerics contract of MSO_MK_PB (pb 15 in the served build, kernels/megakernel/serving.py DEFAULT_MODE):
// value-identical. Each lane polls the same 32 bytes the PB = 0 build polls as two 16-byte vectors, as one 256-bit
// volatile load (the same strength), into the same fragment registers, and keeps the same exit vote, so every fragment
// value, every sum and every output is bitwise the PB = 0 build's. With MSO_MK_BH, bit 2's site is bh's pool plane
// polls (stages/moe_hops.cuh bhPollPlane / pollVecs256Tail): the same argument. In check mode the PB builds reproduce
// the PB = 0 build's output digit for digit.
#ifndef MSO_MK_PB
#define MSO_MK_PB 0
#endif
#if MSO_MK_PB < 0 || MSO_MK_PB > 15
#error MSO_MK_PB must be a mask of 1 (MhUnpool), 2 (MhPool), 4 (QkvRun early x) and 8 (AhOproj)
#endif
// ---- end MSO_MK_PB ----
// ---- MSO_MK_AH: the sliding-window layer's attention front, a bit mask (ah 903 in the served build,
// kernels/megakernel/serving.py DEFAULT_MODE):
//  1 (GM): each merging AhSwaAttn unit sums its token's two gate heads from the gate runs' partials itself
//    (stages/attnHalf.cuh ahMerge), and the host plan holds no gate AhQkvFin unit (schedule_attn.plan_attention).
//  2 (FF): host plan only (no device code under this bit): the 20 q / k / v AhQkvFin units are role 0 of early CTAs
//    whose QKV run is pure gate, so they start right after the dependency wait instead of after a QKV run.
//  4 (FFP, with FF only): with early x and a pre-wait attention half (kIFlagXIn; kIFlagI2bPreWait in the served SWA
//    layers, or kIFlagPreWaitAttn), a CTA whose roles start with an AhQkvFin unit followed by a QKV run runs both,
//    and the attention-half units after them, before griddepcontrol.wait (the kernel's `first`), so FF's fins start
//    at entry instead of at the wait.
//  8 (GP, with FFP only): the FF CTAs hold no QKV run; their pure-gate runs are the second run of plain early
//    CTAs (QKV run, o_proj only), so GM's gate partials do not wait for a fin on the run's CTA
//    (schedule_attn.plan_attention gate_runs_off_fin). The kernel side: FFP's pre-wait rule also takes a CTA whose
//    AhQkvFin unit (role 0) is followed by any pre-wait stage.
//  16 (WD): a merging AhSwaAttn unit's merge (ahMerge) runs once dry (no polled load, no global store, no stamp;
//    shared-memory scratch only) in a slack window before its real pass, so the real pass runs from a warm instruction
//    cache. Without 32: before the CTA's QKV run, when the prologue's sample of the CTA's early x read it unpublished
//    (the x wait); with 32 (WD-Q): at the unit's start, before the pool-page conversion and the Q poll (the Q wait).
//  128 (GS, with GM, without WD): GM's gate is spread over warps 4-7 of the merging unit: thread (r, d) of
//    threads 128-255 polls the gate's run-partial words of its one gate value (row r, dim d) together with its O
//    vectors, sums them, takes the sigmoid and writes sig[r][d] (rather than warp 7 polling 8 gate vectors per lane
//    and running 4 sigmoids in a row); the gate addresses use 32-bit run arithmetic (the same integers).
//  256 (SP): the SWA block pass's key-half staging rows (stageAhSwaAttn o1) padded to 72 floats, so a warp's
//    float2 stores (half 1) and loads (half 0) of O take 2 shared-memory wavefronts instead of 8 (its 8 rows do not
//    share two banks).
//  512 (ME, with GS, without WD): the merge early, on key half 1 alone. In a merging unit warps 4-7 run the
//    whole merge (bar 1 over 128 threads) right after the block pass's key-half staging barrier, while half 0 combines
//    and publishes the unit's own partial and then waits at the unit's closing barrier: thread 128 + p polls and weighs
//    thread p's split group and its own (warp 4 also warp 0's (m, l) trees), and forms the final value (r, d) whose
//    gate it holds (GS), so sig stays in a register. There is no pre-merge barrier; each slot is emptied by the thread
//    that polled it.
//  1024 (MX, with GS, without WD and ME): the merge early, hand-off form. The merging unit's merge is called
//    by all threads at the unit's end without the pre-merge barrier, so half 1 (no combine) enters it while half 0
//    combines and publishes: thread 128 + p polls its own O vectors, the gate words and thread p's O vectors (warp 4
//    also warp 0's (m, l)) in one poll and hands thread p's vectors over in the reduction area (one barrier, which half
//    0 joins after publishing); half 0 polls nothing; the rest is GS's 256-thread merge. Each slot is emptied by the
//    thread that polled it.
// Numerics contract of MSO_MK_AH: value-identical. GM runs the gate fin's statements on the same partials in the same
// order (slices, then runs, from +0; one BF16 rounding; 1 / (1 + expf(-x)); BF16), so every sig value, merge output and
// layer output is bitwise the AH = 0 build's; FF moves units between CTAs and reorders a CTA's roles only; FFP moves
// the dependency wait of those CTAs only; GP moves QKV runs between CTAs (a run's chunks, partial slots and sums are
// its own whatever CTA holds it); WD adds a pass with no global side effect (every shared word it writes is rewritten
// by the real pass before that pass reads it); GS runs GM's statements per gate value on another thread (the gate
// statements are elementwise: the same words, the same slot order from +0, the same roundings); SP moves shared-memory
// staging addresses only; ME runs the merge's statements for half 0's threads on half 1's (the same words, the same
// split groups summed in the same order, the same final sums, roundings and stores); MX moves the polls of half 0's
// slots to half 1 (the same words, copied through shared memory unchanged).
#ifndef MSO_MK_AH
#define MSO_MK_AH 0
#endif
#if MSO_MK_AH < 0 || MSO_MK_AH > 2047 || (MSO_MK_AH & 64) || ((MSO_MK_AH & 4) && !(MSO_MK_AH & 2))                     \
    || ((MSO_MK_AH & 8) && !(MSO_MK_AH & 4)) || ((MSO_MK_AH & 32) && !(MSO_MK_AH & 16))                                \
    || ((MSO_MK_AH & 128) && (!(MSO_MK_AH & 1) || (MSO_MK_AH & 16))) || ((MSO_MK_AH & 512) && !(MSO_MK_AH & 128))      \
    || ((MSO_MK_AH & 1024) && (!(MSO_MK_AH & 128) || (MSO_MK_AH & 512)))
#error MSO_MK_AH must be a mask of 1 (GM: gate in the SWA merge), 2 (FF: fin first, host plan only), 4 (FFP: FF's fin before the wait, needs 2), 8 (GP: FF CTAs without a QKV run, needs 4), 16 (WD: dry merge pass), 32 (WD-Q: the dry pass in the Q wait, needs 16), 128 (GS: the gate over warps 4-7, needs 1, not 16), 256 (SP: padded key-half staging rows), 512 (ME: the merge early on half 1, needs 128) and 1024 (MX: the merge early with a hand-off, needs 128, not 512)
#endif
// ---- end MSO_MK_AH ----
// ---- MSO_MK_LY: code layout of the kernel's text section, attribution only (never served). ptxas places the
// kernel's callees after the kernel body in the lexicographic order of their mangled names, so renaming a function
// moves it in the text section, and every function after its old and before its new place moves by its size. A mask:
//  1: ahMerge is named zzAhMerge (it sorts after every other callee);
//  2: stageAhQkvFin is named zzAhQkvFn (likewise, after zzAhMerge).
// The code is unchanged (value-identical by construction); only addresses of whole functions move.
//  4: AH FFP's block in the kernel body is compiled without MSO_MK_AH bit 4 (a no-op twin: without FF's plan no CTA
//     starts with an AhQkvFin unit followed by a QKV run, so the block never changes `first`); it measures what the
//     block's presence alone costs (its instructions and ptxas's kernel-wide register assignment around it).
//  8: with ME (MSO_MK_AH bit 512): ME's merge entered only after half 0's publication (a barrier before the call):
//     ME's code and allocation without the early entry, so ME - this twin is the early entry alone.
#ifndef MSO_MK_LY
#define MSO_MK_LY 0
#endif
#if MSO_MK_LY < 0 || MSO_MK_LY > 15 || ((MSO_MK_LY & 4) && (MSO_MK_AH & 4)) || ((MSO_MK_LY & 8) && !(MSO_MK_AH & 512))
#error MSO_MK_LY must be a mask of 1 (ahMerge last), 2 (stageAhQkvFin last), 4 (inert FFP block, without MSO_MK_AH bit 4) and 8 (ME entered after the publication, needs MSO_MK_AH bit 512)
#endif
#if MSO_MK_LY & 1
#define ahMerge zzAhMerge
#endif
#if MSO_MK_LY & 2
#define stageAhQkvFin zzAhQkvFn
#endif
// ---- end MSO_MK_LY ----

namespace mso_mk
{

// ----------------------------------------------------------------------------------------------------------------
// Shapes
// ----------------------------------------------------------------------------------------------------------------
constexpr int kCtas = MSO_MK_CTAS;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kTok = 8;          // verify rows of a K=7 step
constexpr int kHidden = 8192;    // model width
constexpr int kMaxRoles = 16;    // units per CTA
constexpr int kBf16Block = 2048; // one (tile, chunk) of a pack_bf16 weight: 16 rows x 64 columns
constexpr int kChunkCols = 64;
constexpr uint32_t kEmpty = 0x80000000u;

// QKV projection of one rank: [8, 8192] x [2304, 8192]^T. 144 tiles of 16 rows x 128 chunks of 64 columns = 18,432
// chunks; CTA c's run is chunks [floor(18432 c / kCtas), floor(18432 (c + 1) / kCtas)) of the [tile][chunk] order.
constexpr int kQkvRows = 2304;
constexpr int kQkvTiles = kQkvRows / 16;
constexpr int kChunksPerRow = kHidden / kChunkCols;
constexpr int kQkvChunks = kQkvTiles * kChunksPerRow;
constexpr int kQkvBlockRows = 64; // a QkvSum unit: one head (q_h, g_h, k_h or v_h) = 4 tiles
constexpr int kQkvBlocks = kQkvRows / kQkvBlockRows;
constexpr int kQkvTilesPerBlock = kQkvBlockRows / 16;
constexpr int kQkvMaxRun = (kQkvChunks + kCtas - 1) / kCtas;
constexpr int kQkvMinRun = kQkvChunks / kCtas;
constexpr int kQkvMaxPerWarp = (kQkvMaxRun + kWarps - 1) / kWarps;
constexpr int kQkvSlices = MSO_MK_QKV_SLICES;
constexpr int kSegChunks = kChunksPerRow / kQkvSlices; // chunks of a segment = one (slice, tile)
static_assert(kQkvSlices == 1 || kQkvSlices == 4, "1 or 4 K slices");
static_assert(kQkvSlices == 1 || kCtas % kQkvSlices == 0, "runs stay inside one K slice");
static_assert(kSegChunks % kWarps == 0, "a warp's chunks keep their position in every segment");
// Segments one run touches (its partial slots), runs one segment is covered by, runs one tile sums.
constexpr int kQkvMaxSlots = (kQkvMaxRun - 2) / kSegChunks + 2;
constexpr int kQkvRunsPerSeg = (kSegChunks - 2) / kQkvMinRun + 2;
constexpr int kQkvMaxRuns = kQkvSlices * kQkvRunsPerSeg;
static_assert(kQkvSlices == 4 || kQkvMaxSlots == 2, "a run covers at most two tiles");
static_assert(kQkvSlices == 1 || kQkvMaxSlots <= 4, "four accumulator slots per warp");
constexpr int kQkvPartSlotBytes = 16 * kTok * 4; // FP32 [8 tokens][16 rows]
constexpr int kQkvPartBytes = kCtas * kQkvMaxSlots * kQkvPartSlotBytes;
constexpr int kSsqChunks = kChunksPerRow;        // XSsq units: one per 64-column chunk

// ---- MoE-side hops: shapes and buffer layouts --------------------------------------------------------------------
#include "stages/moe_hops_layout.cuh"
// ---- end MoE-side hops --------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------------------------------
// Arena (per copy), clear units, counters
// ----------------------------------------------------------------------------------------------------------------
constexpr int kArXSsq = 0;    // FP32 [8 tokens][128 chunks]: sum of bf16(residual)^2 over the chunk's 64 columns
constexpr int kArTest = 4096; // test stages: 64 units x 64 B
// ---- attention half (stages/attnHalf.cuh): arena data with several readers ----
constexpr int kArAhQ = 16384;    // BF16 [2 KV heads][64 rows = 8 token + query head][64]: attention Q
constexpr int kArAhKNew = 32768; // FP8 [2][8 tokens][64]: the step's K rows
constexpr int kArAhVNew = 33792; // FP8 [2][8 tokens][64]: the step's V rows
constexpr int kArAhSig = 34816;  // BF16 [8 tokens][1024]: bf16(sigmoid(gate))
constexpr int kArAhO = 51200;    // BF16 [replicas <= 4][8 tokens][1024]: the gated o_proj input
constexpr int kArenaBytes = 131072;
static_assert(kArAhO + 4 * 8 * 1024 * 2 <= kArenaBytes, "attention-half arena layout");
// ---- end attention half ----
// ---- one-launch full layer: the raw QKV projection BF16 [8 tokens][2304] (checkpoint order) as Lamport words, from
// QkvSum to FaAttn / FaMerge of the same launch (kIFlagQkvArena); after kArAhO's first two replicas (a full layer's
// launch uses at most two) ----
constexpr int kArFaQkv = kArAhO + 2 * 8 * 1024 * 2;
static_assert(kArFaQkv + 8 * 2304 * 2 <= kArenaBytes, "one-launch full layer arena layout");
// ---- end one-launch full layer ----
constexpr int kClearUnitBytes = 16384;
constexpr int kClearUnits = kArenaBytes / kClearUnitBytes;
static_assert(kArTest + 64 * 64 <= kArenaBytes, "arena layout");

enum Counter : int
{
    kCClear = 0,    // clear units done (this launch's)
    kCTest = 1,     // test stages: publisher units done
    kCLmTicket = 2, // MSO_MK_LM bit 1: FaMerge's arrival tickets (stages/fullAttention.cuh lmTicket)
    kNumCounters = 32,
};

// ----------------------------------------------------------------------------------------------------------------
// Stages (role entry = stage << 16 | unit), parameters, flags, wait codes
// ----------------------------------------------------------------------------------------------------------------
enum Stage : int
{
    kStNone = 0,
    kStQkvRun = 1,    // a chunk run of the QKV weight: FP32 partials of its <= 2 tiles
    kStQkvSum = 2,    // a 64-row block: the runs' partials summed, (1 / rms), one BF16 rounding
    kStClear = 3,     // 16 KiB of the next launch's arena copy
    kStXSsq = 4,      // (kFlagXPrime) sum of bf16(residual)^2 of one 64-column chunk per token
    kStTestCopy = 16, // tests: a unit's weight bytes from shared memory to global memory
    kStTestPub = 17,  // tests: 64 Lamport words + the publisher counter
    kStTestSub = 18,  // tests: polls a publisher's words, waits for every publisher, self-clears
    kStTestHang = 19, // tests: polls a word nobody writes (the watchdog's test)
    // ---- the SWA attention half (stages/attnHalf.cuh), stage range 10-29 ----
    kStAhQkvFin = 10,  // a 64-row head block: run sums, prep (norm, RoPE), KV append, Q / gate into the arena
    kStAhSwaAttn = 11, // (KV head, 64-key block) attention partial; b < 32 also merges two rows + norm-gate
    kStAhOproj = 12,   // 2-3 o_proj tiles into the hop1 slot
    // ---- end attention half ----
    // ---- full attention (stages/fullAttention.cuh), range 50-59 ----
    kStFaAttn = 50,  // one unit's page range of both KV heads: partial O (FP16) + log2-sum-exp per row
    kStFaMerge = 51, // one (token, query head) row: merge of the units' partials, BF16, norm-gate
    // ---- end full attention ----
    // ---- MoE experts (stages/experts*.cuh) begin ----
    kStMoeExperts = 60, // routing, placement, pre-expert quant, the routed experts, the weighted sum (one per CTA)
    // ---- end MoE experts ----
    // ---- MoE-side hops (stages/moe_hops.cuh): 80-99 ----
    kStMhOproj = 80,   // stub of the o_proj epilogue: the rank's partial into every rank's hop1 slot
    kStMhH1 = 81,      // hop1 consumer: rank-ordered sum, post-attention norm, gated residual, pre-FFN norm
    kStMhRouter = 82,  // one router expert row (FP32): 8 logits -> every rank
    kStMhPool = 83,    // a pool tile, one K half: FP32 partial -> every rank
    kStMhExpStub = 84, // stub of the experts: F2' reader, then the rank's weighted-sum partial -> every rank
    kStMhFinal = 85,   // finalize all-reduce: rank-ordered sum -> m
    kStMhUnpool = 86,  // an unpool tile: BF16 -> every rank's F1 area
    kStMhF1 = 87,      // F1 consumer: post-FFN norm, gated residual, next pre-QKV norm, capture
    kStMhClear = 88,   // the previous launch's copy of the arena and the F2' areas
    // ---- end MoE-side hops ----
    // ---- integration (stages/integration.cuh): 100-119 ----
    kStIClearX = 100,  // 16 KiB of the early-x buffer the previous launch read (kPIXBuf copy n - 1)
    kStI2bGhost = 101, // one sliding-window attention unit on scratch buffers (keeps that code warm)
    kStI2bWarm = 102,  // L2 prefetch of the next sliding-window layer's working buffers
    // ---- end integration ----
    // ---- BF16 MoE experts of layers 0 / 61 (stages/bf16Experts*.cuh): 130-139 ----
    kStMoeExpertsBf16 = 130, // the BF16 experts stage (one per CTA; the NVFP4 stage's descriptor word kPMoe)
    // ---- end BF16 MoE experts ----
};

enum Param : int
{
    kPMagic,
    kPNumCtas,
    kPMaxRoles,
    kPSmemBytes,
    kPCopy,      // this launch's copy of the arena and the counters
    kPNumCopies, // copies in the rotation (launches per replay are a multiple of it)
    kPCounters,  // u32 [copies][kNumCounters]
    kPArena,     // [copies][kArenaBytes]
    kPErr,       // pinned host int32 [16 + 5 kCtas]: the watchdog record
    kPTimeoutNs,
    kPStamps,    // u64 [kCtas][kStampsPerCta], or 0
    kPFlags,
    kPWQkv,      // pack_bf16(W_qkv): [144][128][2048 B]
    kPGammaQkv,  // (kFlagXPrime) the pre-QKV norm weight w, BF16 [8192]; the Gemma gain is (1 + w)
    kPQkvPart,   // FP32 [kCtas][2 slots][8 tokens][16 rows], every word kEmpty between launches (single reader)
    kPEps,       // (kFlagXPrime) the norm's epsilon, FP32 bits
    kPTestUnits, // tests: int64 [units][2] = (byte offset in the blob, bytes)
    kPTestBlob,  // tests: the TestCopy source bytes
    kPTestMail,  // tests: [64 units][256 B], kEmpty between launches
    kPTestPubs,  // tests: publisher units per launch
    // ---- attention half (stages/attnHalf.cuh) ----
    kPAhWOproj,     // pack_bf16(W_o): [512][16][2048 B]
    kPAhSlot,       // BF16 [8][8192]: this rank's hop1 slot (multicast address) or a plain buffer
    kPAhPool,       // the layer's FP8 pool view [blocks][2][32][64]
    kPAhBlockTable, // int32 [2][stride]: K block ids, V block ids
    kPAhBtStride,   // row stride of the block table (elements)
    kPAhSeqLen,     // int32 [1]: kv length including the step (graph input)
    kPAhPositions,  // int32 [8]: RoPE positions (graph input)
    kPAhCosSin,     // FP32 [positions][2][32]
    kPAhKvScale,    // FP32 bits: kv_scale_orig_quant
    kPAhQScale,     // FP32 bits: the query pre-scale
    kPAhPrepEps,    // FP32 bits: the Q / K norm epsilon
    kPAhGateEps,    // FP32 bits: the output norm epsilon
    kPAhPart,       // the attention partials (single reader, kEmpty between launches)
    kPAhWindow,     // sliding window
    kPAhQFp8,       // Q through the cache FP8
    kPAhOReplicas,  // copies of the gated o_proj input (1-4)
    kPAhAttnOut,    // BF16 [8][1024] ungated attention (check runs) or 0
    kPAhFlags,      // stages/attnHalf.cuh kAhFlag* bits (also keeps the parameter count even, below)
    // ---- end attention half ----
    // ---- full attention (stages/fullAttention.cuh) ----
    kPFaTmap,       // the layer pool's TMA tensor map (128 B, global): [blocks x 64 rows][64 B], box 64 x 32, SW64
    kPFaPool,       // the layer's FP8 pool view [blocks][2][32][64] (the step's K/V append)
    kPFaTable,      // int32 block table [2][stride]: row 0 K block ids, row 1 V block ids
    kPFaTableStride,
    kPFaKvLens,     // int32 [1]: L, the context including the step's 8 rows (device, read before the wait)
    kPFaKvScale,    // FP32 [1]: kv_scale_orig_quant
    kPFaPartO,      // FP16 [128 rows][units][64], kEmpty between launches (single reader)
    kPFaPartL,      // FP32 [128 rows][units], kEmpty between launches (single reader)
    kPFaUnits,      // attention units of the layer (fixed; one per CTA)
    kPFaMinPages,   // fewest pages per keyed unit: eff = min(units, ceil(P / minPages))
    kPFaEntryPages, // pages requested at entry (rounded up to tiles of 4, at most the ring; the rest after the wait)
    kPFaQScale,     // the query pre-scale, FP32 bits
    kPFaEps,        // the Q / K / output-norm epsilon, FP32 bits
    kPFaOut,        // BF16 [8][1024]: the o_proj input
    kPFaOutF32,     // FP32 [8][1024]: the merged attention before its BF16 rounding (check mode), or 0
    kPFaPrefetch,   // tiles beyond the ring prefetched into L2 ahead of their TMA (0: none)
    kPFaDbgO,       // tests: FP16 [128 rows][units][64], the units' partial O in the stock op's layout, or 0
    kPFaDbgL,       // tests: FP32 [128 rows][units], the units' log2-sum-exps (with kPFaDbgO)
    // ---- end full attention ----
    // ---- MoE experts begin ----
    kPMoe,    // the MoE experts stage's layer descriptor (int64 [kMdNumWords], stages/experts_defs.cuh)
    kPMoePad, // unused: keeps kPNumParams even (the role tables after the block are read as int4)
    // ---- end MoE experts ----
    // ---- MoE-side hops (stages/moe_hops.cuh) ----
    kPMhUc,      // the MoE hop workspace, this rank's unicast view: [copies][kMhCopyBytes]
    kPMhMc,      // the same through the multicast address (stores reach every rank)
    kPMhArena,   // the stages' local arena: [copies][kMhArenaBytes]
    kPMhRank,    // this rank in the group of 8
    kPMhWRouter, // FP32 [32][8192]: router rows [32 rank, +32)
    kPMhWPool,   // pack_bf16(W_pool[512 rank, +512)): [32 tiles][128 chunks][2048 B]
    kPMhWUnpool, // pack_bf16(W_unpool[1024 rank, +1024)): [64 tiles][64 chunks][2048 B]
    kPMhCoefH1,  // per MhH1 unit: alpha1 [W] FP32, beta1 [W] FP32, pre_ffn_norm.weight [W] BF16 (W = 8192 / U)
    kPMhCoefF1,  // per MhF1 unit: alpha2, beta2 [W] FP32, post_ffn gain (w + 1) [W] BF16, next pre_qkv weight [W] BF16
    kPMhPeers,   // int64 [8]: every rank's unicast view of the hop workspace (kFlagMhUnicast)
    kPMhScale,   // FP32 bits: the router's metap multiplier
    kPMhCapture, // BF16 [8][stride]: the DFlash capture of this layer, or 0
    kPMhCaptureStride, // elements between capture rows
    kPMhInOproj,       // stub input: BF16 [8][8192], this rank's o_proj partial
    kPMhInExpert,      // stub input: BF16 [8][4096], this rank's weighted-sum partial
    kPMhOutRes1,       // check output (or 0): FP32 [8][8192], the residual after hop1
    kPMhOutRes2,       // check output: FP32 [8][8192], the residual after F1
    kPMhOutXffn,       // check output: BF16 [8][8192], x_ffn
    kPMhOutPool,       // check output: BF16 [8][4096], the gathered pool output
    kPMhOutLogits,     // check output: FP32 [256][8], the gathered logits
    kPMhOutM,          // check output: BF16 [8][4096], the finalize all-reduce
    kPMhOutB,          // check output: BF16 [8][8192], the gathered unpool branch
    kPMhOutXattn,      // BF16 [8][8192]: the next layer's attention input (or 0)
    kPRl,              // rl (MSO_MK_RL builds only): the layer's rl block (int64 [kRlBlockWords], kRlBlock*), or 0;
                       // unused elsewhere. Also keeps kPNumParams even (see the static_assert below)
    // ---- end MoE-side hops ----
    // ---- integration ----
    kPIFlags,  // kIFlag* bits (the stage hand-offs of the integrated launches)
    kPIResOut, // FP32 [8][8192]: MhF1's r'' (instead of updating io.res in place), or 0
    kPIXBuf,   // BF16 [copies][8][8192] Lamport words: x_attn handed to the next launch (kIFlagXOut / kIFlagXIn)
    kPIPad,    // unused: keeps kPNumParams even
    // ---- end integration ----
    // ---- ghost and warm units ----
    kPI2bGhost, // int64 [kPNumParams]: the ghost units' parameter block (scratch buffers; stageI2bGhost), or 0
    kPI2bWarm,  // int64 [1 + 2 n]: n, then (address, bytes) of the buffers stageI2bWarm prefetches into L2, or 0
    // ---- end ghost and warm units ----
    kPNumParams,
};

// The role tables follow the parameter block and are read with 16-byte loads.
static_assert(kPNumParams % 2 == 0, "an even number of int64 parameters");

// ---- integration: kPIFlags bits ----
// FaMerge also stores the gated o_proj input as Lamport words into the arena's kArAhO replicas (kPAhOReplicas of them),
// for AhOproj in the same launch; FaOut = 0 drops the plain store.
constexpr int64_t kIFlagFaOutArena = 1;
// AhOproj publishes into the MoE hop workspace's hop1 area (slot rank, multicast, Lamport words), not kPAhSlot.
constexpr int64_t kIFlagOprojToMh = 2;
// MhF1 reads r' from kPMhOutRes1 (global, MhH1's check output) instead of MhH1's shared-memory region.
constexpr int64_t kIFlagMhRes1Global = 4;
// Early x: MhF1 also stores x_attn as Lamport words into copy n + 1 of kPIXBuf (kIFlagXOut); the next launch's QkvRun
// (its first unit) polls copy n of it and runs before griddepcontrol.wait (kIFlagXIn).
constexpr int64_t kIFlagXOut = 8;
constexpr int64_t kIFlagXIn = 16;
constexpr int kIXBytes = 8 * 8192 * 2;
constexpr int kIClearXUnits = 8;
constexpr int kIClearXUnitBytes = kIXBytes / kIClearXUnits;
// ---- end integration ----
// ---- one-launch full layer: QkvSum stores the raw projection as Lamport words into the arena (kArFaQkv) instead of
// io.qkv, and FaAttn / FaMerge poll it there ----
constexpr int64_t kIFlagQkvArena = 32;
// Reversed keyed units: the FaAttn role unit (= slot) s < eff computes unit eff - 1 - s (its pages and its partial
// slots), so the unit holding the step's rows (eff - 1, the append and the patches) runs on the first CTA, not on the
// last one.
constexpr int64_t kIFlagFaRevUnits = 64;
// The attention half before griddepcontrol.wait: with early x (kIFlagXIn), the CTA's leading units of stages that read
// only this launch's data and step-immutable inputs (QkvRun, QkvSum, FaAttn, FaMerge, AhQkvFin, AhSwaAttn, Clear,
// AhOproj) all run before the wait; the first other unit (the MoE half: MhH1 reads the residual) runs after it.
constexpr int64_t kIFlagPreWaitAttn = 128;
// The QKV weight is packed heads-first (q_0 .. q_15 | k_0 k_1 | v_0 v_1 | g_0 .. g_15, 64-row blocks); with
// kIFlagQkvArena QkvSum stores block b's columns at their checkpoint position (q_h -> 2 h, g_h -> 2 h + 1, k / v -> 32
// .. 35).
constexpr int64_t kIFlagQkvHeadsFirst = 256;
// Lighter FaAttn units on the late CTAs: with every unit keyed (eff == kCtas), slot s >= kCtas - kFaLateSlots gets
// kFaLateWeight / 16 of an early slot's pages (the late CTAs start their unit after the early ones in a one-launch full
// layer: they become resident last and run a QKV run + the gate QkvSum first).
constexpr int64_t kIFlagFaLateLight = 512;
constexpr int kFaLateSlots = 64;
#ifndef MSO_MK_FA_LATE_WEIGHT
#define MSO_MK_FA_LATE_WEIGHT 13
#endif
constexpr int kFaLateWeight = MSO_MK_FA_LATE_WEIGHT;
// ---- end one-launch full layer ----
// ---- kPIFlags bits 20-27 (kIFlagI2b*, kIFlagZ1FinalNorm) ----
// With kIFlagXIn: every leading attention-half unit of the CTA (QkvRun, AhQkvFin, AhSwaAttn, Clear, AhOproj) also runs
// before griddepcontrol.wait. They read only this launch's data (polled), graph inputs and immutable data; the buffers
// they write were last read by launches <= n - 2 (or emptied by their reader before x_attn was published), and launch
// n - 2 completed before any CTA of this launch started (launch n - 1 passed its wait before releasing its dependents).
constexpr int64_t kIFlagI2bPreWait = int64_t(1) << 20;
// Entry hooks: no hook for the gate QkvFin blocks (they read no metadata); the RoPE positions are loaded together with
// the sequence length.
constexpr int64_t kIFlagI2bEntryLite = int64_t(1) << 21;
// The weight requests that follow a unit (runRole) are issued by lane 0 of warp 7 instead of thread 0.
constexpr int64_t kIFlagI2bIssueWarp7 = int64_t(1) << 22;
// MhF1's next norm is the model's final RMSNorm (gain w, not 1 + w): layer 61's output is the LM-head input.
constexpr int64_t kIFlagZ1FinalNorm = int64_t(1) << 26;
// ---- end kPIFlags bits 20-27 ----
// ---- kPIFlags bits 40-49 (kIFlagT3*): canary-gated polls of the multi-reader hand-off areas ----
// A consumer CTA's warp 0 first polls one word of every producer unit whose output it reads (lane l: unit l), then
// the CTA barrier, then the unchanged full poll (which still checks every word it reads). While the producers are not
// done, a CTA re-polls 32 words instead of its whole input (no poll storm on the large hand-off areas).
constexpr int64_t kIFlagT3CanaryX = int64_t(1) << 40;   // MhPool / MhRouter: x_ffn from the 8 U MhH1 units
constexpr int64_t kIFlagT3CanaryM = int64_t(1) << 41;   // MhUnpool: m from the 32 MhFinal units
constexpr int64_t kIFlagT3CanaryXIn = int64_t(1) << 42; // QkvRun with early x in: x_attn from the 8 U MhF1 units
constexpr int64_t kIFlagT3CanaryLogits = int64_t(1)
    << 43;                                              // experts route() (coalesced poll): 32 of the 512 logit vectors
// The boundaries' row-norm exchange (MSO_MK_MH_XCHG = 1, U > 1) with the CTA's slots gathered in shared memory and
// published / polled as 16-byte vectors (same slot values, same summation order).
constexpr int64_t kIFlagT3XchgVec = int64_t(1) << 44;
// MhH1 / MhPool / MhUnpool / MhF1 wait for their (long resident) unit weights before their input poll, so nothing but
// the arithmetic follows the poll.
constexpr int64_t kIFlagT3WaitFirst = int64_t(1) << 45;
// MhF1 loads r' (kIFlagMhRes1Global) before its input poll.
constexpr int64_t kIFlagT3EarlyRes1 = int64_t(1) << 46;
// MhFinal over all 256 threads (4 columns each, 8-byte polls) instead of 128 threads x 8 columns (same sums).
constexpr int64_t kIFlagT3Final256 = int64_t(1) << 47;
// kIFlagT3WaitFirst for the o_proj units (AhOproj, the hop-1 producer) and for the early-x QKV runs (QkvRun sliced).
constexpr int64_t kIFlagT3WaitFirstOproj = int64_t(1) << 48;
constexpr int64_t kIFlagT3WaitFirstQkv = int64_t(1) << 49;
// ---- end kPIFlags bits 40-49 ----
// ---- kPIFlags bits 56-61 (kIFlagW1*): the attention half's units run their code once before their inputs
// exist, so the pass on the real inputs runs from a warm instruction cache ----
// A unit that reaches its input poll before the input is published first runs a warm pass: the same instructions with
// a run-time predicate that turns off every global load of polled data, every global store, every barrier wait on
// weights and every stamp (its math runs on zeros and on whatever the weight region holds). Then the real pass.
// QkvRun with early x in (its x not yet published):
constexpr int64_t kIFlagW1WarmQkv = int64_t(1) << 56;
// AhOproj (its gated input not yet published):
constexpr int64_t kIFlagW1WarmOproj = int64_t(1) << 57;
// AhOproj's weights, when their issue point is the CTA's QKV run, are requested by the o_proj unit itself once one word
// of attention Q is published, not by runRole right after the QKV run: the o_proj tiles (16 MB per rank) then stream
// while SwaAttn computes instead of during the QKV run -> QkvFin hand-off (for a single request, the hand-off latency
// grows with the bytes in flight). Progress: the unit waits only on QkvFin, which waits only on QKV runs (every CTA's
// first unit).
constexpr int64_t kIFlagW1OprojAfterQ = int64_t(1) << 58;
// AhSwaAttn converts its pages (entry copies of keys before the step) to the BF16 tiles before its Q poll; after Q only
// the step's own rows are converted, straight into the tiles (the same conversion, one barrier less after Q).
constexpr int64_t kIFlagW1SwaPreConvert = int64_t(1) << 60;
// ---- end kPIFlags bits 56-61 ----
// ---- MSO_MK_IC_PI_*: kPIFlags bits fixed at compile time (build modes *_icp_s4 / *_icfp_s4: MSO_MK_IC_PI_ZERO = the
// bits fixed to 0; *_icfpo_s4: also MSO_MK_IC_PI_ONE = the bits fixed to 1) ----
// Every read of the launch flags goes through icPIFlags, so the code of the other value of a fixed bit folds away (a
// smaller kernel; the same operations in the same order for the value a launch sets). The prologue traps on a launch
// that sets a bit fixed to 0 or clears a bit fixed to 1.
#ifndef MSO_MK_IC_PI_ZERO
#define MSO_MK_IC_PI_ZERO 0
#endif
#ifndef MSO_MK_IC_PI_ONE
#define MSO_MK_IC_PI_ONE 0
#endif
constexpr int64_t kIcPIZero = static_cast<int64_t>(MSO_MK_IC_PI_ZERO);
constexpr int64_t kIcPIOne = static_cast<int64_t>(MSO_MK_IC_PI_ONE);
static_assert((kIcPIZero & kIcPIOne) == 0, "MSO_MK_IC_PI_*: a kPIFlags bit is fixed to both 0 and 1");

__device__ __forceinline__ int64_t icPIFlags(int64_t raw)
{
    return (raw & ~kIcPIZero) | kIcPIOne;
}

// ---- end MSO_MK_IC_PI_* ----
// ---- MSO_MK_IC_FA_*: kPIFlags bits the served full-attention layers fix, folded only where full layers alone read
// them (build modes *_icfpoxa_s4: MSO_MK_IC_FA_ZERO / MSO_MK_IC_FA_ONE). Only a full layer runs FaAttn / FaMerge and
// the FaAttn prologue helpers, so their reads (icFaFlags) take the full layers' values; each of those entry points
// checks the launch's word (icFaCheck) and traps when the bits differ. Identity in the other builds. ----
#ifndef MSO_MK_IC_FA_ZERO
#define MSO_MK_IC_FA_ZERO 0
#endif
#ifndef MSO_MK_IC_FA_ONE
#define MSO_MK_IC_FA_ONE 0
#endif
constexpr int64_t kIcFaZero = static_cast<int64_t>(MSO_MK_IC_FA_ZERO);
constexpr int64_t kIcFaOne = static_cast<int64_t>(MSO_MK_IC_FA_ONE);
static_assert((kIcFaZero & kIcFaOne) == 0, "MSO_MK_IC_FA_*: a full-attention kPIFlags bit is fixed to both 0 and 1");
static_assert(((kIcFaZero | kIcFaOne) & (kIcPIZero | kIcPIOne)) == 0,
    "MSO_MK_IC_FA_*: the full-attention bits are not the MSO_MK_IC_PI_* bits");

__device__ __forceinline__ int64_t icFaFlags(int64_t raw)
{
    return (icPIFlags(raw) & ~kIcFaZero) | kIcFaOne;
}

__device__ __forceinline__ void icFaCheck(int64_t raw)
{
    if ((kIcFaZero | kIcFaOne) != 0 && (raw & (kIcFaZero | kIcFaOne)) != kIcFaOne)
    {
        __trap(); // a full-attention stage on a launch whose bits are not the served full layers'
    }
}

// ---- end MSO_MK_IC_FA_* ----
// ---- MSO_MK_FT_GATE: the one-launch full layer's FaMerge sums its row's gate from the gate runs' FP32 partials
// itself (MSO_MK_FT_GATE = 1, build modes *_icfpoxft_s4; the host plan then holds no gate QkvSum units) ----
#ifndef MSO_MK_FT_GATE
#define MSO_MK_FT_GATE 0
#endif
#if MSO_MK_FT_GATE != 0 && MSO_MK_FT_GATE != 1
#error MSO_MK_FT_GATE must be 0 or 1
#endif
// ---- MSO_MK_FT_DRY: FaMerge runs its merge code once on zeros (no stores) before its polls, so the real pass after
// the last partial runs from a warm instruction cache (MSO_MK_FT_DRY = 1, build modes *_icfpoxftd_s4) ----
#ifndef MSO_MK_FT_DRY
#define MSO_MK_FT_DRY 0
#endif
#if MSO_MK_FT_DRY != 0 && MSO_MK_FT_DRY != 1
#error MSO_MK_FT_DRY must be 0 or 1
#endif
// ---- end MSO_MK_FT_* ----
// ---- MSO_MK_LD: code a served lean launch never executes, compiled out of the lean builds (a bit mask; no effect
// unless MSO_MK_LEAN = 1). Every removal is a path the served launches never take; where a served precondition replaces
// a run-time test, the launch traps when the precondition does not hold.
//   1 (KF): the kernel Flags word is the served one, kFlagEvictFirst | kFlagOnePiece (no kFlagXPrime / kFlagCheck /
//     kFlagL2Deferred / kFlagMhUnicast paths); the prologue traps on another word (kFlagNoStamps aside).
//   2 (CK): no check / test outputs (kPAhAttnOut, kPFaOut, kPFaOutF32, kPFaDbgO, kPMhOut{Xffn,Pool,Logits,M,Res2,B},
//     io.qkv / io.qkvF32); the prologue traps on a launch that passes one.
//   4 (DS): the dispatchers hold only the served stages (no test stages, XSsq, ghost / warm units, MoE hop stubs, the
//     rank-local router); an unknown stage traps.
//   8 (SW): the sliding-window stages take the served SWA layers' values: kIFlagW1SwaPreConvert and
//     kIFlagI2bEntryLite set, kPAhQFp8 = 1 (AhSwaAttn / AhQkvFin / the entry hook trap otherwise).
//  16 (FL): QkvSum and the full-attention stages read their launch bits through icFaFlags with kIFlagQkvHeadsFirst set,
//     and FaMerge publishes one o_proj replica (QkvSum traps on another launch word, FaMerge on kPAhOReplicas != 1).
//  32 (FG): the served full layers' FaAttn geometry: kPFaUnits = kCtas, kPFaMinPages = 4, kPFaEntryPages = the ring,
//     kPFaPrefetch = 0 (faPrologue, FaAttn and FaMerge trap on other fields; stages/fullAttention.cuh ldFgCheck).
//  64 (R1): one runRole instance in the kernel body: the roles before and after the grid-dependency wait run in one
//     loop that reaches the wait block at role `first` (the same operations in the same order).
// 128 (OP, needs KF): a weight request is KF's one piece written out: piece 0 carries every byte, pieces 1.. arrive
//     with 0 bytes (issuePieces' loop with onePiece = true, without the empty pieces' range tests; the request loops
//     stay rolled).
// 256 (SC): served constants: the workspace / arena copy neighbours without an integer division (the prologue traps
//     unless 0 <= copy < numCopies), AhOproj's o_proj replica as a mask (kPAhOReplicas a power of two, else a trap),
//     and no kMdOut copy under MhF2 (the experts stage traps on a descriptor with one). ----
#ifndef MSO_MK_LD
#define MSO_MK_LD 0
#endif
#define MSO_MK_LD_KF ((MSO_MK_LD & 1) && MSO_MK_LEAN)
#define MSO_MK_LD_CK ((MSO_MK_LD & 2) && MSO_MK_LEAN)
#define MSO_MK_LD_DS ((MSO_MK_LD & 4) && MSO_MK_LEAN)
#define MSO_MK_LD_SW ((MSO_MK_LD & 8) && MSO_MK_LEAN)
#define MSO_MK_LD_FL ((MSO_MK_LD & 16) && MSO_MK_LEAN)
#define MSO_MK_LD_FG ((MSO_MK_LD & 32) && MSO_MK_LEAN)
#define MSO_MK_LD_R1 ((MSO_MK_LD & 64) && MSO_MK_LEAN)
#define MSO_MK_LD_OP ((MSO_MK_LD & 128) && MSO_MK_LEAN)
#define MSO_MK_LD_SC ((MSO_MK_LD & 256) && MSO_MK_LEAN)
#if MSO_MK_LD_OP && !MSO_MK_LD_KF
#error "ld OP writes out KF's one-piece request: MSO_MK_LD needs bit 1 with bit 128"
#endif
// MSO_MK_COPY_NEXT(c, n) / MSO_MK_COPY_PREV(c, n): copy (c + 1) % n / (c - 1) % n of n cyclic copies, 0 <= c < n
// (SC: a select instead of the division).
#if MSO_MK_LD_SC
#define MSO_MK_COPY_NEXT(c, n) ((c) + 1 == (n) ? 0 : (c) + 1)
#define MSO_MK_COPY_PREV(c, n) ((c) == 0 ? (n) -1 : (c) -1)
#else
#define MSO_MK_COPY_NEXT(c, n) (c + 1) % n
#define MSO_MK_COPY_PREV(c, n) (c + n - 1) % n
#endif
// MSO_MK_KFLAGS(f): a read of the kernel Flags word (the word itself unless KF).
#if MSO_MK_LD_KF
#define MSO_MK_KFLAGS(f) (int64_t(kFlagEvictFirst) | int64_t(kFlagOnePiece))
#else
#define MSO_MK_KFLAGS(f) f
#endif
// MSO_MK_CKP(v): a check / test output field (0 with CK); MSO_MK_CKPTR(T, ctx, idx): its pointer (nullptr with CK).
// MSO_MK_CKIO(p): a check output pointer of Io (nullptr with CK).
#if MSO_MK_LD_CK
#define MSO_MK_CKP(v) int64_t(0)
#define MSO_MK_CKPTR(T, ...) static_cast<T*>(nullptr)
#define MSO_MK_CKIO(p) static_cast<void*>(nullptr)
#else
#define MSO_MK_CKP(v) v
#define MSO_MK_CKPTR(T, ...) ptr<T>(__VA_ARGS__)
#define MSO_MK_CKIO(p) p
#endif
// MSO_MK_FL_QKVSUM(raw): QkvSum's read of the launch bits (FL: the full layers' bits, which every served QkvSum unit
// runs under, with kIFlagQkvHeadsFirst).
#if MSO_MK_LD_FL
#define MSO_MK_FL_QKVSUM(raw) (icFaFlags(raw) | kIFlagQkvHeadsFirst)
#else
#define MSO_MK_FL_QKVSUM(raw) icPIFlags(raw)
#endif
// ---- end MSO_MK_LD ----
// ---- MSO_MK_RL: the router logits by linearity (a bit mask; default 0 = none of the rl code is compiled) ----
//   1 (build): kIFlagRlOn. MhRouter's units compute the rank's A / B logit partials from the layer's residual and the
//     gated o_proj input (stageRlLogit, the weight slabs of the rl block), and MhH1 forms the logits from the 8 ranks'
//     partials with its own s1, s2 (rlFinalize) into its rank's F2' logit area.
//   2 (record): kIFlagRlRecord. MhF1 copies its slice of the F2' areas and MhH1 its residual and o_proj input slices
//     into the rl block's record buffer (the FP64 check of rl.py).
//   4 (oracle, attribution only, never served): kIFlagRlOracleLg / kIFlagRlOraclePool. MhH1 stores the recorded logit
//     (and pool) words of its slice into this launch's F2' areas right after its boundary arithmetic.
//   8 (with 1; needs MSO_MK_RP bit 1 and MSO_MK_NB bit 1): the rank's logits formed once by rp's publisher CTA at its
//     stage entry (rlPublisherLogits, stages/moe_hops.cuh) instead of by the MhH1 units, which then run as stock.
//   16 (with 1): router-free: every router slot runs rl's units (kIFlagRlOn on every layer, served only with
//     MUSESPARK_OPEN_MEGAKERNEL_RL=all), so stageMhRouter is a trap and the stock router code leaves the binary (bh's
//     bit 8 does the same for a bh build). The same values as the build without bit 16 at RL=all.
// Numerics contract of MSO_MK_RL: the build's router input differs from stock's. The logits are formed from FP32
// composite weights (W_R diag(g alpha1) and W_R diag(g beta1) W_o,r, made in FP64 and rounded to FP32 once) and FP32
// sums in a fixed order, so they skip stock's BF16 roundings of the o_proj sum, the gated residual and x_ffn; the
// residual stream is unchanged. The routing can therefore differ from stock's on near-tied top-8 scores. The served
// switch is MUSESPARK_OPEN_MEGAKERNEL_RL (serving.py; unset = the build's rl, on for the served build) on an
// MSO_MK_RL build.
// Why it is not a precision change: every value the router reads stays at least as precise as stock's (FP32
// composites and FP32 sums with FP32 accumulation instead of BF16-rounded inputs to an FP32-accumulated GEMV); no
// stored tensor changes its type, and the residual stream, x_ffn and the expert inputs keep their bits. What moves is
// the association of the logit sum (A + s1 B over the 8 ranks' partials, in rank order) and where rounding happens.
// The record bit (2) feeds rl.py's check of the logits against an FP64 reference (rl.reference) and of their top-8
// sets against the reference's (rl.top8_stats).
// The switch is kept because the build is not bitwise the stock layers.
#ifndef MSO_MK_RL
#define MSO_MK_RL 0
#endif
#if MSO_MK_RL & ~31
#error "MSO_MK_RL: bits 1 (build), 2 (record), 4 (oracle), 8 (the publisher forms the logits) and 16 (router-free) only"
#endif
#if (MSO_MK_RL & 8) && !(MSO_MK_RL & 1)
#error "MSO_MK_RL bit 8 is a placement of bit 1's logits"
#endif
#if (MSO_MK_RL & 16) && !(MSO_MK_RL & 1)
#error "MSO_MK_RL bit 16 (router-free) needs bit 1's rl units in every router slot"
#endif
// kPRl points at the layer's rl block: int64 [kRlBlockWords] = the record / oracle buffer (kRlRecBytes: the F2' areas
// as the hop workspace lays them out (pool [2][8][4096] FP32, logits [256][8] FP32), then the residual FP32 [8][8192]
// and this rank's gated o_proj input BF16 [8][1024]), the rl workspace through the multicast address and this rank's
// unicast view ([copies][kRlCopyBytes] each), and the rl units' weight slabs ([kRlUnits][kRlUnitBytes] FP32).
constexpr int kRlBlockRec = 0;
constexpr int kRlBlockMc = 1;
constexpr int kRlBlockUc = 2;
constexpr int kRlBlockW = 3;
constexpr int kRlBlockWords = 4;
constexpr int64_t kIFlagRlOracleLg = int64_t(1) << 52;
constexpr int64_t kIFlagRlRecord = int64_t(1) << 53;
constexpr int64_t kIFlagRlOraclePool = int64_t(1) << 54;
constexpr int64_t kIFlagRlOn = int64_t(1) << 55;

// ---- end MSO_MK_RL ----

enum Flag : int64_t
{
    kFlagEvictFirst = 1, // weight copies, KV pages and clears under the L2 evict-first policy
    kFlagNoStamps = 2,   // no stamps even with a stamp buffer
    kFlagXPrime = 4, // QKV input = the FP32 residual: x' = bf16(bf16(r) (1 + w)) in registers, 1 / rms after the GEMV
    kFlagCheck = 8,  // check mode: FP32 station outputs (QkvSum: the FP32 sums)
    kFlagL2Deferred = 16,              // at entry, the weights of the units whose slot is busy are prefetched into L2
    kFlagOnePiece = 32,                // every weight copy as one request on its first barrier (else kPieces requests)
    kFlagMhUnicast = int64_t(1) << 32, // MoE-side hop publications as one unicast store per rank (else multicast)
};

// Watchdog wait-site codes (err[0] of a trap record, err[16 + 4 cta] of a waiter's log).
enum WaitCode : int
{
    kCodeWeights = 101, // waitBar: a unit's weight copy
    kCodeQkvPart = 110, // QkvSum: the runs' partials
    kCodeQkvSsq = 111,  // QkvSum: the XSsq partials
    kCodeCounter = 120, // waitCounter
    kCodeTestSub = 130,
    kCodeTestHang = 131,
    // ---- full attention ----
    kCodeFaFull = 150,  // a tile slot's TMA bytes
    kCodeFaS = 151,     // an S buffer's QK commit
    kCodeFaDrain = 152, // a tile's PVs complete (O rescale, epilogue)
    kCodeFaPart = 153,  // FaMerge: the units' partials
    kCodeFaEmpty = 154, // a tile slot's PV commit (refill)
    kCodeFaQkv = 155,   // FaAttn / FaMerge: the raw projection in the arena (kIFlagQkvArena)
    // ---- end full attention ----
    // ---- MoE-side hops ----
    kCodeMhHop1 = 180, // MhH1: the hop1 slots
    kCodeMhSsq = 181,  // MhH1 / MhF1: a row's norm-exchange slots
    kCodeMhX = 182,    // MhPool / MhRouter: x_ffn
    kCodeMhF2 = 183,   // MhExpStub: the gathered pool output / logits
    kCodeMhFin = 184,  // MhFinal: the finalize slots
    kCodeMhM = 185,    // MhUnpool: m
    kCodeMhF1 = 186,   // MhF1: the gathered unpool branch
    // ---- end MoE-side hops ----
};

constexpr int64_t kMagic = 0x4d534f4d4b503241LL; // "MSOMKP2A"

// Stamps per CTA (u64 %globaltimer): [role][start, inputs in (thread 0, last poll), weights in, end], then entry,
// grid-dependency wait return, exit, the SM id, entry weight requests issued, then kSubStamps points.
constexpr int kStampEntry = kMaxRoles * 4;
constexpr int kStampWait = kStampEntry + 1;
constexpr int kStampExit = kStampEntry + 2;
constexpr int kStampSmid = kStampEntry + 3;
constexpr int kStampIssued = kStampEntry + 4;
constexpr int kStampFirst = kStampEntry + 5; // thread 0: the entry-issued units' first requests are out
// ---- prologue stamps (thread 0) ----
constexpr int kStampCtx = kStampEntry + 6; // the parameter block and Ctx are in shared memory
constexpr int kStampPro = kStampEntry + 7; // warp 0's entry issue loop (weights, entry copies) is done
// ---- end prologue stamps ----
constexpr int kStampSub = kStampEntry + 8;
constexpr int kSubStamps = 32;
// ---- the SM clock (clock64) at entry and at exit, for the CTA's mean SM clock rate ----
constexpr int kStampClk0 = kStampSub + kSubStamps;
constexpr int kStampClk1 = kStampClk0 + 1;
// ---- end SM clock ----
// ---- the experts stage's attribution words (kStampI2a*) ----
constexpr int kStampI2aRaw6 = kStampClk1 + 1;    // raw (E << 32) | units of the CTA's experts unit
constexpr int kStampI2aXStaged = kStampClk1 + 2; // experts thread 0: X staged (end of gatherX)
// ---- end experts attribution words ----
// ---- kStampE3: the experts stage's per-unit stamp record (p4a::kE3Tr* offsets, kE3TrWords words) ----
constexpr int kStampE3 = kStampClk1 + 3;
constexpr int kStampE3Words = 104;
// ---- kStampFa: the deferred FaAttn prologue (faAfterQkvRun, thread 0): start, warp 0's part done, ring issued,
// spare ----
constexpr int kStampFa = kStampE3 + kStampE3Words;
constexpr int kStampFaWords = 4;
// ---- end kStampFa ----
// ---- kStampSy: the poll probe (thread 0, stamped builds): per role two classes of polls (syClass: 0 = a stage's input
// poll, 1 = poll1 and the merge / experts-pool polls), each [start, second round's loads out, end] clock64 of its last
// poll and (wait code << 32) | exit-test rounds; then the anchors: clock64 before / after the wait's and the exit's
// %globaltimer read ----
constexpr int kStampSy = kStampFa + kStampFaWords;
constexpr int kStampSyRole = 8;
constexpr int kStampSyAnchor = kStampSy + kStampSyRole * kMaxRoles;
constexpr int kStampSyWords = kStampSyRole * kMaxRoles + 4;
constexpr int kStampsPerCta = kStampSy + kStampSyWords;
// ---- end kStampSy ----
// ---- end kStampE3 ----

// Shared memory: unit regions [0, kSmemMain), then the reduction area and the small area.
#ifndef MSO_MK_SMEM_MAIN
#define MSO_MK_SMEM_MAIN 200704
#endif
constexpr int kSmemMain = MSO_MK_SMEM_MAIN;
constexpr int kSmemRed = kSmemMain;
constexpr int kSmemRedBytes = 16384;
constexpr int kSmemSmall = kSmemRed + kSmemRedBytes;
constexpr int kSmemSmallBytes = 4096;
// ---- kIFlagFaEarlyMeta's meta area (a deferred FaAttn unit's geometry and page ids, filled at kernel entry) ----
constexpr int kSmemFaMeta = kSmemSmall + kSmemSmallBytes;
constexpr int kSmemFaMetaBytes = 2048;
constexpr int kSmemBytes = kSmemFaMeta + kSmemFaMetaBytes;
// ---- end kIFlagFaEarlyMeta's meta area ----
static_assert(kSmemMain % 1024 == 0, "unit regions are 1 KiB aligned");

// A unit's weight copy goes out as kPieces requests, each completing on its own barrier, so the unit can start on the
// first piece while the others land.
constexpr int kPieces = 4;
constexpr int kSmallBars = 0;       // kMaxRoles x kPieces barriers (u64): role i piece p at 8 (kPieces i + p)
constexpr int kSmallRoles = 512;    // 3 x kMaxRoles int32: units, offsets, issue points
constexpr int kSmallScalars = 768;  // 128 floats
constexpr int kSmallParams = 1280;  // the parameter block (int64), copied at entry
constexpr int kSmallCtx = 2304;     // struct Ctx
constexpr int kSmallWeights = 2560; // the CTA's weight-table row: kMaxRoles x (src0, src1, n0 | n1 << 32)
static_assert(kMaxRoles * kPieces * 8 <= kSmallRoles, "barriers");
static_assert(kSmallRoles + 3 * kMaxRoles * 4 <= kSmallScalars, "role rows");
static_assert(kSmallParams + kPNumParams * 8 <= kSmallCtx, "parameter block does not fit");
static_assert(kSmallWeights + kMaxRoles * 3 * 8 <= kSmemSmallBytes, "weight-table row does not fit");
// ---- the parameter copy uses threads 0-63 and 192-255 (up to 128 fields) ----
static_assert(kPNumParams <= 128 && kThreads == 256, "the parameter copy uses threads 0-63 and 192-255");
// ---- end parameter copy ----
// ---- MoE experts begin ----
static_assert(kPNumParams % 2 == 0, "the role tables after the parameter block are read as int4: keep the count even");
// ---- end MoE experts ----
// The role tables follow the parameter block and the kernel reads them as int4: the block must end on 16 bytes.
static_assert(kPNumParams % 2 == 0, "pad the parameter block to an even number of fields");

// The role tables (int32 [3][kCtas][kMaxRoles]) and the weight table (int64 [kCtas][kMaxRoles][3]) after the
// parameter block, in int64 words.
constexpr int kTableWords = 3 * kCtas * kMaxRoles / 2;
constexpr int kWeightWords = kCtas * kMaxRoles * 3;
static_assert((3 * kCtas * kMaxRoles) % 2 == 0, "tables end on an int64 boundary");
// Units whose entry-time weight request thread 0 makes before the parameter block is in shared memory.
constexpr int kFirstEarlyRoles = 4;

// Per-step pointers (a kernel argument, so a captured graph keeps the addresses of the buffers it was captured with).
struct Io
{
    void const* x;   // BF16 [8][8192]: the normed QKV input (mode A)
    void const* res; // FP32 [8][8192]: the residual (kFlagXPrime)
    void* qkv;       // BF16 [8][2304]: the QKV output
    void* qkvF32;    // FP32 [8][2304] (kFlagCheck), or null
    void* out0;      // tests
    void* out1;      // tests
    void const* in0; // tests
    void const* in1; // tests
};

// ---- full attention ----
#include "stages/fullAttentionConfig.cuh"
// ---- end full attention ----
// ---- MoE experts begin ----
#include "stages/experts_defs.cuh"
// ---- end MoE experts ----
// ---- BF16 MoE experts ----
#include "stages/bf16Experts_defs.cuh"
// ---- end BF16 MoE experts ----

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000

// ----------------------------------------------------------------------------------------------------------------
// PTX helpers
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smemU32(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint64_t globalTimer()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// ---- stamp clock: the value of a stamp (MSO_MK_SY_CLK: %clock64 cycles, else %globaltimer ns) ----
__device__ __forceinline__ uint64_t stampTime()
{
#if MSO_MK_SY_CLK
    return clock64();
#else
    return globalTimer();
#endif
}

// clock64 once the register t holds its value (the read waits for t), so [clock before, this] brackets t's read.
__device__ __forceinline__ uint64_t clockAfter(uint64_t t)
{
    uint64_t c;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) : "l"(t));
    return c;
}

// A %globaltimer anchor of the CTA's cycle stamps: words [k, k + 1] = clock64 before / after the read; returns it.
__device__ __forceinline__ uint64_t syAnchor(uint64_t* stamps, int k)
{
    uint64_t const c0 = clock64();
    uint64_t const g = globalTimer();
    uint64_t const c1 = clockAfter(g);
    stamps[kStampSyAnchor + k] = c0;
    stamps[kStampSyAnchor + k + 1] = c1;
    return g;
}

// ---- end stamp clock ----

__device__ __forceinline__ uint32_t smid()
{
    uint32_t v;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
    return v;
}

__device__ __forceinline__ void mbarInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void fenceMbarInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarExpectTx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbarTryWait(uint32_t bar, uint32_t parity)
{
    uint32_t ok;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
    return ok != 0;
}

__device__ __forceinline__ void bulkCopy(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

// The same with an L2 cache policy (read-once streams: evict first, so they do not push the kernel's code, the
// activations and the KV rows out of L2).
__device__ __forceinline__ void bulkCopyHint(
    uint32_t dst, void const* src, uint32_t bytes, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
            "r"(dst),
        "l"(src), "r"(bytes), "r"(bar), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void prefetchBulkL2(void const* src, uint32_t bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
}

__device__ __forceinline__ void prefetchBulkL2Hint(void const* src, uint32_t bytes, uint64_t policy)
{
    asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(src), "r"(bytes), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ uint64_t policyEvictFirst()
{
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

__device__ __forceinline__ void fenceProxyAsyncSmem()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void fenceProxyAsyncGlobal()
{
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

__device__ __forceinline__ uint32_t ldAcquire(uint32_t const* p)
{
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void redReleaseAdd(uint32_t* p, uint32_t v)
{
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void redRelaxedAdd(uint32_t* p, uint32_t v)
{
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint4 ldcg128(void const* p)
{
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float4 ldcgF4(void const* p)
{
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ float2 ldcgF2(void const* p)
{
    float2 v;
    asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ldPoll128(void const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldPoll32(void const* p)
{
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void st128(void* p, uint4 v)
{
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// st128 under an L2 cache policy (createpolicy).
__device__ __forceinline__ void st128Hint(void* p, uint4 v, uint64_t policy)
{
    asm volatile("st.global.L2::cache_hint.v4.b32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ void st64(void* p, uint2 v)
{
    asm volatile("st.global.v2.b32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void st32(void* p, uint32_t v)
{
    asm volatile("st.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint4 lds128(uint32_t addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float bf16Round(float x)
{
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ uint32_t packBf16(float lo, float hi)
{
    __nv_bfloat162 const v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t const*>(&v);
}

__device__ __forceinline__ float bf16Lo(uint32_t v)
{
    return __uint_as_float(v << 16);
}

__device__ __forceinline__ float bf16Hi(uint32_t v)
{
    return __uint_as_float(v & 0xffff0000u);
}

// ----------------------------------------------------------------------------------------------------------------
// Kernel context
// ----------------------------------------------------------------------------------------------------------------
struct Ctx
{
    int64_t const* p; // the parameter block (shared memory)
    uint8_t* smem;
    uint32_t smemBase;
    uint32_t* counters; // this launch's set
    int* err;
    uint64_t timeoutNs;
    uint8_t* arena;   // this launch's copy
    uint64_t* stamps; // this CTA's stamps, or null
    int64_t flags;
    int copy;
    int numCopies;
    int stampRole; // the running unit's index in the CTA's list
    int w1Early;   // the CTA's early x was not published when the prologue sampled it (kIFlagW1WarmQkv)
};

static_assert(sizeof(Ctx) <= 256, "Ctx exceeds its shared-memory slot");

template <typename T>
__device__ __forceinline__ T* ptr(Ctx const& ctx, int idx)
{
    return reinterpret_cast<T*>(ctx.p[idx]);
}

template <typename T>
__device__ __forceinline__ T* arena(Ctx const& ctx, int off)
{
    return reinterpret_cast<T*>(ctx.arena + off);
}

__device__ __forceinline__ void subStamp(Ctx const& ctx, int k)
{
    if (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && threadIdx.x == 0)
    {
        MSO_MK_STAMPS(ctx)[kStampSub + k] = stampTime();
    }
}

// The running unit's inputs are in (thread 0's view; the last call of the unit wins).
__device__ __forceinline__ void stampInput(Ctx const& ctx)
{
    if (MSO_MK_STAMPS(ctx) != nullptr && threadIdx.x == 0)
    {
        MSO_MK_STAMPS(ctx)[ctx.stampRole * 4 + 1] = stampTime();
    }
}

// A watchdog expired: the record goes to host memory (readable after the trap), then the kernel traps.
__device__ __noinline__ void fail(Ctx const& ctx, int code, int a, int b)
{
    int volatile* err = ctx.err;
    err[1] = blockIdx.x;
    err[2] = threadIdx.x;
    err[3] = a;
    err[4] = b;
    err[0] = code;
    __threadfence_system();
    __trap();
}

// Half way to the timeout, a waiting warp's lane 0 logs (code, thread, a, b) into its CTA's slot of the record
// (entries 16 + 4 cta), so a hang shows what every CTA waits for, not only the first to trap.
__device__ __noinline__ void logWaiter(Ctx const& ctx, int code, int a, int b)
{
    int volatile* err = ctx.err + 16 + 4 * blockIdx.x;
    err[1] = static_cast<int>(threadIdx.x);
    err[2] = a;
    err[3] = b;
    err[0] = code;
    __threadfence_system();
}

// The watchdog of a spin (out of line, so every wait inlines only its turn count): the wait began at about t0.
__device__ __noinline__ void spinSlow(Ctx const& ctx, uint64_t t0, int code, int a, int b)
{
    uint64_t const waited = globalTimer() - t0;
    if (waited > ctx.timeoutNs)
    {
        fail(ctx, code, a, b);
    }
    if (waited > ctx.timeoutNs / 2 && (threadIdx.x & 31) == 0)
    {
        logWaiter(ctx, code, a, b);
    }
}

// One more turn of a spin. t0: the clock at the first turn, plus one per turn since; every 256 turns the watchdog
// runs. No back-off: it would delay every hand-off.
__device__ __forceinline__ void spinCheck(Ctx const& ctx, uint64_t& t0, int code, int a, int b)
{
#if MSO_MK_LEAN
    return;
#else
    if (t0 == 0)
    {
        t0 = globalTimer();
    }
    else if ((++t0 & 0xffull) == 0)
    {
        spinSlow(ctx, t0, code, a, b);
    }
#endif
}

// The exit test of a poll loop, taken by the whole warp at once (all 32 lanes call it). A lane that leaves a spin
// early is let go on alone after a long spin and the warp stays split for the rest of the kernel (ptxas drops a
// __syncwarp() after the loop as redundant), so every later warp collective takes its slow path.
__device__ __forceinline__ bool warpAll(bool ok)
{
    return __all_sync(0xffffffffu, ok);
}

// ----------------------------------------------------------------------------------------------------------------
// Data as its own flag: every 32-bit word of a polled buffer is kEmpty until its producer stores it.
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lamport(uint32_t w)
{
    return w == kEmpty ? 0u : w;
}

__device__ __forceinline__ uint4 lamport4(uint4 v)
{
    return make_uint4(lamport(v.x), lamport(v.y), lamport(v.z), lamport(v.w));
}

__device__ __forceinline__ bool full4(uint4 v)
{
    return (v.x != kEmpty) & (v.y != kEmpty) & (v.z != kEmpty) & (v.w != kEmpty);
}

__device__ __forceinline__ uint4 emptyVec()
{
    return make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
}

// ---- the poll probe of thread 0 (stamped builds; kStampSy): the poll's first loads are out at clock c0, its
// second round's at c1 (0: one round), it left after `rounds` exit tests; class k of the running role takes
// [c0, c1, exit clock, (code << 32) | rounds]. Stores only, so the probe adds no round trip to the chain. ----
__device__ __forceinline__ uint64_t* syProbe(Ctx const& ctx)
{
    return (MSO_MK_STAMPS_DETAIL(ctx) != nullptr && threadIdx.x == 0)
        ? MSO_MK_STAMPS(ctx) + kStampSy + kStampSyRole * ctx.stampRole
        : nullptr;
}

// 1: poll1 (canaries, exchange slots) and the merge's / the experts' pool polls; 0: every other pollVecs.
__device__ __forceinline__ int syClass(bool vecs, int code)
{
    return (!vecs || code == 144 || code == 161) ? 1 : 0;
}

__device__ __forceinline__ void syProbeEnd(uint64_t* sy, int k, uint64_t c0, uint64_t c1, uint32_t rounds, int code)
{
    if (sy != nullptr)
    {
        sy[4 * k] = c0;
        sy[4 * k + 1] = c1;
        sy[4 * k + 2] = clock64();
        sy[4 * k + 3] = (static_cast<uint64_t>(static_cast<uint32_t>(code)) << 32) | rounds;
    }
}

// The probe's second-round clock (stamped builds): taken once, when the second round's loads are out.
__device__ __forceinline__ void syProbeRound(uint64_t const* sy, uint32_t rounds, uint64_t& c1)
{
    if (sy != nullptr && rounds == 1)
    {
        c1 = clock64();
    }
}

// ---- end poll probe ----

// One word, the whole warp (every lane calls it; a lane without a word passes p = nullptr).
__device__ __forceinline__ uint32_t poll1(Ctx const& ctx, void const* p, int code)
{
    uint32_t v = p != nullptr ? ldPoll32(p) : 0u;
    uint64_t* const sy = syProbe(ctx);                      // poll probe
    uint64_t const syC0 = sy != nullptr ? clock64() : 0ull; // poll probe
    uint64_t syC1 = 0;                                      // poll probe
    uint32_t syRounds = 1;                                  // poll probe
    uint64_t t0 = 0;
    while (!warpAll(v != kEmpty))
    {
        spinCheck(ctx, t0, code, 0, 0);
        if (v == kEmpty)
        {
            v = ldPoll32(p);
        }
        syProbeRound(sy, syRounds, syC1);                             // poll probe
        ++syRounds;                                                   // poll probe
    }
    syProbeEnd(sy, syClass(false, code), syC0, syC1, syRounds, code); // poll probe
    return v;
}

// kN 16-byte vectors at addresses a[i] (nullptr: none), all loads in flight together, then again for the ones not
// complete; the whole warp calls it.
template <int kN>
__device__ __forceinline__ void pollVecs(Ctx const& ctx, uint4 (&v)[kN], uint8_t const* const (&a)[kN], int code)
{
#if MSO_MK_SY_POLL
    // ---- MSO_MK_SY_POLL = 1: one loop, whose first pass issues every load from the code the later passes run
    // (the same loads, the same exit test: the first complete value of every word is kept, as below) ----
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[i] = a[i] != nullptr ? emptyVec() : make_uint4(0, 0, 0, 0);
    }
    uint64_t* const sy = syProbe(ctx);
    uint64_t syC0 = 0;
    uint64_t syC1 = 0;
    uint32_t syRounds = 0;
    uint64_t t0 = 0;
    while (true)
    {
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(v[i]))
            {
                v[i] = ldPoll128(a[i]);
            }
        }
        if (sy != nullptr)
        {
            if (syRounds == 0)
            {
                syC0 = clock64();
            }
            else if (syRounds == 1)
            {
                syC1 = clock64();
            }
        }
        ++syRounds;
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, kN, 0);
    }
    syProbeEnd(sy, syClass(true, code), syC0, syC1, syRounds, code);
    // ---- end MSO_MK_SY_POLL = 1 ----
#else
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[i] = a[i] != nullptr ? ldPoll128(a[i]) : make_uint4(0, 0, 0, 0);
    }
    uint64_t* const sy = syProbe(ctx);                      // poll probe
    uint64_t const syC0 = sy != nullptr ? clock64() : 0ull; // poll probe
    uint64_t syC1 = 0;                                      // poll probe
    uint32_t syRounds = 1;                                  // poll probe
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, kN, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(v[i]))
            {
                v[i] = ldPoll128(a[i]);
            }
        }
        syProbeRound(sy, syRounds, syC1);                            // poll probe
        ++syRounds;                                                  // poll probe
    }
    syProbeEnd(sy, syClass(true, code), syC0, syC1, syRounds, code); // poll probe
#endif
}

// ---- MSO_MK_PB: the B fragment of mma m16n8k16 from a polled BF16 [8][ld] activation is, for lane l, the 32 bytes
// at 32 (l & 3) of row l >> 2 of a 64-column chunk. pollVecs reads it as two 16-byte vectors in two instructions, each
// of which touches 8 rows, i.e. 8 different 128-byte lines; the time of a poll round grows with the lines its
// instructions touch, so a 64 KiB fragment poll takes a longer round than a coalesced one.
// pollVecs256 reads each lane's 32 bytes with one 256-bit load: 8 whole lines per instruction and half the
// instructions. Same addresses, same lanes, the same strength as ldPoll128 (ld.volatile). A pair is re-read whole while
// either half is incomplete, straight into the fragment registers so that every re-read of a round is in flight at
// once; a complete half re-reads the same bits, because a polled word is stored once per copy and the clears write
// another copy than the one the launch polls (MSO_MK_COPY_PREV / MSO_MK_COPY_NEXT), the invariant pollVecs's reads
// rest on too. So the protocol and the fragment values are pollVecs's. ----
__device__ __forceinline__ void ldPoll256(void const* p, uint4& lo, uint4& hi)
{
    unsigned long long q0;
    unsigned long long q1;
    unsigned long long q2;
    unsigned long long q3;
    asm volatile("ld.volatile.global.v4.b64 {%0, %1, %2, %3}, [%4];" : "=l"(q0), "=l"(q1), "=l"(q2), "=l"(q3) : "l"(p));
    lo = make_uint4(static_cast<uint32_t>(q0), static_cast<uint32_t>(q0 >> 32), static_cast<uint32_t>(q1),
        static_cast<uint32_t>(q1 >> 32));
    hi = make_uint4(static_cast<uint32_t>(q2), static_cast<uint32_t>(q2 >> 32), static_cast<uint32_t>(q3),
        static_cast<uint32_t>(q3 >> 32));
}

// pollVecs<2 kN> over the same address array when every pair is one 32-byte vector: a[2 i] 32-byte aligned and
// a[2 i + 1] = a[2 i] + 16, or both nullptr (none). The whole warp calls it.
template <int kN>
__device__ __forceinline__ void pollVecs256(
    Ctx const& ctx, uint4 (&v)[2 * kN], uint8_t const* const (&a)[2 * kN], int code)
{
#if MSO_MK_SY_POLL
    // the one-loop form of pollVecs (MSO_MK_SY_POLL = 1)
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[2 * i] = a[2 * i] != nullptr ? emptyVec() : make_uint4(0, 0, 0, 0);
        v[2 * i + 1] = v[2 * i];
    }
    uint64_t* const sy = syProbe(ctx);
    uint64_t syC0 = 0;
    uint64_t syC1 = 0;
    uint32_t syRounds = 0;
    uint64_t t0 = 0;
    while (true)
    {
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!(full4(v[2 * i]) & full4(v[2 * i + 1])))
            {
                ldPoll256(a[2 * i], v[2 * i], v[2 * i + 1]);
            }
        }
        if (sy != nullptr)
        {
            if (syRounds == 0)
            {
                syC0 = clock64();
            }
            else if (syRounds == 1)
            {
                syC1 = clock64();
            }
        }
        ++syRounds;
        bool ok = true;
#pragma unroll
        for (int i = 0; i < 2 * kN; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, 2 * kN, 0);
    }
    syProbeEnd(sy, syClass(true, code), syC0, syC1, syRounds, code);
#else
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        if (a[2 * i] != nullptr)
        {
            ldPoll256(a[2 * i], v[2 * i], v[2 * i + 1]);
        }
        else
        {
            v[2 * i] = make_uint4(0, 0, 0, 0);
            v[2 * i + 1] = v[2 * i];
        }
    }
    uint64_t* const sy = syProbe(ctx);                      // poll probe
    uint64_t const syC0 = sy != nullptr ? clock64() : 0ull; // poll probe
    uint64_t syC1 = 0;                                      // poll probe
    uint32_t syRounds = 1;                                  // poll probe
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < 2 * kN; ++i)
        {
            ok &= full4(v[i]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, 2 * kN, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!(full4(v[2 * i]) & full4(v[2 * i + 1])))
            {
                ldPoll256(a[2 * i], v[2 * i], v[2 * i + 1]);
            }
        }
        syProbeRound(sy, syRounds, syC1);                            // poll probe
        ++syRounds;                                                  // poll probe
    }
    syProbeEnd(sy, syClass(true, code), syC0, syC1, syRounds, code); // poll probe
#endif
}

// ---- end MSO_MK_PB ----

// kN 16-byte vectors at src + stride * i.
template <int kN>
__device__ __forceinline__ void pollStrided(Ctx const& ctx, uint4 (&v)[kN], uint8_t const* src, int stride, int code)
{
    uint8_t const* a[kN];
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        a[i] = src + stride * i;
    }
    pollVecs<kN>(ctx, v, a, code);
}

// Thread 0 waits until counter >= target, then the CTA proceeds (bar.sync orders the CTA's later reads); returns the
// value thread 0 read last (other threads: 0).
__device__ __forceinline__ uint32_t waitCounter(Ctx const& ctx, int idx, uint32_t target, int code)
{
    uint32_t v = 0;
    if (threadIdx.x == 0)
    {
        uint32_t const* c = ctx.counters + idx;
        uint64_t t0 = 0;
        // One acquire load per turn (the watchdog takes the value already read).
        while ((v = ldAcquire(c)) < target)
        {
            spinCheck(ctx, t0, code, idx, static_cast<int>(v));
        }
    }
    __syncthreads();
    return v;
}

// The CTA's global writes so far, then +1 on the counter (the barrier orders them before thread 0's release).
__device__ __forceinline__ void signalCounter(Ctx const& ctx, int idx)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        redReleaseAdd(ctx.counters + idx, 1u);
    }
}

__device__ __forceinline__ void waitBar(Ctx const& ctx, uint32_t bar, uint32_t parity, int code)
{
    if (!mbarTryWait(bar, parity))
    {
        uint64_t t0 = 0;
        while (!mbarTryWait(bar, parity))
        {
            spinCheck(ctx, t0, code, static_cast<int>(bar), static_cast<int>(parity));
        }
    }
}

// The running unit's weights are all in (thread 0's view).
__device__ __forceinline__ void stampWeights(Ctx const& ctx)
{
    if (MSO_MK_STAMPS(ctx) != nullptr && threadIdx.x == 0)
    {
        MSO_MK_STAMPS(ctx)[ctx.stampRole * 4 + 2] = stampTime();
    }
}

// Byte offset where piece p of an n-byte weight copy ends: whole 2 KiB GEMM blocks when n is a multiple of them, else
// 16-byte granules.
__device__ __forceinline__ uint32_t pieceEnd(uint32_t n, int p, bool onePiece)
{
    if (onePiece)
    {
        return n;
    }
    uint32_t const g = n % kBf16Block == 0 ? kBf16Block : 16u;
    return (n / g * static_cast<uint32_t>(p + 1) / kPieces) * g;
}

// Every piece of the unit whose first barrier is bar.
__device__ __forceinline__ void waitPieces(Ctx const& ctx, uint32_t bar, int code)
{
#pragma unroll 1
    for (int p = 0; p < kPieces; ++p)
    {
        waitBar(ctx, bar + 8 * p, 0, code);
    }
    stampWeights(ctx);
}

// Bulk copy of [src, src + bytes) to shared address dst on barrier bar (one thread), in requests of <= 512 KiB; with
// evictFirst under the L2 evict-first policy.
__device__ __forceinline__ void copyRange(
    uint32_t dst, uint8_t const* src, uint32_t bytes, uint32_t bar, bool evictFirst)
{
    constexpr uint32_t kPiece = 1u << 19;
    uint64_t const policy = evictFirst ? policyEvictFirst() : 0;
    for (uint32_t off = 0; off < bytes; off += kPiece)
    {
        uint32_t const n = bytes - off < kPiece ? bytes - off : kPiece;
        if (evictFirst)
        {
            bulkCopyHint(dst + off, src + off, n, bar, policy);
        }
        else
        {
            bulkCopy(dst + off, src + off, n, bar);
        }
    }
}

#if MSO_MK_LD_OP
// MSO_MK_LD_OP: copyRange with its request loop kept rolled (the same requests in the same order).
__device__ __forceinline__ void ldCopyRange1(
    uint32_t dst, uint8_t const* src, uint32_t bytes, uint32_t bar, bool evictFirst)
{
    constexpr uint32_t kPiece = 1u << 19;
    uint64_t const policy = evictFirst ? policyEvictFirst() : 0;
#pragma unroll 1
    for (uint32_t off = 0; off < bytes; off += kPiece)
    {
        uint32_t const n = bytes - off < kPiece ? bytes - off : kPiece;
        if (evictFirst)
        {
            bulkCopyHint(dst + off, src + off, n, bar, policy);
        }
        else
        {
            bulkCopy(dst + off, src + off, n, bar);
        }
    }
}
#endif

// One thread: the n0 + n1 bytes [src0, +n0) | [src1, +n1) to shared address dst as kPieces requests, piece p on
// barrier bar + 8 p (each barrier expects its piece's bytes; an empty piece completes at once).
__device__ __forceinline__ void issuePieces(uint32_t dst, uint8_t const* src0, uint32_t n0, uint8_t const* src1,
    uint32_t n1, uint32_t bar, bool evictFirst, bool onePiece)
{
    uint32_t const n = n0 + n1;
#if MSO_MK_LD_OP
    // MSO_MK_LD_OP: the loop below with onePiece = true (piece 0 = [0, n); pieces 1.. are empty: 0 bytes, no copy).
    (void) onePiece;
    mbarExpectTx(bar, n);
    if (0 < n0)
    {
        uint32_t const e = n < n0 ? n : n0;
        ldCopyRange1(dst, src0, e, bar, evictFirst);
    }
    if (n > n0)
    {
        ldCopyRange1(dst + n0, src1, n - n0, bar, evictFirst);
    }
#pragma unroll
    for (int p = 1; p < kPieces; ++p)
    {
        mbarExpectTx(bar + 8 * p, 0u);
    }
#else
    uint32_t a = 0;
#pragma unroll 1
    for (int p = 0; p < kPieces; ++p)
    {
        uint32_t const b = pieceEnd(n, p, onePiece);
        uint32_t const pb = bar + 8 * p;
        mbarExpectTx(pb, b - a);
        if (a < n0)
        {
            uint32_t const e = b < n0 ? b : n0;
            copyRange(dst + a, src0 + a, e - a, pb, evictFirst);
        }
        if (b > n0)
        {
            uint32_t const s = a > n0 ? a : n0;
            copyRange(dst + s, src1 + (s - n0), b - s, pb, evictFirst);
        }
        a = b;
    }
#endif
}

// ----------------------------------------------------------------------------------------------------------------
// GEMV building blocks: M = weight rows (16 per tile), N = 8 rows, K in 64-wide chunks.
//
// A packed (BF16): per (tile, chunk) 2048 B = 4 pieces x 32 threads x 16 B; thread (g, c) holds row g's elements
// [16c, +16) of the chunk in pieces 0-1 and row g + 8's in pieces 2-3. B natural [8][K]: thread (g, c) reads row g's
// elements [64 chunk + 16c, +16). In k step s of a chunk, MMA k slots {2c, 2c + 1} take elements 4s, 4s + 1 of the
// thread's 16 and slots {2c + 8, 2c + 9} take 4s + 2, 4s + 3, in A and in B alike.
// ----------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void loadB(uint4 (&b)[2], T const* gB, int ldb, int chunk)
{
    int const lane = threadIdx.x & 31;
    T const* src = gB + static_cast<size_t>(lane >> 2) * ldb + chunk * kChunkCols + 16 * (lane & 3);
    b[0] = ldcg128(src);
    b[1] = ldcg128(src + 8);
}

// kFlagXPrime: the B fragment of x' = bf16(bf16(r) (1 + w)) from the FP32 residual r [8][8192] and the norm weight w.
__device__ __forceinline__ void loadXPrime(uint4 (&b)[2], float const* res, __nv_bfloat16 const* w, int chunk)
{
    int const lane = threadIdx.x & 31;
    int const col = chunk * kChunkCols + 16 * (lane & 3);
    float const* r = res + static_cast<size_t>(lane >> 2) * kHidden + col;
    float4 const r0 = ldcgF4(r);
    float4 const r1 = ldcgF4(r + 4);
    float4 const r2 = ldcgF4(r + 8);
    float4 const r3 = ldcgF4(r + 12);
    uint4 const w0 = __ldg(reinterpret_cast<uint4 const*>(w + col));
    uint4 const w1 = __ldg(reinterpret_cast<uint4 const*>(w + col + 8));
    auto xp = [](float rv, float wv) { return bf16Round(rv) * (1.f + wv); };
    b[0].x = packBf16(xp(r0.x, bf16Lo(w0.x)), xp(r0.y, bf16Hi(w0.x)));
    b[0].y = packBf16(xp(r0.z, bf16Lo(w0.y)), xp(r0.w, bf16Hi(w0.y)));
    b[0].z = packBf16(xp(r1.x, bf16Lo(w0.z)), xp(r1.y, bf16Hi(w0.z)));
    b[0].w = packBf16(xp(r1.z, bf16Lo(w0.w)), xp(r1.w, bf16Hi(w0.w)));
    b[1].x = packBf16(xp(r2.x, bf16Lo(w1.x)), xp(r2.y, bf16Hi(w1.x)));
    b[1].y = packBf16(xp(r2.z, bf16Lo(w1.y)), xp(r2.w, bf16Hi(w1.y)));
    b[1].z = packBf16(xp(r3.x, bf16Lo(w1.z)), xp(r3.y, bf16Hi(w1.z)));
    b[1].w = packBf16(xp(r3.z, bf16Lo(w1.w)), xp(r3.w, bf16Hi(w1.w)));
}

// acc += A(tile, chunk) x B(chunk), BF16.
__device__ __forceinline__ void mmaChunkBf16(float (&acc)[4], uint32_t aBlock, uint4 const (&b)[2])
{
    int const lane = threadIdx.x & 31;
    uint32_t const a = aBlock + lane * 16;
    uint4 const p0 = lds128(a);
    uint4 const p1 = lds128(a + 512);
    uint4 const p2 = lds128(a + 1024);
    uint4 const p3 = lds128(a + 1536);
    mmaBf16(acc, p0.x, p2.x, p0.y, p2.y, b[0].x, b[0].y);
    mmaBf16(acc, p0.z, p2.z, p0.w, p2.w, b[0].z, b[0].w);
    mmaBf16(acc, p1.x, p3.x, p1.y, p3.y, b[1].x, b[1].y);
    mmaBf16(acc, p1.z, p3.z, p1.w, p3.w, b[1].z, b[1].w);
}

// The warps' accumulators of kSlots tiles summed in warp order (from +0), as out[slot][token][16 rows] (FP32) in
// shared memory at red + 8 KiB. Accumulator element r of lane (g, c): row g + 8 (r >= 2), token 2c + (r & 1).
template <int kSlots>
__device__ __forceinline__ void reduceWarpsTokRow(Ctx const& ctx, float const (&acc)[kSlots][4], float* out)
{
    static_assert(kSlots * 8 * 32 * 4 * 4 <= 8192, "reduction staging");
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
#pragma unroll
    for (int s = 0; s < kSlots; ++s)
    {
        *reinterpret_cast<float4*>(red + ((warp * kSlots + s) * 32 + lane) * 4)
            = make_float4(acc[s][0], acc[s][1], acc[s][2], acc[s][3]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kSlots * 128; i += kThreads)
    {
        int const s = i / 128;
        int const tok = (i % 128) / 16;
        int const row = i % 16;
        int const l = (row & 7) * 4 + tok / 2;
        int const r = (row >= 8 ? 2 : 0) + (tok & 1);
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            v += red[((w * kSlots + s) * 32 + l) * 4 + r];
        }
        out[i] = v;
    }
    __syncthreads();
}

// ----------------------------------------------------------------------------------------------------------------
// QKV cut (mirrored by schedule.py)
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ int qkvRunStart(int c)
{
    return static_cast<int>((static_cast<int64_t>(kQkvChunks) * c) / kCtas);
}

// The CTA whose run holds chunk j: the largest c with qkvRunStart(c) <= j.
__device__ __forceinline__ int qkvCtaOfChunk(int j)
{
    return static_cast<int>((static_cast<int64_t>(kCtas) * (j + 1) - 1) / kQkvChunks);
}

#if MSO_MK_LANES & 8 // MoE-hop stage guard
// ---- MoE-side hops: device stages ---------------------------------------------------------------------------------
#include "stages/moe_hops.cuh"
// ---- end MoE-side hops --------------------------------------------------------------------------------------------
// ---- integration: the hand-off helpers ----
#include "stages/integration.cuh"
// ---- end integration ----
#endif

// ----------------------------------------------------------------------------------------------------------------
// Unit weights: where they come from and how many bytes (units with a weight copy only).
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void unitWeights(
    Ctx const& ctx, int stage, int unit, uint8_t const*& src0, uint32_t& n0, uint8_t const*& src1, uint32_t& n1)
{
    src0 = nullptr;
    src1 = nullptr;
    n0 = 0;
    n1 = 0;
    switch (stage)
    {
    case kStQkvRun:
    {
        int const j0 = qkvRunStart(unit);
        src0 = ptr<uint8_t const>(ctx, kPWQkv) + static_cast<size_t>(j0) * kBf16Block;
        n0 = static_cast<uint32_t>(qkvRunStart(unit + 1) - j0) * kBf16Block;
        break;
    }
#if MSO_MK_LANES & 1 // attention-half stage guard
    // ---- attention half ----
    case kStAhOproj:
    {
        int const t0 = unit & 1023;
        src0 = ptr<uint8_t const>(ctx, kPAhWOproj) + static_cast<size_t>(t0) * (16 * kBf16Block);
        n0 = static_cast<uint32_t>(unit >> 10) * (16 * kBf16Block);
        break;
    }
    // ---- end attention half ----
#endif
    case kStTestCopy:
    {
        int64_t const* t = ptr<int64_t const>(ctx, kPTestUnits) + 2 * unit;
        src0 = ptr<uint8_t const>(ctx, kPTestBlob) + t[0];
        n0 = static_cast<uint32_t>(t[1]);
        break;
    }
#if MSO_MK_LANES & 8                                                     // MoE-hop stage guard
    default: mhUnitWeights(ctx, stage, unit, src0, n0, src1, n1); break; // MoE-side hops (may set src1)
#else
    default: break;
#endif
    }
}

// Thread 0, its earlier shared-memory reads complete (the caller synchronized): the weights of role j from the CTA's
// weight-table row into its region, completing on barrier bar.
__device__ __noinline__ void issueUnitWeights(Ctx const& ctx, int j, uint32_t bar)
{
    int const* roles = reinterpret_cast<int const*>(ctx.smem + kSmemSmall + kSmallRoles);
    int64_t const* w = reinterpret_cast<int64_t const*>(ctx.smem + kSmemSmall + kSmallWeights) + 3 * j;
    uint8_t const* src0 = reinterpret_cast<uint8_t const*>(w[0]);
    uint8_t const* src1 = reinterpret_cast<uint8_t const*>(w[1]);
    uint32_t const n0 = static_cast<uint32_t>(w[2]);
    uint32_t const n1 = static_cast<uint32_t>(static_cast<uint64_t>(w[2]) >> 32);
    if (n0 + n1 == 0)
    {
        return;
    }
    uint32_t const dst = ctx.smemBase + static_cast<uint32_t>(roles[kMaxRoles + j]);
    fenceProxyAsyncSmem();
    issuePieces(dst, src0, n0, src1, n1, bar, (MSO_MK_KFLAGS(ctx.flags) & kFlagEvictFirst) != 0,
        (MSO_MK_KFLAGS(ctx.flags) & kFlagOnePiece) != 0); // MSO_MK_LD_KF
}

// Thread 0 (kFlagL2Deferred): the weights of role j into L2 (its slot is still busy).
__device__ __noinline__ void prefetchUnitL2(Ctx const& ctx, int j)
{
    int64_t const* w = reinterpret_cast<int64_t const*>(ctx.smem + kSmemSmall + kSmallWeights) + 3 * j;
    uint32_t const n0 = static_cast<uint32_t>(w[2]);
    uint32_t const n1 = static_cast<uint32_t>(static_cast<uint64_t>(w[2]) >> 32);
    uint64_t const policy = policyEvictFirst();
    constexpr uint32_t kPiece = 1u << 19;
    for (uint32_t off = 0; off < n0; off += kPiece)
    {
        prefetchBulkL2Hint(reinterpret_cast<uint8_t const*>(w[0]) + off, n0 - off < kPiece ? n0 - off : kPiece, policy);
    }
    for (uint32_t off = 0; off < n1; off += kPiece)
    {
        prefetchBulkL2Hint(reinterpret_cast<uint8_t const*>(w[1]) + off, n1 - off < kPiece ? n1 - off : kPiece, policy);
    }
}

// ---- kIFlagFaRunL2: thread 0 at entry, a QkvRun unit's weights into L2 without a cache hint (they are read later,
// after the CTA's first run) ----
__device__ __noinline__ void faRunL2(Ctx const& ctx, int j)
{
    int64_t const* w = reinterpret_cast<int64_t const*>(ctx.smem + kSmemSmall + kSmallWeights) + 3 * j;
    uint32_t const n0 = static_cast<uint32_t>(w[2]);
    uint32_t const n1 = static_cast<uint32_t>(static_cast<uint64_t>(w[2]) >> 32);
    constexpr uint32_t kPiece = 1u << 19;
    for (uint32_t off = 0; off < n0; off += kPiece)
    {
        prefetchBulkL2(reinterpret_cast<uint8_t const*>(w[0]) + off, n0 - off < kPiece ? n0 - off : kPiece);
    }
    for (uint32_t off = 0; off < n1; off += kPiece)
    {
        prefetchBulkL2(reinterpret_cast<uint8_t const*>(w[1]) + off, n1 - off < kPiece ? n1 - off : kPiece);
    }
}

// ---- end kIFlagFaRunL2 ----

// Filled once per parameter block, before capture, from the same address/length function the units use.
__global__ void msoWeightTableKernel(int64_t const* params, int64_t* out)
{
    int const c = static_cast<int>(blockIdx.x);
    int const j = static_cast<int>(threadIdx.x);
    Ctx ctx{};
    ctx.p = params;
    int const role = reinterpret_cast<int const*>(params + kPNumParams)[c * kMaxRoles + j];
    uint8_t const* src0 = nullptr;
    uint8_t const* src1 = nullptr;
    uint32_t n0 = 0, n1 = 0;
    if (role != 0)
    {
        unitWeights(ctx, role >> 16, role & 0xffff, src0, n0, src1, n1);
    }
    int64_t* o = out + (static_cast<int64_t>(c) * kMaxRoles + j) * 3;
    o[0] = reinterpret_cast<int64_t>(src0);
    o[1] = reinterpret_cast<int64_t>(src1);
    o[2] = static_cast<int64_t>(n0) | (static_cast<int64_t>(n1) << 32);
}

// ----------------------------------------------------------------------------------------------------------------
// Stages
// ----------------------------------------------------------------------------------------------------------------

// QkvRun: chunks [j0, j1) of the packed QKV weight (its region at wOff, its pieces' barriers from bar) times the B
// fragments of the matching x columns. The unit waits for all its weight pieces before the MMA loop; warp w then takes
// run chunks w, w + 8, ... with four accumulator chains per tile; the two tiles' FP32 partials [8 tokens][16 rows] go
// to the CTA's two slots of kPQkvPart as Lamport words.
__device__ __forceinline__ void qkvChunk(float (&acc)[4], uint32_t a, uint4 const (&b)[2])
{
#if MSO_MK_DBG == 1
    int const lane = threadIdx.x & 31;
    uint4 const p0 = lds128(a + lane * 16);
    uint4 const p1 = lds128(a + lane * 16 + 512);
    uint4 const p2 = lds128(a + lane * 16 + 1024);
    uint4 const p3 = lds128(a + lane * 16 + 1536);
    acc[0] += __uint_as_float((p0.x ^ p1.y ^ p2.z ^ p3.w ^ b[0].x) & 0x3f800000u);
#elif MSO_MK_DBG == 2
    mmaBf16(acc, b[0].x, b[1].x, b[0].y, b[1].y, b[0].x, b[0].y);
    mmaBf16(acc, b[0].z, b[1].z, b[0].w, b[1].w, b[0].z, b[0].w);
    mmaBf16(acc, b[1].x, b[0].x, b[1].y, b[0].y, b[1].x, b[1].y);
    mmaBf16(acc, b[1].z, b[0].z, b[1].w, b[0].w, b[1].z, b[1].w);
    (void) a;
#else
    mmaChunkBf16(acc, a, b);
#endif
}

// QkvRun with K slices (MSO_MK_QKV_SLICES = 4): warp w takes the run's chunks j with j % 8 == w, i.e. positions
// w, w + 8, w + 16, w + 24 of every segment it meets, so it needs 4 x fragments (b[f] = position 8 f + w of the
// run's slice). With kk = the chunk's index counted from position w of the warp's first segment, the fragment is
// b[kk % 4] and the warp-relative segment kk / 4, both known at compile time in the unrolled loop.
__device__ __forceinline__ void stageQkvRunSliced(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, Io const& io)
{
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const j0 = qkvRunStart(unit);
    int const n = qkvRunStart(unit + 1) - j0;
    int const g0 = j0 / kSegChunks;
    int const slice = g0 / kQkvTiles;
    int const iw = (warp - (j0 & 7) + 8) & 7;                                   // the warp's first chunk (run index)
    int const gw = (j0 + iw) / kSegChunks;                                      // its segment
    int const q0 = ((j0 + iw) % kSegChunks) / 8;                                // its position index in that segment
    bool const xPrime = (MSO_MK_KFLAGS(ctx.flags) & kFlagXPrime) != 0;          // MSO_MK_LD_KF
    constexpr int kPos = kSegChunks / kWarps;                                   // positions per warp and segment (4)
    bool const t3Wf = (icPIFlags(ctx.p[kPIFlags]) & kIFlagT3WaitFirstQkv) != 0; // the weights before the x poll
    // ---- kIFlagW1WarmQkv: a warm pass first when this slice's x is not published yet ----
    // (ctx.w1Early: the kernel prologue's sample of one word of this slice's x, taken off the CTA's path.)
    bool const warm = (icPIFlags(ctx.p[kPIFlags]) & kIFlagW1WarmQkv) && (icPIFlags(ctx.p[kPIFlags]) & kIFlagXIn)
        && ctx.w1Early != 0;
#pragma unroll 1
    for (int pass = warm ? 0 : 1; pass < 2; ++pass)
    {
        bool const dry = pass == 0;
        // ---- end kIFlagW1WarmQkv ----
        uint4 b[kPos][2];
        if (t3Wf && !dry) // not in the warm pass
        {
            waitPieces(ctx, bar, kCodeWeights);
        }
        // ---- early x (the previous launch's x_attn, polled; this unit runs before the dependency wait) ----
        if (icPIFlags(ctx.p[kPIFlags]) & kIFlagXIn)
        {
            uint8_t const* xb = ptr<uint8_t const>(ctx, kPIXBuf) + static_cast<size_t>(ctx.copy) * kIXBytes;
            // ---- kIFlagT3CanaryXIn: gate on the MhF1 units of the slice's columns ----
#if MSO_MK_LANES & 8
            if ((icPIFlags(ctx.p[kPIFlags]) & kIFlagT3CanaryXIn) && !dry) // not in the warm pass
            {
                int const c0 = slice * kSegChunks * kChunkCols;
                t3Canary(ctx, t3RowCanary(xb, lane, 0, 8, c0, c0 + kSegChunks * kChunkCols), kCodeQkvPart);
            }
#endif
            // ---- end kIFlagT3CanaryXIn ----
            uint8_t const* a[2 * kPos];
#pragma unroll
            for (int f = 0; f < kPos; ++f)
            {
                int const chunk = slice * kSegChunks + 8 * f + warp;
                a[2 * f] = xb + (static_cast<size_t>(lane >> 2) * kHidden + chunk * kChunkCols + 16 * (lane & 3)) * 2;
                a[2 * f + 1] = a[2 * f] + 16;
                if (dry) // the warm pass loads nothing (pollVecs reads no address that is nullptr)
                {
                    a[2 * f] = nullptr;
                    a[2 * f + 1] = nullptr;
                }
            }
            uint4 v[2 * kPos];
#if MSO_MK_PB & 4 // one 256-bit load per fragment
            static_assert(kIXBytes % 32 == 0 && (kHidden * 2) % 32 == 0 && (kChunkCols * 2) % 32 == 0,
                "MSO_MK_PB bit 4: 32-byte pairs");
            pollVecs256<kPos>(ctx, v, a, kCodeQkvPart);
#else
            pollVecs<2 * kPos>(ctx, v, a, kCodeQkvPart);
#endif
#pragma unroll
            for (int f = 0; f < kPos; ++f)
            {
                b[f][0] = v[2 * f];
                b[f][1] = v[2 * f + 1];
            }
        }
        else
        // ---- end early x ----
        {
#pragma unroll
            for (int f = 0; f < kPos; ++f)
            {
                int const chunk = slice * kSegChunks + 8 * f + warp;
                if (xPrime)
                {
                    loadXPrime(
                        b[f], static_cast<float const*>(io.res), ptr<__nv_bfloat16 const>(ctx, kPGammaQkv), chunk);
                }
                else
                {
                    loadB(b[f], static_cast<__nv_bfloat16 const*>(io.x), kHidden, chunk);
                }
            }
        }
        if (!dry) // not in the warm pass
        {
            stampInput(ctx);
        }
        // Every piece before the MMA loop: a wait inside it (asm volatile) would keep the shared loads of later chunks
        // behind it, and one wait for every piece is faster than four pieces with in-loop waits.
        if (!t3Wf && !dry) // else waited before the x poll; not in the warm pass
        {
            waitPieces(ctx, bar, kCodeWeights);
        }
        if (!dry) // not in the warm pass
        {
            subStamp(ctx, 0);
        }
        float part[4][2][4] = {};
        constexpr int kSteps = kPos - 1 + kQkvMaxPerWarp;
#pragma unroll
        for (int kk = 0; kk < kSteps; ++kk)
        {
            int const i = iw + 8 * (kk - q0);
            if (kk >= q0 && i < n)
            {
                qkvChunk(
                    part[kk / kPos][kk & 1], ctx.smemBase + wOff + static_cast<uint32_t>(i) * kBf16Block, b[kk % kPos]);
            }
        }
        if (!dry) // not in the warm pass
        {
            subStamp(ctx, 1);
        }
        // Warp-relative segments to the CTA's slots: slot = (gw - g0) + rs. Staging [warp][rs][lane][4] FP32 (16 KiB).
        float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
#pragma unroll
        for (int rs = 0; rs < 4; ++rs)
        {
            *reinterpret_cast<float4*>(red + ((warp * 4 + rs) * 32 + lane) * 4)
                = make_float4(part[rs][0][0] + part[rs][1][0], part[rs][0][1] + part[rs][1][1],
                    part[rs][0][2] + part[rs][1][2], part[rs][0][3] + part[rs][1][3]);
        }
        __syncthreads();
        int const slots = (j0 + n - 1) / kSegChunks - g0 + 1;
        uint8_t* part0 = ptr<uint8_t>(ctx, kPQkvPart) + static_cast<size_t>(unit) * kQkvMaxSlots * kQkvPartSlotBytes;
        for (int o = threadIdx.x; o < slots * 128; o += kThreads)
        {
            int const slot = o / 128;
            int const tok = (o % 128) / 16;
            int const row = o % 16;
            int const l = (row & 7) * 4 + tok / 2;
            int const r = (row >= 8 ? 2 : 0) + (tok & 1);
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                int const rs = slot - ((j0 + ((w - (j0 & 7) + 8) & 7)) / kSegChunks - g0);
                if (rs >= 0 && rs < 4)
                {
                    v += red[((w * 4 + rs) * 32 + l) * 4 + r];
                }
            }
            if (!dry) // not in the warm pass
            {
                st32(part0 + 4 * o, lamport(__float_as_uint(v)));
            }
        }
        (void) gw;
        // ---- warm pass: the real pass rewrites the staging area ----
        if (dry)
        {
            __syncthreads();
            continue;
        }
        // ---- end warm pass ----
        subStamp(ctx, 2);
    } // pass
}

__device__ __noinline__ void stageQkvRun(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, Io const& io)
{
    if constexpr (kQkvSlices != 1)
    {
        stageQkvRunSliced(ctx, unit, wOff, bar, io);
        return;
    }
    int const warp = threadIdx.x >> 5;
    int const j0 = qkvRunStart(unit);
    int const n = qkvRunStart(unit + 1) - j0;
    int const jb = (j0 / kChunksPerRow + 1) * kChunksPerRow - j0;      // the run index where the second tile begins
    bool const xPrime = (MSO_MK_KFLAGS(ctx.flags) & kFlagXPrime) != 0; // MSO_MK_LD_KF
    uint4 b[kQkvMaxPerWarp][2];
#pragma unroll
    for (int k = 0; k < kQkvMaxPerWarp; ++k)
    {
        int const i = warp + kWarps * k;
        if (i < n)
        {
            int const chunk = (j0 + i) % kChunksPerRow;
            if (xPrime)
            {
                loadXPrime(b[k], static_cast<float const*>(io.res), ptr<__nv_bfloat16 const>(ctx, kPGammaQkv), chunk);
            }
            else
            {
                loadB(b[k], static_cast<__nv_bfloat16 const*>(io.x), kHidden, chunk);
            }
        }
    }
    stampInput(ctx);
    waitPieces(ctx, bar, kCodeWeights);
    subStamp(ctx, 0);
    float part[2][4][4] = {};
#pragma unroll
    for (int k = 0; k < kQkvMaxPerWarp; ++k)
    {
        int const i = warp + kWarps * k;
        if (i < n)
        {
            uint32_t const a = ctx.smemBase + wOff + static_cast<uint32_t>(i) * kBf16Block;
            if (i < jb)
            {
                qkvChunk(part[0][k & 3], a, b[k]);
            }
            else
            {
                qkvChunk(part[1][k & 3], a, b[k]);
            }
        }
    }
    float acc[2][4];
#pragma unroll
    for (int s = 0; s < 2; ++s)
    {
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            acc[s][r] = (part[s][0][r] + part[s][1][r]) + (part[s][2][r] + part[s][3][r]);
        }
    }
    subStamp(ctx, 1);
    float* out = reinterpret_cast<float*>(ctx.smem + kSmemRed + 8192);
    reduceWarpsTokRow<2>(ctx, acc, out);
    int const slots = jb < n ? 2 : 1;
    uint8_t* part0 = ptr<uint8_t>(ctx, kPQkvPart) + static_cast<size_t>(unit) * kQkvMaxSlots * kQkvPartSlotBytes;
    for (int i = threadIdx.x; i < slots * 128; i += kThreads)
    {
        st32(part0 + 4 * i, lamport(__float_as_uint(out[i])));
    }
    subStamp(ctx, 2);
}

// XSsq (kFlagXPrime): unit u = 64-column chunk u; per token the sum of bf16(r)^2 over its columns, into the arena.
// Warp w = token w; lane l: columns 64 u + 2 l, + 1; then a fixed shuffle tree.
__device__ __noinline__ void stageXSsq(Ctx const& ctx, int unit, Io const& io)
{
    int const tok = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    float2 const r
        = ldcgF2(static_cast<float const*>(io.res) + static_cast<size_t>(tok) * kHidden + unit * kChunkCols + 2 * lane);
    float const c0 = bf16Round(r.x);
    float const c1 = bf16Round(r.y);
    float s = __fmul_rn(c0, c0) + __fmul_rn(c1, c1);
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1)
    {
        s += __shfl_xor_sync(0xffffffffu, s, m);
    }
    stampInput(ctx);
    if (lane == 0)
    {
        st32(arena<uint8_t>(ctx, kArXSsq) + 4 * (tok * kSsqChunks + unit), lamport(__float_as_uint(s)));
    }
}

// QkvSum: unit u = rows [64 u, +64) (tiles 4u .. 4u + 3). Threads 0-127: tile 4u + t / 32, its float4 t % 32 of
// [8 tokens][16 rows] (token q / 4, rows 4 (q % 4) ..+3); the <= 3 runs' partials polled together and summed in
// ascending CTA order from +0; one BF16 rounding (after the 1 / rms scale with kFlagXPrime); then the slots it read
// are emptied (single reader, after the output). Warp 4 (kFlagXPrime): the tokens' 1 / rms from the XSsq partials.
// ---- kIFlagFaQPrep (stages/fullAttention.cuh) ----
__device__ void faQPrepStore(Ctx const& ctx, int tok, int c0, int hq, uint2 o);

// ---- end kIFlagFaQPrep ----

__device__ __noinline__ void stageQkvSum(Ctx const& ctx, int unit, Io const& io)
{
#if MSO_MK_LD_FL
    icFaCheck(ctx.p[kPIFlags]);
    if (!(ctx.p[kPIFlags] & kIFlagQkvHeadsFirst))
    {
        __trap(); // MSO_MK_LD_FL: QkvSum runs only in the served one-launch full layers (heads-first QKV pack)
    }
#endif
    int const tid = threadIdx.x;
    bool const xPrime = (MSO_MK_KFLAGS(ctx.flags) & kFlagXPrime) != 0; // MSO_MK_LD_KF
    float* sRcp = reinterpret_cast<float*>(ctx.smem + kSmemSmall + kSmallScalars);
    uint4 v[kQkvMaxRuns];
    uint8_t const* a[kQkvMaxRuns] = {};
    int tile = 0;
    int q = 0;
    if (tid < 128)
    {
        tile = kQkvTilesPerBlock * unit + tid / 32;
        q = tid % 32;
        uint8_t const* base = ptr<uint8_t const>(ctx, kPQkvPart);
#pragma unroll
        for (int sl = 0; sl < kQkvSlices; ++sl)
        {
            int const g = sl * kQkvTiles + tile;
            int const cFirst = qkvCtaOfChunk(g * kSegChunks);
            int const cLast = qkvCtaOfChunk(g * kSegChunks + kSegChunks - 1);
#pragma unroll
            for (int k = 0; k < kQkvRunsPerSeg; ++k)
            {
                int const c = cFirst + k;
                if (c <= cLast)
                {
                    int const slot = g - qkvRunStart(c) / kSegChunks;
                    a[sl * kQkvRunsPerSeg + k]
                        = base + (static_cast<size_t>(c) * kQkvMaxSlots + slot) * kQkvPartSlotBytes + 16 * q;
                }
            }
        }
        pollVecs<kQkvMaxRuns>(ctx, v, a, kCodeQkvPart);
    }
    else if (xPrime && tid < 160)
    {
        int const lane = tid - 128;
        int const tok = lane / 4;
        uint8_t const* src = arena<uint8_t const>(ctx, kArXSsq) + 4 * (tok * kSsqChunks + 32 * (lane % 4));
        uint4 s[8];
        pollStrided<8>(ctx, s, src, 16, kCodeQkvSsq);
        float S = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            S += __uint_as_float(s[i].x);
            S += __uint_as_float(s[i].y);
            S += __uint_as_float(s[i].z);
            S += __uint_as_float(s[i].w);
        }
        S += __shfl_xor_sync(0xffffffffu, S, 1);
        S += __shfl_xor_sync(0xffffffffu, S, 2);
        if (lane % 4 == 0)
        {
            sRcp[tok] = rsqrtf(S / static_cast<float>(kHidden) + __int_as_float(static_cast<int>(ctx.p[kPEps])));
        }
    }
    stampInput(ctx);
    __syncthreads();
    if (tid < 128)
    {
        int const tok = q / 4;
        int const row0 = 4 * (q % 4);
        float sum[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < kQkvMaxRuns; ++k)
        {
            if (a[k] != nullptr)
            {
                sum[0] += __uint_as_float(v[k].x);
                sum[1] += __uint_as_float(v[k].y);
                sum[2] += __uint_as_float(v[k].z);
                sum[3] += __uint_as_float(v[k].w);
            }
        }
        float const sc = xPrime ? sRcp[tok] : 1.f;
        int const col = 16 * tile + row0;
        uint2 const o = make_uint2(packBf16(sum[0] * sc, sum[1] * sc), packBf16(sum[2] * sc, sum[3] * sc));
        // ---- one-launch full layer: Lamport words for the same launch's FaAttn / FaMerge ----
        if (MSO_MK_FL_QKVSUM(ctx.p[kPIFlags]) & kIFlagQkvArena) // MSO_MK_LD_FL
        {
            int ccol = col;
            if (MSO_MK_FL_QKVSUM(ctx.p[kPIFlags]) & kIFlagQkvHeadsFirst) // MSO_MK_LD_FL
            {
                int const b = col / kQkvBlockRows;
                int const cb = b < 16 ? 2 * b : (b < 20 ? 16 + b : 2 * (b - 20) + 1);
                ccol = cb * kQkvBlockRows + col % kQkvBlockRows;
            }
            st64(ctx.arena + kArFaQkv + (static_cast<size_t>(tok) * kQkvRows + ccol) * 2,
                make_uint2(lamport(o.x), lamport(o.y)));
            // ---- kIFlagFaQRep: the q columns (checkpoint blocks 2 h) once more at kArAhQ [tok][head][64] ----
            int const cbk = ccol / kQkvBlockRows;
            // kIFlagFaQPrep: a q block's FaAttn Q prep (block-uniform: the unit is one block; threads 0-127 together)
            if ((MSO_MK_FL_QKVSUM(ctx.p[kPIFlags]) & kIFlagFaQPrep) && cbk < 32 && (cbk & 1) == 0) // MSO_MK_LD_FL
            {
                faQPrepStore(ctx, tok, ccol % kQkvBlockRows, cbk / 2, o);
            }
            else if ((MSO_MK_FL_QKVSUM(ctx.p[kPIFlags]) & kIFlagFaQRep) && cbk < 32 && (cbk & 1) == 0) // MSO_MK_LD_FL
            {
                st64(ctx.arena + kArAhQ + ((static_cast<size_t>(tok) * 16 + cbk / 2) * 64 + ccol % kQkvBlockRows) * 2,
                    make_uint2(lamport(o.x), lamport(o.y)));
            }
            // ---- end kIFlagFaQRep ----
        }
        else
        // ---- end one-launch full layer ----
        {
            st64(static_cast<uint8_t*>(io.qkv) + (static_cast<size_t>(tok) * kQkvRows + col) * 2, o);
        }
        if ((MSO_MK_KFLAGS(ctx.flags) & kFlagCheck) && io.qkvF32 != nullptr) // MSO_MK_LD_KF
        {
            *reinterpret_cast<float4*>(static_cast<float*>(io.qkvF32) + static_cast<size_t>(tok) * kQkvRows + col)
                = make_float4(sum[0] * sc, sum[1] * sc, sum[2] * sc, sum[3] * sc);
        }
#pragma unroll
        for (int k = 0; k < kQkvMaxRuns; ++k)
        {
            if (a[k] != nullptr)
            {
                st128(const_cast<uint8_t*>(a[k]), emptyVec());
            }
        }
    }
}

// Clear: 16 KiB unit of the next launch's arena copy, under the L2 evict-first policy (only a later launch reads it).
__device__ __noinline__ void stageClear(Ctx const& ctx, int unit)
{
    int const other = MSO_MK_COPY_NEXT(ctx.copy, ctx.numCopies);
    uint64_t const policy = policyEvictFirst();
    uint8_t* base = ptr<uint8_t>(ctx, kPArena) + static_cast<size_t>(other) * kArenaBytes + unit * kClearUnitBytes;
    for (int off = 16 * threadIdx.x; off < kClearUnitBytes; off += 16 * kThreads)
    {
        st128Hint(base + off, emptyVec(), policy);
    }
    signalCounter(ctx, kCClear);
}

// Tests: the unit's weight bytes (TestUnits[unit] = (offset, bytes)) from its shared-memory region to out0 + offset.
__device__ __noinline__ void stageTestCopy(Ctx const& ctx, int unit, uint32_t wOff, uint32_t bar, Io const& io)
{
    int64_t const* t = ptr<int64_t const>(ctx, kPTestUnits) + 2 * unit;
    int64_t const off = t[0];
    int const bytes = static_cast<int>(t[1]);
    waitPieces(ctx, bar, kCodeWeights);
    uint8_t* dst = static_cast<uint8_t*>(io.out0) + off;
    for (int i = 16 * threadIdx.x; i < bytes; i += 16 * kThreads)
    {
        st128(dst + i, lds128(ctx.smemBase + wOff + i));
    }
}

// Tests: 64 words of unit u (seed *in0): w_i = hash, with words i % 16 == 5 equal to kEmpty (stored as +0) into
// mail[u]; 16 words into the arena (cleared by the next launch's clear units); then the publisher counter.
__device__ __forceinline__ uint32_t testWord(int unit, int i, uint32_t seed)
{
    if (i % 16 == 5)
    {
        return kEmpty;
    }
    uint32_t h
        = seed * 0x9e3779b9u + static_cast<uint32_t>(unit) * 0x85ebca6bu + static_cast<uint32_t>(i) * 0xc2b2ae35u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    return h;
}

__device__ __noinline__ void stageTestPub(Ctx const& ctx, int unit, Io const& io)
{
    uint32_t const seed = *static_cast<uint32_t const*>(io.in0);
    int const tid = threadIdx.x;
    if (tid < 64)
    {
        st32(ptr<uint8_t>(ctx, kPTestMail) + 256 * unit + 4 * tid, lamport(testWord(unit, tid, seed)));
    }
    if (tid < 16)
    {
        st32(arena<uint8_t>(ctx, kArTest) + 64 * unit + 4 * tid, lamport(testWord(unit, tid, seed)));
    }
    signalCounter(ctx, kCTest);
}

// Tests: warp 0 polls mail[u] (16 vectors, lanes 0-15), writes it to out0[u]; then every publisher's count, written
// to out1[u] (exactly the publisher count when this launch's counter set was zeroed); then empties mail[u].
__device__ __noinline__ void stageTestSub(Ctx const& ctx, int unit, Io const& io)
{
    int const tid = threadIdx.x;
    uint8_t* mail = ptr<uint8_t>(ctx, kPTestMail) + 256 * unit;
    if (tid < 32)
    {
        uint4 v[1];
        uint8_t const* a[1] = {tid < 16 ? mail + 16 * tid : nullptr};
        pollVecs<1>(ctx, v, a, kCodeTestSub);
        stampInput(ctx);
        if (tid < 16)
        {
            st128(static_cast<uint8_t*>(io.out0) + 256 * unit + 16 * tid, v[0]);
        }
    }
    uint32_t const seen = waitCounter(ctx, kCTest, static_cast<uint32_t>(ctx.p[kPTestPubs]), kCodeCounter);
    if (tid == 0)
    {
        static_cast<uint32_t*>(io.out1)[unit] = seen;
    }
    if (tid < 16)
    {
        st128(mail + 16 * tid, emptyVec());
    }
}

// Tests: warp 0 polls a word nobody writes (in1); only the watchdog ends it.
__device__ __noinline__ void stageTestHang(Ctx const& ctx, Io const& io)
{
    if (threadIdx.x < 32)
    {
        poll1(ctx, threadIdx.x == 0 ? io.in1 : nullptr, kCodeTestHang);
    }
}

// ---- SM-stable roles (MSO_MK_I2B_SMID = 1): the CTA's logical index is the SM it runs on ----
// At one CTA per SM and a grid of kCtas = the SM count, every SM holds exactly one CTA of a launch once the last one is
// resident (no unit of a launch completes before every CTA started: each CTA has an o_proj tile that the MoE half
// waits for), so %smid is a permutation of the block indices. With it, SM s runs the same unit list in every launch of
// a plan, and its instruction cache still holds that code from the previous layer (block-index order hands the freed
// SMs to a new grid in exit order, which shuffles the mapping every launch). From here to the end of the megakernel
// every `blockIdx` (the stage files' too) is the logical index.
#ifndef MSO_MK_I2B_SMID
#define MSO_MK_I2B_SMID 0
#endif
#if MSO_MK_I2B_SMID
static_assert(kMhClusterDim == 1, "SM-stable roles need cluster-free launches");

__device__ __forceinline__ uint3 i2bLogicalBlockIdx()
{
    uint32_t s;
    asm("mov.u32 %0, %%smid;" : "=r"(s));
    return make_uint3(s, 0u, 0u);
}

#define blockIdx (::mso_mk::i2bLogicalBlockIdx())
#endif
// ---- end SM-stable roles ----
#if MSO_MK_LANES & 1 // attention-half stage guard
// ---- attention half ----
#include "stages/attnHalf.cuh"
// ---- end attention half ----
#endif
#if MSO_MK_LANES & 2 // full-attention stage guard
// ---- full attention ----
#include "stages/fullAttention.cuh"
// ---- end full attention ----
#endif
#if MSO_MK_LANES & 4 // experts stage guard
// ---- MoE experts begin ----
#include "stages/experts.cuh"
// ---- end MoE experts ----
#endif
// ---- the BF16 experts stage (MSO_MK_LANES bit 4; needs the MoE experts stage, bit 2) ----
#if (MSO_MK_LANES & 16) && (MSO_MK_LANES & 4)
#include "stages/bf16Experts.cuh"
#endif
// ---- end BF16 experts stage ----

// ----------------------------------------------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------------------------------------------
// ---- integration: IClearX = 16 KiB of the early-x buffer copy the previous launch read ----
__device__ __noinline__ void stageIClearX(Ctx const& ctx, int unit)
{
    int const prev = MSO_MK_COPY_PREV(ctx.copy, ctx.numCopies);
    uint64_t const policy = policyEvictFirst();
    uint8_t* base = ptr<uint8_t>(ctx, kPIXBuf) + static_cast<size_t>(prev) * kIXBytes + unit * kIClearXUnitBytes;
    for (int off = 16 * threadIdx.x; off < kIClearXUnitBytes; off += 16 * kThreads)
    {
        st128Hint(base + off, emptyVec(), policy);
    }
}

// ---- end integration ----

// ---- the attention-half stages that may run before griddepcontrol.wait (kIFlagI2bPreWait) ----
__device__ __forceinline__ bool i2bPreWaitStage(int stage)
{
#if MSO_MK_LANES & 1
    return stage == kStQkvRun || stage == kStAhQkvFin || stage == kStAhSwaAttn || stage == kStClear
        || stage == kStAhOproj;
#else
    return false;
#endif
}

// ---- end kIFlagI2bPreWait stages ----

// ---- ghost and warm units ----
// A full-attention layer runs none of the sliding-window attention code (QkvFin, SwaAttn, the merge), and the next
// sliding-window layer runs each of those units slower while the code both layer types run is not slower. A ghost unit
// (on a CTA of the full layer's launch F that is idle after its experts) runs one such unit through the same functions
// on a scratch parameter block (kPI2bGhost: scratch arena, partials, pool page, block table, positions, RoPE rows; the
// layer's scalars), so that code was fetched recently when the next sliding-window layer needs it. Unit u: 0-3 = QkvFin
// blocks q0, k0, v0, g0; 4 = SwaAttn (h 0, b 0) with its merge. The scratch inputs a unit polls are refilled with zeros
// (a non-empty Lamport word) before it runs, because the unit empties what it read. Nothing outside the scratch buffers
// is written.
__device__ __noinline__ void stageI2bGhost(Ctx const& ctx, int unit, uint32_t bar)
{
#if MSO_MK_LANES & 1
    int64_t const* gp = reinterpret_cast<int64_t const*>(ctx.p[kPI2bGhost]);
    if (gp == nullptr || __ldg(gp + kPMagic) != kMagic)
    {
        return;
    }
    int const tid = threadIdx.x;
    // The ghost's parameter block and Ctx in [0, 2048) of the unit region (free: the experts unit before it is done,
    // and the stages run here use the regions from 98304 on and the small area).
    int64_t* sp = reinterpret_cast<int64_t*>(ctx.smem);
    Ctx* gc = reinterpret_cast<Ctx*>(ctx.smem + 1024);
    for (int i = tid; i < kPNumParams; i += kThreads)
    {
        sp[i] = __ldg(gp + i);
    }
    __syncthreads();
    if (tid == 0)
    {
        *gc = ctx;
        gc->p = sp;
        gc->arena = reinterpret_cast<uint8_t*>(sp[kPArena]);
        gc->copy = 0;
        gc->stamps = nullptr;
    }
    __syncthreads();
    Ctx const& g = *gc;
    Io const gio = {};
    uint4 const full = make_uint4(0u, 0u, 0u, 0u);
    if (unit < 64)
    {
        int const b = unit;
        if (tid < 128)
        {
            int const tile = kQkvTilesPerBlock * b + tid / 32;
            int const q = tid % 32;
            uint8_t* base = ptr<uint8_t>(g, kPQkvPart);
#pragma unroll 1
            for (int sl = 0; sl < kQkvSlices; ++sl)
            {
                int const gs = sl * kQkvTiles + tile;
                int const cFirst = qkvCtaOfChunk(gs * kSegChunks);
                int const cLast = qkvCtaOfChunk(gs * kSegChunks + kSegChunks - 1);
#pragma unroll 1
                for (int c = cFirst; c <= cLast; ++c)
                {
                    int const slot = gs - qkvRunStart(c) / kSegChunks;
                    st128(base + (static_cast<size_t>(c) * kQkvMaxSlots + slot) * kQkvPartSlotBytes + 16 * q, full);
                }
            }
        }
        if (tid == 0)
        {
            ahEntryIssue(g, (kStAhQkvFin << 16) | b, bar);
        }
        __syncthreads();
        stageAhQkvFin(g, b, bar, gio);
    }
    else
    {
        // SwaAttn unit (h 0, b) and its merge of rows {2 b, 2 b + 1}: splits 0-32 of those rows and their (m, l).
        int const b = unit - 64;
#pragma unroll 1
        for (int i = tid; i < kAhBlocks * 2 * (kAhD / 4); i += kThreads)
        {
            int const sp2 = i / (2 * (kAhD / 4));
            int const r = (i / (kAhD / 4)) % 2;
            st128(ahPartO(g, 0, sp2, 2 * b + r) + 16 * (i % (kAhD / 4)), full);
        }
        if (tid < kAhBlocks)
        {
            st128(ahPartMl(g, 0, tid, 2 * b), full);
        }
        if (tid == 0)
        {
            ahEntryIssue(g, (kStAhSwaAttn << 16) | b, bar);
        }
        __syncthreads();
        stageAhSwaAttn(g, b, bar, gio);
    }
#endif
}

// The first sliding-window layer of a forward (after the stock layer 0 and the first full layer) is slower than the
// others, in its QkvFin / SwaAttn / merge units, and refreshing their code (ghost units) does not change that. This
// unit prefetches the buffers those units and the MoE half touch (the table kPI2bWarm, built on the host) into L2 from
// a CTA of a full layer's launch F that is idle after its experts. Unit u takes the ranges u, u + kI2bWarmUnits, ...;
// each thread of the CTA one 512 KiB piece at a time. Prefetch only: no memory is written.
constexpr int kI2bWarmUnits = 4;

__device__ __noinline__ void stageI2bWarm(Ctx const& ctx, int unit)
{
    int64_t const* t = reinterpret_cast<int64_t const*>(ctx.p[kPI2bWarm]);
    if (t == nullptr)
    {
        return;
    }
    int const n = static_cast<int>(__ldg(t));
    constexpr uint32_t kPiece = 1u << 19;
#pragma unroll 1
    for (int r = unit; r < n; r += kI2bWarmUnits)
    {
        uint8_t const* base = reinterpret_cast<uint8_t const*>(__ldg(t + 1 + 2 * r));
        uint64_t const bytes = static_cast<uint64_t>(__ldg(t + 2 + 2 * r));
#pragma unroll 1
        for (uint64_t off = static_cast<uint64_t>(threadIdx.x) * kPiece; off < bytes; off += kThreads * kPiece)
        {
            uint64_t const left = bytes - off;
            prefetchBulkL2(base + off, static_cast<uint32_t>(left < kPiece ? left : kPiece));
        }
    }
}

// ---- end ghost and warm units ----

__device__ __forceinline__ void runUnit(Ctx const& c, int stage, int unit, uint32_t off, uint32_t bar, Io const& io)
{
    switch (stage)
    {
    case kStQkvRun: stageQkvRun(c, unit, off, bar, io); break;
    case kStQkvSum: stageQkvSum(c, unit, io); break;
    case kStClear: stageClear(c, unit); break;
#if !MSO_MK_LD_DS // the served plans hold none of these
    case kStXSsq: stageXSsq(c, unit, io); break;
    case kStTestCopy: stageTestCopy(c, unit, off, bar, io); break;
    case kStTestPub: stageTestPub(c, unit, io); break;
    case kStTestSub: stageTestSub(c, unit, io); break;
    case kStTestHang: stageTestHang(c, io); break;
#endif
    case kStIClearX: stageIClearX(c, unit); break;        // integration
#if !MSO_MK_LD_DS                                         // nor any of these
    case kStI2bGhost: stageI2bGhost(c, unit, bar); break; // ghost unit
    case kStI2bWarm: stageI2bWarm(c, unit); break;        // L2 warm-up
#endif
#if MSO_MK_LANES & 1                                      // attention-half stage guard
    // ---- attention half ----
    case kStAhQkvFin: stageAhQkvFin(c, unit, bar, io); break;
    case kStAhSwaAttn: stageAhSwaAttn(c, unit, bar, io); break;
    case kStAhOproj:
        stageAhOproj(c, unit, off, bar, io);
        break;
        // ---- end attention half ----
#endif
#if MSO_MK_LANES & 2                                      // full-attention stage guard
    case kStFaAttn: stageFaAttn(c, unit, off, io); break; // full attention
    case kStFaMerge: stageFaMerge(c, unit, io); break;    // full attention
#endif
#if MSO_MK_LANES & 4                                      // experts stage guard
    // ---- MoE experts begin ----
    case kStMoeExperts:
        stageMoeExperts(c, unit, off);
        break;
        // ---- end MoE experts ----
#endif
#if (MSO_MK_LANES & 16) && (MSO_MK_LANES & 4) // BF16 experts stage guard
    case kStMoeExpertsBf16: stageMoeExpertsBf16(c, unit, off); break;
#endif
#if MSO_MK_LANES & 8 // MoE-hop stage guard
#if MSO_MK_LD_DS     // a stage this build does not hold traps
    default:
        if (!mhRunUnit(c, stage, unit, off, bar, io))
        {
            __trap();
        }
        break;
#else
    default: mhRunUnit(c, stage, unit, off, bar, io); break; // MoE-side hops
#endif
#else
    default: break;
#endif
    }
}

// Role i of the CTA: the unit, then (thread 0) the weight requests of the later units issued after it.
__device__ __forceinline__ void runRole(
    Ctx& ctx, int const* sRoles, int i, int nRoles, uint32_t sb, int cta, Io const& io)
{
    int const tid = threadIdx.x;
    int const role = sRoles[i];
    int const stage = role >> 16;
    int const unit = role & 0xffff;
    uint32_t const off = static_cast<uint32_t>(sRoles[kMaxRoles + i]);
    uint32_t const bar = sb + kSmemSmall + kSmallBars + 8 * kPieces * i;
    if (MSO_MK_STAMPS(ctx) != nullptr && tid == 0)
    {
        MSO_MK_STAMPS(ctx)[i * 4] = stampTime();
#if MSO_MK_SY_PROG
        // The CTA's progress in host memory (stamped runs): after a trap, the unit each CTA was in.
        reinterpret_cast<int volatile*>(ctx.err)[16 + 4 * kCtas + cta] = (stage << 16) | (i << 8) | 1;
#endif
    }
    ctx.stampRole = i;
    runUnit(ctx, stage, unit, off, bar, io);
    __syncthreads();
    if (MSO_MK_STAMPS(ctx) != nullptr && tid == 0)
    {
        MSO_MK_STAMPS(ctx)[i * 4 + 3] = stampTime();
    }
    // With kIFlagI2bIssueWarp7 the later units' weight requests come from lane 0 of the last warp, so warp 0 (the
    // poller of most stages) starts the next unit at once; the issue code runs from a cold instruction cache.
    int const issuer = (icPIFlags(ctx.p[kPIFlags]) & kIFlagI2bIssueWarp7) ? kThreads - 32 : 0;
    // kIFlagW1OprojAfterQ: AhOproj requests its own weights (stageAhOproj) when they would follow the QKV run.
    bool const w1Defer = (icPIFlags(ctx.p[kPIFlags]) & kIFlagW1OprojAfterQ) && stage == kStQkvRun;
    if (tid == issuer)
    {
#pragma unroll 1
        for (int j = i + 1; j < nRoles; ++j)
        {
            if (sRoles[2 * kMaxRoles + j] == i && !(w1Defer && (sRoles[j] >> 16) == kStAhOproj))
            {
                issueUnitWeights(ctx, j, sb + kSmemSmall + kSmallBars + 8 * kPieces * j);
            }
        }
    }
#if MSO_MK_LANES & 2 // full-attention stage guard
    // ---- one-launch full layer: a FaAttn unit after the CTA's QkvRun (and QkvSum) gets its prologue and first ring
    // tiles right after the unit before it (before the dependency wait when the run polls early x) ----
    if ((sRoles[0] >> 16) == kStQkvRun && stage != kStFaAttn)
    {
        faAfterQkvRun(ctx, sRoles, i, nRoles);
    }
    // ---- end one-launch full layer ----
#endif
}

__global__ void __launch_bounds__(kThreads, 1) msoMegakernel(int64_t const* __restrict__ params, Io const io)
{
    extern __shared__ __align__(1024) uint8_t smemRaw[];
    int const tid = threadIdx.x;
    int const cta = static_cast<int>(blockIdx.x);
    uint32_t const sb = smemU32(smemRaw);
    // Weights first: thread 0 reads the CTA's first kFirstEarlyRoles table entries and requests every one of them whose
    // slot is free at entry (issue -1), before the parameter block, the setup and the grid-dependency wait.
    uint32_t firstIssued = 0;
    uint64_t tFirst = 0;
    uint64_t tCtx = 0;                                             // kStampCtx
    uint64_t const clk0 = MSO_MK_STAMPS_MODE == 0 ? clock64() : 0; // kStampClk0
    uint64_t tPro = 0;                                             // kStampPro
    if (tid == 0)
    {
        int const* tables = reinterpret_cast<int const*>(params + kPNumParams);
        int64_t const* wt = params + kPNumParams + kTableWords + static_cast<size_t>(cta) * 3 * kMaxRoles;
        int4 const iss = __ldg(reinterpret_cast<int4 const*>(tables + (2 * kCtas + cta) * kMaxRoles));
        int4 const offs = __ldg(reinterpret_cast<int4 const*>(tables + (kCtas + cta) * kMaxRoles));
        int64_t w[3 * kFirstEarlyRoles];
#pragma unroll
        for (int i = 0; i < 3 * kFirstEarlyRoles; ++i)
        {
            w[i] = __ldg(wt + i);
        }
        int64_t const flags = MSO_MK_KFLAGS(__ldg(params + kPFlags)); // MSO_MK_LD_KF
        int const issue[kFirstEarlyRoles] = {iss.x, iss.y, iss.z, iss.w};
        int const off[kFirstEarlyRoles] = {offs.x, offs.y, offs.z, offs.w};
#pragma unroll
        for (int i = 0; i < kFirstEarlyRoles; ++i)
        {
            uint32_t const n0 = static_cast<uint32_t>(w[3 * i + 2]);
            uint32_t const n1 = static_cast<uint32_t>(static_cast<uint64_t>(w[3 * i + 2]) >> 32);
            if (issue[i] == -1 && n0 + n1 != 0)
            {
                uint32_t const bar = sb + kSmemSmall + kSmallBars + 8 * kPieces * i;
#pragma unroll
                for (int p = 0; p < kPieces; ++p)
                {
                    mbarInit(bar + 8 * p, 1);
                }
                fenceMbarInit();
                issuePieces(sb + static_cast<uint32_t>(off[i]), reinterpret_cast<uint8_t const*>(w[3 * i]), n0,
                    reinterpret_cast<uint8_t const*>(w[3 * i + 1]), n1, bar, (flags & kFlagEvictFirst) != 0,
                    (flags & kFlagOnePiece) != 0);
                firstIssued |= 1u << i;
            }
        }
        tFirst = stampTime();
    }
    uint64_t const entry = globalTimer();
    // The parameter block, the CTA's rows of the role tables and its weight-table row into shared memory in one round
    // trip: every later read of them is a shared load.
    int64_t* sParams = reinterpret_cast<int64_t*>(smemRaw + kSmemSmall + kSmallParams);
    int* sRoles = reinterpret_cast<int*>(smemRaw + kSmemSmall + kSmallRoles);
    if (tid < 64)
    {
#if MSO_MK_I2B_PRO // thread 0 is busy with the entry weight requests; field 0 is thread 176's
        if (tid > 0 && tid < kPNumParams)
#else
        if (tid < kPNumParams)
#endif
        {
            sParams[tid] = params[tid];
        }
    }
    else if (tid >= 64 && tid < 64 + 3 * kMaxRoles)
    {
        int const* tables = reinterpret_cast<int const*>(params + kPNumParams);
        int const i = tid - 64;
        int const table = i / kMaxRoles;
        sRoles[i] = tables[(table * kCtas + cta) * kMaxRoles + i % kMaxRoles];
    }
    else if (tid >= 128 && tid < 128 + 3 * kMaxRoles)
    {
        int64_t const* table = params + kPNumParams + kTableWords;
        reinterpret_cast<int64_t*>(smemRaw + kSmemSmall + kSmallWeights)[tid - 128]
            = table[static_cast<size_t>(cta) * 3 * kMaxRoles + tid - 128];
    }
    // ---- parameter fields 64-127 ----
    else if (tid >= 192 && tid - 128 < kPNumParams)
    {
        sParams[tid - 128] = params[tid - 128];
    }
    // ---- end parameter fields 64-127 ----
#if MSO_MK_I2B_PRO // field 0
    else if (tid == 176)
    {
        sParams[0] = params[0];
    }
#endif
    __syncthreads();
    Ctx& ctx = *reinterpret_cast<Ctx*>(smemRaw + kSmemSmall + kSmallCtx);
    if (tid == 0)
    {
        ctx.p = sParams;
        ctx.smem = smemRaw;
        ctx.smemBase = sb;
        ctx.copy = static_cast<int>(sParams[kPCopy]);
        ctx.numCopies = static_cast<int>(sParams[kPNumCopies]);
        ctx.counters = reinterpret_cast<uint32_t*>(sParams[kPCounters]) + ctx.copy * kNumCounters;
        ctx.arena = reinterpret_cast<uint8_t*>(sParams[kPArena]) + static_cast<size_t>(ctx.copy) * kArenaBytes;
        ctx.err = reinterpret_cast<int*>(sParams[kPErr]);
        ctx.timeoutNs = static_cast<uint64_t>(sParams[kPTimeoutNs]);
        ctx.flags = sParams[kPFlags];
        ctx.stamps = sParams[kPStamps] != 0 && !(sParams[kPFlags] & kFlagNoStamps)
            ? reinterpret_cast<uint64_t*>(sParams[kPStamps]) + static_cast<size_t>(cta) * kStampsPerCta
            : nullptr;
        ctx.stampRole = 0;
        ctx.w1Early = 0; // set from the early-x sample below
    }
    __syncthreads();
    if ((kIcPIZero != 0 && (sParams[kPIFlags] & kIcPIZero) != 0)
        || (kIcPIOne != 0 && (sParams[kPIFlags] & kIcPIOne) != kIcPIOne))
    {
        __trap(); // this launch sets a kPIFlags bit the build fixed to 0 or clears one it fixed to 1
    }
#if MSO_MK_LD_KF
    if ((sParams[kPFlags] & ~int64_t(kFlagNoStamps)) != MSO_MK_KFLAGS(0))
    {
        __trap(); // MSO_MK_LD_KF: the build fixes the kernel Flags word to the served one
    }
#endif
#if MSO_MK_LD_CK
    if ((sParams[kPAhAttnOut] | sParams[kPFaOut] | sParams[kPFaOutF32] | sParams[kPFaDbgO] | sParams[kPMhOutXffn]
            | sParams[kPMhOutPool] | sParams[kPMhOutLogits] | sParams[kPMhOutM] | sParams[kPMhOutRes2]
            | sParams[kPMhOutB])
            != 0
        || io.qkv != nullptr || io.qkvF32 != nullptr)
    {
        __trap(); // MSO_MK_LD_CK: the build has no check / test outputs
    }
#endif
#if MSO_MK_LD_SC
    if (sParams[kPCopy] < 0 || sParams[kPCopy] >= sParams[kPNumCopies])
    {
        __trap(); // MSO_MK_LD_SC: the copy neighbours are selects (0 <= copy < numCopies)
    }
#endif
    if (MSO_MK_STAMPS_MODE == 0 && tid == 0) // kStampCtx
    {
        tCtx = stampTime();
    }
    // ---- kIFlagW1WarmQkv: the last thread samples one word of the CTA's early x; the load completes during the
    // rest of the prologue and its result is kept in ctx.w1Early before the prologue's last barrier ----
    uint32_t w1Probe = 0u;
#if (MSO_MK_AH & 48) == 16 // AH WD: the same sample decides the dry merge pass before the CTA's QKV run
    bool const w1Probing
        = tid == kThreads - 1 && (icPIFlags(sParams[kPIFlags]) & kIFlagXIn) && (sRoles[0] >> 16) == kStQkvRun;
#else
    bool const w1Probing = tid == kThreads - 1 && (icPIFlags(sParams[kPIFlags]) & kIFlagW1WarmQkv)
        && (icPIFlags(sParams[kPIFlags]) & kIFlagXIn) && (sRoles[0] >> 16) == kStQkvRun;
#endif
    if (w1Probing)
    {
        int const slice = (qkvRunStart(sRoles[0] & 0xffff) / kSegChunks) / kQkvTiles;
        uint8_t const* xb
            = reinterpret_cast<uint8_t const*>(sParams[kPIXBuf]) + static_cast<size_t>(ctx.copy) * kIXBytes;
        w1Probe = ldPoll32(xb + slice * kSegChunks * kChunkCols * 2);
    }
    // ---- end kIFlagW1WarmQkv ----
    // Warp 0: the barriers not initialized above; thread 0 then requests the other entry-issued units' weights (and
    // with kFlagL2Deferred prefetches the blocked units' weights into L2).
    if (tid < 32)
    {
        uint32_t const issued = __shfl_sync(0xffffffffu, firstIssued, 0);
        static_assert(kMaxRoles * kPieces == 64, "two barriers per lane");
#pragma unroll
        for (int k = 0; k < 2; ++k)
        {
            int const idx = tid + 32 * k;
            if (!((issued >> (idx / kPieces)) & 1u))
            {
                mbarInit(sb + kSmemSmall + kSmallBars + 8 * idx, 1);
            }
        }
        fenceMbarInit();
        __syncwarp();
        if (tid == 0)
        {
#pragma unroll 1
            for (int i = 0; i < kMaxRoles && sRoles[i] != 0; ++i)
            {
                int const issue = sRoles[2 * kMaxRoles + i];
                if (issue == -1 && !((issued >> i) & 1u))
                {
                    issueUnitWeights(ctx, i, sb + kSmemSmall + kSmallBars + 8 * kPieces * i);
                }
                else if (issue >= 0 && (sRoles[i] >> 16) == kStQkvRun && (icPIFlags(sParams[kPIFlags]) & kIFlagFaRunL2))
                {
                    faRunL2(ctx, i);
                }
                else if (issue >= 0 && (MSO_MK_KFLAGS(ctx.flags) & kFlagL2Deferred)) // MSO_MK_LD_KF
                {
                    prefetchUnitL2(ctx, i);
                }
#if MSO_MK_LANES & 1 // attention-half stage guard
                // ---- attention half: pre-wait copies of graph inputs / immutable data (pages, RoPE rows) ----
                else if (issue == -2 && ((sRoles[i] >> 16) == kStAhSwaAttn || (sRoles[i] >> 16) == kStAhQkvFin)
                    && !ahEntryHookless(ctx, sRoles[i])) // kIFlagI2bEntryLite
                {
                    ahEntryIssue(ctx, sRoles[i], sb + kSmemSmall + kSmallBars + 8 * kPieces * i);
                }
                // ---- end attention half ----
#endif
            }
            if (MSO_MK_STAMPS_MODE == 0) // kStampPro
            {
                tPro = stampTime();
            }
        }
    }
    // The stamps (one word per thread; the barrier below orders them before the units' stamps).
    if (MSO_MK_STAMPS(ctx) != nullptr && tid >= 64)
    {
        for (int i = tid - 64; i < kStampsPerCta; i += kThreads - 64)
        {
            MSO_MK_STAMPS(ctx)[i] = i == kStampEntry ? entry : i == kStampSmid ? smid() : 0ull;
        }
    }
    // ---- full attention: the units' KV pages are requested before the grid-dependency wait ----
#if MSO_MK_LANES & 2 // full-attention stage guard
    if (tid < 32)
    {
#pragma unroll 1
        for (int i = 0; i < kMaxRoles && sRoles[i] != 0; ++i)
        {
            if (i == 0 && (sRoles[i] >> 16) == kStFaAttn) // a later FaAttn unit (after QkvRun): faAfterQkvRun
            {
                faPrologue(ctx, sRoles[i] & 0xffff, static_cast<uint32_t>(sRoles[kMaxRoles + i]));
            }
            else if ((sRoles[i] >> 16) == kStFaAttn) // a later FaAttn unit: its metadata into L2 now
            {
                faPrefetchMeta(ctx, sRoles[i] & 0xffff);
            }
        }
    }
#endif
    // ---- end full attention ----
    // ---- kIFlagFaEarlyMeta: warp 1 fills the meta area of the CTA's FaAttn unit that follows its first unit ----
#if MSO_MK_LANES & 2 // full-attention stage guard
    if (tid >= 32 && tid < 64 && (icPIFlags(sParams[kPIFlags]) & kIFlagFaEarlyMeta))
    {
#pragma unroll 1
        for (int i = 1; i < kMaxRoles && sRoles[i] != 0; ++i)
        {
            if ((sRoles[i] >> 16) == kStFaAttn)
            {
                faEarlyMeta(ctx, sRoles[i] & 0xffff);
                break;
            }
        }
    }
#endif
    // ---- end kIFlagFaEarlyMeta ----
    // ---- the boundaries' cluster exchange (stages/moe_hops_xchg.cuh): slot area to kEmpty, cluster arrive ----
#if MSO_MK_LANES & 8 // MoE-hop stage guard (the exchange belongs to the MoE hop stages)
    mhXchgEntry(smemRaw);
#endif
    // ---- end cluster exchange ----
    if (sParams[kPMagic] != kMagic || sParams[kPNumCtas] != static_cast<int64_t>(gridDim.x) || gridDim.x != kCtas
        || cta >= kCtas // SM-stable roles: the SM id must be a block index
        || sParams[kPMaxRoles] != kMaxRoles || (sb & 1023u) != 0)
    {
        __trap();
    }
    if (w1Probing) // kIFlagW1WarmQkv
    {
        ctx.w1Early = w1Probe == kEmpty ? 1 : 0;
    }
    __syncthreads();
    // ---- full attention: every warp issues its share of the units' first ring tiles (still before the wait) ----
#if MSO_MK_LANES & 2 // full-attention stage guard
#pragma unroll 1
    for (int i = 0; i < kMaxRoles && sRoles[i] != 0; ++i)
    {
        if (i == 0 && (sRoles[i] >> 16) == kStFaAttn) // role 0 only (see faAfterQkvRun)
        {
            faPrologueIssue(ctx, static_cast<uint32_t>(sRoles[kMaxRoles + i]));
        }
    }
#endif
    // ---- end full attention ----
    if (MSO_MK_STAMPS(ctx) != nullptr && tid == 0)
    {
        MSO_MK_STAMPS(ctx)[kStampIssued] = stampTime();
        MSO_MK_STAMPS(ctx)[kStampFirst] = tFirst;
        MSO_MK_STAMPS(ctx)[kStampCtx] = tCtx;
        MSO_MK_STAMPS(ctx)[kStampPro] = tPro;
        MSO_MK_STAMPS(ctx)[kStampClk0] = clk0;
    }
    int nRoles = 0;
#pragma unroll 1
    while (nRoles < kMaxRoles && sRoles[nRoles] != 0)
    {
        ++nRoles;
    }
    // ---- early x: the first unit (QkvRun) polls the previous launch's x_attn before the wait ----
    int first = 0;
    if ((icPIFlags(sParams[kPIFlags]) & kIFlagXIn) && nRoles > 0 && (sRoles[0] >> 16) == kStQkvRun)
    {
        first = 1;
    }
    // ---- kIFlagPreWaitAttn: the attention-half units that follow the QKV run also run before the wait ----
    if (first == 1 && (icPIFlags(sParams[kPIFlags]) & kIFlagPreWaitAttn))
    {
        while (first < nRoles)
        {
            int const st = sRoles[first] >> 16;
            if (st != kStQkvSum && st != kStFaAttn && st != kStFaMerge && st != kStClear && st != kStAhOproj
                && st != kStAhQkvFin && st != kStAhSwaAttn)
            {
                break;
            }
            ++first;
        }
    }
    // ---- end kIFlagPreWaitAttn ----
#if (MSO_MK_AH & 4) || (MSO_MK_LY & 4) // LY bit 4 compiles the block without FF's plan (inert)
    // ---- AH FFP: an AhQkvFin unit ahead of the CTA's QKV run (FF's plan) and the run run before the wait, then the
    // attention-half units that follow them, as the kIFlagPreWaitAttn and kIFlagI2bPreWait loops do for a CTA whose
    // run is role 0; only when the fin's stage is a pre-wait stage under the launch's flags. The fin reads only this
    // launch's run partials (polled) and its step-immutable RoPE entry copy, and writes what it writes when it follows
    // the run before the wait; the run polls its x copy as role 0 does. ----
    {
        auto const fl = icPIFlags(sParams[kPIFlags]);
        auto const preWait = [fl](int st)
        {
            return ((fl & kIFlagPreWaitAttn)
                       && (st == kStQkvSum || st == kStFaAttn || st == kStFaMerge || st == kStClear || st == kStAhOproj
                           || st == kStAhQkvFin || st == kStAhSwaAttn))
                || ((fl & kIFlagI2bPreWait) && i2bPreWaitStage(st));
        };
        if (first == 0 && (fl & kIFlagXIn) && nRoles > 1 && (sRoles[0] >> 16) == kStAhQkvFin
            && (sRoles[1] >> 16) == kStQkvRun && preWait(kStAhQkvFin))
        {
            first = 2;
            while (first < nRoles && preWait(sRoles[first] >> 16))
            {
                ++first;
            }
        }
#if MSO_MK_AH & 8
        // ---- AH GP: FF's fin CTA without a QKV run (its gate run is another CTA's second run): the fin and the
        // pre-wait units after it ----
        else if (first == 0 && (fl & kIFlagXIn) && nRoles > 1 && (sRoles[0] >> 16) == kStAhQkvFin
            && preWait(kStAhQkvFin))
        {
            first = 1;
            while (first < nRoles && preWait(sRoles[first] >> 16))
            {
                ++first;
            }
        }
        // ---- end AH GP ----
#endif
    }
    // ---- end AH FFP ----
#endif
    // The kIFlagI2bPreWait loop below is a no-op once the kIFlagPreWaitAttn loop advanced (its stage set is a subset).
    // ---- kIFlagI2bPreWait: the CTA's leading attention-half units run before the wait too ----
    if (first == 1 && (icPIFlags(sParams[kPIFlags]) & kIFlagI2bPreWait))
    {
        while (first < nRoles && i2bPreWaitStage(sRoles[first] >> 16))
        {
            ++first;
        }
    }
    // ---- end kIFlagI2bPreWait ----
#if (MSO_MK_AH & 48) == 16 && (MSO_MK_LANES & 1)
    // ---- AH WD: a CTA whose pre-wait roles are its QKV run (role 0, early x) and a merging AhSwaAttn unit runs that
    // unit's merge once dry before the run, when the prologue's sample read the CTA's early x unpublished (the run
    // would wait for it). The dry pass reads no global memory and writes only kSmemRed (no async copy targets it). ----
    if (ctx.w1Early != 0 && first > 1 && (sRoles[0] >> 16) == kStQkvRun)
    {
#pragma unroll 1
        for (int i = 1; i < first; ++i)
        {
            int const u = sRoles[i] & 0xffff;
            if ((sRoles[i] >> 16) == kStAhSwaAttn && u % kAhBlocks < kAhBlocks - 1)
            {
                ahMerge(ctx, u / kAhBlocks, u % kAhBlocks, io, true);
                break;
            }
        }
    }
    // ---- end AH WD ----
#endif
#if MSO_MK_LD_R1
    // ---- MSO_MK_LD_R1: the roles before the wait (0 .. first - 1), the wait block, then the roles after it, as one
    // loop with one runRole instance (first <= nRoles: the wait block runs once, before role first or after the last
    // role) ----
#pragma unroll 1
    for (int i = 0;; ++i)
    {
        if (i == first)
        {
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#if MSO_MK_LANES & 8       // MoE-hop stage guard
            mhXchgReady(); // every peer of the cluster initialised its slot area
#endif
            if (MSO_MK_STAMPS(ctx) != nullptr && tid == 0)
            {
                MSO_MK_STAMPS(ctx)[kStampWait] = syAnchor(MSO_MK_STAMPS(ctx), 0); // with its clock64 bracket
            }
            // The next launch's counters (last used by launch n + 1 - copies <= n - 2, complete before the wait).
            {
                uint32_t* next = reinterpret_cast<uint32_t*>(sParams[kPCounters])
                    + (MSO_MK_COPY_NEXT(ctx.copy, ctx.numCopies)) * kNumCounters;
                for (int c = cta * kThreads + tid; c < kNumCounters; c += kCtas * kThreads)
                {
                    next[c] = 0;
                }
            }
        }
        if (i >= nRoles)
        {
            break;
        }
        runRole(ctx, sRoles, i, nRoles, sb, cta, io);
    }
    // ---- end MSO_MK_LD_R1 ----
#else
    for (int i = 0; i < first; ++i)
    {
        runRole(ctx, sRoles, i, nRoles, sb, cta, io);
    }
    // ---- end early x ----
    // The previous kernel's outputs (x, the residual, the step inputs) are read after the grid-dependency wait. Before
    // it: weight copies and L2 prefetches only (immutable), the parameter block and the tables.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#if MSO_MK_LANES & 8 // MoE-hop stage guard
    mhXchgReady(); // every peer of the cluster initialised its slot area
#endif
    if (MSO_MK_STAMPS(ctx) != nullptr && tid == 0)
    {
        MSO_MK_STAMPS(ctx)[kStampWait] = syAnchor(MSO_MK_STAMPS(ctx), 0); // with its clock64 bracket
    }
    // The next launch's counters (last used by launch n + 1 - copies <= n - 2, complete before this wait returned).
    {
        uint32_t* next = reinterpret_cast<uint32_t*>(sParams[kPCounters])
            + (MSO_MK_COPY_NEXT(ctx.copy, ctx.numCopies)) * kNumCounters;
        for (int i = cta * kThreads + tid; i < kNumCounters; i += kCtas * kThreads)
        {
            next[i] = 0;
        }
    }

    for (int i = first; i < nRoles; ++i)
    {
        runRole(ctx, sRoles, i, nRoles, sb, cta, io);
    }
#endif
    __syncthreads();
    if (MSO_MK_STAMPS(ctx) != nullptr && tid == 0)
    {
        MSO_MK_STAMPS(ctx)[kStampExit] = syAnchor(MSO_MK_STAMPS(ctx), 2); // with its clock64 bracket
        MSO_MK_STAMPS(ctx)[kStampClk1] = clock64();
#if MSO_MK_SY_PROG
        reinterpret_cast<int volatile*>(ctx.err)[16 + 4 * kCtas + cta] = 0xffff;
#endif
    }
}

// A latency-bound predecessor for the benchmarks: waits for its own predecessor, releases its dependents at entry
// (early) or at exit, spins spinNs, then copies nVec 16-byte vectors src -> dst (the next kernel's input). quiet: one
// thread per CTA reads the clock, with __nanosleep between reads, while the others wait at the barrier (else every
// thread reads %globaltimer in a tight loop).
#if MSO_MK_I2B_SMID // the SM-stable block index ends with the megakernel
#undef blockIdx
#endif
__global__ void msoSpinKernel(uint64_t spinNs, int early, int quiet, uint4 const* src, uint4* dst, int nVec)
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (early)
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    uint64_t const t0 = globalTimer();
    if (quiet)
    {
        if (threadIdx.x == 0)
        {
            while (globalTimer() - t0 < spinNs)
            {
                __nanosleep(256);
            }
        }
        __syncthreads();
    }
    else
    {
        while (globalTimer() - t0 < spinNs)
        {
        }
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nVec; i += gridDim.x * blockDim.x)
    {
        dst[i] = src[i];
    }
}

#else

__global__ void msoMegakernel(int64_t const*, Io const) {}

__global__ void msoWeightTableKernel(int64_t const*, int64_t*) {}

__global__ void msoSpinKernel(uint64_t, int, int, uint4 const*, uint4*, int) {}

#endif

} // namespace mso_mk

// ----------------------------------------------------------------------------------------------------------------
// C ABI (loaded with ctypes by library.py); errors are returned as cudaError_t values, never thrown.
// ----------------------------------------------------------------------------------------------------------------
extern "C"
{

    int msoMkSmemBytes()
    {
        return mso_mk::kSmemBytes;
    }

    int msoMkSmemMain()
    {
        return mso_mk::kSmemMain;
    }

    int msoMkNumParams()
    {
        return mso_mk::kPNumParams;
    }

    int64_t msoMkMagic()
    {
        return mso_mk::kMagic;
    }

    int msoMkCtas()
    {
        return mso_mk::kCtas;
    }

    int msoMkMaxRoles()
    {
        return mso_mk::kMaxRoles;
    }

    int msoMkTableWords()
    {
        return mso_mk::kTableWords;
    }

    int msoMkWeightWords()
    {
        return mso_mk::kWeightWords;
    }

    int msoMkArenaBytes()
    {
        return mso_mk::kArenaBytes;
    }

    int msoMkClearUnits()
    {
        return mso_mk::kClearUnits;
    }

    int msoMkNumCounters()
    {
        return mso_mk::kNumCounters;
    }

    int msoMkStampsPerCta()
    {
        return mso_mk::kStampsPerCta;
    }

    int msoMkQkvPartBytes()
    {
        return mso_mk::kQkvPartBytes;
    }

    int msoMkQkvSlices()
    {
        return mso_mk::kQkvSlices;
    }

    int msoMkLean()
    {
        return MSO_MK_LEAN;
    }

    int msoMkStampsMode()
    {
        return MSO_MK_STAMPS_MODE;
    }

    // ---- the stamps' clock (0: %globaltimer ns; 1: clock64 cycles with the per-CTA anchors) and the probe layout ----
    int msoMkStampsClock()
    {
        return MSO_MK_SY_CLK;
    }

    int msoMkStampSy()
    {
        return mso_mk::kStampSy;
    }

    // ---- end stamps' clock and probe layout ----

    // ---- attention half ----
    // The attention partials' bytes (stages/attnHalf.cuh kAhPartBytes; the header is device-only).
    int msoMkAhPartBytes()
    {
        return 2 * 33 * 64 * 64 * 4 + 2 * 33 * 64 * 2 * 4;
    }

    // ---- end attention half ----
    // ---- full attention ----
    int msoMkFaRegionBytes()
    {
        return mso_mk::fa::kRegionBytes;
    }

    int msoMkFaRing()
    {
        return mso_mk::fa::kRing;
    }

    int msoMkFaMaxUnitPages()
    {
        return mso_mk::fa::kMaxUnitPages;
    }

    // The 2D tensor map of a layer's FP8 pool view (numBlocks blocks of 4 KiB = 64 rows of 64 B): box 64 x boxRows (32:
    // one head; 64: both heads of a block), SWIZZLE_64B, L2 promotion MSO_MK_FA_L2PROMO, written to out (128 B host
    // memory).
    int msoMkFaTensorMap(void* pool, long long numBlocks, void* out, int boxRows)
    {
        using Encode = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, cuuint64_t const*,
            cuuint64_t const*, cuuint32_t const*, cuuint32_t const*, CUtensorMapInterleave, CUtensorMapSwizzle,
            CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
        void* fn = nullptr;
        cudaDriverEntryPointQueryResult q{};
        cudaError_t e = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
        if (e != cudaSuccess || fn == nullptr || q != cudaDriverEntryPointSuccess)
        {
            return e != cudaSuccess ? static_cast<int>(e) : static_cast<int>(cudaErrorSymbolNotFound);
        }
        cuuint64_t const dims[2] = {64, static_cast<cuuint64_t>(numBlocks) * 64};
        cuuint64_t const strides[1] = {64};
        cuuint32_t const box[2] = {64, static_cast<cuuint32_t>(boxRows)};
        cuuint32_t const es[2] = {1, 1};
        CUresult const r = reinterpret_cast<Encode>(fn)(static_cast<CUtensorMap*>(out), CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2, pool, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
            MSO_MK_FA_L2PROMO == 256       ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                : MSO_MK_FA_L2PROMO == 128 ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                : MSO_MK_FA_L2PROMO == 64  ? CU_TENSOR_MAP_L2_PROMOTION_L2_64B
                                           : CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        return r == CUDA_SUCCESS ? 0 : 1000 + static_cast<int>(r);
    }

    // ---- end full attention ----
    // ---- the MoE hop layout (moe_hops.py checks its mirror), and the FMA-contraction mode of the boundaries ----
    int msoMkMhLayout(int64_t* out)
    {
        using namespace mso_mk;
        int64_t const v[]
            = { kMhCopyBytes,
                  kMhH1Off,
                  kMhPoolOff,
                  kMhLogitOff,
                  kMhFinOff,
                  kMhF1Off,
                  kMhArenaBytes,
                  kMhArX,
                  kMhArM,
                  kMhArSsq,
                  kMhClearUnits,
                  kMhClearUnitBytes,
                  MSO_MK_MH_FMA,
                  kPMhUc,
                  kPMhOutXattn,
                  kMhRowU,
                  kMhCoefH1Bytes,
                  kMhCoefF1Bytes
#if MSO_MK_NB // nb: the one-exchange mask and its arena rows (records of 18 fields: MSO_MK_NB = 0)
                  ,
                  MSO_MK_NB,
                  kMhArAbc
#if MSO_MK_BH // bh: the two-term mask and its first arena plane (records of 20 fields: MSO_MK_BH = 0)
                  ,
                  MSO_MK_BH,
                  kMhArRA
#elif MSO_MK_FZ // fz without bh: bh's pair as (0, 0)
                  ,
                  0,
                  0
#endif
#if MSO_MK_FZ // fz: the leaf mask, the fz block per copy and its r' area (records of at most 22 fields: MSO_MK_FZ = 0)
                  ,
                  MSO_MK_FZ,
                  kMhFzBytes,
                  kMhFzMaxCopies
#endif
#endif
              };
        int const n = static_cast<int>(sizeof(v) / sizeof(v[0]));
        for (int i = 0; i < n; ++i)
        {
            out[i] = v[i];
        }
        return n;
    }

    // ---- end MoE hop layout ----

    // ---- rl: the rl layout (rl.py checks its mirror) ----
    int msoMkRlLayout(int64_t* out)
    {
        using namespace mso_mk;
        int64_t const v[] = {kRlUnits, kRlUnitBytes, kRlTermBytes, kRlCopyBytes, kRlRecBytes, kRlRecResOff, kRlRecOOff,
            kRlOracleBytes, kPRl, kRlBlockWords, kIFlagRlOracleLg, kIFlagRlRecord, kIFlagRlOraclePool, kIFlagRlOn,
            kRlOk ? 1 : 0, MSO_MK_RL};
        int const n = static_cast<int>(sizeof(v) / sizeof(v[0]));
        for (int i = 0; i < n; ++i)
        {
            out[i] = v[i];
        }
        return n;
    }

    // ---- end rl ----

    // ---- the boundaries' exchange mode and the launch's cluster dimension; cluster admission of the megakernel ----
    // 1: every cluster of a launch can be co-resident (set by msoMkInit); a cluster launch is refused otherwise.
    static int gMhClusterAdmitted = -1;

    int msoMkMhXchg(int* out)
    {
        out[0] = mso_mk::kMhXchg;
        out[1] = mso_mk::kMhClusterDim;
        out[2] = mso_mk::kMhF1Full ? 1 : 0;
        return 3;
    }

    // The number of clusters of `dim` CTAs of the megakernel (kCtas CTAs, kSmemBytes) that can be co-resident
    // (cudaOccupancyMaxActiveClusters); the launch needs kCtas / dim of them.
    int msoMkMaxActiveClusters(int dim, int* out)
    {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(mso_mk::kCtas);
        config.blockDim = dim3(mso_mk::kThreads);
        config.dynamicSmemBytes = mso_mk::kSmemBytes;
        cudaLaunchAttribute attrs[1]{};
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim.x = dim;
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
        int n = 0;
        cudaError_t const e = cudaOccupancyMaxActiveClusters(&n, mso_mk::msoMegakernel, &config);
        out[0] = n;
        return static_cast<int>(e);
    }

    // ---- end exchange mode and cluster admission ----

    int msoMkInit()
    {
        int dev = 0;
        int optin = 0;
        cudaError_t e = cudaGetDevice(&dev);
        if (e != cudaSuccess)
        {
            return static_cast<int>(e);
        }
        e = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (e != cudaSuccess)
        {
            return static_cast<int>(e);
        }
        if (mso_mk::kSmemBytes > optin)
        {
            return static_cast<int>(cudaErrorInvalidValue);
        }
        e = cudaFuncSetAttribute(
            mso_mk::msoMegakernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mso_mk::kSmemBytes);
        if (e != cudaSuccess)
        {
            return static_cast<int>(e);
        }
        e = cudaFuncSetAttribute(
            mso_mk::msoMegakernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        if (e != cudaSuccess)
        {
            return static_cast<int>(e);
        }
        // ---- cluster admission, once, outside any capture (msoMkLaunch refuses a launch that cannot co-reside) ----
        if (mso_mk::kMhClusterDim > 1)
        {
            int n = 0;
            int const rc = msoMkMaxActiveClusters(mso_mk::kMhClusterDim, &n);
            gMhClusterAdmitted = (rc == 0 && n * mso_mk::kMhClusterDim >= mso_mk::kCtas) ? 1 : 0;
        }
        // ---- end cluster admission ----
        return 0;
    }

    // Registers per thread and local (spill) bytes of the megakernel, for the build report.
    int msoMkAttributes(int* out)
    {
        cudaFuncAttributes a{};
        cudaError_t const e = cudaFuncGetAttributes(&a, mso_mk::msoMegakernel);
        out[0] = a.numRegs;
        out[1] = static_cast<int>(a.localSizeBytes);
        out[2] = static_cast<int>(a.maxThreadsPerBlock);
        out[3] = a.binaryVersion;
        return static_cast<int>(e);
    }

    // The weight table of a parameter block (eager, before capture): out = params + kPNumParams + kTableWords.
    int msoMkWeightTable(void const* params, void* out, void* stream)
    {
        mso_mk::msoWeightTableKernel<<<mso_mk::kCtas, mso_mk::kMaxRoles, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<int64_t const*>(params), static_cast<int64_t*>(out));
        return static_cast<int>(cudaGetLastError());
    }

    // io: 8 device pointers in mso_mk::Io order (host memory).
    int msoMkLaunch(void const* params, void const* const* io, void* stream, int pdl)
    {
        mso_mk::Io args{io[0], io[1], const_cast<void*>(io[2]), const_cast<void*>(io[3]), const_cast<void*>(io[4]),
            const_cast<void*>(io[5]), io[6], io[7]};
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(mso_mk::kCtas);
        config.blockDim = dim3(mso_mk::kThreads);
        config.dynamicSmemBytes = mso_mk::kSmemBytes;
        config.stream = static_cast<cudaStream_t>(stream);
        cudaLaunchAttribute attrs[2]{};
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
        config.attrs = attrs;
        config.numAttrs = 1;
        // ---- the MoE hop boundaries' row clusters (MSO_MK_MH_XCHG = 2) ----
        if (mso_mk::kMhClusterDim > 1)
        {
            // The persistent kernel's CTAs wait on each other: every cluster must be co-resident (msoMkInit checked
            // it).
            if (gMhClusterAdmitted != 1)
            {
                return static_cast<int>(cudaErrorInvalidConfiguration);
            }
            attrs[1].id = cudaLaunchAttributeClusterDimension;
            attrs[1].val.clusterDim.x = mso_mk::kMhClusterDim;
            attrs[1].val.clusterDim.y = 1;
            attrs[1].val.clusterDim.z = 1;
            config.numAttrs = 2;
        }
        // ---- end row clusters ----
        cudaError_t const e
            = cudaLaunchKernelEx(&config, mso_mk::msoMegakernel, static_cast<int64_t const*>(params), args);
        if (e != cudaSuccess)
        {
            return static_cast<int>(e);
        }
        return static_cast<int>(cudaGetLastError());
    }

    int msoMkSpin(unsigned long long spinNs, int early, int quiet, void const* src, void* dst, int nVec, int ctas,
        int threads, void* stream, int pdl)
    {
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(ctas);
        config.blockDim = dim3(threads);
        config.dynamicSmemBytes = 0;
        config.stream = static_cast<cudaStream_t>(stream);
        cudaLaunchAttribute attrs[1]{};
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
        config.attrs = attrs;
        config.numAttrs = 1;
        cudaError_t const e = cudaLaunchKernelEx(&config, mso_mk::msoSpinKernel, static_cast<uint64_t>(spinNs), early,
            quiet, static_cast<uint4 const*>(src), static_cast<uint4*>(dst), nVec);
        if (e != cudaSuccess)
        {
            return static_cast<int>(e);
        }
        return static_cast<int>(cudaGetLastError());
    }

// ---- MoE experts begin ----
#include "stages/experts_abi.cuh"
// ---- end MoE experts ----
// ---- BF16 MoE experts ----
#include "stages/bf16Experts_abi.cuh"
    // ---- end BF16 MoE experts ----

} // extern "C"
