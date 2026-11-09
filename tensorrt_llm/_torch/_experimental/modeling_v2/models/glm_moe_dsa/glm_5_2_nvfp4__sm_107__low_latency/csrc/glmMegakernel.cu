/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// One GLM-5.2-NVFP4 plain decoder layer (a MoE layer that reuses its leader's top-k) as one kernel, for one shape:
// Rubin VR200 (212 SMs), attention TP16 + MoE TP16, 8 query rows (batch 1, MTP 7), FP8 KV in 64-token blocks.
//
// The launch is 212 CTAs of 256 threads, one per SM. Every CTA runs a list of work units fixed on the host (stage,
// unit, shared-memory offset, and after which unit its weights are requested), so the weights of a unit are in shared
// memory before its input exists whenever its slot is free early.
//
// Data is its own flag. Every activation buffer starts as the empty word 0x80000000 (FP32 -0.0) in every 32-bit word;
// producers store whole words (a payload word equal to the empty word is stored as 0, the same value in every format
// used here), consumers poll exactly the words they read. Buffers read by one consumer are cleared by it after use;
// the others live in an arena with C >= 3 copies (kPNumCopies): launch n (of the replaced layers, in order) uses copy
// n % C and clears copy (n + 1) % C for the next launch (its readers, launch n + 1 - C <= n - 2, finished before launch
// n began). The counters come in C sets the same way. The two all-reduces over the 16 GPUs work the same way across
// GPUs: producers store their shard through the multicast address into every GPU's slot buffer. The stores are weak and
// the polls volatile: PTX promises a whole-word read only between strong accesses, so this relies on the hardware
// storing aligned 32-bit words whole, and no system-scope fence follows the multicast stores (fence.sc.sys there cost
// +52..+58 us per slice).
//
// Stages, in dependency order: fused-A (q_a | c_kv | k_pe), q_b' (absorbed, q_a RMSNorm folded, RoPE, E4M3 Q), the
// KV append, the sparse attention over the 2048 selected rows (per row 16 blocks of 128), the merge, o_proj with v_b
// folded in (W_OV, K = 4 heads x 512; publishes the attention all-reduce), the attention all-reduce + residual +
// RMSNorm + NVFP4 quantization, the router GEMV and the noaux_tc top-8, the shared expert, the routed experts (FC1
// blocks with SwiGLU + NVFP4, FC2 eighths, dealt over all CTAs), the top-8 combine + shared add (publishes the MoE
// all-reduce), and the MoE all-reduce + residual + next RMSNorm.
//
// Every GEMM is mma.sync m16n8k16 with M = weight rows and N = the 8 rows: weights are repacked on the host into the
// fragment order of the MMA (BF16 directly, NVFP4 with their E4M3 block scales applied to exact FP16 values), with a
// K permutation inside each 64-wide chunk that lets every thread read its activation fragment as 32 contiguous bytes.

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/csrc/glmMegakernel.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#define GLM52_STAMPS(c) static_cast<uint64_t*>(nullptr)
#define GLM52_FLAGS(c) specializedFlags(c)

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::megakernel
{

namespace glm52
{

// ----------------------------------------------------------------------------------------------------------------
// Shape
// ----------------------------------------------------------------------------------------------------------------
constexpr int kThreads = 256;
constexpr int kWarps = 8;
constexpr int kCtas = 212;
constexpr int kTok = 8;
constexpr int kHidden = 6144;
constexpr int kRanks = 16;
constexpr int kHeads = 4;
constexpr int kFaRows = 2624;
constexpr int kFaChunks = 96;
constexpr int kQbRows = 2304;
constexpr int kQbChunks = 32;
constexpr int kQbUnitTiles = 3;
constexpr int kTopK = 2048;
constexpr int kRowBytes = 576;
constexpr int kSmemRowBytes = kRowBytes + 16;
constexpr int kAttnBlocks = 16;
constexpr int kAttnKeys = kTopK / kAttnBlocks;
constexpr int kPRowBytes = kAttnKeys + 16;
constexpr int kDv = 512;
constexpr int kOChunks = 32;
constexpr int kRChunks = 96;
constexpr int kExperts = 256;
constexpr int kColsPerHopUnit = kHidden / 4;
constexpr int kBf16Block = 2048;
constexpr float kPScale = 448.f;
constexpr float kEps = 1e-5f;
constexpr uint32_t kEmpty = 0x80000000u;

// Routed experts: 16E FC1 pieces (g = 16 slot + 8 tile + K piece: 128 rows x 6 K blocks of 128) and 8E FC2 pieces
// (j = 8 slot + eighth: output rows [768 eighth, +768) = 6 tiles of 128 rows), each list cut into 212 equal runs, CTA c
// taking FC1 pieces [16E c / 212, 16E (c + 1) / 212) and then FC2 pieces [8E c / 212, 8E (c + 1) / 212): every
// CTA's FC1 work is one round of 4-5 pieces (E = 57), so all FC1 partials are in at about the same time. A piece is
// 48 KiB of E2M1 in SWIZZLE_64B atoms (128 rows x 64 B per K block of 128) and its 12 K steps of block scales (512 B
// each, the 128x4 layout).
constexpr int kPieceW = 6 * 128 * 64;          // 49,152
constexpr int kPieceS = 12 * 512;              // 6,144
constexpr int kPieceBytes = kPieceW + kPieceS; // 55,296
constexpr int kRingSlots = 3;
constexpr int kHfp4Bytes = kTok * 3072;        // h in E2M1, [8][3072]
constexpr int kHsfBytes = kTok * 384;          // its E4M3 block scales, [8][384]

// One all-reduce buffer (one launch copy): [8 tokens][16 ranks][6144] BF16.
constexpr int kHopBytes = kTok * kRanks * kHidden * 2;
// The leader's top-k exchange (one launch copy, multicast like the all-reduce buffers): int32 [8 rows][2048], each row
// published by the even rank of the pair that selects it (the row shard; unused with kFlagIdxAllRows).
constexpr int kTkBytes = kTok * kTopK * 4;
// GLM52_ROUTER_LG, the router logits by linearity: W_r RMSNorm_post(h') x rms = W_r diag(gamma) (res_in + sum_r W_OV,r
// O_r), so each rank's 16 logit units (experts [16 u, +16)) multiply [O_r | res_in's columns [384 r, +384)] by the
// host-made M_r = W_r diag(gamma) [W_OV,r | I] (K = 2048 + 384) as soon as O is merged, and publish the FP32 partials
// through the multicast like the hops: [16 ranks][16 units][8 tokens][16 experts] per launch copy, after the top-k
// exchange's copies in the same workspace. The same unit then sums its slot column u of the 16 ranks in rank order into
// kArLg and empties it (single reader); the routing scales by 1 / rms as before.
constexpr int kRouterLg = GLM52_ROUTER_LG;
constexpr int kLgUnits = 16;
constexpr int kLgResChunks = kHidden / kRanks / 64;        // 6: the rank's 384 residual columns
constexpr int kLgChunks = kOChunks + kLgResChunks;         // 38 K chunks of 64 per unit (host layout)
constexpr int kLgSlotBytes = kTok * 16 * 4;                // 512
constexpr int kLgBytes = kRanks * kLgUnits * kLgSlotBytes; // 131,072 per launch copy
constexpr int kLgOprojFirst = 148;                         // logit unit u rides on o_proj unit kLgOprojFirst + u (host)

// Arena of multi-reader activation buffers, per launch copy (byte offsets).
constexpr int kArY = 0;      // BF16 [8][2624]      fused-A output
constexpr int kArQ8 = 41984; // E4M3 [8][2304]      Q
constexpr int kArKv = 60416; // E4M3 [8][576]       this step's KV rows
constexpr int kArO = 65024;  // BF16 [8][4][512]    merged attention output
// Leader layers (the DSA indexer) in the 16 KiB v_b no longer needs (folded into o_proj):
constexpr int kArIdxKw = 97792;     // BF16 [8][160]       fused-A rows 2624-2783: indexer K (128), head weights (32)
constexpr int kArIdxAmax = 100352;  // FP32 [8][32][4]     indexer Q: |max| of each head's four 32-row quarters
constexpr int kArIdxKNew = 104448;  // E4M3 [8][128]       this step's indexer keys
constexpr int kArIdxKNewS = 105472; // FP32 [8]           their scales
constexpr int kArPStats = 105504;   // FP32 [8][16][4][2]  kFlagSliceMerge: the attention blocks' (max, sum) per head
static_assert(kArIdxKNewS + 32 <= kArPStats && kArPStats + 8 * 16 * 4 * 8 <= 114176, "arena layout");
constexpr int kArLg = 105600;      // FP32 [8][256]       GLM52_ROUTER_LG: the router logits (16 ranks' partials summed,
                                   // before the 1 / rms scale); shares kArPStats' space (no kFlagSliceMerge with it)
static_assert(kArIdxKNewS + 32 <= kArLg && kArLg % 128 == 0 && kArLg + 8 * 256 * 4 <= 114176, "arena layout");
constexpr int kArH = 114176;       // BF16 [8][6144]      h (attention all-reduce, RMSNorm)
constexpr int kArHfp4 = 212480;    // E2M1 [8][3072]      NVFP4(h)
constexpr int kArHsf = 237056;     // E4M3 [8][384]       its block scales (right after the E2M1 values)
constexpr int kArSsq1 = 240128;    // FP32 [8][4]         sum-of-squares partials, attention all-reduce
constexpr int kArSsq2 = 240256;    // FP32 [8][4]         sum-of-squares partials, MoE all-reduce
constexpr int kArLogit = 240384;   // FP32 [4][8][256]    router logits per K quarter
constexpr int kArRoute = 273152;   // int32 [256]         routing (E, slot experts, slot of (t, k), weights)
constexpr int kArSh = 274176;      // FP32 [4][8][32][8]  shared gate_up partials per K quarter
constexpr int kArP1 = 306944;      // FP32 [212][2][128][8] FC1 partials (CTA, run, row, token); kFlagFc2Direct uses
                                   // [212][2][8][4][16][2] (CTA, run, token, feature block, feature, up/gate)
constexpr int kArIdxQ8 = 2043648;  // E4M3 [8][32][128]   indexer Q (RoPE'd, per-head UE8M0 scale), leader layers
constexpr int kArIdxW = 2076416;   // FP32 [8][32]        indexer head weights x q_scale x 128^-0.5 x 32^-0.5
constexpr int kArIdxBnd = 2077440; // u32 [215][8]        per CTA and row: a lower bound of its 10th-largest key; T, T2,
                                   // E (or u64 [8][212]: per row and CTA, its lanes' 9th and 10th maxima)
constexpr int kArIdxTopk = 2109184; // int32 [8][2048]    the leader's top-k again (its attention polls these words)
// BF16 [8][32][128]: the leader's indexer Q before E4M3 under the row shard (its IdxSel CTAs quantize it; the row shard
// leaves the top-k's arena copy unused: its attention reads the exchange).
constexpr int kArIdxQb = kArIdxTopk;
static_assert(8 * 32 * 128 * 2 == 8 * 2048 * 4, "the BF16 indexer Q in the top-k's arena copy");
constexpr int kArX = 2174720; // BF16 [8][6144]      this launch's x, written by the previous launch's H2
static_assert(kArIdxBnd + (212 + 3) * 8 * 4 <= kArIdxTopk, "arena layout"); // the CTAs' words, then the rows' T, T2, E
static_assert(kArIdxBnd + 212 * 8 * 8 <= kArIdxTopk, "arena layout");       // or per (row, CTA) one 64-bit word
static_assert(kArIdxTopk + 8 * 2048 * 4 <= kArX, "arena layout");
constexpr int kORepBytes = kTok * 4 * 512 * 2;
constexpr int kArORep = kArX + kTok * kHidden * 2;
constexpr int kArHp = kArORep + 3 * kORepBytes; // BF16 [8][6144] kFlagHPrime: h' = res x gamma, before the RMS scale
constexpr int kArXSsq = kArHp + kTok * kHidden * 2; // FP32 [8][4] kFlagXPrime: this launch's x' sum-of-squares partials
constexpr int kArenaBytes = kArXSsq + kTok * 4 * 4;

// Clear units (16 KiB): the next launch's hop buffers (2 x kHopBytes) and top-k exchange, then its arena copy.
constexpr int kClearUnitBytes = 16384;
constexpr int kHopClearUnits = (2 * kHopBytes + kTkBytes) / kClearUnitBytes;
static_assert((2 * kHopBytes) % kClearUnitBytes == 0 && kTkBytes % kClearUnitBytes == 0, "clear units");
constexpr int kArenaClearUnits = (kArenaBytes + kClearUnitBytes - 1) / kClearUnitBytes;
constexpr int kClearUnits = kHopClearUnits + kArenaClearUnits;
// Every clear unit signals kCClear (the owners and the next launch's fused-A wait for kCountedClear of them): that
// fused-A stores its rows (kArY; kArIdxKw in leaders) before its grid-dependency wait, so its wait for the count is
// what orders those clears first, as it does the clears of x it polls.
constexpr int kCountedClearUnits = kHopClearUnits + kArenaClearUnits;

// ----------------------------------------------------------------------------------------------------------------
// Leader layers: the DSA indexer (32 heads x 128, keys in a paged FP8 cache) and its exact top-2048
// ----------------------------------------------------------------------------------------------------------------
constexpr int kIdxHeads = 32;
constexpr int kIdxDim = 128;
constexpr int kIdxRows = kIdxHeads * kIdxDim; // 4096 rows of the indexer wq_b
constexpr int kIdxQUnitRows = 32;             // an IdxQ unit: two 16-row tiles, a quarter of one head
constexpr int kIdxQChunks = 32;               // K = 2048 in 64-wide chunks
constexpr int kFaTilesLeader = 174;           // fused-A with the indexer rows: 2784 = 174 x 16
constexpr int kIdxKwCols = 160;
static_assert(kFaTilesLeader * 16 == kFaRows + kIdxKwCols, "leader fused-A tiles");
constexpr int kIdxPageKeys = 64;
constexpr int kIdxPageBytes = kIdxPageKeys * (kIdxDim + 4); // 8448: 64 keys x 128 B, then their 64 FP32 scales
static_assert(kIdxPageBytes % 128 == 0 && kIdxRows == 128 * kIdxQUnitRows, "indexer layout: 128 IdxQ units");
constexpr int kIdxMaxPages = 16;                            // pages per IdxSel round (its K region's)
constexpr int kSelBatchPages = kIdxMaxPages / 2;            // the row shard's page batches (half the K region each)
#if GLM52_TOPK_UNION
// Reserve one page for speculative lookahead beyond a 1M-token logical context.
constexpr int kCandList = (1 << 20) + kIdxPageKeys;
constexpr int kCandCap = kCandList + 8192; // per row: list A, then the union's 13,568 words
#else
constexpr int kCandList = 1 << 18;      // candidates per list (list A, then list B): any row's keys
constexpr int kCandCap = 2 * kCandList; // top-k candidates per row: list A, then list B
#endif
static_assert(kCtas * kIdxMaxPages * kIdxPageKeys + 8 <= kCandList, "a list holds every key of a row");
constexpr int kCandSmem = 16384;             // the candidates the shared-memory selection copies (A, then B)
constexpr int kCandSmemList = kCandSmem / 2; // per list; a list taken past it is selected in global memory
constexpr uint32_t kTightGap = 3u << 17;     // T2 = E - this (order-key units: 2.3-4.7 % of the value)
constexpr uint32_t kGallop = 1u << 17;       // the selector's first step from E (0.8-1.6 %)
constexpr int kSelSmall = 512;               // a selection window this small is ranked directly
constexpr int kSelWin = 64;                  // idxSelectRow: the search's window ranked directly
constexpr int kHistBins = 64;                // per row: list A's histogram, bins from T2
constexpr int kHistShift = 14;               // its bins' width in order-key units (0.1-0.2 % of the value)
constexpr uint32_t kHistSpan = static_cast<uint32_t>(kHistBins) << kHistShift; // [T2, T2 + this): ~E - 3 % .. E + 5 %
constexpr int kIdxQUnits = 128;              // IdxQ units = quarter heads, on CTAs [0, 128)
constexpr int kIdxSelBase = 204;             // the CTA selecting row r: kIdxSelBase + r
constexpr int kH1Cta0 = 180;                 // the H1 / H2 units' CTAs: 180-211 (host build_roles)
constexpr float kIdxWeightScale = 0.015625f; // 128^-0.5 * 32^-0.5
// IdxSel's shared memory (from its offset, 1024-aligned): the K pages (SWIZZLE_128B atoms; once a 128-key tile's
// MMAs are done, its first 4 KiB hold the tile's logits, FP32 [8 tokens][128 keys]), Q in E4M3 ([256 (token, head)
// rows][128] SWIZZLE_128B), the pages' key scales, the head weights. The top-k histograms use the reduction area.
constexpr int kSelK = 0;
constexpr int kSelQ8 = kSelK + kIdxMaxPages * 8192; // 32 KiB
constexpr int kSelKs = kSelQ8 + 256 * 128;          // FP32 [16][64]
constexpr int kSelW = kSelKs + kIdxMaxPages * 256;  // FP32 [8][32]
constexpr int kSelBytes = kSelW + 1024;
constexpr int kSelLaneMax
    = 8192; // row shard: the tile epilogues' largest keys, u32 [8 warps][32 lanes], reduction area
constexpr int kSelLaneMax2 = 9216; // row shard: their second largest keys, u32 [8 warps][32 lanes]
static_assert(kSelQ8 % 1024 == 0 && kSelBytes <= 200704, "IdxSel shared memory");
// The row shard: the row's logits, FP32 [local key] (key 64 j + r of the CTA's page j), past the rounds' region; IdxSel
// takes kSelTotal bytes (the host's unit size).
constexpr int kSelLog = kSelBytes;
constexpr int kSelTotal = 200704;
constexpr int kIdxMaxPagesRs = (kSelTotal - kSelLog) / (4 * 2 * kIdxPageKeys) * 2; // a CTA's pages, whole tiles: 124
// The longest context the indexer takes (its keys: the cached ones and the step's kTok). kFlagPageBalance (always on)
// puts kIdxBalDeficit pages fewer on each IdxQ CTA than on each CTA without an IdxQ unit (idxPageOf), at most
// kIdxMaxPagesRs on a CTA; a candidate list holds every key of a row.
constexpr int kIdxBalDeficit = 7;
constexpr int kIdxBalCap = (kCtas - kIdxQUnits) * kIdxBalDeficit + kCtas * (kIdxMaxPagesRs - kIdxBalDeficit);
constexpr int kIdxMaxKeys = kIdxBalCap * kIdxPageKeys < kCandList ? kIdxBalCap * kIdxPageKeys : kCandList;
// RS-H's lattice (rsBin): bins of 1/128 binade (the order key's top 16 bits) for |logit| in [2^-13, 2^10) on each sign
// (kRsW bins per sign), each sign's two tails, and +0.0 / -0.0 exactly, in key order: [0] logit <= -2^10, [1, kRsW]
// the negative window, [kRsW + 1] (-2^-13, -0.0), [kRsW + 2] -0.0, [kRsW + 3] +0.0, [kRsW + 4] (+0.0, 2^-13),
// [kRsW + 5, 2 kRsW + 4] the positive window, [2 kRsW + 5] >= 2^10. The window covers every 2048th logit seen so far
// (synthetic rows 0.0028-0.83, served rows -24..-100); a cut in a tail bin is still exact (the global selection).
// The global histogram's bin b sits at word 4 (b / kRsPer + kThreads * (b % kRsPer / 4)) + b % 4: thread t's six
// coalesced 16-byte polls hold its bins [kRsPer t, + kRsPer) in order.
constexpr int kRsW = 23 * 128;
constexpr uint32_t kRsN0 = 0x3B80u; // the negative window's top-16 keys [kRsN0, kRsN0 + kRsW)
constexpr uint32_t kRsP0 = 0xB900u; // the positive window's [kRsP0, kRsP0 + kRsW)
constexpr uint32_t kRsBinNZ = kRsW + 2;
constexpr uint32_t kRsBinPZ = kRsW + 3;
constexpr int kRsPer = 24;
constexpr int kRsBins = kRsPer * kThreads;
static_assert(2 * kRsW + 6 <= kRsBins && kRsPer % 4 == 0, "RS-H lattice");
static_assert(kRsN0 + kRsW == 0x4700u && kRsP0 + kRsW == 0xC480u, "RS-H window: |logit| in [2^-13, 2^10)");
// Each CTA adds to the global histogram only its bins from the one holding its kRsLocalK-th largest key up (its keys
// counted in shared memory during the tile epilogues): ~27k adds per row whatever the context. Per-key adds of every
// key backed up on the hot bins (one-GPU test: 12 us at 335k, 32 us at 900k on served-like rows; same-address adds
// serialize at ~7-9 ns). Served rows hold at most ~69 of the row's top 2048 in one CTA (one hot page).
constexpr uint32_t kRsLocalK = 128;
constexpr uint32_t kRsPairs = 512; // a cut bin of at most this many keys is ranked by the row CTA in shared memory
constexpr uint32_t kRsDirect = 1;  // descriptor modes: the cut bin's keys go straight into the row (taken whole, or
constexpr uint32_t kRsRank = 2;    // equal keys), ranked by the row CTA, or selected exactly in global memory;
constexpr uint32_t kRsGlobal = 3;  // kRsMore: the cut lies below a CTA's added bins, every CTA adds the rest first
constexpr uint32_t kRsMore = 4;

// Stage codes of a work unit (role entry = stage << 16 | unit).
enum Stage : int
{
    kStNone = 0,
    kStFusedA = 1,
    kStQb = 2,
    kStKvAppend = 3,
    kStAttn = 4,
    kStMerge = 5,
    kStOproj = 7,
    kStClear = 8,
    kStH1 = 9,
    kStRouterGemv = 10,
    kStSharedGu = 12,
    kStMoe = 13,
    kStOwner = 14,
    kStH2 = 15,
    kStOwnerShared = 16,
    kStIdxK = 17,    // leader: this step's indexer keys (LayerNorm, RoPE, E4M3) into the cache and the arena
    kStIdxQ = 18,    // leader: 32 rows of the indexer Q (q_a norm folded, RoPE), BF16, and their |max| per head quarter
    kStIdxSel = 19,  // leader: the logits of the CTA's key pages, then the exact top-2048 with every other CTA
    kStOprojLg = 20, // GLM52_ROUTER_LG: o_proj unit u (CTAs 148-163), then on the same O this rank's partial router
                     // logits of experts [16 (u - 148), +16) with its residual columns, multicast, and their ranks' sum
};

// Counters (u32, the next copy zeroed in every launch's prologue): edges that cannot carry their own flag.
enum Counter : int
{
    kCClear = 0,   // hop-buffer clear units done (ordered before the MoE all-reduce's publish)
    kCExit = 1,
    kCMoeDone = 2, // CTAs past their MoE unit (kFlagHoldExit)
    kCXReady = 3,  // (unused)
    // Leader: the top-k rows done (polled by every CTA past its part of the top-k: a 128-byte line of its own, away
    // from the candidate counts the selectors poll).
    kCIdxDone = 16,
    // Leader top-k: per row the bound T (its complement, stored by the row's selector for the record), and per row a
    // 64-bit pair: the CTAs that have taken their candidate slots and list A's candidates ((CTAs << 24) | count), then
    // list B's candidates (low 18 bits) and, with the row shard, list A's keys in its histogram (high 14 bits).
    kCIdxT = 32,
    kCIdxCand = kCIdxT + kTok,
    kCIdxTies = kCIdxCand + 2 * kTok, // per row: keys equal to T2 taken from list A's pool (past each CTA's free ones)
    kCIdxTiesB = kCIdxTies + kTok,    // per row: keys equal to T taken from list B's pool
    kCIdxWritten = kCIdxTiesB + kTok, // CTAs done storing candidates (a release after the join; the selector acquires)
    kCIdxHist = 128, // per row: list A's histogram (kHistBins bins of keys - T2, from T2); the row shard uses its row's
    // Optional cooperative bin emission, one row per rank. This tail is included in the C ABI counter count and
    // the existing next-copy clear; no word is reused from a live histogram or another row's candidate metadata.
    kCIdxParState = kCIdxHist + kTok * kHistBins, // 0: pending, 1: original selector, 2: descriptor published
    kCIdxParCount,                                // sure positions in low 16 bits, boundary entries in high 16
    kCIdxParDone,                                 // eight CTAs' release-published completed scans
    kCIdxParN,
    kCIdxParWin,                                  // uint4: the histogram window
    kCIdxParKey = kCIdxParWin + 4,
    kCIdxParPos = kCIdxParKey + kSelWin,
    kNumCounters = (kCIdxParPos + kSelWin + 31) & ~31, // preserve the original 128-byte alignment of every copy
};

static_assert(kCIdxWritten < kCIdxHist, "counters");

// Parameter block (int64 words, filled by the host).
enum Param : int
{
    kPMagic,
    kPKvBase,
    kPStrideRows,
    kPLayerRows,
    kPMaxBlocks,
    kPRope,
    kPWFa,
    kPWQb,
    kPWLg, // GLM52_ROUTER_LG: W_r diag(gamma_post) [W_OV of this rank | this rank's 384 residual columns], packed
           // BF16 [16 tiles][38 chunks][2048 B] (the slot of the folded-away v_b weights)
    kPWO,
    kPWR,
    kPWGu,
    kPWD,
    kPGammaKv,
    kPGammaPost,
    kPGammaNext,
    kPRouterBias,
    kPW13,
    kPS13,
    kPW2,
    kPS2,
    kPAlpha1,
    kPAlpha2,
    kPArena,
    kPPartO,
    kPPartStats,
    kPFc2out,
    kPH1Mc,
    kPH1Uc,
    kPH2Mc,
    kPH2Uc,
    kPHopBytes,
    kPRank,
    kPCounters,
    kPState,
    kPErr,
    kPRoles,
    kPRoleSmem,
    kPRoleIssue,
    kPMaxRoles,
    kPTimeoutNs,
    kPStamps,
    kPFlags,
    kPCopy,      // this launch's copy of the arena, hop buffers and counters (launch index mod kPNumCopies)
    kPNumCopies, // copies in the rotation (the launches per replay are a multiple of it)
    kPLeader,    // 1: a leader layer (the role table has the indexer stages; the attention waits for the own top-k)
    kPIdxWq,     // indexer wq_b with the q_a norm folded in, packed BF16 [256 tiles][32 chunks][2048 B]
    kPIdxKnW,    // indexer K LayerNorm weight, FP32 [128]
    kPIdxKnB,    // its bias, FP32 [128]
    kPIdxRope,   // indexer cos/sin, FP32 [positions][64] = [cos 0-31 | sin 0-31]
    kPIdxPool,   // this layer's indexer K cache: block b (kIdxPageBytes) at kPIdxPool + b * kPIdxBlockStride
    kPIdxBlockStride, // bytes between the layer's blocks (the pool interleaves the indexer layers), a multiple of 128
    kPIdxTmap,        // the pool as a 2-D tensor map [rows of 128 B], SWIZZLE_128B boxes of 64 rows (CUtensorMap)
    kPIdxBlocks,      // the request's indexer K cache block table, int32
    kPIdxCand,        // top-k boundary candidates, uint2 (key, position) [copies][8][kCandCap]
    kPTopkOut,        // the leader's top-k, int32 [8][2048] (the next plain layers' input)
    kPIdxDebug,       // check mode: the leader's logits, FP32 [8][kPIdxDebugLd] (0: off)
    kPIdxDebugLd,
    kPIdxDebug2, // check mode: CTA 0's first key page (64 x 128 B, unswizzled), its Q8 (256 x 128 B), its tile 0's D
                 // (FP32 [128 keys][256]); 0: off
    kPTkUc,      // the top-k exchange (kTkBytes per launch copy): this GPU's copy, and its multicast address
    kPTkMc,
    kPNumParams,
};

// Variant bits (kPFlags): wait on one canary word per producer unit before the bulk poll of an activation many CTAs
// read at once (the bulk poll of every waiting CTA otherwise re-reads the whole region every round trip).
enum Flag : int64_t
{
    kFlagFc2Masked = 1ll << 38, // skip arithmetic/shuffles for tokens outside the uniform expert mask
    kFlagMergeDry = 1ll << 39,  // merge: its code once on zeros (no stores) before the poll, so the real pass runs warm
    kFlagHopSelfClear
    = 1ll << 37,                // (always on) the H1 units empty the 16 hop-1 slots they read and the previous hop-2's
                                // of their columns (single readers); the host schedules no clear units for the hop
                                // buffers (each signalled with a release: ~1 us, and their stores after o_proj's)
    kFlagXFrag = 1ll << 40,     // (always on) x' in the arena in fused-A's fragment order: per 64-column chunk two
                                // 512-byte halves, lane l's 16 bytes of half h at 16 l (each poll instruction reads 4
                                // whole lines once, not half of 8 lines twice)
    kFlagPageBalance = 1ll << 55, // leader: more key pages on the CTAs without an IdxQ unit (idxPageOf)
    kFlagAttnIlp = 1ll << 56,     // attention: four Q.K accumulator chains, two P.V output blocks per iteration
    kFlagSliceMerge = 1ll << 58,  // attention unit (t, b) merges dims [32 b, +32) of row t itself (attnSliceMerge): no
                                  // merge units (the host layout drops them)
    kFlagXPrime
    = 1ll << 51, // (always on) H2 publishes the next launch's x' = res' x gamma_next (kArX) and its per-unit sum of
                 // squares (kArXSsq) before its exchange; that launch's fused-A (early x) scales its
                 // outputs by 1 / rms (every output row is linear in x): the exchange leaves the x path
    kFlagWarmInHop = 1ll << 47, // (always on) H1 CTAs (180-211): the MoE warm-up (kFlagWarmMoe) in H1's hop-1 wait, not
                                // between H1 and the routing (those CTAs set the MoE tail)
    kFlagSelFast = 1ll << 42,   // (always on) leader top-k (row shard, kFlagIdxParallel): every selector CTA takes
                                // the counts and the window itself, the boundary keys are polled 64-bit slots (no
                                // release counters)
    kFlagHopClearLate
    = 1ll << 32, // (always on) the H1 units' hop-slot self-clear at the start of the same CTA's H2 unit (it waits for
                 // hop 2 there), not at H1's end (~1.4 of H1's 1.9 us, before those CTAs' MoE)
    kFlagLandProbe = 1ll << 41,  // stamped runs: sub-stamp 128 when the setup's weight requests went out, 129 when the
                                 // first fused-A / q_b' unit's weights landed (thread 0 waits for them before its poll)
    kFlagFirstEarly = 1ll << 50, // (always on) thread 0 requests the first unit's weights (setup-issued ones) first,
                                 // from the tables in one round trip, before the parameter block and the setup (the
                                 // CTAs that start last, on the SMs the previous layer's H2 units held, get q_b''s
                                 // weights ~0.6-0.9 us after its input otherwise)
    kFlagHPrime = 1ll << 48, // (always on) H1 publishes h' = res x gamma (kArHp) before its sum-of-squares exchange;
                             // the router and the shared gate_up run on it, the routing and the shared expert's combine
                             // scale by 1 / rms (from kArSsq1): the exchange leaves the routing path
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
    kFlagSparseP1 = 1ll << 54, // direct P1 only: publish/gather routed tokens, zero inactive FC2 operands
#endif
    kFlagCanaryMoe = 1,        // MoE CTAs: the router units' logits, then the routing, then NVFP4(h)
    kFlagCanaryOproj = 2,      // o_proj: the v_b units
    kFlagCanaryH = 4,          // router GEMV and shared gate_up: the H1 units of the K quarter
    kFlagCanaryOwner = 8,      // owner: the shared gate_up units
    kFlagHoldExit = 16,    // CTAs exit only once every CTA's MoE unit is done (the next kernel's prologue loads then
                           // stay out of the MoE weight stream)
    kFlagNextChained = 32, // (set by the host per launch) the next launch is this kernel's and reads x' from the arena
                           // (kFlagEarlyX): H2 leaves out the normalized x and its sum-of-squares exchange
    kFlagEvictFirst = 64,  // weight bulk copies with the L2 evict-first policy
    kFlagWarmMoe = 128,    // MoE: one FC1 and one FC2 MMA batch on stale operands before the routing (the MMA code
                           // paths' first run otherwise lands on the expert pieces' critical path)
    kFlagOsInMoe = 512,    // the shared-expert unit runs inside the MoE stage after the first expert requests (its
                           // own unit before the MoE stage returns at once)
    kFlagNoStamps = 2048,  // no timing stamps (the timed comparison with the stock layer)
    kFlagMoeL2 = 4096, // MoE: the CTA's pieces beyond the ring are prefetched into L2 right after the ring's requests
                       // (they then land from L2 when a ring slot frees, not from HBM a round trip later)
    kFlagMoeL2Normal = 8192, // those prefetches with the default L2 policy (else evict-first with kFlagEvictFirst)
    kFlagL2Deferred = 16384, // leader, at entry: the CTA's indexer Q weights (requested once fused-A frees their slot)
    kFlagEarlyX = 32768,     // the previous launch is this kernel: fused-A polls x from the arena (its H2 wrote it
                             // there), before the grid-dependency wait
    kFlagCheckX = 65536, // with kFlagEarlyX: CTA 0 compares that x with the kernel argument after the wait (trap 98)
    kFlagOsPrefetch = 131072, // MoE CTAs with a shared-expert unit: its down weights into L2 before the expert requests
    kFlagIdxQEarly = 262144,  // leader fused-A: its IdxQ unit's weights into L2 once its own landed, and into the slot
                              // as soon as its MMAs are done (not after the unit)
    kFlagTopkOneList = 524288,      // leader top-k without the bounds' estimate: one list (keys >= T), plain bisection
    kFlagMergeClearFirst = 1048576, // merge units clear the attention partials before their output (else after it)
    kFlagOwnerClearFirst = 2097152, // owner units clear the routed rows they read before their shard (else after it)
    kFlagIdxAllRows = 4194304, // leader: every rank scores and selects all 8 rows (else the row shard: rank r scores
                               // only row r / 2 (logits N = 32), the even rank of the pair publishes its selection to
                               // every rank through the top-k exchange, and the attention takes all 8 rows from there)
    kFlagTopkNoHist = 8388608, // the selectors without list A's histogram: the search starts from the whole list, not
                               // from the bin of the row's need0-th key
    kFlagTopkNoBin
    = 16777216, // with the histogram's bin window: the packed search from it (else one pass, idxSelectBin)
    kFlagTopkTCta = 33554432,    // T, T2 and E from one CTA (the lanes' top four, ten warp rounds, two hops; else every
                                 // CTA takes them from the words of the lanes' maxima itself)
    kFlagIdxTilePairs
    = 536870912,                 // row shard: the logits' tiles two at a time in 256-column TMEM buffers (as all rows)
    kFlagBoundLoop = 1073741824, // row shard: the bound's lane maxima from a pass over the logits (else from the tile
                                 // epilogues)
    kFlagBinCopy = 268435456,    // the bin selector reads list A from its bulk copy (else straight from L2 in its pass)
    kFlagMoeAllFirst = 67108864, // MoE: the first three pieces requested at once (else piece 0 first, pieces 1-2 after
                                 // the FC2 groups' inputs)
    kFlagBinNoPreload = 256, // the direct bin pass loads its first pairs itself (else the caller, before the window)
    kFlagBinVolatile = 1024, // the direct bin pass reads the list with volatile loads (else ld.global.cg)
    kFlagRstdShared = 1ll << 33, // (always on) MoE (kFlagHPrime): the routing polls the tokens' sum-of-squares partials
                                 // with the logits and leaves each token's 1 / rms in the MoE list, where the shared
                                 // expert inside the MoE stage reads it (else four L2 round trips per thread there)
    kFlagCandWarps
    = 1ll << 36, // (always on) leader (row shard): the CTA's candidates counted and written by all 8 warps (warp w: its
                 // lanes' keys m in [w nPer / 8, (w + 1) nPer / 8)), one slot atomic per CTA
    kFlagUnionLean
    = 1ll << 31, // (always on) leader union T: on CTA 127 (an IdxQ CTA, the fewest key pages) while it holds fewer
                 // than CTAs 128-211's 14, else on CTA 204; the union's slots zeroed after T's publish
    kFlagUnionSide = 1ll << 27, // (always on) leader union T: pass 1 over the side of zero that holds the need-th
                                // largest key (served rows: a few positive logits, the 2048th negative; the window
                                // below the largest key held too few and a full-range fallback pass ran)
    kFlagHop1Lines = 1ll << 34, // (always on) hop 1's buffer as [rank][o_proj unit][token][32 columns]: an o_proj
                                // unit's 8 tokens are one 512 B block (full lines in the multicast; else 8 half-lines)
    kFlagPageMap2 = 1ll << 35,  // leader: the key pages by the CTA groups' IdxSel starts, for every context length
                                // (idxPageMap; else kFlagPageBalance's map, tuned at 147k)
    kFlagFc2Direct = 1ll << 46, // FC1 partials as [token][block][feature][up/gate]; FC2 sums its own features in
                                // registers, without the shared-memory partial transpose
    kFlagIdxParallel = 1ll << 49, // even row shards: CTAs 204..211 emit a small-bin selection cooperatively
};

constexpr int64_t kMagic = 0x474c4d3532504c32LL; // "GLM52PL2"
constexpr int kMaxRoles = 16;
// Stamps per CTA (u64 %globaltimer): [role][start, inputs in (thread 0, last poll), weights in, end], then entry, PDL
// wait return, exit, MoE weights requested, then kSubStamps points inside q_b' (0-4), the attention (5-11) and the MoE
// (12-15, piece p landed / consumed at 16 + 2p / 17 + 2p, the FC1 partials of the CTA's item k in at 32 + k; FC2 group
// g < 2 polling its partials from 73 + g to 75 + g; piece p < 8 waited for from 80 + p); the SM clock (%clock64) of
// sub-stamps 40-71 at 96 + k - 40 (cycles against ns: the SM's speed in a step).
constexpr int kSubStamps = 136;
constexpr int kStampsPerCta = kMaxRoles * 4 + 4 + kSubStamps;

// Shared memory: unit regions [0, kSmemMain), then fixed areas. Past the 227 KB opt-in limit (VR200: up to
// 334,848 B per block) the kernel runs in the oversized shared-memory mode (8 KB of L1 left).
#ifndef GLM52_SMEM_MAIN
#define GLM52_SMEM_MAIN 200704
#endif
constexpr int kSmemMain = GLM52_SMEM_MAIN;
constexpr int kSmemRed = kSmemMain;
constexpr int kSmemRedBytes = 16384;
constexpr int kSmemRes = kSmemRed + kSmemRedBytes;
constexpr int kSmemResBytes = 3072;
constexpr int kSmemSmall = kSmemRes + kSmemResBytes;
#if GLM52_SETUP_MODE == 0
constexpr int kSmemSmallBytes = 2688;
#else
constexpr int kSmemSmallBytes = 3072;
#endif
constexpr int kSmemBytes = kSmemSmall + kSmemSmallBytes;

// Small area layout (bytes from kSmemSmall).
constexpr int kSmallBars = 0; // kMaxRoles unit barriers + 8 attention + kRingSlots MoE ring + MoE MMAs +
                              // leader IdxSel: its key pages, its two TMEM buffers' MMAs, the row shard's odd
                              // page batches
constexpr int kBarIdxK = kMaxRoles + 8 + kRingSlots + 1;
constexpr int kBarIdxK2 = kBarIdxK + 3;
constexpr int kNumBars = kBarIdxK2 + 1;
static_assert(kNumBars * 8 <= 256, "barriers");
constexpr int kSmallRoles = 256;   // 3 x kMaxRoles int32
constexpr int kSmallScalars = 512; // misc (128 floats)
constexpr int kSmallMoe = 1024;    // MoE work list (int32)
constexpr int kSmallParams = 2048; // the parameter block (int64), copied at entry
constexpr int kSmallCtx = 2560;    // the kernel context (struct Ctx)
static_assert(kSmallParams + kPNumParams * 8 <= kSmallCtx, "parameter block does not fit");
#if GLM52_SETUP_MODE == 1
constexpr int kSmallWeights = 2688;
static_assert(kSmallWeights + kMaxRoles * 3 * 8 <= kSmemSmallBytes, "weight table row does not fit");
#endif

// MoE work list (int32 words at kSmallMoe): E, piece count, the CTA's FC1 and FC2 piece ranges, the slot of every
// (token, k), the slot experts, the routing weights, the slots' token masks (bytes).
constexpr int kMoeE = 0;
constexpr int kMoePieces = 1;
constexpr int kMoeF1A = 2;
constexpr int kMoeF1B = 3;
constexpr int kMoeF2A = 4;
constexpr int kMoeF2B = 5;
constexpr int kMoeWarmed = 6;      // nonzero: an earlier unit of the CTA ran the MoE warm-up (zeroed at entry)
constexpr int kMoeRcp = 8;         // [8] float bits: the tokens' 1 / rms (kFlagRstdShared)
constexpr int kMoeSlotOfTk = 32;   // [64]
constexpr int kMoeSlotExpert = 96; // [64]
constexpr int kMoeWeight = 160;    // [64] float bits
constexpr int kMoeSlotMask = 224;  // [64] bytes

// Route block layout (int32 words) in the arena; the vectors are 16-byte aligned.
constexpr int kRouteE = 0;
constexpr int kRouteSlotExpert = 4; // [64]
constexpr int kRouteSlotOfTk = 68;  // [8][8]
constexpr int kRouteWeight = 132;   // [8][8] float bits

// The MoE cut's weights per CTA class (W, OS, SG, H1: see moeCutWeight), written with kFc2ListsPacked by gen_cut.py.
constexpr int kFc2ListsCut[4] = {6, 4, 5, 5};
#if GLM52_LISTS_MODE >= 1
__device__ __constant__ __align__(16) uint32_t kFc2ListsPacked[4160][3] = {
    {0x1105c168u, 0xa118722eu, 0x00000612u},
    {0x4a20ce48u, 0x32e156a0u, 0x00000d36u},
    {0x0882c0a8u, 0x40883817u, 0x000002e1u},
    {0x1f8e4688u, 0xd9509a46u, 0x00000612u},
    {0x3d9c8d28u, 0xd2511883u, 0x00000a04u},
    {0x58aad4c8u, 0x6b197eb8u, 0x00000d36u},
    {0x0581c068u, 0xd85c260fu, 0x000001e0u},
    {0x15098448u, 0xc8d8642eu, 0x000003d1u},
    {0x25918828u, 0xe960a84fu, 0x00000612u},
    {0x391b0cc8u, 0x4a0cfa78u, 0x000008f4u},
    {0x4e261288u, 0x5aa148a0u, 0x00000af5u},
    {0x5e2e1668u, 0x7b298cc1u, 0x00000d36u},
    {0x04014048u, 0xa0441c0bu, 0x00000170u},
    {0x0f870348u, 0x58a04a22u, 0x000002e1u},
    {0x1b8d0628u, 0x10fc7839u, 0x00000462u},
    {0x28134928u, 0xf16cae54u, 0x00000612u},
    {0x371a4ca8u, 0xf9ecec72u, 0x00000833u},
    {0x48231108u, 0xea692e94u, 0x00000a04u},
    {0x54a99468u, 0xa2c55cabu, 0x00000b85u},
    {0x612fd768u, 0x833592c6u, 0x00000d36u},
    {0x03010048u, 0x80341609u, 0x00000120u},
    {0x0c85c288u, 0x10803c1bu, 0x00000251u},
    {0x160a44e8u, 0xa8c8602eu, 0x00000371u},
    {0x1f0f0728u, 0x49188640u, 0x000004b2u},
    {0x2a1449c8u, 0xf970b257u, 0x00000612u},
    {0x3599cc88u, 0xc9d4e46eu, 0x000007c3u},
    {0x43a0d008u, 0xa2451c8au, 0x00000974u},
    {0x4f26d328u, 0x3a9144a0u, 0x00000a95u},
    {0x582b9568u, 0xd2e16ab2u, 0x00000bd5u},
    {0x6330d808u, 0x8b3996c8u, 0x00000d36u},
    {0x0280c028u, 0x682c1207u, 0x000000f0u},
    {0x0a84c228u, 0xe06c3217u, 0x000001e0u},
    {0x12088408u, 0x60a85026u, 0x000002e1u},
    {0x1a0c8608u, 0xd8e46e36u, 0x000003d1u},
    {0x221047e8u, 0x692c9046u, 0x000004f2u},
    {0x2b150a28u, 0xf974b658u, 0x00000612u},
    {0x34998c68u, 0xa9c8de6cu, 0x00000783u},
    {0x401f4f68u, 0x62250c83u, 0x000008f4u},
    {0x4b251248u, 0xf2713498u, 0x00000a04u},
    {0x53291448u, 0x6aad52a8u, 0x00000af5u},
    {0x5aacd628u, 0xfaf174b8u, 0x00000c15u},
    {0x64319868u, 0x8b3d9acau, 0x00000d36u},
    {0x0200c028u, 0x58241006u, 0x000000d0u},
    {0x090401c8u, 0xc05c2a13u, 0x000001a0u},
    {0x0f874388u, 0x30904421u, 0x00000271u},
    {0x160ac528u, 0x98c45e2eu, 0x00000341u},
    {0x1c8e06c8u, 0x00f87a3bu, 0x00000422u},
    {0x24118888u, 0x8138984au, 0x00000522u},
    {0x2c158a88u, 0x0178b85au, 0x00000613u},
    {0x34194c68u, 0x91bcda6au, 0x00000743u},
    {0x3e1e4ee8u, 0x320d027eu, 0x00000884u},
    {0x48235168u, 0xc2592892u, 0x00000994u},
    {0x4f275368u, 0x2a8d42a0u, 0x00000a65u},
    {0x55aa9508u, 0x92c15eadu, 0x00000b45u},
    {0x5d2e16c8u, 0x13017cbcu, 0x00000c46u},
    {0x653218c8u, 0x93419cccu, 0x00000d36u},
    {0x02008028u, 0x50200e05u, 0x000000b0u},
    {0x078381a8u, 0xa8502411u, 0x00000170u},
    {0x0d868308u, 0x087c3c1cu, 0x00000221u},
    {0x13894488u, 0x60ac5228u, 0x000002e1u},
    {0x190c45e8u, 0xc0dc6a34u, 0x00000391u},
    {0x1f0f0768u, 0x2108823fu, 0x00000462u},
    {0x25924908u, 0x91409e4du, 0x00000542u},
    {0x2c95cac8u, 0x0178b85bu, 0x00000613u},
    {0x33994c68u, 0x81b8d669u, 0x00000723u},
    {0x3c1d8e88u, 0x09fcfa7bu, 0x00000834u},
    {0x452210a8u, 0x92411c8cu, 0x00000944u},
    {0x4ca5d2c8u, 0xf275369au, 0x00000a04u},
    {0x5228d428u, 0x52a54ea6u, 0x00000ab5u},
    {0x582b95a8u, 0xb2d166b1u, 0x00000b85u},
    {0x5eaed728u, 0x230980bfu, 0x00000c66u},
    {0x65b258e8u, 0x93419ccdu, 0x00000d36u},
    {0x01808028u, 0x481c0c05u, 0x000000a0u},
    {0x07030168u, 0x9848200fu, 0x00000140u},
    {0x0c05c2a8u, 0xe8703619u, 0x000001e0u},
    {0x11084408u, 0x38984a24u, 0x00000291u},
    {0x168ac548u, 0x90c05e2eu, 0x00000331u},
    {0x1b8d8688u, 0xe0ec7238u, 0x000003d1u},
    {0x209007e8u, 0x39188843u, 0x00000492u},
    {0x27130968u, 0xa148a24fu, 0x00000552u},
    {0x2d160ae8u, 0x017cba5cu, 0x00000613u},
    {0x33190c68u, 0x71b0d468u, 0x00000703u},
    {0x3b1d0e48u, 0xe9f0f478u, 0x000007f3u},
    {0x42a0d028u, 0x6a2d1287u, 0x000008f4u},
    {0x4a24d228u, 0xca612e96u, 0x000009b4u},
    {0x4fa75388u, 0x228942a0u, 0x00000a55u},
    {0x54aa14c8u, 0x72b556aau, 0x00000af5u},
    {0x59ac9628u, 0xcae16cb5u, 0x00000bb5u},
    {0x5faf9788u, 0x331186c1u, 0x00000c76u},
    {0x66329928u, 0x93419ecdu, 0x00000d36u},
    {0x01808028u, 0x40180a04u, 0x00000090u},
    {0x0602c148u, 0x88401e0du, 0x00000120u},
    {0x0b050268u, 0xd0643017u, 0x000001b0u},
    {0x0f878388u, 0x18884220u, 0x00000251u},
    {0x1409c4c8u, 0x68b05429u, 0x000002e1u},
    {0x188c05e8u, 0xb0d46832u, 0x00000371u},
    {0x1d8e4708u, 0xf8f87a3cu, 0x00000401u},
    {0x2290c848u, 0x51248e46u, 0x000004b2u},
    {0x281389a8u, 0xa950a451u, 0x00000572u},
    {0x2d964b08u, 0x017cbc5cu, 0x00000613u},
    {0x33190c68u, 0x69acd267u, 0x000006e3u},
    {0x3a1c8e08u, 0xd9e4ee75u, 0x000007c3u},
    {0x41200fc8u, 0x421d0a83u, 0x000008a4u},
    {0x47a39188u, 0xaa512691u, 0x00000974u},
    {0x4d265308u, 0xfa79389bu, 0x00000a04u},
    {0x51a89428u, 0x429d4ca4u, 0x00000a95u},
    {0x56aad548u, 0x8ac15eaeu, 0x00000b25u},
    {0x5b2d5688u, 0xe2e972b8u, 0x00000bd5u},
    {0x60b017e8u, 0x3b1988c3u, 0x00000c86u},
    {0x66b2d948u, 0x93459eceu, 0x00000d36u},
    {0x01808028u, 0x38180a04u, 0x00000080u},
    {0x05828128u, 0x78381a0cu, 0x00000100u},
    {0x0a048228u, 0xc05c2c15u, 0x00000190u},
    {0x0e06c348u, 0x007c3c1du, 0x00000211u},
    {0x1208c448u, 0x48a04c25u, 0x000002a1u},
    {0x168b0568u, 0x88c05e2eu, 0x00000321u},
    {0x1a8d0668u, 0xc8e06e36u, 0x000003a1u},
    {0x1f0f4788u, 0x1104803fu, 0x00000432u},
    {0x239188a8u, 0x612c9448u, 0x000004d2u},
    {0x289409e8u, 0xb154a853u, 0x00000582u},
    {0x2d968b28u, 0x017cbc5du, 0x00000613u},
    {0x32990c68u, 0x61a8d067u, 0x000006d3u},
    {0x391c0de8u, 0xc1dcea73u, 0x000007a3u},
    {0x3f9f4f68u, 0x2a0d0480u, 0x00000864u},
    {0x45a29108u, 0x8a411c8du, 0x00000934u},
    {0x4b255288u, 0xda693097u, 0x000009c4u},
    {0x4fa793a8u, 0x1a8942a0u, 0x00000a45u},
    {0x53a994a8u, 0x5aa952a8u, 0x00000ac5u},
    {0x582bd5c8u, 0xa2cd64b1u, 0x00000b55u},
    {0x5cae16c8u, 0xf2f576bau, 0x00000bf5u},
    {0x61b09828u, 0x431d8cc4u, 0x00000c96u},
    {0x66b31968u, 0x9345a0ceu, 0x00000d36u},
    {0x01004008u, 0x30140803u, 0x00000070u},
    {0x05024108u, 0x7034180bu, 0x000000f0u},
    {0x09044208u, 0xb0542813u, 0x00000170u},
    {0x0d064307u, 0xf074381bu, 0x00000000u},
    {0x108803e8u, 0x28904622u, 0x00000261u},
    {0x148a04e8u, 0x68b0562au, 0x000002e1u},
    {0x188c05e8u, 0xa8d06632u, 0x00000361u},
    {0x1c8e06e7u, 0xe8f0763au, 0x00000001u},
    {0x200fc7c8u, 0x29108441u, 0x00000462u},
    {0x249208e8u, 0x7134984bu, 0x000004f2u},
    {0x29944a08u, 0xb958aa54u, 0x00000582u},
    {0x2e16cb47u, 0x097cbc5du, 0x00000003u},
    {0x3298cc48u, 0x59a4d066u, 0x000006c3u},
    {0x381bcdc8u, 0xb1d4e672u, 0x00000783u},
    {0x3e1ecf28u, 0x1200fe7du, 0x00000834u},
    {0x442190a8u, 0x6a311489u, 0x000008f4u},
    {0x49a49208u, 0xba592a94u, 0x00000984u},
    {0x4da69328u, 0xfa793a9cu, 0x00000a04u},
    {0x51a89428u, 0x3a994aa4u, 0x00000a85u},
    {0x55aa9527u, 0x7ab95aacu, 0x00000005u},
    {0x592c5608u, 0xbad568b3u, 0x00000b85u},
    {0x5dae9728u, 0x02fd7abcu, 0x00000c16u},
    {0x6230d848u, 0x4b218ec6u, 0x00000ca6u},
    {0x67335968u, 0x9345a0cfu, 0x00000d36u},
    {0x01004008u, 0x30140803u, 0x00000070u},
    {0x05024107u, 0x7034180bu, 0x00000000u},
    {0x088401e7u, 0xa8502612u, 0x00000000u},
    {0x0c05c2c7u, 0xe06c3419u, 0x00000000u},
    {0x0f8783a7u, 0x18884220u, 0x00000001u},
    {0x13094487u, 0x50a45027u, 0x00000001u},
    {0x168b0567u, 0x88c05e2eu, 0x00000001u},
    {0x1a0cc648u, 0xc0dc6c35u, 0x00000391u},
    {0x1d8e8738u, 0xf8f87a3cu, 0x00000401u},
    {0x21908828u, 0x39188a44u, 0x00000482u},
    {0x25928928u, 0x81389a4cu, 0x00000512u},
    {0x2a14ca48u, 0xc15cac55u, 0x00000592u},
    {0x2e16cb47u, 0x0980bc5du, 0x00000003u},
    {0x3298cc48u, 0x51a4ce66u, 0x000006b3u},
    {0x379b8da8u, 0xa9cce471u, 0x00000763u},
    {0x3d1e4ee8u, 0xf9f8fa7bu, 0x00000813u},
    {0x42a0d048u, 0x52250e86u, 0x000008b4u},
    {0x47a391a8u, 0xa24d2491u, 0x00000954u},
    {0x4c25d2c7u, 0xe26d3499u, 0x00000004u},
    {0x4fa793a7u, 0x1a8942a0u, 0x00000005u},
    {0x53295488u, 0x52a550a7u, 0x00000ab5u},
    {0x56ab1578u, 0x8ac15eaeu, 0x00000b25u},
    {0x5b2d1667u, 0xd2e570b7u, 0x00000005u},
    {0x5eaf1768u, 0x0b017ebeu, 0x00000c26u},
    {0x63315868u, 0x532590c7u, 0x00000cb6u},
    {0x67335988u, 0x9345a0cfu, 0x00000d36u},
    {0x01004007u, 0x30140803u, 0x00000000u},
    {0x048200e7u, 0x6830160au, 0x00000000u},
    {0x0803c1c6u, 0x004c2411u, 0x00000000u},
    {0x0b054287u, 0xd0643017u, 0x00000000u},
    {0x0e870367u, 0x08803e1eu, 0x00000001u},
    {0x11888437u, 0x38984a24u, 0x00000001u},
    {0x150a4507u, 0x70b4582bu, 0x00000001u},
    {0x188c05e6u, 0x00d06632u, 0x00000000u},
    {0x1b8d86a7u, 0xd8e87238u, 0x00000001u},
    {0x1f0f4787u, 0x1104803fu, 0x00000002u},
    {0x22910868u, 0x49208e46u, 0x000004a2u},
    {0x26930968u, 0x89409e4eu, 0x00000522u},
    {0x2a950a68u, 0xc960ae56u, 0x000005a2u},
    {0x2e970b67u, 0x0980be5eu, 0x00000003u},
    {0x3218cc48u, 0x49a0ce65u, 0x000006a3u},
    {0x371b4d88u, 0x99c8e26fu, 0x00000743u},
    {0x3c1dcec8u, 0xe9f0f679u, 0x000007e3u},
    {0x41205008u, 0x3a190883u, 0x00000884u},
    {0x4622d128u, 0x8a411c8du, 0x00000924u},
    {0x4aa51267u, 0xca612e96u, 0x00000004u},
    {0x4e26d347u, 0x027d3c9du, 0x00000005u},
    {0x51a89426u, 0x02994aa4u, 0x00000000u},
    {0x54aa14e7u, 0x6ab156aau, 0x00000005u},
    {0x582bd5c7u, 0xa2cd64b1u, 0x00000005u},
    {0x5bad96a8u, 0xdae972b8u, 0x00000bc5u},
    {0x5faf97a8u, 0x1b0982c0u, 0x00000c46u},
    {0x63b198a8u, 0x5b2992c8u, 0x00000cc6u},
    {0x67b399a7u, 0x9b49a2d0u, 0x00000006u},
    {0x01004007u, 0x30140803u, 0x00000000u},
    {0x0401c0d7u, 0x602c1409u, 0x00000000u},
    {0x078381a6u, 0x00482210u, 0x00000000u},
    {0x0a850266u, 0x00602e16u, 0x00000000u},
    {0x0d868326u, 0x00783a1cu, 0x00000000u},
    {0x108803e7u, 0x28904622u, 0x00000001u},
    {0x138984b7u, 0x58a85228u, 0x00000001u},
    {0x170b4586u, 0x00c4602fu, 0x00000000u},
    {0x1a0cc646u, 0x00dc6c35u, 0x00000000u},
    {0x1d0e4706u, 0x00f4783bu, 0x00000000u},
    {0x200fc7c7u, 0x210c8441u, 0x00000002u},
    {0x239188a7u, 0x59289248u, 0x00000002u},
    {0x27134988u, 0x9144a04fu, 0x00000532u},
    {0x2b154a87u, 0xd164b057u, 0x00000002u},
    {0x2e970b67u, 0x0980be5eu, 0x00000003u},
    {0x3218cc48u, 0x499ccc65u, 0x000006a3u},
    {0x369b0d68u, 0x91c4e06eu, 0x00000733u},
    {0x3b9d4e88u, 0xd9e8f278u, 0x000007c3u},
    {0x401fcfa8u, 0x220d0481u, 0x00000864u},
    {0x44a210e8u, 0x7235168au, 0x000008f4u},
    {0x49245208u, 0xb2552893u, 0x00000974u},
    {0x4ca612f7u, 0xea71369au, 0x00000004u},
    {0x5027d3c6u, 0x028d44a1u, 0x00000000u},
    {0x53295486u, 0x02a550a7u, 0x00000000u},
    {0x562ad546u, 0x02bd5cadu, 0x00000000u},
    {0x592c5607u, 0xb2d568b3u, 0x00000005u},
    {0x5cae16e7u, 0xeaf176bau, 0x00000005u},
    {0x602fd7c8u, 0x230d84c1u, 0x00000c56u},
    {0x6431d8c7u, 0x632d94c9u, 0x00000006u},
    {0x67b399a7u, 0x9b49a2d0u, 0x00000006u},
    {0x01004006u, 0x00140803u, 0x00000000u},
    {0x0401c0c6u, 0x002c1409u, 0x00000000u},
    {0x07034186u, 0x0044200fu, 0x00000000u},
    {0x0a04c246u, 0x005c2c15u, 0x00000000u},
    {0x0c8602f6u, 0x0070361au, 0x00000000u},
    {0x0f8783a6u, 0x00884220u, 0x00000000u},
    {0x12890466u, 0x00a04e26u, 0x00000000u},
    {0x158a8526u, 0x00b85a2cu, 0x00000000u},
    {0x188c05e6u, 0x00d06632u, 0x00000000u},
    {0x1b0d4696u, 0x00e47037u, 0x00000000u},
    {0x1e0ec746u, 0x00fc7c3du, 0x00000000u},
    {0x21104807u, 0x31148843u, 0x00000002u},
    {0x249208e7u, 0x6930964au, 0x00000002u},
    {0x2813c9c7u, 0xa14ca451u, 0x00000002u},
    {0x2b958aa7u, 0xd968b258u, 0x00000002u},
    {0x2f174b86u, 0x0184c05fu, 0x00000000u},
    {0x3218cc48u, 0x419ccc65u, 0x00000693u},
    {0x361acd48u, 0x89c0de6eu, 0x00000723u},
    {0x3a9d0e68u, 0xc9e0ee76u, 0x000007b3u},
    {0x3f1f4f88u, 0x1205007fu, 0x00000834u},
    {0x43a15088u, 0x5a291288u, 0x000008c4u},
    {0x47a391a7u, 0xa2492290u, 0x00000004u},
    {0x4ba592a6u, 0x02693298u, 0x00000000u},
    {0x4ea71366u, 0x02813e9eu, 0x00000000u},
    {0x51a89426u, 0x02994aa4u, 0x00000000u},
    {0x5429d4d6u, 0x02ad54a9u, 0x00000000u},
    {0x572b5586u, 0x02c560afu, 0x00000000u},
    {0x5a2cd647u, 0xc2dd6cb5u, 0x00000005u},
    {0x5dae9727u, 0xfaf97abcu, 0x00000005u},
    {0x61305807u, 0x331588c3u, 0x00000006u},
    {0x6431d8d8u, 0x632d94c9u, 0x00000cd6u},
    {0x67b399b7u, 0x9b49a2d0u, 0x00000006u},
    {0x01004006u, 0x00140803u, 0x00000000u},
    {0x0401c0c5u, 0x00001409u, 0x00000000u},
    {0x06830166u, 0x00401e0eu, 0x00000000u},
    {0x09848225u, 0x00002a14u, 0x00000000u},
    {0x0c05c2c6u, 0x006c3419u, 0x00000000u},
    {0x0e870376u, 0x00803e1eu, 0x00000000u},
    {0x11888426u, 0x00984a24u, 0x00000000u},
    {0x1409c4d6u, 0x00ac5429u, 0x00000000u},
    {0x170b4586u, 0x00c4602fu, 0x00000000u},
    {0x198c8636u, 0x00d86a34u, 0x00000000u},
    {0x1c8e06e5u, 0x0000763au, 0x00000000u},
    {0x1f0f4786u, 0x0104803fu, 0x00000000u},
    {0x2210c847u, 0x411c8c45u, 0x00000002u},
    {0x25124917u, 0x7134984bu, 0x00000002u},
    {0x289409e7u, 0xa950a652u, 0x00000002u},
    {0x2b958ab7u, 0xd968b258u, 0x00000002u},
    {0x2f174b86u, 0x0184c05fu, 0x00000000u},
    {0x3218cc48u, 0x419ccc65u, 0x00000693u},
    {0x361acd48u, 0x81bcdc6du, 0x00000713u},
    {0x3a1cce48u, 0xc1dcec75u, 0x00000793u},
    {0x3e1ecf48u, 0x01fcfc7du, 0x00000814u},
    {0x4220d048u, 0x421d0c85u, 0x00000894u},
    {0x4622d148u, 0x8a411c8du, 0x00000924u},
    {0x4aa51266u, 0x02612e96u, 0x00000000u},
    {0x4d265316u, 0x0275389bu, 0x00000000u},
    {0x5027d3c6u, 0x028d44a1u, 0x00000000u},
    {0x52a91476u, 0x02a14ea6u, 0x00000000u},
    {0x55aa9525u, 0x00015aacu, 0x00000000u},
    {0x582bd5c6u, 0x02cd64b1u, 0x00000000u},
    {0x5b2d5686u, 0x02e570b7u, 0x00000000u},
    {0x5e2ed747u, 0x02fd7cbdu, 0x00000006u},
    {0x61b09826u, 0x03198ac4u, 0x00000000u},
    {0x64b218e7u, 0x6b3196cau, 0x00000006u},
    {0x6833d9c6u, 0x034da4d1u, 0x00000000u},
    {0x01004006u, 0x00140803u, 0x00000000u},
    {0x038180b6u, 0x00281208u, 0x00000000u},
    {0x0602c156u, 0x003c1c0du, 0x00000000u},
    {0x09044205u, 0x00002813u, 0x00000000u},
    {0x0b8582a5u, 0x00003218u, 0x00000000u},
    {0x0e06c345u, 0x00003c1du, 0x00000000u},
    {0x108803e6u, 0x00904622u, 0x00000000u},
    {0x13094496u, 0x00a45027u, 0x00000000u},
    {0x158a8536u, 0x00b85a2cu, 0x00000000u},
    {0x180bc5d6u, 0x00cc6431u, 0x00000000u},
    {0x1b0d4685u, 0x00007037u, 0x00000000u},
    {0x1d8e8725u, 0x00007a3cu, 0x00000000u},
    {0x200fc7c6u, 0x010c8441u, 0x00000000u},
    {0x23114886u, 0x01249047u, 0x00000000u},
    {0x2612c946u, 0x013c9c4du, 0x00000000u},
    {0x29144a06u, 0x0154a853u, 0x00000000u},
    {0x2c15cac7u, 0xe16cb459u, 0x00000002u},
    {0x2f174b96u, 0x0184c05fu, 0x00000000u},
    {0x3218cc47u, 0x419ccc65u, 0x00000003u},
    {0x359a8d28u, 0x79b8da6cu, 0x00000703u},
    {0x399c8e28u, 0xb9d8ea74u, 0x00000783u},
    {0x3d9e8f27u, 0xf9f8fa7cu, 0x00000003u},
    {0x41205008u, 0x32150883u, 0x00000874u},
    {0x45225108u, 0x7235188bu, 0x000008f4u},
    {0x49245207u, 0xb2552893u, 0x00000004u},
    {0x4c25d2d6u, 0x026d3499u, 0x00000000u},
    {0x4ea71376u, 0x02813e9eu, 0x00000000u},
    {0x51285416u, 0x029548a3u, 0x00000000u},
    {0x5429d4c5u, 0x000154a9u, 0x00000000u},
    {0x56ab1565u, 0x00015eaeu, 0x00000000u},
    {0x592c5606u, 0x02d568b3u, 0x00000000u},
    {0x5bad96b7u, 0xdae972b8u, 0x00000005u},
    {0x5f2f5786u, 0x030580bfu, 0x00000000u},
    {0x6230d846u, 0x031d8cc5u, 0x00000000u},
    {0x65325906u, 0x033598cbu, 0x00000000u},
    {0x6833d9c6u, 0x034da4d1u, 0x00000000u},
    {0x01004005u, 0x00000803u, 0x00000000u},
    {0x038180a5u, 0x00001208u, 0x00000000u},
    {0x0602c145u, 0x00001c0du, 0x00000000u},
    {0x088401e5u, 0x00002612u, 0x00000000u},
    {0x0b054285u, 0x00003017u, 0x00000000u},
    {0x0d064316u, 0x0074381bu, 0x00000000u},
    {0x0f8783b6u, 0x00884220u, 0x00000000u},
    {0x1208c456u, 0x009c4c25u, 0x00000000u},
    {0x148a04f5u, 0x0000562au, 0x00000000u},
    {0x170b4585u, 0x0000602fu, 0x00000000u},
    {0x198c8625u, 0x00006a34u, 0x00000000u},
    {0x1c0dc6c5u, 0x00007439u, 0x00000000u},
    {0x1e8f0765u, 0x00007e3eu, 0x00000000u},
    {0x21104805u, 0x00008843u, 0x00000000u},
    {0x239188a6u, 0x01289248u, 0x00000000u},
    {0x26930966u, 0x01409e4eu, 0x00000000u},
    {0x29948a26u, 0x0158aa54u, 0x00000000u},
    {0x2c960ae6u, 0x0170b65au, 0x00000000u},
    {0x2f978ba5u, 0x0000c260u, 0x00000000u},
    {0x3218cc47u, 0x419ccc65u, 0x00000003u},
    {0x359a8d27u, 0x79b8da6cu, 0x00000003u},
    {0x391c4e07u, 0xb1d4e873u, 0x00000003u},
    {0x3c9e0ee8u, 0xe9f0f67au, 0x000007e3u},
    {0x40a00fe7u, 0x2a110682u, 0x00000004u},
    {0x4421d0c7u, 0x622d1489u, 0x00000004u},
    {0x47a391a8u, 0x9a492290u, 0x00000944u},
    {0x4b255296u, 0x02653097u, 0x00000000u},
    {0x4da69335u, 0x00013a9cu, 0x00000000u},
    {0x5027d3c5u, 0x000144a1u, 0x00000000u},
    {0x52a91465u, 0x00014ea6u, 0x00000000u},
    {0x552a5505u, 0x000158abu, 0x00000000u},
    {0x57ab95a5u, 0x000162b0u, 0x00000000u},
    {0x5a2cd645u, 0x00016cb5u, 0x00000000u},
    {0x5cae16e6u, 0x02f176bau, 0x00000000u},
    {0x5faf97a6u, 0x030982c0u, 0x00000000u},
    {0x62b11866u, 0x03218ec6u, 0x00000000u},
    {0x65b29926u, 0x03399accu, 0x00000000u},
    {0x6833d9d6u, 0x034da4d1u, 0x00000000u},
    {0x01004005u, 0x00000803u, 0x00000000u},
    {0x038180a5u, 0x00001208u, 0x00000000u},
    {0x05828135u, 0x00001a0cu, 0x00000000u},
    {0x0803c1c5u, 0x00002411u, 0x00000000u},
    {0x0a850265u, 0x00002e16u, 0x00000000u},
    {0x0c8602f5u, 0x0000361au, 0x00000000u},
    {0x0f074385u, 0x0000401fu, 0x00000000u},
    {0x11888425u, 0x00004a24u, 0x00000000u},
    {0x138984b5u, 0x00005228u, 0x00000000u},
    {0x160ac545u, 0x00005c2du, 0x00000000u},
    {0x180bc5d5u, 0x00006431u, 0x00000000u},
    {0x1a8d0665u, 0x00006e36u, 0x00000000u},
    {0x1d0e4705u, 0x0000783bu, 0x00000000u},
    {0x1f0f4795u, 0x0000803fu, 0x00000000u},
    {0x21908826u, 0x01188a44u, 0x00000000u},
    {0x249208e5u, 0x0000964au, 0x00000000u},
    {0x27134986u, 0x0144a04fu, 0x00000000u},
    {0x2a14ca46u, 0x015cac55u, 0x00000000u},
    {0x2c960af6u, 0x0170b65au, 0x00000000u},
    {0x2f978ba5u, 0x0000c260u, 0x00000000u},
    {0x3218cc46u, 0x019ccc65u, 0x00000000u},
    {0x351a4d07u, 0x71b4d86bu, 0x00000003u},
    {0x389c0de7u, 0xa9d0e672u, 0x00000003u},
    {0x3c1dcec7u, 0xe1ecf479u, 0x00000003u},
    {0x3f9f8fa7u, 0x1a090280u, 0x00000004u},
    {0x43215087u, 0x52251087u, 0x00000004u},
    {0x46a31167u, 0x8a411e8eu, 0x00000004u},
    {0x4a24d246u, 0x025d2c95u, 0x00000000u},
    {0x4ca612f5u, 0x0001369au, 0x00000000u},
    {0x4f275385u, 0x0001409fu, 0x00000000u},
    {0x51285415u, 0x000148a3u, 0x00000000u},
    {0x53a994a5u, 0x000152a8u, 0x00000000u},
    {0x562ad545u, 0x00015cadu, 0x00000000u},
    {0x582bd5d5u, 0x000164b1u, 0x00000000u},
    {0x5aad1666u, 0x02e16eb6u, 0x00000000u},
    {0x5d2e5716u, 0x02f578bbu, 0x00000000u},
    {0x602fd7c6u, 0x030d84c1u, 0x00000000u},
    {0x63315885u, 0x000190c7u, 0x00000000u},
    {0x65b29926u, 0x03399accu, 0x00000000u},
    {0x68b419e5u, 0x0001a6d2u, 0x00000000u},
    {0x01004005u, 0x00000803u, 0x00000000u},
    {0x03014095u, 0x00001007u, 0x00000000u},
    {0x05828125u, 0x00001a0cu, 0x00000000u},
    {0x078381b5u, 0x00002210u, 0x00000000u},
    {0x0a04c245u, 0x00002c15u, 0x00000000u},
    {0x0c05c2d5u, 0x00003419u, 0x00000000u},
    {0x0e870364u, 0x0000001eu, 0x00000000u},
    {0x108803e5u, 0x00004622u, 0x00000000u},
    {0x12890475u, 0x00004e26u, 0x00000000u},
    {0x150a4505u, 0x0000582bu, 0x00000000u},
    {0x170b4595u, 0x0000602fu, 0x00000000u},
    {0x198c8624u, 0x00000034u, 0x00000000u},
    {0x1b8d86a5u, 0x00007238u, 0x00000000u},
    {0x1d8e8735u, 0x00007a3cu, 0x00000000u},
    {0x200fc7c5u, 0x00008441u, 0x00000000u},
    {0x22910865u, 0x00008e46u, 0x00000000u},
    {0x25124906u, 0x0134984bu, 0x00000000u},
    {0x279389b6u, 0x0148a250u, 0x00000000u},
    {0x2a14ca56u, 0x015cac55u, 0x00000000u},
    {0x2d164b05u, 0x0000b85bu, 0x00000000u},
    {0x2f978ba5u, 0x0000c260u, 0x00000000u},
    {0x3218cc46u, 0x019ccc65u, 0x00000000u},
    {0x351a4d07u, 0x71b4d86bu, 0x00000003u},
    {0x389c0de6u, 0x01d0e672u, 0x00000000u},
    {0x3b9d8ea7u, 0xd9e8f278u, 0x00000003u},
    {0x3f1f4f87u, 0x1205007fu, 0x00000004u},
    {0x4220d057u, 0x421d0c85u, 0x00000004u},
    {0x45a29127u, 0x7a391a8cu, 0x00000004u},
    {0x49245206u, 0x02552893u, 0x00000000u},
    {0x4ba592b5u, 0x00013298u, 0x00000000u},
    {0x4e26d345u, 0x00013c9du, 0x00000000u},
    {0x5027d3d5u, 0x000144a1u, 0x00000000u},
    {0x52a91464u, 0x000000a6u, 0x00000000u},
    {0x54aa14e5u, 0x000156aau, 0x00000000u},
    {0x56ab1575u, 0x00015eaeu, 0x00000000u},
    {0x592c5605u, 0x000168b3u, 0x00000000u},
    {0x5b2d5696u, 0x02e570b7u, 0x00000000u},
    {0x5e2ed745u, 0x00017cbdu, 0x00000000u},
    {0x60b017e6u, 0x031186c2u, 0x00000000u},
    {0x63315896u, 0x032590c7u, 0x00000000u},
    {0x6632d945u, 0x00019ccdu, 0x00000000u},
    {0x68b419e5u, 0x0001a6d2u, 0x00000000u},
    {0x01004005u, 0x00000803u, 0x00000000u},
    {0x03014095u, 0x00001007u, 0x00000000u},
    {0x05024115u, 0x0000180bu, 0x00000000u},
    {0x078381a4u, 0x00000010u, 0x00000000u},
    {0x09848225u, 0x00002a14u, 0x00000000u},
    {0x0b8582b5u, 0x00003218u, 0x00000000u},
    {0x0d868335u, 0x00003a1cu, 0x00000000u},
    {0x1007c3c4u, 0x00000021u, 0x00000000u},
    {0x1208c444u, 0x00000025u, 0x00000000u},
    {0x1409c4c5u, 0x00005429u, 0x00000000u},
    {0x160ac555u, 0x00005c2du, 0x00000000u},
    {0x180bc5d5u, 0x00006431u, 0x00000000u},
    {0x1a8d0664u, 0x00000036u, 0x00000000u},
    {0x1c8e06e4u, 0x0000003au, 0x00000000u},
    {0x1e8f0765u, 0x00007e3eu, 0x00000000u},
    {0x209007f5u, 0x00008642u, 0x00000000u},
    {0x23114885u, 0x00009047u, 0x00000000u},
    {0x25928925u, 0x00009a4cu, 0x00000000u},
    {0x2813c9c6u, 0x014ca451u, 0x00000000u},
    {0x2a950a76u, 0x0160ae56u, 0x00000000u},
    {0x2d164b16u, 0x0174b85bu, 0x00000000u},
    {0x2f978bb5u, 0x0000c260u, 0x00000000u},
    {0x3218cc46u, 0x019ccc65u, 0x00000000u},
    {0x351a4d06u, 0x01b4d86bu, 0x00000000u},
    {0x381bcdc6u, 0x01cce471u, 0x00000000u},
    {0x3b1d4e87u, 0xd1e4f077u, 0x00000003u},
    {0x3e9f0f66u, 0x0200fe7eu, 0x00000000u},
    {0x41a09026u, 0x02190a84u, 0x00000000u},
    {0x44a210e7u, 0x6a31168au, 0x00000004u},
    {0x4823d1c6u, 0x024d2491u, 0x00000000u},
    {0x4b255284u, 0x00000097u, 0x00000000u},
    {0x4d265305u, 0x0001389bu, 0x00000000u},
    {0x4f275395u, 0x0001409fu, 0x00000000u},
    {0x51285415u, 0x000148a3u, 0x00000000u},
    {0x53a994a4u, 0x000000a8u, 0x00000000u},
    {0x55aa9524u, 0x000000acu, 0x00000000u},
    {0x57ab95a5u, 0x000162b0u, 0x00000000u},
    {0x59ac9635u, 0x00016ab4u, 0x00000000u},
    {0x5c2dd6c5u, 0x000174b9u, 0x00000000u},
    {0x5eaf1765u, 0x00017ebeu, 0x00000000u},
    {0x61305805u, 0x000188c3u, 0x00000000u},
    {0x63b198a5u, 0x000192c8u, 0x00000000u},
    {0x6632d945u, 0x00019ccdu, 0x00000000u},
    {0x68b419e5u, 0x0001a6d2u, 0x00000000u},
    {0x01004005u, 0x00000803u, 0x00000000u},
    {0x03014095u, 0x00001007u, 0x00000000u},
    {0x05024115u, 0x0000180bu, 0x00000000u},
    {0x07034195u, 0x0000200fu, 0x00000000u},
    {0x09044215u, 0x00002813u, 0x00000000u},
    {0x0b054295u, 0x00003017u, 0x00000000u},
    {0x0d064315u, 0x0000381bu, 0x00000000u},
    {0x0f074395u, 0x0000401fu, 0x00000000u},
    {0x11084415u, 0x00004823u, 0x00000000u},
    {0x13094495u, 0x00005027u, 0x00000000u},
    {0x150a4515u, 0x0000582bu, 0x00000000u},
    {0x170b4595u, 0x0000602fu, 0x00000000u},
    {0x190c4615u, 0x00006833u, 0x00000000u},
    {0x1b0d4695u, 0x00007037u, 0x00000000u},
    {0x1d0e4715u, 0x0000783bu, 0x00000000u},
    {0x1f0f4795u, 0x0000803fu, 0x00000000u},
    {0x21104816u, 0x01148843u, 0x00000000u},
    {0x239188b6u, 0x01289248u, 0x00000000u},
    {0x2612c955u, 0x00009c4du, 0x00000000u},
    {0x289409e5u, 0x0000a652u, 0x00000000u},
    {0x2b154a85u, 0x0000b057u, 0x00000000u},
    {0x2d968b25u, 0x0000ba5cu, 0x00000000u},
    {0x2f978bb5u, 0x0000c260u, 0x00000000u},
    {0x3218cc46u, 0x019ccc65u, 0x00000000u},
    {0x349a0cf7u, 0x69b0d66au, 0x00000003u},
    {0x379b8db7u, 0x99c8e270u, 0x00000003u},
    {0x3a9d0e77u, 0xc9e0ee76u, 0x00000003u},
    {0x3d9e8f37u, 0xf9f8fa7cu, 0x00000003u},
    {0x40a00ff7u, 0x2a110682u, 0x00000004u},
    {0x4421d0c6u, 0x022d1489u, 0x00000000u},
    {0x47235186u, 0x0245208fu, 0x00000000u},
    {0x4a24d245u, 0x00012c95u, 0x00000000u},
    {0x4c25d2d5u, 0x00013499u, 0x00000000u},
    {0x4e26d355u, 0x00013c9du, 0x00000000u},
    {0x5027d3d5u, 0x000144a1u, 0x00000000u},
    {0x5228d455u, 0x00014ca5u, 0x00000000u},
    {0x5429d4d5u, 0x000154a9u, 0x00000000u},
    {0x562ad555u, 0x00015cadu, 0x00000000u},
    {0x582bd5d5u, 0x000164b1u, 0x00000000u},
    {0x5a2cd655u, 0x00016cb5u, 0x00000000u},
    {0x5cae16e5u, 0x000176bau, 0x00000000u},
    {0x5f2f5785u, 0x000180bfu, 0x00000000u},
    {0x61b09825u, 0x00018ac4u, 0x00000000u},
    {0x6431d8c5u, 0x000194c9u, 0x00000000u},
    {0x6632d956u, 0x033d9ccdu, 0x00000000u},
    {0x68b419f5u, 0x0001a6d2u, 0x00000000u},
    {0x01004004u, 0x00000003u, 0x00000000u},
    {0x03014084u, 0x00000007u, 0x00000000u},
    {0x05024104u, 0x0000000bu, 0x00000000u},
    {0x07034184u, 0x0000000fu, 0x00000000u},
    {0x088401f5u, 0x00002612u, 0x00000000u},
    {0x0a850275u, 0x00002e16u, 0x00000000u},
    {0x0c8602f5u, 0x0000361au, 0x00000000u},
    {0x0e870374u, 0x0000001eu, 0x00000000u},
    {0x108803e4u, 0x00000022u, 0x00000000u},
    {0x12890464u, 0x00000026u, 0x00000000u},
    {0x148a04e4u, 0x0000002au, 0x00000000u},
    {0x160ac555u, 0x00005c2du, 0x00000000u},
    {0x180bc5d5u, 0x00006431u, 0x00000000u},
    {0x1a0cc655u, 0x00006c35u, 0x00000000u},
    {0x1c0dc6d4u, 0x00000039u, 0x00000000u},
    {0x1e0ec744u, 0x0000003du, 0x00000000u},
    {0x200fc7c4u, 0x00000041u, 0x00000000u},
    {0x2210c845u, 0x00008c45u, 0x00000000u},
    {0x2411c8d6u, 0x012c9449u, 0x00000000u},
    {0x26930975u, 0x00009e4eu, 0x00000000u},
    {0x29144a05u, 0x0000a853u, 0x00000000u},
    {0x2b154a95u, 0x0000b057u, 0x00000000u},
    {0x2d968b25u, 0x0000ba5cu, 0x00000000u},
    {0x3017cbc4u, 0x00000061u, 0x00000000u},
    {0x3218cc45u, 0x0000cc65u, 0x00000000u},
    {0x349a0ce6u, 0x01b0d66au, 0x00000000u},
    {0x379b8da6u, 0x01c8e270u, 0x00000000u},
    {0x3a9d0e66u, 0x01e0ee76u, 0x00000000u},
    {0x3d1e4f16u, 0x01f4f87bu, 0x00000000u},
    {0x401fcfc6u, 0x020d0481u, 0x00000000u},
    {0x43215086u, 0x02251087u, 0x00000000u},
    {0x4622d146u, 0x023d1c8du, 0x00000000u},
    {0x49245205u, 0x00012893u, 0x00000000u},
    {0x4ba592a4u, 0x00000098u, 0x00000000u},
    {0x4da69324u, 0x0000009cu, 0x00000000u},
    {0x4f275395u, 0x0001409fu, 0x00000000u},
    {0x51285415u, 0x000148a3u, 0x00000000u},
    {0x53295495u, 0x000150a7u, 0x00000000u},
    {0x552a5514u, 0x000000abu, 0x00000000u},
    {0x572b5584u, 0x000000afu, 0x00000000u},
    {0x592c5604u, 0x000000b3u, 0x00000000u},
    {0x5b2d5685u, 0x000170b7u, 0x00000000u},
    {0x5d2e5715u, 0x000178bbu, 0x00000000u},
    {0x5faf97a5u, 0x000182c0u, 0x00000000u},
    {0x6230d845u, 0x00018cc5u, 0x00000000u},
    {0x6431d8d5u, 0x000194c9u, 0x00000000u},
    {0x66b31965u, 0x00019eceu, 0x00000000u},
    {0x68b419f5u, 0x0001a6d2u, 0x00000000u},
    {0x01004004u, 0x00000003u, 0x00000000u},
    {0x03014084u, 0x00000007u, 0x00000000u},
    {0x048200f5u, 0x0000160au, 0x00000000u},
    {0x06830174u, 0x0000000eu, 0x00000000u},
    {0x088401e4u, 0x00000012u, 0x00000000u},
    {0x0a04c255u, 0x00002c15u, 0x00000000u},
    {0x0c05c2d4u, 0x00000019u, 0x00000000u},
    {0x0e06c344u, 0x0000001du, 0x00000000u},
    {0x1007c3c4u, 0x00000021u, 0x00000000u},
    {0x11888435u, 0x00004a24u, 0x00000000u},
    {0x138984b4u, 0x00000028u, 0x00000000u},
    {0x158a8524u, 0x0000002cu, 0x00000000u},
    {0x170b4595u, 0x0000602fu, 0x00000000u},
    {0x190c4614u, 0x00000033u, 0x00000000u},
    {0x1b0d4684u, 0x00000037u, 0x00000000u},
    {0x1d0e4704u, 0x0000003bu, 0x00000000u},
    {0x1e8f0775u, 0x00007e3eu, 0x00000000u},
    {0x209007f5u, 0x00008642u, 0x00000000u},
    {0x22910875u, 0x00008e46u, 0x00000000u},
    {0x25124904u, 0x0000004bu, 0x00000000u},
    {0x27134985u, 0x0000a04fu, 0x00000000u},
    {0x29144a15u, 0x0000a853u, 0x00000000u},
    {0x2b958aa5u, 0x0000b258u, 0x00000000u},
    {0x2d968b35u, 0x0000ba5cu, 0x00000000u},
    {0x3017cbc4u, 0x00000061u, 0x00000000u},
    {0x3218cc45u, 0x0000cc65u, 0x00000000u},
    {0x349a0ce6u, 0x01b0d66au, 0x00000000u},
    {0x371b4d96u, 0x01c4e06fu, 0x00000000u},
    {0x3a1cce46u, 0x01dcec75u, 0x00000000u},
    {0x3c9e0ef6u, 0x01f0f67au, 0x00000000u},
    {0x3f9f8fa6u, 0x02090280u, 0x00000000u},
    {0x42a11065u, 0x00010e86u, 0x00000000u},
    {0x45225106u, 0x0235188bu, 0x00000000u},
    {0x4823d1c6u, 0x024d2491u, 0x00000000u},
    {0x4aa51275u, 0x00012e96u, 0x00000000u},
    {0x4ca612f4u, 0x0000009au, 0x00000000u},
    {0x4ea71364u, 0x0000009eu, 0x00000000u},
    {0x5027d3d5u, 0x000144a1u, 0x00000000u},
    {0x5228d454u, 0x000000a5u, 0x00000000u},
    {0x5429d4c4u, 0x000000a9u, 0x00000000u},
    {0x562ad544u, 0x000000adu, 0x00000000u},
    {0x57ab95b5u, 0x000162b0u, 0x00000000u},
    {0x59ac9634u, 0x000000b4u, 0x00000000u},
    {0x5bad96a5u, 0x000172b8u, 0x00000000u},
    {0x5dae9735u, 0x00017abcu, 0x00000000u},
    {0x602fd7c5u, 0x000184c1u, 0x00000000u},
    {0x6230d855u, 0x00018cc5u, 0x00000000u},
    {0x64b218e5u, 0x000196cau, 0x00000000u},
    {0x66b31975u, 0x00019eceu, 0x00000000u},
    {0x69345a04u, 0x000000d3u, 0x00000000u},
    {0x01004004u, 0x00000003u, 0x00000000u},
    {0x03014084u, 0x00000007u, 0x00000000u},
    {0x048200f4u, 0x0000000au, 0x00000000u},
    {0x06830164u, 0x0000000eu, 0x00000000u},
    {0x0803c1d4u, 0x00000011u, 0x00000000u},
    {0x0a04c244u, 0x00000015u, 0x00000000u},
    {0x0b8582b4u, 0x00000018u, 0x00000000u},
    {0x0d868324u, 0x0000001cu, 0x00000000u},
    {0x0f074395u, 0x0000401fu, 0x00000000u},
    {0x11084414u, 0x00000023u, 0x00000000u},
    {0x13094484u, 0x00000027u, 0x00000000u},
    {0x148a04f4u, 0x0000002au, 0x00000000u},
    {0x168b0564u, 0x0000002eu, 0x00000000u},
    {0x180bc5d4u, 0x00000031u, 0x00000000u},
    {0x1a0cc644u, 0x00000035u, 0x00000000u},
    {0x1b8d86b5u, 0x00007238u, 0x00000000u},
    {0x1d8e8734u, 0x0000003cu, 0x00000000u},
    {0x1f8f87a4u, 0x00000040u, 0x00000000u},
    {0x21104815u, 0x00008843u, 0x00000000u},
    {0x23114895u, 0x00009047u, 0x00000000u},
    {0x25928924u, 0x0000004cu, 0x00000000u},
    {0x279389a5u, 0x0000a250u, 0x00000000u},
    {0x29948a35u, 0x0000aa54u, 0x00000000u},
    {0x2b958ab5u, 0x0000b258u, 0x00000000u},
    {0x2e16cb44u, 0x0000005du, 0x00000000u},
    {0x3017cbc4u, 0x00000061u, 0x00000000u},
    {0x3218cc45u, 0x0000cc65u, 0x00000000u},
    {0x349a0ce5u, 0x0000d66au, 0x00000000u},
    {0x371b4d86u, 0x01c4e06fu, 0x00000000u},
    {0x399c8e36u, 0x01d8ea74u, 0x00000000u},
    {0x3c9e0ee5u, 0x0000f67au, 0x00000000u},
    {0x3f1f4f86u, 0x0205007fu, 0x00000000u},
    {0x41a09036u, 0x02190a84u, 0x00000000u},
    {0x44a210e5u, 0x0001168au, 0x00000000u},
    {0x47235186u, 0x0245208fu, 0x00000000u},
    {0x49a49235u, 0x00012a94u, 0x00000000u},
    {0x4c25d2c4u, 0x00000099u, 0x00000000u},
    {0x4da69334u, 0x0000009cu, 0x00000000u},
    {0x4fa793a4u, 0x000000a0u, 0x00000000u},
    {0x51285414u, 0x000000a3u, 0x00000000u},
    {0x53295484u, 0x000000a7u, 0x00000000u},
    {0x54aa14f5u, 0x000156aau, 0x00000000u},
    {0x56ab1574u, 0x000000aeu, 0x00000000u},
    {0x58ac15e4u, 0x000000b2u, 0x00000000u},
    {0x5a2cd655u, 0x00016cb5u, 0x00000000u},
    {0x5c2dd6d5u, 0x000174b9u, 0x00000000u},
    {0x5e2ed755u, 0x00017cbdu, 0x00000000u},
    {0x60b017e4u, 0x000000c2u, 0x00000000u},
    {0x62b11865u, 0x00018ec6u, 0x00000000u},
    {0x64b218f5u, 0x000196cau, 0x00000000u},
    {0x66b31975u, 0x00019eceu, 0x00000000u},
    {0x69345a04u, 0x000000d3u, 0x00000000u},
    {0x01004004u, 0x00000003u, 0x00000000u},
    {0x02810074u, 0x00000006u, 0x00000000u},
    {0x048200e4u, 0x0000000au, 0x00000000u},
    {0x0602c154u, 0x0000000du, 0x00000000u},
    {0x078381b4u, 0x00000010u, 0x00000000u},
    {0x09044214u, 0x00000013u, 0x00000000u},
    {0x0b054283u, 0x00000000u, 0x00000000u},
    {0x0c8602e4u, 0x0000001au, 0x00000000u},
    {0x0e06c354u, 0x0000001du, 0x00000000u},
    {0x0f8783b4u, 0x00000020u, 0x00000000u},
    {0x11084414u, 0x00000023u, 0x00000000u},
    {0x13094483u, 0x00000000u, 0x00000000u},
    {0x148a04e4u, 0x0000002au, 0x00000000u},
    {0x160ac554u, 0x0000002du, 0x00000000u},
    {0x178b85b4u, 0x00000030u, 0x00000000u},
    {0x190c4614u, 0x00000033u, 0x00000000u},
    {0x1b0d4684u, 0x00000037u, 0x00000000u},
    {0x1c8e06f4u, 0x0000003au, 0x00000000u},
    {0x1e0ec754u, 0x0000003du, 0x00000000u},
    {0x200fc7c3u, 0x00000000u, 0x00000000u},
    {0x21908825u, 0x00008a44u, 0x00000000u},
    {0x239188b5u, 0x00009248u, 0x00000000u},
    {0x25928935u, 0x00009a4cu, 0x00000000u},
    {0x279389b5u, 0x0000a250u, 0x00000000u},
    {0x2a14ca44u, 0x00000055u, 0x00000000u},
    {0x2c15cac4u, 0x00000059u, 0x00000000u},
    {0x2e16cb45u, 0x0000bc5du, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x3218cc45u, 0x0000cc65u, 0x00000000u},
    {0x349a0ce6u, 0x01b4d66au, 0x00000000u},
    {0x379b8db7u, 0x99c8e270u, 0x00000003u},
    {0x3a9d0e77u, 0xc9e0ee76u, 0x00000003u},
    {0x3d9e8f37u, 0xf9f8fa7cu, 0x00000003u},
    {0x40a00ff7u, 0x2a110682u, 0x00000004u},
    {0x43a190b7u, 0x5a291288u, 0x00000004u},
    {0x46a31177u, 0x8a411e8eu, 0x00000004u},
    {0x49a49235u, 0x00012a94u, 0x00000000u},
    {0x4c25d2c3u, 0x00000000u, 0x00000000u},
    {0x4da69324u, 0x0000009cu, 0x00000000u},
    {0x4f275394u, 0x0000009fu, 0x00000000u},
    {0x50a813f4u, 0x000000a2u, 0x00000000u},
    {0x5228d455u, 0x00014ca5u, 0x00000000u},
    {0x5429d4d4u, 0x000000a9u, 0x00000000u},
    {0x55aa9534u, 0x000000acu, 0x00000000u},
    {0x572b5594u, 0x000000afu, 0x00000000u},
    {0x592c5603u, 0x00000000u, 0x00000000u},
    {0x5aad1664u, 0x000000b6u, 0x00000000u},
    {0x5cae16e4u, 0x000000bau, 0x00000000u},
    {0x5eaf1765u, 0x00017ebeu, 0x00000000u},
    {0x60b017f5u, 0x000186c2u, 0x00000000u},
    {0x62b11875u, 0x00018ec6u, 0x00000000u},
    {0x64b218f5u, 0x000196cau, 0x00000000u},
    {0x67335984u, 0x000000cfu, 0x00000000u},
    {0x69345a04u, 0x000000d3u, 0x00000000u},
    {0x01004004u, 0x00000003u, 0x00000000u},
    {0x02810074u, 0x00000006u, 0x00000000u},
    {0x0401c0d4u, 0x00000009u, 0x00000000u},
    {0x05828134u, 0x0000000cu, 0x00000000u},
    {0x07034194u, 0x0000000fu, 0x00000000u},
    {0x088401f4u, 0x00000012u, 0x00000000u},
    {0x0a850264u, 0x00000016u, 0x00000000u},
    {0x0c05c2d4u, 0x00000019u, 0x00000000u},
    {0x0d868334u, 0x0000001cu, 0x00000000u},
    {0x0f074394u, 0x0000001fu, 0x00000000u},
    {0x108803f4u, 0x00000022u, 0x00000000u},
    {0x1208c454u, 0x00000025u, 0x00000000u},
    {0x1409c4c3u, 0x00000000u, 0x00000000u},
    {0x158a8523u, 0x00000000u, 0x00000000u},
    {0x170b4583u, 0x00000000u, 0x00000000u},
    {0x188c05e3u, 0x00000000u, 0x00000000u},
    {0x1a0cc643u, 0x00000000u, 0x00000000u},
    {0x1b8d86a3u, 0x00000000u, 0x00000000u},
    {0x1d0e4704u, 0x0000003bu, 0x00000000u},
    {0x1e8f0774u, 0x0000003eu, 0x00000000u},
    {0x200fc7d4u, 0x00000041u, 0x00000000u},
    {0x2210c844u, 0x00000045u, 0x00000000u},
    {0x2411c8c4u, 0x00000049u, 0x00000000u},
    {0x2612c945u, 0x00009c4du, 0x00000000u},
    {0x2813c9d4u, 0x00000051u, 0x00000000u},
    {0x2a14ca44u, 0x00000055u, 0x00000000u},
    {0x2c15cac5u, 0x0000b459u, 0x00000000u},
    {0x2e16cb55u, 0x0000bc5du, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x3218cc44u, 0x00000065u, 0x00000000u},
    {0x3419ccc6u, 0x01acd469u, 0x00000000u},
    {0x371b4d85u, 0x0000e26fu, 0x00000000u},
    {0x3a1cce46u, 0x01dcec75u, 0x00000000u},
    {0x3d1e4f06u, 0x01f4f87bu, 0x00000000u},
    {0x401f8fb6u, 0x020d0481u, 0x00000000u},
    {0x43215086u, 0x02251087u, 0x00000000u},
    {0x45a29137u, 0x7a391a8cu, 0x00000004u},
    {0x49245205u, 0x00012893u, 0x00000000u},
    {0x4ba592a3u, 0x00000000u, 0x00000000u},
    {0x4d265303u, 0x00000000u, 0x00000000u},
    {0x4ea71363u, 0x00000000u, 0x00000000u},
    {0x5027d3c3u, 0x00000000u, 0x00000000u},
    {0x51a89423u, 0x00000000u, 0x00000000u},
    {0x53295484u, 0x000000a7u, 0x00000000u},
    {0x54aa14f4u, 0x000000aau, 0x00000000u},
    {0x562ad554u, 0x000000adu, 0x00000000u},
    {0x57ab95b4u, 0x000000b0u, 0x00000000u},
    {0x592c5614u, 0x000000b3u, 0x00000000u},
    {0x5aad1675u, 0x00016eb6u, 0x00000000u},
    {0x5d2e5704u, 0x000000bbu, 0x00000000u},
    {0x5f2f5784u, 0x000000bfu, 0x00000000u},
    {0x61305805u, 0x000188c3u, 0x00000000u},
    {0x63315895u, 0x000190c7u, 0x00000000u},
    {0x65325915u, 0x000198cbu, 0x00000000u},
    {0x67335995u, 0x0001a0cfu, 0x00000000u},
    {0x69345a14u, 0x000000d3u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x02810063u, 0x00000000u, 0x00000000u},
    {0x0401c0c4u, 0x00000009u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x07034184u, 0x0000000fu, 0x00000000u},
    {0x088401f4u, 0x00000012u, 0x00000000u},
    {0x0a04c254u, 0x00000015u, 0x00000000u},
    {0x0b8582b4u, 0x00000018u, 0x00000000u},
    {0x0d064314u, 0x0000001bu, 0x00000000u},
    {0x0e870374u, 0x0000001eu, 0x00000000u},
    {0x1007c3d4u, 0x00000021u, 0x00000000u},
    {0x11888434u, 0x00000024u, 0x00000000u},
    {0x13094494u, 0x00000027u, 0x00000000u},
    {0x148a04f4u, 0x0000002au, 0x00000000u},
    {0x160ac554u, 0x0000002du, 0x00000000u},
    {0x178b85b4u, 0x00000030u, 0x00000000u},
    {0x190c4614u, 0x00000033u, 0x00000000u},
    {0x1a8d0674u, 0x00000036u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d8e8724u, 0x0000003cu, 0x00000000u},
    {0x1f0f4794u, 0x0000003fu, 0x00000000u},
    {0x209007f4u, 0x00000042u, 0x00000000u},
    {0x22910864u, 0x00000046u, 0x00000000u},
    {0x249208e4u, 0x0000004au, 0x00000000u},
    {0x2612c955u, 0x00009c4du, 0x00000000u},
    {0x2813c9d5u, 0x0000a451u, 0x00000000u},
    {0x2a14ca55u, 0x0000ac55u, 0x00000000u},
    {0x2c960ae4u, 0x0000005au, 0x00000000u},
    {0x2e16cb55u, 0x0000bc5du, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c34u, 0x00000064u, 0x00000000u},
    {0x3419ccc5u, 0x0000d469u, 0x00000000u},
    {0x369b0d66u, 0x01c0de6eu, 0x00000000u},
    {0x3a1c8e25u, 0x0000ec75u, 0x00000000u},
    {0x3c9e0ee6u, 0x01f0f67au, 0x00000000u},
    {0x3f1f4f96u, 0x0205007fu, 0x00000000u},
    {0x4220d046u, 0x021d0c85u, 0x00000000u},
    {0x45a29105u, 0x00011a8cu, 0x00000000u},
    {0x4823d1c5u, 0x00012491u, 0x00000000u},
    {0x4aa51264u, 0x00000096u, 0x00000000u},
    {0x4c25d2d4u, 0x00000099u, 0x00000000u},
    {0x4da69334u, 0x0000009cu, 0x00000000u},
    {0x4f275394u, 0x0000009fu, 0x00000000u},
    {0x51285403u, 0x00000000u, 0x00000000u},
    {0x5228d454u, 0x000000a5u, 0x00000000u},
    {0x53a994b4u, 0x000000a8u, 0x00000000u},
    {0x552a5514u, 0x000000abu, 0x00000000u},
    {0x56ab1574u, 0x000000aeu, 0x00000000u},
    {0x582bd5d4u, 0x000000b1u, 0x00000000u},
    {0x59ac9634u, 0x000000b4u, 0x00000000u},
    {0x5bad96a4u, 0x000000b8u, 0x00000000u},
    {0x5dae9724u, 0x000000bcu, 0x00000000u},
    {0x5f2f5795u, 0x000180bfu, 0x00000000u},
    {0x61305815u, 0x000188c3u, 0x00000000u},
    {0x63315895u, 0x000190c7u, 0x00000000u},
    {0x65325915u, 0x000198cbu, 0x00000000u},
    {0x67335995u, 0x0001a0cfu, 0x00000000u},
    {0x69345a14u, 0x000000d3u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x02810063u, 0x00000000u, 0x00000000u},
    {0x038180b4u, 0x00000008u, 0x00000000u},
    {0x05828123u, 0x00000000u, 0x00000000u},
    {0x06830174u, 0x0000000eu, 0x00000000u},
    {0x0803c1d4u, 0x00000011u, 0x00000000u},
    {0x09848234u, 0x00000014u, 0x00000000u},
    {0x0b054294u, 0x00000017u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x0e06c344u, 0x0000001du, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x11084404u, 0x00000023u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x1409c4c3u, 0x00000000u, 0x00000000u},
    {0x150a4514u, 0x0000002bu, 0x00000000u},
    {0x170b4583u, 0x00000000u, 0x00000000u},
    {0x180bc5d4u, 0x00000031u, 0x00000000u},
    {0x198c8634u, 0x00000034u, 0x00000000u},
    {0x1b0d4694u, 0x00000037u, 0x00000000u},
    {0x1c8e06f4u, 0x0000003au, 0x00000000u},
    {0x1e0ec754u, 0x0000003du, 0x00000000u},
    {0x1f8f87b4u, 0x00000040u, 0x00000000u},
    {0x21104814u, 0x00000043u, 0x00000000u},
    {0x22910875u, 0x00008e46u, 0x00000000u},
    {0x249208f5u, 0x0000964au, 0x00000000u},
    {0x26930975u, 0x00009e4eu, 0x00000000u},
    {0x289409f4u, 0x00000052u, 0x00000000u},
    {0x2a950a64u, 0x00000056u, 0x00000000u},
    {0x2c960ae4u, 0x0000005au, 0x00000000u},
    {0x2e970b64u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x3218cc44u, 0x00000065u, 0x00000000u},
    {0x3419ccc5u, 0x0000d469u, 0x00000000u},
    {0x369b0d65u, 0x0000de6eu, 0x00000000u},
    {0x391c4e06u, 0x01d4e873u, 0x00000000u},
    {0x3b9d8eb6u, 0x01e8f278u, 0x00000000u},
    {0x3e9f0f65u, 0x0001007fu, 0x00000000u},
    {0x41a09025u, 0x00010c84u, 0x00000000u},
    {0x4421d0d6u, 0x022d1489u, 0x00000000u},
    {0x47235175u, 0x0001208fu, 0x00000000u},
    {0x49a49225u, 0x00012a94u, 0x00000000u},
    {0x4ba592b4u, 0x00000098u, 0x00000000u},
    {0x4d265314u, 0x0000009bu, 0x00000000u},
    {0x4ea71374u, 0x0000009eu, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x51a89423u, 0x00000000u, 0x00000000u},
    {0x53295483u, 0x00000000u, 0x00000000u},
    {0x54aa14e3u, 0x00000000u, 0x00000000u},
    {0x562ad543u, 0x00000000u, 0x00000000u},
    {0x572b5594u, 0x000000afu, 0x00000000u},
    {0x592c5603u, 0x00000000u, 0x00000000u},
    {0x5a2cd654u, 0x000000b5u, 0x00000000u},
    {0x5c2dd6c4u, 0x000000b9u, 0x00000000u},
    {0x5e2ed744u, 0x000000bdu, 0x00000000u},
    {0x602fd7c4u, 0x000000c1u, 0x00000000u},
    {0x61b09835u, 0x00018ac4u, 0x00000000u},
    {0x63b198b4u, 0x000000c8u, 0x00000000u},
    {0x65b29924u, 0x000000ccu, 0x00000000u},
    {0x67b399a4u, 0x000000d0u, 0x00000000u},
    {0x69345a14u, 0x000000d3u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x02810063u, 0x00000000u, 0x00000000u},
    {0x0401c0c3u, 0x00000000u, 0x00000000u},
    {0x05024114u, 0x0000000bu, 0x00000000u},
    {0x06830174u, 0x0000000eu, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x09848223u, 0x00000000u, 0x00000000u},
    {0x0b054283u, 0x00000000u, 0x00000000u},
    {0x0c05c2d4u, 0x00000019u, 0x00000000u},
    {0x0d868334u, 0x0000001cu, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x108803e3u, 0x00000000u, 0x00000000u},
    {0x1208c443u, 0x00000000u, 0x00000000u},
    {0x13094494u, 0x00000027u, 0x00000000u},
    {0x148a04f4u, 0x0000002au, 0x00000000u},
    {0x160ac554u, 0x0000002du, 0x00000000u},
    {0x178b85b4u, 0x00000030u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x1a8d0663u, 0x00000000u, 0x00000000u},
    {0x1c0dc6c3u, 0x00000000u, 0x00000000u},
    {0x1d0e4714u, 0x0000003bu, 0x00000000u},
    {0x1e8f0774u, 0x0000003eu, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21908824u, 0x00000044u, 0x00000000u},
    {0x23114894u, 0x00000047u, 0x00000000u},
    {0x25124904u, 0x0000004bu, 0x00000000u},
    {0x27134984u, 0x0000004fu, 0x00000000u},
    {0x289409f4u, 0x00000052u, 0x00000000u},
    {0x2a950a64u, 0x00000056u, 0x00000000u},
    {0x2c960ae4u, 0x0000005au, 0x00000000u},
    {0x2e970b64u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c34u, 0x00000064u, 0x00000000u},
    {0x33998ca5u, 0x0000d268u, 0x00000000u},
    {0x361acd46u, 0x01bcdc6du, 0x00000000u},
    {0x389c0df5u, 0x0000e872u, 0x00000000u},
    {0x3b9d8ea5u, 0x0000f278u, 0x00000000u},
    {0x3e1ecf46u, 0x01fcfc7du, 0x00000000u},
    {0x41205005u, 0x00010883u, 0x00000000u},
    {0x43a190a6u, 0x02291288u, 0x00000000u},
    {0x4622d156u, 0x023d1c8du, 0x00000000u},
    {0x49245205u, 0x00012893u, 0x00000000u},
    {0x4b255294u, 0x00000097u, 0x00000000u},
    {0x4ca612f3u, 0x00000000u, 0x00000000u},
    {0x4e26d343u, 0x00000000u, 0x00000000u},
    {0x4fa793a3u, 0x00000000u, 0x00000000u},
    {0x50a813f4u, 0x000000a2u, 0x00000000u},
    {0x5228d454u, 0x000000a5u, 0x00000000u},
    {0x53a994b4u, 0x000000a8u, 0x00000000u},
    {0x552a5514u, 0x000000abu, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x582bd5c3u, 0x00000000u, 0x00000000u},
    {0x59ac9623u, 0x00000000u, 0x00000000u},
    {0x5aad1674u, 0x000000b6u, 0x00000000u},
    {0x5cae16e4u, 0x000000bau, 0x00000000u},
    {0x5e2ed754u, 0x000000bdu, 0x00000000u},
    {0x602fd7c4u, 0x000000c1u, 0x00000000u},
    {0x6230d844u, 0x000000c5u, 0x00000000u},
    {0x63b198b4u, 0x000000c8u, 0x00000000u},
    {0x65b29924u, 0x000000ccu, 0x00000000u},
    {0x67b399a4u, 0x000000d0u, 0x00000000u},
    {0x69345a14u, 0x000000d3u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x02810063u, 0x00000000u, 0x00000000u},
    {0x0401c0c3u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x06830163u, 0x00000000u, 0x00000000u},
    {0x078381b4u, 0x00000010u, 0x00000000u},
    {0x09044214u, 0x00000013u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x0c05c2c3u, 0x00000000u, 0x00000000u},
    {0x0d064314u, 0x0000001bu, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x1007c3c3u, 0x00000000u, 0x00000000u},
    {0x11888423u, 0x00000000u, 0x00000000u},
    {0x12890474u, 0x00000026u, 0x00000000u},
    {0x1409c4d4u, 0x00000029u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x168b0574u, 0x0000002eu, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x198c8623u, 0x00000000u, 0x00000000u},
    {0x1a8d0674u, 0x00000036u, 0x00000000u},
    {0x1c0dc6d4u, 0x00000039u, 0x00000000u},
    {0x1d8e8734u, 0x0000003cu, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x200fc7d4u, 0x00000041u, 0x00000000u},
    {0x2210c844u, 0x00000045u, 0x00000000u},
    {0x239188b5u, 0x00009248u, 0x00000000u},
    {0x25928934u, 0x0000004cu, 0x00000000u},
    {0x27134994u, 0x0000004fu, 0x00000000u},
    {0x29144a04u, 0x00000053u, 0x00000000u},
    {0x2a950a74u, 0x00000056u, 0x00000000u},
    {0x2c960ae4u, 0x0000005au, 0x00000000u},
    {0x2e16cb55u, 0x0000bc5du, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c34u, 0x00000064u, 0x00000000u},
    {0x33194c95u, 0x0000d067u, 0x00000000u},
    {0x361acd45u, 0x0000dc6du, 0x00000000u},
    {0x389c0de5u, 0x0000e672u, 0x00000000u},
    {0x3b9d4e84u, 0x00000078u, 0x00000000u},
    {0x3d9e8f26u, 0x01f8fa7cu, 0x00000000u},
    {0x40a00fe5u, 0x00010682u, 0x00000000u},
    {0x43215085u, 0x00011087u, 0x00000000u},
    {0x45225116u, 0x0235188bu, 0x00000000u},
    {0x4823d1c5u, 0x00012491u, 0x00000000u},
    {0x4aa51263u, 0x00000000u, 0x00000000u},
    {0x4c25d2c3u, 0x00000000u, 0x00000000u},
    {0x4da69323u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x5027d3c3u, 0x00000000u, 0x00000000u},
    {0x51a89423u, 0x00000000u, 0x00000000u},
    {0x52a91474u, 0x000000a6u, 0x00000000u},
    {0x5429d4d4u, 0x000000a9u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x572b5583u, 0x00000000u, 0x00000000u},
    {0x58ac15e3u, 0x00000000u, 0x00000000u},
    {0x59ac9634u, 0x000000b4u, 0x00000000u},
    {0x5b2d5694u, 0x000000b7u, 0x00000000u},
    {0x5cae16f5u, 0x000176bau, 0x00000000u},
    {0x5eaf1774u, 0x000000beu, 0x00000000u},
    {0x602fd7d5u, 0x000184c1u, 0x00000000u},
    {0x6230d855u, 0x00018cc5u, 0x00000000u},
    {0x6431d8d4u, 0x000000c9u, 0x00000000u},
    {0x65b29935u, 0x00019accu, 0x00000000u},
    {0x67b399b4u, 0x000000d0u, 0x00000000u},
    {0x69345a14u, 0x000000d3u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x02810063u, 0x00000000u, 0x00000000u},
    {0x038180b3u, 0x00000000u, 0x00000000u},
    {0x05024103u, 0x00000000u, 0x00000000u},
    {0x0602c154u, 0x0000000du, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x09044203u, 0x00000000u, 0x00000000u},
    {0x0a04c254u, 0x00000015u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x0d064303u, 0x00000000u, 0x00000000u},
    {0x0e06c354u, 0x0000001du, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x11084403u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x138984a3u, 0x00000000u, 0x00000000u},
    {0x150a4503u, 0x00000000u, 0x00000000u},
    {0x160ac554u, 0x0000002du, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f4u, 0x00000032u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b8d86a3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f4u, 0x0000003au, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x1f8f87a3u, 0x00000000u, 0x00000000u},
    {0x209007f4u, 0x00000042u, 0x00000000u},
    {0x22910864u, 0x00000046u, 0x00000000u},
    {0x2411c8d4u, 0x00000049u, 0x00000000u},
    {0x25928934u, 0x0000004cu, 0x00000000u},
    {0x279389a4u, 0x00000050u, 0x00000000u},
    {0x29144a14u, 0x00000053u, 0x00000000u},
    {0x2b154a84u, 0x00000057u, 0x00000000u},
    {0x2c960af4u, 0x0000005au, 0x00000000u},
    {0x2e970b63u, 0x00000000u, 0x00000000u},
    {0x3017cbc4u, 0x00000061u, 0x00000000u},
    {0x31988c34u, 0x00000064u, 0x00000000u},
    {0x33194c96u, 0x01a4d067u, 0x00000000u},
    {0x359a8d35u, 0x0000da6cu, 0x00000000u},
    {0x381bcdc6u, 0x01cce471u, 0x00000000u},
    {0x3a9d0e75u, 0x0000ee76u, 0x00000000u},
    {0x3d1e4f06u, 0x01f4f87bu, 0x00000000u},
    {0x3f9f8fb5u, 0x00010280u, 0x00000000u},
    {0x42a11065u, 0x00010e86u, 0x00000000u},
    {0x44a210f5u, 0x0001168au, 0x00000000u},
    {0x47235185u, 0x00012290u, 0x00000000u},
    {0x49a49235u, 0x00012a94u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x4ca612f4u, 0x0000009au, 0x00000000u},
    {0x4e26d354u, 0x0000009du, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x51285403u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x53a994a3u, 0x00000000u, 0x00000000u},
    {0x54aa14f4u, 0x000000aau, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x57ab95a3u, 0x00000000u, 0x00000000u},
    {0x592c5603u, 0x00000000u, 0x00000000u},
    {0x5a2cd654u, 0x000000b5u, 0x00000000u},
    {0x5bad96b4u, 0x000000b8u, 0x00000000u},
    {0x5d2e5714u, 0x000000bbu, 0x00000000u},
    {0x5f2f5784u, 0x000000bfu, 0x00000000u},
    {0x61305803u, 0x00000000u, 0x00000000u},
    {0x62b11863u, 0x00000000u, 0x00000000u},
    {0x6431d8c4u, 0x000000c9u, 0x00000000u},
    {0x6632d944u, 0x000000cdu, 0x00000000u},
    {0x67b399b4u, 0x000000d0u, 0x00000000u},
    {0x69345a14u, 0x000000d3u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x038180a3u, 0x00000000u, 0x00000000u},
    {0x048200f4u, 0x0000000au, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x078381a3u, 0x00000000u, 0x00000000u},
    {0x088401f4u, 0x00000012u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x0b8582a3u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x0e06c343u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x108803e3u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x13094483u, 0x00000000u, 0x00000000u},
    {0x1409c4d4u, 0x00000029u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x170b4583u, 0x00000000u, 0x00000000u},
    {0x180bc5d4u, 0x00000031u, 0x00000000u},
    {0x198c8633u, 0x00000000u, 0x00000000u},
    {0x1a8d0673u, 0x00000000u, 0x00000000u},
    {0x1c0dc6c3u, 0x00000000u, 0x00000000u},
    {0x1d0e4714u, 0x0000003bu, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x200fc7c3u, 0x00000000u, 0x00000000u},
    {0x21104814u, 0x00000043u, 0x00000000u},
    {0x22910874u, 0x00000046u, 0x00000000u},
    {0x249208e4u, 0x0000004au, 0x00000000u},
    {0x2612c954u, 0x0000004du, 0x00000000u},
    {0x279389b4u, 0x00000050u, 0x00000000u},
    {0x29948a24u, 0x00000054u, 0x00000000u},
    {0x2b154a94u, 0x00000057u, 0x00000000u},
    {0x2c960af4u, 0x0000005au, 0x00000000u},
    {0x2e970b64u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c34u, 0x00000064u, 0x00000000u},
    {0x33194c95u, 0x0000d067u, 0x00000000u},
    {0x351a4d16u, 0x01b4d86bu, 0x00000000u},
    {0x379b8db6u, 0x01c8e270u, 0x00000000u},
    {0x3a9d0e65u, 0x0000ee76u, 0x00000000u},
    {0x3c9e0ef6u, 0x01f0f67au, 0x00000000u},
    {0x3f1f4f96u, 0x0205007fu, 0x00000000u},
    {0x4220d044u, 0x00000085u, 0x00000000u},
    {0x4421d0c5u, 0x00011489u, 0x00000000u},
    {0x4622d156u, 0x023d1c8du, 0x00000000u},
    {0x49245205u, 0x00012893u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x4ca612e3u, 0x00000000u, 0x00000000u},
    {0x4da69334u, 0x0000009cu, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x51a89423u, 0x00000000u, 0x00000000u},
    {0x52a91474u, 0x000000a6u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x552a5514u, 0x000000abu, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x582bd5c3u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x5aad1664u, 0x000000b6u, 0x00000000u},
    {0x5c2dd6d4u, 0x000000b9u, 0x00000000u},
    {0x5dae9734u, 0x000000bcu, 0x00000000u},
    {0x5f2f5794u, 0x000000bfu, 0x00000000u},
    {0x61305804u, 0x000000c3u, 0x00000000u},
    {0x62b11874u, 0x000000c6u, 0x00000000u},
    {0x64b218e4u, 0x000000cau, 0x00000000u},
    {0x6632d954u, 0x000000cdu, 0x00000000u},
    {0x6833d9c3u, 0x00000000u, 0x00000000u},
    {0x69b49a23u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x038180a3u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x0602c143u, 0x00000000u, 0x00000000u},
    {0x07034194u, 0x0000000fu, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x0b054283u, 0x00000000u, 0x00000000u},
    {0x0c05c2d3u, 0x00000000u, 0x00000000u},
    {0x0d868323u, 0x00000000u, 0x00000000u},
    {0x0e870374u, 0x0000001eu, 0x00000000u},
    {0x1007c3d3u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x12890463u, 0x00000000u, 0x00000000u},
    {0x138984b3u, 0x00000000u, 0x00000000u},
    {0x150a4503u, 0x00000000u, 0x00000000u},
    {0x160ac554u, 0x0000002du, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x1a0cc643u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x1c8e06e3u, 0x00000000u, 0x00000000u},
    {0x1d8e8734u, 0x0000003cu, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21908824u, 0x00000044u, 0x00000000u},
    {0x23114894u, 0x00000047u, 0x00000000u},
    {0x249208f4u, 0x0000004au, 0x00000000u},
    {0x26930964u, 0x0000004eu, 0x00000000u},
    {0x2813c9d4u, 0x00000051u, 0x00000000u},
    {0x29948a34u, 0x00000054u, 0x00000000u},
    {0x2b154a94u, 0x00000057u, 0x00000000u},
    {0x2d164b04u, 0x0000005bu, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x33194c85u, 0x0000d067u, 0x00000000u},
    {0x351a4d15u, 0x0000d86bu, 0x00000000u},
    {0x379b8da5u, 0x0000e270u, 0x00000000u},
    {0x399c8e36u, 0x01d8ea74u, 0x00000000u},
    {0x3c1dced5u, 0x0000f479u, 0x00000000u},
    {0x3e9f0f65u, 0x0000fe7eu, 0x00000000u},
    {0x41205005u, 0x00010883u, 0x00000000u},
    {0x43a190a5u, 0x00011288u, 0x00000000u},
    {0x45a29136u, 0x02391a8cu, 0x00000000u},
    {0x4823d1d5u, 0x00012491u, 0x00000000u},
    {0x4a24d254u, 0x00000095u, 0x00000000u},
    {0x4ba592b4u, 0x00000098u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x4ea71363u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x51285403u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x53295494u, 0x000000a7u, 0x00000000u},
    {0x54aa14f3u, 0x00000000u, 0x00000000u},
    {0x562ad543u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x58ac15e3u, 0x00000000u, 0x00000000u},
    {0x59ac9633u, 0x00000000u, 0x00000000u},
    {0x5aad1674u, 0x000000b6u, 0x00000000u},
    {0x5cae16e4u, 0x000000bau, 0x00000000u},
    {0x5e2ed754u, 0x000000bdu, 0x00000000u},
    {0x602fd7c3u, 0x00000000u, 0x00000000u},
    {0x61b09824u, 0x000000c4u, 0x00000000u},
    {0x63315894u, 0x000000c7u, 0x00000000u},
    {0x64b218f4u, 0x000000cau, 0x00000000u},
    {0x6632d954u, 0x000000cdu, 0x00000000u},
    {0x6833d9c3u, 0x00000000u, 0x00000000u},
    {0x69b49a23u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c054u, 0x00000005u, 0x00000000u},
    {0x038180b3u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x0602c143u, 0x00000000u, 0x00000000u},
    {0x07034193u, 0x00000000u, 0x00000000u},
    {0x088401e3u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x0c05c2c3u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0e870363u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x108803f3u, 0x00000000u, 0x00000000u},
    {0x1208c443u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x148a04e3u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x180bc5c3u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x1a8d0663u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f4u, 0x0000003au, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x1f8f87a3u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x2210c843u, 0x00000000u, 0x00000000u},
    {0x239188a3u, 0x00000000u, 0x00000000u},
    {0x25124904u, 0x0000004bu, 0x00000000u},
    {0x26930974u, 0x0000004eu, 0x00000000u},
    {0x2813c9d4u, 0x00000051u, 0x00000000u},
    {0x29948a34u, 0x00000054u, 0x00000000u},
    {0x2b958aa3u, 0x00000000u, 0x00000000u},
    {0x2d164b03u, 0x00000000u, 0x00000000u},
    {0x2e970b64u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c75u, 0x0000ce66u, 0x00000000u},
    {0x351a4d05u, 0x0000d86bu, 0x00000000u},
    {0x371b4d96u, 0x01c4e06fu, 0x00000000u},
    {0x3a1cce44u, 0x00000075u, 0x00000000u},
    {0x3c1dcec5u, 0x0000f479u, 0x00000000u},
    {0x3e9f0f64u, 0x0000007eu, 0x00000000u},
    {0x40a00fe5u, 0x00010682u, 0x00000000u},
    {0x43215084u, 0x00000087u, 0x00000000u},
    {0x45225105u, 0x0001188bu, 0x00000000u},
    {0x47a391a4u, 0x00000090u, 0x00000000u},
    {0x49a49224u, 0x00000094u, 0x00000000u},
    {0x4b255294u, 0x00000097u, 0x00000000u},
    {0x4ca612f3u, 0x00000000u, 0x00000000u},
    {0x4da69334u, 0x0000009cu, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x51a89423u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x5429d4c3u, 0x00000000u, 0x00000000u},
    {0x552a5513u, 0x00000000u, 0x00000000u},
    {0x56ab1563u, 0x00000000u, 0x00000000u},
    {0x57ab95b3u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x5a2cd643u, 0x00000000u, 0x00000000u},
    {0x5b2d5694u, 0x000000b7u, 0x00000000u},
    {0x5cae16f4u, 0x000000bau, 0x00000000u},
    {0x5eaf1763u, 0x00000000u, 0x00000000u},
    {0x602fd7c3u, 0x00000000u, 0x00000000u},
    {0x61b09824u, 0x000000c4u, 0x00000000u},
    {0x63315894u, 0x000000c7u, 0x00000000u},
    {0x64b218f4u, 0x000000cau, 0x00000000u},
    {0x6632d954u, 0x000000cdu, 0x00000000u},
    {0x6833d9c3u, 0x00000000u, 0x00000000u},
    {0x69b49a23u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c054u, 0x00000005u, 0x00000000u},
    {0x038180b3u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x07034183u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0a850263u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x0d064303u, 0x00000000u, 0x00000000u},
    {0x0e06c353u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x000803e2u, 0x00000000u, 0x00000000u},
    {0x11888423u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x138984b4u, 0x00000028u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x160ac554u, 0x0000002du, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x198c8633u, 0x00000000u, 0x00000000u},
    {0x1b0d4683u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x1e8f0763u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x209007f4u, 0x00000042u, 0x00000000u},
    {0x2210c854u, 0x00000045u, 0x00000000u},
    {0x239188b4u, 0x00000048u, 0x00000000u},
    {0x25124914u, 0x0000004bu, 0x00000000u},
    {0x26930974u, 0x0000004eu, 0x00000000u},
    {0x2813c9d4u, 0x00000051u, 0x00000000u},
    {0x2a14ca44u, 0x00000055u, 0x00000000u},
    {0x2b958ab4u, 0x00000058u, 0x00000000u},
    {0x2d164b14u, 0x0000005bu, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c75u, 0x0000ce66u, 0x00000000u},
    {0x349a0cf6u, 0x01b0d66au, 0x00000000u},
    {0x371b4d95u, 0x0000e06fu, 0x00000000u},
    {0x391c4e16u, 0x01d4e873u, 0x00000000u},
    {0x3b9d8eb5u, 0x0000f278u, 0x00000000u},
    {0x3d9e8f35u, 0x0000fa7cu, 0x00000000u},
    {0x401fcfc5u, 0x00010481u, 0x00000000u},
    {0x42a11064u, 0x00000086u, 0x00000000u},
    {0x44a210e4u, 0x0000008au, 0x00000000u},
    {0x46a31165u, 0x00011e8eu, 0x00000000u},
    {0x48a411f6u, 0x02512692u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x4da69323u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x51285403u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x54aa14e3u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x572b5583u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x5a2cd654u, 0x000000b5u, 0x00000000u},
    {0x5bad96b4u, 0x000000b8u, 0x00000000u},
    {0x5d2e5714u, 0x000000bbu, 0x00000000u},
    {0x5eaf1774u, 0x000000beu, 0x00000000u},
    {0x602fd7d4u, 0x000000c1u, 0x00000000u},
    {0x6230d843u, 0x00000000u, 0x00000000u},
    {0x63b198a3u, 0x00000000u, 0x00000000u},
    {0x65325903u, 0x00000000u, 0x00000000u},
    {0x66b31963u, 0x00000000u, 0x00000000u},
    {0x6833d9c3u, 0x00000000u, 0x00000000u},
    {0x69b49a23u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x000180a2u, 0x00000000u, 0x00000000u},
    {0x048200e3u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x0803c1c3u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0c8602e3u, 0x00000000u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x0f074383u, 0x00000000u, 0x00000000u},
    {0x1007c3d3u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x13094494u, 0x00000027u, 0x00000000u},
    {0x148a04f3u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x000b4582u, 0x00000000u, 0x00000000u},
    {0x180bc5c3u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b8d86a3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f3u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x200fc7c3u, 0x00000000u, 0x00000000u},
    {0x21104814u, 0x00000043u, 0x00000000u},
    {0x22910874u, 0x00000046u, 0x00000000u},
    {0x2411c8d4u, 0x00000049u, 0x00000000u},
    {0x25928934u, 0x0000004cu, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x289409e4u, 0x00000052u, 0x00000000u},
    {0x2a14ca54u, 0x00000055u, 0x00000000u},
    {0x2b958ab3u, 0x00000000u, 0x00000000u},
    {0x2d164b04u, 0x0000005bu, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c75u, 0x0000ce66u, 0x00000000u},
    {0x349a0cf5u, 0x0000d66au, 0x00000000u},
    {0x369b0d75u, 0x0000de6eu, 0x00000000u},
    {0x391c4e04u, 0x00000073u, 0x00000000u},
    {0x3b1d4e85u, 0x0000f077u, 0x00000000u},
    {0x3d9e8f25u, 0x0000fa7cu, 0x00000000u},
    {0x3f9f8fb5u, 0x00010280u, 0x00000000u},
    {0x41a09035u, 0x00010a84u, 0x00000000u},
    {0x4421d0c4u, 0x00000089u, 0x00000000u},
    {0x4622d145u, 0x00011c8du, 0x00000000u},
    {0x4823d1d5u, 0x00012491u, 0x00000000u},
    {0x4a24d254u, 0x00000095u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x4d265303u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x5027d3d4u, 0x000000a1u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x53a994b3u, 0x00000000u, 0x00000000u},
    {0x552a5503u, 0x00000000u, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x002c15e2u, 0x00000000u, 0x00000000u},
    {0x59ac9623u, 0x00000000u, 0x00000000u},
    {0x5aad1674u, 0x000000b6u, 0x00000000u},
    {0x5c2dd6d3u, 0x00000000u, 0x00000000u},
    {0x5dae9724u, 0x000000bcu, 0x00000000u},
    {0x5f2f5794u, 0x000000bfu, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x6230d844u, 0x000000c5u, 0x00000000u},
    {0x63b198b4u, 0x000000c8u, 0x00000000u},
    {0x65325914u, 0x000000cbu, 0x00000000u},
    {0x66b31974u, 0x000000ceu, 0x00000000u},
    {0x6833d9d4u, 0x000000d1u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x048200e3u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x09044203u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0c05c2d3u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0e870363u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x108803f3u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x13094483u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x160ac553u, 0x00000000u, 0x00000000u},
    {0x178b85a3u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x198c8633u, 0x00000000u, 0x00000000u},
    {0x1a8d0673u, 0x00000000u, 0x00000000u},
    {0x1c0dc6c3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21908823u, 0x00000000u, 0x00000000u},
    {0x23114883u, 0x00000000u, 0x00000000u},
    {0x249208e3u, 0x00000000u, 0x00000000u},
    {0x25928934u, 0x0000004cu, 0x00000000u},
    {0x27134994u, 0x0000004fu, 0x00000000u},
    {0x289409f4u, 0x00000052u, 0x00000000u},
    {0x2a14ca54u, 0x00000055u, 0x00000000u},
    {0x2c15cac3u, 0x00000000u, 0x00000000u},
    {0x2d164b14u, 0x0000005bu, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c74u, 0x00000066u, 0x00000000u},
    {0x349a0ce5u, 0x0000d66au, 0x00000000u},
    {0x369b0d75u, 0x0000de6eu, 0x00000000u},
    {0x389c0df5u, 0x0000e672u, 0x00000000u},
    {0x3b1d4e84u, 0x00000077u, 0x00000000u},
    {0x3d1e4f04u, 0x0000007bu, 0x00000000u},
    {0x3f1f4f84u, 0x0000007fu, 0x00000000u},
    {0x41205005u, 0x00010883u, 0x00000000u},
    {0x43215095u, 0x00011087u, 0x00000000u},
    {0x45a29124u, 0x0000008cu, 0x00000000u},
    {0x47a391a5u, 0x00012290u, 0x00000000u},
    {0x4a24d243u, 0x00000000u, 0x00000000u},
    {0x002592a2u, 0x00000000u, 0x00000000u},
    {0x4ca612e3u, 0x00000000u, 0x00000000u},
    {0x4da69333u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x00285402u, 0x00000000u, 0x00000000u},
    {0x5228d443u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x55aa9523u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x57ab95b3u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x59ac9633u, 0x00000000u, 0x00000000u},
    {0x5b2d5684u, 0x000000b7u, 0x00000000u},
    {0x5cae16f3u, 0x00000000u, 0x00000000u},
    {0x5e2ed743u, 0x00000000u, 0x00000000u},
    {0x5f2f5794u, 0x000000bfu, 0x00000000u},
    {0x61305803u, 0x00000000u, 0x00000000u},
    {0x62b11863u, 0x00000000u, 0x00000000u},
    {0x63b198b4u, 0x000000c8u, 0x00000000u},
    {0x65325914u, 0x000000cbu, 0x00000000u},
    {0x66b31974u, 0x000000ceu, 0x00000000u},
    {0x6833d9d4u, 0x000000d1u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x05828123u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x0005c2c2u, 0x00000000u, 0x00000000u},
    {0x0d064303u, 0x00000000u, 0x00000000u},
    {0x0e06c353u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x000803e2u, 0x00000000u, 0x00000000u},
    {0x00088422u, 0x00000000u, 0x00000000u},
    {0x12890463u, 0x00000000u, 0x00000000u},
    {0x138984b3u, 0x00000000u, 0x00000000u},
    {0x148a04f3u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x000b4582u, 0x00000000u, 0x00000000u},
    {0x180bc5c3u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x1e8f0763u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x21908834u, 0x00000044u, 0x00000000u},
    {0x239188a3u, 0x00000000u, 0x00000000u},
    {0x249208f4u, 0x0000004au, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x279389a3u, 0x00000000u, 0x00000000u},
    {0x29144a03u, 0x00000000u, 0x00000000u},
    {0x2a950a63u, 0x00000000u, 0x00000000u},
    {0x2b958ab4u, 0x00000058u, 0x00000000u},
    {0x2d164b14u, 0x0000005bu, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c74u, 0x00000066u, 0x00000000u},
    {0x349a0ce4u, 0x0000006au, 0x00000000u},
    {0x361acd55u, 0x0000dc6du, 0x00000000u},
    {0x389c0de4u, 0x00000072u, 0x00000000u},
    {0x3a9d0e65u, 0x0000ee76u, 0x00000000u},
    {0x3c9e0ef5u, 0x0000f67au, 0x00000000u},
    {0x3e9f0f75u, 0x0000fe7eu, 0x00000000u},
    {0x40a00ff5u, 0x00010682u, 0x00000000u},
    {0x43215084u, 0x00000087u, 0x00000000u},
    {0x45225104u, 0x0000008bu, 0x00000000u},
    {0x47235184u, 0x0000008fu, 0x00000000u},
    {0x48a411f5u, 0x00012692u, 0x00000000u},
    {0x4b255283u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x51a89423u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x53a994b3u, 0x00000000u, 0x00000000u},
    {0x54aa14f3u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x572b5583u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x5a2cd653u, 0x00000000u, 0x00000000u},
    {0x5b2d5694u, 0x000000b7u, 0x00000000u},
    {0x5cae16f4u, 0x000000bau, 0x00000000u},
    {0x5e2ed754u, 0x000000bdu, 0x00000000u},
    {0x5faf97b4u, 0x000000c0u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x62b11863u, 0x00000000u, 0x00000000u},
    {0x6431d8c3u, 0x00000000u, 0x00000000u},
    {0x65b29923u, 0x00000000u, 0x00000000u},
    {0x66b31974u, 0x000000ceu, 0x00000000u},
    {0x6833d9d4u, 0x000000d1u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x07034194u, 0x0000000fu, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x00074382u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x00084402u, 0x00000000u, 0x00000000u},
    {0x1208c443u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x160ac553u, 0x00000000u, 0x00000000u},
    {0x000b85a2u, 0x00000000u, 0x00000000u},
    {0x000c05e2u, 0x00000000u, 0x00000000u},
    {0x198c8623u, 0x00000000u, 0x00000000u},
    {0x1a8d0673u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f3u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x21104803u, 0x00000000u, 0x00000000u},
    {0x2210c854u, 0x00000045u, 0x00000000u},
    {0x239188b3u, 0x00000000u, 0x00000000u},
    {0x25124903u, 0x00000000u, 0x00000000u},
    {0x2612c954u, 0x0000004du, 0x00000000u},
    {0x279389b4u, 0x00000050u, 0x00000000u},
    {0x29144a14u, 0x00000053u, 0x00000000u},
    {0x2a950a73u, 0x00000000u, 0x00000000u},
    {0x2c15cac3u, 0x00000000u, 0x00000000u},
    {0x2d164b14u, 0x0000005bu, 0x00000000u},
    {0x2e970b73u, 0x00000000u, 0x00000000u},
    {0x3017cbc3u, 0x00000000u, 0x00000000u},
    {0x00188c22u, 0x00000000u, 0x00000000u},
    {0x32990c64u, 0x00000066u, 0x00000000u},
    {0x3419ccd5u, 0x0000d469u, 0x00000000u},
    {0x361acd55u, 0x0000dc6du, 0x00000000u},
    {0x381bcdd5u, 0x0000e471u, 0x00000000u},
    {0x3a9d0e64u, 0x00000076u, 0x00000000u},
    {0x3c9e0ee4u, 0x0000007au, 0x00000000u},
    {0x3e1ecf55u, 0x0000fc7du, 0x00000000u},
    {0x401fcfd5u, 0x00010481u, 0x00000000u},
    {0x4220d055u, 0x00010c85u, 0x00000000u},
    {0x4421d0d5u, 0x00011489u, 0x00000000u},
    {0x46a31164u, 0x0000008eu, 0x00000000u},
    {0x48a411e4u, 0x00000092u, 0x00000000u},
    {0x4aa51263u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x4ca612f3u, 0x00000000u, 0x00000000u},
    {0x4da69333u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x51285403u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x552a5513u, 0x00000000u, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x572b5594u, 0x000000afu, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x59ac9633u, 0x00000000u, 0x00000000u},
    {0x5aad1673u, 0x00000000u, 0x00000000u},
    {0x5bad96b4u, 0x000000b8u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x5eaf1763u, 0x00000000u, 0x00000000u},
    {0x602fd7c3u, 0x00000000u, 0x00000000u},
    {0x61b09823u, 0x00000000u, 0x00000000u},
    {0x62b11874u, 0x000000c6u, 0x00000000u},
    {0x6431d8d4u, 0x000000c9u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x67335983u, 0x00000000u, 0x00000000u},
    {0x68b419e3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x07034193u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0c05c2d4u, 0x00000019u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x108803f3u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x138984b3u, 0x00000000u, 0x00000000u},
    {0x148a04f3u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f4u, 0x00000032u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21104814u, 0x00000043u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x2411c8c3u, 0x00000000u, 0x00000000u},
    {0x25124914u, 0x0000004bu, 0x00000000u},
    {0x26930973u, 0x00000000u, 0x00000000u},
    {0x2813c9c3u, 0x00000000u, 0x00000000u},
    {0x29144a14u, 0x00000053u, 0x00000000u},
    {0x2a950a74u, 0x00000056u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d968b23u, 0x00000000u, 0x00000000u},
    {0x2f174b83u, 0x00000000u, 0x00000000u},
    {0x3017cbd4u, 0x00000061u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x3419ccc4u, 0x00000069u, 0x00000000u},
    {0x361acd44u, 0x0000006du, 0x00000000u},
    {0x379b8db5u, 0x0000e270u, 0x00000000u},
    {0x399c8e35u, 0x0000ea74u, 0x00000000u},
    {0x3b9d8eb5u, 0x0000f278u, 0x00000000u},
    {0x3d9e8f35u, 0x0000fa7cu, 0x00000000u},
    {0x3f9f8fb4u, 0x00000080u, 0x00000000u},
    {0x41a09025u, 0x00010a84u, 0x00000000u},
    {0x43a190b4u, 0x00000088u, 0x00000000u},
    {0x45a29125u, 0x00011a8cu, 0x00000000u},
    {0x47a391b4u, 0x00000090u, 0x00000000u},
    {0x49a49224u, 0x00000094u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x00271362u, 0x00000000u, 0x00000000u},
    {0x002793a2u, 0x00000000u, 0x00000000u},
    {0x50a813e3u, 0x00000000u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x53a994b3u, 0x00000000u, 0x00000000u},
    {0x54aa14f3u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x57ab95b3u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x59ac9633u, 0x00000000u, 0x00000000u},
    {0x5b2d5683u, 0x00000000u, 0x00000000u},
    {0x5c2dd6d4u, 0x000000b9u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5f2f5783u, 0x00000000u, 0x00000000u},
    {0x60b017e3u, 0x00000000u, 0x00000000u},
    {0x61b09834u, 0x000000c4u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x6431d8d4u, 0x000000c9u, 0x00000000u},
    {0x65b29934u, 0x000000ccu, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x68b419e3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x07034193u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0c05c2d3u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0e06c353u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x1007c3d3u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x160ac553u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x000c8622u, 0x00000000u, 0x00000000u},
    {0x000d0662u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x000f0762u, 0x00000000u, 0x00000000u},
    {0x000f87a2u, 0x00000000u, 0x00000000u},
    {0x001007e2u, 0x00000000u, 0x00000000u},
    {0x21908823u, 0x00000000u, 0x00000000u},
    {0x22910874u, 0x00000046u, 0x00000000u},
    {0x2411c8d4u, 0x00000049u, 0x00000000u},
    {0x25928933u, 0x00000000u, 0x00000000u},
    {0x27134983u, 0x00000000u, 0x00000000u},
    {0x2813c9d3u, 0x00000000u, 0x00000000u},
    {0x29948a23u, 0x00000000u, 0x00000000u},
    {0x2a950a74u, 0x00000056u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d968b23u, 0x00000000u, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00188c22u, 0x00000000u, 0x00000000u},
    {0x32990c63u, 0x00000000u, 0x00000000u},
    {0x3419ccc4u, 0x00000069u, 0x00000000u},
    {0x359a8d35u, 0x0000da6cu, 0x00000000u},
    {0x379b8db5u, 0x0000e270u, 0x00000000u},
    {0x399c8e34u, 0x00000074u, 0x00000000u},
    {0x3b9d8ea4u, 0x00000078u, 0x00000000u},
    {0x3d1e4f15u, 0x0000f87bu, 0x00000000u},
    {0x3f9f8fa4u, 0x00000080u, 0x00000000u},
    {0x41205015u, 0x00010883u, 0x00000000u},
    {0x43215094u, 0x00000087u, 0x00000000u},
    {0x45225105u, 0x0001188bu, 0x00000000u},
    {0x47235195u, 0x0001208fu, 0x00000000u},
    {0x49245214u, 0x00000093u, 0x00000000u},
    {0x00255282u, 0x00000000u, 0x00000000u},
    {0x4c25d2c3u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x0027d3c2u, 0x00000000u, 0x00000000u},
    {0x51285403u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x552a5513u, 0x00000000u, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x5a2cd653u, 0x00000000u, 0x00000000u},
    {0x5b2d5693u, 0x00000000u, 0x00000000u},
    {0x5cae16e3u, 0x00000000u, 0x00000000u},
    {0x5dae9734u, 0x000000bcu, 0x00000000u},
    {0x5f2f5793u, 0x00000000u, 0x00000000u},
    {0x60b017e3u, 0x00000000u, 0x00000000u},
    {0x61b09834u, 0x000000c4u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x64b218e3u, 0x00000000u, 0x00000000u},
    {0x65b29934u, 0x000000ccu, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x68b419e3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0002c152u, 0x00000000u, 0x00000000u},
    {0x07034183u, 0x00000000u, 0x00000000u},
    {0x0003c1d2u, 0x00000000u, 0x00000000u},
    {0x00044202u, 0x00000000u, 0x00000000u},
    {0x0004c242u, 0x00000000u, 0x00000000u},
    {0x00054282u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x00064302u, 0x00000000u, 0x00000000u},
    {0x0e06c343u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x0007c3d2u, 0x00000000u, 0x00000000u},
    {0x00084402u, 0x00000000u, 0x00000000u},
    {0x0008c442u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x0009c4c2u, 0x00000000u, 0x00000000u},
    {0x000a4502u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x198c8633u, 0x00000000u, 0x00000000u},
    {0x1a8d0673u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f3u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x21908834u, 0x00000044u, 0x00000000u},
    {0x23114893u, 0x00000000u, 0x00000000u},
    {0x249208e3u, 0x00000000u, 0x00000000u},
    {0x25928933u, 0x00000000u, 0x00000000u},
    {0x27134983u, 0x00000000u, 0x00000000u},
    {0x289409e3u, 0x00000000u, 0x00000000u},
    {0x29948a33u, 0x00000000u, 0x00000000u},
    {0x2b154a83u, 0x00000000u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d968b23u, 0x00000000u, 0x00000000u},
    {0x2e970b73u, 0x00000000u, 0x00000000u},
    {0x3017cbc3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc54u, 0x00000065u, 0x00000000u},
    {0x33998cb4u, 0x00000068u, 0x00000000u},
    {0x359a8d25u, 0x0000da6cu, 0x00000000u},
    {0x379b8db4u, 0x00000070u, 0x00000000u},
    {0x399c8e24u, 0x00000074u, 0x00000000u},
    {0x3b9d8ea4u, 0x00000078u, 0x00000000u},
    {0x3d1e4f14u, 0x0000007bu, 0x00000000u},
    {0x3f1f4f84u, 0x0000007fu, 0x00000000u},
    {0x40a00ff5u, 0x00010682u, 0x00000000u},
    {0x43215084u, 0x00000087u, 0x00000000u},
    {0x44a210f5u, 0x0001168au, 0x00000000u},
    {0x46a31174u, 0x0000008eu, 0x00000000u},
    {0x48a411e4u, 0x00000092u, 0x00000000u},
    {0x4a24d254u, 0x00000095u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x002612f2u, 0x00000000u, 0x00000000u},
    {0x00269322u, 0x00000000u, 0x00000000u},
    {0x00271362u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x50a813e3u, 0x00000000u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x53a994b3u, 0x00000000u, 0x00000000u},
    {0x002a14f2u, 0x00000000u, 0x00000000u},
    {0x002a9522u, 0x00000000u, 0x00000000u},
    {0x002b1562u, 0x00000000u, 0x00000000u},
    {0x002b95a2u, 0x00000000u, 0x00000000u},
    {0x002c15e2u, 0x00000000u, 0x00000000u},
    {0x002c9622u, 0x00000000u, 0x00000000u},
    {0x5a2cd653u, 0x00000000u, 0x00000000u},
    {0x5bad96a3u, 0x00000000u, 0x00000000u},
    {0x5cae16f4u, 0x000000bau, 0x00000000u},
    {0x5e2ed754u, 0x000000bdu, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x60b017f4u, 0x000000c2u, 0x00000000u},
    {0x6230d853u, 0x00000000u, 0x00000000u},
    {0x63b198a3u, 0x00000000u, 0x00000000u},
    {0x64b218f3u, 0x00000000u, 0x00000000u},
    {0x6632d943u, 0x00000000u, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x68b419e3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0001c0d2u, 0x00000000u, 0x00000000u},
    {0x00024102u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x0004c242u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x108803f3u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x000984b2u, 0x00000000u, 0x00000000u},
    {0x000a04e2u, 0x00000000u, 0x00000000u},
    {0x000a8522u, 0x00000000u, 0x00000000u},
    {0x160ac553u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21104813u, 0x00000000u, 0x00000000u},
    {0x2210c853u, 0x00000000u, 0x00000000u},
    {0x239188a3u, 0x00000000u, 0x00000000u},
    {0x249208f3u, 0x00000000u, 0x00000000u},
    {0x2612c943u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x289409e3u, 0x00000000u, 0x00000000u},
    {0x29948a34u, 0x00000054u, 0x00000000u},
    {0x2b154a93u, 0x00000000u, 0x00000000u},
    {0x2c15cad4u, 0x00000059u, 0x00000000u},
    {0x2d968b33u, 0x00000000u, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00188c22u, 0x00000000u, 0x00000000u},
    {0x32990c63u, 0x00000000u, 0x00000000u},
    {0x33998cb4u, 0x00000068u, 0x00000000u},
    {0x359a8d24u, 0x0000006cu, 0x00000000u},
    {0x371b4d94u, 0x0000006fu, 0x00000000u},
    {0x391c4e05u, 0x0000e873u, 0x00000000u},
    {0x3b1d4e94u, 0x00000077u, 0x00000000u},
    {0x3c9e0ef4u, 0x0000007au, 0x00000000u},
    {0x3e9f0f64u, 0x0000007eu, 0x00000000u},
    {0x40a00fe4u, 0x00000082u, 0x00000000u},
    {0x42a11064u, 0x00000086u, 0x00000000u},
    {0x4421d0d4u, 0x00000089u, 0x00000000u},
    {0x4622d144u, 0x0000008du, 0x00000000u},
    {0x47a391b4u, 0x00000090u, 0x00000000u},
    {0x49a49224u, 0x00000094u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x0027d3c2u, 0x00000000u, 0x00000000u},
    {0x51285403u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x00295492u, 0x00000000u, 0x00000000u},
    {0x0029d4c2u, 0x00000000u, 0x00000000u},
    {0x002a5502u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x57ab95b3u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x59ac9633u, 0x00000000u, 0x00000000u},
    {0x5aad1673u, 0x00000000u, 0x00000000u},
    {0x5c2dd6c3u, 0x00000000u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x5eaf1763u, 0x00000000u, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x61305803u, 0x00000000u, 0x00000000u},
    {0x6230d854u, 0x000000c5u, 0x00000000u},
    {0x63b198b3u, 0x00000000u, 0x00000000u},
    {0x64b218f4u, 0x000000cau, 0x00000000u},
    {0x6632d953u, 0x00000000u, 0x00000000u},
    {0x67335994u, 0x000000cfu, 0x00000000u},
    {0x68b419f3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x01004003u, 0x00000000u, 0x00000000u},
    {0x0200c053u, 0x00000000u, 0x00000000u},
    {0x00014092u, 0x00000000u, 0x00000000u},
    {0x0001c0c2u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x00048232u, 0x00000000u, 0x00000000u},
    {0x00050262u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0c05c2d3u, 0x00000000u, 0x00000000u},
    {0x00068322u, 0x00000000u, 0x00000000u},
    {0x00070362u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x1007c3d3u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x0009c4d2u, 0x00000000u, 0x00000000u},
    {0x000a4502u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x198c8633u, 0x00000000u, 0x00000000u},
    {0x1a8d0673u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f3u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x000f0772u, 0x00000000u, 0x00000000u},
    {0x000f87a2u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21104813u, 0x00000000u, 0x00000000u},
    {0x2210c854u, 0x00000045u, 0x00000000u},
    {0x239188b4u, 0x00000048u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x279389a3u, 0x00000000u, 0x00000000u},
    {0x289409f4u, 0x00000052u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b154a93u, 0x00000000u, 0x00000000u},
    {0x2c960ae3u, 0x00000000u, 0x00000000u},
    {0x2d968b33u, 0x00000000u, 0x00000000u},
    {0x2e970b74u, 0x0000005eu, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc54u, 0x00000065u, 0x00000000u},
    {0x33998cb4u, 0x00000068u, 0x00000000u},
    {0x351a4d15u, 0x0000d86bu, 0x00000000u},
    {0x371b4d94u, 0x0000006fu, 0x00000000u},
    {0x391c4e04u, 0x00000073u, 0x00000000u},
    {0x3a9d0e75u, 0x0000ee76u, 0x00000000u},
    {0x3c9e0ef4u, 0x0000007au, 0x00000000u},
    {0x3e1ecf54u, 0x0000007du, 0x00000000u},
    {0x401fcfc4u, 0x00000081u, 0x00000000u},
    {0x41a09035u, 0x00010a84u, 0x00000000u},
    {0x43a190b4u, 0x00000088u, 0x00000000u},
    {0x45a29124u, 0x0000008cu, 0x00000000u},
    {0x47a391a4u, 0x00000090u, 0x00000000u},
    {0x49245214u, 0x00000093u, 0x00000000u},
    {0x4aa51273u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x4ca612f3u, 0x00000000u, 0x00000000u},
    {0x4da69333u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x002793b2u, 0x00000000u, 0x00000000u},
    {0x50a813e3u, 0x00000000u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x002994b2u, 0x00000000u, 0x00000000u},
    {0x002a14e2u, 0x00000000u, 0x00000000u},
    {0x552a5513u, 0x00000000u, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x002cd652u, 0x00000000u, 0x00000000u},
    {0x5b2d5683u, 0x00000000u, 0x00000000u},
    {0x5c2dd6d4u, 0x000000b9u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5f2f5783u, 0x00000000u, 0x00000000u},
    {0x602fd7d3u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x62b11863u, 0x00000000u, 0x00000000u},
    {0x63b198b3u, 0x00000000u, 0x00000000u},
    {0x65325903u, 0x00000000u, 0x00000000u},
    {0x6632d953u, 0x00000000u, 0x00000000u},
    {0x67335994u, 0x000000cfu, 0x00000000u},
    {0x68b419f3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x0000c042u, 0x00000000u, 0x00000000u},
    {0x00014082u, 0x00000000u, 0x00000000u},
    {0x038180b3u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x000381b2u, 0x00000000u, 0x00000000u},
    {0x000401e2u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0c05c2d3u, 0x00000000u, 0x00000000u},
    {0x00064312u, 0x00000000u, 0x00000000u},
    {0x0006c342u, 0x00000000u, 0x00000000u},
    {0x00074382u, 0x00000000u, 0x00000000u},
    {0x0007c3c2u, 0x00000000u, 0x00000000u},
    {0x108803f3u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x138984b3u, 0x00000000u, 0x00000000u},
    {0x000a04f2u, 0x00000000u, 0x00000000u},
    {0x000a8522u, 0x00000000u, 0x00000000u},
    {0x160ac553u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x000d4682u, 0x00000000u, 0x00000000u},
    {0x000dc6c2u, 0x00000000u, 0x00000000u},
    {0x000e4702u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x2411c8c3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x26930963u, 0x00000000u, 0x00000000u},
    {0x279389b3u, 0x00000000u, 0x00000000u},
    {0x289409f4u, 0x00000052u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b154a93u, 0x00000000u, 0x00000000u},
    {0x2c960ae3u, 0x00000000u, 0x00000000u},
    {0x2d968b33u, 0x00000000u, 0x00000000u},
    {0x2f174b83u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc53u, 0x00000000u, 0x00000000u},
    {0x33998ca4u, 0x00000068u, 0x00000000u},
    {0x351a4d14u, 0x0000006bu, 0x00000000u},
    {0x371b4d84u, 0x0000006fu, 0x00000000u},
    {0x391c4e03u, 0x00000000u, 0x00000000u},
    {0x3a1cce55u, 0x0000ec75u, 0x00000000u},
    {0x3c1dced4u, 0x00000079u, 0x00000000u},
    {0x3e1ecf44u, 0x0000007du, 0x00000000u},
    {0x3f9f8fb5u, 0x00010280u, 0x00000000u},
    {0x41a09034u, 0x00000084u, 0x00000000u},
    {0x43215094u, 0x00000087u, 0x00000000u},
    {0x45225104u, 0x0000008bu, 0x00000000u},
    {0x46a31174u, 0x0000008eu, 0x00000000u},
    {0x48a411e4u, 0x00000092u, 0x00000000u},
    {0x00251262u, 0x00000000u, 0x00000000u},
    {0x002592a2u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x0028d442u, 0x00000000u, 0x00000000u},
    {0x00295482u, 0x00000000u, 0x00000000u},
    {0x0029d4c2u, 0x00000000u, 0x00000000u},
    {0x54aa14f3u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x57ab95b3u, 0x00000000u, 0x00000000u},
    {0x002c15f2u, 0x00000000u, 0x00000000u},
    {0x002c9622u, 0x00000000u, 0x00000000u},
    {0x5a2cd653u, 0x00000000u, 0x00000000u},
    {0x5bad96a3u, 0x00000000u, 0x00000000u},
    {0x5cae16f3u, 0x00000000u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5f2f5783u, 0x00000000u, 0x00000000u},
    {0x602fd7d4u, 0x000000c1u, 0x00000000u},
    {0x61b09833u, 0x00000000u, 0x00000000u},
    {0x62b11873u, 0x00000000u, 0x00000000u},
    {0x6431d8c3u, 0x00000000u, 0x00000000u},
    {0x65325913u, 0x00000000u, 0x00000000u},
    {0x6632d953u, 0x00000000u, 0x00000000u},
    {0x67b399a3u, 0x00000000u, 0x00000000u},
    {0x68b419f3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x0000c042u, 0x00000000u, 0x00000000u},
    {0x00014082u, 0x00000000u, 0x00000000u},
    {0x038180b3u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x00028132u, 0x00000000u, 0x00000000u},
    {0x00030162u, 0x00000000u, 0x00000000u},
    {0x000381a2u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0004c252u, 0x00000000u, 0x00000000u},
    {0x00054282u, 0x00000000u, 0x00000000u},
    {0x0005c2c2u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x000803f2u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x000a4512u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x000f4792u, 0x00000000u, 0x00000000u},
    {0x000fc7c2u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x21908834u, 0x00000044u, 0x00000000u},
    {0x23114893u, 0x00000000u, 0x00000000u},
    {0x2411c8d3u, 0x00000000u, 0x00000000u},
    {0x25124914u, 0x0000004bu, 0x00000000u},
    {0x26930973u, 0x00000000u, 0x00000000u},
    {0x279389b4u, 0x00000050u, 0x00000000u},
    {0x29144a13u, 0x00000000u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b958aa3u, 0x00000000u, 0x00000000u},
    {0x2c960af3u, 0x00000000u, 0x00000000u},
    {0x2d968b34u, 0x0000005cu, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc53u, 0x00000000u, 0x00000000u},
    {0x33998ca3u, 0x00000000u, 0x00000000u},
    {0x349a0cf5u, 0x0000d66au, 0x00000000u},
    {0x369b0d74u, 0x0000006eu, 0x00000000u},
    {0x389c0de4u, 0x00000072u, 0x00000000u},
    {0x3a1cce54u, 0x00000075u, 0x00000000u},
    {0x3b9d8eb5u, 0x0000f278u, 0x00000000u},
    {0x3d9e8f34u, 0x0000007cu, 0x00000000u},
    {0x3f1f4f94u, 0x0000007fu, 0x00000000u},
    {0x41205004u, 0x00000083u, 0x00000000u},
    {0x42a11075u, 0x00010e86u, 0x00000000u},
    {0x44a210f4u, 0x0000008au, 0x00000000u},
    {0x4622d154u, 0x0000008du, 0x00000000u},
    {0x4823d1c4u, 0x00000091u, 0x00000000u},
    {0x49a49234u, 0x00000094u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x0025d2d2u, 0x00000000u, 0x00000000u},
    {0x00265302u, 0x00000000u, 0x00000000u},
    {0x0026d342u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x002813f2u, 0x00000000u, 0x00000000u},
    {0x00289422u, 0x00000000u, 0x00000000u},
    {0x00291462u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x002a5512u, 0x00000000u, 0x00000000u},
    {0x002ad542u, 0x00000000u, 0x00000000u},
    {0x002b5582u, 0x00000000u, 0x00000000u},
    {0x002bd5c2u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x5aad1663u, 0x00000000u, 0x00000000u},
    {0x5bad96b3u, 0x00000000u, 0x00000000u},
    {0x5d2e5703u, 0x00000000u, 0x00000000u},
    {0x5e2ed753u, 0x00000000u, 0x00000000u},
    {0x002f97a2u, 0x00000000u, 0x00000000u},
    {0x60b017e3u, 0x00000000u, 0x00000000u},
    {0x61b09833u, 0x00000000u, 0x00000000u},
    {0x62b11874u, 0x000000c6u, 0x00000000u},
    {0x6431d8d3u, 0x00000000u, 0x00000000u},
    {0x65325913u, 0x00000000u, 0x00000000u},
    {0x6632d954u, 0x000000cdu, 0x00000000u},
    {0x67b399b3u, 0x00000000u, 0x00000000u},
    {0x68b419f3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x0000c042u, 0x00000000u, 0x00000000u},
    {0x02810073u, 0x00000000u, 0x00000000u},
    {0x038180b3u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x00030172u, 0x00000000u, 0x00000000u},
    {0x07034193u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x00044212u, 0x00000000u, 0x00000000u},
    {0x0004c242u, 0x00000000u, 0x00000000u},
    {0x00054282u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x00068332u, 0x00000000u, 0x00000000u},
    {0x00070362u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x1007c3d3u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x0008c452u, 0x00000000u, 0x00000000u},
    {0x00094482u, 0x00000000u, 0x00000000u},
    {0x138984b3u, 0x00000000u, 0x00000000u},
    {0x000a04f2u, 0x00000000u, 0x00000000u},
    {0x000a8522u, 0x00000000u, 0x00000000u},
    {0x000b0562u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x000d4682u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f3u, 0x00000000u, 0x00000000u},
    {0x000e8732u, 0x00000000u, 0x00000000u},
    {0x000f0762u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21104813u, 0x00000000u, 0x00000000u},
    {0x2210c853u, 0x00000000u, 0x00000000u},
    {0x23114893u, 0x00000000u, 0x00000000u},
    {0x249208e3u, 0x00000000u, 0x00000000u},
    {0x25928933u, 0x00000000u, 0x00000000u},
    {0x00134982u, 0x00000000u, 0x00000000u},
    {0x2813c9c3u, 0x00000000u, 0x00000000u},
    {0x29144a13u, 0x00000000u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b958aa3u, 0x00000000u, 0x00000000u},
    {0x2c960af3u, 0x00000000u, 0x00000000u},
    {0x0016cb42u, 0x00000000u, 0x00000000u},
    {0x2f174b83u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc53u, 0x00000000u, 0x00000000u},
    {0x33998ca3u, 0x00000000u, 0x00000000u},
    {0x351a4d03u, 0x00000000u, 0x00000000u},
    {0x369b0d64u, 0x0000006eu, 0x00000000u},
    {0x381bcdd5u, 0x0000e471u, 0x00000000u},
    {0x3a1cce54u, 0x00000075u, 0x00000000u},
    {0x3b9d8eb4u, 0x00000078u, 0x00000000u},
    {0x3d1e4f14u, 0x0000007bu, 0x00000000u},
    {0x3f1f4f84u, 0x0000007fu, 0x00000000u},
    {0x40a00ff4u, 0x00000082u, 0x00000000u},
    {0x42a11063u, 0x00000000u, 0x00000000u},
    {0x4421d0c4u, 0x00000089u, 0x00000000u},
    {0x45a29134u, 0x0000008cu, 0x00000000u},
    {0x47a391a4u, 0x00000090u, 0x00000000u},
    {0x49245214u, 0x00000093u, 0x00000000u},
    {0x4aa51273u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x4ca612f3u, 0x00000000u, 0x00000000u},
    {0x00269332u, 0x00000000u, 0x00000000u},
    {0x00271362u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x0028d442u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x53a994b3u, 0x00000000u, 0x00000000u},
    {0x54aa14f3u, 0x00000000u, 0x00000000u},
    {0x002a9532u, 0x00000000u, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x002c5612u, 0x00000000u, 0x00000000u},
    {0x002cd642u, 0x00000000u, 0x00000000u},
    {0x002d5682u, 0x00000000u, 0x00000000u},
    {0x5c2dd6c3u, 0x00000000u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x5e2ed753u, 0x00000000u, 0x00000000u},
    {0x5faf97a3u, 0x00000000u, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x0030d842u, 0x00000000u, 0x00000000u},
    {0x63315883u, 0x00000000u, 0x00000000u},
    {0x6431d8d3u, 0x00000000u, 0x00000000u},
    {0x65325913u, 0x00000000u, 0x00000000u},
    {0x66b31963u, 0x00000000u, 0x00000000u},
    {0x67b399b3u, 0x00000000u, 0x00000000u},
    {0x68b419f3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x0000c042u, 0x00000000u, 0x00000000u},
    {0x02810073u, 0x00000000u, 0x00000000u},
    {0x038180b3u, 0x00000000u, 0x00000000u},
    {0x000200f2u, 0x00000000u, 0x00000000u},
    {0x00028122u, 0x00000000u, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x07034193u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x00044212u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x000582b2u, 0x00000000u, 0x00000000u},
    {0x000602e2u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0e06c353u, 0x00000000u, 0x00000000u},
    {0x00074392u, 0x00000000u, 0x00000000u},
    {0x0007c3c2u, 0x00000000u, 0x00000000u},
    {0x108803f3u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x00090472u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x000b4582u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x000dc6d2u, 0x00000000u, 0x00000000u},
    {0x000e4702u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x001007f2u, 0x00000000u, 0x00000000u},
    {0x21908823u, 0x00000000u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x239188b3u, 0x00000000u, 0x00000000u},
    {0x249208f3u, 0x00000000u, 0x00000000u},
    {0x25928933u, 0x00000000u, 0x00000000u},
    {0x27134983u, 0x00000000u, 0x00000000u},
    {0x2813c9d3u, 0x00000000u, 0x00000000u},
    {0x29144a14u, 0x00000053u, 0x00000000u},
    {0x2a950a73u, 0x00000000u, 0x00000000u},
    {0x2b958ab3u, 0x00000000u, 0x00000000u},
    {0x2c960af3u, 0x00000000u, 0x00000000u},
    {0x2d968b33u, 0x00000000u, 0x00000000u},
    {0x2f174b83u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc53u, 0x00000000u, 0x00000000u},
    {0x33194c94u, 0x00000067u, 0x00000000u},
    {0x349a0cf4u, 0x0000006au, 0x00000000u},
    {0x361acd54u, 0x0000006du, 0x00000000u},
    {0x381bcdc4u, 0x00000071u, 0x00000000u},
    {0x399c8e34u, 0x00000074u, 0x00000000u},
    {0x3b9d8ea4u, 0x00000078u, 0x00000000u},
    {0x3d1e4f14u, 0x0000007bu, 0x00000000u},
    {0x3e9f0f74u, 0x0000007eu, 0x00000000u},
    {0x401fcfd4u, 0x00000081u, 0x00000000u},
    {0x4220d044u, 0x00000085u, 0x00000000u},
    {0x43a190b4u, 0x00000088u, 0x00000000u},
    {0x45a29123u, 0x00000000u, 0x00000000u},
    {0x47235183u, 0x00000000u, 0x00000000u},
    {0x48a411e4u, 0x00000092u, 0x00000000u},
    {0x4a24d253u, 0x00000000u, 0x00000000u},
    {0x002592a2u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x002813f2u, 0x00000000u, 0x00000000u},
    {0x00289422u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x002a5512u, 0x00000000u, 0x00000000u},
    {0x002ad542u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x57ab95b3u, 0x00000000u, 0x00000000u},
    {0x002c15f2u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x5a2cd653u, 0x00000000u, 0x00000000u},
    {0x5b2d5693u, 0x00000000u, 0x00000000u},
    {0x5c2dd6d3u, 0x00000000u, 0x00000000u},
    {0x5dae9723u, 0x00000000u, 0x00000000u},
    {0x5eaf1773u, 0x00000000u, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x6230d843u, 0x00000000u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x6431d8d3u, 0x00000000u, 0x00000000u},
    {0x65b29923u, 0x00000000u, 0x00000000u},
    {0x66b31973u, 0x00000000u, 0x00000000u},
    {0x67b399b3u, 0x00000000u, 0x00000000u},
    {0x68b419f3u, 0x00000000u, 0x00000000u},
    {0x69b49a33u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x02810073u, 0x00000000u, 0x00000000u},
    {0x000180b2u, 0x00000000u, 0x00000000u},
    {0x000200e2u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x00034192u, 0x00000000u, 0x00000000u},
    {0x0003c1c2u, 0x00000000u, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x00050272u, 0x00000000u, 0x00000000u},
    {0x000582a2u, 0x00000000u, 0x00000000u},
    {0x0c05c2d3u, 0x00000000u, 0x00000000u},
    {0x00064312u, 0x00000000u, 0x00000000u},
    {0x0006c342u, 0x00000000u, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x000783b2u, 0x00000000u, 0x00000000u},
    {0x000803e2u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x00094492u, 0x00000000u, 0x00000000u},
    {0x0009c4c2u, 0x00000000u, 0x00000000u},
    {0x148a04f3u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x000b0572u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x000c4612u, 0x00000000u, 0x00000000u},
    {0x198c8633u, 0x00000000u, 0x00000000u},
    {0x000d0672u, 0x00000000u, 0x00000000u},
    {0x000d86a2u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x000ec752u, 0x00000000u, 0x00000000u},
    {0x000f4782u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x2411c8c3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x2813c9d3u, 0x00000000u, 0x00000000u},
    {0x29144a13u, 0x00000000u, 0x00000000u},
    {0x2a950a63u, 0x00000000u, 0x00000000u},
    {0x2b958ab3u, 0x00000000u, 0x00000000u},
    {0x2c960af3u, 0x00000000u, 0x00000000u},
    {0x2e16cb43u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc53u, 0x00000000u, 0x00000000u},
    {0x33194c94u, 0x00000067u, 0x00000000u},
    {0x349a0cf4u, 0x0000006au, 0x00000000u},
    {0x361acd54u, 0x0000006du, 0x00000000u},
    {0x379b8db4u, 0x00000070u, 0x00000000u},
    {0x399c8e24u, 0x00000074u, 0x00000000u},
    {0x3b1d4e94u, 0x00000077u, 0x00000000u},
    {0x3c9e0ef4u, 0x0000007au, 0x00000000u},
    {0x3e9f0f64u, 0x0000007eu, 0x00000000u},
    {0x401fcfd4u, 0x00000081u, 0x00000000u},
    {0x41a09033u, 0x00000000u, 0x00000000u},
    {0x43215084u, 0x00000087u, 0x00000000u},
    {0x44a210f4u, 0x0000008au, 0x00000000u},
    {0x46a31163u, 0x00000000u, 0x00000000u},
    {0x4823d1c4u, 0x00000091u, 0x00000000u},
    {0x49a49234u, 0x00000094u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x0025d2d2u, 0x00000000u, 0x00000000u},
    {0x00265302u, 0x00000000u, 0x00000000u},
    {0x4da69333u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x002793b2u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x002994b2u, 0x00000000u, 0x00000000u},
    {0x002a14e2u, 0x00000000u, 0x00000000u},
    {0x552a5513u, 0x00000000u, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x002b5592u, 0x00000000u, 0x00000000u},
    {0x002bd5c2u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x59ac9633u, 0x00000000u, 0x00000000u},
    {0x5aad1673u, 0x00000000u, 0x00000000u},
    {0x5bad96b3u, 0x00000000u, 0x00000000u},
    {0x5cae16f3u, 0x00000000u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5eaf1773u, 0x00000000u, 0x00000000u},
    {0x602fd7c3u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x6230d853u, 0x00000000u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x64b218e3u, 0x00000000u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x66b31973u, 0x00000000u, 0x00000000u},
    {0x0033d9c2u, 0x00000000u, 0x00000000u},
    {0x00345a02u, 0x00000000u, 0x00000000u},
    {0x0034da42u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x02810073u, 0x00000000u, 0x00000000u},
    {0x000180b2u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0002c152u, 0x00000000u, 0x00000000u},
    {0x00034182u, 0x00000000u, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x000401f2u, 0x00000000u, 0x00000000u},
    {0x00048222u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x00054292u, 0x00000000u, 0x00000000u},
    {0x0005c2c2u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x00070372u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x1007c3d3u, 0x00000000u, 0x00000000u},
    {0x00084412u, 0x00000000u, 0x00000000u},
    {0x0008c442u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x000984b2u, 0x00000000u, 0x00000000u},
    {0x000a04e2u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x000b4582u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x000c05f2u, 0x00000000u, 0x00000000u},
    {0x000c8622u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x000d4682u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x1c8e06f3u, 0x00000000u, 0x00000000u},
    {0x000e8732u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x000fc7d2u, 0x00000000u, 0x00000000u},
    {0x00104802u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x23114883u, 0x00000000u, 0x00000000u},
    {0x2411c8d3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x289409e3u, 0x00000000u, 0x00000000u},
    {0x29948a33u, 0x00000000u, 0x00000000u},
    {0x2a950a73u, 0x00000000u, 0x00000000u},
    {0x2b958ab3u, 0x00000000u, 0x00000000u},
    {0x2d164b03u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x31184c13u, 0x00000000u, 0x00000000u},
    {0x3218cc53u, 0x00000000u, 0x00000000u},
    {0x33194c94u, 0x00000067u, 0x00000000u},
    {0x349a0cf3u, 0x00000000u, 0x00000000u},
    {0x361acd44u, 0x0000006du, 0x00000000u},
    {0x379b8db4u, 0x00000070u, 0x00000000u},
    {0x391c4e14u, 0x00000073u, 0x00000000u},
    {0x3a9d0e74u, 0x00000076u, 0x00000000u},
    {0x3c9e0ee3u, 0x00000000u, 0x00000000u},
    {0x3e1ecf44u, 0x0000007du, 0x00000000u},
    {0x3f9f8fb4u, 0x00000080u, 0x00000000u},
    {0x41205014u, 0x00000083u, 0x00000000u},
    {0x42a11074u, 0x00000086u, 0x00000000u},
    {0x44a210e3u, 0x00000000u, 0x00000000u},
    {0x4622d144u, 0x0000008du, 0x00000000u},
    {0x47a391b4u, 0x00000090u, 0x00000000u},
    {0x49245214u, 0x00000093u, 0x00000000u},
    {0x00255282u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x002612f2u, 0x00000000u, 0x00000000u},
    {0x00269322u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x002813f2u, 0x00000000u, 0x00000000u},
    {0x00289422u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x00295492u, 0x00000000u, 0x00000000u},
    {0x0029d4c2u, 0x00000000u, 0x00000000u},
    {0x54aa14f3u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x002b1572u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x002c5612u, 0x00000000u, 0x00000000u},
    {0x002cd642u, 0x00000000u, 0x00000000u},
    {0x5aad1673u, 0x00000000u, 0x00000000u},
    {0x5bad96b3u, 0x00000000u, 0x00000000u},
    {0x002e5702u, 0x00000000u, 0x00000000u},
    {0x5e2ed743u, 0x00000000u, 0x00000000u},
    {0x5f2f5793u, 0x00000000u, 0x00000000u},
    {0x602fd7d3u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x62b11863u, 0x00000000u, 0x00000000u},
    {0x63b198b3u, 0x00000000u, 0x00000000u},
    {0x64b218f3u, 0x00000000u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x66b31973u, 0x00000000u, 0x00000000u},
    {0x0033d9c2u, 0x00000000u, 0x00000000u},
    {0x00345a02u, 0x00000000u, 0x00000000u},
    {0x0034da42u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x00010072u, 0x00000000u, 0x00000000u},
    {0x000180a2u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x00024112u, 0x00000000u, 0x00000000u},
    {0x0002c142u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x000381b2u, 0x00000000u, 0x00000000u},
    {0x000401e2u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0004c252u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x000602f2u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0006c352u, 0x00000000u, 0x00000000u},
    {0x00074382u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x000803f2u, 0x00000000u, 0x00000000u},
    {0x00088422u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x00094492u, 0x00000000u, 0x00000000u},
    {0x0009c4c2u, 0x00000000u, 0x00000000u},
    {0x148a04f3u, 0x00000000u, 0x00000000u},
    {0x000a8532u, 0x00000000u, 0x00000000u},
    {0x160ac553u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x000bc5d2u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x000d0662u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x000dc6d2u, 0x00000000u, 0x00000000u},
    {0x000e4702u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x000f0772u, 0x00000000u, 0x00000000u},
    {0x000f87a2u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21104813u, 0x00000000u, 0x00000000u},
    {0x2210c853u, 0x00000000u, 0x00000000u},
    {0x23114893u, 0x00000000u, 0x00000000u},
    {0x2411c8d3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x001389a2u, 0x00000000u, 0x00000000u},
    {0x289409e3u, 0x00000000u, 0x00000000u},
    {0x29948a33u, 0x00000000u, 0x00000000u},
    {0x2a950a73u, 0x00000000u, 0x00000000u},
    {0x2b958ab3u, 0x00000000u, 0x00000000u},
    {0x2c960af3u, 0x00000000u, 0x00000000u},
    {0x2d968b33u, 0x00000000u, 0x00000000u},
    {0x2f174b83u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x0018cc42u, 0x00000000u, 0x00000000u},
    {0x32990c74u, 0x00000066u, 0x00000000u},
    {0x3419ccd4u, 0x00000069u, 0x00000000u},
    {0x359a8d34u, 0x0000006cu, 0x00000000u},
    {0x379b8da3u, 0x00000000u, 0x00000000u},
    {0x391c4e03u, 0x00000000u, 0x00000000u},
    {0x3a9d0e63u, 0x00000000u, 0x00000000u},
    {0x3c1dcec4u, 0x00000079u, 0x00000000u},
    {0x3d9e8f34u, 0x0000007cu, 0x00000000u},
    {0x3f1f4f94u, 0x0000007fu, 0x00000000u},
    {0x40a00ff4u, 0x00000082u, 0x00000000u},
    {0x4220d054u, 0x00000085u, 0x00000000u},
    {0x43a190b4u, 0x00000088u, 0x00000000u},
    {0x45225114u, 0x0000008bu, 0x00000000u},
    {0x47235183u, 0x00000000u, 0x00000000u},
    {0x48a411e3u, 0x00000000u, 0x00000000u},
    {0x4a24d243u, 0x00000000u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x0025d2d2u, 0x00000000u, 0x00000000u},
    {0x00265302u, 0x00000000u, 0x00000000u},
    {0x4da69333u, 0x00000000u, 0x00000000u},
    {0x00271372u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x00291472u, 0x00000000u, 0x00000000u},
    {0x002994a2u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x002a5512u, 0x00000000u, 0x00000000u},
    {0x002ad542u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x002b95b2u, 0x00000000u, 0x00000000u},
    {0x002c15e2u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x002cd652u, 0x00000000u, 0x00000000u},
    {0x002d5682u, 0x00000000u, 0x00000000u},
    {0x5c2dd6c3u, 0x00000000u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x5e2ed753u, 0x00000000u, 0x00000000u},
    {0x5f2f5793u, 0x00000000u, 0x00000000u},
    {0x602fd7d3u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x00311862u, 0x00000000u, 0x00000000u},
    {0x63b198a3u, 0x00000000u, 0x00000000u},
    {0x64b218f3u, 0x00000000u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x66b31973u, 0x00000000u, 0x00000000u},
    {0x67b399b3u, 0x00000000u, 0x00000000u},
    {0x68b419f3u, 0x00000000u, 0x00000000u},
    {0x0034da42u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x0000c042u, 0x00000000u, 0x00000000u},
    {0x02810073u, 0x00000000u, 0x00000000u},
    {0x000180b2u, 0x00000000u, 0x00000000u},
    {0x000200e2u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0002c152u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x000381b2u, 0x00000000u, 0x00000000u},
    {0x000401e2u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0004c252u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x000582b2u, 0x00000000u, 0x00000000u},
    {0x000602e2u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0006c352u, 0x00000000u, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x000783b2u, 0x00000000u, 0x00000000u},
    {0x000803e2u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x0008c452u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x000984b2u, 0x00000000u, 0x00000000u},
    {0x000a04e2u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x000b85b2u, 0x00000000u, 0x00000000u},
    {0x000c05e2u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x1a8d0673u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x000e06f2u, 0x00000000u, 0x00000000u},
    {0x000e8722u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x000f4792u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x001007f2u, 0x00000000u, 0x00000000u},
    {0x00108822u, 0x00000000u, 0x00000000u},
    {0x22910863u, 0x00000000u, 0x00000000u},
    {0x239188b3u, 0x00000000u, 0x00000000u},
    {0x249208f3u, 0x00000000u, 0x00000000u},
    {0x25928933u, 0x00000000u, 0x00000000u},
    {0x26930973u, 0x00000000u, 0x00000000u},
    {0x279389b3u, 0x00000000u, 0x00000000u},
    {0x289409f3u, 0x00000000u, 0x00000000u},
    {0x29948a33u, 0x00000000u, 0x00000000u},
    {0x2a950a73u, 0x00000000u, 0x00000000u},
    {0x0015cac2u, 0x00000000u, 0x00000000u},
    {0x00164b02u, 0x00000000u, 0x00000000u},
    {0x2e16cb43u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x0018cc42u, 0x00000000u, 0x00000000u},
    {0x32990c74u, 0x00000066u, 0x00000000u},
    {0x3419ccd4u, 0x00000069u, 0x00000000u},
    {0x359a8d35u, 0x0000da6cu, 0x00000000u},
    {0x379b8db4u, 0x00000070u, 0x00000000u},
    {0x391c4e14u, 0x00000073u, 0x00000000u},
    {0x3a9d0e74u, 0x00000076u, 0x00000000u},
    {0x3c1dced4u, 0x00000079u, 0x00000000u},
    {0x3d9e8f34u, 0x0000007cu, 0x00000000u},
    {0x3f1f4f94u, 0x0000007fu, 0x00000000u},
    {0x40a00ff4u, 0x00000082u, 0x00000000u},
    {0x4220d054u, 0x00000085u, 0x00000000u},
    {0x43a190b4u, 0x00000088u, 0x00000000u},
    {0x45225114u, 0x0000008bu, 0x00000000u},
    {0x46a31174u, 0x0000008eu, 0x00000000u},
    {0x4823d1d4u, 0x00000091u, 0x00000000u},
    {0x49a49234u, 0x00000094u, 0x00000000u},
    {0x00255292u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x002612f2u, 0x00000000u, 0x00000000u},
    {0x00269322u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x50a813f3u, 0x00000000u, 0x00000000u},
    {0x00289432u, 0x00000000u, 0x00000000u},
    {0x00291462u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x0029d4d2u, 0x00000000u, 0x00000000u},
    {0x54aa14f3u, 0x00000000u, 0x00000000u},
    {0x002a9532u, 0x00000000u, 0x00000000u},
    {0x002b1562u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x002bd5d2u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x002d1662u, 0x00000000u, 0x00000000u},
    {0x5b2d5693u, 0x00000000u, 0x00000000u},
    {0x5c2dd6d3u, 0x00000000u, 0x00000000u},
    {0x002e9722u, 0x00000000u, 0x00000000u},
    {0x5eaf1763u, 0x00000000u, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x61b09833u, 0x00000000u, 0x00000000u},
    {0x62b11873u, 0x00000000u, 0x00000000u},
    {0x63b198b3u, 0x00000000u, 0x00000000u},
    {0x64b218f3u, 0x00000000u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x66b31973u, 0x00000000u, 0x00000000u},
    {0x67b399b3u, 0x00000000u, 0x00000000u},
    {0x00345a02u, 0x00000000u, 0x00000000u},
    {0x0034da42u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x02810073u, 0x00000000u, 0x00000000u},
    {0x000180b2u, 0x00000000u, 0x00000000u},
    {0x0401c0d3u, 0x00000000u, 0x00000000u},
    {0x00024112u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x00030172u, 0x00000000u, 0x00000000u},
    {0x000381a2u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x00044212u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x00050272u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0005c2d2u, 0x00000000u, 0x00000000u},
    {0x00064302u, 0x00000000u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x00070372u, 0x00000000u, 0x00000000u},
    {0x000783a2u, 0x00000000u, 0x00000000u},
    {0x1007c3d3u, 0x00000000u, 0x00000000u},
    {0x00084412u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x00090472u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x0009c4d2u, 0x00000000u, 0x00000000u},
    {0x000a4502u, 0x00000000u, 0x00000000u},
    {0x000a8532u, 0x00000000u, 0x00000000u},
    {0x000b0562u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x000bc5d2u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x000dc6d2u, 0x00000000u, 0x00000000u},
    {0x000e4702u, 0x00000000u, 0x00000000u},
    {0x000e8732u, 0x00000000u, 0x00000000u},
    {0x000f0762u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x000fc7d2u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x239188b3u, 0x00000000u, 0x00000000u},
    {0x249208f3u, 0x00000000u, 0x00000000u},
    {0x25928933u, 0x00000000u, 0x00000000u},
    {0x26930973u, 0x00000000u, 0x00000000u},
    {0x279389b4u, 0x00000050u, 0x00000000u},
    {0x29144a13u, 0x00000000u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b154a93u, 0x00000000u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d164b13u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x0018cc42u, 0x00000000u, 0x00000000u},
    {0x32990c74u, 0x00000066u, 0x00000000u},
    {0x3419ccd4u, 0x00000069u, 0x00000000u},
    {0x359a8d34u, 0x0000006cu, 0x00000000u},
    {0x379b8da3u, 0x00000000u, 0x00000000u},
    {0x389c0df4u, 0x00000072u, 0x00000000u},
    {0x3a1cce54u, 0x00000075u, 0x00000000u},
    {0x3b9d8eb4u, 0x00000078u, 0x00000000u},
    {0x3d1e4f14u, 0x0000007bu, 0x00000000u},
    {0x3e9f0f74u, 0x0000007eu, 0x00000000u},
    {0x401fcfd4u, 0x00000081u, 0x00000000u},
    {0x41a09033u, 0x00000000u, 0x00000000u},
    {0x43215084u, 0x00000087u, 0x00000000u},
    {0x44a210f4u, 0x0000008au, 0x00000000u},
    {0x4622d154u, 0x0000008du, 0x00000000u},
    {0x47a391b4u, 0x00000090u, 0x00000000u},
    {0x49245214u, 0x00000093u, 0x00000000u},
    {0x4aa51273u, 0x00000000u, 0x00000000u},
    {0x002592b2u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x00265312u, 0x00000000u, 0x00000000u},
    {0x0026d342u, 0x00000000u, 0x00000000u},
    {0x00271372u, 0x00000000u, 0x00000000u},
    {0x002793a2u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x51285413u, 0x00000000u, 0x00000000u},
    {0x0028d452u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x002994b2u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x002a5512u, 0x00000000u, 0x00000000u},
    {0x002ad542u, 0x00000000u, 0x00000000u},
    {0x002b1572u, 0x00000000u, 0x00000000u},
    {0x002b95a2u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x002c5612u, 0x00000000u, 0x00000000u},
    {0x59ac9633u, 0x00000000u, 0x00000000u},
    {0x002d1672u, 0x00000000u, 0x00000000u},
    {0x5bad96a3u, 0x00000000u, 0x00000000u},
    {0x5cae16f3u, 0x00000000u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5eaf1773u, 0x00000000u, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x61b09833u, 0x00000000u, 0x00000000u},
    {0x62b11873u, 0x00000000u, 0x00000000u},
    {0x0031d8c2u, 0x00000000u, 0x00000000u},
    {0x00325902u, 0x00000000u, 0x00000000u},
    {0x0032d942u, 0x00000000u, 0x00000000u},
    {0x00335982u, 0x00000000u, 0x00000000u},
    {0x0033d9c2u, 0x00000000u, 0x00000000u},
    {0x00345a02u, 0x00000000u, 0x00000000u},
    {0x0034da42u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x00010072u, 0x00000000u, 0x00000000u},
    {0x000180a2u, 0x00000000u, 0x00000000u},
    {0x0001c0d2u, 0x00000000u, 0x00000000u},
    {0x00024102u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x00030172u, 0x00000000u, 0x00000000u},
    {0x07034193u, 0x00000000u, 0x00000000u},
    {0x0003c1d2u, 0x00000000u, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x00050272u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0005c2d2u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x00068332u, 0x00000000u, 0x00000000u},
    {0x0e06c353u, 0x00000000u, 0x00000000u},
    {0x00074392u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x000803f2u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x0008c452u, 0x00000000u, 0x00000000u},
    {0x00094482u, 0x00000000u, 0x00000000u},
    {0x138984b3u, 0x00000000u, 0x00000000u},
    {0x000a04f2u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x000b85b2u, 0x00000000u, 0x00000000u},
    {0x000c05e2u, 0x00000000u, 0x00000000u},
    {0x000c4612u, 0x00000000u, 0x00000000u},
    {0x000cc642u, 0x00000000u, 0x00000000u},
    {0x000d0672u, 0x00000000u, 0x00000000u},
    {0x000d86a2u, 0x00000000u, 0x00000000u},
    {0x1c0dc6d3u, 0x00000000u, 0x00000000u},
    {0x000e4712u, 0x00000000u, 0x00000000u},
    {0x000ec742u, 0x00000000u, 0x00000000u},
    {0x000f0772u, 0x00000000u, 0x00000000u},
    {0x000f87a2u, 0x00000000u, 0x00000000u},
    {0x000fc7d2u, 0x00000000u, 0x00000000u},
    {0x00104802u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x2411c8c3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x0013c9d2u, 0x00000000u, 0x00000000u},
    {0x00144a02u, 0x00000000u, 0x00000000u},
    {0x0014ca42u, 0x00000000u, 0x00000000u},
    {0x00154a82u, 0x00000000u, 0x00000000u},
    {0x0015cac2u, 0x00000000u, 0x00000000u},
    {0x2d164b03u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x0018cc42u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x3419ccc3u, 0x00000000u, 0x00000000u},
    {0x359a8d23u, 0x00000000u, 0x00000000u},
    {0x371b4d84u, 0x0000006fu, 0x00000000u},
    {0x389c0df3u, 0x00000000u, 0x00000000u},
    {0x3a1cce43u, 0x00000000u, 0x00000000u},
    {0x3b9d8ea3u, 0x00000000u, 0x00000000u},
    {0x3c9e0ef4u, 0x0000007au, 0x00000000u},
    {0x3e9f0f63u, 0x00000000u, 0x00000000u},
    {0x401fcfc3u, 0x00000000u, 0x00000000u},
    {0x41a09023u, 0x00000000u, 0x00000000u},
    {0x42a11074u, 0x00000086u, 0x00000000u},
    {0x4421d0d4u, 0x00000089u, 0x00000000u},
    {0x45a29134u, 0x0000008cu, 0x00000000u},
    {0x47235194u, 0x0000008fu, 0x00000000u},
    {0x48a411f4u, 0x00000092u, 0x00000000u},
    {0x4a24d253u, 0x00000000u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x0025d2d2u, 0x00000000u, 0x00000000u},
    {0x4ca612f3u, 0x00000000u, 0x00000000u},
    {0x00269332u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x50a813f3u, 0x00000000u, 0x00000000u},
    {0x00289432u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x00295492u, 0x00000000u, 0x00000000u},
    {0x53a994b3u, 0x00000000u, 0x00000000u},
    {0x002a14f2u, 0x00000000u, 0x00000000u},
    {0x002a9522u, 0x00000000u, 0x00000000u},
    {0x002ad552u, 0x00000000u, 0x00000000u},
    {0x002b5582u, 0x00000000u, 0x00000000u},
    {0x002b95b2u, 0x00000000u, 0x00000000u},
    {0x002c15e2u, 0x00000000u, 0x00000000u},
    {0x002c9622u, 0x00000000u, 0x00000000u},
    {0x002cd652u, 0x00000000u, 0x00000000u},
    {0x002d5682u, 0x00000000u, 0x00000000u},
    {0x002dd6c2u, 0x00000000u, 0x00000000u},
    {0x002e5702u, 0x00000000u, 0x00000000u},
    {0x002ed742u, 0x00000000u, 0x00000000u},
    {0x002f5782u, 0x00000000u, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x61b09834u, 0x000000c4u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x6431d8d3u, 0x00000000u, 0x00000000u},
    {0x65325913u, 0x00000000u, 0x00000000u},
    {0x6632d953u, 0x00000000u, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x00010072u, 0x00000000u, 0x00000000u},
    {0x000180a2u, 0x00000000u, 0x00000000u},
    {0x0001c0d2u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x00028132u, 0x00000000u, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x07034193u, 0x00000000u, 0x00000000u},
    {0x0003c1d2u, 0x00000000u, 0x00000000u},
    {0x088401f3u, 0x00000000u, 0x00000000u},
    {0x00048232u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x00054292u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x000602f2u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0006c352u, 0x00000000u, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x000783b2u, 0x00000000u, 0x00000000u},
    {0x000803e2u, 0x00000000u, 0x00000000u},
    {0x00084412u, 0x00000000u, 0x00000000u},
    {0x0008c442u, 0x00000000u, 0x00000000u},
    {0x00090472u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x000a4512u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x000b0572u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x000bc5d2u, 0x00000000u, 0x00000000u},
    {0x188c05f3u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x000d4692u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x000e06f2u, 0x00000000u, 0x00000000u},
    {0x000e8722u, 0x00000000u, 0x00000000u},
    {0x000ec752u, 0x00000000u, 0x00000000u},
    {0x000f4782u, 0x00000000u, 0x00000000u},
    {0x000f87b2u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x21104813u, 0x00000000u, 0x00000000u},
    {0x2210c853u, 0x00000000u, 0x00000000u},
    {0x23114893u, 0x00000000u, 0x00000000u},
    {0x2411c8d3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x2813c9d3u, 0x00000000u, 0x00000000u},
    {0x29144a13u, 0x00000000u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b154a93u, 0x00000000u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d164b13u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x0018cc42u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x3419ccc3u, 0x00000000u, 0x00000000u},
    {0x351a4d14u, 0x0000006bu, 0x00000000u},
    {0x371b4d83u, 0x00000000u, 0x00000000u},
    {0x389c0de3u, 0x00000000u, 0x00000000u},
    {0x399c8e34u, 0x00000074u, 0x00000000u},
    {0x3b1d4e93u, 0x00000000u, 0x00000000u},
    {0x3c9e0ee4u, 0x0000007au, 0x00000000u},
    {0x3e1ecf54u, 0x0000007du, 0x00000000u},
    {0x3f9f8fb3u, 0x00000000u, 0x00000000u},
    {0x41205003u, 0x00000000u, 0x00000000u},
    {0x4220d054u, 0x00000085u, 0x00000000u},
    {0x4421d0c3u, 0x00000000u, 0x00000000u},
    {0x45a29123u, 0x00000000u, 0x00000000u},
    {0x46a31174u, 0x0000008eu, 0x00000000u},
    {0x4823d1d3u, 0x00000000u, 0x00000000u},
    {0x49a49224u, 0x00000094u, 0x00000000u},
    {0x00255292u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x002612f2u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x0026d352u, 0x00000000u, 0x00000000u},
    {0x4ea71373u, 0x00000000u, 0x00000000u},
    {0x002793b2u, 0x00000000u, 0x00000000u},
    {0x002813e2u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x0028d442u, 0x00000000u, 0x00000000u},
    {0x00291472u, 0x00000000u, 0x00000000u},
    {0x002994a2u, 0x00000000u, 0x00000000u},
    {0x0029d4d2u, 0x00000000u, 0x00000000u},
    {0x002a5502u, 0x00000000u, 0x00000000u},
    {0x002a9532u, 0x00000000u, 0x00000000u},
    {0x002b1562u, 0x00000000u, 0x00000000u},
    {0x572b5593u, 0x00000000u, 0x00000000u},
    {0x002bd5d2u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x5a2cd653u, 0x00000000u, 0x00000000u},
    {0x5b2d5693u, 0x00000000u, 0x00000000u},
    {0x5c2dd6d3u, 0x00000000u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x5e2ed753u, 0x00000000u, 0x00000000u},
    {0x5f2f5793u, 0x00000000u, 0x00000000u},
    {0x602fd7d3u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x6230d853u, 0x00000000u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x6431d8d3u, 0x00000000u, 0x00000000u},
    {0x65325913u, 0x00000000u, 0x00000000u},
    {0x6632d953u, 0x00000000u, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x00010072u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0001c0d2u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x00028132u, 0x00000000u, 0x00000000u},
    {0x00030162u, 0x00000000u, 0x00000000u},
    {0x00034192u, 0x00000000u, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x000401f2u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0004c252u, 0x00000000u, 0x00000000u},
    {0x0a850273u, 0x00000000u, 0x00000000u},
    {0x000582b2u, 0x00000000u, 0x00000000u},
    {0x000602e2u, 0x00000000u, 0x00000000u},
    {0x00064312u, 0x00000000u, 0x00000000u},
    {0x0d868333u, 0x00000000u, 0x00000000u},
    {0x00070372u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x0007c3d2u, 0x00000000u, 0x00000000u},
    {0x00084402u, 0x00000000u, 0x00000000u},
    {0x00088432u, 0x00000000u, 0x00000000u},
    {0x00090462u, 0x00000000u, 0x00000000u},
    {0x00094492u, 0x00000000u, 0x00000000u},
    {0x138984b3u, 0x00000000u, 0x00000000u},
    {0x000a04f2u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x000b4582u, 0x00000000u, 0x00000000u},
    {0x000b85b2u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x000c4612u, 0x00000000u, 0x00000000u},
    {0x198c8633u, 0x00000000u, 0x00000000u},
    {0x000d0672u, 0x00000000u, 0x00000000u},
    {0x1b0d4693u, 0x00000000u, 0x00000000u},
    {0x000dc6d2u, 0x00000000u, 0x00000000u},
    {0x000e4702u, 0x00000000u, 0x00000000u},
    {0x000e8732u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x000f4792u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x001007f2u, 0x00000000u, 0x00000000u},
    {0x00108822u, 0x00000000u, 0x00000000u},
    {0x00110862u, 0x00000000u, 0x00000000u},
    {0x23114893u, 0x00000000u, 0x00000000u},
    {0x2411c8d3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x2813c9d3u, 0x00000000u, 0x00000000u},
    {0x29144a13u, 0x00000000u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b154a93u, 0x00000000u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d164b13u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x0017cbd2u, 0x00000000u, 0x00000000u},
    {0x00184c02u, 0x00000000u, 0x00000000u},
    {0x0018cc42u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x33998cb4u, 0x00000068u, 0x00000000u},
    {0x351a4d13u, 0x00000000u, 0x00000000u},
    {0x369b0d64u, 0x0000006eu, 0x00000000u},
    {0x381bcdd3u, 0x00000000u, 0x00000000u},
    {0x399c8e24u, 0x00000074u, 0x00000000u},
    {0x3b1d4e93u, 0x00000000u, 0x00000000u},
    {0x3c9e0ee3u, 0x00000000u, 0x00000000u},
    {0x3d9e8f34u, 0x0000007cu, 0x00000000u},
    {0x3f1f4f94u, 0x0000007fu, 0x00000000u},
    {0x40a00ff4u, 0x00000082u, 0x00000000u},
    {0x4220d053u, 0x00000000u, 0x00000000u},
    {0x43a190a3u, 0x00000000u, 0x00000000u},
    {0x44a210f4u, 0x0000008au, 0x00000000u},
    {0x46a31163u, 0x00000000u, 0x00000000u},
    {0x47a391b4u, 0x00000090u, 0x00000000u},
    {0x49245213u, 0x00000000u, 0x00000000u},
    {0x4aa51263u, 0x00000000u, 0x00000000u},
    {0x002592b2u, 0x00000000u, 0x00000000u},
    {0x4c25d2d3u, 0x00000000u, 0x00000000u},
    {0x00265312u, 0x00000000u, 0x00000000u},
    {0x4da69333u, 0x00000000u, 0x00000000u},
    {0x00271372u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x0027d3c2u, 0x00000000u, 0x00000000u},
    {0x50a813f3u, 0x00000000u, 0x00000000u},
    {0x00289432u, 0x00000000u, 0x00000000u},
    {0x5228d453u, 0x00000000u, 0x00000000u},
    {0x00295492u, 0x00000000u, 0x00000000u},
    {0x002994b2u, 0x00000000u, 0x00000000u},
    {0x002a14e2u, 0x00000000u, 0x00000000u},
    {0x552a5513u, 0x00000000u, 0x00000000u},
    {0x002ad552u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x002b95b2u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x002c5612u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x002d1662u, 0x00000000u, 0x00000000u},
    {0x002d96a2u, 0x00000000u, 0x00000000u},
    {0x002e16e2u, 0x00000000u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x5e2ed753u, 0x00000000u, 0x00000000u},
    {0x5f2f5793u, 0x00000000u, 0x00000000u},
    {0x602fd7d3u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x6230d853u, 0x00000000u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x6431d8d3u, 0x00000000u, 0x00000000u},
    {0x65325913u, 0x00000000u, 0x00000000u},
    {0x6632d953u, 0x00000000u, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x00010072u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0001c0d2u, 0x00000000u, 0x00000000u},
    {0x048200f3u, 0x00000000u, 0x00000000u},
    {0x00028132u, 0x00000000u, 0x00000000u},
    {0x0602c153u, 0x00000000u, 0x00000000u},
    {0x00034192u, 0x00000000u, 0x00000000u},
    {0x000381b2u, 0x00000000u, 0x00000000u},
    {0x000401e2u, 0x00000000u, 0x00000000u},
    {0x09044213u, 0x00000000u, 0x00000000u},
    {0x0004c252u, 0x00000000u, 0x00000000u},
    {0x00050272u, 0x00000000u, 0x00000000u},
    {0x000582a2u, 0x00000000u, 0x00000000u},
    {0x0005c2d2u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x00068332u, 0x00000000u, 0x00000000u},
    {0x0e06c353u, 0x00000000u, 0x00000000u},
    {0x00074392u, 0x00000000u, 0x00000000u},
    {0x0f8783b3u, 0x00000000u, 0x00000000u},
    {0x000803f2u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x0008c452u, 0x00000000u, 0x00000000u},
    {0x12890473u, 0x00000000u, 0x00000000u},
    {0x000984b2u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x000a4512u, 0x00000000u, 0x00000000u},
    {0x000a8532u, 0x00000000u, 0x00000000u},
    {0x000b0562u, 0x00000000u, 0x00000000u},
    {0x170b4593u, 0x00000000u, 0x00000000u},
    {0x000bc5d2u, 0x00000000u, 0x00000000u},
    {0x000c05f2u, 0x00000000u, 0x00000000u},
    {0x000c8622u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x000d4682u, 0x00000000u, 0x00000000u},
    {0x000d86b2u, 0x00000000u, 0x00000000u},
    {0x000e06e2u, 0x00000000u, 0x00000000u},
    {0x000e4712u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x000f0772u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x000fc7d2u, 0x00000000u, 0x00000000u},
    {0x00104802u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x00110872u, 0x00000000u, 0x00000000u},
    {0x239188a3u, 0x00000000u, 0x00000000u},
    {0x249208f3u, 0x00000000u, 0x00000000u},
    {0x00128932u, 0x00000000u, 0x00000000u},
    {0x00130962u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x2813c9d3u, 0x00000000u, 0x00000000u},
    {0x29144a13u, 0x00000000u, 0x00000000u},
    {0x2a14ca53u, 0x00000000u, 0x00000000u},
    {0x2b154a93u, 0x00000000u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d164b13u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x33998cb4u, 0x00000068u, 0x00000000u},
    {0x351a4d14u, 0x0000006bu, 0x00000000u},
    {0x369b0d73u, 0x00000000u, 0x00000000u},
    {0x381bcdc3u, 0x00000000u, 0x00000000u},
    {0x399c8e23u, 0x00000000u, 0x00000000u},
    {0x3b1d4e83u, 0x00000000u, 0x00000000u},
    {0x3c1dced3u, 0x00000000u, 0x00000000u},
    {0x3d9e8f23u, 0x00000000u, 0x00000000u},
    {0x3f1f4f83u, 0x00000000u, 0x00000000u},
    {0x40a00fe3u, 0x00000000u, 0x00000000u},
    {0x41a09034u, 0x00000084u, 0x00000000u},
    {0x43215094u, 0x00000087u, 0x00000000u},
    {0x44a210f3u, 0x00000000u, 0x00000000u},
    {0x4622d143u, 0x00000000u, 0x00000000u},
    {0x47235194u, 0x0000008fu, 0x00000000u},
    {0x48a411f4u, 0x00000092u, 0x00000000u},
    {0x4a24d253u, 0x00000000u, 0x00000000u},
    {0x4b255293u, 0x00000000u, 0x00000000u},
    {0x0025d2d2u, 0x00000000u, 0x00000000u},
    {0x4ca612f3u, 0x00000000u, 0x00000000u},
    {0x00269332u, 0x00000000u, 0x00000000u},
    {0x0026d352u, 0x00000000u, 0x00000000u},
    {0x00275382u, 0x00000000u, 0x00000000u},
    {0x002793b2u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x0028d442u, 0x00000000u, 0x00000000u},
    {0x00291472u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x0029d4d2u, 0x00000000u, 0x00000000u},
    {0x002a5502u, 0x00000000u, 0x00000000u},
    {0x002a9532u, 0x00000000u, 0x00000000u},
    {0x562ad553u, 0x00000000u, 0x00000000u},
    {0x002b5592u, 0x00000000u, 0x00000000u},
    {0x002b95b2u, 0x00000000u, 0x00000000u},
    {0x002c15e2u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x002cd652u, 0x00000000u, 0x00000000u},
    {0x5aad1673u, 0x00000000u, 0x00000000u},
    {0x5bad96b3u, 0x00000000u, 0x00000000u},
    {0x5cae16f3u, 0x00000000u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5eaf1773u, 0x00000000u, 0x00000000u},
    {0x002f97b2u, 0x00000000u, 0x00000000u},
    {0x003017e2u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x6230d853u, 0x00000000u, 0x00000000u},
    {0x003198a2u, 0x00000000u, 0x00000000u},
    {0x6431d8d3u, 0x00000000u, 0x00000000u},
    {0x65325913u, 0x00000000u, 0x00000000u},
    {0x6632d953u, 0x00000000u, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x00010072u, 0x00000000u, 0x00000000u},
    {0x00014092u, 0x00000000u, 0x00000000u},
    {0x0001c0c2u, 0x00000000u, 0x00000000u},
    {0x000200f2u, 0x00000000u, 0x00000000u},
    {0x00028122u, 0x00000000u, 0x00000000u},
    {0x0002c152u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x000381b2u, 0x00000000u, 0x00000000u},
    {0x0803c1d3u, 0x00000000u, 0x00000000u},
    {0x00044212u, 0x00000000u, 0x00000000u},
    {0x09848233u, 0x00000000u, 0x00000000u},
    {0x00050272u, 0x00000000u, 0x00000000u},
    {0x00054292u, 0x00000000u, 0x00000000u},
    {0x0005c2c2u, 0x00000000u, 0x00000000u},
    {0x000602f2u, 0x00000000u, 0x00000000u},
    {0x00068322u, 0x00000000u, 0x00000000u},
    {0x0006c352u, 0x00000000u, 0x00000000u},
    {0x0e870373u, 0x00000000u, 0x00000000u},
    {0x000783b2u, 0x00000000u, 0x00000000u},
    {0x000803e2u, 0x00000000u, 0x00000000u},
    {0x00084412u, 0x00000000u, 0x00000000u},
    {0x11888433u, 0x00000000u, 0x00000000u},
    {0x00090472u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x0009c4d2u, 0x00000000u, 0x00000000u},
    {0x000a04f2u, 0x00000000u, 0x00000000u},
    {0x000a8522u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x000b4582u, 0x00000000u, 0x00000000u},
    {0x000b85b2u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x000c4612u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x000d0662u, 0x00000000u, 0x00000000u},
    {0x000d4692u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x000e06f2u, 0x00000000u, 0x00000000u},
    {0x000e8722u, 0x00000000u, 0x00000000u},
    {0x000ec752u, 0x00000000u, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x000f87b2u, 0x00000000u, 0x00000000u},
    {0x200fc7d3u, 0x00000000u, 0x00000000u},
    {0x00104812u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x239188b3u, 0x00000000u, 0x00000000u},
    {0x249208f3u, 0x00000000u, 0x00000000u},
    {0x25928933u, 0x00000000u, 0x00000000u},
    {0x26930973u, 0x00000000u, 0x00000000u},
    {0x279389b3u, 0x00000000u, 0x00000000u},
    {0x001409f2u, 0x00000000u, 0x00000000u},
    {0x00148a22u, 0x00000000u, 0x00000000u},
    {0x00150a62u, 0x00000000u, 0x00000000u},
    {0x00158aa2u, 0x00000000u, 0x00000000u},
    {0x2c15cad3u, 0x00000000u, 0x00000000u},
    {0x2d164b13u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x0018cc42u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x33998cb3u, 0x00000000u, 0x00000000u},
    {0x351a4d03u, 0x00000000u, 0x00000000u},
    {0x361acd54u, 0x0000006du, 0x00000000u},
    {0x379b8db4u, 0x00000070u, 0x00000000u},
    {0x391c4e13u, 0x00000000u, 0x00000000u},
    {0x3a9d0e63u, 0x00000000u, 0x00000000u},
    {0x3c1dcec3u, 0x00000000u, 0x00000000u},
    {0x3d1e4f14u, 0x0000007bu, 0x00000000u},
    {0x3e9f0f73u, 0x00000000u, 0x00000000u},
    {0x401fcfc3u, 0x00000000u, 0x00000000u},
    {0x41a09023u, 0x00000000u, 0x00000000u},
    {0x42a11074u, 0x00000086u, 0x00000000u},
    {0x4421d0d4u, 0x00000089u, 0x00000000u},
    {0x45a29133u, 0x00000000u, 0x00000000u},
    {0x47235183u, 0x00000000u, 0x00000000u},
    {0x4823d1d4u, 0x00000091u, 0x00000000u},
    {0x49a49233u, 0x00000000u, 0x00000000u},
    {0x00255282u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x002612f2u, 0x00000000u, 0x00000000u},
    {0x00265312u, 0x00000000u, 0x00000000u},
    {0x0026d342u, 0x00000000u, 0x00000000u},
    {0x00271372u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x0027d3d2u, 0x00000000u, 0x00000000u},
    {0x002813f2u, 0x00000000u, 0x00000000u},
    {0x00289422u, 0x00000000u, 0x00000000u},
    {0x0028d452u, 0x00000000u, 0x00000000u},
    {0x00295482u, 0x00000000u, 0x00000000u},
    {0x002994b2u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x002a5512u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x002b1572u, 0x00000000u, 0x00000000u},
    {0x002b5592u, 0x00000000u, 0x00000000u},
    {0x002bd5c2u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x002cd652u, 0x00000000u, 0x00000000u},
    {0x002d5682u, 0x00000000u, 0x00000000u},
    {0x002dd6c2u, 0x00000000u, 0x00000000u},
    {0x002e5702u, 0x00000000u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5eaf1773u, 0x00000000u, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x61b09833u, 0x00000000u, 0x00000000u},
    {0x62b11873u, 0x00000000u, 0x00000000u},
    {0x63b198b3u, 0x00000000u, 0x00000000u},
    {0x003218f2u, 0x00000000u, 0x00000000u},
    {0x00329922u, 0x00000000u, 0x00000000u},
    {0x00331962u, 0x00000000u, 0x00000000u},
    {0x003399a2u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x01808033u, 0x00000000u, 0x00000000u},
    {0x00010072u, 0x00000000u, 0x00000000u},
    {0x03014093u, 0x00000000u, 0x00000000u},
    {0x0001c0d2u, 0x00000000u, 0x00000000u},
    {0x000200f2u, 0x00000000u, 0x00000000u},
    {0x05024113u, 0x00000000u, 0x00000000u},
    {0x0002c152u, 0x00000000u, 0x00000000u},
    {0x06830173u, 0x00000000u, 0x00000000u},
    {0x000381b2u, 0x00000000u, 0x00000000u},
    {0x0003c1d2u, 0x00000000u, 0x00000000u},
    {0x00044202u, 0x00000000u, 0x00000000u},
    {0x00048232u, 0x00000000u, 0x00000000u},
    {0x0a04c253u, 0x00000000u, 0x00000000u},
    {0x00054292u, 0x00000000u, 0x00000000u},
    {0x0b8582b3u, 0x00000000u, 0x00000000u},
    {0x000602f2u, 0x00000000u, 0x00000000u},
    {0x00064312u, 0x00000000u, 0x00000000u},
    {0x0006c342u, 0x00000000u, 0x00000000u},
    {0x00070372u, 0x00000000u, 0x00000000u},
    {0x0f074393u, 0x00000000u, 0x00000000u},
    {0x0007c3d2u, 0x00000000u, 0x00000000u},
    {0x108803f3u, 0x00000000u, 0x00000000u},
    {0x00088432u, 0x00000000u, 0x00000000u},
    {0x0008c452u, 0x00000000u, 0x00000000u},
    {0x00094482u, 0x00000000u, 0x00000000u},
    {0x000984b2u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x000a4512u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x000b0572u, 0x00000000u, 0x00000000u},
    {0x000b4592u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x000c05f2u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x000d0672u, 0x00000000u, 0x00000000u},
    {0x000d86a2u, 0x00000000u, 0x00000000u},
    {0x000dc6d2u, 0x00000000u, 0x00000000u},
    {0x000e4702u, 0x00000000u, 0x00000000u},
    {0x000e8732u, 0x00000000u, 0x00000000u},
    {0x1e0ec753u, 0x00000000u, 0x00000000u},
    {0x000f4792u, 0x00000000u, 0x00000000u},
    {0x1f8f87b3u, 0x00000000u, 0x00000000u},
    {0x001007f2u, 0x00000000u, 0x00000000u},
    {0x00104812u, 0x00000000u, 0x00000000u},
    {0x0010c842u, 0x00000000u, 0x00000000u},
    {0x00114882u, 0x00000000u, 0x00000000u},
    {0x2411c8c3u, 0x00000000u, 0x00000000u},
    {0x00124912u, 0x00000000u, 0x00000000u},
    {0x0012c942u, 0x00000000u, 0x00000000u},
    {0x26930973u, 0x00000000u, 0x00000000u},
    {0x279389b3u, 0x00000000u, 0x00000000u},
    {0x289409f3u, 0x00000000u, 0x00000000u},
    {0x29948a33u, 0x00000000u, 0x00000000u},
    {0x00150a72u, 0x00000000u, 0x00000000u},
    {0x00158aa2u, 0x00000000u, 0x00000000u},
    {0x00160ae2u, 0x00000000u, 0x00000000u},
    {0x2d164b13u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x33998cb3u, 0x00000000u, 0x00000000u},
    {0x351a4d03u, 0x00000000u, 0x00000000u},
    {0x369b0d63u, 0x00000000u, 0x00000000u},
    {0x379b8db3u, 0x00000000u, 0x00000000u},
    {0x391c4e03u, 0x00000000u, 0x00000000u},
    {0x3a9d0e63u, 0x00000000u, 0x00000000u},
    {0x3b9d8eb3u, 0x00000000u, 0x00000000u},
    {0x3d1e4f03u, 0x00000000u, 0x00000000u},
    {0x3e9f0f63u, 0x00000000u, 0x00000000u},
    {0x3f9f8fb4u, 0x00000080u, 0x00000000u},
    {0x41205013u, 0x00000000u, 0x00000000u},
    {0x4220d054u, 0x00000085u, 0x00000000u},
    {0x43a190b4u, 0x00000088u, 0x00000000u},
    {0x45225113u, 0x00000000u, 0x00000000u},
    {0x4622d154u, 0x0000008du, 0x00000000u},
    {0x4823d1c3u, 0x00000000u, 0x00000000u},
    {0x49a49223u, 0x00000000u, 0x00000000u},
    {0x00251272u, 0x00000000u, 0x00000000u},
    {0x002592a2u, 0x00000000u, 0x00000000u},
    {0x0025d2d2u, 0x00000000u, 0x00000000u},
    {0x00265302u, 0x00000000u, 0x00000000u},
    {0x00269332u, 0x00000000u, 0x00000000u},
    {0x4e26d353u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x4fa793b3u, 0x00000000u, 0x00000000u},
    {0x002813f2u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x00291472u, 0x00000000u, 0x00000000u},
    {0x53295493u, 0x00000000u, 0x00000000u},
    {0x0029d4d2u, 0x00000000u, 0x00000000u},
    {0x002a14f2u, 0x00000000u, 0x00000000u},
    {0x002a9522u, 0x00000000u, 0x00000000u},
    {0x002ad552u, 0x00000000u, 0x00000000u},
    {0x56ab1573u, 0x00000000u, 0x00000000u},
    {0x002b95b2u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x002c5612u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x002d1662u, 0x00000000u, 0x00000000u},
    {0x5b2d5693u, 0x00000000u, 0x00000000u},
    {0x5c2dd6d3u, 0x00000000u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x002ed752u, 0x00000000u, 0x00000000u},
    {0x002f5782u, 0x00000000u, 0x00000000u},
    {0x5faf97b3u, 0x00000000u, 0x00000000u},
    {0x60b017f3u, 0x00000000u, 0x00000000u},
    {0x61b09833u, 0x00000000u, 0x00000000u},
    {0x62b11873u, 0x00000000u, 0x00000000u},
    {0x003198b2u, 0x00000000u, 0x00000000u},
    {0x64b218e3u, 0x00000000u, 0x00000000u},
    {0x00329932u, 0x00000000u, 0x00000000u},
    {0x00331962u, 0x00000000u, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x00008032u, 0x00000000u, 0x00000000u},
    {0x00010062u, 0x00000000u, 0x00000000u},
    {0x00014092u, 0x00000000u, 0x00000000u},
    {0x0001c0c2u, 0x00000000u, 0x00000000u},
    {0x000200f2u, 0x00000000u, 0x00000000u},
    {0x00024112u, 0x00000000u, 0x00000000u},
    {0x0002c142u, 0x00000000u, 0x00000000u},
    {0x00030172u, 0x00000000u, 0x00000000u},
    {0x000381a2u, 0x00000000u, 0x00000000u},
    {0x0003c1d2u, 0x00000000u, 0x00000000u},
    {0x000401f2u, 0x00000000u, 0x00000000u},
    {0x00048222u, 0x00000000u, 0x00000000u},
    {0x0004c252u, 0x00000000u, 0x00000000u},
    {0x00054282u, 0x00000000u, 0x00000000u},
    {0x000582b2u, 0x00000000u, 0x00000000u},
    {0x000602e2u, 0x00000000u, 0x00000000u},
    {0x00064312u, 0x00000000u, 0x00000000u},
    {0x00068332u, 0x00000000u, 0x00000000u},
    {0x00070362u, 0x00000000u, 0x00000000u},
    {0x00074392u, 0x00000000u, 0x00000000u},
    {0x0007c3c2u, 0x00000000u, 0x00000000u},
    {0x000803f2u, 0x00000000u, 0x00000000u},
    {0x00084412u, 0x00000000u, 0x00000000u},
    {0x0008c442u, 0x00000000u, 0x00000000u},
    {0x00090472u, 0x00000000u, 0x00000000u},
    {0x13094493u, 0x00000000u, 0x00000000u},
    {0x0009c4d2u, 0x00000000u, 0x00000000u},
    {0x000a04f2u, 0x00000000u, 0x00000000u},
    {0x150a4513u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x000b85b2u, 0x00000000u, 0x00000000u},
    {0x180bc5d3u, 0x00000000u, 0x00000000u},
    {0x000c4612u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x1a0cc653u, 0x00000000u, 0x00000000u},
    {0x000d4692u, 0x00000000u, 0x00000000u},
    {0x1b8d86b3u, 0x00000000u, 0x00000000u},
    {0x000e06f2u, 0x00000000u, 0x00000000u},
    {0x000e4712u, 0x00000000u, 0x00000000u},
    {0x1d8e8733u, 0x00000000u, 0x00000000u},
    {0x000f0772u, 0x00000000u, 0x00000000u},
    {0x1f0f4793u, 0x00000000u, 0x00000000u},
    {0x000fc7d2u, 0x00000000u, 0x00000000u},
    {0x209007f3u, 0x00000000u, 0x00000000u},
    {0x00108832u, 0x00000000u, 0x00000000u},
    {0x2210c853u, 0x00000000u, 0x00000000u},
    {0x23114893u, 0x00000000u, 0x00000000u},
    {0x2411c8d3u, 0x00000000u, 0x00000000u},
    {0x00124912u, 0x00000000u, 0x00000000u},
    {0x0012c942u, 0x00000000u, 0x00000000u},
    {0x26930973u, 0x00000000u, 0x00000000u},
    {0x0013c9c2u, 0x00000000u, 0x00000000u},
    {0x289409f3u, 0x00000000u, 0x00000000u},
    {0x00148a32u, 0x00000000u, 0x00000000u},
    {0x2a950a63u, 0x00000000u, 0x00000000u},
    {0x00158ab2u, 0x00000000u, 0x00000000u},
    {0x00160ae2u, 0x00000000u, 0x00000000u},
    {0x00168b22u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x3017cbd3u, 0x00000000u, 0x00000000u},
    {0x00184c12u, 0x00000000u, 0x00000000u},
    {0x00188c32u, 0x00000000u, 0x00000000u},
    {0x00190c62u, 0x00000000u, 0x00000000u},
    {0x33998ca3u, 0x00000000u, 0x00000000u},
    {0x349a0cf3u, 0x00000000u, 0x00000000u},
    {0x361acd43u, 0x00000000u, 0x00000000u},
    {0x379b8da3u, 0x00000000u, 0x00000000u},
    {0x391c4e03u, 0x00000000u, 0x00000000u},
    {0x3a1cce53u, 0x00000000u, 0x00000000u},
    {0x3b1d4e94u, 0x00000077u, 0x00000000u},
    {0x3c9e0ef4u, 0x0000007au, 0x00000000u},
    {0x3e1ecf53u, 0x00000000u, 0x00000000u},
    {0x3f9f8fa3u, 0x00000000u, 0x00000000u},
    {0x41205003u, 0x00000000u, 0x00000000u},
    {0x4220d053u, 0x00000000u, 0x00000000u},
    {0x43a190a3u, 0x00000000u, 0x00000000u},
    {0x45225103u, 0x00000000u, 0x00000000u},
    {0x4622d153u, 0x00000000u, 0x00000000u},
    {0x47235194u, 0x0000008fu, 0x00000000u},
    {0x48a411f3u, 0x00000000u, 0x00000000u},
    {0x4a24d243u, 0x00000000u, 0x00000000u},
    {0x00255292u, 0x00000000u, 0x00000000u},
    {0x4ba592b3u, 0x00000000u, 0x00000000u},
    {0x002612f2u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x0026d352u, 0x00000000u, 0x00000000u},
    {0x00271372u, 0x00000000u, 0x00000000u},
    {0x4f275393u, 0x00000000u, 0x00000000u},
    {0x0027d3d2u, 0x00000000u, 0x00000000u},
    {0x50a813f3u, 0x00000000u, 0x00000000u},
    {0x00289432u, 0x00000000u, 0x00000000u},
    {0x0028d452u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x002994b2u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x002a5512u, 0x00000000u, 0x00000000u},
    {0x55aa9533u, 0x00000000u, 0x00000000u},
    {0x002b1572u, 0x00000000u, 0x00000000u},
    {0x002b5592u, 0x00000000u, 0x00000000u},
    {0x57ab95b3u, 0x00000000u, 0x00000000u},
    {0x002c15f2u, 0x00000000u, 0x00000000u},
    {0x592c5613u, 0x00000000u, 0x00000000u},
    {0x002cd652u, 0x00000000u, 0x00000000u},
    {0x002d1672u, 0x00000000u, 0x00000000u},
    {0x002d96a2u, 0x00000000u, 0x00000000u},
    {0x002e16e2u, 0x00000000u, 0x00000000u},
    {0x5d2e5713u, 0x00000000u, 0x00000000u},
    {0x5e2ed753u, 0x00000000u, 0x00000000u},
    {0x5f2f5793u, 0x00000000u, 0x00000000u},
    {0x002fd7d2u, 0x00000000u, 0x00000000u},
    {0x61305803u, 0x00000000u, 0x00000000u},
    {0x0030d852u, 0x00000000u, 0x00000000u},
    {0x00315882u, 0x00000000u, 0x00000000u},
    {0x63b198b3u, 0x00000000u, 0x00000000u},
    {0x64b218f3u, 0x00000000u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x00331972u, 0x00000000u, 0x00000000u},
    {0x67335993u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x00008032u, 0x00000000u, 0x00000000u},
    {0x00010062u, 0x00000000u, 0x00000000u},
    {0x00014092u, 0x00000000u, 0x00000000u},
    {0x0001c0c2u, 0x00000000u, 0x00000000u},
    {0x000200f2u, 0x00000000u, 0x00000000u},
    {0x00024112u, 0x00000000u, 0x00000000u},
    {0x05828133u, 0x00000000u, 0x00000000u},
    {0x00030172u, 0x00000000u, 0x00000000u},
    {0x00034192u, 0x00000000u, 0x00000000u},
    {0x0003c1c2u, 0x00000000u, 0x00000000u},
    {0x000401f2u, 0x00000000u, 0x00000000u},
    {0x00048222u, 0x00000000u, 0x00000000u},
    {0x0004c252u, 0x00000000u, 0x00000000u},
    {0x00050272u, 0x00000000u, 0x00000000u},
    {0x0b054293u, 0x00000000u, 0x00000000u},
    {0x0005c2d2u, 0x00000000u, 0x00000000u},
    {0x0c8602f3u, 0x00000000u, 0x00000000u},
    {0x00068332u, 0x00000000u, 0x00000000u},
    {0x0006c352u, 0x00000000u, 0x00000000u},
    {0x00074382u, 0x00000000u, 0x00000000u},
    {0x000783b2u, 0x00000000u, 0x00000000u},
    {0x0007c3d2u, 0x00000000u, 0x00000000u},
    {0x00084402u, 0x00000000u, 0x00000000u},
    {0x00088432u, 0x00000000u, 0x00000000u},
    {0x1208c453u, 0x00000000u, 0x00000000u},
    {0x00094492u, 0x00000000u, 0x00000000u},
    {0x000984b2u, 0x00000000u, 0x00000000u},
    {0x1409c4d3u, 0x00000000u, 0x00000000u},
    {0x000a4512u, 0x00000000u, 0x00000000u},
    {0x158a8533u, 0x00000000u, 0x00000000u},
    {0x000b0572u, 0x00000000u, 0x00000000u},
    {0x000b4592u, 0x00000000u, 0x00000000u},
    {0x178b85b3u, 0x00000000u, 0x00000000u},
    {0x000c05f2u, 0x00000000u, 0x00000000u},
    {0x190c4613u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x000d0672u, 0x00000000u, 0x00000000u},
    {0x000d86a2u, 0x00000000u, 0x00000000u},
    {0x000dc6d2u, 0x00000000u, 0x00000000u},
    {0x000e06f2u, 0x00000000u, 0x00000000u},
    {0x1d0e4713u, 0x00000000u, 0x00000000u},
    {0x000ec752u, 0x00000000u, 0x00000000u},
    {0x1e8f0773u, 0x00000000u, 0x00000000u},
    {0x000f87b2u, 0x00000000u, 0x00000000u},
    {0x000fc7d2u, 0x00000000u, 0x00000000u},
    {0x00104802u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x00110872u, 0x00000000u, 0x00000000u},
    {0x001188a2u, 0x00000000u, 0x00000000u},
    {0x2411c8d3u, 0x00000000u, 0x00000000u},
    {0x25124913u, 0x00000000u, 0x00000000u},
    {0x0012c952u, 0x00000000u, 0x00000000u},
    {0x00134982u, 0x00000000u, 0x00000000u},
    {0x0013c9c2u, 0x00000000u, 0x00000000u},
    {0x00144a02u, 0x00000000u, 0x00000000u},
    {0x29948a33u, 0x00000000u, 0x00000000u},
    {0x00150a72u, 0x00000000u, 0x00000000u},
    {0x2b958aa3u, 0x00000000u, 0x00000000u},
    {0x00160af2u, 0x00000000u, 0x00000000u},
    {0x00168b22u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x0017cbd2u, 0x00000000u, 0x00000000u},
    {0x00184c02u, 0x00000000u, 0x00000000u},
    {0x31988c33u, 0x00000000u, 0x00000000u},
    {0x32990c73u, 0x00000000u, 0x00000000u},
    {0x33998cb3u, 0x00000000u, 0x00000000u},
    {0x349a0cf4u, 0x0000006au, 0x00000000u},
    {0x361acd53u, 0x00000000u, 0x00000000u},
    {0x371b4d93u, 0x00000000u, 0x00000000u},
    {0x389c0de3u, 0x00000000u, 0x00000000u},
    {0x3a1cce43u, 0x00000000u, 0x00000000u},
    {0x3b9d8ea3u, 0x00000000u, 0x00000000u},
    {0x3c9e0ef3u, 0x00000000u, 0x00000000u},
    {0x3d9e8f34u, 0x0000007cu, 0x00000000u},
    {0x3f1f4f94u, 0x0000007fu, 0x00000000u},
    {0x40a00ff3u, 0x00000000u, 0x00000000u},
    {0x41a09033u, 0x00000000u, 0x00000000u},
    {0x43215083u, 0x00000000u, 0x00000000u},
    {0x44a210e3u, 0x00000000u, 0x00000000u},
    {0x4622d143u, 0x00000000u, 0x00000000u},
    {0x47235193u, 0x00000000u, 0x00000000u},
    {0x4823d1d4u, 0x00000091u, 0x00000000u},
    {0x49a49233u, 0x00000000u, 0x00000000u},
    {0x4aa51273u, 0x00000000u, 0x00000000u},
    {0x002592b2u, 0x00000000u, 0x00000000u},
    {0x0025d2d2u, 0x00000000u, 0x00000000u},
    {0x00265302u, 0x00000000u, 0x00000000u},
    {0x00269332u, 0x00000000u, 0x00000000u},
    {0x00271362u, 0x00000000u, 0x00000000u},
    {0x00275392u, 0x00000000u, 0x00000000u},
    {0x002793b2u, 0x00000000u, 0x00000000u},
    {0x5027d3d3u, 0x00000000u, 0x00000000u},
    {0x00285412u, 0x00000000u, 0x00000000u},
    {0x51a89433u, 0x00000000u, 0x00000000u},
    {0x00291472u, 0x00000000u, 0x00000000u},
    {0x00295492u, 0x00000000u, 0x00000000u},
    {0x0029d4c2u, 0x00000000u, 0x00000000u},
    {0x002a14f2u, 0x00000000u, 0x00000000u},
    {0x552a5513u, 0x00000000u, 0x00000000u},
    {0x002ad552u, 0x00000000u, 0x00000000u},
    {0x002b1572u, 0x00000000u, 0x00000000u},
    {0x002b95a2u, 0x00000000u, 0x00000000u},
    {0x002bd5d2u, 0x00000000u, 0x00000000u},
    {0x58ac15f3u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x002cd652u, 0x00000000u, 0x00000000u},
    {0x5aad1673u, 0x00000000u, 0x00000000u},
    {0x5bad96b3u, 0x00000000u, 0x00000000u},
    {0x002e16f2u, 0x00000000u, 0x00000000u},
    {0x002e9722u, 0x00000000u, 0x00000000u},
    {0x5e2ed753u, 0x00000000u, 0x00000000u},
    {0x5f2f5793u, 0x00000000u, 0x00000000u},
    {0x602fd7d3u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x0030d852u, 0x00000000u, 0x00000000u},
    {0x62b11873u, 0x00000000u, 0x00000000u},
    {0x0031d8c2u, 0x00000000u, 0x00000000u},
    {0x64b218f3u, 0x00000000u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x00331972u, 0x00000000u, 0x00000000u},
    {0x003399a2u, 0x00000000u, 0x00000000u},
    {0x6833d9d3u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
    {0x00004002u, 0x00000000u, 0x00000000u},
    {0x00008032u, 0x00000000u, 0x00000000u},
    {0x00010062u, 0x00000000u, 0x00000000u},
    {0x00014092u, 0x00000000u, 0x00000000u},
    {0x000180b2u, 0x00000000u, 0x00000000u},
    {0x000200e2u, 0x00000000u, 0x00000000u},
    {0x00024112u, 0x00000000u, 0x00000000u},
    {0x0002c142u, 0x00000000u, 0x00000000u},
    {0x00030172u, 0x00000000u, 0x00000000u},
    {0x00034192u, 0x00000000u, 0x00000000u},
    {0x078381b3u, 0x00000000u, 0x00000000u},
    {0x000401f2u, 0x00000000u, 0x00000000u},
    {0x00044212u, 0x00000000u, 0x00000000u},
    {0x0004c242u, 0x00000000u, 0x00000000u},
    {0x00050272u, 0x00000000u, 0x00000000u},
    {0x00054292u, 0x00000000u, 0x00000000u},
    {0x0005c2c2u, 0x00000000u, 0x00000000u},
    {0x000602f2u, 0x00000000u, 0x00000000u},
    {0x0d064313u, 0x00000000u, 0x00000000u},
    {0x0006c352u, 0x00000000u, 0x00000000u},
    {0x00070372u, 0x00000000u, 0x00000000u},
    {0x000783a2u, 0x00000000u, 0x00000000u},
    {0x0007c3d2u, 0x00000000u, 0x00000000u},
    {0x000803f2u, 0x00000000u, 0x00000000u},
    {0x11084413u, 0x00000000u, 0x00000000u},
    {0x0008c452u, 0x00000000u, 0x00000000u},
    {0x00090472u, 0x00000000u, 0x00000000u},
    {0x000984a2u, 0x00000000u, 0x00000000u},
    {0x0009c4d2u, 0x00000000u, 0x00000000u},
    {0x148a04f3u, 0x00000000u, 0x00000000u},
    {0x000a8532u, 0x00000000u, 0x00000000u},
    {0x000ac552u, 0x00000000u, 0x00000000u},
    {0x168b0573u, 0x00000000u, 0x00000000u},
    {0x000b85b2u, 0x00000000u, 0x00000000u},
    {0x000bc5d2u, 0x00000000u, 0x00000000u},
    {0x000c4602u, 0x00000000u, 0x00000000u},
    {0x000c8632u, 0x00000000u, 0x00000000u},
    {0x000cc652u, 0x00000000u, 0x00000000u},
    {0x000d4682u, 0x00000000u, 0x00000000u},
    {0x000d86b2u, 0x00000000u, 0x00000000u},
    {0x000dc6d2u, 0x00000000u, 0x00000000u},
    {0x000e4702u, 0x00000000u, 0x00000000u},
    {0x000e8732u, 0x00000000u, 0x00000000u},
    {0x000f0762u, 0x00000000u, 0x00000000u},
    {0x000f4792u, 0x00000000u, 0x00000000u},
    {0x000f87b2u, 0x00000000u, 0x00000000u},
    {0x001007e2u, 0x00000000u, 0x00000000u},
    {0x00104812u, 0x00000000u, 0x00000000u},
    {0x21908833u, 0x00000000u, 0x00000000u},
    {0x22910873u, 0x00000000u, 0x00000000u},
    {0x239188b3u, 0x00000000u, 0x00000000u},
    {0x249208f3u, 0x00000000u, 0x00000000u},
    {0x00128932u, 0x00000000u, 0x00000000u},
    {0x2612c953u, 0x00000000u, 0x00000000u},
    {0x27134993u, 0x00000000u, 0x00000000u},
    {0x0013c9d2u, 0x00000000u, 0x00000000u},
    {0x00144a02u, 0x00000000u, 0x00000000u},
    {0x0014ca42u, 0x00000000u, 0x00000000u},
    {0x2a950a73u, 0x00000000u, 0x00000000u},
    {0x00158ab2u, 0x00000000u, 0x00000000u},
    {0x00160ae2u, 0x00000000u, 0x00000000u},
    {0x00168b22u, 0x00000000u, 0x00000000u},
    {0x2e16cb53u, 0x00000000u, 0x00000000u},
    {0x2f174b93u, 0x00000000u, 0x00000000u},
    {0x0017cbd2u, 0x00000000u, 0x00000000u},
    {0x00184c02u, 0x00000000u, 0x00000000u},
    {0x00188c32u, 0x00000000u, 0x00000000u},
    {0x00190c62u, 0x00000000u, 0x00000000u},
    {0x33194c94u, 0x00000067u, 0x00000000u},
    {0x349a0cf3u, 0x00000000u, 0x00000000u},
    {0x359a8d34u, 0x0000006cu, 0x00000000u},
    {0x371b4d93u, 0x00000000u, 0x00000000u},
    {0x389c0de3u, 0x00000000u, 0x00000000u},
    {0x3a1cce43u, 0x00000000u, 0x00000000u},
    {0x3b1d4e93u, 0x00000000u, 0x00000000u},
    {0x3c1dced4u, 0x00000079u, 0x00000000u},
    {0x3d9e8f33u, 0x00000000u, 0x00000000u},
    {0x3f1f4f83u, 0x00000000u, 0x00000000u},
    {0x401fcfd3u, 0x00000000u, 0x00000000u},
    {0x41a09023u, 0x00000000u, 0x00000000u},
    {0x42a11074u, 0x00000086u, 0x00000000u},
    {0x4421d0d3u, 0x00000000u, 0x00000000u},
    {0x45225113u, 0x00000000u, 0x00000000u},
    {0x46a31163u, 0x00000000u, 0x00000000u},
    {0x4823d1c3u, 0x00000000u, 0x00000000u},
    {0x49245213u, 0x00000000u, 0x00000000u},
    {0x00251262u, 0x00000000u, 0x00000000u},
    {0x00255292u, 0x00000000u, 0x00000000u},
    {0x0025d2c2u, 0x00000000u, 0x00000000u},
    {0x002612f2u, 0x00000000u, 0x00000000u},
    {0x4d265313u, 0x00000000u, 0x00000000u},
    {0x0026d352u, 0x00000000u, 0x00000000u},
    {0x00271372u, 0x00000000u, 0x00000000u},
    {0x002793a2u, 0x00000000u, 0x00000000u},
    {0x0027d3d2u, 0x00000000u, 0x00000000u},
    {0x002813f2u, 0x00000000u, 0x00000000u},
    {0x00289422u, 0x00000000u, 0x00000000u},
    {0x0028d452u, 0x00000000u, 0x00000000u},
    {0x52a91473u, 0x00000000u, 0x00000000u},
    {0x002994b2u, 0x00000000u, 0x00000000u},
    {0x5429d4d3u, 0x00000000u, 0x00000000u},
    {0x002a5512u, 0x00000000u, 0x00000000u},
    {0x002a9532u, 0x00000000u, 0x00000000u},
    {0x002ad552u, 0x00000000u, 0x00000000u},
    {0x002b5582u, 0x00000000u, 0x00000000u},
    {0x002b95b2u, 0x00000000u, 0x00000000u},
    {0x582bd5d3u, 0x00000000u, 0x00000000u},
    {0x002c5612u, 0x00000000u, 0x00000000u},
    {0x002c9632u, 0x00000000u, 0x00000000u},
    {0x002d1662u, 0x00000000u, 0x00000000u},
    {0x002d5692u, 0x00000000u, 0x00000000u},
    {0x002dd6c2u, 0x00000000u, 0x00000000u},
    {0x5cae16f3u, 0x00000000u, 0x00000000u},
    {0x5dae9733u, 0x00000000u, 0x00000000u},
    {0x5eaf1773u, 0x00000000u, 0x00000000u},
    {0x002f97b2u, 0x00000000u, 0x00000000u},
    {0x003017e2u, 0x00000000u, 0x00000000u},
    {0x61305813u, 0x00000000u, 0x00000000u},
    {0x6230d853u, 0x00000000u, 0x00000000u},
    {0x63315893u, 0x00000000u, 0x00000000u},
    {0x0031d8d2u, 0x00000000u, 0x00000000u},
    {0x64b218f3u, 0x00000000u, 0x00000000u},
    {0x65b29933u, 0x00000000u, 0x00000000u},
    {0x66b31973u, 0x00000000u, 0x00000000u},
    {0x003399b2u, 0x00000000u, 0x00000000u},
    {0x003419e2u, 0x00000000u, 0x00000000u},
    {0x69345a13u, 0x00000000u, 0x00000000u},
    {0x0034da52u, 0x00000000u, 0x00000000u},
};
static_assert(sizeof(kFc2ListsPacked) == 49920, "packed list constant extent");
#endif

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

__device__ __forceinline__ void mbarInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
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

// The same with an L2 cache policy (weights streamed once per layer: evict first, so they do not push the kernel's
// code, the activations and the KV rows out of L2).
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

__device__ __forceinline__ uint4 ldVolatile128(void const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
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

__device__ __forceinline__ uint4 lds128(uint32_t addr)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    return v;
}

__device__ __forceinline__ uint16_t lds16(uint32_t addr)
{
    uint16_t v;
    asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr)
{
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ uint2 lds64(uint32_t addr)
{
    uint2 v;
    asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr));
    return v;
}

__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v)
{
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}

__device__ __forceinline__ void sts64(uint32_t addr, uint2 v)
{
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};" ::"r"(addr), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void sts128(uint32_t addr, uint4 v)
{
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ void redsAdd(uint32_t addr, uint32_t v)
{
    asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}

// As redsAdd, without the memory clobber: the compiler keeps it in order with the other volatile asm (the shared loads
// and stores here) but need not treat it as touching every address.
__device__ __forceinline__ void redsAddUnordered(uint32_t addr, uint32_t v)
{
    asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(addr), "r"(v));
}

__device__ __forceinline__ uint32_t atomsAdd(uint32_t addr, uint32_t v)
{
    uint32_t old;
    asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(addr), "r"(v) : "memory");
    return old;
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

__device__ __forceinline__ void mmaF16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrixX4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX4Trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX2(uint32_t addr, uint32_t& r0, uint32_t& r1)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// Two E4M3 values (low 16 bits) as two exact FP16 values.
__device__ __forceinline__ uint32_t e4m3x2ToHalf2(uint32_t pair)
{
    __half2_raw const h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xffffU), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

// Two E2M1 values (one byte: element 2k in the low nibble) as two exact FP16 values.
__device__ __forceinline__ uint32_t e2m1x2ToHalf2(uint32_t byte)
{
    __half2_raw const h = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte & 0xffU), __NV_E2M1);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b)
{
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint8_t toE4m3(float x)
{
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

// Two values as E4M3 (round to nearest, saturating) in the low 16 bits: lo in bits 0-7.
__device__ __forceinline__ uint32_t toE4m3x2(float lo, float hi)
{
    return static_cast<uint32_t>(__nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float rcpApproxFtz(float x)
{
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float ex2ApproxFtz(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// E2M1 of (lo, hi) (round to nearest, saturating): lo in the low nibble of the returned byte.
__device__ __forceinline__ uint32_t toE2m1x2(float lo, float hi)
{
    uint32_t v;
    asm("{\n"
        ".reg .b8 b;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "mov.b32 %0, {b, b, b, b};\n"
        "}"
        : "=r"(v)
        : "f"(lo), "f"(hi));
    return v & 0xffu;
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

__device__ __forceinline__ uint4 ldPoll128(void const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ldPoll64(void const* p)
{
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldPoll32(void const* p)
{
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

// One 64-bit word (single-copy atomic against stU64's store, unlike a two-element vector).
__device__ __forceinline__ uint64_t ldPollU64(void const* p)
{
    uint64_t v;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

// A strong (relaxed, GPU-scope) store: with ldPollU64's volatile load a morally strong pair, so a poller sees the whole
// word or none of it (PTX promises single-copy atomicity only between morally strong accesses; a weak store is not).
__device__ __forceinline__ void stU64(void* p, uint64_t v)
{
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// Read-only loads the compiler keeps where they are written (issued before the polls that follow them).
__device__ __forceinline__ int ldPinnedS32(int const* p)
{
    int v;
    asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ldPinnedU16(void const* p)
{
    uint16_t v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float4 ldPinnedF4(void const* p)
{
    float4 v;
    asm volatile("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ void st64(void* p, uint2 v)
{
    asm volatile("st.global.v2.b32 [%0], {%1, %2};" ::"l"(p), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void st32(void* p, uint32_t v)
{
    asm volatile("st.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void prefetchL2(void const* p, uint32_t bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// ----------------------------------------------------------------------------------------------------------------
// Tensor-core NVFP4 MMA (tcgen05 kind::mxf4nvf4 block16): D[128 weight rows][8 tokens] FP32 in TMEM, A and B E2M1 in
// shared memory (K-major SWIZZLE_64B), their UE4M3 block scales in TMEM.
// ----------------------------------------------------------------------------------------------------------------
constexpr uint32_t kIdescNvf4 = (1u << 7) | (1u << 10) | ((8u >> 3) << 17) | ((128u >> 4) << 24); // M 128, N 8
// K-major SWIZZLE_64B operand descriptor without its start address: 8-row x 64 B atoms 512 B apart, version 1.
constexpr uint64_t kDescSw64 = (1ull << 16) | (static_cast<uint64_t>(512 >> 4) << 32) | (1ull << 46) | (4ull << 61);
// tcgen05.cp 32x128b source descriptor without its start address: 8-row x 16 B core matrices 128 B apart.
constexpr uint64_t kDescSf = (static_cast<uint64_t>(128 >> 4) << 32) | (1ull << 46);
constexpr uint32_t kTmemCols = 512;
constexpr uint32_t kTmemD = 0;      // accumulators: FC1 8 columns, FC2 6 tiles x 8
constexpr uint32_t kTmemSfa = 64;   // weight scales of ring slot rs: [64 + 48 rs, +48), 4 columns per K step
constexpr uint32_t kTmemSfb = 256;  // FC1 token scales of the 96 K steps: per K step a column of scales and a zero one
constexpr uint32_t kTmemSfb2 = 448; // FC2 token scales (2 K steps)

__device__ __forceinline__ void tcFenceBefore()
{
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcFenceAfter()
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// The following five are executed by a whole (converged) warp.
__device__ __forceinline__ void tmemAlloc(uint32_t dst, uint32_t cols)
{
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst), "r"(cols) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmemDealloc(uint32_t taddr, uint32_t cols)
{
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr), "r"(cols) : "memory");
}

// 512 B of block scales (shared memory, 128x4 layout) into 4 TMEM columns; the elected lane issues.
__device__ __forceinline__ void utccpElect(uint32_t taddr, uint32_t src)
{
    uint64_t const sdesc = kDescSf | ((src & 0x3ffffu) >> 4);
    asm volatile(
        "{\n"
        ".reg .pred e;\n"
        "elect.sync _|e, 0xffffffff;\n"
        "@e tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
        "}\n" ::"r"(taddr),
        "l"(sdesc)
        : "memory");
}

// D (+)= A B for one K step of 64; a and b are shared addresses of the K step's first byte (atom-aligned rows).
__device__ __forceinline__ void mmaNvf4Elect(
    uint32_t d, uint32_t a, uint32_t b, uint32_t sfa, uint32_t sfb, uint32_t acc)
{
    uint64_t const da = kDescSw64 | ((a & 0x3ffffu) >> 4);
    uint64_t const db = kDescSw64 | ((b & 0x3ffffu) >> 4);
    asm volatile(
        "{\n"
        ".reg .pred e, p;\n"
        "elect.sync _|e, 0xffffffff;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "@e tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(kIdescNvf4), "r"(sfa), "r"(sfb), "r"(acc)
        : "memory");
}

// One lane of the (converged) warp: true in exactly one lane. A batch of the single-thread tcgen05 instructions
// below under one election compiles to straight code (an election per instruction added a divergence check and a
// fallback copy of every instruction).
__device__ __forceinline__ bool electOne()
{
    uint32_t e;
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "elect.sync _|p, 0xffffffff;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(e));
    return e != 0;
}

#if GLM52_IO_MODE == 2 || GLM52_IO_MODE == 3
// As utccp1, all four 8-row groups from the same 128 bytes (stride 0): rows 8-31 repeat rows 0-7.
__device__ __forceinline__ void utccp1Rows8(uint32_t taddr, uint32_t src)
{
    uint64_t const sdesc = (1ull << 46) | ((src & 0x3ffffu) >> 4);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}

#endif

__device__ __forceinline__ void utccp1(uint32_t taddr, uint32_t src)
{
    uint64_t const sdesc = kDescSf | ((src & 0x3ffffu) >> 4);
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr), "l"(sdesc) : "memory");
}

__device__ __forceinline__ void mmaNvf4(uint32_t d, uint32_t a, uint32_t b, uint32_t sfa, uint32_t sfb, uint32_t acc)
{
    uint64_t const da = kDescSw64 | ((a & 0x3ffffu) >> 4);
    uint64_t const db = kDescSw64 | ((b & 0x3ffffu) >> 4);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n"
        "}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(kIdescNvf4), "r"(sfa), "r"(sfb), "r"(acc)
        : "memory");
}

__device__ __forceinline__ void commit1(uint32_t bar)
{
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar) : "memory");
}

// K-major SWIZZLE_128B operand descriptor without its start address: 8-row x 128 B atoms 1024 B apart, version 1.
constexpr uint64_t kDescSw128 = (1ull << 16) | (static_cast<uint64_t>(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
// tcgen05.mma kind::f8f6f4: E4M3 x E4M3 -> FP32, both operands K-major, M 128, N 256.
constexpr uint32_t kIdescF8 = (1u << 4) | ((256u >> 3) << 17) | ((128u >> 4) << 24);
constexpr uint32_t kIdescF8N32 = (1u << 4) | ((32u >> 3) << 17) | ((128u >> 4) << 24); // one row's 32 heads

// D (+)= A B^T for one K step of 32 E4M3 values; a and b are shared addresses of the step's first byte in
// SWIZZLE_128B atoms (A: 128 rows, B: 256 rows).
__device__ __forceinline__ void mmaF8(uint32_t d, uint32_t a, uint32_t b, uint32_t acc, uint32_t idesc = kIdescF8)
{
    uint64_t const da = kDescSw128 | ((a & 0x3ffffu) >> 4);
    uint64_t const db = kDescSw128 | ((b & 0x3ffffu) >> 4);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
        "}\n" ::"r"(d),
        "l"(da), "l"(db), "r"(idesc), "r"(acc)
        : "memory");
}

// One box of a 2-D tensor map (global tensor-map address) into shared memory on barrier bar (one thread).
__device__ __forceinline__ void tmaLoad2d(uint32_t dst, void const* tmap, int x, int y, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
            dst),
        "l"(tmap), "r"(x), "r"(y), "r"(bar)
        : "memory");
}

// The same with an L2 cache policy.
__device__ __forceinline__ void tmaLoad2dHint(
    uint32_t dst, void const* tmap, int x, int y, uint32_t bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
        "l"(tmap), "r"(x), "r"(y), "r"(bar), "l"(policy)
        : "memory");
}

// 32 columns of the warp's TMEM lane quarter, one register each (the caller waits).
__device__ __forceinline__ void tmemLd32(uint32_t taddr, uint32_t (&r)[32])
{
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
        : "r"(taddr)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;"
                 : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]),
                 "+r"(r[8]), "+r"(r[9]), "+r"(r[10]), "+r"(r[11]), "+r"(r[12]), "+r"(r[13]), "+r"(r[14]), "+r"(r[15]),
                 "+r"(r[16]), "+r"(r[17]), "+r"(r[18]), "+r"(r[19]), "+r"(r[20]), "+r"(r[21]), "+r"(r[22]), "+r"(r[23]),
                 "+r"(r[24]), "+r"(r[25]), "+r"(r[26]), "+r"(r[27]), "+r"(r[28]), "+r"(r[29]), "+r"(r[30]), "+r"(r[31])
                 :
                 : "memory");
}

// Columns [taddr, taddr + 128) of the warp's TMEM lane quarter: four 32-column loads in flight, one wait (the empty
// statements after it order the other registers' uses behind it).
__device__ __forceinline__ void tmemLd128(uint32_t taddr, uint32_t (&r)[128])
{
#pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=r"(r[32 * c + 0]), "=r"(r[32 * c + 1]), "=r"(r[32 * c + 2]), "=r"(r[32 * c + 3]), "=r"(r[32 * c + 4]),
            "=r"(r[32 * c + 5]), "=r"(r[32 * c + 6]), "=r"(r[32 * c + 7]), "=r"(r[32 * c + 8]), "=r"(r[32 * c + 9]),
            "=r"(r[32 * c + 10]), "=r"(r[32 * c + 11]), "=r"(r[32 * c + 12]), "=r"(r[32 * c + 13]),
            "=r"(r[32 * c + 14]), "=r"(r[32 * c + 15]), "=r"(r[32 * c + 16]), "=r"(r[32 * c + 17]),
            "=r"(r[32 * c + 18]), "=r"(r[32 * c + 19]), "=r"(r[32 * c + 20]), "=r"(r[32 * c + 21]),
            "=r"(r[32 * c + 22]), "=r"(r[32 * c + 23]), "=r"(r[32 * c + 24]), "=r"(r[32 * c + 25]),
            "=r"(r[32 * c + 26]), "=r"(r[32 * c + 27]), "=r"(r[32 * c + 28]), "=r"(r[32 * c + 29]),
            "=r"(r[32 * c + 30]), "=r"(r[32 * c + 31])
            : "r"(taddr + 32u * static_cast<uint32_t>(c))
            : "memory");
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;"
                 : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]),
                 "+r"(r[8]), "+r"(r[9]), "+r"(r[10]), "+r"(r[11]), "+r"(r[12]), "+r"(r[13]), "+r"(r[14]), "+r"(r[15]),
                 "+r"(r[16]), "+r"(r[17]), "+r"(r[18]), "+r"(r[19]), "+r"(r[20]), "+r"(r[21]), "+r"(r[22]), "+r"(r[23]),
                 "+r"(r[24]), "+r"(r[25]), "+r"(r[26]), "+r"(r[27]), "+r"(r[28]), "+r"(r[29]), "+r"(r[30]), "+r"(r[31])
                 :
                 : "memory");
    asm volatile(""
                 : "+r"(r[32]), "+r"(r[33]), "+r"(r[34]), "+r"(r[35]), "+r"(r[36]), "+r"(r[37]), "+r"(r[38]),
                 "+r"(r[39]), "+r"(r[40]), "+r"(r[41]), "+r"(r[42]), "+r"(r[43]), "+r"(r[44]), "+r"(r[45]), "+r"(r[46]),
                 "+r"(r[47]), "+r"(r[48]), "+r"(r[49]), "+r"(r[50]), "+r"(r[51]), "+r"(r[52]), "+r"(r[53]), "+r"(r[54]),
                 "+r"(r[55]), "+r"(r[56]), "+r"(r[57]), "+r"(r[58]), "+r"(r[59]), "+r"(r[60]), "+r"(r[61]), "+r"(r[62]),
                 "+r"(r[63]));
    asm volatile(""
                 : "+r"(r[64]), "+r"(r[65]), "+r"(r[66]), "+r"(r[67]), "+r"(r[68]), "+r"(r[69]), "+r"(r[70]),
                 "+r"(r[71]), "+r"(r[72]), "+r"(r[73]), "+r"(r[74]), "+r"(r[75]), "+r"(r[76]), "+r"(r[77]), "+r"(r[78]),
                 "+r"(r[79]), "+r"(r[80]), "+r"(r[81]), "+r"(r[82]), "+r"(r[83]), "+r"(r[84]), "+r"(r[85]), "+r"(r[86]),
                 "+r"(r[87]), "+r"(r[88]), "+r"(r[89]), "+r"(r[90]), "+r"(r[91]), "+r"(r[92]), "+r"(r[93]), "+r"(r[94]),
                 "+r"(r[95]));
    asm volatile(""
                 : "+r"(r[96]), "+r"(r[97]), "+r"(r[98]), "+r"(r[99]), "+r"(r[100]), "+r"(r[101]), "+r"(r[102]),
                 "+r"(r[103]), "+r"(r[104]), "+r"(r[105]), "+r"(r[106]), "+r"(r[107]), "+r"(r[108]), "+r"(r[109]),
                 "+r"(r[110]), "+r"(r[111]), "+r"(r[112]), "+r"(r[113]), "+r"(r[114]), "+r"(r[115]), "+r"(r[116]),
                 "+r"(r[117]), "+r"(r[118]), "+r"(r[119]), "+r"(r[120]), "+r"(r[121]), "+r"(r[122]), "+r"(r[123]),
                 "+r"(r[124]), "+r"(r[125]), "+r"(r[126]), "+r"(r[127]));
}

// The indexer's E4M3 scale of a row with this |max|: max(amax, 1e-12) / 448 (FP32 constants), rounded up to a power
// of two (UE8M0), as the stock prologue and Q tail.
__device__ __forceinline__ float ue8m0Scale(float amax)
{
    float const ratio = __fmul_rn(fmaxf(amax, __uint_as_float(0x2B8CBCCCu)), __uint_as_float(0x3B124925u));
    uint32_t const bits = __float_as_uint(ratio);
    uint32_t const e = (bits & 0x7F800000u) + ((bits & 0x007FFFFFu) != 0 ? 0x00800000u : 0u);
    return __uint_as_float(e);
}

// Order-preserving u32 of an FP32 value (larger value, larger key).
__device__ __forceinline__ uint32_t orderKey(float f)
{
    uint32_t const u = __float_as_uint(f);
    return u ^ ((u >> 31) != 0 ? 0xffffffffu : 0x80000000u);
}

// The barrier at shared address bar arrives once the MMAs issued so far are complete.
__device__ __forceinline__ void commitElect(uint32_t bar)
{
    asm volatile(
        "{\n"
        ".reg .pred e;\n"
        "elect.sync _|e, 0xffffffff;\n"
        "@e tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
        "}\n" ::"r"(bar)
        : "memory");
}

// Columns [taddr, taddr + kCols) of the warp's TMEM lane quarter (lane = row).
template <int kCols>
__device__ __forceinline__ void tmemLd(uint32_t taddr, float (&v)[kCols])
{
    static_assert(kCols == 8 || kCols == 16, "8 or 16 columns");
    uint32_t r[kCols];
    if constexpr (kCols == 8)
    {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "r"(taddr)
                     : "memory");
    }
    else
    {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
            "%15}, [%16];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
            "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
            : "r"(taddr)
            : "memory");
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#pragma unroll
    for (int i = 0; i < kCols; ++i)
    {
        v[i] = __uint_as_float(r[i]);
    }
}

// Columns [taddr, taddr + 24) of the warp's TMEM lane quarter as three 8-column tiles: both loads are issued before the
// one wait (the registers are tied to the wait, so no use moves above it).
__device__ __forceinline__ void tmemLd3x8(uint32_t taddr, float (&v)[3][8])
{
    uint32_t r[24];
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
        "%15}, [%16];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
        : "r"(taddr)
        : "memory");
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23])
        : "r"(taddr + 16)
        : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;"
                 : "+r"(r[0]), "+r"(r[1]), "+r"(r[2]), "+r"(r[3]), "+r"(r[4]), "+r"(r[5]), "+r"(r[6]), "+r"(r[7]),
                 "+r"(r[8]), "+r"(r[9]), "+r"(r[10]), "+r"(r[11]), "+r"(r[12]), "+r"(r[13]), "+r"(r[14]), "+r"(r[15]),
                 "+r"(r[16]), "+r"(r[17]), "+r"(r[18]), "+r"(r[19]), "+r"(r[20]), "+r"(r[21]), "+r"(r[22]), "+r"(r[23])
                 :
                 : "memory");
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            v[i][j] = __uint_as_float(r[8 * i + j]);
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Kernel context
// ----------------------------------------------------------------------------------------------------------------
struct Ctx
{
    int64_t const* p;
    uint8_t* smem;
    uint32_t smemBase;
    uint32_t* counters;
    int* err;
    uint64_t timeoutNs;
    uint8_t* arena; // this launch's arena copy
    int parity;
    int rank;
    uint64_t* stamps; // this CTA's stamps, or null
    int stampRole;    // the running unit's index in the CTA's list
    int64_t flags;    // variant bits (Flag)
    int numCopies;    // copies in the launch rotation
    int leader;       // a leader layer (kPLeader)
};

static_assert(sizeof(Ctx) <= 128, "Ctx exceeds its shared-memory slot");

#if GLM52_FLAGS_MODE != 0
#ifndef GLM52_STATIC_FLAGS
#define GLM52_STATIC_FLAGS 0ll
#endif
// Options selected per arm: the host passes an arm's flags as GLM52_STATIC_FLAGS, and these bits of them are fixed into
// the build (kFlagMergeDry, kFlagLandProbe, kFlagMoeL2, kFlagMoeL2Normal, kFlagSliceMerge, kFlagPageMap2; bits 43-45:
// the layout pad). kFlagFirstEarly,
// kFlagHopSelfClear, kFlagSelFast, kFlagPageBalance, kFlagAttnIlp, kFlagHopClearLate, kFlagWarmInHop, kFlagHPrime,
// kFlagXPrime, kFlagHop1Lines, kFlagRstdShared, kFlagXFrag, kFlagUnionSide, kFlagUnionLean and kFlagCandWarps are
// always on.
constexpr int64_t kArmOptionFlags
    = kFlagMergeDry | kFlagLandProbe | kFlagMoeL2 | kFlagMoeL2Normal | kFlagSliceMerge | kFlagPageMap2 | (7ll << 43);
constexpr int64_t kSpecializedDynamicFlags = kFlagEarlyX | kFlagNoStamps | kFlagNextChained;
constexpr int64_t kSpecializedStaticFlags = 704ll
#if GLM52_FLAGS_MODE >= 2
    | (1ll << 38) | (1ll << 46) | (1ll << 49)
#endif
#if GLM52_FLAGS_MODE == 3 || GLM52_FLAGS_MODE == 5
    | (1ll << 54)
#endif
    | kFlagFirstEarly | kFlagHopSelfClear | kFlagSelFast | kFlagPageBalance | kFlagAttnIlp | kFlagHopClearLate
    | kFlagWarmInHop | kFlagHPrime | kFlagXPrime | kFlagHop1Lines | kFlagRstdShared | kFlagXFrag | kFlagUnionSide
    | kFlagUnionLean | kFlagCandWarps | (static_cast<int64_t>(GLM52_STATIC_FLAGS) & kArmOptionFlags);
// The clear units the owners (and the next launch's fused-A) wait for: with kFlagHopSelfClear the hop buffers' units
// are not scheduled (the H1 units empty those slots), the top-k exchange's and the arena's units still count.
constexpr int kHopSlotClearUnits = 2 * kHopBytes / kClearUnitBytes;
constexpr int kCountedClear
    = kCountedClearUnits - ((kSpecializedStaticFlags & kFlagHopSelfClear) != 0 ? kHopSlotClearUnits : 0);
static_assert(kRouterLg == 0 || (kSpecializedStaticFlags & kFlagSliceMerge) == 0, "kArLg shares kArPStats' space");
static_assert(kRouterLg == 0
        || ((kSpecializedStaticFlags & kFlagHPrime) != 0 && (kSpecializedStaticFlags & kFlagRstdShared) != 0),
    "the reduced logits are h' logits: the routing scales them by 1 / rms");

// Inline at each read, including noinline helper bodies: optional branches
// become constants while capture-dependent EarlyX/NoStamps remain runtime.
__device__ __forceinline__ int64_t specializedFlags(Ctx const& ctx)
{
    return (ctx.flags & kSpecializedDynamicFlags) | kSpecializedStaticFlags;
}
#endif

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
    if (GLM52_STAMPS(ctx) != nullptr && threadIdx.x == 0)
    {
        GLM52_STAMPS(ctx)[kMaxRoles * 4 + 4 + k] = globalTimer();
        if (k >= 40 && k < 72)
        {
            GLM52_STAMPS(ctx)[kMaxRoles * 4 + 4 + 96 + k - 40] = clock64();
        }
    }
}

// The running unit's inputs are in (thread 0's view; the last call of the unit wins).
__device__ __forceinline__ void stampInput(Ctx const& ctx)
{
    if (GLM52_STAMPS(ctx) != nullptr && threadIdx.x == 0)
    {
        GLM52_STAMPS(ctx)[ctx.stampRole * 4 + 1] = globalTimer();
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

// One more turn of a spin. t0: the clock at the first turn, plus one per turn since (1 ns per turn of ~0.5 us); every
// 256 turns the watchdog runs. (Backing off here measured slower: 8 us per layer.)
__device__ __forceinline__ void spinCheck(Ctx const& ctx, uint64_t& t0, int code, int a, int b)
{
#if GLM52_LEAN
    return;
#endif
    if (t0 == 0)
    {
        t0 = globalTimer();
    }
    else if ((++t0 & 0xffull) == 0)
    {
        spinSlow(ctx, t0, code, a, b);
    }
}

// The exit test of a poll loop, taken by the whole warp at once (all 32 lanes call it). A lane that leaves a spin early
// waits at the reconvergence point while the others spin, and after a long spin it is let go on alone: the warp then
// stays split (a __syncwarp() after the loop is dropped by ptxas as redundant), so every later warp collective takes
// its slow path and the rest issues once per part (the leader top-k's candidate step ran 8.4 instead of 2.5 us).
__device__ __forceinline__ bool warpAll(bool ok)
{
    return __all_sync(0xffffffffu, ok);
}

// ----------------------------------------------------------------------------------------------------------------
// Data as its own flag: every 32-bit word of a buffer is kEmpty until its producer stores it.
// ----------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lamport(uint32_t w)
{
    return w == kEmpty ? 0u : w;
}

__device__ __forceinline__ bool full4(uint4 v)
{
    return (v.x != kEmpty) & (v.y != kEmpty) & (v.z != kEmpty) & (v.w != kEmpty);
}

__device__ __forceinline__ bool full2(uint2 v)
{
    return (v.x != kEmpty) & (v.y != kEmpty);
}

__device__ __forceinline__ uint4 poll4(Ctx const& ctx, void const* p, int code)
{
    uint4 v = ldPoll128(p);
    uint64_t t0 = 0;
    while (!full4(v))
    {
        spinCheck(ctx, t0, code, 0, 0);
        v = ldPoll128(p);
    }
    return v;
}

__device__ __forceinline__ uint32_t poll1(Ctx const& ctx, void const* p, int code)
{
    uint32_t v = ldPoll32(p);
    uint64_t t0 = 0;
    while (v == kEmpty)
    {
        spinCheck(ctx, t0, code, 0, 0);
        v = ldPoll32(p);
    }
    return v;
}

// kN 16-byte vectors at src + stride * i: all loads in flight together, then again for the ones not complete.
template <int kN>
__device__ __forceinline__ void pollStrided(Ctx const& ctx, uint4 (&v)[kN], uint8_t const* src, int stride, int code)
{
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        v[i] = ldPoll128(src + stride * i);
    }
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
        spinCheck(ctx, t0, code, stride, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(v[i]))
            {
                v[i] = ldPoll128(src + stride * i);
            }
        }
    }
}

// nVec 16-byte vectors from src to shared memory at dst; vector j is handled by thread j mod nThreads of the threads
// [first, first + nThreads) (kN >= its count).
template <int kN>
__device__ __forceinline__ void pollCopy(
    Ctx const& ctx, uint8_t* dst, uint8_t const* src, int nVec, int first, int nThreads, int code)
{
    int const me = static_cast<int>(threadIdx.x) - first;
    uint4 v[kN];
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        int const j = me + nThreads * i;
        v[i] = j < nVec ? ldPoll128(src + 16 * j) : make_uint4(0, 0, 0, 0);
    }
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
        spinCheck(ctx, t0, code, nVec, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(v[i]))
            {
                v[i] = ldPoll128(src + 16 * (me + nThreads * i));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        int const j = me + nThreads * i;
        if (j < nVec)
        {
            *reinterpret_cast<uint4*>(dst + 16 * j) = v[i];
        }
    }
}

// Warp 0 polls one canary word per producer unit (addrOf(i), i < n <= 64) until none is empty; then the CTA proceeds.
template <typename AddrOf>
__device__ __forceinline__ void waitCanaries(Ctx const& ctx, int n, AddrOf addrOf, int code)
{
    if (threadIdx.x < 32)
    {
        int const lane = threadIdx.x;
        uint8_t const* a0 = lane < n ? addrOf(lane) : nullptr;
        uint8_t const* a1 = lane + 32 < n ? addrOf(lane + 32) : nullptr;
        uint32_t v0 = a0 != nullptr ? ldPoll32(a0) : 0u;
        uint32_t v1 = a1 != nullptr ? ldPoll32(a1) : 0u;
        uint64_t t0 = 0;
        while (!__all_sync(0xffffffffu, (v0 != kEmpty) & (v1 != kEmpty)))
        {
            spinCheck(ctx, t0, code, n, 0);
            if (v0 == kEmpty)
            {
                v0 = ldPoll32(a0);
            }
            if (v1 == kEmpty)
            {
                v1 = ldPoll32(a1);
            }
        }
    }
    __syncthreads();
}

// Thread 0 waits until counter >= target; the CTA then proceeds (bar.sync orders the CTA's later reads).
__device__ __forceinline__ void waitCounter(Ctx const& ctx, int idx, uint32_t target, int code)
{
    if (threadIdx.x == 0)
    {
        uint32_t const* c = ctx.counters + idx;
        uint64_t t0 = 0;
        // One acquire load per turn (the watchdog takes the value already read: a second load doubled the turn).
        uint32_t v;
        while ((v = ldAcquire(c)) < target)
        {
            spinCheck(ctx, t0, code, idx, static_cast<int>(v));
        }
    }
    __syncthreads();
}

// Every CTA of the grid: the CTA's global writes so far, then +1 on the counter, then wait until all CTAs have
// arrived (the counter's set is this launch's; the CTA then proceeds). The barrier orders the CTA's writes before
// thread 0's release (red.release.gpu: MEMBAR.ALL.GPU, then the add), so no separate fence.
__device__ __forceinline__ void gridBarrier(Ctx const& ctx, int idx, int code)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        redReleaseAdd(ctx.counters + idx, 1u);
        uint64_t t0 = 0;
        while (ldAcquire(ctx.counters + idx) < static_cast<uint32_t>(kCtas))
        {
            spinCheck(ctx, t0, code, idx, 0);
        }
    }
    __syncthreads();
}

// The CTA's global writes so far, then +1 on the counter (as gridBarrier: the barrier, then one release).
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
    if (GLM52_STAMPS(ctx) != nullptr && threadIdx.x == 0)
    {
        uint64_t* s = GLM52_STAMPS(ctx) + ctx.stampRole * 4 + 2;
        if (*s == 0)
        {
            *s = globalTimer();
        }
    }
}

// Bulk copy of [src, src + bytes) to shared offset dst on barrier bar (thread 0 only), one request (a request takes
// ~0.3 us to issue); with evictFirst under the L2 evict-first policy.
__device__ __forceinline__ void copyPieces(
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

// ----------------------------------------------------------------------------------------------------------------
// GEMM building blocks: M = weight rows (16 per tile), N = 8 rows, K in 64-wide chunks.
//
// A packed (BF16): per (tile, chunk) 2048 B = 4 pieces x 32 threads x 16 B; thread (g, c) holds row g's elements
// [64q + 16c, +16) in pieces 0-1 and row g + 8's in pieces 2-3. A packed (NVFP4): per (tile, chunk) 512 B = 32 threads
// x 16 B (row g's bytes [32q + 8c, +8), then row g + 8's), and 64 B of scales (thread: row g's and row g + 8's E4M3
// scale of the 16-block 4q + c). B natural [8][K]: thread (g, c) reads row g's elements [64q + 16c, +16).
// In k step s of a chunk, MMA k slots {2c, 2c + 1} take elements 4s, 4s + 1 of the thread's 16, slots {2c + 8, 2c + 9}
// take 4s + 2, 4s + 3, in A and in B alike.
// ----------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void loadB(uint4 (&b)[2], T const* gB, int ldb, int chunk)
{
    int const lane = threadIdx.x & 31;
    T const* src = gB + static_cast<size_t>(lane >> 2) * ldb + chunk * 64 + 16 * (lane & 3);
    b[0] = ldcg128(src);
    b[1] = ldcg128(src + 8);
}

template <typename T>
__device__ __forceinline__ void loadBSmem(uint4 (&b)[2], T const* sB, int ldb, int chunk)
{
    int const lane = threadIdx.x & 31;
    T const* src = sB + (lane >> 2) * ldb + chunk * 64 + 16 * (lane & 3);
    b[0] = *reinterpret_cast<uint4 const*>(src);
    b[1] = *reinterpret_cast<uint4 const*>(src + 8);
}

// The B fragments of chunks [chunk0, chunk0 + kN) of a 16-bit [8][ld] activation in the arena, polled.
template <int kN>
__device__ __forceinline__ void pollB(
    Ctx const& ctx, uint4 (&b)[kN][2], uint8_t const* gB, int ld, int chunk0, int code)
{
    int const lane = threadIdx.x & 31;
    uint8_t const* src = gB + ((lane >> 2) * ld + chunk0 * 64 + 16 * (lane & 3)) * 2;
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        b[i][0] = ldPoll128(src + 128 * i);
        b[i][1] = ldPoll128(src + 128 * i + 16);
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(b[i][0]) & full4(b[i][1]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, chunk0, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(b[i][0]))
            {
                b[i][0] = ldPoll128(src + 128 * i);
            }
            if (!full4(b[i][1]))
            {
                b[i][1] = ldPoll128(src + 128 * i + 16);
            }
        }
    }
}

// Byte offset of x' (token t, column col) in the arena's x (kFlagXFrag: fused-A's fragment order, else [8][6144]).
__device__ __forceinline__ size_t xArOff(int t, int col)
{
    if constexpr ((kSpecializedStaticFlags & kFlagXFrag) != 0)
    {
        return static_cast<size_t>(col >> 6) * 1024 + ((col >> 3) & 1) * 512 + (4 * t + ((col >> 4) & 3)) * 16
            + 2 * (col & 7);
    }
    return (static_cast<size_t>(t) * kHidden + col) * 2;
}

// pollB for x' in fragment order (kFlagXFrag): chunk j's fragment b[.][h] of lane l at j 1024 + 512 h + 16 l.
template <int kN>
__device__ __forceinline__ void pollBFrag(Ctx const& ctx, uint4 (&b)[kN][2], uint8_t const* gB, int chunk0, int code)
{
    int const lane = threadIdx.x & 31;
    uint8_t const* src = gB + chunk0 * 1024 + 16 * lane;
#pragma unroll
    for (int i = 0; i < kN; ++i)
    {
        b[i][0] = ldPoll128(src + 1024 * i);
        b[i][1] = ldPoll128(src + 1024 * i + 512);
    }
    uint64_t t0 = 0;
    while (true)
    {
        bool ok = true;
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            ok &= full4(b[i][0]) & full4(b[i][1]);
        }
        if (warpAll(ok))
        {
            break;
        }
        spinCheck(ctx, t0, code, chunk0, 0);
#pragma unroll
        for (int i = 0; i < kN; ++i)
        {
            if (!full4(b[i][0]))
            {
                b[i][0] = ldPoll128(src + 1024 * i);
            }
            if (!full4(b[i][1]))
            {
                b[i][1] = ldPoll128(src + 1024 * i + 512);
            }
        }
    }
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

// Sum of the warps' accumulators of kTiles tiles in warp order, as out[row][token] (FP32) in the reduction area.
// Rows: 16 * tile + (g, g + 8); tokens 2c, 2c + 1.
template <int kTiles>
__device__ __forceinline__ void reduceWarps(Ctx const& ctx, float const (&acc)[kTiles][4], float* out)
{
    float* red = reinterpret_cast<float*>(ctx.smem + kSmemRed);
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
    {
        *reinterpret_cast<float4*>(red + ((warp * kTiles + t) * 32 + lane) * 4)
            = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kTiles * 32 * 4; i += kThreads)
    {
        int const t = i / 128;
        int const l = (i / 4) % 32;
        int const r = i % 4;
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            s += red[((w * kTiles + t) * 32 + l) * 4 + r];
        }
        int const row = 16 * t + (l >> 2) + (r >= 2 ? 8 : 0);
        int const tok = 2 * (l & 3) + (r & 1);
        out[row * kTok + tok] = s;
    }
    __syncthreads();
}

// Threads 0-63: rows [16 tile, +16) of out[row][token] as BF16 pairs into the [8][ld] activation at dst (one word
// each: token t, rows 2 pr and 2 pr + 1), times scale[t] if given.
__device__ __forceinline__ void storeTileBf16(
    float const* out, uint8_t* dst, int ld, int row0, float const* scale = nullptr)
{
    if (threadIdx.x < 64)
    {
        int const tok = threadIdx.x / 8;
        int const pr = threadIdx.x % 8;
        float const sc = scale != nullptr ? scale[tok] : 1.f;
        uint32_t const w = packBf16(out[(2 * pr) * kTok + tok] * sc, out[(2 * pr + 1) * kTok + tok] * sc);
        st32(dst + (tok * ld + row0 + 2 * pr) * 2, lamport(w));
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Unit weights: where they come from and how many bytes (static units only).
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
    case kStFusedA:
        src0 = ptr<uint8_t const>(ctx, kPWFa) + static_cast<size_t>(unit) * kFaChunks * kBf16Block;
        n0 = kFaChunks * kBf16Block;
        break;
    case kStQb:
        src0 = ptr<uint8_t const>(ctx, kPWQb) + static_cast<size_t>(unit) * kQbUnitTiles * kQbChunks * kBf16Block;
        n0 = kQbUnitTiles * kQbChunks * kBf16Block;
        break;
    case kStOproj:
        src0 = ptr<uint8_t const>(ctx, kPWO) + static_cast<size_t>(unit) * 2 * kOChunks * kBf16Block;
        n0 = 2 * kOChunks * kBf16Block;
        break;
    case kStRouterGemv:
    {
        int const et = unit / 4;
        int const kq = unit % 4;
        src0 = ptr<uint8_t const>(ctx, kPWR) + (static_cast<size_t>(et) * kRChunks + 24 * kq) * kBf16Block;
        n0 = 24 * kBf16Block;
        break;
    }
    case kStIdxQ:
        src0 = ptr<uint8_t const>(ctx, kPIdxWq) + static_cast<size_t>(unit) * 2 * kIdxQChunks * kBf16Block;
        n0 = 2 * kIdxQChunks * kBf16Block;
        break;
    case kStOprojLg:
        // o_proj's two tiles, then the logit tile's O part (chunks 0-31; the residual chunks' fragments come from
        // global memory).
        src0 = ptr<uint8_t const>(ctx, kPWO) + static_cast<size_t>(unit) * 2 * kOChunks * kBf16Block;
        n0 = 2 * kOChunks * kBf16Block;
        src1 = ptr<uint8_t const>(ctx, kPWLg) + static_cast<size_t>(unit - kLgOprojFirst) * kLgChunks * kBf16Block;
        n1 = kOChunks * kBf16Block;
        break;
    case kStSharedGu:
    {
        int const ft = unit / 4;
        int const kq = unit % 4;
        src0 = ptr<uint8_t const>(ctx, kPWGu) + (static_cast<size_t>(ft) * kRChunks + 24 * kq) * kBf16Block;
        src1 = ptr<uint8_t const>(ctx, kPWGu) + (static_cast<size_t>(8 + ft) * kRChunks + 24 * kq) * kBf16Block;
        n0 = 24 * kBf16Block;
        n1 = 24 * kBf16Block;
        break;
    }
    default: break;
    }
}

// Thread 0 at entry (kFlagL2Deferred, leaders): the indexer Q weights of the CTA's IdxQ unit, whose copy waits for
// fused-A's slot, into L2 behind the first requests.
__device__ __noinline__ void prefetchDeferred(Ctx const& ctx, int const* sRoles)
{
#pragma unroll 1
    for (int i = 0; i < kMaxRoles && sRoles[i] != 0; ++i)
    {
        if ((sRoles[i] >> 16) == kStIdxQ)
        {
            uint8_t const* src0;
            uint8_t const* src1;
            uint32_t n0, n1;
            unitWeights(ctx, kStIdxQ, sRoles[i] & 0xffff, src0, n0, src1, n1);
            prefetchBulkL2(src0, n0);
        }
    }
}

#if GLM52_SETUP_MODE == 0
__device__ __noinline__ void issueUnitWeights(Ctx const& ctx, int role, int smemOff, uint32_t bar)
{
    // Thread 0 only; every thread's earlier shared-memory reads are complete (the caller synchronized).
    uint8_t const* src0;
    uint8_t const* src1;
    uint32_t n0, n1;
    unitWeights(ctx, role >> 16, role & 0xffff, src0, n0, src1, n1);
    if (n0 + n1 == 0)
    {
        return;
    }
    fenceProxyAsyncSmem();
    mbarExpectTx(bar, n0 + n1);
    bool const evictFirst = (GLM52_FLAGS(ctx) & kFlagEvictFirst) != 0;
    copyPieces(ctx.smemBase + smemOff, src0, n0, bar, evictFirst);
    if (n1)
    {
        copyPieces(ctx.smemBase + smemOff + n0, src1, n1, bar, evictFirst);
    }
}
#else
__device__ __noinline__ void issueUnitWeights(Ctx const& ctx, int j, uint32_t bar)
{
    // Thread 0 after the original caller's join; the per-layer table is immutable.
    int const* roles = reinterpret_cast<int const*>(ctx.smem + kSmemSmall + kSmallRoles);
    int64_t const* w = reinterpret_cast<int64_t const*>(ctx.smem + kSmemSmall + kSmallWeights) + 3 * j;
    uint8_t const* src0 = reinterpret_cast<uint8_t const*>(w[0]);
    uint8_t const* src1 = reinterpret_cast<uint8_t const*>(w[1]);
    uint32_t const n0 = static_cast<uint32_t>(w[2]);
    uint32_t const n1 = static_cast<uint32_t>(static_cast<uint64_t>(w[2]) >> 32);
    if (n0 + n1 == 0)
        return;
    uint32_t const dst = ctx.smemBase + static_cast<uint32_t>(roles[kMaxRoles + j]);
    fenceProxyAsyncSmem();
    mbarExpectTx(bar, n0 + n1);
    bool const evictFirst = (GLM52_FLAGS(ctx) & kFlagEvictFirst) != 0;
    copyPieces(dst, src0, n0, bar, evictFirst);
    if (n1)
        copyPieces(dst + n0, src1, n1, bar, evictFirst);
}

// Filled once, before capture, using the unchanged address/length function.
__global__ void glm52WeightTableKernel(int64_t const* params, int64_t* out)
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
        unitWeights(ctx, role >> 16, role & 0xffff, src0, n0, src1, n1);
    int64_t* o = out + (static_cast<int64_t>(c) * kMaxRoles + j) * 3;
    o[0] = reinterpret_cast<int64_t>(src0);
    o[1] = reinterpret_cast<int64_t>(src1);
    o[2] = static_cast<int64_t>(n0) | (static_cast<int64_t>(n1) << 32);
}
#endif

// Thread 0, inside unit `blocker` once its weight slot is free: the weights of its stage-`stage` units waiting for that
// slot go out now, and those units are marked issued for the main loop.
__device__ __noinline__ void issueBlockedEarly(Ctx const& ctx, int blocker, int stage)
{
    int* sRoles = reinterpret_cast<int*>(ctx.smem + kSmemSmall + kSmallRoles);
#pragma unroll 1
    for (int j = blocker + 1; j < kMaxRoles && sRoles[j] != 0; ++j)
    {
        if ((sRoles[j] >> 16) == stage && sRoles[2 * kMaxRoles + j] == blocker)
        {
#if GLM52_SETUP_MODE == 0
            issueUnitWeights(ctx, sRoles[j], sRoles[kMaxRoles + j], ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
#else
            issueUnitWeights(ctx, j, ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
#endif
            sRoles[2 * kMaxRoles + j] = -3;
        }
    }
}

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_107__low_latency/csrc/glmMegakernelStages.cuh"

// ----------------------------------------------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------------------------------------------
// Runs one unit of the CTA's list.
__device__ __forceinline__ void runUnit(Ctx const& c, int stage, int unit, uint32_t off, uint32_t bar,
    __nv_bfloat16 const* x, __nv_bfloat16 const* residual, __nv_bfloat16* xOut, __nv_bfloat16* resOut, int const* topk,
    int const* blockTable, int const* kvLen)
{
    switch (stage)
    {
    case kStFusedA: stageFusedA(c, unit, off, bar, x); break;
    case kStQb: stageQb(c, unit, off, bar, kvLen); break;
    case kStKvAppend: stageKvAppend(c, kvLen, blockTable); break;
    case kStIdxK: stageIdxK(c, kvLen); break;
    case kStIdxQ: stageIdxQ(c, unit, off, bar, kvLen); break;
    case kStIdxSel: stageIdxSel(c, off, kvLen); break;
    case kStAttn: stageAttn(c, unit, off, kvLen, blockTable, topk); break;
    case kStMerge: stageMerge(c, unit); break;
    case kStOproj: stageOproj(c, unit, off, bar); break;
    case kStClear: stageClear(c, unit); break;
    case kStH1: stageHop(c, unit, false, residual, xOut, resOut); break;
    case kStRouterGemv:
        if constexpr (kRouterLg != 2)
        {
            stageRouterGemv(c, unit, off, bar);
        }
        break;
    case kStOprojLg:
        if constexpr (kRouterLg != 0)
        {
            stageOprojLg(c, unit, off, bar, residual);
        }
        break;
    case kStSharedGu: stageSharedGu(c, unit, off, bar); break;
    case kStMoe: stageMoe(c, (GLM52_FLAGS(c) & kFlagOsInMoe) ? unit - 1 : -1, off); break;
    case kStOwnerShared:
        if (!(GLM52_FLAGS(c) & kFlagOsInMoe))
        {
            stageOwnerShared(c, unit);
        }
        break;
    case kStOwner: stageOwner(c, unit); break;
    case kStH2: stageHop(c, unit, true, residual, xOut, resOut); break;
    default: break;
    }
}

__global__ void __launch_bounds__(kThreads, 1) glm52PlainLayerKernel(int64_t const* __restrict__ params,
    __nv_bfloat16 const* __restrict__ x, __nv_bfloat16 const* __restrict__ residual, __nv_bfloat16* __restrict__ xOut,
    __nv_bfloat16* __restrict__ resOut, int const* __restrict__ topk, int const* __restrict__ blockTable,
    int const* __restrict__ kvLen)
{
    extern __shared__ __align__(1024) uint8_t smemRaw[];
#if GLM52_SETUP_MODE == 1
    // kFlagFirstEarly: the first unit's weights (when the setup would request them) before anything else.
    bool firstIssued = false;
    if constexpr ((kSpecializedStaticFlags & kFlagFirstEarly) != 0)
    {
        if (threadIdx.x == 0)
        {
            int const* tables = reinterpret_cast<int const*>(params + kPNumParams);
            int64_t const* wt
                = params + kPNumParams + 3 * kCtas * kMaxRoles / 2 + static_cast<size_t>(blockIdx.x) * 3 * kMaxRoles;
            int const issue0 = __ldg(tables + (2 * kCtas + static_cast<int>(blockIdx.x)) * kMaxRoles);
            int const off0 = __ldg(tables + (kCtas + static_cast<int>(blockIdx.x)) * kMaxRoles);
            int64_t const w0 = __ldg(wt);
            int64_t const w1 = __ldg(wt + 1);
            int64_t const w2 = __ldg(wt + 2);
            uint32_t const n0 = static_cast<uint32_t>(w2);
            uint32_t const n1 = static_cast<uint32_t>(static_cast<uint64_t>(w2) >> 32);
            if (issue0 == -1 && n0 + n1 != 0)
            {
                uint32_t const sb = smemU32(smemRaw);
                uint32_t const bar0 = sb + kSmemSmall + kSmallBars;
                mbarInit(bar0, 1);
                asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
                mbarExpectTx(bar0, n0 + n1);
                bool const evictFirst = (kSpecializedStaticFlags & kFlagEvictFirst) != 0;
                uint32_t const dst = sb + static_cast<uint32_t>(off0);
                copyPieces(dst, reinterpret_cast<uint8_t const*>(w0), n0, bar0, evictFirst);
                if (n1)
                {
                    copyPieces(dst + n0, reinterpret_cast<uint8_t const*>(w1), n1, bar0, evictFirst);
                }
                firstIssued = true;
            }
        }
    }
#endif
    // The parameter block into shared memory: every later read of it is a shared load (L1 is a few KiB next to the
    // units' shared memory, so global reads of it went to L2 again after each poll or barrier).
    int64_t* sParams = reinterpret_cast<int64_t*>(smemRaw + kSmemSmall + kSmallParams);
    int* sRoles = reinterpret_cast<int*>(smemRaw + kSmemSmall + kSmallRoles);
    if (threadIdx.x < kPNumParams)
    {
        sParams[threadIdx.x] = params[threadIdx.x];
    }
    // The CTA's rows of the role tables, appended to the parameter block (units, shared offsets, issue points), in the
    // same round trip.
    if (threadIdx.x < 3 * kMaxRoles)
    {
        int const* tables = reinterpret_cast<int const*>(params + kPNumParams);
        int const table = threadIdx.x / kMaxRoles;
        sRoles[threadIdx.x]
            = tables[(table * kCtas + static_cast<int>(blockIdx.x)) * kMaxRoles + threadIdx.x % kMaxRoles];
    }
#if GLM52_SETUP_MODE == 1
    if (threadIdx.x >= 64 && threadIdx.x < 64 + 3 * kMaxRoles)
    {
        int64_t const* table = params + kPNumParams + 3 * kCtas * kMaxRoles / 2;
        reinterpret_cast<int64_t*>(smemRaw + kSmemSmall + kSmallWeights)[threadIdx.x - 64]
            = table[static_cast<int>(blockIdx.x) * 3 * kMaxRoles + threadIdx.x - 64];
    }
#endif
    __syncthreads();
    params = sParams;
    // The context in shared memory: thread 0 fills it (its reads in the units are shared loads, never evicted).
    Ctx& ctx = *reinterpret_cast<Ctx*>(smemRaw + kSmemSmall + kSmallCtx);
    if (threadIdx.x == 0)
    {
        ctx.p = params;
        ctx.smem = smemRaw;
        ctx.smemBase = smemU32(smemRaw);
        ctx.parity = static_cast<int>(params[kPCopy]);
        ctx.counters = reinterpret_cast<uint32_t*>(params[kPCounters]) + ctx.parity * kNumCounters;
        ctx.err = reinterpret_cast<int*>(params[kPErr]);
        ctx.timeoutNs = static_cast<uint64_t>(params[kPTimeoutNs]);
        ctx.rank = static_cast<int>(params[kPRank]);
        ctx.stamps = params[kPStamps] != 0 && !(params[kPFlags] & kFlagNoStamps)
            ? reinterpret_cast<uint64_t*>(params[kPStamps]) + static_cast<size_t>(blockIdx.x) * kStampsPerCta
            : nullptr;
        ctx.stampRole = 0;
        ctx.flags = params[kPFlags];
        ctx.numCopies = static_cast<int>(params[kPNumCopies]);
        ctx.leader = static_cast<int>(params[kPLeader]);
        ctx.arena = reinterpret_cast<uint8_t*>(params[kPArena]) + static_cast<size_t>(ctx.parity) * kArenaBytes;
    }
    __syncthreads();
    int const tid = threadIdx.x;
    uint64_t const entry = globalTimer();
    // The weights first: thread 0 initializes the barriers and requests the weights of the units that need no free
    // slot right away (a CTA that starts late, on an SM the previous layer released last, has them in flight ~1.5 us
    // sooner than after the rest of the setup); warp 1 takes the MoE stage's TMEM meanwhile. The TMEM is taken before
    // the dependents are released: a later kernel's CTA sharing this SM then waits for it instead of holding TMEM this
    // CTA needs while it waits for this grid (released at the MoE stage's end).
#if GLM52_SETUP_MODE == 0
    if (tid == 0)
    {
#pragma unroll 1
        for (int i = 0; i < kNumBars; ++i)
        {
            mbarInit(ctx.smemBase + kSmemSmall + kSmallBars + 8 * i, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll 1
        for (int i = 0; i < kMaxRoles && sRoles[i] != 0; ++i)
        {
            if (sRoles[2 * kMaxRoles + i] == -1)
            {
#if GLM52_SETUP_MODE == 0
                issueUnitWeights(ctx, sRoles[i], sRoles[kMaxRoles + i], ctx.smemBase + kSmemSmall + kSmallBars + 8 * i);
#else
                issueUnitWeights(ctx, i, ctx.smemBase + kSmemSmall + kSmallBars + 8 * i);
#endif
            }
        }
        if (ctx.leader && (GLM52_FLAGS(ctx) & kFlagL2Deferred))
        {
            prefetchDeferred(ctx, sRoles);
        }
    }
#else
    uint64_t landIssued = 0;
    if (tid < 32)
    {
        static_assert(kNumBars <= 32, "one lane per barrier");
        if (tid < kNumBars && !(tid == 0 && firstIssued))
            mbarInit(ctx.smemBase + kSmemSmall + kSmallBars + 8 * tid, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        __syncwarp();
        if (tid == 0)
        {
#pragma unroll 1
            for (int i = 0; i < kMaxRoles && sRoles[i] != 0; ++i)
            {
                if (sRoles[2 * kMaxRoles + i] == -1 && !(i == 0 && firstIssued))
                    issueUnitWeights(ctx, i, ctx.smemBase + kSmemSmall + kSmallBars + 8 * i);
            }
            if constexpr ((kSpecializedStaticFlags & kFlagLandProbe) != 0)
            {
                landIssued = globalTimer();
            }
            if (ctx.leader && (GLM52_FLAGS(ctx) & kFlagL2Deferred))
                prefetchDeferred(ctx, sRoles);
        }
    }
#endif
    else if (tid >= 32 && tid < 64)
    {
        tmemAlloc(smemU32(smemRaw + kSmemSmall + kSmallScalars + 4 * 120), kTmemCols);
    }
    // The stamps (one word per thread; the CTA's synchronization below orders them before the units' stamps).
    if (GLM52_STAMPS(ctx) != nullptr && tid >= 64)
    {
        for (int i = tid - 64; i < kStampsPerCta; i += kThreads - 64)
        {
            GLM52_STAMPS(ctx)[i] = i == kMaxRoles * 4 ? entry : 0ull;
        }
    }
    // The SWIZZLE_64B operands need 512-byte aligned shared addresses (the MoE ring and h start at offsets of 512 B).
    if (params[kPHopBytes] != kHopBytes || params[kPMagic] != kMagic || (ctx.smemBase & 1023u) != 0
        || params[kPMaxRoles] != kMaxRoles)
    {
        __trap();
    }
    reinterpret_cast<int*>(smemRaw + kSmemSmall + kSmallMoe)[tid] = 0;
    tcFenceBefore();
    __syncthreads();
    tcFenceAfter();
    int nRoles = 0;
#pragma unroll 1
    while (nRoles < kMaxRoles && sRoles[nRoles] != 0)
    {
        ++nRoles;
    }
    // The next launch's counters (last used by launch n + 1 - copies <= n - 2, finished before this launch began).
    {
        uint32_t* next
            = reinterpret_cast<uint32_t*>(params[kPCounters]) + ((ctx.parity + 1) % ctx.numCopies) * kNumCounters;
        for (int i = static_cast<int>(blockIdx.x) * kThreads + tid; i < kNumCounters; i += kCtas * kThreads)
        {
            next[i] = 0;
        }
    }
    // The previous kernel's outputs (x, residual, the step inputs, the leader's top-k) are read after the
    // grid-dependency wait; with kFlagEarlyX a CTA whose first unit is fused-A runs it before the wait, its x from the
    // arena.
    bool const early = (GLM52_FLAGS(ctx) & kFlagEarlyX) && nRoles > 0 && (sRoles[0] >> 16) == kStFusedA;
    auto dependencyWait = [&]()
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
        {
            GLM52_STAMPS(ctx)[kMaxRoles * 4 + 1] = globalTimer();
        }
        if (early && (GLM52_FLAGS(ctx) & kFlagCheckX) && blockIdx.x == 0)
        {
            uint8_t const* axb = arena<uint8_t const>(ctx, kArX);
            uint32_t const* gx = reinterpret_cast<uint32_t const*>(x);
            for (int i = tid; i < kTok * kHidden / 2; i += kThreads)
            {
                uint32_t const* const aw
                    = reinterpret_cast<uint32_t const*>(axb + xArOff(2 * i / kHidden, 2 * i % kHidden));
                if constexpr ((kSpecializedStaticFlags & kFlagXPrime) != 0)
                {
                    // x' x 1 / rms against x: the same math, rounded at other points (about 2^-7 relative apart).
                    float const* q = arena<float const>(ctx, kArXSsq) + 4 * (2 * i / kHidden);
                    float const r = rsqrtf(((((0.f + __ldcg(q)) + __ldcg(q + 1)) + __ldcg(q + 2)) + __ldcg(q + 3))
                            / static_cast<float>(kHidden)
                        + kEps);
                    uint32_t const a = __ldcg(aw);
                    uint32_t const g = __ldcg(gx + i);
                    if (fabsf(bf16Lo(a) * r - bf16Lo(g)) > 0.015625f * fabsf(bf16Lo(g)) + 1e-30f
                        || fabsf(bf16Hi(a) * r - bf16Hi(g)) > 0.015625f * fabsf(bf16Hi(g)) + 1e-30f)
                    {
                        fail(ctx, 98, i, static_cast<int>(a));
                    }
                }
                else if (__ldcg(aw) != lamport(__ldcg(gx + i)))
                {
                    fail(ctx, 98, i, static_cast<int>(__ldcg(aw)));
                }
            }
        }
    };
    if constexpr ((kSpecializedStaticFlags & kFlagLandProbe) != 0)
    {
        if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
        {
            GLM52_STAMPS(ctx)[kMaxRoles * 4 + 4 + 128] = landIssued;
        }
    }
    if (!early)
    {
        dependencyWait();
    }

    for (int i = 0; i < nRoles; ++i)
    {
        if (early && i == 1)
        {
            dependencyWait();
        }
        int const role = sRoles[i];
        int const stage = role >> 16;
        int const unit = role & 0xffff;
        uint32_t const off = static_cast<uint32_t>(sRoles[kMaxRoles + i]);
        uint32_t const bar = ctx.smemBase + kSmemSmall + kSmallBars + 8 * i;
        if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
        {
            GLM52_STAMPS(ctx)[i * 4] = globalTimer();
        }
        if (tid == 0)
        {
            ctx.stampRole = i; // read by thread 0 only (its stamps)
        }
        if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
        {
            // The CTA's progress in host memory (stamped runs): after a trap, the unit each CTA was in.
            reinterpret_cast<int volatile*>(ctx.err)[16 + 4 * kCtas + blockIdx.x] = (stage << 16) | (i << 8) | 1;
        }
        runUnit(ctx, stage, unit, off, bar, x, residual, xOut, resOut, topk, blockTable, kvLen);
        // Every thread fences its generic shared accesses of the unit for the async proxy before the join (the
        // documented pattern): later units' weight copies and gathers, issued by one thread after it, land on them.
        fenceProxyAsyncSmem();
        __syncthreads();
        if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
        {
            GLM52_STAMPS(ctx)[i * 4 + 3] = globalTimer();
        }
        if (stage == kStMoe && (GLM52_FLAGS(ctx) & kFlagHoldExit) && tid == 0)
        {
            atomicAdd(ctx.counters + kCMoeDone, 1u);
        }
        if (tid == 0)
        {
            bool const attnNext = i + 1 < nRoles && (sRoles[i + 1] >> 16) == kStAttn;
#pragma unroll 1
            for (int j = i + 1; j < nRoles; ++j)
            {
                if (sRoles[2 * kMaxRoles + j] == i)
                {
                    if (attnNext && (sRoles[j] >> 16) == kStOproj)
                    {
                        sRoles[2 * kMaxRoles + j] = -4; // requested by the attention unit (after its gathers / Q)
                    }
                    else
                    {
#if GLM52_SETUP_MODE == 0
                        issueUnitWeights(
                            ctx, sRoles[j], sRoles[kMaxRoles + j], ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
#else
                        issueUnitWeights(ctx, j, ctx.smemBase + kSmemSmall + kSmallBars + 8 * j);
#endif
                    }
                }
            }
        }
    }

    if (early && nRoles == 1)
    {
        dependencyWait();
    }
    __syncthreads();
    if (GLM52_FLAGS(ctx) & kFlagHoldExit)
    {
        waitCounter(ctx, kCMoeDone, kCtas, 150);
    }
    if (GLM52_STAMPS(ctx) != nullptr && tid == 0)
    {
        GLM52_STAMPS(ctx)[kMaxRoles * 4 + 2] = globalTimer();
        reinterpret_cast<int volatile*>(ctx.err)[16 + 4 * kCtas + blockIdx.x] = 0xffff;
    }
    // The layout pad: code placement alone moves the slice by up to ~7 us (lessons/megakernel/code-placement-moves-the-
    // slice.md), so the stage functions' offset in the 128-byte fetch line is a build parameter (k never-run stores).
    constexpr int kLayoutPad = static_cast<int>((kSpecializedStaticFlags >> 43) & 7);
    if constexpr (kLayoutPad > 0)
    {
        if (ctx.p[kPMagic] == 0)
        {
            uint32_t const a = smemU32(smemRaw);
            asm volatile("st.shared.u32 [%0], %1;" ::"r"(a), "r"(tid) : "memory");
            if constexpr (kLayoutPad > 1)
            {
                asm volatile("st.shared.u32 [%0+16], %1;" ::"r"(a), "r"(tid) : "memory");
            }
            if constexpr (kLayoutPad > 2)
            {
                asm volatile("st.shared.u32 [%0+32], %1;" ::"r"(a), "r"(tid) : "memory");
            }
            if constexpr (kLayoutPad > 3)
            {
                asm volatile("st.shared.u32 [%0+48], %1;" ::"r"(a), "r"(tid) : "memory");
            }
            if constexpr (kLayoutPad > 4)
            {
                asm volatile("st.shared.u32 [%0+64], %1;" ::"r"(a), "r"(tid) : "memory");
            }
            if constexpr (kLayoutPad > 5)
            {
                asm volatile("st.shared.u32 [%0+80], %1;" ::"r"(a), "r"(tid) : "memory");
            }
            if constexpr (kLayoutPad > 6)
            {
                asm volatile("st.shared.u32 [%0+96], %1;" ::"r"(a), "r"(tid) : "memory");
            }
        }
    }
}

#else

__global__ void glm52PlainLayerKernel(int64_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    __nv_bfloat16*, int const*, int const*, int const*)
{
}

#if GLM52_SETUP_MODE == 1
__global__ void glm52WeightTableKernel(int64_t const*, int64_t*) {}
#endif

#endif

} // namespace glm52

// ----------------------------------------------------------------------------------------------------------------
// Native launch and layout queries for the Torch binding.
// ----------------------------------------------------------------------------------------------------------------

int glm52PlainLayerSmemBytes()
{
    return glm52::kSmemBytes;
}

int glm52PlainLayerNumParams()
{
    return glm52::kPNumParams;
}

int64_t glm52PlainLayerMagic()
{
    return glm52::kMagic;
}

int glm52PlainLayerArenaBytes()
{
    return glm52::kArenaBytes;
}

int glm52PlainLayerClearUnits()
{
    return glm52::kClearUnits;
}

int glm52PlainLayerNumCounters()
{
    return glm52::kNumCounters;
}

int glm52PlainLayerCandCap()
{
    return glm52::kCandCap;
}

int glm52PlainLayerMaxKeys()
{
    return glm52::kIdxMaxKeys;
}

// GLM52_ROUTER_LG of this build (the host's role tables: 1 adds the logit units, 2 also drops the router GEMV units).
int glm52PlainLayerRouterLg()
{
    return glm52::kRouterLg;
}

// GLM52_ROUTER_LG's multicast workspace per launch copy, after the top-k exchange's copies.
int glm52PlainLayerLgBytes()
{
    return glm52::kLgBytes;
}

// The MoE cut's class weight i (W, OS, SG, H1; glm52::kFc2ListsCut), for the host's check of slot 0's FC1 partials.
int glm52PlainLayerCutWeight(int i)
{
    return i >= 0 && i < 4 ? glm52::kFc2ListsCut[i] : 0;
}

// The 128-byte CUtensorMap of a leader's indexer K cache (rows of 128 B from pool, numRows of them) with SWIZZLE_128B
// boxes of 64 rows (one block's keys), written to out (host memory); a CUresult.
int glm52PlainLayerIdxTmap(void* out, void* pool, unsigned long long numRows)
{
    cuuint64_t dims[2] = {128, static_cast<cuuint64_t>(numRows)};
    cuuint64_t strides[1] = {128};
    cuuint32_t box[2] = {128, 64};
    cuuint32_t elem[2] = {1, 1};
    CUresult const r = cuTensorMapEncodeTiled(static_cast<CUtensorMap*>(out), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, pool,
        dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return static_cast<int>(r);
}

#if GLM52_SETUP_MODE == 1
int glm52PlainLayerWeightTable(void const* params, void* out, void* stream)
{
    glm52::glm52WeightTableKernel<<<glm52::kCtas, glm52::kMaxRoles, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t const*>(params), static_cast<int64_t*>(out));
    return static_cast<int>(cudaGetLastError());
}
#endif

int glm52PlainLayerInit()
{
#if defined(PLACEHOLDER_KERNELS)
    return static_cast<int>(cudaErrorNotSupported);
#else
    int dev = 0;
    int optInBytes = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optInBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (glm52::kSmemBytes > optInBytes)
    {
        // Larger than the portable opt-in: the oversized shared-memory mode (the max-dynamic-size attribute is
        // ignored in it).
        return static_cast<int>(cudaFuncSetAttribute(glm52::glm52PlainLayerKernel, cudaFuncAttributeSharedMemoryMode,
            cudaSharedMemoryModePreferOversizedSharedMemory));
    }
    cudaError_t e = cudaFuncSetAttribute(
        glm52::glm52PlainLayerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, glm52::kSmemBytes);
    if (e != cudaSuccess)
    {
        return static_cast<int>(e);
    }
    e = cudaFuncSetAttribute(
        glm52::glm52PlainLayerKernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    return static_cast<int>(e);
#endif
}

int glm52PlainLayerLaunch(void const* params, void const* x, void const* residual, void* xOut, void* resOut,
    void const* topk, void const* blockTable, void const* kvLen, void* stream, int pdl)
{
#if defined(PLACEHOLDER_KERNELS)
    return static_cast<int>(cudaErrorNotSupported);
#else
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(glm52::kCtas);
    config.blockDim = dim3(glm52::kThreads);
    config.dynamicSmemBytes = glm52::kSmemBytes;
    config.stream = static_cast<cudaStream_t>(stream);
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attrs[1].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
    attrs[1].val.sharedMemCarveout = 100;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaError_t const e = cudaLaunchKernelEx(&config, glm52::glm52PlainLayerKernel, static_cast<int64_t const*>(params),
        static_cast<__nv_bfloat16 const*>(x), static_cast<__nv_bfloat16 const*>(residual),
        static_cast<__nv_bfloat16*>(xOut), static_cast<__nv_bfloat16*>(resOut), static_cast<int const*>(topk),
        static_cast<int const*>(blockTable), static_cast<int const*>(kvLen));
    return static_cast<int>(e);
#endif
}

} // namespace kernels::glm_moe_dsa::megakernel

TRTLLM_MV2_NAMESPACE_END
