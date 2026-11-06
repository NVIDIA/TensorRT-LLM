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

// The target head's sampler fold on an interleaved vocabulary shard, with the online-softmax states exchanged
// instead of the logits (``distributed_head_sampler``; kernels/s2_sampler/sampler.py).
//
// The served sampler (ops/spec/_head_sampler/head_sampler.cu) folds a gathered BF16 row of V = 4 * vecCount logits:
// virtual thread vt (0 .. 1023) folds the vectors v = vt + 1024 j (4 logits each), in increasing j, each vector's
// four logits in order, into one OnlineSoftmax {max, sum, argmax}; kernel B reduces / scans the 1024 states. When
// rank r's shard holds the vectors v = world * m + r (m = 0 .. vecCount / world - 1; local row 4m + q is global row
// 4(world m + r) + q), virtual thread vt = world * a + r folds exactly the local vectors m = a + (1024 / world) j,
// all on rank r. So each rank folds its 1024 / world virtual threads per row from its own shard, with the served
// cooperative fold's statements (splitFoldCoopKernel: the soft cap of every element by the whole CTA into shared
// memory, then one thread per virtual thread), and every state is the served state bit for bit: the same function
// of the same elements in the same order. The argmax a state carries is the global element index 4 (vt + 1024 j) + e.
//
// Transport of the states: LL elements over the NVLink multicast fabric, no reset (the one-shot logits gather's
// protocol, kernels/pv_logits/pv_logits_gather.cu). A state travels as four 64-bit elements (epoch << 32) | word with
// the words {max, sum, argmax, 0}, written with two 16-byte multicast stores to (row, rank, a) of the area of the
// call's epoch parity, i.e. into every rank's copy; a reader polls its own copy with 16-byte relaxed system-scope loads
// until the four elements of a state carry the call's epoch. A 64-bit element is written and read as one access, so a
// matching epoch half implies the data half of the same call.
//
// Ownership and progress.
// - Writers: element (row, r, a, w) of rank R's copy is written only by rank r's fold kernel, CTA (row, a / VTS),
//   thread a % VTS, through the multicast address, once per call. Readers: only rank R's poll kernel or the draw
//   kernel (a call runs one of the two) reads rank R's copy. Nothing stores sentinels into fabric memory.
// - Epochs: the fold kernel's CTA b keeps foldCounters[b] and the poll kernel's CTA row keeps pollCounters[row] (this
//   rank's device memory, each read and written only by its own CTA): a call uses counter + 1 and stores it back at
//   the CTA's end. Grids are fixed per instance (rows x 1024 / world / VTS fold CTAs, rows poll CTAs), every call runs
//   every CTA of both kernels once, and every rank runs the same sequence of calls, so all CTAs of all ranks agree
//   on a call's epoch. The areas start zeroed (epoch 0 is never used); a stale element carries an older epoch.
// - Reuse: call e uses area e & 1. A peer writes the area rank R reads in call e next in its call e + 2. The peer's
//   fold of call e + 2 runs after its poll of call e + 1 completed (stream order through the kernels between them,
//   each of which waits for its predecessor before it completes), and that poll completed only after it saw rank R's
//   call-(e + 1) states, which rank R's fold publishes only after its own poll of call e completed (same argument).
//   So calls may follow each other with no other collective in between.
// - Counter reads: the poll kernel of call e is launched on the fold's trigger, which the fold issues after its
//   griddepcontrol.wait, so its predecessors (the poll of call e - 1 among them) have completed; the poll kernel
//   itself waits for the fold grid (griddepcontrol.wait) before it completes, which keeps that chain transitive.
// - PDL: the fold waits before its first global access and releases its dependent right after (as the served
//   cooperative fold with EARLY_TRIGGER), so the poll kernel is resident while the fold runs.
//
// The draw (s2DrawKernel) stands in for the served kernel B (head_sampler.cu splitDrawKernel), one 1024-thread
// CTA per row on every rank. A neutral row (no top-k / top-p / min-p): thread vt polls state (row, vt % world,
// vt / world) instead of loading it, then kernel B's code runs verbatim (block reduce, rebase, scan, Philox draw,
// claimants). The walk runs only on the rank that holds the walker's vectors (walker % world: its local vectors
// walker / world + localThreads q, the global vectors walker + 1024 q), with kernel B's statements; that rank
// publishes its result into its entry of a per-row token slot and every rank polls the entry. With several
// claimants (two scan spans overlapping around the target), every rank that holds claimants walks them and
// publishes its latest crossing as key = position * 2048 + tid + 1 (the order of kernel B's 64-bit key
// (position << 32) | (tid + 1), tid + 1 <= 1024), and every rank takes the largest key over those ranks -- kernel B's
// rule. A row whose params enable a filter: the fold kernel also publishes that row's local vectors into the row
// area (LL, as the states), the draw polls every rank's vectors into a [rows, vocab] BF16 scratch row in vocabulary
// order and runs the shipped body on it (kernel B's filtered path on the same bits).
// - Token slot and row area ownership: entry (row, r) of rank R's copy is written only by rank r's draw CTA row
//   (slot) or rank r's fold CTAs of the row (row area), through the multicast address, once per call, with the call's
//   epoch; only rank R's draw reads its copy. Reuse follows the states' argument: every draw CTA polls data of every
//   rank (states, or the row area) published in the same call, so a peer reaches its call e + 2 only after this
//   rank's call-(e + 1) fold, which runs after this rank's draw of call e completed.
// - The draw keeps one epoch counter per row (its CTA's), read at entry and stored at its end; it is launched on the
//   fold's trigger, after the fold's wait, so the previous call's draw has completed. It waits for the fold grid
//   (griddepcontrol.wait) before it completes. With `trigger` it releases its dependent at entry (the dependent
//   waits for this grid before it reads the tokens).
//
// Rejection sampling (rejection_sampler; s2FoldPublishKernel<..., kRs = true> + s2RsDrawKernel, ops
// fold_publish_rs / rs_draw; rs_common.cuh holds the q record and the RNG). Rows 0 .. 6 test the drafts d_t against
// the q record of the request's slot (records[slotIds[0]]), row 7 is the bonus row. Per row the fold and draw compute
// out_t = accepted ? d_t : y_t with y_t = argmax_v (log rho(v) - log E_v), one Exp(1) variate E_v per element
// (rs::logE of the row's race subsequence): rho = e^x off the candidates and e^M max(w_c - q_c Z, 0) on candidates
// c != d_t (the residual), or rho = e^x everywhere on a strict row (record not usable) and on row 7.
// - Fold: pair 0 of every state is the non-RS fold's {max, sum}, bit for bit. Pair 1 carries {raceKey, raceId}:
//   warps 1 .. 7, idle during phase 2, fold each virtual thread's race x - log E over its elements, the row's
//   candidates held by this CTA excluded on rows the record makes RS rows (thread 32 + a % VTS writes it). Warp 7
//   publishes the candidate slots (row t < 7, lane k, rank r) = {claim, bits(x)}: the CTA holding id = ids[t][k] if
//   rank r owns it, else the designated CTA blockIdx.y == k, so every rank publishes all 7 x 16 slots on every RS
//   call whatever the record says (one writer per element). Warp 6 of CTA (t, 0) publishes the record digest
//   (t, r) = {hD, hQ}.
// - Draw: a neutral row polls the 1024 states (both pairs), for t < 7 the 16 x world candidate slots and the world
//   digests (published before pair 0), reduces {max, sum, key, id} in one block reduce and decides in warp 0 from
//   the candidates' x, the record's q and the off-candidate winner; no scan, walk or token slot. The candidate claims
//   must agree across ranks (else trap code 1) and so must hD (kind, valid bit, drafts, draft input; else trap code
//   2); differing hQ (q values) make the row strict on every rank.
// - Filtered row (top-k / top-p / min-p; rsFilteredRow): every rank polls the same candidate slots, digests and
//   states, then one owner rank, row % world, decides alone. The owner gathers the row from every rank's row area
//   (the fold publishes the row area for filtered rows, as for s2DrawKernel's filtered branch) into a survivor list,
//   computes the cutoff, the accept test and the residual / plain draw, and publishes {token, flags} into its entry
//   (row, owner) of the token slot with the call's epoch; the flags carry an 18-bit hash of the row's sampling
//   parameters (rowParamHash). The other ranks poll only that slot entry, take the owner's token and flags, and trap
//   with code 7 when the hash differs from their own row's (the ranks staged different parameters). Under the strict
//   filter (Knobs.filter), or with a non-finite row max / sum, every rank runs s2DrawKernel's filtered branch instead
//   (rsStrictFilteredRow: acceptance against the filtered target sample).
// - Areas: the candidate-slot and digest areas (two parities each) follow the states' epoch, parity and reuse rules:
//   written by the fold of call e, read only by the draw of call e on every rank, and every neutral or filtered draw
//   polls every rank's states of its call, so a peer reaches call e + 2 only after this rank's draw of call e. On a
//   filtered row only the owner polls the row area (the strict body polls it on every rank), and the token slot
//   entry (row, owner) is written once per call by the owner's draw CTA and read by every other rank's.
// - Consume: the draw CTA of row t < 7 clears the record's valid bit t after its final griddepcontrol.wait, i.e.
//   after every fold CTA of this rank read the record; a record therefore serves one verify.

#include "fusedSamplingKernelsCommon.cuh"
#include "rs_common.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <climits>
#include <cstdint>
#include <optional>

namespace tensorrt_llm
{
namespace kernels
{
namespace fusedSampling
{

// ---- Verbatim copies of head_sampler.cu's soft-capped BF16 row reads (SplitCap, capOf, sSplitCap, CapBf16, toFloat,
// forEachLogit): the shipped body instantiated on CapBf16 here reads the scratch row exactly as the served kernel B's
// filtered path reads the gathered row.

//! The soft cap's three FP32 constants: multiplier m, 1/cap (rounded once to FP32, as
//! ATen's division by a scalar multiplies by it) and cap.
struct SplitCap
{
    float multiplier;
    float invCap;
    float cap;
};

//! The torch chain logits.float() * m, / cap, tanh, cap * t, statement for statement.
__device__ __forceinline__ float capOf(float x, SplitCap const& c)
{
    float y = __fmul_rn(x, c.multiplier);
    y = __fmul_rn(y, c.invCap);
    y = tanhf(y);
    return __fmul_rn(c.cap, y);
}

//! The constants for reads through CapBf16 inside the shipped body: the draw stores its
//! argument here before anything reads a logit.
__shared__ SplitCap sSplitCap;

//! A BF16 LM-head logit read as the soft-capped FP32 logit.
struct CapBf16
{
    __nv_bfloat16 v;
};

__device__ inline float toFloat(CapBf16 v)
{
    return capOf(__bfloat162float(v.v), sSplitCap);
}

//! forEachLogit<float>'s traversal of a row of soft-capped logits held as BF16: vector v
//! (4 elements) on thread v % blockDim.x, in increasing v, read as 8 bytes instead of
//! 16. The FP32 rows the built-in samples are 16-byte aligned, so the shipped sweep takes
//! its vector path; this one always does.
template <typename Fn>
__device__ inline void forEachLogit(CapBf16 const* row, int vocabSize, Fn fn)
{
    int const vecCount = vocabSize / 4;
    uint2 const* vrow = reinterpret_cast<uint2 const*>(row);
    for (int v = threadIdx.x; v < vecCount; v += blockDim.x)
    {
        uint2 const packed = vrow[v];
        __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            fn(v * 4 + j, capOf(__bfloat162float(elems[j]), sSplitCap));
        }
    }
    for (int i = vecCount * 4 + threadIdx.x; i < vocabSize; i += blockDim.x)
    {
        fn(i, toFloat(row[i]));
    }
}

// ---- end of the copies.

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm

namespace
{

namespace fs = tensorrt_llm::kernels::fusedSampling;
namespace tk = tensorrt_llm::kernels;

//! Virtual threads of the served sampler's row (its 1024-thread block, head_sampler.cu kSplitBlock).
constexpr int kVirtualThreads = fs::kWideBlock;
//! Vectors one virtual thread folds at most (head_sampler.cu kSplitWalkVectors, the walk's shared-memory stage).
constexpr int kWalkVectors = 64;
//! 64-bit LL elements per state: max, sum, argmax, pad.
constexpr int kStateElems = 4;
//! Poll kernel block: one thread per virtual thread of a row.
constexpr int kPollThreads = kVirtualThreads;

//! The soft cap's FP32 constants (head_sampler.cu SplitCap).
struct Cap
{
    float multiplier;
    float invCap;
    float cap;
};

//! The torch chain logits.float() * m, / cap, tanh, cap * t, statement for statement (head_sampler.cu capOf).
__device__ __forceinline__ float capOf(float x, Cap const& c)
{
    float y = __fmul_rn(x, c.multiplier);
    y = __fmul_rn(y, c.invCap);
    y = tanhf(y);
    return __fmul_rn(c.cap, y);
}

struct FoldParams
{
    //! Row params (temperature; with the row area also top-k / top-p / min-p, which decide whether a row is filtered).
    tk::FusedSamplingParams fs;
    //! This rank's interleaved shard of the BF16 logits, [rows][localVecs] vectors of 4.
    uint2 const* local{nullptr};
    //! Multicast address of the two state areas, each [rows][world][localThreads][kStateElems] 64-bit elements.
    uint64_t mcArea{0};
    uint64_t areaBytes{0};
    //! Optional (0: off): multicast address of the two row areas, each [rows][world][localVecs] pairs of 64-bit
    //! elements; a filtered row's local vectors are published there.
    uint64_t mcRows{0};
    uint64_t rowAreaBytes{0};
    uint32_t* counters{nullptr};
    //! Optional (tests): this rank's own states, [rows][kVirtualThreads] at their global virtual thread.
    fs::OnlineSoftmax* ownStates{nullptr};
    //! Optional (bench): [slots][4] globaltimer stamps; slot `slot`: [0] min post-wait, [1] max publish end.
    unsigned long long* stamps{nullptr};
    int slot{0};
    Cap cap{};
    int rank{0};
    int world{1};
    int localVecs{0};
    int localThreads{0};
};

//! The RS fold's inputs next to FoldParams (rejection sampling; see the file header).
struct RsFoldParams
{
    //! rs.records [numSlots][1024] and the slot table (request 0's slot at [0]).
    uint8_t const* records{nullptr};
    int numSlots{0};
    int64_t const* slotIds{nullptr};
    //! The verify's drafts [7], int32 (draftInBytes 4) or int64 (8), read in place.
    void const* draftIn{nullptr};
    int draftInBytes{4};
    //! Per-row seed / offset (the race subsequence of row j is rs::raceSubseq(offset[j])).
    int64_t const* seed{nullptr};
    int64_t const* offset{nullptr};
    //! Multicast addresses of the two candidate-slot areas ([7][16][world] pairs) and the two digest areas
    //! ([7][world] pairs).
    uint64_t mcCand{0};
    uint64_t candBytes{0};
    uint64_t mcDig{0};
    uint64_t digBytes{0};
    int vocab{0};
    //! Tests only: [2][rows][vocab] FP32, [0] log E and [1] x of every element this rank folds; no exclusion.
    float* exportE{nullptr};
    int noExclude{0};
};

//! The verify's draft t (t < 7) as int32.
__device__ __forceinline__ int32_t draftAt(void const* draftIn, int bytes, int t)
{
    return bytes == 8 ? static_cast<int32_t>(static_cast<int64_t const*>(draftIn)[t])
                      : static_cast<int32_t const*>(draftIn)[t];
}

//! The record of request 0, or nullptr when its slot is out of range.
__device__ __forceinline__ rs::QRecord const* recordOf(uint8_t const* records, int numSlots, int64_t slot)
{
    return slot >= 0 && slot < numSlots ? reinterpret_cast<rs::QRecord const*>(records + slot * rs::kRecordBytes)
                                        : nullptr;
}

//! Byte offset of candidate slot (t, k, r) in one candidate area, and of digest (t, r) in one digest area.
__device__ __forceinline__ uint64_t candOffset(int t, int k, int r, int world)
{
    return (static_cast<uint64_t>(t * rs::kLanes + k) * world + r) * 16u;
}

__device__ __forceinline__ uint64_t digOffset(int t, int r, int world)
{
    return static_cast<uint64_t>(t * world + r) * 16u;
}

//! The record digest of row t (rs_common.cuh): hD over the protocol words and the drafts, hQ over q of row t.
__device__ __forceinline__ void recordDigest(
    rs::QRecord const* rec, void const* draftIn, int draftInBytes, int t, uint32_t& hD, uint32_t& hQ)
{
    uint8_t const kind = rec != nullptr ? rec->kind : static_cast<uint8_t>(rs::Kind::kNone);
    uint8_t const mode = rec != nullptr ? rec->mode : 0;
    uint32_t const valid = rec != nullptr ? rec->valid : 0u;
    uint32_t h = rs::hashStep(rs::kHashInit, rs::digestHead(kind, mode, valid >> t, t));
#pragma unroll
    for (int i = 0; i < rs::kPos; ++i)
    {
        h = rs::hashStep(h, static_cast<uint32_t>(rec != nullptr ? rec->draft[i] : -1));
    }
#pragma unroll
    for (int i = 0; i < rs::kPos; ++i)
    {
        h = rs::hashStep(h, static_cast<uint32_t>(draftAt(draftIn, draftInBytes, i)));
    }
    hD = h;
    h = rs::kHashInit;
#pragma unroll
    for (int k = 0; k < rs::kLanes; ++k)
    {
        h = rs::hashStep(h, rec != nullptr ? __float_as_uint(rec->qf[t][k]) : 0u);
    }
    hQ = rs::hashStep(h, rec != nullptr ? rec->posFlags[t] : 0u);
}

__device__ __forceinline__ bool filteredRow(fs::RowParams const& rp)
{
    return rp.needTopK || rp.needTopP || rp.needMinP;
}

//! The 18-bit hash of a row's sampling parameters (temperature reciprocal, top-k, top-p, clamped min-p) that a
//! filtered row's owner publishes with its token.
__device__ __forceinline__ uint32_t rowParamHash(fs::RowParams const& rp)
{
    uint32_t h = rs::hashStep(rs::kHashInit, __float_as_uint(rp.tempInv));
    h = rs::hashStep(h, static_cast<uint32_t>(rp.topK));
    h = rs::hashStep(h, __float_as_uint(rp.topP));
    h = rs::hashStep(h, __float_as_uint(rp.minP));
    return h & 0x3FFFFu;
}

struct PollParams
{
    uint64_t ucArea{0};
    uint64_t areaBytes{0};
    uint32_t* counters{nullptr};
    //! [rows][kVirtualThreads] states at their virtual thread (the layout the served kernel B reads).
    fs::OnlineSoftmax* states{nullptr};
    //! Optional (bench): slot `slot`: [2] max time a CTA saw all its states, [3] max poll-kernel entry.
    unsigned long long* stamps{nullptr};
    int slot{0};
    int world{1};
    int localThreads{0};
    unsigned long long timeoutNs{0};
};

__device__ __forceinline__ void gridDependencyWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void gridDependencyTrigger()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ unsigned long long globalTimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void publishPair(uint64_t address, unsigned long long a, unsigned long long b)
{
    asm volatile("st.global.v2.b64 [%0], {%1, %2};" ::"l"(address), "l"(a), "l"(b) : "memory");
}

__device__ __forceinline__ void pollPair(uint64_t address, unsigned long long& a, unsigned long long& b)
{
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(address) : "memory");
}

__device__ __forceinline__ bool fresh(unsigned long long a, unsigned long long b, uint32_t epoch)
{
    return static_cast<uint32_t>(a >> 32) == epoch && static_cast<uint32_t>(b >> 32) == epoch;
}

//! Byte offset of state (row, r, a) in one area.
__device__ __forceinline__ uint64_t stateOffset(int row, int r, int a, int world, int localThreads)
{
    return (static_cast<uint64_t>(row * world + r) * localThreads + a) * (kStateElems * 8u);
}

//! The fold of this rank's virtual threads and the publish of their states.
//!
//! Grid (rows, localThreads / VTS); block THREADS >= VTS. The CTA owns the local virtual threads a0 .. a0 + VTS - 1
//! (a0 = blockIdx.y * VTS; global vt = world * a + rank), i.e. the local vectors m = a0 + t + localThreads * j. Phase
//! 1 (all threads, coalesced over t) stores each element's __fmul_rn(capOf(logit), tempInv) at [j][t] in shared
//! memory; phase 2 (thread t < VTS) folds its slots j = 0, 1, ... with splitFoldCoopKernel's statements. FORM 0 runs
//! the coop kernel's two passes as written; FORM 1 runs them (running max / branch weight, then the sum chain) in one
//! loop: the same operations on the same values, each in the same order, only interleaved; FORM 2 is FORM 1 with phase
//! 1's loads all issued before its first cap and phase 2's next slot read from shared memory one slot ahead (again
//! the same values in the same order).
//!
//! kRs (rejection sampling, see the file header): phase 1 / phase 2 and pair 0 as above; warps 1 .. 7 fold the race
//! into pair 1 while phase 2 runs (FORM >= 1 never writes sCapped in phase 2), warp 7 publishes the candidate slots
//! and warp 6 of CTA (t, 0) the record digest.
template <int VTS, int THREADS, int FORM, bool kRs = false>
__global__ void __launch_bounds__(THREADS) s2FoldPublishKernel(FoldParams const p, RsFoldParams const q)
{
    static_assert(THREADS >= VTS && THREADS % VTS == 0, "CTA shape");
    static_assert(FORM >= 0 && FORM <= 2, "fold form");
    static_assert(!kRs || (FORM >= 1 && THREADS == 256 && VTS == 4), "RS fold: form 1 / 2 at 4 x 256");
    //! FORM 2: slots one thread loads at most (a virtual thread folds at most kWalkVectors vectors).
    constexpr int kPer = (kWalkVectors * VTS + THREADS - 1) / THREADS;
    //! RS: the race threads (warps 1 .. 7), the candidate-slot warp and the digest warp.
    constexpr int kRaceThreads = THREADS - 32;
    constexpr int kCandWarp = THREADS / 32 - 1;
    constexpr int kDigestWarp = THREADS / 32 - 2;
    extern __shared__ float4 sCapped[];
    __shared__ uint32_t sEpoch;
    //! RS: sCapped float index of each held candidate to exclude (-1: none), the race winners per (warp, vt).
    __shared__ int sIdx[kRs ? rs::kLanes : 1];
    __shared__ float sRaceKey[kRs ? THREADS / 32 - 1 : 1][VTS];
    __shared__ int sRaceId[kRs ? THREADS / 32 - 1 : 1][VTS];
    gridDependencyWait();
    gridDependencyTrigger();
    if (p.stamps != nullptr && threadIdx.x == 0)
    {
        atomicMin(p.stamps + p.slot * 4 + 0, globalTimer());
    }
    int const row = blockIdx.x;
    int const cta = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0)
    {
        sEpoch = p.counters[cta] + 1u;
    }
    int const a0 = blockIdx.y * VTS;
    int const localVecs = p.localVecs;
    int const lt = p.localThreads;
    fs::RowParams const rp = fs::loadRowParams(p.fs, row);
    float const tempInv = rp.tempInv;
    uint2 const* localRow = p.local + static_cast<size_t>(row) * localVecs;
    // Slots of the CTA's first virtual thread: the most any of its threads folds.
    int const steps = localVecs > a0 ? (localVecs - a0 + lt - 1) / lt : 0;
    // RS: candidate lane k = lane of the candidate warp reads its id (and whether the row is an RS row) before phase
    // 1's loads complete.
    int candId = -1;
    bool rsRow = false;
    if constexpr (kRs)
    {
        if (threadIdx.x / 32 == kCandWarp && threadIdx.x % 32 < rs::kLanes && row < rs::kPos)
        {
            rs::QRecord const* rec = recordOf(q.records, q.numSlots, q.slotIds[0]);
            if (rec != nullptr)
            {
                candId = rec->ids[row][threadIdx.x % 32];
                rsRow = !filteredRow(rp) && rs::isUsableKind(rec->kind) && ((rec->valid >> row) & 1u) != 0u;
            }
        }
    }
    if constexpr (FORM == 2)
    {
        uint2 packed[kPer];
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const i = threadIdx.x + k * THREADS;
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (i < steps * VTS && m < localVecs)
            {
                packed[k] = __ldg(localRow + m);
            }
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const i = threadIdx.x + k * THREADS;
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (i < steps * VTS && m < localVecs)
            {
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed[k]);
                float4 x;
                x.x = __fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), tempInv);
                x.y = __fmul_rn(capOf(__bfloat162float(elems[1]), p.cap), tempInv);
                x.z = __fmul_rn(capOf(__bfloat162float(elems[2]), p.cap), tempInv);
                x.w = __fmul_rn(capOf(__bfloat162float(elems[3]), p.cap), tempInv);
                sCapped[i] = x;
            }
        }
    }
    else
    {
        for (int i = threadIdx.x; i < steps * VTS; i += THREADS)
        {
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (m < localVecs)
            {
                uint2 const packed = __ldg(localRow + m);
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
                float4 x;
                x.x = __fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), tempInv);
                x.y = __fmul_rn(capOf(__bfloat162float(elems[1]), p.cap), tempInv);
                x.z = __fmul_rn(capOf(__bfloat162float(elems[2]), p.cap), tempInv);
                x.w = __fmul_rn(capOf(__bfloat162float(elems[3]), p.cap), tempInv);
                sCapped[i] = x;
            }
        }
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    if (p.mcRows != 0 && (rp.needTopK || rp.needTopP || rp.needMinP))
    {
        // A filtered row: the draw runs the shipped body on the full row, so publish this CTA's local vectors
        // (the CTA's virtual threads' vectors; the row's CTAs together cover the shard) into every rank's row area.
        unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
        uint64_t const dst = p.mcRows + static_cast<uint64_t>(epoch & 1u) * p.rowAreaBytes
            + static_cast<uint64_t>(row * p.world + p.rank) * localVecs * 16u;
        for (int i = threadIdx.x; i < steps * VTS; i += THREADS)
        {
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (m < localVecs)
            {
                uint2 const packed = __ldg(localRow + m);
                publishPair(dst + static_cast<uint64_t>(m) * 16u, tag | packed.x, tag | packed.y);
            }
        }
    }
    if (threadIdx.x < VTS)
    {
        int const a = a0 + threadIdx.x;
        int const vt = a * p.world + p.rank;
        int const mine = localVecs > a ? (localVecs - a + lt - 1) / lt : 0;
        float runMax = -FLT_MAX;
        int runArg = 0;
        float runSum = 0.0f;
        if constexpr (FORM == 2)
        {
            float4 next = mine > 0 ? sCapped[threadIdx.x] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int j = 0; j < mine; ++j)
            {
                int const base = (vt + j * kVirtualThreads) * 4;
                float4 const x = next;
                if (j + 1 < mine)
                {
                    next = sCapped[(j + 1) * VTS + threadIdx.x];
                }
                float const xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    bool const keep = runMax >= xs[e];
                    float const w = keep ? __expf(xs[e] - runMax) : __expf(runMax - xs[e]);
                    runArg = keep ? runArg : base + e;
                    runMax = keep ? runMax : xs[e];
                    runSum = keep ? runSum + 1.0f * w : 1.0f + runSum * w;
                }
            }
        }
        else if constexpr (FORM == 1)
        {
            for (int j = 0; j < mine; ++j)
            {
                // Global vector vt + 1024 j = world (a + lt j) + rank: its first element's index.
                int const base = (vt + j * kVirtualThreads) * 4;
                float4 const x = sCapped[j * VTS + threadIdx.x];
                float const xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    bool const keep = runMax >= xs[e];
                    float const w = keep ? __expf(xs[e] - runMax) : __expf(runMax - xs[e]);
                    runArg = keep ? runArg : base + e;
                    runMax = keep ? runMax : xs[e];
                    runSum = keep ? runSum + 1.0f * w : 1.0f + runSum * w;
                }
            }
        }
        else
        {
            for (int j = 0; j < mine; ++j)
            {
                int const base = (vt + j * kVirtualThreads) * 4;
                float4 x = sCapped[j * VTS + threadIdx.x];
                float* xs = reinterpret_cast<float*>(&x);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    bool const keep = runMax >= xs[e];
                    float const w = keep ? __expf(xs[e] - runMax) : __expf(runMax - xs[e]);
                    runArg = keep ? runArg : base + e;
                    runMax = keep ? runMax : xs[e];
                    xs[e] = keep ? w : -w;
                }
                sCapped[j * VTS + threadIdx.x] = x;
            }
            for (int j = 0; j < mine; ++j)
            {
                float4 const x = sCapped[j * VTS + threadIdx.x];
                float const xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    bool const keep = (__float_as_uint(xs[e]) >> 31) == 0u;
                    float const w = fabsf(xs[e]);
                    runSum = keep ? runSum + 1.0f * w : 1.0f + runSum * w;
                }
            }
        }
        unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
        uint64_t const dst
            = p.mcArea + static_cast<uint64_t>(epoch & 1u) * p.areaBytes + stateOffset(row, p.rank, a, p.world, lt);
        publishPair(dst, tag | __float_as_uint(runMax), tag | __float_as_uint(runSum));
        if constexpr (!kRs)
        {
            publishPair(dst + 16u, tag | static_cast<uint32_t>(runArg), tag);
        }
        if (p.ownStates != nullptr)
        {
            p.ownStates[static_cast<size_t>(row) * kVirtualThreads + vt] = fs::OnlineSoftmax{runMax, runSum, runArg};
        }
    }
    if constexpr (kRs)
    {
        if (threadIdx.x >= 32)
        {
            unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
            uint64_t const par = epoch & 1u;
            float const* xs = reinterpret_cast<float const*>(sCapped);
            int const lane = threadIdx.x % 32;
            int const warp = threadIdx.x / 32;
            if (warp == kCandWarp && lane < rs::kLanes)
            {
                // Candidate slot (row, k, rank) and the exclusion list: the holder CTA publishes {owner | id, x}, a
                // rank that does not own the id publishes {id} from the designated CTA.
                int const k = lane;
                int idx = -1;
                bool const ownsRank = candId >= 0 && candId < q.vocab && ((candId >> 2) % p.world) == p.rank;
                int const m = (candId >> 2) / p.world;
                int const a = m % lt;
                bool const here = ownsRank && a >= a0 && a < a0 + VTS;
                if (row < rs::kPos)
                {
                    uint64_t const addr = q.mcCand + par * q.candBytes + candOffset(row, k, p.rank, p.world);
                    if (here)
                    {
                        int const e = 4 * ((m / lt) * VTS + a - a0) + (candId & 3);
                        publishPair(addr, tag | rs::claimWord(candId, true), tag | __float_as_uint(xs[e]));
                        idx = rsRow && q.noExclude == 0 ? e : -1;
                    }
                    else if (!ownsRank && static_cast<int>(blockIdx.y) == k)
                    {
                        publishPair(addr, tag | rs::claimWord(candId, false), tag);
                    }
                }
                sIdx[k] = idx;
            }
            asm volatile("bar.sync 1, %0;" ::"n"(kRaceThreads) : "memory");
            // The race: thread i folds slots i, i + kRaceThreads, ... (kRaceThreads % VTS == 0: the same virtual
            // thread tv = i % VTS), each element's key x - log E (excluded / NaN: -inf), larger key then lower id.
            uint64_t const seedR = static_cast<uint64_t>(q.seed[row]);
            uint64_t const subseq = rs::raceSubseq(q.offset[row]);
            float bestKey = -__int_as_float(0x7F800000);
            int bestId = INT_MAX;
            int const tv = (threadIdx.x - 32) % VTS;
            for (int i = threadIdx.x - 32; i < steps * VTS; i += kRaceThreads)
            {
                int const jv = i / VTS;
                int const mv = a0 + tv + jv * lt;
                if (mv < localVecs)
                {
                    uint32_t excl = 0u;
#pragma unroll
                    for (int k = 0; k < rs::kLanes; ++k)
                    {
                        int const s = sIdx[k];
                        excl |= (s >> 2) == i ? 1u << (s & 3) : 0u;
                    }
                    uint64_t const v = static_cast<uint64_t>(p.world) * (a0 + tv) + p.rank
                        + static_cast<uint64_t>(kVirtualThreads) * jv;
                    uint4 const hi = rs::Ph(seedR, subseq, 2u * v);
                    uint4 const lo = rs::Ph(seedR, subseq, 2u * v + 1u);
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        float const x = xs[4 * i + e];
                        int const id = static_cast<int>(4u * v) + e;
                        float const logE = logf(rs::expo(rs::philoxWord(hi, e), rs::philoxWord(lo, e)));
                        if (q.exportE != nullptr)
                        {
                            size_t const at = static_cast<size_t>(row) * q.vocab + id;
                            q.exportE[at] = logE;
                            q.exportE[static_cast<size_t>(p.fs.numRows) * q.vocab + at] = x;
                        }
                        float const key = (((excl >> e) & 1u) != 0u || !(x == x)) ? -__int_as_float(0x7F800000)
                                                                                  : __fsub_rn(x, logE);
                        if (rs::better(key, id, bestKey, bestId))
                        {
                            bestKey = key;
                            bestId = id;
                        }
                    }
                }
            }
#pragma unroll
            for (int off = VTS; off < 32; off <<= 1)
            {
                float const k2 = __shfl_xor_sync(0xFFFFFFFFu, bestKey, off);
                int const i2 = __shfl_xor_sync(0xFFFFFFFFu, bestId, off);
                if (rs::better(k2, i2, bestKey, bestId))
                {
                    bestKey = k2;
                    bestId = i2;
                }
            }
            if (lane < VTS)
            {
                sRaceKey[warp - 1][lane] = bestKey;
                sRaceId[warp - 1][lane] = bestId;
            }
            asm volatile("bar.sync 1, %0;" ::"n"(kRaceThreads) : "memory");
            if (threadIdx.x < 32 + VTS)
            {
                int const t = threadIdx.x - 32;
                float key = sRaceKey[0][t];
                int id = sRaceId[0][t];
#pragma unroll
                for (int w = 1; w < THREADS / 32 - 1; ++w)
                {
                    if (rs::better(sRaceKey[w][t], sRaceId[w][t], key, id))
                    {
                        key = sRaceKey[w][t];
                        id = sRaceId[w][t];
                    }
                }
                uint64_t const dst = p.mcArea + par * p.areaBytes + stateOffset(row, p.rank, a0 + t, p.world, lt);
                publishPair(dst + 16u, tag | __float_as_uint(key), tag | static_cast<uint32_t>(id));
            }
            if (warp == kDigestWarp && lane == 0 && blockIdx.y == 0 && row < rs::kPos)
            {
                rs::QRecord const* rec = recordOf(q.records, q.numSlots, q.slotIds[0]);
                uint32_t hD, hQ;
                recordDigest(rec, q.draftIn, q.draftInBytes, row, hD, hQ);
                publishPair(q.mcDig + par * q.digBytes + digOffset(row, p.rank, p.world), tag | hD, tag | hQ);
            }
        }
    }
    if (p.stamps != nullptr)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicMax(p.stamps + p.slot * 4 + 1, globalTimer());
        }
    }
    if (threadIdx.x == 0)
    {
        p.counters[cta] = epoch;
    }
}

//! Every rank's states of one row, polled from this rank's copy: thread vt (one CTA per row) waits until the four
//! elements of state (row, vt % world, vt / world) carry the call's epoch, then stores the state at [row][vt].
__global__ void __launch_bounds__(kPollThreads, 1) s2PollStatesKernel(PollParams const p)
{
    __shared__ uint32_t sEpoch;
    int const row = blockIdx.x;
    int const vt = threadIdx.x;
    if (p.stamps != nullptr && threadIdx.x == 0)
    {
        atomicMax(p.stamps + p.slot * 4 + 3, globalTimer());
    }
    if (threadIdx.x == 0)
    {
        sEpoch = p.counters[row] + 1u;
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    uint64_t const src = p.ucArea + static_cast<uint64_t>(epoch & 1u) * p.areaBytes
        + stateOffset(row, vt % p.world, vt / p.world, p.world, p.localThreads);
    unsigned long long e0, e1, e2, e3;
    pollPair(src, e0, e1);
    pollPair(src + 16u, e2, e3);
    if (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)))
    {
        unsigned long long const start = globalTimer();
        do
        {
            if (globalTimer() - start > p.timeoutNs)
            {
                __trap();
            }
            pollPair(src, e0, e1);
            pollPair(src + 16u, e2, e3);
        } while (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)));
    }
    if (p.stamps != nullptr)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicMax(p.stamps + p.slot * 4 + 2, globalTimer());
        }
    }
    // Completion implies the fold grid's (see the header: counter reads of the next call).
    gridDependencyWait();
    gridDependencyTrigger();
    p.states[static_cast<size_t>(row) * kVirtualThreads + vt]
        = fs::OnlineSoftmax{__uint_as_float(static_cast<uint32_t>(e0)), __uint_as_float(static_cast<uint32_t>(e1)),
            static_cast<int>(static_cast<uint32_t>(e2))};
    if (threadIdx.x == 0)
    {
        p.counters[row] = epoch;
    }
}

struct DrawParams
{
    //! Row params, seed / offset and outputTokens of the served kernel B; `logits` is the [rows][vocab] BF16 scratch
    //! row the filtered path fills (and the shipped body then reads).
    tk::FusedSamplingParams fs;
    //! This rank's interleaved shard of the BF16 logits, [rows][localVecs] vectors of 4.
    uint2 const* local{nullptr};
    //! This rank's copy of the two state areas.
    uint64_t ucStates{0};
    uint64_t areaBytes{0};
    //! The two token-slot areas, each [rows][world] entries of two 64-bit elements: this rank's copy, multicast.
    uint64_t ucSlot{0};
    uint64_t mcSlot{0};
    uint64_t slotBytes{0};
    //! This rank's copy of the two row areas (see FoldParams::mcRows).
    uint64_t ucRows{0};
    uint64_t rowAreaBytes{0};
    //! One epoch counter per row (the draw CTA's).
    uint32_t* counters{nullptr};
    //! Optional (bench): slot `slot`: [2] max time a CTA saw all its states, [3] max time a CTA stored its token.
    unsigned long long* stamps{nullptr};
    int slot{0};
    //! Tests only: a neutral row with one claimant takes the several-claimants path (the same token).
    int forceMulti{0};
    Cap cap{};
    int rank{0};
    int world{1};
    int localVecs{0};
    int localThreads{0};
    unsigned long long timeoutNs{0};
};

//! Spin until the 64-bit pair at `address` carries `epoch` in both elements; a wait past the timeout traps.
__device__ __forceinline__ void awaitPair(
    uint64_t address, uint32_t epoch, unsigned long long timeoutNs, unsigned long long& a, unsigned long long& b)
{
    pollPair(address, a, b);
    if (!fresh(a, b, epoch))
    {
        unsigned long long const start = globalTimer();
        do
        {
            if (globalTimer() - start > timeoutNs)
            {
                __trap();
            }
            pollPair(address, a, b);
        } while (!fresh(a, b, epoch));
    }
}

//! The draw of one row on every rank (see the file header). Grid rows; block kVirtualThreads.
template <bool kTrigger>
__global__ void __launch_bounds__(kVirtualThreads, 1) s2DrawKernel(DrawParams const p)
{
    using BlockReduceOnline = cub::BlockReduce<fs::OnlineSoftmax, kVirtualThreads>;
    using BlockScanF = cub::BlockScan<float, kVirtualThreads>;
    __shared__ fs::FusedSamplingShared<kVirtualThreads> shared;
    __shared__ __align__(16) float wbuf[4 * kWalkVectors];
    __shared__ int sWalker;
    __shared__ float sBase;
    __shared__ unsigned long long sLastCrossing;
    __shared__ int sCrossIdx;
    __shared__ uint32_t sOwners;
    __shared__ uint32_t sEpoch;
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    if constexpr (kTrigger)
    {
        gridDependencyTrigger();
    }
    if (tid == 0)
    {
        fs::sSplitCap = fs::SplitCap{p.cap.multiplier, p.cap.invCap, p.cap.cap};
        sEpoch = p.counters[row] + 1u;
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    uint64_t const parity = epoch & 1u;
    int const world = p.world;
    int const vocabSize = p.fs.vocabSize;
    int const vecCount = vocabSize / 4;
    fs::RowParams const rp = fs::loadRowParams(p.fs, row);
    if (rp.needTopK || rp.needTopP || rp.needMinP)
    {
        // Every rank's local vectors of this row, from the row area, into the scratch row in vocabulary order:
        // global vector v is rank (v % world)'s local vector v / world.
        uint64_t const src
            = p.ucRows + parity * p.rowAreaBytes + static_cast<uint64_t>(row) * world * p.localVecs * 16u;
        uint2* const dst
            = reinterpret_cast<uint2*>(const_cast<void*>(p.fs.logits)) + static_cast<size_t>(row) * vecCount;
        for (int v = tid; v < vecCount; v += kVirtualThreads)
        {
            unsigned long long a, b;
            awaitPair(
                src + (static_cast<uint64_t>(v % world) * p.localVecs + v / world) * 16u, epoch, p.timeoutNs, a, b);
            dst[v] = make_uint2(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
        }
        __syncthreads();
        fs::fusedSamplingBody<fs::CapBf16, kVirtualThreads, true, false>(p.fs, shared);
    }
    else
    {
        // Kernel B's state load, as a poll of this rank's copy of the state areas.
        unsigned long long e0, e1, e2, e3;
        uint64_t const src
            = p.ucStates + parity * p.areaBytes + stateOffset(row, tid % world, tid / world, world, p.localThreads);
        pollPair(src, e0, e1);
        pollPair(src + 16u, e2, e3);
        if (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)))
        {
            unsigned long long const start = globalTimer();
            do
            {
                if (globalTimer() - start > p.timeoutNs)
                {
                    __trap();
                }
                pollPair(src, e0, e1);
                pollPair(src + 16u, e2, e3);
            } while (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)));
        }
        fs::OnlineSoftmax const cachedThreadStats{__uint_as_float(static_cast<uint32_t>(e0)),
            __uint_as_float(static_cast<uint32_t>(e1)), static_cast<int>(static_cast<uint32_t>(e2))};
        if (p.stamps != nullptr)
        {
            __syncthreads();
            if (tid == 0)
            {
                atomicMax(p.stamps + p.slot * 4 + 2, globalTimer());
            }
        }
        // ---- From here to the walk: kernel B's statements (head_sampler.cu splitDrawKernel), in its order.
        fs::OnlineSoftmax const rowStats
            = BlockReduceOnline(shared.temp.reduceOnline).Reduce(cachedThreadStats, fs::OnlineSoftmaxOp());
        if (tid == 0)
        {
            shared.maxScaled = rowStats.max;
            shared.argmax = rowStats.argmax;
            shared.totalMass = rowStats.sum;
            sWalker = -1;
        }
        __syncthreads();
        float const maxScaled = shared.maxScaled;
        float const cachedLocalMass = cachedThreadStats.sum * __expf(cachedThreadStats.max - maxScaled);
        if (tid == 0)
        {
            fs::initRowRng(p.fs, row, &shared.rejectRng);
            shared.pivot = -1.0f;
            shared.token = -1;
        }
        __syncthreads();
        float const localMass = cachedLocalMass;
        float base = 0.0f;
        float supportMass = 0.0f;
        BlockScanF(shared.temp.scanF).ExclusiveSum(localMass, base, supportMass);
        __syncthreads();
        if (tid == 0)
        {
            shared.massTarget = rp.topP * supportMass;
            shared.target = curand_uniform(&shared.rejectRng) * supportMass;
            shared.candIdx = -1;
            sLastCrossing = 0ull;
            sCrossIdx = -1;
            sOwners = 0u;
        }
        __syncthreads();
        float const target = shared.target;
        bool const claims = target >= base && target < base + localMass;
        if (claims)
        {
            sWalker = tid;
            sBase = base;
            atomicOr(&sOwners, 1u << (tid % world));
        }
        int const claimants = __syncthreads_count(claims);
        // ---- The walk: on the rank holding the walker's vectors; the result through the token slot.
        uint2 const* const localRow = p.local + static_cast<size_t>(row) * p.localVecs;
        uint64_t const slotRow = parity * p.slotBytes + static_cast<uint64_t>(row) * world * 16u;
        unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
        int token = -1;
        if (claimants == 1 && p.forceMulti == 0)
        {
            int const walker = sWalker;
            int const owner = walker % world;
            if (owner == p.rank)
            {
                // Kernel B's single-walker walk on the local copy of the walker's vectors.
                int const walkVectors = (vecCount - walker + kVirtualThreads - 1) / kVirtualThreads;
                if (tid < walkVectors)
                {
                    uint2 const packed = localRow[walker / world + tid * p.localThreads];
                    __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        wbuf[tid * 4 + j]
                            = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), p.cap), rp.tempInv) - maxScaled);
                    }
                }
                __syncthreads();
                if (tid == 0)
                {
                    float running = sBase;
                    int cand = -1;
                    float4 const* w4 = reinterpret_cast<float4 const*>(wbuf);
                    for (int q = 0; q < walkVectors && cand < 0; ++q)
                    {
                        float4 const quad = w4[q];
                        float const ws[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                        {
                            float const w = ws[j];
                            // supportFloor 0 and pivot -1 in the shipped walk's membership test.
                            if (cand < 0 && w >= 0.0f && w > -1.0f)
                            {
                                running += w;
                                if (running > target)
                                {
                                    cand = (walker + q * kVirtualThreads) * 4 + j;
                                }
                            }
                        }
                    }
                    publishPair(p.mcSlot + slotRow + static_cast<uint64_t>(p.rank) * 16u,
                        tag | static_cast<uint32_t>(cand), tag);
                    token = cand;
                }
            }
            else if (tid == 0)
            {
                unsigned long long a, b;
                awaitPair(p.ucSlot + slotRow + static_cast<uint64_t>(owner) * 16u, epoch, p.timeoutNs, a, b);
                token = static_cast<int>(static_cast<uint32_t>(a));
            }
        }
        else if (claimants >= 1)
        {
            // Kernel B's several-claimants walk, each claimant on the rank holding its vectors.
            int crossIdx = -1;
            unsigned long long crossing = 0ull;
            if (claims && tid % world == p.rank)
            {
                float running = base;
                int position = 0;
                for (int q = 0; tid + q * kVirtualThreads < vecCount && crossIdx < 0; ++q)
                {
                    int const v = tid + q * kVirtualThreads;
                    uint2 const packed = localRow[tid / world + q * p.localThreads];
                    __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
                    for (int j = 0; j < 4; ++j)
                    {
                        float const w
                            = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), p.cap), rp.tempInv) - maxScaled);
                        ++position;
                        if (!(w >= 0.0f && w > -1.0f))
                        {
                            continue;
                        }
                        running += w;
                        if (running > target)
                        {
                            crossIdx = v * 4 + j;
                            crossing = (static_cast<unsigned long long>(position) << 32)
                                | static_cast<unsigned long long>(tid + 1);
                            break;
                        }
                    }
                }
                if (crossIdx >= 0)
                {
                    atomicMax(&sLastCrossing, crossing);
                }
            }
            __syncthreads();
            if (crossIdx >= 0 && crossing == sLastCrossing)
            {
                sCrossIdx = crossIdx;
            }
            __syncthreads();
            if (tid == 0)
            {
                uint32_t const owners = sOwners;
                if ((owners >> p.rank) & 1u)
                {
                    unsigned long long const best = sLastCrossing;
                    uint32_t const key
                        = best == 0ull ? 0u : static_cast<uint32_t>((best >> 32) * 2048ull + (best & 0xFFFFFFFFull));
                    int const idx = best == 0ull ? -1 : sCrossIdx;
                    publishPair(p.mcSlot + slotRow + static_cast<uint64_t>(p.rank) * 16u,
                        tag | static_cast<uint32_t>(idx), tag | key);
                }
                uint32_t bestKey = 0u;
                for (int r = 0; r < world; ++r)
                {
                    if (((owners >> r) & 1u) == 0u)
                    {
                        continue;
                    }
                    unsigned long long a, b;
                    awaitPair(p.ucSlot + slotRow + static_cast<uint64_t>(r) * 16u, epoch, p.timeoutNs, a, b);
                    int const idx = static_cast<int>(static_cast<uint32_t>(a));
                    uint32_t const key = static_cast<uint32_t>(b);
                    if (idx >= 0 && key > bestKey)
                    {
                        bestKey = key;
                        token = idx;
                    }
                }
            }
        }
        if (tid == 0)
        {
            p.fs.outputTokens[row] = token >= 0 ? token : shared.argmax;
            if (p.stamps != nullptr)
            {
                atomicMax(p.stamps + p.slot * 4 + 3, globalTimer());
            }
        }
    }
    // Completion implies the fold grid's (the next call's counter reads).
    gridDependencyWait();
    if (tid == 0)
    {
        p.counters[row] = epoch;
    }
}

//! FP32 words of one row's test export (RsDrawParams::exports): kExport* scalars (ints stored as their bits), then
//! per candidate lane w, qZ, the race key, x and log E.
constexpr int kExportM = 0;
constexpr int kExportZ = 1;
constexpr int kExportKoff = 2;
constexpr int kExportIoff = 3;
constexpr int kExportKd = 4;
constexpr int kExportAcc = 5;
constexpr int kExportY = 6;
constexpr int kExportOut = 7;
constexpr int kExportUsable = 8;
constexpr int kExportReason = 9;
constexpr int kExportUa = 10;
constexpr int kExportKstar = 11;
constexpr int kExportW = 16;
constexpr int kExportQz = 32;
constexpr int kExportKc = 48;
constexpr int kExportX = 64;
constexpr int kExportLogE = 80;
constexpr int kExportWords = 96;

//! FP32 words of one filtered row's owner export (RsDrawParams::fexports): kFExport* scalars (ints stored as their
//! bits), then per candidate lane w [w >= tau], fl(qf K) and the residual mass.
constexpr int kFExportPhi = 0;
constexpr int kFExportRawS = 1;
constexpr int kFExportTau = 2;
constexpr int kFExportKept = 3;
constexpr int kFExportS1 = 4;
constexpr int kFExportN = 5;
constexpr int kFExportNmax = 6;
constexpr int kFExportROff = 7;
constexpr int kFExportRC = 8;
constexpr int kFExportTarget = 9;
constexpr int kFExportUr = 10;
constexpr int kFExportClaim = 11;
constexpr int kFExportOut = 12;
constexpr int kFExportFlags = 13;
constexpr int kFExportUa = 14;
constexpr int kFExportKd = 15;
constexpr int kFExportY = 16;
constexpr int kFExportM = 17;
constexpr int kFExportZ = 18;
constexpr int kFExportCase = 19;
constexpr int kFExportKeptCount = 20;
constexpr int kFExportWk = 32;
constexpr int kFExportQk = 48;
constexpr int kFExportRho = 64;
constexpr int kFExportWords = 80;

//! Bits of a filtered row's flags word, which the row's owner publishes with the token: bit 0 accepted, 1 usable
//! record, 2 .. 5 RsReason, 6 the draft outside the filtered target support, 7 the screen fell back to the whole row,
//! 8 a rounding fallback of the residual draw, 9 a plain row (drawn from the filtered target alone), 10 .. 27 the
//! 18-bit hash of the row's sampling parameters.
constexpr uint32_t kFlagAcc = 1u << 0;
constexpr uint32_t kFlagUsable = 1u << 1;
constexpr int kFlagReasonShift = 2;
constexpr uint32_t kFlagDUnkept = 1u << 6;
constexpr uint32_t kFlagScreenFallback = 1u << 7;
constexpr uint32_t kFlagResidFallback = 1u << 8;
constexpr uint32_t kFlagPlain = 1u << 9;
constexpr int kFlagHashShift = 10;
constexpr uint32_t kFlagHashMask = 0x3FFFFu;
//! Knobs.check trap detail of a residual rounding fallback: kCheckResidDetail + row.
constexpr long long kCheckResidDetail = 1000;

//! Why a neutral row t < 7 is not decided by the chain test (0: usable), in the order the draw tests them; the RsDiag
//! counter each reason increments.
enum class RsReason : int
{
    kUsable = 0,
    kKind = 1,
    kPoison = 2,
    kDrafts = 3,
    kQdiv = 4,
    kQinvalid = 5,
    kKd = 6,
    kDup = 7,
    kQzero = 8,
};

__device__ __forceinline__ rs::Diag reasonCounter(RsReason r)
{
    switch (r)
    {
    case RsReason::kKind: return rs::Diag::kStrictKind;
    case RsReason::kPoison: return rs::Diag::kStrictPoison;
    case RsReason::kDrafts: return rs::Diag::kStrictDrafts;
    case RsReason::kQdiv: return rs::Diag::kStrictQdiv;
    case RsReason::kQinvalid: return rs::Diag::kStrictQinvalid;
    case RsReason::kKd: return rs::Diag::kStrictKd;
    case RsReason::kDup: return rs::Diag::kStrictDup;
    default: return rs::Diag::kStrictQzero;
    }
}

struct RsDrawParams
{
    //! rs.records (the draw clears valid bit t of its row t < 7 at its end) and the slot table.
    uint8_t* records{nullptr};
    int numSlots{0};
    int dummySlot{-1};
    int64_t const* slotIds{nullptr};
    //! The verify's drafts [7], int32 (4) or int64 (8).
    void const* draftIn{nullptr};
    int draftInBytes{4};
    //! This rank's copy of the two candidate-slot areas and the two digest areas.
    uint64_t ucCand{0};
    uint64_t candBytes{0};
    uint64_t ucDig{0};
    uint64_t digBytes{0};
    int vocab{0};
    int check{0};
    //! rs.diag (int64 [rs::kDiagWords]).
    unsigned long long* diag{nullptr};
    //! Optional calibration ring [ringSize][4] {w_d / qZ_d, u_a, accepted, row} and its head.
    float* ring{nullptr};
    int* ringHead{nullptr};
    int ringSize{0};
    //! Tests only: forced accept uniforms [rows], forced candidate log E [rows][16], exports [rows][kExportWords].
    float const* forceUa{nullptr};
    float const* forceLogE{nullptr};
    float* exports{nullptr};
    //! Filtered rows (rsFilteredRow): the owner's survivor list, [ceil(rows / world)][listCap] values x - M (FP32)
    //! and ids (int32), list slot row / world; rs::FilterMode.
    float* listX{nullptr};
    int32_t* listId{nullptr};
    int listCap{0};
    int filterMode{0};
    //! Tests only: 1 drops the raw-logit screen, 2 the whole screen (floor 0); owner exports [rows][kFExportWords].
    int forceScreen{0};
    float* fexports{nullptr};
    //! Optional (bench): [slots][4] globaltimer stamps of filtered rows, slot DrawParams::slot: [0] owner gather done,
    //! [1] owner cutoff and kept mass done, [2] owner decision done, [3] token known (every rank).
    unsigned long long* fstamps{nullptr};
};

//! (max, sum) of the online softmax and the race winner (key, id) of the polled states.
struct RsStat
{
    float max;
    float sum;
    float key;
    int id;
};

struct RsStatOp
{
    __device__ inline RsStat operator()(RsStat const& a, RsStat const& b) const
    {
        fs::OnlineSoftmax const m
            = fs::combineOnlineSoftmax(fs::OnlineSoftmax{a.max, a.sum, 0}, fs::OnlineSoftmax{b.max, b.sum, 0});
        bool const takeB = rs::better(b.key, b.id, a.key, a.id);
        return RsStat{m.max, m.sum, takeB ? b.key : a.key, takeB ? b.id : a.id};
    }
};

using BlockReduceRs = cub::BlockReduce<RsStat, kVirtualThreads>;

struct FloatMaxOp
{
    __device__ inline float operator()(float a, float b) const
    {
        return fmaxf(a, b);
    }
};

//! The RS draw's shared scalars (one CTA per row).
struct RsDrawScalars
{
    uint32_t epoch;
    int id[rs::kLanes];
    float qf[rs::kLanes];
    float logE[rs::kLanes];
    float x[rs::kLanes];
    float ua;
    int d;
    int rs;
    int same;
    int flags;
    int validBit;
    int dummy;
    int abort;
    int dBad;
    int qBad;
    float M;
    float Z;
    float koff;
    int ioff;
    // Filtered rows (rsFilteredRow): the owner's floor, raw screen, cutoff, kept mass, top-p base mass, list sizes,
    // residual masses and target, claimant; the row's token and flags word (every rank); a parameter-hash mismatch.
    float phi;
    float rawS;
    float tau;
    float kept;
    float s1;
    int nList;
    int nMax;
    float rOff;
    float rC;
    float target;
    int claim;
    int out;
    uint32_t oflags;
    int owner;
    int hashBad;
    float rho[rs::kLanes];
    float listMass;
    float claimBase;
    int claimN;
    int lastTid;
    unsigned long long best;
    unsigned long long ft[4];
};

// The RS draw's scratch at namespace scope (as sSplitCap): referenced only by s2RsDrawKernel and its filtered-row
// bodies, which therefore address it as shared memory even when not inlined; the non-RS kernels do not change.
__shared__ fs::FusedSamplingShared<kVirtualThreads> sRsShared;
__shared__ RsDrawScalars sRsS;
__shared__ typename BlockReduceRs::TempStorage sRsReduce;

//! The record of request 0's slot and the row's randomness (no fabric dependency): candidate ids, q, log E, the
//! draft, the record's flags into sRsS; thread 32 the accept uniform.
__device__ __forceinline__ void rsPrologue(DrawParams const& p, RsDrawParams const& q, int row)
{
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int64_t const slot = q.slotIds[0];
    rs::QRecord const* rec = slot >= 0 && slot < q.numSlots
        ? reinterpret_cast<rs::QRecord const*>(q.records + slot * rs::kRecordBytes)
        : nullptr;
    uint64_t const seedR = static_cast<uint64_t>(p.fs.seed[row]);
    int64_t const offR = static_cast<int64_t>(p.fs.offset[row]);
    if (tid < 32)
    {
        uint8_t const kind = rec != nullptr ? rec->kind : static_cast<uint8_t>(rs::Kind::kNone);
        uint32_t const valid = rec != nullptr ? rec->valid : 0u;
        bool const rsRow = row < rs::kPos && rs::isUsableKind(kind) && ((valid >> row) & 1u) != 0u;
        bool const differs = rsRow && lane < rs::kPos && rec->draft[lane] != draftAt(q.draftIn, q.draftInBytes, lane);
        bool const same = rsRow && __ballot_sync(0xFFFFFFFFu, differs) == 0u;
        if (lane < rs::kLanes)
        {
            int const id = row < rs::kPos && rec != nullptr ? rec->ids[row][lane] : -1;
            sRsS.id[lane] = id;
            sRsS.qf[lane] = rsRow ? rec->qf[row][lane] : 0.0f;
            float logE = 0.0f;
            if (rsRow && id >= 0 && id < q.vocab)
            {
                logE = q.forceLogE != nullptr ? q.forceLogE[row * rs::kLanes + lane]
                                              : rs::logE(seedR, rs::raceSubseq(offR), id);
            }
            sRsS.logE[lane] = logE;
        }
        if (lane == 0)
        {
            sRsS.rs = rsRow;
            sRsS.same = same;
            sRsS.d = row < rs::kPos ? draftAt(q.draftIn, q.draftInBytes, row) : -1;
            sRsS.flags = rsRow ? rec->posFlags[row] : 0;
            sRsS.validBit = row < rs::kPos && ((valid >> row) & 1u) != 0u;
            sRsS.dummy = slot == q.dummySlot;
            sRsS.abort = 0;
            sRsS.dBad = 0;
            sRsS.qBad = 0;
        }
    }
    else if (tid == 32 && row < rs::kPos)
    {
        sRsS.ua = q.forceUa != nullptr ? q.forceUa[row] : rs::unif(rs::Ph(seedR, rs::acceptSubseq(offR), 0).x);
    }
    __syncthreads();
}

//! This call's candidate slots (claims and the holder's x) and record digests of row t < 7, published before pair 0
//! of the states: sRsS.x, sRsS.abort, sRsS.dBad, sRsS.qBad.
__device__ __forceinline__ void rsPollCandDig(
    DrawParams const& p, RsDrawParams const& q, int row, uint32_t epoch, uint64_t parity)
{
    float const kNegInf = -__int_as_float(0x7F800000);
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const world = p.world;
    if (row < rs::kPos && tid >= 32 && tid < 32 + 4 * 32)
    {
        // Thread 32 + world k + r: candidate slot (row, k, r); the world lanes of lane k form one group.
        int const c = tid - 32;
        int const k = c / world;
        int const r = c % world;
        bool const act = c < rs::kLanes * world;
        unsigned long long cw = 0ull, xb = 0ull;
        if (act)
        {
            awaitPair(q.ucCand + parity * q.candBytes + candOffset(row, k, r, world), epoch, p.timeoutNs, cw, xb);
        }
        uint32_t const idw = static_cast<uint32_t>(cw) & ~rs::kClaimOwner;
        bool const own = act && (static_cast<uint32_t>(cw) & rs::kClaimOwner) != 0u;
        int const leadLane = lane & ~(world - 1);
        uint32_t const gm = (world >= 32 ? 0xFFFFFFFFu : ((1u << world) - 1u)) << leadLane;
        uint32_t const lead = __shfl_sync(0xFFFFFFFFu, idw, leadLane);
        bool const allEq = (__ballot_sync(0xFFFFFFFFu, idw == lead) & gm) == gm;
        uint32_t const ob = __ballot_sync(0xFFFFFFFFu, own) & gm;
        int const ownCount = __popc(ob);
        uint32_t const xo = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(xb), ownCount != 0 ? __ffs(ob) - 1 : lane);
        if (act && r == 0)
        {
            int const id = sRsS.id[k];
            uint32_t const expect = id < 0 ? rs::kClaimEmpty : static_cast<uint32_t>(id);
            int const owners = id >= 0 && id < q.vocab ? 1 : 0;
            sRsS.x[k] = ownCount == 1 ? __uint_as_float(xo) : kNegInf;
            if (!(allEq && lead == expect && ownCount == owners))
            {
                sRsS.abort = 1;
            }
        }
    }
    else if (row < rs::kPos && tid >= 32 + 4 * 32 && tid < 32 + 5 * 32)
    {
        // Thread 160 + r: digest (row, r).
        int const r = tid - (32 + 4 * 32);
        bool const act = r < world;
        unsigned long long hd = 0ull, hq = 0ull;
        if (act)
        {
            awaitPair(q.ucDig + parity * q.digBytes + digOffset(row, r, world), epoch, p.timeoutNs, hd, hq);
        }
        uint32_t const mask = world >= 32 ? 0xFFFFFFFFu : (1u << world) - 1u;
        uint32_t const d0 = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(hd), 0);
        uint32_t const q0 = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(hq), 0);
        uint32_t const dSame = __ballot_sync(0xFFFFFFFFu, act && static_cast<uint32_t>(hd) == d0);
        uint32_t const qSame = __ballot_sync(0xFFFFFFFFu, act && static_cast<uint32_t>(hq) == q0);
        if (lane == 0)
        {
            sRsS.dBad = dSame != mask;
            sRsS.qBad = qSame != mask;
        }
    }
}

//! Every rank's states of the row (both pairs) reduced to {max, sum, race key, race id} in sRsS; then the coded
//! traps of rsPollCandDig's checks (claims: code 1, hD: code 2).
__device__ __forceinline__ void rsPollStates(
    DrawParams const& p, RsDrawParams const& q, int row, uint32_t epoch, uint64_t parity)
{
    int const tid = threadIdx.x;
    int const world = p.world;
    unsigned long long e0, e1, e2, e3;
    {
        uint64_t const src
            = p.ucStates + parity * p.areaBytes + stateOffset(row, tid % world, tid / world, world, p.localThreads);
        pollPair(src, e0, e1);
        pollPair(src + 16u, e2, e3);
        if (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)))
        {
            unsigned long long const start = globalTimer();
            do
            {
                if (globalTimer() - start > p.timeoutNs)
                {
                    __trap();
                }
                pollPair(src, e0, e1);
                pollPair(src + 16u, e2, e3);
            } while (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)));
        }
    }
    if (p.stamps != nullptr)
    {
        __syncthreads();
        if (tid == 0)
        {
            atomicMax(p.stamps + p.slot * 4 + 2, globalTimer());
        }
    }
    RsStat const st{__uint_as_float(static_cast<uint32_t>(e0)), __uint_as_float(static_cast<uint32_t>(e1)),
        __uint_as_float(static_cast<uint32_t>(e2)), static_cast<int>(static_cast<uint32_t>(e3))};
    RsStat const agg = BlockReduceRs(sRsReduce).Reduce(st, RsStatOp());
    if (tid == 0)
    {
        sRsS.M = agg.max;
        sRsS.Z = agg.sum;
        sRsS.koff = agg.key;
        sRsS.ioff = agg.id;
    }
    __syncthreads();
    if (tid == 0 && sRsS.abort != 0)
    {
        // The candidate ids differ across ranks: a writer bug, never data.
        rs::trap(q.diag, rs::TrapCode::kClaims, row);
    }
    if (tid == 0 && sRsS.dBad != 0)
    {
        // Drafts, kind or validity differ across ranks: the TP forward already ran on divergent inputs.
        rs::trap(q.diag, rs::TrapCode::kDigestD, row);
    }
}

//! A filtered row's strict body on every rank: s2DrawKernel's filtered branch (every rank's local vectors from the
//! row area into the scratch row, then the shipped body). At 8 rows the body is deterministic, so every rank draws
//! the same token from the same gathered row.
__device__ __forceinline__ void rsStrictFilteredRow(DrawParams const& p, int row, uint32_t epoch, uint64_t parity)
{
    int const world = p.world;
    int const vecCount = p.fs.vocabSize / 4;
    uint64_t const src = p.ucRows + parity * p.rowAreaBytes + static_cast<uint64_t>(row) * world * p.localVecs * 16u;
    uint2* const dst = reinterpret_cast<uint2*>(const_cast<void*>(p.fs.logits)) + static_cast<size_t>(row) * vecCount;
    for (int v = threadIdx.x; v < vecCount; v += kVirtualThreads)
    {
        unsigned long long a, b;
        awaitPair(src + (static_cast<uint64_t>(v % world) * p.localVecs + v / world) * 16u, epoch, p.timeoutNs, a, b);
        dst[v] = make_uint2(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
    }
    __syncthreads();
    fs::fusedSamplingBody<fs::CapBf16, kVirtualThreads, true, false>(p.fs, sRsShared);
}

//! The reason ladder of a row t < 7 (the order the draw tests them): why the row is not decided by the chain test.
__device__ __forceinline__ RsReason rsReason(
    bool rsRow, bool poison, bool same, bool qBad, int flags, int kd, uint32_t hit, bool dupAny, bool rangeOk, float qd)
{
    if (!rsRow)
    {
        return RsReason::kKind;
    }
    if (poison)
    {
        return RsReason::kPoison;
    }
    if (!same)
    {
        return RsReason::kDrafts;
    }
    if (qBad)
    {
        return RsReason::kQdiv;
    }
    if ((flags & rs::kPosQInvalid) != 0)
    {
        return RsReason::kQinvalid;
    }
    if (kd < 0)
    {
        return RsReason::kKd;
    }
    if (__popc(hit) != 1 || dupAny || !rangeOk)
    {
        return RsReason::kDup;
    }
    if (!(qd > 0.0f))
    {
        return RsReason::kQzero;
    }
    return RsReason::kUsable;
}

//! The floor phi of a filtered row's screen (every kept weight is >= phi) and its case (RS_SCREEN_CASES): min-p
//! keeps w >= minP; top-k without min-p keeps the k largest, which are >= phi whenever at least k weights are
//! (checked after the gather); top-p alone keeps everything above its cut, and with 1 - topP >= 2^-10 the weights
//! >= phi = (1 - topP) Z / (2V) carry more than topP Z of the mass, so the cut lies above phi. 0: no screen.
__device__ __forceinline__ float rsScreenFloor(fs::RowParams const& rp, float Z, int vocab, int& screenCase)
{
    float const rest = __fsub_rn(1.0f, rp.topP);
    bool const restScreen = rp.needTopP && rest >= 0x1p-10f;
    float const inv2V = __frcp_rn(__int2float_rn(2 * vocab));
    if (rp.needMinP)
    {
        screenCase = 2;
        return rp.minP;
    }
    if (rp.needTopK)
    {
        screenCase = 3;
        return restScreen ? __fmul_rn(rest, inv2V) : 0x1p-24f;
    }
    if (restScreen)
    {
        screenCase = 1;
        return __fmul_rn(__fmul_rn(rest, Z), inv2V);
    }
    screenCase = 0;
    return 0.0f;
}

//! The raw (BF16 logit) value below which an element's weight is below phi / 2: capOf is non-decreasing, and
//! w >= phi needs capOf(raw) >= T (M + ln phi), so raw >= (cap / m) atanh(T (M + ln(phi / 2)) / cap) keeps every such
//! element with a factor-2 margin; rounded down. -inf: no raw screen (also without a positive soft cap).
__device__ __forceinline__ float rsRawScreen(float phi, float M, float tempInv, Cap const& cap)
{
    float const kNegInf = -__int_as_float(0x7F800000);
    double const temperature = 1.0 / static_cast<double>(tempInv);
    if (!(phi > 0.0f) || !(temperature >= 1e-3) || !(cap.multiplier > 0.0f))
    {
        return kNegInf;
    }
    if (!(cap.cap > 0.0f))
    {
        // capOf has no cap-free form (it scales by 1 / cap), so there is no bound to invert.
        return kNegInf;
    }
    double const capped = temperature * (static_cast<double>(M) + log(0.5 * static_cast<double>(phi)));
    double const a = capped / static_cast<double>(cap.cap);
    if (!(a > -1.0 && a < 1.0))
    {
        return kNegInf;
    }
    return __double2float_rd(static_cast<double>(cap.cap) / static_cast<double>(cap.multiplier) * atanh(a));
}

//! The gather: thread tid's elements of the row from every rank's row area (vectors v = tid + 1024 j, increasing j,
//! four polls in flight), each x - M = fl(fl(capOf(raw) / T) - M) kept iff raw >= rawS and w = exp(x - M) >= phi,
//! appended at listX / listId[i * 1024 + tid] (thread-major, so the list is a function of the row). Returns the count.
__device__ __forceinline__ int rsGatherList(DrawParams const& p, int row, uint32_t epoch, uint64_t parity, float M,
    float tempInv, float phi, float rawS, float* listX, int32_t* listId)
{
    int const tid = threadIdx.x;
    int const world = p.world;
    int const vecCount = p.fs.vocabSize / 4;
    uint64_t const src = p.ucRows + parity * p.rowAreaBytes + static_cast<uint64_t>(row) * world * p.localVecs * 16u;
    int n = 0;
    for (int v0 = tid; v0 < vecCount; v0 += 4 * kVirtualThreads)
    {
        unsigned long long a[4], b[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const v = v0 + k * kVirtualThreads;
            if (v < vecCount)
            {
                pollPair(src + (static_cast<uint64_t>(v % world) * p.localVecs + v / world) * 16u, a[k], b[k]);
            }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            int const v = v0 + k * kVirtualThreads;
            if (v < vecCount)
            {
                if (!fresh(a[k], b[k], epoch))
                {
                    awaitPair(src + (static_cast<uint64_t>(v % world) * p.localVecs + v / world) * 16u, epoch,
                        p.timeoutNs, a[k], b[k]);
                }
                uint32_t const halves[2] = {static_cast<uint32_t>(a[k]), static_cast<uint32_t>(b[k])};
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(halves);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    float const raw = __bfloat162float(elems[e]);
                    if (!(raw < rawS))
                    {
                        float const xm = __fsub_rn(__fmul_rn(capOf(raw, p.cap), tempInv), M);
                        if (__expf(xm) >= phi)
                        {
                            listX[n * kVirtualThreads + tid] = xm;
                            listId[n * kVirtualThreads + tid] = 4 * v + e;
                            ++n;
                        }
                    }
                }
            }
        }
    }
    return n;
}

//! sum_{i < n} [w_i >= floor] w_i over thread tid's list column in i order, w_i = exp(listX[i * 1024 + tid]), then
//! the block sum (cub's fixed tree), returned to every thread: the same value on every call with the same list.
__device__ __forceinline__ float rsListMass(float const* listX, int n, float floor)
{
    float local = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        float const w = __expf(listX[i * kVirtualThreads + threadIdx.x]);
        local = w >= floor ? __fadd_rn(local, w) : local;
    }
    float const total = fs::FusedSamplingShared<kVirtualThreads>::BlockReduceF(sRsShared.temp.reduceF).Sum(local);
    if (threadIdx.x == 0)
    {
        sRsS.listMass = total;
    }
    __syncthreads();
    return sRsS.listMass;
}

//! The residual mass of a list entry: w [w >= tau], 0 on the record's candidates when `excludeCands`.
__device__ __forceinline__ float rsListRho(float xm, int id, float tau, bool excludeCands)
{
    float const w = __expf(xm);
    bool cand = false;
    if (excludeCands)
    {
#pragma unroll
        for (int k = 0; k < rs::kLanes; ++k)
        {
            cand = cand || sRsS.id[k] == id;
        }
    }
    return w >= tau && !cand ? w : 0.0f;
}

//! A filtered row by exact filtered rejection sampling. The owner rank row % world gathers the row's elements above
//! the screen floor into its list (case K re-gathers unscreened when fewer than top-k survive), finds the stock
//! cutoff tau and kept mass K with the shipped findThreshold on the list (min-p -> top-k -> top-p, top-p against the
//! mass top-p sees), and decides: accept iff fl(u_a fl(q_d K)) < w_d [w_d >= tau]; otherwise the
//! token is drawn from the residual relu(p' - q') (usable record) or p' (row 7, unusable record) by an inverse CDF
//! over the list (thread-major, block scan, a unique claimant whose column is staged in shared memory) then the
//! candidate lanes, with u_r = word y of the accept subsequence. The owner then publishes {token, flags} into its
//! entry of the row's token slot; every other rank polls it and checks the parameter hash. Every floating decision is
//! the owner's, so the token is the same on every rank. Leaves sRsS.out / oflags / owner / hashBad / nList / ft.
__device__ __forceinline__ void rsFilteredRow(
    DrawParams const& p, RsDrawParams const& q, int row, uint32_t epoch, uint64_t parity)
{
    float const kNegInf = -__int_as_float(0x7F800000);
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const world = p.world;
    int const owner = row % world;
    fs::RowParams const rp = fs::loadRowParams(p.fs, row);
    uint32_t const hash = rowParamHash(rp);
    uint64_t const slotAt = parity * p.slotBytes + static_cast<uint64_t>(row * world + owner) * 16u;
    unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
    if (owner != p.rank)
    {
        if (tid == 0)
        {
            unsigned long long a, b;
            awaitPair(p.ucSlot + slotAt, epoch, p.timeoutNs, a, b);
            uint32_t const fl = static_cast<uint32_t>(b);
            sRsS.out = static_cast<int>(static_cast<uint32_t>(a));
            sRsS.oflags = fl;
            sRsS.owner = 0;
            sRsS.nList = 0;
            sRsS.hashBad = ((fl >> kFlagHashShift) & kFlagHashMask) != hash;
            sRsS.ft[3] = globalTimer();
        }
        __syncthreads();
        return;
    }
    float const M = sRsS.M;
    float const Z = sRsS.Z;
    float* const listX = q.listX + static_cast<size_t>(row / world) * q.listCap;
    int32_t* const listId = q.listId + static_cast<size_t>(row / world) * q.listCap;
    // ---- The screen floor and the gather (then the pad and the top-k validity check).
    int screenCase = 0;
    float phi = rsScreenFloor(rp, Z, p.fs.vocabSize, screenCase);
    float rawS = rsRawScreen(phi, M, rp.tempInv, p.cap);
    if (q.forceScreen == 2)
    {
        phi = 0.0f;
    }
    if (q.forceScreen != 0)
    {
        rawS = kNegInf;
    }
    bool screenFallback = false;
    int n = rsGatherList(p, row, epoch, parity, M, rp.tempInv, phi, rawS, listX, listId);
    using BlockReduceF = fs::FusedSamplingShared<kVirtualThreads>::BlockReduceF;
    float total = BlockReduceF(sRsShared.temp.reduceF).Sum(static_cast<float>(n));
    if (tid == 0)
    {
        sRsS.nList = static_cast<int>(total);
    }
    __syncthreads();
    if (rp.needTopK && !rp.needMinP && phi > 0.0f && sRsS.nList < rp.topK)
    {
        // Fewer than top-k weights >= phi: the k-th largest may lie below phi, so take the whole row.
        screenFallback = true;
        phi = 0.0f;
        rawS = kNegInf;
        n = rsGatherList(p, row, epoch, parity, M, rp.tempInv, phi, rawS, listX, listId);
        total = BlockReduceF(sRsShared.temp.reduceF).Sum(static_cast<float>(n));
        if (tid == 0)
        {
            sRsS.nList = static_cast<int>(total);
        }
        __syncthreads();
    }
    float const nMaxF = BlockReduceF(sRsShared.temp.reduceF).Reduce(static_cast<float>(n), FloatMaxOp());
    if (tid == 0)
    {
        sRsS.nMax = static_cast<int>(nMaxF);
    }
    __syncthreads();
    int const nMax = sRsS.nMax;
    for (int i = n; i < nMax; ++i)
    {
        listX[i * kVirtualThreads + tid] = kNegInf;
        listId[i * kVirtualThreads + tid] = -1;
    }
    __syncthreads();
    if (tid == 0)
    {
        sRsS.ft[0] = globalTimer();
    }
    // ---- The stock cutoff on the list (findThreshold sees w = exp(fl(x - M) * 1 - 0), the list's weights).
    int const length = nMax * kVirtualThreads;
    float thr = rp.needMinP ? rp.minP : 0.0f;
    if (rp.needTopK)
    {
        float const topKThreshold
            = fs::findThreshold<float>(listX, length, 1.0f, 0.0f, thr, static_cast<long long>(rp.topK), 0.0f,
                /*byCount=*/true, sRsShared.count, sRsShared.mass, sRsShared.cand, &sRsShared.candCount,
                &sRsShared.bucketCount, &sRsShared.chosen, &sRsShared.countHi, &sRsShared.massHi, &sRsShared.fired);
        thr = fmaxf(thr, topKThreshold);
    }
    float s1 = Z;
    if (rp.needTopP)
    {
        if (thr > 0.0f)
        {
            s1 = rsListMass(listX, n, thr);
        }
        float const topPThreshold = fs::findThreshold<float>(listX, length, 1.0f, 0.0f, thr, 0, __fmul_rn(rp.topP, s1),
            /*byCount=*/false, sRsShared.count, sRsShared.mass, sRsShared.cand, &sRsShared.candCount,
            &sRsShared.bucketCount, &sRsShared.chosen, &sRsShared.countHi, &sRsShared.massHi, &sRsShared.fired);
        thr = fmaxf(thr, topPThreshold);
    }
    float const tau = thr;
    // With tau = 0 the kept set is the whole row and its mass the states' sum (the stock pass-1 total).
    float const kept = tau > 0.0f ? rsListMass(listX, n, tau) : Z;
    if (tid == 0)
    {
        sRsS.tau = tau;
        sRsS.kept = kept;
        sRsS.s1 = s1;
        sRsS.phi = phi;
        sRsS.rawS = rawS;
        sRsS.ft[1] = globalTimer();
    }
    // ---- The decision: warp 0 decides on the candidate lanes; lane k < 16 holds candidate k.
    if (tid < 32)
    {
        int const d = sRsS.d;
        bool const rsRow = sRsS.rs != 0;
        int const id = lane < rs::kLanes ? sRsS.id[lane] : -1;
        bool const has = lane < rs::kLanes && id >= 0 && id < q.vocab;
        float const x = has ? sRsS.x[lane] : kNegInf;
        float const w = has ? __expf(__fsub_rn(x, M)) : 0.0f;
        float const wk = has && w >= tau ? w : 0.0f;
        float const qv = lane < rs::kLanes ? sRsS.qf[lane] : 0.0f;
        float const qk = has ? __fmul_rn(qv, kept) : 0.0f;
        uint32_t const hit = __ballot_sync(0xFFFFFFFFu, has && id == d);
        int const kd = hit != 0u ? __ffs(hit) - 1 : -1;
        uint32_t const peers = __match_any_sync(0xFFFFFFFFu, has ? id : -(lane + 2));
        bool const dupAny = __ballot_sync(0xFFFFFFFFu, has && __popc(peers) > 1) != 0u;
        bool const rangeOk = __ballot_sync(0xFFFFFFFFu, lane < rs::kLanes && !(id >= -1 && id < q.vocab)) == 0u;
        bool const poison = !(M > kNegInf && M < -kNegInf) || !(Z >= 1.0f && Z < -kNegInf)
            || __ballot_sync(0xFFFFFFFFu, has && !(w == w)) != 0u;
        int const kdc = kd < 0 ? 0 : kd;
        float const qd = __shfl_sync(0xFFFFFFFFu, qv, kdc);
        float const wkd = __shfl_sync(0xFFFFFFFFu, wk, kdc);
        float const qkd = __shfl_sync(0xFFFFFFFFu, qk, kdc);
        RsReason const reason
            = rsReason(rsRow, poison, sRsS.same != 0, sRsS.qBad != 0, sRsS.flags, kd, hit, dupAny, rangeOk, qd);
        bool const usable = reason == RsReason::kUsable;
        float const ua = sRsS.ua;
        bool const acc = usable && __fmul_rn(ua, qkd) < wkd;
        // The residual on the candidate lanes: max(w_c [w_c >= tau] - q_c K, 0), 0 at the draft; none on a plain row.
        float const rho = usable && !acc && has && lane != kd ? fmaxf(__fsub_rn(wk, qk), 0.0f) : 0.0f;
        if (lane < rs::kLanes)
        {
            sRsS.rho[lane] = rho;
            if (q.fexports != nullptr)
            {
                float* ex = q.fexports + static_cast<size_t>(row) * kFExportWords;
                ex[kFExportWk + lane] = wk;
                ex[kFExportQk + lane] = qk;
                ex[kFExportRho + lane] = rho;
            }
        }
        __syncwarp();
        if (lane == 0)
        {
            float rC = 0.0f;
            for (int k = 0; k < rs::kLanes; ++k)
            {
                rC = __fadd_rn(rC, sRsS.rho[k]);
            }
            sRsS.rC = rC;
            sRsS.out = acc ? d : -1;
            sRsS.claim = INT_MAX;
            sRsS.oflags = (acc ? kFlagAcc : 0u) | (usable ? kFlagUsable : 0u)
                | (static_cast<uint32_t>(reason) << kFlagReasonShift) | (usable && !(wkd > 0.0f) ? kFlagDUnkept : 0u)
                | (screenFallback ? kFlagScreenFallback : 0u) | (usable ? 0u : kFlagPlain)
                | ((hash & kFlagHashMask) << kFlagHashShift);
            if (q.fexports != nullptr)
            {
                float* ex = q.fexports + static_cast<size_t>(row) * kFExportWords;
                ex[kFExportUa] = row < rs::kPos ? ua : 0.0f;
                ex[kFExportKd] = __int_as_float(kd);
            }
        }
    }
    __syncthreads();
    bool const draw = sRsS.out < 0;
    bool const excludeCands = draw && (sRsS.oflags & kFlagUsable) != 0u;
    float ur = 0.0f;
    if (draw)
    {
        // ---- The residual's list part: this thread's column in i order, then the block scan.
        float partial = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            partial = __fadd_rn(partial,
                rsListRho(listX[i * kVirtualThreads + tid], listId[i * kVirtualThreads + tid], tau, excludeCands));
        }
        float excl = 0.0f;
        float rOff = 0.0f;
        fs::FusedSamplingShared<kVirtualThreads>::BlockScanF(sRsShared.temp.scanF).ExclusiveSum(partial, excl, rOff);
        if (tid == 0)
        {
            uint64_t const seedR = static_cast<uint64_t>(p.fs.seed[row]);
            int64_t const offR = static_cast<int64_t>(p.fs.offset[row]);
            ur = rs::unif(rs::Ph(seedR, rs::acceptSubseq(offR), 0).y);
            sRsS.rOff = rOff;
            sRsS.target = __fmul_rn(ur, __fadd_rn(rOff, sRsS.rC));
            sRsS.claim = INT_MAX;
            sRsS.lastTid = -1;
            sRsS.claimBase = 0.0f;
            sRsS.claimN = 0;
            sRsS.best = 0ull;
        }
        __syncthreads();
        float const target = sRsS.target;
        bool const listPart = target < rOff;
        if (listPart && partial > 0.0f)
        {
            // The claimant spans the target; several (rounding) or none are possible, so the smallest thread wins
            // and, with none, the last thread holding residual mass.
            if (excl <= target && target < __fadd_rn(excl, partial))
            {
                atomicMin(&sRsS.claim, tid);
            }
            atomicMax(&sRsS.lastTid, tid);
        }
        __syncthreads();
        if (listPart)
        {
            bool const noClaim = sRsS.claim == INT_MAX;
            int const claim = noClaim ? sRsS.lastTid : sRsS.claim;
            // Every thread has read the claim before the claimant overwrites it (noClaim decides the fallback flag).
            __syncthreads();
            if (tid == claim)
            {
                sRsS.claimBase = excl;
                sRsS.claimN = n;
                sRsS.claim = claim;
            }
            __syncthreads();
            // Stage the claimant's column (entry i: residual at cand[i], id at count[i]) and walk it in i order.
            int const cn = sRsS.claimN;
            for (int i = tid; i < cn; i += kVirtualThreads)
            {
                int const at = i * kVirtualThreads + claim;
                int const id = listId[at];
                sRsShared.cand[i] = rsListRho(listX[at], id, tau, excludeCands);
                sRsShared.count[i] = id;
            }
            __syncthreads();
            if (tid == 0)
            {
                float running = sRsS.claimBase;
                int y = -1;
                int last = -1;
                for (int i = 0; i < cn && y < 0; ++i)
                {
                    float const r = sRsShared.cand[i];
                    if (r > 0.0f)
                    {
                        last = sRsShared.count[i];
                        running = __fadd_rn(running, r);
                        if (running > target)
                        {
                            y = last;
                        }
                    }
                }
                sRsS.out = y >= 0 ? y : last;
                if (y < 0 || noClaim)
                {
                    sRsS.oflags |= kFlagResidFallback;
                }
            }
        }
        else if (tid == 0)
        {
            // The candidate part, in lane order.
            float running = rOff;
            int y = -1;
            int last = -1;
            for (int k = 0; k < rs::kLanes && y < 0; ++k)
            {
                float const r = sRsS.rho[k];
                if (r > 0.0f)
                {
                    last = sRsS.id[k];
                    running = __fadd_rn(running, r);
                    if (running > target)
                    {
                        y = last;
                    }
                }
            }
            sRsS.out = y >= 0 ? y : last;
            if (y < 0)
            {
                sRsS.oflags |= kFlagResidFallback;
            }
        }
        __syncthreads();
        if (sRsS.out < 0)
        {
            // No residual mass where the target fell (not reachable in exact arithmetic): the kept element other
            // than the draft with the largest weight, ties to the smaller id.
            unsigned long long best = 0ull;
            int const d = excludeCands ? sRsS.d : -1;
            for (int i = 0; i < n; ++i)
            {
                int const id = listId[i * kVirtualThreads + tid];
                float const w = __expf(listX[i * kVirtualThreads + tid]);
                if (id >= 0 && id != d && w >= tau)
                {
                    unsigned long long const key = static_cast<unsigned long long>(__float_as_uint(w)) << 32
                        | (0xFFFFFFFFu - static_cast<uint32_t>(id));
                    best = key > best ? key : best;
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                unsigned long long const o = __shfl_xor_sync(0xFFFFFFFFu, best, off);
                best = o > best ? o : best;
            }
            if (lane == 0 && best != 0ull)
            {
                atomicMax(&sRsS.best, best);
            }
            __syncthreads();
            if (tid == 0)
            {
                unsigned long long const b = sRsS.best;
                sRsS.out = b == 0ull ? sRsS.ioff : static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(b));
                sRsS.oflags |= kFlagResidFallback;
            }
            __syncthreads();
        }
    }
    // ---- The token and flags through the owner's entry of the token slot.
    if (tid == 0)
    {
        sRsS.ft[2] = globalTimer();
        publishPair(p.mcSlot + slotAt, tag | static_cast<uint32_t>(sRsS.out), tag | sRsS.oflags);
        sRsS.owner = 1;
        sRsS.hashBad = 0;
        sRsS.ft[3] = globalTimer();
        if (q.fexports != nullptr)
        {
            float* ex = q.fexports + static_cast<size_t>(row) * kFExportWords;
            ex[kFExportPhi] = sRsS.phi;
            ex[kFExportRawS] = sRsS.rawS;
            ex[kFExportTau] = sRsS.tau;
            ex[kFExportKept] = kept;
            ex[kFExportS1] = sRsS.s1;
            ex[kFExportN] = __int_as_float(sRsS.nList);
            ex[kFExportNmax] = __int_as_float(nMax);
            ex[kFExportROff] = draw ? sRsS.rOff : 0.0f;
            ex[kFExportRC] = draw ? sRsS.rC : 0.0f;
            ex[kFExportTarget] = draw ? sRsS.target : 0.0f;
            ex[kFExportUr] = ur;
            ex[kFExportClaim] = __int_as_float(draw ? sRsS.claim : -1);
            ex[kFExportOut] = __int_as_float(sRsS.out);
            ex[kFExportFlags] = __uint_as_float(sRsS.oflags);
            ex[kFExportY] = __int_as_float(draw ? sRsS.out : -1);
            ex[kFExportM] = M;
            ex[kFExportZ] = Z;
            ex[kFExportCase] = __int_as_float(screenCase);
        }
    }
    if (q.fexports != nullptr)
    {
        __syncthreads();
        // Kept elements with a positive weight (the stock probabilities' support).
        int keptCount = 0;
        for (int i = 0; i < n; ++i)
        {
            float const w = __expf(listX[i * kVirtualThreads + tid]);
            keptCount += w >= tau && w > 0.0f ? 1 : 0;
        }
        float const kc = BlockReduceF(sRsShared.temp.reduceF).Sum(static_cast<float>(keptCount));
        if (tid == 0)
        {
            q.fexports[static_cast<size_t>(row) * kFExportWords + kFExportKeptCount]
                = __int_as_float(static_cast<int>(kc));
        }
    }
    __syncthreads();
}

//! Thread 0 of a row rsFilteredRow decided: the token, the stamps, the RsDiag counters from the row's flags word
//! (equal on every rank but the owner-only words), a parameter-hash mismatch (trap code 7), Knobs.check.
__device__ __forceinline__ void rsFilteredTail(DrawParams const& p, RsDrawParams const& q, int row)
{
    uint32_t const fl = sRsS.oflags;
    if (sRsS.hashBad != 0)
    {
        // This rank's sampling parameters of the row differ from its owner's: the ranks staged divergent params.
        rs::trap(q.diag, rs::TrapCode::kFilterParams, row);
    }
    p.fs.outputTokens[row] = sRsS.out;
    if (p.stamps != nullptr)
    {
        atomicMax(p.stamps + p.slot * 4 + 3, globalTimer());
    }
    bool const owner = sRsS.owner != 0;
    if (q.fstamps != nullptr)
    {
        unsigned long long* const f = q.fstamps + p.slot * 4;
        if (owner)
        {
            atomicMax(f + 0, sRsS.ft[0]);
            atomicMax(f + 1, sRsS.ft[1]);
            atomicMax(f + 2, sRsS.ft[2]);
        }
        atomicMax(f + 3, sRsS.ft[3]);
    }
    bool const usable = (fl & kFlagUsable) != 0u;
    if (row < rs::kPos)
    {
        rs::diagAdd(q.diag, rs::Diag::kFilteredRows);
        if (usable)
        {
            rs::diagAdd(q.diag, rs::Diag::kFiltRsRows);
            rs::diagAdd(q.diag, (fl & kFlagAcc) != 0u ? rs::Diag::kFiltAccepts : rs::Diag::kFiltRejects);
            if ((fl & kFlagDUnkept) != 0u)
            {
                rs::diagAdd(q.diag, rs::Diag::kFiltDUnkept);
            }
        }
        else
        {
            rs::diagAdd(q.diag, reasonCounter(static_cast<RsReason>((fl >> kFlagReasonShift) & 0xFu)));
            if (sRsS.validBit != 0)
            {
                rs::diagAdd(q.diag, rs::Diag::kConsumedUnused);
            }
        }
    }
    if ((fl & kFlagPlain) != 0u)
    {
        rs::diagAdd(q.diag, rs::Diag::kFiltPlainRows);
    }
    if ((fl & kFlagResidFallback) != 0u)
    {
        rs::diagAdd(q.diag, rs::Diag::kFiltResidFallback);
    }
    if (owner)
    {
        rs::diagAdd(q.diag, rs::Diag::kFiltOwnerRows);
        rs::diagAdd(q.diag, rs::Diag::kFiltListEntries, static_cast<unsigned long long>(sRsS.nList));
        rs::diagMax(q.diag, rs::Diag::kFiltListMax, static_cast<unsigned long long>(sRsS.nList));
        if ((fl & kFlagScreenFallback) != 0u)
        {
            rs::diagAdd(q.diag, rs::Diag::kFiltScreenFallback);
        }
    }
    if (q.check != 0 && row < rs::kPos && sRsS.dummy == 0)
    {
        if (!usable)
        {
            rs::trap(q.diag, rs::TrapCode::kCheckStrict, row);
        }
        if ((fl & kFlagResidFallback) != 0u)
        {
            rs::trap(q.diag, rs::TrapCode::kCheckStrict, kCheckResidDetail + row);
        }
    }
}

//! The RS draw of one row on every rank (see the file header). Grid rows; block kVirtualThreads. Same counters,
//! trigger and final wait as s2DrawKernel. Every row polls the candidate slots, digests (t < 7) and states; a
//! neutral row then decides in warp 0, a filtered row runs rsFilteredRow (one owner rank per row, the token through
//! the token slot) or, under the strict filter or a non-finite max / sum, rsStrictFilteredRow.
template <bool kTrigger>
__global__ void __launch_bounds__(kVirtualThreads, 1)
    s2RsDrawKernel(__grid_constant__ DrawParams const p, __grid_constant__ RsDrawParams const q)
{
    float const kNegInf = -__int_as_float(0x7F800000);
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    if constexpr (kTrigger)
    {
        gridDependencyTrigger();
    }
    if (tid == 0)
    {
        fs::sSplitCap = fs::SplitCap{p.cap.multiplier, p.cap.invCap, p.cap.cap};
        sRsS.epoch = p.counters[row] + 1u;
    }
    __syncthreads();
    uint32_t const epoch = sRsS.epoch;
    uint64_t const parity = epoch & 1u;
    fs::RowParams const rp = fs::loadRowParams(p.fs, row);
    if (row == 0 && tid == 0)
    {
        int64_t const slot = q.slotIds[0];
        if (!(slot >= 0 && slot < q.numSlots))
        {
            rs::diagAdd(q.diag, rs::Diag::kSlotOob);
            if (q.check != 0)
            {
                rs::trap(q.diag, rs::TrapCode::kCheckSlot, slot);
            }
        }
    }
    // The neutral body first: its decision then follows the polls near the start of the kernel text instead of
    // behind the filtered bodies, where it would be fetched cold after the LM-head GEMM has flushed L2
    // (test_modeling_v2_rs_filtered.py checks the placement of its token store).
    if (!filteredRow(rp))
    {
        rsPrologue(p, q, row);
        rsPollCandDig(p, q, row, epoch, parity);
        rsPollStates(p, q, row, epoch, parity);
        // ---- Warp 0: the decision; lanes 0 .. 15 are the candidate lanes, lane 16 the off-candidate winner.
        if (tid < 32)
        {
            float const M = sRsS.M;
            float const Z = sRsS.Z;
            int const d = sRsS.d;
            bool const rsRow = sRsS.rs != 0;
            int const id = lane < rs::kLanes ? sRsS.id[lane] : -1;
            bool const has = lane < rs::kLanes && id >= 0 && id < q.vocab;
            float const x = has ? sRsS.x[lane] : kNegInf;
            float const w = has ? __expf(__fsub_rn(x, M)) : 0.0f;
            float const qv = lane < rs::kLanes ? sRsS.qf[lane] : 0.0f;
            float const qZ = has ? __fmul_rn(qv, Z) : 0.0f;
            float const logE = lane < rs::kLanes ? sRsS.logE[lane] : 0.0f;
            uint32_t const hit = __ballot_sync(0xFFFFFFFFu, has && id == d);
            int const kd = hit != 0u ? __ffs(hit) - 1 : -1;
            uint32_t const peers = __match_any_sync(0xFFFFFFFFu, has ? id : -(lane + 2));
            bool const dupAny = __ballot_sync(0xFFFFFFFFu, has && __popc(peers) > 1) != 0u;
            bool const rangeOk = __ballot_sync(0xFFFFFFFFu, lane < rs::kLanes && !(id >= -1 && id < q.vocab)) == 0u;
            bool const poison = !(M > kNegInf && M < -kNegInf) || !(Z >= 1.0f && Z < -kNegInf)
                || __ballot_sync(0xFFFFFFFFu, has && !(w == w)) != 0u;
            int const kdc = kd < 0 ? 0 : kd;
            float const qd = __shfl_sync(0xFFFFFFFFu, qv, kdc);
            float const wd = __shfl_sync(0xFFFFFFFFu, w, kdc);
            float const qZd = __shfl_sync(0xFFFFFFFFu, qZ, kdc);
            RsReason const reason
                = rsReason(rsRow, poison, sRsS.same != 0, sRsS.qBad != 0, sRsS.flags, kd, hit, dupAny, rangeOk, qd);
            bool const usable = reason == RsReason::kUsable;
            float const ua = sRsS.ua;
            bool const acc = usable && __fmul_rn(ua, qZd) < wd;
            float kc = kNegInf;
            if (usable && !acc)
            {
                // The residual: rho(c) = e^M max(w_c - q_c Z, 0), rho(d) = 0.
                float const r = lane == kd ? 0.0f : fmaxf(__fsub_rn(w, qZ), 0.0f);
                kc = has && r > 0.0f ? __fsub_rn(__fadd_rn(logf(r), M), logE) : kNegInf;
            }
            else if (rsRow)
            {
                // Strict: the plain race; the fold excluded this row's candidates, so they enter here.
                kc = has ? __fsub_rn(x, logE) : kNegInf;
            }
            kc = kc == kc ? kc : kNegInf;
            int kid = has ? id : INT_MAX;
            float key = kc;
            if (lane == rs::kLanes)
            {
                key = sRsS.koff;
                kid = sRsS.ioff;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                float const k2 = __shfl_xor_sync(0xFFFFFFFFu, key, off);
                int const i2 = __shfl_xor_sync(0xFFFFFFFFu, kid, off);
                if (rs::better(k2, i2, key, kid))
                {
                    key = k2;
                    kid = i2;
                }
            }
            int const out = acc ? d : kid;
            if (q.exports != nullptr)
            {
                float* ex = q.exports + static_cast<size_t>(row) * kExportWords;
                if (lane < rs::kLanes)
                {
                    ex[kExportW + lane] = w;
                    ex[kExportQz + lane] = qZ;
                    ex[kExportKc + lane] = kc;
                    ex[kExportX + lane] = x;
                    ex[kExportLogE + lane] = logE;
                }
                if (lane == 0)
                {
                    ex[kExportM] = M;
                    ex[kExportZ] = Z;
                    ex[kExportKoff] = sRsS.koff;
                    ex[kExportIoff] = __int_as_float(sRsS.ioff);
                    ex[kExportKd] = __int_as_float(kd);
                    ex[kExportAcc] = __int_as_float(acc ? 1 : 0);
                    ex[kExportY] = __int_as_float(kid);
                    ex[kExportOut] = __int_as_float(out);
                    ex[kExportUsable] = __int_as_float(usable ? 1 : 0);
                    ex[kExportReason] = __int_as_float(static_cast<int>(reason));
                    ex[kExportUa] = row < rs::kPos ? ua : 0.0f;
                    ex[kExportKstar] = key;
                }
            }
            if (lane == 0)
            {
                p.fs.outputTokens[row] = out;
                if (p.stamps != nullptr)
                {
                    atomicMax(p.stamps + p.slot * 4 + 3, globalTimer());
                }
                if (row < rs::kPos)
                {
                    rs::diagAdd(q.diag, rs::Diag::kRsRows);
                    if (usable)
                    {
                        rs::diagAdd(q.diag, acc ? rs::Diag::kAccepts : rs::Diag::kRejects);
                    }
                    else
                    {
                        rs::diagAdd(q.diag, reasonCounter(reason));
                        if (sRsS.validBit != 0)
                        {
                            rs::diagAdd(q.diag, rs::Diag::kConsumedUnused);
                        }
                    }
                    if (sRsS.dummy != 0)
                    {
                        rs::diagAdd(q.diag, rs::Diag::kDummyRows);
                    }
                    else if (q.check != 0 && !usable)
                    {
                        rs::trap(q.diag, rs::TrapCode::kCheckStrict, row);
                    }
                    if (q.ring != nullptr && usable)
                    {
                        // Unsigned: the head wraps past 2^31 on a long run and must still index inside the ring.
                        unsigned const at
                            = static_cast<unsigned>(atomicAdd(q.ringHead, 1)) % static_cast<unsigned>(q.ringSize);
                        float* entry = q.ring + static_cast<size_t>(at) * 4;
                        entry[0] = __fdiv_rn(wd, qZd);
                        entry[1] = ua;
                        entry[2] = acc ? 1.0f : 0.0f;
                        entry[3] = static_cast<float>(row);
                    }
                }
            }
        }
    }
    else
    {
        rsPrologue(p, q, row);
        rsPollCandDig(p, q, row, epoch, parity);
        rsPollStates(p, q, row, epoch, parity);
        float const M = sRsS.M;
        float const Z = sRsS.Z;
        bool const strict = q.filterMode == static_cast<int>(rs::FilterMode::kStrict);
        bool const lastResort = !(M > kNegInf && M < -kNegInf) || !(Z >= 1.0f && Z < -kNegInf);
        if (strict || lastResort)
        {
            rsStrictFilteredRow(p, row, epoch, parity);
            if (tid == 0 && row < rs::kPos)
            {
                rs::diagAdd(q.diag, rs::Diag::kFilteredRows);
                rs::diagAdd(q.diag, strict ? rs::Diag::kFiltStrictRows : rs::Diag::kFiltStrictLastResort);
            }
        }
        else
        {
            rsFilteredRow(p, q, row, epoch, parity);
            if (tid == 0)
            {
                rsFilteredTail(p, q, row);
            }
        }
    }
    if (tid == 0 && row == rs::kPos)
    {
        rs::diagAdd(q.diag, rs::Diag::kRsDrawCalls);
    }
    // Completion implies the fold grid's (the next call's counter reads); after it no fold CTA of this rank still
    // reads the record, so the row's position is consumed here.
    gridDependencyWait();
    if (tid == 0)
    {
        p.counters[row] = sRsS.epoch;
        int64_t const slot = q.slotIds[0];
        if (row < rs::kPos && slot >= 0 && slot < q.numSlots)
        {
            atomicAnd(
                reinterpret_cast<unsigned int*>(q.records + slot * rs::kRecordBytes + offsetof(rs::QRecord, valid)),
                ~(1u << row));
        }
    }
}

//! Bench helper: one thread stores the globaltimer into stamps[index] after its PDL wait.
__global__ void s2StampKernel(unsigned long long* stamps, int index)
{
    gridDependencyWait();
    stamps[index] = globalTimer();
    gridDependencyTrigger();
}

__global__ void s2DryKernel() {}

template <typename Kernel, typename... Args>
void launchEx(Kernel kernel, dim3 grid, dim3 block, size_t smem, bool pdl, Args... args)
{
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

template <int VTS, int THREADS, int FORM, bool kRs = false>
void launchFoldShape(FoldParams const& p, RsFoldParams const& q, int rows, bool pdl)
{
    auto kernel = s2FoldPublishKernel<VTS, THREADS, FORM, kRs>;
    int const steps = (p.localVecs + p.localThreads - 1) / p.localThreads;
    size_t const smem = static_cast<size_t>(steps) * VTS * sizeof(float4);
    static bool const raised = []
    {
        size_t const most = static_cast<size_t>(kWalkVectors) * VTS * sizeof(float4);
        C10_CUDA_CHECK(cudaFuncSetAttribute(s2FoldPublishKernel<VTS, THREADS, FORM, kRs>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(most)));
        return true;
    }();
    (void) raised;
    launchEx(kernel, dim3(rows, p.localThreads / VTS), dim3(THREADS), smem, pdl, p, q);
}

template <int FORM>
void launchFold(FoldParams const& p, int rows, int64_t vts, int64_t threads, bool pdl)
{
#define S2_FOLD_SHAPE(V, T)                                                                                            \
    if (vts == V && threads == T)                                                                                      \
    {                                                                                                                  \
        launchFoldShape<V, T, FORM>(p, RsFoldParams{}, rows, pdl);                                                     \
        return;                                                                                                        \
    }
    S2_FOLD_SHAPE(2, 128)
    S2_FOLD_SHAPE(4, 64)
    S2_FOLD_SHAPE(4, 128)
    S2_FOLD_SHAPE(4, 256)
    S2_FOLD_SHAPE(8, 128)
    S2_FOLD_SHAPE(8, 256)
    S2_FOLD_SHAPE(16, 256)
    S2_FOLD_SHAPE(8, 512)
    S2_FOLD_SHAPE(16, 512)
    S2_FOLD_SHAPE(32, 512)
#undef S2_FOLD_SHAPE
    TORCH_CHECK(false, "s2 fold: unsupported (virtual threads per CTA, threads) = (", vts, ", ", threads, ")");
}

int64_t foldCtas(int64_t rows, int64_t world, int64_t vts)
{
    return rows * (kVirtualThreads / world) / vts;
}

void checkCounters(at::Tensor const& counters, int64_t need, char const* name)
{
    TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == at::kInt && counters.is_contiguous()
            && counters.numel() >= need,
        "s2: ", name, " must be a contiguous int32 CUDA tensor of at least ", need, " entries");
}

void checkStamps(std::optional<at::Tensor> const& stamps, int64_t slot)
{
    if (stamps.has_value())
    {
        auto const& s = stamps.value();
        TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kLong && s.is_contiguous() && s.dim() == 2 && s.size(1) == 4
                && slot >= 0 && slot < s.size(0),
            "s2: stamps must be a contiguous int64 [slots, 4] CUDA tensor and slot inside it");
    }
}

void checkRowParam(at::Tensor const& t, char const* name, at::ScalarType dtype, int64_t rows)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.dim() == 1 && t.size(0) >= rows && t.is_contiguous(),
        "s2: ", name, " must be a contiguous 1-D CUDA tensor of the right dtype with at least numRows entries");
}

//! Bytes of one row area: every rank's local vectors of `rows` rows, one pair of 64-bit elements per vector.
int64_t rowAreaNeed(int64_t rows, int64_t world, int64_t localVecs)
{
    return rows * world * localVecs * 16;
}

//! The checked FoldParams of a fold_publish call (see s2FoldPublishKernel).
FoldParams foldParams(at::Tensor const& local, at::Tensor const& temperatures, at::Tensor const& counters,
    int64_t mcArea, int64_t areaBytes, int64_t rank, int64_t world, double multiplier, double invCap, double cap,
    int64_t vts, std::optional<at::Tensor> const& ownStates, std::optional<at::Tensor> const& stamps, int64_t slot,
    std::optional<at::Tensor> const& topKs, std::optional<at::Tensor> const& topPs,
    std::optional<at::Tensor> const& minPs, int64_t mcRows, int64_t rowAreaBytes)
{
    TORCH_CHECK(local.is_cuda() && local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous(),
        "s2 fold: local must be a contiguous BF16 [rows, shard] CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0, "s2 fold: local must be 8-byte aligned");
    int64_t const rows = local.size(0);
    int64_t const shard = local.size(1);
    TORCH_CHECK(rows >= 1 && rows <= fs::kWideBlockMaxRows, "s2 fold: 1..", fs::kWideBlockMaxRows, " rows");
    TORCH_CHECK(world >= 1 && kVirtualThreads % world == 0 && rank >= 0 && rank < world, "s2 fold: bad rank / world");
    TORCH_CHECK(shard % 4 == 0, "s2 fold: the shard is a whole number of 4-element vectors");
    int64_t const localThreads = kVirtualThreads / world;
    int64_t const localVecs = shard / 4;
    TORCH_CHECK((localVecs + localThreads - 1) / localThreads <= kWalkVectors, "s2 fold: vocabulary too wide");
    TORCH_CHECK(vts >= 1 && localThreads % vts == 0, "s2 fold: virtual threads per CTA must divide ", localThreads);
    TORCH_CHECK(temperatures.is_cuda() && temperatures.scalar_type() == at::kFloat && temperatures.dim() == 1
            && temperatures.size(0) >= rows && temperatures.is_contiguous(),
        "s2 fold: temperatures must be a contiguous FP32 [>= rows] CUDA tensor");
    checkCounters(counters, foldCtas(rows, world, vts), "fold counters");
    TORCH_CHECK(mcArea % 16 == 0, "s2 fold: misaligned area");
    TORCH_CHECK(areaBytes >= rows * kVirtualThreads * kStateElems * 8, "s2 fold: area too small");
    if (ownStates.has_value())
    {
        auto const& s = ownStates.value();
        TORCH_CHECK(
            s.is_cuda() && s.scalar_type() == at::kInt && s.is_contiguous() && s.numel() >= rows * kVirtualThreads * 3,
            "s2 fold: own_states holds 3 words per virtual thread");
    }
    checkStamps(stamps, slot);
    FoldParams p;
    if (mcRows != 0)
    {
        TORCH_CHECK(topKs.has_value() && topPs.has_value() && minPs.has_value(),
            "s2 fold: the row area needs top_ks / top_ps / min_ps (they decide which rows are filtered)");
        checkRowParam(topKs.value(), "top_ks", at::kInt, rows);
        checkRowParam(topPs.value(), "top_ps", at::kFloat, rows);
        checkRowParam(minPs.value(), "min_ps", at::kFloat, rows);
        TORCH_CHECK(mcRows % 16 == 0 && rowAreaBytes >= rowAreaNeed(rows, world, shard / 4),
            "s2 fold: misaligned or too small row area");
        p.fs.topKs = topKs.value().data_ptr<int32_t>();
        p.fs.topPs = topPs.value().data_ptr<float>();
        p.fs.minPs = minPs.value().data_ptr<float>();
        p.mcRows = static_cast<uint64_t>(mcRows);
        p.rowAreaBytes = static_cast<uint64_t>(rowAreaBytes);
    }
    p.fs.temperatures = temperatures.data_ptr<float>();
    p.fs.numRows = static_cast<int32_t>(rows);
    p.fs.vocabSize = static_cast<int32_t>(shard * world);
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    p.mcArea = static_cast<uint64_t>(mcArea);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.ownStates = ownStates.has_value() ? reinterpret_cast<fs::OnlineSoftmax*>(ownStates.value().data_ptr()) : nullptr;
    p.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
    p.slot = static_cast<int>(slot);
    p.cap = Cap{static_cast<float>(multiplier), static_cast<float>(invCap), static_cast<float>(cap)};
    p.rank = static_cast<int>(rank);
    p.world = static_cast<int>(world);
    p.localVecs = static_cast<int>(localVecs);
    p.localThreads = static_cast<int>(localThreads);
    return p;
}

//! This rank's fold of its virtual threads and the publish of their states (see s2FoldPublishKernel).
void foldPublish(at::Tensor const& local, at::Tensor const& temperatures, at::Tensor const& counters, int64_t mcArea,
    int64_t areaBytes, int64_t rank, int64_t world, double multiplier, double invCap, double cap, int64_t vts,
    int64_t threads, int64_t form, bool pdl, std::optional<at::Tensor> const& ownStates,
    std::optional<at::Tensor> const& stamps, int64_t slot, std::optional<at::Tensor> const& topKs,
    std::optional<at::Tensor> const& topPs, std::optional<at::Tensor> const& minPs, int64_t mcRows,
    int64_t rowAreaBytes)
{
    at::cuda::CUDAGuard guard(local.device());
    FoldParams const p = foldParams(local, temperatures, counters, mcArea, areaBytes, rank, world, multiplier, invCap,
        cap, vts, ownStates, stamps, slot, topKs, topPs, minPs, mcRows, rowAreaBytes);
    int64_t const rows = local.size(0);
    if (form == 2)
    {
        launchFold<2>(p, static_cast<int>(rows), vts, threads, pdl);
    }
    else if (form == 1)
    {
        launchFold<1>(p, static_cast<int>(rows), vts, threads, pdl);
    }
    else
    {
        TORCH_CHECK(form == 0, "s2 fold: form 0, 1 or 2");
        launchFold<0>(p, static_cast<int>(rows), vts, threads, pdl);
    }
}

//! Every rank's states of `rows` rows into `states` ([rows][1024] OnlineSoftmax, 3 int32 words each).
void pollStates(at::Tensor const& states, at::Tensor const& counters, int64_t ucArea, int64_t areaBytes, int64_t rows,
    int64_t world, int64_t timeoutNs, bool pdl, std::optional<at::Tensor> const& stamps, int64_t slot)
{
    at::cuda::CUDAGuard guard(states.device());
    TORCH_CHECK(states.is_cuda() && states.scalar_type() == at::kInt && states.is_contiguous()
            && states.numel() >= rows * kVirtualThreads * 3,
        "s2 poll: states holds 3 words per virtual thread");
    TORCH_CHECK(rows >= 1 && rows <= fs::kWideBlockMaxRows, "s2 poll: 1..", fs::kWideBlockMaxRows, " rows");
    TORCH_CHECK(world >= 1 && kVirtualThreads % world == 0, "s2 poll: bad world");
    checkCounters(counters, rows, "poll counters");
    TORCH_CHECK(ucArea % 16 == 0, "s2 poll: misaligned area");
    TORCH_CHECK(areaBytes >= rows * kVirtualThreads * kStateElems * 8, "s2 poll: area too small");
    checkStamps(stamps, slot);
    PollParams p;
    p.ucArea = static_cast<uint64_t>(ucArea);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.states = reinterpret_cast<fs::OnlineSoftmax*>(states.data_ptr());
    p.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
    p.slot = static_cast<int>(slot);
    p.world = static_cast<int>(world);
    p.localThreads = static_cast<int>(kVirtualThreads / world);
    p.timeoutNs = static_cast<unsigned long long>(timeoutNs);
    launchEx(s2PollStatesKernel, dim3(rows), dim3(kPollThreads), 0, pdl, p);
}

//! The checked DrawParams of a draw call and its int32 [rows] tokens (see draw).
DrawParams drawParams(at::Tensor const& local, at::Tensor const& scratch, at::Tensor const& temperatures,
    at::Tensor const& topKs, at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& counters, int64_t ucStates, int64_t areaBytes,
    int64_t ucSlot, int64_t mcSlot, int64_t slotBytes, int64_t ucRows, int64_t rowAreaBytes, int64_t rank,
    int64_t world, double multiplier, double invCap, double cap, int64_t timeoutNs,
    std::optional<at::Tensor> const& stamps, int64_t slot, int64_t forceMulti, at::Tensor& tokens)
{
    TORCH_CHECK(local.is_cuda() && local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous(),
        "s2 draw: local must be a contiguous BF16 [rows, shard] CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0, "s2 draw: local must be 8-byte aligned");
    int64_t const rows = local.size(0);
    int64_t const shard = local.size(1);
    TORCH_CHECK(rows >= 1 && rows <= fs::kWideBlockMaxRows, "s2 draw: 1..", fs::kWideBlockMaxRows, " rows");
    TORCH_CHECK(world >= 1 && kVirtualThreads % world == 0 && rank >= 0 && rank < world, "s2 draw: bad rank / world");
    TORCH_CHECK(world <= 32, "s2 draw: the claimant mask holds 32 ranks");
    TORCH_CHECK(shard % 4 == 0, "s2 draw: the shard is a whole number of 4-element vectors");
    int64_t const localThreads = kVirtualThreads / world;
    int64_t const localVecs = shard / 4;
    TORCH_CHECK((localVecs + localThreads - 1) / localThreads <= kWalkVectors, "s2 draw: vocabulary too wide");
    TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == at::kBFloat16 && scratch.dim() == 2
            && scratch.is_contiguous() && scratch.size(0) >= rows && scratch.size(1) == shard * world,
        "s2 draw: scratch must be a contiguous BF16 [>= rows, world * shard] CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(scratch.data_ptr()) % 8 == 0, "s2 draw: scratch must be 8-byte aligned");
    checkRowParam(temperatures, "temperatures", at::kFloat, rows);
    checkRowParam(topKs, "top_ks", at::kInt, rows);
    checkRowParam(topPs, "top_ps", at::kFloat, rows);
    checkRowParam(minPs, "min_ps", at::kFloat, rows);
    checkCounters(counters, rows, "draw counters");
    TORCH_CHECK(ucStates % 16 == 0 && ucSlot % 16 == 0 && mcSlot % 16 == 0 && ucRows % 16 == 0, "s2 draw: misaligned");
    TORCH_CHECK(areaBytes >= rows * kVirtualThreads * kStateElems * 8, "s2 draw: state area too small");
    TORCH_CHECK(slotBytes >= rows * world * 16, "s2 draw: token slot too small");
    TORCH_CHECK(rowAreaBytes >= rowAreaNeed(rows, world, localVecs), "s2 draw: row area too small");
    checkStamps(stamps, slot);
    DrawParams p;
    p.fs.logits = scratch.data_ptr();
    p.fs.temperatures = temperatures.data_ptr<float>();
    p.fs.topKs = topKs.data_ptr<int32_t>();
    p.fs.topPs = topPs.data_ptr<float>();
    p.fs.minPs = minPs.data_ptr<float>();
    p.fs.numRows = static_cast<int32_t>(rows);
    p.fs.vocabSize = static_cast<int32_t>(shard * world);
    TORCH_CHECK(seed.has_value() == offset.has_value(), "seed and offset must be provided together");
    if (seed.has_value())
    {
        auto const& s = seed.value();
        auto const& o = offset.value();
        TORCH_CHECK(s.is_cuda() && o.is_cuda(), "seed and offset must be CUDA tensors");
        TORCH_CHECK(s.scalar_type() == at::kLong && o.scalar_type() == at::kLong, "seed and offset must be int64");
        TORCH_CHECK(s.numel() == o.numel(), "seed and offset must have the same length");
        TORCH_CHECK(s.numel() == 1 || s.numel() >= rows, "seed/offset must hold 1 or numRows entries");
        p.fs.seed = reinterpret_cast<uint64_t const*>(s.data_ptr<int64_t>());
        p.fs.offset = reinterpret_cast<uint64_t const*>(o.data_ptr<int64_t>());
        p.fs.perRowRng = s.numel() >= rows;
    }
    tokens = at::empty({rows}, local.options().dtype(at::kInt));
    p.fs.outputTokens = tokens.data_ptr<int32_t>();
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    p.ucStates = static_cast<uint64_t>(ucStates);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.ucSlot = static_cast<uint64_t>(ucSlot);
    p.mcSlot = static_cast<uint64_t>(mcSlot);
    p.slotBytes = static_cast<uint64_t>(slotBytes);
    p.ucRows = static_cast<uint64_t>(ucRows);
    p.rowAreaBytes = static_cast<uint64_t>(rowAreaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
    p.slot = static_cast<int>(slot);
    p.forceMulti = static_cast<int>(forceMulti);
    p.cap = Cap{static_cast<float>(multiplier), static_cast<float>(invCap), static_cast<float>(cap)};
    p.rank = static_cast<int>(rank);
    p.world = static_cast<int>(world);
    p.localVecs = static_cast<int>(localVecs);
    p.localThreads = static_cast<int>(localThreads);
    p.timeoutNs = static_cast<unsigned long long>(timeoutNs);
    return p;
}

//! The draw of `rows` rows on this rank (see s2DrawKernel); the int32 [rows] tokens. `scratch` is the
//! [rows, world * shard] BF16 row the filtered path fills. The row-parameter and seed / offset checks are the served
//! draw's (head_sampler.cu rowParams).
at::Tensor draw(at::Tensor const& local, at::Tensor const& scratch, at::Tensor const& temperatures,
    at::Tensor const& topKs, at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& counters, int64_t ucStates, int64_t areaBytes,
    int64_t ucSlot, int64_t mcSlot, int64_t slotBytes, int64_t ucRows, int64_t rowAreaBytes, int64_t rank,
    int64_t world, double multiplier, double invCap, double cap, int64_t timeoutNs, bool pdl, bool trigger,
    std::optional<at::Tensor> const& stamps, int64_t slot, int64_t forceMulti)
{
    at::cuda::CUDAGuard guard(local.device());
    at::Tensor tokens;
    DrawParams const p = drawParams(local, scratch, temperatures, topKs, topPs, minPs, seed, offset, counters, ucStates,
        areaBytes, ucSlot, mcSlot, slotBytes, ucRows, rowAreaBytes, rank, world, multiplier, invCap, cap, timeoutNs,
        stamps, slot, forceMulti, tokens);
    int64_t const rows = local.size(0);
    if (trigger)
    {
        launchEx(s2DrawKernel<true>, dim3(rows), dim3(kVirtualThreads), 0, pdl, p);
    }
    else
    {
        launchEx(s2DrawKernel<false>, dim3(rows), dim3(kVirtualThreads), 0, pdl, p);
    }
    return tokens;
}

//! Bytes of one candidate-slot area and of one digest area (rows t < 7).
int64_t candAreaNeed(int64_t world)
{
    return int64_t{rs::kPos} * rs::kLanes * world * 16;
}

int64_t digAreaNeed(int64_t world)
{
    return int64_t{rs::kPos} * world * 16;
}

//! Entries of one filtered row's survivor list: every element of the row, thread-major (4 per vector, vector
//! tid + 1024 j on thread tid), so 4 ceil(vecCount / 1024) entries per thread.
int64_t rsListCap(int64_t vocab)
{
    return 4 * ((vocab / 4 + kVirtualThreads - 1) / kVirtualThreads) * kVirtualThreads;
}

void checkRsInputs(int64_t world, at::Tensor const& records, int64_t numSlots, at::Tensor const& slotIds,
    at::Tensor const& draftIn, int64_t candAddr, int64_t candBytes, int64_t digAddr, int64_t digBytes)
{
    TORCH_CHECK(world >= 1 && world <= 8, "s2 rs: world 1 .. 8 (a candidate's claims form one group of lanes)");
    TORCH_CHECK(records.is_cuda() && records.scalar_type() == at::kByte && records.is_contiguous() && numSlots >= 0
            && records.numel() >= numSlots * rs::kRecordBytes
            && reinterpret_cast<uintptr_t>(records.data_ptr()) % alignof(rs::QRecord) == 0,
        "s2 rs: records must be a contiguous 16-byte aligned uint8 [num_slots, 1024] CUDA tensor");
    TORCH_CHECK(slotIds.is_cuda() && slotIds.scalar_type() == at::kLong && slotIds.numel() >= 1,
        "s2 rs: slot_ids must be an int64 CUDA tensor (request 0's slot first)");
    TORCH_CHECK(draftIn.is_cuda() && (draftIn.scalar_type() == at::kInt || draftIn.scalar_type() == at::kLong)
            && draftIn.is_contiguous() && draftIn.numel() >= rs::kPos,
        "s2 rs: draft_in must be a contiguous int32 / int64 CUDA tensor of at least 7 drafts");
    TORCH_CHECK(candAddr % 16 == 0 && digAddr % 16 == 0 && candAddr != 0 && digAddr != 0,
        "s2 rs: the candidate-slot and digest areas are not reserved (rejection_sampler) or misaligned");
    TORCH_CHECK(candBytes >= candAreaNeed(world) && digBytes >= digAreaNeed(world),
        "s2 rs: candidate-slot or digest area too small");
}

void checkRowRng(std::optional<at::Tensor> const& seed, std::optional<at::Tensor> const& offset, int64_t rows)
{
    TORCH_CHECK(seed.has_value() && offset.has_value(), "s2 rs: per-row seed and offset are required");
    checkRowParam(seed.value(), "seed", at::kLong, rows);
    checkRowParam(offset.value(), "offset", at::kLong, rows);
}

//! The RS fold (s2FoldPublishKernel<4, 256, 2, true>): fold_publish's arguments (row params mandatory) + the RS inputs.
void foldPublishRs(at::Tensor const& local, at::Tensor const& temperatures, at::Tensor const& counters, int64_t mcArea,
    int64_t areaBytes, int64_t rank, int64_t world, double multiplier, double invCap, double cap, int64_t vts,
    int64_t threads, int64_t form, bool pdl, std::optional<at::Tensor> const& ownStates,
    std::optional<at::Tensor> const& stamps, int64_t slot, at::Tensor const& topKs, at::Tensor const& topPs,
    at::Tensor const& minPs, int64_t mcRows, int64_t rowAreaBytes, at::Tensor const& seed, at::Tensor const& offset,
    at::Tensor const& records, int64_t numSlots, at::Tensor const& slotIds, at::Tensor const& draftIn, int64_t mcCand,
    int64_t candBytes, int64_t mcDig, int64_t digBytes, int64_t vocab, std::optional<at::Tensor> const& exportE,
    bool noExclude)
{
    at::cuda::CUDAGuard guard(local.device());
    TORCH_CHECK(vts == 4 && threads == 256 && form == 2, "s2 rs fold: shape 4 x 256, form 2");
    TORCH_CHECK(mcRows != 0, "s2 rs fold: the row area is required (filtered rows take the row-area path)");
    FoldParams const p = foldParams(local, temperatures, counters, mcArea, areaBytes, rank, world, multiplier, invCap,
        cap, vts, ownStates, stamps, slot, topKs, topPs, minPs, mcRows, rowAreaBytes);
    int64_t const rows = local.size(0);
    TORCH_CHECK(p.localThreads / vts >= rs::kLanes, "s2 rs fold: a designated CTA per candidate lane");
    TORCH_CHECK(vocab == static_cast<int64_t>(p.fs.vocabSize), "s2 rs fold: vocab must be world * shard");
    checkRsInputs(world, records, numSlots, slotIds, draftIn, mcCand, candBytes, mcDig, digBytes);
    checkRowRng(seed, offset, rows);
    RsFoldParams q;
    q.records = records.data_ptr<uint8_t>();
    q.numSlots = static_cast<int>(numSlots);
    q.slotIds = slotIds.data_ptr<int64_t>();
    q.draftIn = draftIn.data_ptr();
    q.draftInBytes = static_cast<int>(draftIn.element_size());
    q.seed = seed.data_ptr<int64_t>();
    q.offset = offset.data_ptr<int64_t>();
    q.mcCand = static_cast<uint64_t>(mcCand);
    q.candBytes = static_cast<uint64_t>(candBytes);
    q.mcDig = static_cast<uint64_t>(mcDig);
    q.digBytes = static_cast<uint64_t>(digBytes);
    q.vocab = static_cast<int>(vocab);
    if (exportE.has_value())
    {
        auto const& e = exportE.value();
        TORCH_CHECK(e.is_cuda() && e.scalar_type() == at::kFloat && e.is_contiguous() && e.numel() >= 2 * rows * vocab,
            "s2 rs fold: export_e must be a contiguous FP32 [2, rows, vocab] CUDA tensor");
        q.exportE = e.data_ptr<float>();
    }
    q.noExclude = noExclude ? 1 : 0;
    launchFoldShape<4, 256, 2, true>(p, q, static_cast<int>(rows), pdl);
}

//! The RS draw (s2RsDrawKernel) of `rows` rows on this rank: draw's arguments + the RS inputs; the int32 [rows] tokens.
at::Tensor rsDraw(at::Tensor const& local, at::Tensor const& scratch, at::Tensor const& temperatures,
    at::Tensor const& topKs, at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& counters, int64_t ucStates, int64_t areaBytes,
    int64_t ucSlot, int64_t mcSlot, int64_t slotBytes, int64_t ucRows, int64_t rowAreaBytes, int64_t rank,
    int64_t world, double multiplier, double invCap, double cap, int64_t timeoutNs, bool pdl, bool trigger,
    std::optional<at::Tensor> const& stamps, int64_t slot, at::Tensor const& records, int64_t numSlots,
    int64_t dummySlot, at::Tensor const& slotIds, at::Tensor const& draftIn, int64_t ucCand, int64_t candBytes,
    int64_t ucDig, int64_t digBytes, int64_t vocab, int64_t check, at::Tensor const& diag,
    std::optional<at::Tensor> const& ring, std::optional<at::Tensor> const& ringHead,
    std::optional<at::Tensor> const& forceUa, std::optional<at::Tensor> const& forceLogE,
    std::optional<at::Tensor> const& exports, std::optional<at::Tensor> const& listX,
    std::optional<at::Tensor> const& listId, int64_t filterMode, int64_t forceScreen,
    std::optional<at::Tensor> const& fexports, std::optional<at::Tensor> const& fstamps)
{
    at::cuda::CUDAGuard guard(local.device());
    at::Tensor tokens;
    DrawParams const p = drawParams(local, scratch, temperatures, topKs, topPs, minPs, seed, offset, counters, ucStates,
        areaBytes, ucSlot, mcSlot, slotBytes, ucRows, rowAreaBytes, rank, world, multiplier, invCap, cap, timeoutNs,
        stamps, slot, 0, tokens);
    int64_t const rows = local.size(0);
    // One request of K + 1 rows: the filtered rows' strict body is deterministic across ranks only up to 8 rows.
    TORCH_CHECK(rows == rs::kPos + 1, "s2 rs draw: one request of K + 1 = ", rs::kPos + 1, " rows, not ", rows);
    TORCH_CHECK(vocab == static_cast<int64_t>(p.fs.vocabSize), "s2 rs draw: vocab must be world * shard");
    checkRsInputs(world, records, numSlots, slotIds, draftIn, ucCand, candBytes, ucDig, digBytes);
    checkRowRng(seed, offset, rows);
    // diag may be pinned host memory (tests read the trap code after a trap).
    TORCH_CHECK((diag.is_cuda() || diag.is_pinned()) && diag.scalar_type() == at::kLong && diag.is_contiguous()
            && diag.numel() >= rs::kDiagWords,
        "s2 rs draw: diag must be a contiguous int64 [", rs::kDiagWords, "] CUDA (or pinned) tensor");
    RsDrawParams q;
    q.records = records.data_ptr<uint8_t>();
    q.numSlots = static_cast<int>(numSlots);
    q.dummySlot = static_cast<int>(dummySlot);
    q.slotIds = slotIds.data_ptr<int64_t>();
    q.draftIn = draftIn.data_ptr();
    q.draftInBytes = static_cast<int>(draftIn.element_size());
    q.ucCand = static_cast<uint64_t>(ucCand);
    q.candBytes = static_cast<uint64_t>(candBytes);
    q.ucDig = static_cast<uint64_t>(ucDig);
    q.digBytes = static_cast<uint64_t>(digBytes);
    q.vocab = static_cast<int>(vocab);
    q.check = static_cast<int>(check);
    q.diag = reinterpret_cast<unsigned long long*>(diag.data_ptr());
    TORCH_CHECK(ring.has_value() == ringHead.has_value(), "s2 rs draw: ring and ring_head go together");
    if (ring.has_value())
    {
        auto const& r = ring.value();
        auto const& h = ringHead.value();
        TORCH_CHECK(r.is_cuda() && r.scalar_type() == at::kFloat && r.is_contiguous() && r.dim() == 2 && r.size(1) == 4
                && r.size(0) >= 1 && h.is_cuda() && h.scalar_type() == at::kInt && h.numel() >= 1,
            "s2 rs draw: ring must be a contiguous FP32 [N, 4] and ring_head an int32 [1] CUDA tensor");
        q.ring = r.data_ptr<float>();
        q.ringHead = h.data_ptr<int32_t>();
        q.ringSize = static_cast<int>(r.size(0));
    }
    if (forceUa.has_value())
    {
        checkRowParam(forceUa.value(), "force_ua", at::kFloat, rows);
        q.forceUa = forceUa.value().data_ptr<float>();
    }
    if (forceLogE.has_value())
    {
        auto const& f = forceLogE.value();
        TORCH_CHECK(f.is_cuda() && f.scalar_type() == at::kFloat && f.is_contiguous() && f.numel() >= rows * rs::kLanes,
            "s2 rs draw: force_loge must be a contiguous FP32 [rows, 16] CUDA tensor");
        q.forceLogE = f.data_ptr<float>();
    }
    if (exports.has_value())
    {
        auto const& e = exports.value();
        TORCH_CHECK(
            e.is_cuda() && e.scalar_type() == at::kFloat && e.is_contiguous() && e.numel() >= rows * kExportWords,
            "s2 rs draw: exports must be a contiguous FP32 [rows, ", kExportWords, "] CUDA tensor");
        q.exports = e.data_ptr<float>();
    }
    TORCH_CHECK(filterMode == static_cast<int64_t>(rs::FilterMode::kExact)
            || filterMode == static_cast<int64_t>(rs::FilterMode::kStrict),
        "s2 rs draw: filter_mode is rs::FilterMode (0 exact, 1 strict)");
    TORCH_CHECK(forceScreen >= 0 && forceScreen <= 2, "s2 rs draw: force_screen 0, 1 or 2");
    q.filterMode = static_cast<int>(filterMode);
    q.forceScreen = static_cast<int>(forceScreen);
    if (filterMode == static_cast<int64_t>(rs::FilterMode::kExact))
    {
        // The owner of row r keeps its list in slot r / world; one list holds every element of a row.
        int64_t const need = rsListCap(vocab);
        TORCH_CHECK(listX.has_value() && listId.has_value(), "s2 rs draw: exact filtering needs list_x and list_id");
        auto const& lx = listX.value();
        auto const& li = listId.value();
        TORCH_CHECK(lx.is_cuda() && lx.scalar_type() == at::kFloat && lx.is_contiguous() && lx.dim() == 2
                && lx.size(0) >= (rows + world - 1) / world && lx.size(1) >= need && li.is_cuda()
                && li.scalar_type() == at::kInt && li.is_contiguous() && li.sizes() == lx.sizes(),
            "s2 rs draw: list_x (FP32) and list_id (int32) must be contiguous [>= ceil(rows / world), >= ", need,
            "] CUDA tensors of one shape");
        q.listX = lx.data_ptr<float>();
        q.listId = li.data_ptr<int32_t>();
        q.listCap = static_cast<int>(lx.size(1));
    }
    if (fexports.has_value())
    {
        auto const& e = fexports.value();
        TORCH_CHECK(
            e.is_cuda() && e.scalar_type() == at::kFloat && e.is_contiguous() && e.numel() >= rows * kFExportWords,
            "s2 rs draw: fexports must be a contiguous FP32 [rows, ", kFExportWords, "] CUDA tensor");
        q.fexports = e.data_ptr<float>();
    }
    if (fstamps.has_value())
    {
        checkStamps(fstamps, slot);
        q.fstamps = reinterpret_cast<unsigned long long*>(fstamps.value().data_ptr());
    }
    if (trigger)
    {
        launchEx(s2RsDrawKernel<true>, dim3(rows), dim3(kVirtualThreads), 0, pdl, p, q);
    }
    else
    {
        launchEx(s2RsDrawKernel<false>, dim3(rows), dim3(kVirtualThreads), 0, pdl, p, q);
    }
    return tokens;
}

//! Registers, local (spill) bytes, static shared bytes and max threads per block of `kernel`.
template <typename Kernel>
void kernelAttributes(int64_t* out, Kernel* kernel)
{
    cudaFuncAttributes attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
    out[0] = attr.numRegs;
    out[1] = static_cast<int64_t>(attr.localSizeBytes);
    out[2] = static_cast<int64_t>(attr.sharedSizeBytes);
    out[3] = attr.maxThreadsPerBlock;
}

//! Resolves the RS kernels (and the non-RS fold / draw for reference) and returns their attributes, int64 [5, 4] on
//! the CPU: rows RS fold <4, 256, 2>, RS draw <false>, RS draw <true>, non-RS fold <4, 256, 2>, non-RS draw <false>;
//! columns registers, local (spill) bytes, static shared bytes, max threads per block.
at::Tensor rsAttributes(at::Tensor const& anchor)
{
    at::cuda::CUDAGuard guard(anchor.device());
    auto out = at::empty({5, 4}, at::TensorOptions().dtype(at::kLong));
    int64_t* const o = out.data_ptr<int64_t>();
    kernelAttributes(o + 0, s2FoldPublishKernel<4, 256, 2, true>);
    kernelAttributes(o + 4, s2RsDrawKernel<false>);
    kernelAttributes(o + 8, s2RsDrawKernel<true>);
    kernelAttributes(o + 12, s2FoldPublishKernel<4, 256, 2, false>);
    kernelAttributes(o + 16, s2DrawKernel<false>);
    return out;
}

void stamp(at::Tensor const& stamps, int64_t index, bool pdl)
{
    at::cuda::CUDAGuard guard(stamps.device());
    TORCH_CHECK(stamps.is_cuda() && stamps.scalar_type() == at::kLong && stamps.is_contiguous() && index >= 0
            && index < stamps.numel(),
        "s2 stamp: contiguous int64 CUDA stamps and an index inside them");
    launchEx(s2StampKernel, dim3(1), dim3(1), 0, pdl, reinterpret_cast<unsigned long long*>(stamps.data_ptr()),
        static_cast<int>(index));
}

void dry(at::Tensor const& anchor)
{
    at::cuda::CUDAGuard guard(anchor.device());
    launchEx(s2DryKernel, dim3(1), dim3(1), 0, false);
}

} // namespace

TORCH_LIBRARY(musespark_open_s2_sampler, m)
{
    m.def(
        "fold_publish(Tensor local, Tensor temperatures, Tensor(a!) counters, int mc_area, int area_bytes, int rank, "
        "int world, float multiplier, float inv_cap, float cap, int vts, int threads, int form, bool pdl, "
        "Tensor(b!)? own_states, Tensor(c!)? stamps, int slot, Tensor? top_ks=None, Tensor? top_ps=None, "
        "Tensor? min_ps=None, int mc_rows=0, int row_area_bytes=0) -> ()");
    m.def(
        "poll_states(Tensor(a!) states, Tensor(b!) counters, int uc_area, int area_bytes, int rows, int world, "
        "int timeout_ns, bool pdl, Tensor(c!)? stamps, int slot) -> ()");
    m.def(
        "draw(Tensor local, Tensor(a!) scratch, Tensor temperatures, Tensor top_ks, Tensor top_ps, Tensor min_ps, "
        "Tensor? seed, Tensor? offset, Tensor(b!) counters, int uc_states, int area_bytes, int uc_slot, int mc_slot, "
        "int slot_bytes, int uc_rows, int row_area_bytes, int rank, int world, float multiplier, float inv_cap, "
        "float cap, int timeout_ns, bool pdl, bool trigger, Tensor(c!)? stamps, int slot, int force_multi=0) "
        "-> Tensor");
    m.def(
        "fold_publish_rs(Tensor local, Tensor temperatures, Tensor(a!) counters, int mc_area, int area_bytes, "
        "int rank, int world, float multiplier, float inv_cap, float cap, int vts, int threads, int form, bool pdl, "
        "Tensor(b!)? own_states, Tensor(c!)? stamps, int slot, Tensor top_ks, Tensor top_ps, Tensor min_ps, "
        "int mc_rows, int row_area_bytes, Tensor seed, Tensor offset, Tensor records, int num_slots, "
        "Tensor slot_ids, Tensor draft_in, int mc_cand, int cand_bytes, int mc_dig, int dig_bytes, int vocab, "
        "Tensor(d!)? export_e=None, bool no_exclude=False) -> ()");
    m.def(
        "rs_draw(Tensor local, Tensor(a!) scratch, Tensor temperatures, Tensor top_ks, Tensor top_ps, "
        "Tensor min_ps, Tensor? seed, Tensor? offset, Tensor(b!) counters, int uc_states, int area_bytes, "
        "int uc_slot, int mc_slot, int slot_bytes, int uc_rows, int row_area_bytes, int rank, int world, "
        "float multiplier, float inv_cap, float cap, int timeout_ns, bool pdl, bool trigger, Tensor(c!)? stamps, "
        "int slot, Tensor(d!) records, int num_slots, int dummy_slot, Tensor slot_ids, Tensor draft_in, "
        "int uc_cand, int cand_bytes, int uc_dig, int dig_bytes, int vocab, int check, Tensor(e!) diag, "
        "Tensor(f!)? ring=None, Tensor(g!)? ring_head=None, Tensor? force_ua=None, Tensor? force_loge=None, "
        "Tensor(h!)? exports=None, Tensor(i!)? list_x=None, Tensor(j!)? list_id=None, int filter_mode=0, "
        "int force_screen=0, Tensor(k!)? fexports=None, Tensor(l!)? fstamps=None) -> Tensor");
    m.def("rs_attributes(Tensor anchor) -> Tensor");
    m.def("stamp(Tensor(a!) stamps, int index, bool pdl) -> ()");
    m.def("dry(Tensor anchor) -> ()");
}

TORCH_LIBRARY_IMPL(musespark_open_s2_sampler, CUDA, m)
{
    m.impl("fold_publish", &foldPublish);
    m.impl("poll_states", &pollStates);
    m.impl("draw", &draw);
    m.impl("fold_publish_rs", &foldPublishRs);
    m.impl("rs_draw", &rsDraw);
    m.impl("rs_attributes", &rsAttributes);
    m.impl("stamp", &stamp);
    m.impl("dry", &dry);
}
