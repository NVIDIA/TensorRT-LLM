/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The DFlash 2 draft head's tail after the vocab-sharded LM-head GEMV in three launches (``tail.py``).
 *
 * The built-in chain after the draft LM head is ``draft_shard_topk`` (Triton) -> the MNNVL one-shot FP32 all-reduce of
 * the packed keys -> ``draft_topk_merge`` (Triton) -> the TF32 einsum (cuBLAS, unchanged here) ->
 * ``draft_path_walk_tail`` (Triton). The kernels below replace the first three and the last one, statement for
 * statement where a value is produced:
 *
 * * ``drSelectPublishKernel`` (one CTA per (row, 4096-column chunk)): the shard keys of ``draft_shard_topk`` --
 *   ``key = ((key16 ^ 0x8000) << 16) | (0xFFFF - column)``, torch's radix key of the BF16 bits, NaN at the top,
 *   ``INT32_MIN`` for a column past the shard -- and the top 16 of each chunk, largest first (signed int32 order,
 *   every key unique). The selection is a sorting network per thread and bitonic merges of sorted top-16 lists
 *   across lanes and warps; it selects the same set in the same order as 16 rounds of ``tl.max``. The 16 keys are
 *   then published to every rank (one-shot multicast, below) instead of being all-reduced as FP32 halves.
 * * ``drMergeKernel`` (one CTA per block position): polls the row's keys of every rank, rebuilds
 *   ``draft_topk_merge``'s 64-bit candidates ``(key16 << 32) | (0xFFFFFFFF - vocab id)``, selects the top 16
 *   (bitonic merges of the ranks' sorted chunk lists), arranges them as ``torch.topk``'s ``gatherTopK`` does
 *   (above the k-th value first, then equal to it, each by ascending id) and runs torch's ``bitonicSortKVInPlace``
 *   network on 32 slots, one slot per lane, exactly as the Triton merge writes it. Then the selector's operands:
 *   ``successor_codebook[ids]`` widened to FP32 and ``predecessor_codebook[prev] * gate`` (one FP32 multiply of two
 *   widened BF16 values, exact). Integer selection and exact products: the outputs equal the Triton merge's bit for
 *   bit.
 * * ``drWalkTailKernel`` (one warp per request): ``draft_path_walk_tail``'s walk -- per position one FP32 add
 *   ``unary + edge``, NaN count, maximum, hits, lowest hit position, lowest hit id, the ``-inf`` rule -- with ballots
 *   and shuffles instead of Triton reductions, then its next-token row and KV-length rewind.
 *
 * Transport (key exchange): LL pairs over the NVLink multicast fabric, no reset -- the protocol of
 * ``kernels/pv_logits/pv_logits_gather.cu``. Each 32-bit key travels as one 64-bit element ``(epoch << 32) |
 * key``, published with 16-byte multicast stores at ``row * world * keysPerRank + rank * keysPerRank + chunk * 16 +
 * j`` of the area of the call's epoch parity; a reader polls its own copy with relaxed system-scope 16-byte loads
 * until both elements of a vector carry the call's epoch. A 64-bit element is written and read as one access, so a
 * matching epoch half implies the key half of the same call.
 *
 * Epochs: every CTA of both kernels keeps its own counter (``counters``, read and written only by that CTA of that
 * kernel) and uses ``counter + 1``, storing it back at its end. Grids are fixed, every call runs every CTA of both
 * kernels, and every rank runs the same sequence of calls, so all CTAs of all ranks agree on a call's epoch. The
 * areas start zeroed (epoch 0 is never used); a stale element carries an older epoch. Two areas alternate by epoch
 * parity: a peer writes the area this rank reads in call ``e`` next in its call ``e + 2``, whose publish follows
 * that peer's call-``e + 1`` merge, which polled this rank's call-``e + 1`` publish, which this rank's stream orders
 * after its call-``e`` merge completed.
 *
 * Variant 2 (``variant = 2``; variant 1 is the above): the walk's two reductions by other instructions, the same
 * values -- the maximum as ``__reduce_max_sync`` over an order-preserving 32-bit image of the scores (non-NaN floats in
 * IEEE order, ``-0`` below ``+0``: only ``score == best`` and ``best == -inf`` read it, and both are blind to the sign
 * of a zero), the lowest hit id as ``__reduce_min_sync`` over the ids as unsigned 32-bit (every candidate id is below
 * 2^31; a non-hit lane offers ``0xFFFFFFFF``, and every step has a hit); and the tail's loads (batch index, accepted
 * count, the accepted token, KV length, rewind) issued before the walk's wait -- their producers precede the select
 * launch, which passed its own wait before any later kernel of the chain could start; the stores stay after the wait.
 * The merge is the same in all variants. Variant 3: variant 2 with the walk's edge rows staged through registers (every
 * lane's 16-byte loads issued together, then the shared-memory stores) instead of a load-store loop.
 *
 * PDL: the select kernel waits before its first global access and releases its dependent (the merge) after its
 * publish. The merge reads only its own counter, the gate, the anchor id and the codebooks before its wait (each
 * final before the select kernel passed its own wait: the gate's producer joins the stream before the select
 * launch), polls, then waits for the select grid before any global store. The walk waits before every access.
 *
 * Rejection sampling (``walk_tail_rs``: ``drWalkTailKernel<3, 1>`` for kMode 1, ``drWalkSampledKernel`` for kMode 2;
 * kMode 0 is the walk above): the draft at position t is drawn from q_t, the softmax over the position's 16 candidates
 * of ``z = R * tau`` (R the walk score ``unary + edge``, tau = 1 / T of the request's row 0; option ``score``: f(R),
 * option ``unary``: the walk runs on f(unary), f(x) = cap tanh(x m / cap) in capOf's statement order, or x m when cap
 * = 0). One Philox word U_t per position (``rs::draftSubseq``, counter 0, word x) picks lane k_t = min{k : U_t < T_k},
 * where T_k = floor(C_k / S * 2^32) over the sequential FP32 prefix sums C_k of e_k = exp(z_k - max z) and T_15 =
 * 2^32: P(k_t = k) = n_k / 2^32 exactly with n_k = T_k - T_{k-1}, and the record stores qf_k = fl(n_k) 2^-32. kMode 1
 * keeps the walk's greedy path (the predecessor of position t + 1 is the greedy lane): with the ``raw`` or ``score``
 * transform the path, the next-token row's first entry, the rewind and every candidate row are kMode 0's bit for bit;
 * with ``unary`` the walk scores are f(unary) + edge and its greedy path may differ from kMode 0's. kMode 1 computes
 * position t's softmax inside the walk's step t; the 7 threshold scans then run once, lane t < 7 scanning position t
 * (lanes 7..31 repeat position 6 and drop it). kMode 2 (the default chain) follows the sampled lane instead. A position
 * whose scores or T are not finite is poisoned: its draft is the greedy token and q is one-hot at that token's lane
 * (all zero if the token is not a candidate). The q record of the request's slot (``rs_common.cuh``) receives the 16
 * candidate ids and qf of every position, the drafts, the position flags, the mode, then kind = kSampled and valid =
 * kValidAll. The slot id, the request's row-0 seed, offset and temperature are read before the wait: they are staged by
 * host-to-device copies before the graph launch, the class of operands variant 2 already reads there; ``check``
 * re-reads them after the wait and traps on a difference. Every record store follows the wait.
 *
 * kMode 2 computes every row the chain can visit instead of the 7 visited rows in series: rows (t, pi) for t = 1..6 and
 * each predecessor lane pi (96 rows, one per lane, 3 rolled rounds) form a table of the next lane and of x_k =
 * fl(C_k fl(1 / S)), k < 15; row (0, 0) is computed by the warp (lane k holds candidate k), and the chain is then 6
 * table lookups. Every value is produced by the statements above in their order; only where it is computed changes:
 * the maximum is an ``fmaxf`` tree (it differs from the ordered maximum only in the sign of a zero maximum, which no
 * e_k sees); the drawn lane is k_t = #{k < 15 : x_k < c_u} with c_u = RU(U + 1) 2^-32 (x 2^32 is exact and monotone,
 * and RU(U + 1) is the smallest float >= U + 1, so x_k < c_u <=> floor(fl(x_k 2^32)) <= U); a position is poisoned iff
 * S is not in (0, inf) (a NaN score, tau or max z makes some e_k NaN, and otherwise S is in [1, 16]); n_k is converted
 * through 32 bits, n_k = 2^32 giving qf = 1. Poisoned table rows get their greedy lane in a separate pass, and the
 * greedy tokens of poisoned positions (and the test exports) are computed only when needed, after the chain.
 *
 * kMode 2 draws from the request's filtered law when the request has top-k (0 < k < 16), top-p (< 1) or min-p (> 0)
 * among the filters the deploy's sampling mode applies (``filterMask``) and the filter mode is exact: the stock order
 * min-p, top-k (ties at the k-th value kept), top-p over the survivors' mass (the mass strictly above a lane at most
 * fl(topP S1); the maximum always kept) on the row's 16 e_k, the removed lanes' e_k set to 0 before the prefix sums
 * (``rsFilterRow``). Such a request does not use the table: its 7 visited rows are computed in series in the warp form.
 * A position is poisoned iff a pre-filter e_k is NaN or S is not in (0, inf). A lane k < 15 whose prefix sum already
 * equals S gets x_k = 1, so T_k = 2^32: the lanes after it (lane 15 included) get n = 0 and qf = 0 exactly. Filtered
 * positions carry kPosFiltered; the request's top-k / top-p / min-p are read before the wait with the temperature.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <optional>

#include "rs_common.cuh"

namespace
{

//! Candidates kept per chunk and per block position (``selector_top_k``).
constexpr int kTopK = 16;
//! Columns per select CTA (``draft_shard_topk``'s ``CHUNK``).
constexpr int kChunk = 4096;
constexpr int kSelectThreads = 256;
constexpr int kSelectWarps = kSelectThreads / 32;
constexpr int kColumnsPerThread = kChunk / kSelectThreads;
static_assert(kColumnsPerThread == 16, "the select kernel loads two 16-byte vectors per thread");
//! Selector rank (codebook columns): one merge thread per column.
constexpr int kRankDim = 256;
constexpr int kMergeThreads = kRankDim;
constexpr int kMaxWorld = 8;
//! A shard of at most 65536 columns has at most 16 chunks.
constexpr int kMaxKeysPerRank = 16 * kTopK;
constexpr int kMaxPositions = 8;
//! ``torch.topk``'s bitonic slots for k <= 32.
constexpr int kSlots = 32;
//! Area vectors (two 64-bit elements) one merge thread polls at most.
constexpr int kPollPerThread = kMaxWorld * kMaxKeysPerRank / 2 / kMergeThreads;
constexpr int32_t kEmptyKey = INT32_MIN;
constexpr unsigned kFullMask = 0xFFFFFFFFu;

__device__ __forceinline__ void gridDependencyWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void launchDependents()
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

__device__ __forceinline__ void pollPair(uint64_t address, unsigned long long& a, unsigned long long& b)
{
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(address) : "memory");
}

__device__ __forceinline__ void publishPair(uint64_t address, unsigned long long a, unsigned long long b)
{
    asm volatile("st.global.v2.b64 [%0], {%1, %2};" ::"l"(address), "l"(a), "l"(b) : "memory");
}

__device__ __forceinline__ float bf16Bits(uint16_t bits)
{
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

// ---- sorted top-16 lists, largest first (a sorting network and bitonic merges) ----

template <typename T>
__device__ __forceinline__ void orderDescending(T& a, T& b)
{
    T const hi = a > b ? a : b;
    T const lo = a > b ? b : a;
    a = hi;
    b = lo;
}

//! Sort 16 values largest first (bitonic sorting network).
template <typename T>
__device__ __forceinline__ void sort16(T (&v)[kTopK])
{
#pragma unroll
    for (int k = 2; k <= kTopK; k <<= 1)
    {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1)
        {
#pragma unroll
            for (int i = 0; i < kTopK; ++i)
            {
                int const l = i ^ j;
                if (l > i)
                {
                    if ((i & k) == 0)
                    {
                        orderDescending(v[i], v[l]);
                    }
                    else
                    {
                        orderDescending(v[l], v[i]);
                    }
                }
            }
        }
    }
}

//! ``v`` := the 16 largest of ``v`` and ``p`` (both sorted largest first), sorted largest first.
template <typename T>
__device__ __forceinline__ void mergeTop16(T (&v)[kTopK], T const (&p)[kTopK])
{
#pragma unroll
    for (int i = 0; i < kTopK; ++i)
    {
        T const q = p[kTopK - 1 - i];
        v[i] = v[i] > q ? v[i] : q;
    }
    // ``v`` is now bitonic and holds the 16 largest of the union; the half cleaners sort it.
#pragma unroll
    for (int j = kTopK >> 1; j > 0; j >>= 1)
    {
#pragma unroll
        for (int i = 0; i < kTopK; ++i)
        {
            int const l = i ^ j;
            if (l > i)
            {
                orderDescending(v[i], v[l]);
            }
        }
    }
}

//! Merge every lane's list with its partners' over the lanes ``1 .. width - 1`` apart: the group's top 16.
template <typename T>
__device__ __forceinline__ void mergeAcrossLanes(T (&v)[kTopK], int width)
{
    for (int m = 1; m < width; m <<= 1)
    {
        T p[kTopK];
#pragma unroll
        for (int i = 0; i < kTopK; ++i)
        {
            p[i] = __shfl_xor_sync(kFullMask, v[i], m);
        }
        mergeTop16(v, p);
    }
}

// ---- select + publish ----

struct SelectParams
{
    uint16_t const* logits; // [rows][rowStride] BF16 bits
    int64_t rowStride;      // elements
    int shard;
    int chunks;
    uint64_t mcArea;    // multicast address of the two areas
    uint64_t areaBytes; // bytes of one area
    uint32_t* counters; // [rows * chunks]
    int rank;
    int world;
    int keysPerRank;
};

__global__ void __launch_bounds__(kSelectThreads, 1) drSelectPublishKernel(SelectParams const p)
{
    __shared__ int32_t sWarpTop[kSelectWarps][kTopK];
    __shared__ int32_t sTop[kTopK];
    __shared__ uint32_t sEpoch;
    gridDependencyWait();
    int const row = blockIdx.x / p.chunks;
    int const chunk = blockIdx.x - row * p.chunks;
    int const lane = threadIdx.x % 32;
    int const warp = threadIdx.x / 32;
    if (threadIdx.x == 0)
    {
        sEpoch = p.counters[blockIdx.x] + 1u;
    }

    // This thread's 16 columns (two 16-byte vectors; the shard is a whole number of 8-column vectors).
    int const local0 = chunk * kChunk + threadIdx.x * kColumnsPerThread;
    uint16_t const* src = p.logits + static_cast<int64_t>(row) * p.rowStride + local0;
    bool const ok0 = local0 < p.shard;
    bool const ok1 = local0 + 8 < p.shard;
    uint4 v0 = make_uint4(0, 0, 0, 0);
    uint4 v1 = make_uint4(0, 0, 0, 0);
    if (ok0)
    {
        v0 = *reinterpret_cast<uint4 const*>(src);
    }
    if (ok1)
    {
        v1 = *reinterpret_cast<uint4 const*>(src + 8);
    }
    uint32_t const words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    int32_t key[kTopK];
#pragma unroll
    for (int e = 0; e < kTopK; ++e)
    {
        uint32_t const bits = (words[e / 2] >> (16 * (e % 2))) & 0xFFFFu;
        uint32_t key16 = (bits & 0x8000u) != 0 ? (bits ^ 0xFFFFu) : (bits ^ 0x8000u);
        bool const isNan = (bits & 0x7F80u) == 0x7F80u && (bits & 0x7Fu) != 0;
        key16 = isNan ? 0xFFFFu : key16;
        uint32_t const local = static_cast<uint32_t>(local0 + e);
        uint32_t const packed = ((key16 ^ 0x8000u) << 16) | (0xFFFFu - local);
        key[e] = (e < 8 ? ok0 : ok1) ? static_cast<int32_t>(packed) : kEmptyKey;
    }
    sort16(key);
    mergeAcrossLanes(key, 32);
    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < kTopK; ++i)
        {
            sWarpTop[warp][i] = key[i];
        }
    }
    __syncthreads();
    if (warp == 0)
    {
#pragma unroll
        for (int i = 0; i < kTopK; ++i)
        {
            key[i] = lane < kSelectWarps ? sWarpTop[lane][i] : kEmptyKey;
        }
        mergeAcrossLanes(key, kSelectWarps);
        if (lane == 0)
        {
#pragma unroll
            for (int i = 0; i < kTopK; ++i)
            {
                sTop[i] = key[i];
            }
        }
        __syncwarp();
        if (lane < kTopK / 2)
        {
            uint32_t const epoch = sEpoch;
            unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
            uint64_t const element = (static_cast<uint64_t>(row) * p.world + p.rank) * p.keysPerRank
                + static_cast<uint64_t>(chunk) * kTopK + 2 * lane;
            uint64_t const address = p.mcArea + static_cast<uint64_t>(epoch & 1u) * p.areaBytes + element * 8u;
            publishPair(
                address, tag | static_cast<uint32_t>(sTop[2 * lane]), tag | static_cast<uint32_t>(sTop[2 * lane + 1]));
        }
    }
    launchDependents();
    if (threadIdx.x == 0)
    {
        p.counters[blockIdx.x] = sEpoch;
    }
}

// ---- merge ----

struct MergeParams
{
    uint64_t ucArea;    // this rank's copy of the two areas
    uint64_t areaBytes;
    uint32_t* counters; // [positions]
    int world;
    int shard;
    int chunks;
    int keysPerRank;
    int positions;
    uint16_t const* gate; // [positions][gateStride] BF16 bits (one request)
    int64_t gateStride;
    void const* anchor;   // [1] int32 or int64
    int anchorIs64;
    uint16_t const* predCodebook;
    int64_t predStride;
    uint16_t const* succCodebook;
    int64_t succStride;
    int64_t* ids; // [positions][16]
    float* unary; // [positions][16]
    float* preds; // [positions][16][256]
    float* succs; // [positions][16][256]
    unsigned long long timeoutNs;
};

//! An order-preserving unsigned image of a non-NaN float (``-0`` maps below ``+0``).
__device__ __forceinline__ uint32_t orderKey(float v)
{
    uint32_t const u = __float_as_uint(v);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float orderValue(uint32_t k)
{
    return __uint_as_float((k & 0x80000000u) != 0 ? (k & 0x7FFFFFFFu) : ~k);
}

//! fp32 value of a 16-bit radix key (``draft_topk_merge``'s ``_decode_value``).
__device__ __forceinline__ float decodeKey16(long long key)
{
    uint32_t const k = static_cast<uint32_t>(key) & 0xFFFFu;
    uint32_t const bits = (k & 0x8000u) != 0 ? (k ^ 0x8000u) : (k ^ 0xFFFFu);
    return __uint_as_float(bits << 16);
}

//! One ``bitonicSwap`` round of torch's ``bitonicSort`` over 32 slots, slot ``lane`` in ``s`` (``_bitonic_step``).
template <int kStride, int kSize, bool kFinal>
__device__ __forceinline__ long long bitonicStep(long long s, int lane)
{
    long long const other = __shfl_xor_sync(kFullMask, s, kStride);
    bool const lowSide = (lane & kStride) == 0;
    long long const a = lowSide ? s : other;
    long long const b = lowSide ? other : s;
    int const pos = lane & ~kStride;
    bool const flag = kFinal ? false : ((pos & kSize) != 0);
    float const fa = decodeKey16(a);
    float const fb = decodeKey16(b);
    bool const greater = ((fa != fa) && (fb == fb)) || (fa > fb);
    bool const swap = (greater && a >= 0) || (b < 0);
    bool const exchange = swap == flag;
    long long const low = exchange ? b : a;
    long long const high = exchange ? a : b;
    return lowSide ? low : high;
}

__global__ void __launch_bounds__(kMergeThreads, 1) drMergeKernel(MergeParams const p)
{
    __shared__ int32_t sKeys[kMaxWorld * kMaxKeysPerRank];
    __shared__ long long sChosen[kTopK];
    __shared__ long long sSlot[kSlots];
    __shared__ long long sIds[kTopK];
    __shared__ uint32_t sEpoch;
    int const t = blockIdx.x;
    int const d = threadIdx.x;
    int const lane = d % 32;
    int const warp = d / 32;
    if (d == 0)
    {
        sEpoch = p.counters[t] + 1u;
    }
    // Inputs that do not depend on the keys (final before the select kernel passed its wait).
    bool const hasNext = t + 1 < p.positions;
    float const gateNext = hasNext ? bf16Bits(p.gate[static_cast<int64_t>(t + 1) * p.gateStride + d]) : 0.0f;
    float gate0 = 0.0f;
    float pred0 = 0.0f;
    if (t == 0)
    {
        long long const anchor = p.anchorIs64 ? static_cast<long long const*>(p.anchor)[0]
                                              : static_cast<long long>(static_cast<int32_t const*>(p.anchor)[0]);
        gate0 = bf16Bits(p.gate[d]);
        pred0 = bf16Bits(p.predCodebook[anchor * p.predStride + d]);
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;

    // Poll this row's keys of every rank: all vectors in flight together, then the stale ones again.
    int const elements = p.world * p.keysPerRank;
    int const vectors = elements / 2;
    uint64_t const rowBase = p.ucArea + static_cast<uint64_t>(epoch & 1u) * p.areaBytes
        + static_cast<uint64_t>(t) * static_cast<uint64_t>(elements) * 8u;
    unsigned long long lo[kPollPerThread];
    unsigned long long hi[kPollPerThread];
    uint32_t stale = 0;
#pragma unroll
    for (int k = 0; k < kPollPerThread; ++k)
    {
        int const v = d + k * kMergeThreads;
        lo[k] = 0;
        hi[k] = 0;
        if (v < vectors)
        {
            pollPair(rowBase + static_cast<uint64_t>(v) * 16u, lo[k], hi[k]);
            bool const old = static_cast<uint32_t>(lo[k] >> 32) != epoch || static_cast<uint32_t>(hi[k] >> 32) != epoch;
            stale |= old ? (1u << k) : 0u;
        }
    }
    if (stale != 0)
    {
        unsigned long long const start = globalTimer();
        while (stale != 0)
        {
            uint32_t const polled = stale;
            stale = 0;
#pragma unroll
            for (int k = 0; k < kPollPerThread; ++k)
            {
                if ((polled >> k) & 1u)
                {
                    int const v = d + k * kMergeThreads;
                    pollPair(rowBase + static_cast<uint64_t>(v) * 16u, lo[k], hi[k]);
                    bool const old
                        = static_cast<uint32_t>(lo[k] >> 32) != epoch || static_cast<uint32_t>(hi[k] >> 32) != epoch;
                    stale |= old ? (1u << k) : 0u;
                }
            }
            if (stale != 0 && globalTimer() - start > p.timeoutNs)
            {
                __trap();
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kPollPerThread; ++k)
    {
        int const v = d + k * kMergeThreads;
        if (v < vectors)
        {
            sKeys[2 * v] = static_cast<int32_t>(static_cast<uint32_t>(lo[k]));
            sKeys[2 * v + 1] = static_cast<int32_t>(static_cast<uint32_t>(hi[k]));
        }
    }
    // The select grid has completed before any store of this kernel.
    gridDependencyWait();
    __syncthreads();

    if (warp == 0)
    {
        // Selection over the ranks' chunk lists (each sorted largest first in the 64-bit order too):
        // list L = rank * chunks + chunk; lane l folds lists l, l + 32, ..., then the lanes merge.
        int const lists = p.world * p.chunks;
        long long v[kTopK];
#pragma unroll
        for (int i = 0; i < kTopK; ++i)
        {
            v[i] = -1;
        }
        for (int list = lane; list < lists; list += 32)
        {
            int const r = list / p.chunks;
            int const c = list - r * p.chunks;
            long long q[kTopK];
#pragma unroll
            for (int i = 0; i < kTopK; ++i)
            {
                int32_t const key = sKeys[r * p.keysPerRank + c * kTopK + i];
                long long const key16 = static_cast<long long>(((key >> 16) & 0xFFFF) ^ 0x8000);
                long long const vocab = static_cast<long long>(r) * p.shard + (0xFFFF - (key & 0xFFFF));
                q[i] = key != kEmptyKey ? ((key16 << 32) | (0xFFFFFFFFLL - vocab)) : -1;
            }
            mergeTop16(v, q);
        }
        mergeAcrossLanes(v, 32);
        if (lane == 0)
        {
#pragma unroll
            for (int i = 0; i < kTopK; ++i)
            {
                sChosen[i] = v[i];
            }
        }
        __syncwarp();
        // Arrangement: above the k-th value by ascending id, then equal to it by ascending id.
        if (lane < kTopK)
        {
            long long const kth16 = sChosen[kTopK - 1] >> 32;
            long long const chosen = sChosen[lane];
            long long const chosen16 = chosen >> 32;
            long long const chosenId = 0xFFFFFFFFLL - (chosen & 0xFFFFFFFFLL);
            long long const order = (static_cast<long long>(chosen16 == kth16) << 40) | chosenId;
            int position = 0;
#pragma unroll
            for (int i = 0; i < kTopK; ++i)
            {
                long long const other = sChosen[i];
                long long const otherOrder
                    = (static_cast<long long>((other >> 32) == kth16) << 40) | (0xFFFFFFFFLL - (other & 0xFFFFFFFFLL));
                position += otherOrder < order ? 1 : 0;
            }
            sSlot[position] = (chosenId << 16) | chosen16;
        }
        else
        {
            sSlot[lane] = -1;
        }
        __syncwarp();
        long long s = sSlot[lane];
        // torch's bitonicSort<32>: sizes 2..16 with the thread-index direction, then the final merge.
        s = bitonicStep<1, 2, false>(s, lane);
        s = bitonicStep<2, 4, false>(s, lane);
        s = bitonicStep<1, 4, false>(s, lane);
        s = bitonicStep<4, 8, false>(s, lane);
        s = bitonicStep<2, 8, false>(s, lane);
        s = bitonicStep<1, 8, false>(s, lane);
        s = bitonicStep<8, 16, false>(s, lane);
        s = bitonicStep<4, 16, false>(s, lane);
        s = bitonicStep<2, 16, false>(s, lane);
        s = bitonicStep<1, 16, false>(s, lane);
        s = bitonicStep<16, 0, true>(s, lane);
        s = bitonicStep<8, 0, true>(s, lane);
        s = bitonicStep<4, 0, true>(s, lane);
        s = bitonicStep<2, 0, true>(s, lane);
        s = bitonicStep<1, 0, true>(s, lane);
        if (lane < kTopK)
        {
            long long const id = s >> 16;
            sIds[lane] = id;
            p.ids[t * kTopK + lane] = id;
            p.unary[t * kTopK + lane] = decodeKey16(s);
        }
    }
    __syncthreads();

    // Selector operands: this row's successors, the next row's predecessors (and row 0's from the anchor).
    uint16_t succRaw[kTopK];
    uint16_t predRaw[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        long long const id = sIds[k];
        succRaw[k] = p.succCodebook[id * p.succStride + d];
        predRaw[k] = hasNext ? p.predCodebook[id * p.predStride + d] : static_cast<uint16_t>(0);
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        int64_t const outRow = static_cast<int64_t>(t * kTopK + k) * kRankDim;
        p.succs[outRow + d] = bf16Bits(succRaw[k]);
        if (hasNext)
        {
            p.preds[outRow + kTopK * kRankDim + d] = __fmul_rn(bf16Bits(predRaw[k]), gateNext);
        }
        if (t == 0)
        {
            p.preds[outRow + d] = __fmul_rn(pred0, gate0);
        }
    }
    if (d == 0)
    {
        p.counters[t] = epoch;
    }
}

// ---- walk + tail ----

struct WalkParams
{
    float const* edges;      // [B][K][16][16]
    float const* unary;      // [B][K][16]
    long long const* ids;    // [B][K][16]
    int positions;           // K
    int32_t* tokens;         // [B][K]
    int32_t* nextTokens;     // [B][K + 1]
    int32_t const* accepted; // rows [K + 1], row stride acceptedStride
    int64_t acceptedStride;
    int32_t const* numAccepted;
    int32_t const* batchIndices;
    int32_t* kvLens;
    int32_t const* rewind;
    int kvRewind;
};

//! The rejection-sampling operands of the walk (kMode 1 / 2; unused by kMode 0).
struct RsWalkParams
{
    uint8_t* records; // [numSlots][rs::kRecordBytes] q records
    int numSlots;
    int dummySlot;
    int64_t const* slotIds; // request b's slot at [b]
    int64_t const* seeds;   // per token: request b's row 0 at [b * (K + 1)]
    int64_t const* offsets;
    float const* temperatures;
    int transform; // rs::Transform
    float mult;
    float invCap;
    float cap;
    int check;
    unsigned long long* diag; // [rs::kDiagWords]
    float* exportE;           // tests only (nullptr served): [B][7][16] e_k
    long long* exportU; // tests only: [B][7] U_t | path lane << 32 | greedy lane << 40 | k_t << 48 | poisoned << 56
    // kMode 2's draft-side filter (unused by kMode 0 / 1): per token, request b's row 0 at b * (K + 1); nullptr when
    // filterMask is 0.
    int32_t const* topKs;
    float const* topPs;
    float const* minPs;
    int filterMode; // rs::FilterMode
    int filterMask; // the filters the deploy's sampling mode applies: bit 0 top-k, bit 1 top-p, bit 2 min-p
};

//! The q transform f, capOf's statements (cap == 0: the scale alone).
__device__ __forceinline__ float rsTransform(float x, RsWalkParams const& rs)
{
    float y = __fmul_rn(x, rs.mult);
    if (rs.cap == 0.0f)
    {
        return y;
    }
    y = __fmul_rn(y, rs.invCap);
    y = tanhf(y);
    return __fmul_rn(rs.cap, y);
}

//! One position's softmax numerators e_k = exp(z_k - max z), z = s * tau, into eRow[lane] for the 16 candidate lanes;
//! whether the position is poisoned (a NaN score, tau not finite, max z not finite). Lane-uniform result; every lane of
//! the warp calls it.
__device__ __forceinline__ bool rsSoftmaxRow(
    float s, bool valid, float tau, RsWalkParams const& rs, float* eRow, int lane)
{
    if (rs.transform == static_cast<int>(rs::Transform::kScore))
    {
        s = rsTransform(s, rs);
    }
    bool const isNumber = s == s;
    float const z = (valid && isNumber) ? __fmul_rn(s, tau) : -INFINITY;
    bool bad = __ballot_sync(kFullMask, valid && !isNumber) != 0 || !(tau == tau);
    float const zmax = orderValue(__reduce_max_sync(kFullMask, orderKey(z)));
    bad = bad || !(zmax > -INFINITY && zmax < INFINITY);
    if (valid)
    {
        eRow[lane] = __expf(__fsub_rn(z, zmax));
    }
    return bad;
}

//! One position's integer thresholds over eRow: T_k = min(floor(C_k (1 / C_15) 2^32), 2^32) (T_15 = 2^32) from the
//! sequential FP32 prefix sums C_k; the drawn lane min{k : u < T_k} (``greedyLane`` when the position is or becomes
//! poisoned, in ``bad``); the lane with ``owner`` stores qRow[k] = fl(T_k - T_{k-1}) 2^-32.
__device__ __forceinline__ int rsThresholdRow(
    float const* eRow, bool& bad, uint32_t u, bool owner, float* qRow, int greedyLane)
{
    float prefix[kTopK];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        sum = __fadd_rn(sum, eRow[k]);
        prefix[k] = sum;
    }
    bad = bad || !(sum > 0.0f && sum < INFINITY);
    float const inv = __frcp_rn(sum);
    constexpr unsigned long long kOne = 1ull << 32;
    unsigned long long previous = 0;
    int drawn = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        unsigned long long threshold = kOne;
        if (k < kTopK - 1)
        {
            unsigned long long const scaled = __float2ull_rd(__fmul_rn(__fmul_rn(prefix[k], inv), 0x1p32f));
            threshold = scaled < kOne ? scaled : kOne;
        }
        // Monotone already (C_k is, and so are the rounded products and the floor); the guard keeps n_k >= 0.
        threshold = threshold > previous ? threshold : previous;
        if (owner)
        {
            qRow[k] = __fmul_rn(__ull2float_rn(threshold - previous), 0x1p-32f);
        }
        // min{k : u < T_k} as the count of T_k <= u (T is monotone and T_15 = 2^32 > u).
        drawn += threshold <= static_cast<unsigned long long>(u) ? 1 : 0;
        previous = threshold;
    }
    return bad ? greedyLane : drawn;
}

template <int kVariant, int kMode>
__global__ void __launch_bounds__(32, 1) drWalkTailKernel(WalkParams const p, RsWalkParams const rs)
{
    static_assert(kMode >= 0 && kMode <= 1, "kMode 0 (greedy) or 1 (RS, greedy chain); kMode 2 is drWalkSampledKernel");
    static_assert(kMode == 0 || kVariant >= 3, "the rejection-sampling walk is built on variant 3");
    constexpr bool kRedux = kVariant >= 2;
    constexpr bool kRs = kMode != 0;
    constexpr int kRsPositions = kRs ? rs::kPos : 1;
    __shared__ float4 sEdges[kMaxPositions * kTopK * kTopK / 4];
    __shared__ float sE[kRsPositions][kTopK + 1];
    __shared__ float sQ[kRsPositions][kTopK];
    __shared__ int sGreedyLane[kRsPositions];
    __shared__ bool sBad[kRsPositions];
    int const b = blockIdx.x;
    int const lane = threadIdx.x;
    int const positions = p.positions;
    // Variants 2 and 3: the tail's operands before the wait. Their producers (the verify step's accept, the attention
    // metadata) precede the select launch, which passed its own wait before anything after it could start.
    int32_t firstToken = 0;
    int32_t kvLen = 0;
    int32_t kvRewindBy = 0;
    if (kRedux && lane == 0)
    {
        int64_t const sourceRow = p.batchIndices[b];
        int column = p.numAccepted[b] - 1;
        column = column < 0 ? column + positions + 1 : column;
        firstToken = p.accepted[sourceRow * p.acceptedStride + column];
        if (p.kvRewind)
        {
            kvLen = p.kvLens[b];
            kvRewindBy = p.rewind[b];
        }
    }
    // Rejection sampling: the slot, the row-0 seed / offset / temperature and the position's draft word before the
    // wait (staged by host-to-device copies before the graph launch).
    int64_t slot = 0;
    int64_t seed = 0;
    int64_t base = 0;
    float temperature = 0.0f;
    uint32_t draftWord = 0;
    if constexpr (kRs)
    {
        int64_t const row0 = static_cast<int64_t>(b) * (positions + 1);
        slot = rs.slotIds[b];
        seed = rs.seeds[row0];
        base = rs.offsets[row0];
        temperature = rs.temperatures[row0];
        if (lane < rs::kPos)
        {
            draftWord = rs::Ph(static_cast<uint64_t>(seed), rs::draftSubseq(base, lane), 0).x;
        }
    }
    gridDependencyWait();
    if constexpr (kRs)
    {
        if (rs.check)
        {
            int64_t const row0 = static_cast<int64_t>(b) * (positions + 1);
            if (rs.slotIds[b] != slot || rs.seeds[row0] != seed || rs.offsets[row0] != base
                || __float_as_uint(rs.temperatures[row0]) != __float_as_uint(temperature))
            {
                rs::trap(rs.diag, rs::TrapCode::kCheckReread, b);
            }
        }
    }
    float4 const* src = reinterpret_cast<float4 const*>(p.edges + static_cast<int64_t>(b) * positions * kTopK * kTopK);
    int const edgeVectors = positions * kTopK * kTopK / 4;
    if constexpr (kVariant >= 3)
    {
        // Every lane's 16-byte loads in flight together, then the shared-memory stores.
        constexpr int kPerLane = kMaxPositions * kTopK * kTopK / 4 / 32;
        float4 staged[kPerLane];
#pragma unroll
        for (int k = 0; k < kPerLane; ++k)
        {
            int const i = lane + 32 * k;
            if (i < edgeVectors)
            {
                staged[k] = src[i];
            }
        }
#pragma unroll
        for (int k = 0; k < kPerLane; ++k)
        {
            int const i = lane + 32 * k;
            if (i < edgeVectors)
            {
                sEdges[i] = staged[k];
            }
        }
    }
    else
    {
        for (int i = lane; i < edgeVectors; i += 32)
        {
            sEdges[i] = src[i];
        }
    }
    bool const valid = lane < kTopK;
    float unary[kMaxPositions];
    long long ids[kMaxPositions];
#pragma unroll
    for (int t = 0; t < kMaxPositions; ++t)
    {
        unary[t] = 0.0f;
        ids[t] = 0;
        if (t < positions && valid)
        {
            int64_t const at = static_cast<int64_t>(b * positions + t) * kTopK + lane;
            unary[t] = p.unary[at];
            ids[t] = p.ids[at];
        }
    }
    float tau = 0.0f;
    if constexpr (kRs)
    {
        tau = (temperature > 0.0f && temperature < INFINITY) ? __frcp_rn(temperature) : __int_as_float(0x7FC00000);
        if (rs.transform == static_cast<int>(rs::Transform::kUnary) && valid)
        {
#pragma unroll
            for (int t = 0; t < rs::kPos; ++t)
            {
                unary[t] = rsTransform(unary[t], rs);
            }
        }
    }
    __syncwarp();
    float const* edges = reinterpret_cast<float const*>(sEdges);
    int32_t* nextRow = p.nextTokens + static_cast<int64_t>(b) * (positions + 1);
    int predecessor = 0;
    // Rejection sampling, per position (lane-uniform): the path lane, the greedy token, the drawn lane, poisoned.
    int pathLane[kMaxPositions];
    int32_t greedyToken[kMaxPositions];
    int drawnLane[kMaxPositions];
    bool poisoned[kMaxPositions];
#pragma unroll
    for (int t = 0; t < kMaxPositions; ++t)
    {
        if (t < positions)
        {
            if constexpr (kRs)
            {
                pathLane[t] = predecessor;
            }
            float const edge = valid ? edges[(t * kTopK + predecessor) * kTopK + lane] : 0.0f;
            float const score = __fadd_rn(unary[t], edge);
            bool const isNan = valid && score != score;
            unsigned const nanMask = __ballot_sync(kFullMask, isNan);
            float best = (!valid || isNan) ? -INFINITY : score;
            if constexpr (kRedux)
            {
                best = orderValue(__reduce_max_sync(kFullMask, orderKey(best)));
            }
            else
            {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                {
                    best = fmaxf(best, __shfl_xor_sync(kFullMask, best, o));
                }
            }
            bool const hit = valid && (nanMask != 0 ? isNan : score == best);
            unsigned const hitMask = __ballot_sync(kFullMask, hit);
            predecessor = __ffs(static_cast<int>(hitMask)) - 1;
            long long token;
            if constexpr (kRedux)
            {
                token = static_cast<long long>(
                    __reduce_min_sync(kFullMask, hit ? static_cast<uint32_t>(ids[t]) : 0xFFFFFFFFu));
            }
            else
            {
                token = hit ? ids[t] : 9223372036854775807LL;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                {
                    long long const other = __shfl_xor_sync(kFullMask, token, o);
                    token = other < token ? other : token;
                }
            }
            if (nanMask == 0 && best == -INFINITY)
            {
                token = 0;
            }
            if constexpr (kRs)
            {
                // The positions' softmax passes are independent of the path after them: each runs in its step.
                greedyToken[t] = static_cast<int32_t>(token);
                bool bad = rsSoftmaxRow(score, valid, tau, rs, sE[t], lane);
                if (lane == 0)
                {
                    sBad[t] = bad;
                    sGreedyLane[t] = predecessor;
                }
            }
            else
            {
                if (lane == 0)
                {
                    p.tokens[b * positions + t] = static_cast<int32_t>(token);
                    nextRow[1 + t] = static_cast<int32_t>(token);
                }
            }
        }
    }
    if constexpr (kMode == 1)
    {
        // One threshold scan for all positions: lane t < 7 scans position t, lanes 7..31 repeat position 6.
        __syncwarp();
        int const row = lane < rs::kPos ? lane : rs::kPos - 1;
        bool bad = sBad[row];
        int const drawn = rsThresholdRow(
            sE[row], bad, __shfl_sync(kFullMask, draftWord, row), lane == row, sQ[row], sGreedyLane[row]);
#pragma unroll
        for (int t = 0; t < rs::kPos; ++t)
        {
            drawnLane[t] = __shfl_sync(kFullMask, drawn, t);
            poisoned[t] = __shfl_sync(kFullMask, bad ? 1 : 0, t) != 0;
        }
    }
    if constexpr (kRs)
    {
        // The drafts and the q record.
        __syncwarp();
        bool const slotOk = slot >= 0 && slot < rs.numSlots;
        rs::QRecord* record = slotOk ? reinterpret_cast<rs::QRecord*>(rs.records + slot * rs::kRecordBytes) : nullptr;
        uint32_t poisonedMask = 0;
#pragma unroll
        for (int t = 0; t < rs::kPos; ++t)
        {
            int32_t const id = static_cast<int32_t>(ids[t]);
            // The lane of the greedy token (-1 if it is not a candidate: the -inf rule's token 0).
            int const greedyTokenLane
                = __ffs(static_cast<int>(__ballot_sync(kFullMask, valid && id == greedyToken[t]))) - 1;
            int32_t const sampled = __shfl_sync(kFullMask, id, drawnLane[t]);
            int32_t const draft = poisoned[t] ? greedyToken[t] : sampled;
            if (rs.check && valid)
            {
                unsigned const peers = __match_any_sync(0xFFFFu, id);
                if (peers != (1u << lane))
                {
                    rs::diagAdd(rs.diag, rs::Diag::kWalkDupIds);
                    rs::trap(rs.diag, rs::TrapCode::kCheckDupIds, static_cast<long long>(b) * rs::kPos + t);
                }
            }
            if (record != nullptr && valid)
            {
                record->ids[t][lane] = id;
                record->qf[t][lane] = poisoned[t] ? (lane == greedyTokenLane ? 1.0f : 0.0f) : sQ[t][lane];
            }
            if (record != nullptr && lane == t)
            {
                record->draft[t] = draft;
                record->posFlags[t] = poisoned[t] ? rs::kPosPoisoned : 0;
            }
            if (rs.exportE != nullptr && valid)
            {
                rs.exportE[(static_cast<int64_t>(b) * rs::kPos + t) * kTopK + lane] = sE[t][lane];
            }
            if (rs.exportU != nullptr && lane == t)
            {
                rs.exportU[static_cast<int64_t>(b) * rs::kPos + t] = static_cast<long long>(draftWord)
                    | static_cast<long long>(pathLane[t]) << 32 | static_cast<long long>(sGreedyLane[t]) << 40
                    | static_cast<long long>(drawnLane[t]) << 48 | static_cast<long long>(poisoned[t] ? 1 : 0) << 56;
            }
            if (lane == 0)
            {
                p.tokens[b * positions + t] = draft;
                nextRow[1 + t] = draft;
            }
            poisonedMask |= poisoned[t] ? (1u << t) : 0u;
        }
        __syncwarp();
        if (lane == 0)
        {
            if (record != nullptr)
            {
                record->mode = rs::modeByte(false, static_cast<rs::Transform>(rs.transform));
                record->kind = static_cast<uint8_t>(rs::Kind::kSampled);
                record->valid = rs::kValidAll;
            }
            else
            {
                rs::diagAdd(rs.diag, rs::Diag::kSlotOob);
                if (rs.check && slot != rs.dummySlot)
                {
                    rs::trap(rs.diag, rs::TrapCode::kCheckSlot, slot);
                }
            }
            rs::diagAdd(rs.diag, rs::Diag::kWalkRsCalls);
            if (poisonedMask != 0)
            {
                rs::diagAdd(rs.diag, rs::Diag::kDraftPoisoned, __popc(poisonedMask));
            }
        }
    }
    if (lane == 0)
    {
        if (!kRedux)
        {
            int64_t const sourceRow = p.batchIndices[b];
            int column = p.numAccepted[b] - 1;
            column = column < 0 ? column + positions + 1 : column;
            firstToken = p.accepted[sourceRow * p.acceptedStride + column];
            if (p.kvRewind)
            {
                kvLen = p.kvLens[b];
                kvRewindBy = p.rewind[b];
            }
        }
        nextRow[0] = firstToken;
        if (p.kvRewind)
        {
            int32_t const rewound = kvLen - kvRewindBy;
            p.kvLens[b] = rewound > 0 ? rewound : 0;
        }
    }
}

// ---- the sampled-chain walk (kMode 2) ----

//! Floats per edge row in shared memory: 16 + 4 of padding, so a lane reading its own 16-byte row is conflict-free.
constexpr int kEdgeRowF = 20;
//! Rows of the draw table: (0, 0) at row 0, then (t, pi) at 1 + 16 (t - 1) + pi for t = 1..6.
constexpr int kDrawRows = 1 + (rs::kPos - 1) * kTopK;
//! Floats per x row of the table (15 used; 17 makes the lane-per-row stores conflict-free).
constexpr int kXStride = kTopK + 1;
constexpr float kTwo32 = 0x1p32f;
//! Table entry of a poisoned row before its greedy lane is filled in.
constexpr uint32_t kRowPoisoned = 0x20u;
//! Next-lane bit of a poisoned position (next lane = the greedy lane).
constexpr uint32_t kNextPoisoned = 0x10u;

__device__ __forceinline__ int drawRowOf(int t, int pi)
{
    return t == 0 ? 0 : 1 + kTopK * (t - 1) + pi;
}

//! c_u: the smallest float c with c 2^32 >= u + 1, so that floor(x 2^32) <= u  <=>  x < c for every float x >= 0.
__device__ __forceinline__ float drawBound(uint32_t u)
{
    return __fmul_rn(__ull2float_ru(static_cast<unsigned long long>(u) + 1ull), 0x1p-32f);
}

//! The greedy lane of a row held by one lane: the lowest NaN lane, else the lowest lane equal to the maximum (== is
//! blind to the sign of a zero, as the warp form's ordered maximum and == are).
__device__ __forceinline__ int greedyLaneOf(float const (&s)[kTopK])
{
    int nanLane = -1;
    float best = -INFINITY;
#pragma unroll
    for (int k = kTopK - 1; k >= 0; --k)
    {
        if (s[k] != s[k])
        {
            nanLane = k;
        }
        else
        {
            best = fmaxf(best, s[k]);
        }
    }
    if (nanLane >= 0)
    {
        return nanLane;
    }
    int g = 0;
#pragma unroll
    for (int k = kTopK - 1; k >= 0; --k)
    {
        g = s[k] == best ? k : g;
    }
    return g;
}

//! The greedy walk step (kMode 0) on a row in the warp form (lane k < 16 holds candidate k): the greedy lane and the
//! greedy token (the -inf rule's 0).
__device__ __forceinline__ void greedyWarp(
    float score, bool valid, int32_t id, int lane, int& greedyLane, int32_t& token)
{
    bool const isNan = valid && score != score;
    unsigned const nanMask = __ballot_sync(kFullMask, isNan);
    float best = (!valid || isNan) ? -INFINITY : score;
    best = orderValue(__reduce_max_sync(kFullMask, orderKey(best)));
    bool const hit = valid && (nanMask != 0 ? isNan : score == best);
    unsigned const hitMask = __ballot_sync(kFullMask, hit);
    greedyLane = __ffs(static_cast<int>(hitMask)) - 1;
    uint32_t const tok = __reduce_min_sync(kFullMask, hit ? static_cast<uint32_t>(id) : 0xFFFFFFFFu);
    token = (nanMask == 0 && best == -INFINITY) ? 0 : static_cast<int32_t>(tok);
}

//! One table row (t, pi) on one lane, raw or unary transform: the next lane, or kRowPoisoned; xRow[k] = x_k, k < 15.
__device__ __forceinline__ uint32_t drawTableRow(
    float const* edgeRow, float const* uRow, float tau, float cU, float* xRow)
{
    float z[kTopK];
#pragma unroll
    for (int c = 0; c < kTopK / 4; ++c)
    {
        float4 const e4 = reinterpret_cast<float4 const*>(edgeRow)[c];
        float4 const u4 = reinterpret_cast<float4 const*>(uRow)[c];
        z[4 * c + 0] = __fmul_rn(__fadd_rn(u4.x, e4.x), tau);
        z[4 * c + 1] = __fmul_rn(__fadd_rn(u4.y, e4.y), tau);
        z[4 * c + 2] = __fmul_rn(__fadd_rn(u4.z, e4.z), tau);
        z[4 * c + 3] = __fmul_rn(__fadd_rn(u4.w, e4.w), tau);
    }
    float m[kTopK / 2];
#pragma unroll
    for (int k = 0; k < kTopK / 2; ++k)
    {
        m[k] = fmaxf(z[2 * k], z[2 * k + 1]);
    }
#pragma unroll
    for (int k = 0; k < kTopK / 4; ++k)
    {
        m[k] = fmaxf(m[2 * k], m[2 * k + 1]);
    }
    float const zmax = fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3]));
    float prefix[kTopK];
    float c = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        c = __fadd_rn(c, __expf(__fsub_rn(z[k], zmax)));
        prefix[k] = c;
    }
    float const inv = __frcp_rn(c);
    uint32_t drawn = 0;
#pragma unroll
    for (int k = 0; k < kTopK - 1; ++k)
    {
        float const x = __fmul_rn(prefix[k], inv);
        xRow[k] = x;
        drawn += x < cU ? 1u : 0u;
    }
    return (c > 0.0f && c < INFINITY) ? drawn : kRowPoisoned;
}

//! One table row (t, pi) on one lane, score transform: the next lane, or the greedy lane | kNextPoisoned when the row
//! is poisoned; xRow[k] = x_k, k < 15.
__device__ __forceinline__ uint32_t drawTableRowScore(
    float const* edgeRow, float const* uRow, float tau, float cU, RsWalkParams const& rs, float* xRow)
{
    float s[kTopK];
#pragma unroll
    for (int c = 0; c < kTopK / 4; ++c)
    {
        float4 const e4 = reinterpret_cast<float4 const*>(edgeRow)[c];
        float4 const u4 = reinterpret_cast<float4 const*>(uRow)[c];
        s[4 * c + 0] = __fadd_rn(u4.x, e4.x);
        s[4 * c + 1] = __fadd_rn(u4.y, e4.y);
        s[4 * c + 2] = __fadd_rn(u4.z, e4.z);
        s[4 * c + 3] = __fadd_rn(u4.w, e4.w);
    }
    float z[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        z[k] = __fmul_rn(rsTransform(s[k], rs), tau);
    }
    float m[kTopK / 2];
#pragma unroll
    for (int k = 0; k < kTopK / 2; ++k)
    {
        m[k] = fmaxf(z[2 * k], z[2 * k + 1]);
    }
#pragma unroll
    for (int k = 0; k < kTopK / 4; ++k)
    {
        m[k] = fmaxf(m[2 * k], m[2 * k + 1]);
    }
    float const zmax = fmaxf(fmaxf(m[0], m[1]), fmaxf(m[2], m[3]));
    float prefix[kTopK];
    float c = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        c = __fadd_rn(c, __expf(__fsub_rn(z[k], zmax)));
        prefix[k] = c;
    }
    float const inv = __frcp_rn(c);
    uint32_t drawn = 0;
#pragma unroll
    for (int k = 0; k < kTopK - 1; ++k)
    {
        float const x = __fmul_rn(prefix[k], inv);
        xRow[k] = x;
        drawn += x < cU ? 1u : 0u;
    }
    if (!(c > 0.0f && c < INFINITY))
    {
        drawn = static_cast<uint32_t>(greedyLaneOf(s)) | kNextPoisoned;
    }
    return drawn;
}

//! Row (t, pi) in the warp form (lane k < 16 holds candidate k): the next lane | kNextPoisoned when poisoned (then the
//! greedy lane), lane-uniform; xRow[k] = x_k for k < 15; sE receives the row's e_k.
__device__ __forceinline__ uint32_t drawWarpRow(int t, int pi, float const* sUF, float const* sEdgeF,
    int32_t const* sIds, float tau, float cU, RsWalkParams const& rs, bool scoreT, float* xRow, float* sE, int lane)
{
    bool const valid = lane < kTopK;
    float const score = valid ? __fadd_rn(sUF[t * kTopK + lane], sEdgeF[(t * kTopK + pi) * kEdgeRowF + lane]) : 0.0f;
    float const sv = scoreT ? rsTransform(score, rs) : score;
    float const z = valid ? __fmul_rn(sv, tau) : -INFINITY;
    float const zmax = orderValue(__reduce_max_sync(kFullMask, orderKey(z)));
    float const e = __expf(__fsub_rn(z, zmax));
    if (valid)
    {
        sE[lane] = e;
    }
    __syncwarp();
    float c = 0.0f;
    float mine = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        c = __fadd_rn(c, sE[k]);
        mine = k == lane ? c : mine;
    }
    bool const bad = !(c > 0.0f && c < INFINITY);
    float const x = __fmul_rn(mine, __frcp_rn(c));
    uint32_t next = __popc(__ballot_sync(kFullMask, lane < kTopK - 1 && x < cU));
    if (lane < kTopK - 1)
    {
        xRow[lane] = x;
    }
    if (bad)
    {
        int greedyLane;
        int32_t token;
        greedyWarp(score, valid, sIds[t * kTopK + (lane & (kTopK - 1))], lane, greedyLane, token);
        next = static_cast<uint32_t>(greedyLane) | kNextPoisoned;
    }
    __syncwarp();
    return next;
}

//! A request's draft-side filter: which of min-p, top-k, top-p act, and their values (min-p clamped to [0, 1]).
struct WalkFilter
{
    bool minP;
    bool topK;
    bool topP;
    float minPValue;
    int k;
    float p;
};

//! The request's filter of one row in the warp form, the stock order: min-p (keep e_k >= minP), then top-k (keep
//! #{j : e_j > e_k} < k: every tie at the k-th value is kept), then top-p over the survivors' mass S1 (keep when the
//! survivors' mass strictly above e_k is at most fl(topP S1); the maximum, e = 1, is always kept). Sums in lane order.
//! The counts and the mass above run over every lane: for a lane that survives min-p and top-k, every lane above it
//! survives them too, so they equal the survivors-only forms there. eRow holds the row's e_k (the caller synced);
//! eRow[lane] becomes e_k or 0; sKeepRow is a 17-float scratch row. Returns whether this lane is kept (lane-varying).
__device__ __forceinline__ bool rsFilterRow(
    float* eRow, float* sKeepRow, float e, bool valid, WalkFilter const& f, int lane)
{
    int above = 0;
    float massAbove = 0.0f;
#pragma unroll
    for (int j = 0; j < kTopK; ++j)
    {
        float const ej = eRow[j];
        bool const gt = ej > e;
        above += gt ? 1 : 0;
        massAbove = gt ? __fadd_rn(massAbove, ej) : massAbove;
    }
    bool const keepMK = valid && (!f.minP || e >= f.minPValue) && (!f.topK || above < f.k);
    sKeepRow[lane < kTopK ? lane : kTopK] = keepMK ? e : 0.0f;
    __syncwarp();
    float s1 = 0.0f;
#pragma unroll
    for (int j = 0; j < kTopK; ++j)
    {
        s1 = __fadd_rn(s1, sKeepRow[j]);
    }
    bool const keep = keepMK && (!f.topP || e == 1.0f || massAbove <= __fmul_rn(f.p, s1));
    __syncwarp();
    if (valid)
    {
        eRow[lane] = keep ? e : 0.0f;
    }
    __syncwarp();
    return keep;
}

//! Row (t, pi) of a filtered request in the warp form: as drawWarpRow, with the request's filter between e_k and the
//! prefix sums. The row is poisoned iff a pre-filter e_k is NaN or S is not in (0, inf). A lane k < 15 whose prefix sum
//! is already S (every later lane removed) gets x_k = 1, so no later lane can be drawn and every later n_k is 0.
//! ``removed`` accumulates the lanes the filter removed (lane-uniform).
__device__ __forceinline__ uint32_t drawFilteredRow(int t, int pi, float const* sUF, float const* sEdgeF,
    int32_t const* sIds, float tau, float cU, RsWalkParams const& rs, bool scoreT, WalkFilter const& f, float* xRow,
    float* sE, float* sKeepRow, int lane, uint32_t& removed)
{
    bool const valid = lane < kTopK;
    float const score = valid ? __fadd_rn(sUF[t * kTopK + lane], sEdgeF[(t * kTopK + pi) * kEdgeRowF + lane]) : 0.0f;
    float const sv = scoreT ? rsTransform(score, rs) : score;
    float const z = valid ? __fmul_rn(sv, tau) : -INFINITY;
    float const zmax = orderValue(__reduce_max_sync(kFullMask, orderKey(z)));
    float const e = __expf(__fsub_rn(z, zmax));
    if (valid)
    {
        sE[lane] = e;
    }
    bool const nanPre = __ballot_sync(kFullMask, valid && !(e == e)) != 0;
    __syncwarp();
    bool const keep = rsFilterRow(sE, sKeepRow, e, valid, f, lane);
    removed += __popc(__ballot_sync(kFullMask, valid && !keep));
    float c = 0.0f;
    float mine = 0.0f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        c = __fadd_rn(c, sE[k]);
        mine = k == lane ? c : mine;
    }
    bool const bad = nanPre || !(c > 0.0f && c < INFINITY);
    float const x = mine == c ? 1.0f : __fmul_rn(mine, __frcp_rn(c));
    uint32_t next = __popc(__ballot_sync(kFullMask, lane < kTopK - 1 && x < cU));
    if (lane < kTopK - 1)
    {
        xRow[lane] = x;
    }
    if (bad)
    {
        int greedyLane;
        int32_t token;
        greedyWarp(score, valid, sIds[t * kTopK + (lane & (kTopK - 1))], lane, greedyLane, token);
        next = static_cast<uint32_t>(greedyLane) | kNextPoisoned;
    }
    __syncwarp();
    return next;
}

//! The kMode 2 walk (the sampled chain), one warp per request (K = 7). kExport: the test exports ``export_e`` /
//! ``export_u`` (the served instantiation has none).
template <bool kExport>
__global__ void __launch_bounds__(32, 1) drWalkSampledKernel(WalkParams const p, RsWalkParams const rs)
{
    constexpr int kP = rs::kPos;
    constexpr int kCells = kP * kTopK;
    __shared__ float4 sEdges[kP * kTopK * kEdgeRowF / 4];
    __shared__ float4 sU4[kCells / 4];
    __shared__ int32_t sIds[kCells];
    __shared__ float sX[kDrawRows][kXStride];
    __shared__ uint32_t sNext[kDrawRows];
    __shared__ float sE0[kTopK + 1];
    __shared__ float sKeepRow[kTopK + 1];
    __shared__ int32_t sGTok[kP + 1];
    __shared__ int32_t sGLane[kP + 1];
    int const b = blockIdx.x;
    int const lane = threadIdx.x;
    int const positions = p.positions;
    // The tail's operands and the request's slot / row-0 seed, offset, temperature, top-k / top-p / min-p before the
    // wait (variant 2's class: their producers precede the select launch, or they are staged by host-to-device copies
    // before the graph).
    int32_t firstToken = 0;
    int32_t kvLen = 0;
    int32_t kvRewindBy = 0;
    if (lane == 0)
    {
        int64_t const sourceRow = p.batchIndices[b];
        int column = p.numAccepted[b] - 1;
        column = column < 0 ? column + positions + 1 : column;
        firstToken = p.accepted[sourceRow * p.acceptedStride + column];
        if (p.kvRewind)
        {
            kvLen = p.kvLens[b];
            kvRewindBy = p.rewind[b];
        }
    }
    int64_t const row0 = static_cast<int64_t>(b) * (positions + 1);
    int64_t const slot = rs.slotIds[b];
    int64_t const seed = rs.seeds[row0];
    int64_t const base = rs.offsets[row0];
    float const temperature = rs.temperatures[row0];
    int32_t topK = 0;
    float topP = 1.0f;
    float minP = 0.0f;
    if (rs.filterMask != 0)
    {
        topK = rs.topKs[row0];
        topP = rs.topPs[row0];
        minP = rs.minPs[row0];
    }
    gridDependencyWait();
    if (rs.check)
    {
        if (rs.slotIds[b] != slot || rs.seeds[row0] != seed || rs.offsets[row0] != base
            || __float_as_uint(rs.temperatures[row0]) != __float_as_uint(temperature)
            || (rs.filterMask != 0
                && (rs.topKs[row0] != topK || __float_as_uint(rs.topPs[row0]) != __float_as_uint(topP)
                    || __float_as_uint(rs.minPs[row0]) != __float_as_uint(minP))))
        {
            rs::trap(rs.diag, rs::TrapCode::kCheckReread, b);
        }
    }
    // The request's filter (lane-uniform): the stock activity rules (top-k iff 0 < k < 16, top-p iff p < 1, min-p iff
    // its clamp to [0, 1] is > 0) restricted to the deploy's filters; none under the strict filter mode.
    WalkFilter f;
    f.minPValue = fminf(fmaxf(minP, 0.0f), 1.0f);
    f.k = topK;
    f.p = topP;
    f.topK = (rs.filterMask & 1) != 0 && topK > 0 && topK < kTopK;
    f.topP = (rs.filterMask & 2) != 0 && topP < 1.0f;
    f.minP = (rs.filterMask & 4) != 0 && f.minPValue > 0.0f;
    bool const filt = rs.filterMode == static_cast<int>(rs::FilterMode::kExact) && (f.topK || f.topP || f.minP);
    // Loads first, then the Philox word, 1 / T and c_u under their latency.
    float4 const* src = reinterpret_cast<float4 const*>(p.edges + static_cast<int64_t>(b) * kP * kTopK * kTopK);
    constexpr int kPerLane = kP * kTopK * kTopK / 4 / 32;
    float4 staged[kPerLane];
#pragma unroll
    for (int k = 0; k < kPerLane; ++k)
    {
        staged[k] = src[lane + 32 * k];
    }
    constexpr int kCellRounds = (kCells + 31) / 32;
    float uv[kCellRounds];
    int32_t iv[kCellRounds];
    int64_t const cellBase = static_cast<int64_t>(b) * positions * kTopK;
#pragma unroll
    for (int j = 0; j < kCellRounds; ++j)
    {
        int const cell = lane + 32 * j;
        uv[j] = 0.0f;
        iv[j] = 0;
        if (cell < kCells)
        {
            uv[j] = p.unary[cellBase + cell];
            iv[j] = static_cast<int32_t>(p.ids[cellBase + cell]);
        }
    }
    uint32_t draftWord = 0;
    if (lane < kP)
    {
        draftWord = rs::Ph(static_cast<uint64_t>(seed), rs::draftSubseq(base, lane), 0).x;
    }
    float const tau
        = (temperature > 0.0f && temperature < INFINITY) ? __frcp_rn(temperature) : __int_as_float(0x7FC00000);
    // Lane t < 7: c_u of position t.
    float const cUmine = drawBound(draftWord);
#pragma unroll
    for (int k = 0; k < kPerLane; ++k)
    {
        int const i = lane + 32 * k;
        sEdges[(i >> 2) * (kEdgeRowF / 4) + (i & 3)] = staged[k];
    }
    float* sUF = reinterpret_cast<float*>(sU4);
    bool const unaryT = rs.transform == static_cast<int>(rs::Transform::kUnary);
#pragma unroll
    for (int j = 0; j < kCellRounds; ++j)
    {
        int const cell = lane + 32 * j;
        if (cell < kCells)
        {
            sUF[cell] = unaryT ? rsTransform(uv[j], rs) : uv[j];
            sIds[cell] = iv[j];
        }
    }
    if (lane <= kP)
    {
        sGTok[lane] = 0;
        sGLane[lane] = -1;
    }
    __syncwarp();
    float const* sEdgeF = reinterpret_cast<float const*>(sEdges);
    bool const scoreT = rs.transform == static_cast<int>(rs::Transform::kScore);
    bool const valid = lane < kTopK;
    // pathWord: the lane of position t's predecessor in bits 4t..4t+3 (pi_0 = 0); poisonedMask: bit t.
    uint32_t pathWord = 0;
    uint32_t poisonedMask = 0;
    uint32_t removed = 0;
    if (!filt)
    {
        // (1) The draw table: round i, lanes 0..15 -> (2i + 1, lane), lanes 16..31 -> (2i + 2, lane - 16).
        if (!scoreT)
        {
#pragma unroll 1
            for (int i = 0; i < 3; ++i)
            {
                int const t = 2 * i + 1 + (lane >> 4);
                int const r = 1 + 32 * i + lane;
                float const cU = __shfl_sync(kFullMask, cUmine, t);
                sNext[r] = drawTableRow(
                    sEdgeF + (t * kTopK + (lane & (kTopK - 1))) * kEdgeRowF, sUF + t * kTopK, tau, cU, sX[r]);
            }
        }
        else
        {
#pragma unroll 1
            for (int i = 0; i < 3; ++i)
            {
                int const t = 2 * i + 1 + (lane >> 4);
                int const r = 1 + 32 * i + lane;
                float const cU = __shfl_sync(kFullMask, cUmine, t);
                sNext[r] = drawTableRowScore(
                    sEdgeF + (t * kTopK + (lane & (kTopK - 1))) * kEdgeRowF, sUF + t * kTopK, tau, cU, rs, sX[r]);
            }
        }
        // Poisoned table rows get their greedy lane (the walk scores recomputed with the same statements).
        bool mine = false;
#pragma unroll 1
        for (int i = 0; i < 3; ++i)
        {
            mine = mine || (sNext[1 + 32 * i + lane] & kRowPoisoned) != 0;
        }
        if (__ballot_sync(kFullMask, mine) != 0)
        {
#pragma unroll 1
            for (int i = 0; i < 3; ++i)
            {
                int const t = 2 * i + 1 + (lane >> 4);
                int const r = 1 + 32 * i + lane;
                if ((sNext[r] & kRowPoisoned) != 0)
                {
                    float const* er = sEdgeF + (t * kTopK + (lane & (kTopK - 1))) * kEdgeRowF;
                    float const* ur = sUF + t * kTopK;
                    float s[kTopK];
#pragma unroll
                    for (int k = 0; k < kTopK; ++k)
                    {
                        s[k] = __fadd_rn(ur[k], er[k]);
                    }
                    sNext[r] = static_cast<uint32_t>(greedyLaneOf(s)) | kNextPoisoned;
                }
            }
        }
        // (2) Row (0, 0) in the warp form, then the chain: 6 lookups.
        uint32_t v = drawWarpRow(
            0, 0, sUF, sEdgeF, sIds, tau, __shfl_sync(kFullMask, cUmine, 0), rs, scoreT, sX[0], sE0, lane);
        int pi = static_cast<int>(v & 15u);
        pathWord = static_cast<uint32_t>(pi) << 4;
        poisonedMask = v >> 4;
#pragma unroll
        for (int t = 1; t < kP; ++t)
        {
            v = sNext[drawRowOf(t, pi)];
            pi = static_cast<int>(v & 15u);
            pathWord |= static_cast<uint32_t>(pi) << (4 * (t + 1));
            poisonedMask |= (v >> 4) << t;
        }
    }
    else
    {
        // Filtered requests: the 7 visited rows in series, each filtered (row t of the x table is position t).
        int pi = 0;
#pragma unroll 1
        for (int t = 0; t < kP; ++t)
        {
            uint32_t const v = drawFilteredRow(t, pi, sUF, sEdgeF, sIds, tau, __shfl_sync(kFullMask, cUmine, t), rs,
                scoreT, f, sX[t], sE0, sKeepRow, lane, removed);
            pi = static_cast<int>(v & 15u);
            pathWord |= static_cast<uint32_t>(pi) << (4 * (t + 1));
            poisonedMask |= (v >> 4) << t;
        }
    }
    // (3) Poisoned positions: the greedy token and its lane; tests: the exports of the visited rows.
    if (kExport || poisonedMask != 0)
    {
#pragma unroll 1
        for (int t = 0; t < kP; ++t)
        {
            bool const poisoned = ((poisonedMask >> t) & 1u) != 0;
            if (kExport || poisoned)
            {
                int const pt = static_cast<int>((pathWord >> (4 * t)) & 15u);
                float const score
                    = valid ? __fadd_rn(sUF[t * kTopK + lane], sEdgeF[(t * kTopK + pt) * kEdgeRowF + lane]) : 0.0f;
                int32_t const id = sIds[t * kTopK + (lane & (kTopK - 1))];
                int greedyLane;
                int32_t token;
                greedyWarp(score, valid, id, lane, greedyLane, token);
                int const tokenLane = __ffs(static_cast<int>(__ballot_sync(kFullMask, valid && id == token))) - 1;
                if (lane == 0)
                {
                    sGTok[t] = token;
                    sGLane[t] = tokenLane;
                }
                if constexpr (kExport)
                {
                    uint32_t const ut = __shfl_sync(kFullMask, draftWord, t);
                    if (rs.exportE != nullptr)
                    {
                        __syncwarp();
                        rsSoftmaxRow(score, valid, tau, rs, sE0, lane);
                        __syncwarp();
                        if (filt)
                        {
                            rsFilterRow(sE0, sKeepRow, valid ? sE0[lane] : 0.0f, valid, f, lane);
                        }
                        if (valid)
                        {
                            rs.exportE[(static_cast<int64_t>(b) * kP + t) * kTopK + lane] = sE0[lane];
                        }
                        __syncwarp();
                    }
                    if (rs.exportU != nullptr && lane == 0)
                    {
                        rs.exportU[static_cast<int64_t>(b) * kP + t] = static_cast<long long>(ut)
                            | static_cast<long long>(pt) << 32 | static_cast<long long>(greedyLane) << 40
                            | static_cast<long long>((pathWord >> (4 * (t + 1))) & 15u) << 48
                            | static_cast<long long>(poisoned ? 1 : 0) << 56;
                    }
                }
            }
        }
        __syncwarp();
    }
    bool const slotOk = slot >= 0 && slot < rs.numSlots;
    rs::QRecord* record = slotOk ? reinterpret_cast<rs::QRecord*>(rs.records + slot * rs::kRecordBytes) : nullptr;
    int32_t* nextRow = p.nextTokens + static_cast<int64_t>(b) * (positions + 1);
    if (rs.check)
    {
#pragma unroll 1
        for (int t = 0; t < kP; ++t)
        {
            if (valid)
            {
                unsigned const peers = __match_any_sync(0xFFFFu, sIds[t * kTopK + lane]);
                if (peers != (1u << lane))
                {
                    rs::diagAdd(rs.diag, rs::Diag::kWalkDupIds);
                    rs::trap(rs.diag, rs::TrapCode::kCheckDupIds, static_cast<long long>(b) * kP + t);
                }
            }
        }
    }
    // (4) The record, two positions per round (lanes 0..15 position 2i, lanes 16..31 position 2i + 1), no warp
    // collectives: T_k = min(floor(fl(x_k 2^32)), 2^32) held mod 2^32 (0 when full), n_k = T_k - T_{k-1} in 32 bits;
    // a full T_k after a zero T_{k-1} is n_k = 2^32 (qf = 1).
    int const k = lane & (kTopK - 1);
#pragma unroll
    for (int i = 0; i < (kP + 1) / 2; ++i)
    {
        int const t = 2 * i + (lane >> 4);
        bool const active = t < kP;
        int const tc = active ? t : kP - 1;
        int const pt = static_cast<int>((pathWord >> (4 * tc)) & 15u);
        int const d = static_cast<int>((pathWord >> (4 * (tc + 1))) & 15u);
        bool const poisoned = ((poisonedMask >> tc) & 1u) != 0;
        int const row = filt ? tc : drawRowOf(tc, pt);
        int32_t const id = sIds[tc * kTopK + k];
        int32_t const sampled = sIds[tc * kTopK + d];
        int32_t const gtok = sGTok[tc];
        int const glane = sGLane[tc];
        float const yk = __fmul_rn(k < kTopK - 1 ? sX[row][k] : 1.0f, kTwo32);
        float const ykm1 = __fmul_rn(k > 0 ? sX[row][k - 1] : 0.0f, kTwo32);
        bool const fullK = !(yk < kTwo32);
        bool const fullKm1 = !(ykm1 < kTwo32);
        uint32_t const tK = fullK ? 0u : __float2uint_rd(yk);
        uint32_t const tKm1 = fullKm1 ? 0u : __float2uint_rd(ykm1);
        float const qfS = (fullK && !fullKm1 && tKm1 == 0u) ? 1.0f : __fmul_rn(__uint2float_rn(tK - tKm1), 0x1p-32f);
        float const qf = poisoned ? (k == glane ? 1.0f : 0.0f) : qfS;
        int32_t const draft = poisoned ? gtok : sampled;
        if (active && record != nullptr)
        {
            record->ids[t][k] = id;
            record->qf[t][k] = qf;
        }
        if (active && k == 0)
        {
            if (record != nullptr)
            {
                record->draft[t] = draft;
                record->posFlags[t] = poisoned ? rs::kPosPoisoned : (filt ? rs::kPosFiltered : 0);
            }
            p.tokens[b * positions + t] = draft;
            nextRow[1 + t] = draft;
        }
    }
    __syncwarp();
    if (lane == 0)
    {
        if (record != nullptr)
        {
            record->mode = rs::modeByte(true, static_cast<rs::Transform>(rs.transform));
            record->kind = static_cast<uint8_t>(rs::Kind::kSampled);
            record->valid = rs::kValidAll;
        }
        else
        {
            rs::diagAdd(rs.diag, rs::Diag::kSlotOob);
            if (rs.check && slot != rs.dummySlot)
            {
                rs::trap(rs.diag, rs::TrapCode::kCheckSlot, slot);
            }
        }
        rs::diagAdd(rs.diag, rs::Diag::kWalkRsCalls);
        if (poisonedMask != 0)
        {
            rs::diagAdd(rs.diag, rs::Diag::kDraftPoisoned, __popc(poisonedMask));
        }
        if (filt)
        {
            rs::diagAdd(rs.diag, rs::Diag::kWalkFilteredPositions, kP);
            if (removed != 0)
            {
                rs::diagAdd(rs.diag, rs::Diag::kWalkFilterRemovedLanes, removed);
            }
        }
        nextRow[0] = firstToken;
        if (p.kvRewind)
        {
            int32_t const rewound = kvLen - kvRewindBy;
            p.kvLens[b] = rewound > 0 ? rewound : 0;
        }
    }
}

__global__ void drDryKernel() {}

cudaLaunchConfig_t launchConfig(int grid, int block, bool pdl, cudaLaunchAttribute* attribute)
{
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid);
    config.blockDim = dim3(block);
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    return config;
}

bool aligned(void const* pointer, uintptr_t bytes)
{
    return reinterpret_cast<uintptr_t>(pointer) % bytes == 0;
}

void selectPublishOp(at::Tensor const& logits, at::Tensor const& counters, int64_t mcArea, int64_t areaBytes,
    int64_t rank, int64_t world, bool pdl, bool dry)
{
    c10::cuda::CUDAGuard guard(counters.device());
    cudaLaunchAttribute attribute[1];
    if (dry)
    {
        cudaLaunchConfig_t config = launchConfig(1, 32, pdl, attribute);
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drDryKernel));
        return;
    }
    TORCH_CHECK(logits.is_cuda() && counters.is_cuda(), "dr draft tail: CUDA tensors only");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 2 && logits.stride(1) == 1,
        "dr draft tail: logits must be a [rows, shard] BF16 tensor with unit column stride");
    int64_t const rows = logits.size(0);
    int64_t const shard = logits.size(1);
    TORCH_CHECK(rows >= 1 && rows <= kMaxPositions, "dr draft tail: 1..", kMaxPositions, " rows");
    TORCH_CHECK(shard >= kTopK && shard <= 65536 && shard % 8 == 0,
        "dr draft tail: a shard of 16..65536 columns, "
        "a multiple of 8");
    TORCH_CHECK(logits.stride(0) % 8 == 0 && aligned(logits.data_ptr(), 16), "dr draft tail: misaligned logits");
    TORCH_CHECK(world >= 1 && world <= kMaxWorld && rank >= 0 && rank < world, "dr draft tail: bad rank / world");
    int64_t const chunks = (shard + kChunk - 1) / kChunk;
    TORCH_CHECK(counters.scalar_type() == at::kInt && counters.is_contiguous() && counters.numel() >= rows * chunks,
        "dr draft tail: select counters must be a contiguous int32 tensor of rows * chunks entries");
    TORCH_CHECK(mcArea % 16 == 0 && areaBytes == rows * world * chunks * kTopK * 8,
        "dr draft tail: the area does not match rows x world x keys");
    SelectParams p{};
    p.logits = reinterpret_cast<uint16_t const*>(logits.data_ptr());
    p.rowStride = logits.stride(0);
    p.shard = static_cast<int>(shard);
    p.chunks = static_cast<int>(chunks);
    p.mcArea = static_cast<uint64_t>(mcArea);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.rank = static_cast<int>(rank);
    p.world = static_cast<int>(world);
    p.keysPerRank = static_cast<int>(chunks * kTopK);
    cudaLaunchConfig_t config = launchConfig(static_cast<int>(rows * chunks), kSelectThreads, pdl, attribute);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drSelectPublishKernel, p));
}

void mergeOp(at::Tensor const& counters, int64_t ucArea, int64_t areaBytes, int64_t world, int64_t shard,
    at::Tensor const& gate, at::Tensor const& anchor, at::Tensor const& predCodebook, at::Tensor const& succCodebook,
    at::Tensor const& ids, at::Tensor const& unary, at::Tensor const& preds, at::Tensor const& succs, bool pdl,
    int64_t timeoutNs, int64_t variant)
{
    c10::cuda::CUDAGuard guard(counters.device());
    TORCH_CHECK(
        gate.scalar_type() == at::kBFloat16 && gate.dim() == 2 && gate.size(1) == kRankDim && gate.stride(1) == 1,
        "dr draft tail: gate must be a [positions, 256] BF16 tensor with unit column stride");
    int64_t const positions = gate.size(0);
    TORCH_CHECK(positions >= 1 && positions <= kMaxPositions, "dr draft tail: 1..", kMaxPositions, " positions");
    TORCH_CHECK(world >= 1 && world <= kMaxWorld && shard >= kTopK && shard <= 65536, "dr draft tail: bad geometry");
    int64_t const chunks = (shard + kChunk - 1) / kChunk;
    TORCH_CHECK(counters.scalar_type() == at::kInt && counters.is_contiguous() && counters.numel() >= positions,
        "dr draft tail: merge counters must be a contiguous int32 tensor of positions entries");
    TORCH_CHECK(ucArea % 16 == 0 && areaBytes == positions * world * chunks * kTopK * 8,
        "dr draft tail: the area does not match positions x world x keys");
    TORCH_CHECK((anchor.scalar_type() == at::kLong || anchor.scalar_type() == at::kInt) && anchor.numel() >= 1,
        "dr draft tail: anchor must be an int32 / int64 tensor");
    for (at::Tensor const* cb : {&predCodebook, &succCodebook})
    {
        TORCH_CHECK(cb->scalar_type() == at::kBFloat16 && cb->dim() == 2 && cb->size(1) == kRankDim
                && cb->stride(1) == 1 && cb->size(0) == world * shard,
            "dr draft tail: codebooks must be [vocab, 256] BF16 with unit column stride");
    }
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous() && ids.numel() == positions * kTopK
            && unary.scalar_type() == at::kFloat && unary.is_contiguous() && unary.numel() == positions * kTopK,
        "dr draft tail: ids / unary must be contiguous [positions, 16] int64 / fp32");
    TORCH_CHECK(preds.scalar_type() == at::kFloat && preds.is_contiguous()
            && preds.numel() == positions * kTopK * kRankDim && succs.scalar_type() == at::kFloat
            && succs.is_contiguous() && succs.numel() == positions * kTopK * kRankDim,
        "dr draft tail: preds / succs must be contiguous [positions, 16, 256] fp32");
    MergeParams p{};
    p.ucArea = static_cast<uint64_t>(ucArea);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.world = static_cast<int>(world);
    p.shard = static_cast<int>(shard);
    p.chunks = static_cast<int>(chunks);
    p.keysPerRank = static_cast<int>(chunks * kTopK);
    p.positions = static_cast<int>(positions);
    p.gate = reinterpret_cast<uint16_t const*>(gate.data_ptr());
    p.gateStride = gate.stride(0);
    p.anchor = anchor.data_ptr();
    p.anchorIs64 = anchor.scalar_type() == at::kLong ? 1 : 0;
    p.predCodebook = reinterpret_cast<uint16_t const*>(predCodebook.data_ptr());
    p.predStride = predCodebook.stride(0);
    p.succCodebook = reinterpret_cast<uint16_t const*>(succCodebook.data_ptr());
    p.succStride = succCodebook.stride(0);
    p.ids = ids.data_ptr<int64_t>();
    p.unary = unary.data_ptr<float>();
    p.preds = preds.data_ptr<float>();
    p.succs = succs.data_ptr<float>();
    p.timeoutNs = static_cast<unsigned long long>(timeoutNs);
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config = launchConfig(static_cast<int>(positions), kMergeThreads, pdl, attribute);
    TORCH_CHECK(variant >= 1 && variant <= 3, "dr draft tail: variant 1, 2 or 3");
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drMergeKernel, p));
}

//! The walk's operands from ``walk_tail``'s arguments (shared by ``walk_tail`` and ``walk_tail_rs``).
WalkParams walkParamsOf(at::Tensor const& edges, at::Tensor const& unary, at::Tensor const& ids,
    at::Tensor const& tokens, at::Tensor const& nextTokens, at::Tensor const& accepted, at::Tensor const& numAccepted,
    at::Tensor const& batchIndices, at::Tensor const& kvLens, at::Tensor const& rewind, bool kvRewind)
{
    TORCH_CHECK(
        unary.dim() == 3 && unary.size(2) == kTopK && unary.scalar_type() == at::kFloat && unary.is_contiguous(),
        "dr draft tail: unary must be a contiguous [B, K, 16] fp32 tensor");
    int64_t const requests = unary.size(0);
    int64_t const positions = unary.size(1);
    TORCH_CHECK(positions >= 1 && positions <= kMaxPositions, "dr draft tail: 1..", kMaxPositions, " positions");
    TORCH_CHECK(edges.scalar_type() == at::kFloat && edges.is_contiguous()
            && edges.numel() == requests * positions * kTopK * kTopK && aligned(edges.data_ptr(), 16),
        "dr draft tail: edges must be a contiguous, 16-byte aligned [B, K, 16, 16] fp32 tensor");
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous() && ids.numel() == requests * positions * kTopK,
        "dr draft tail: ids must be a contiguous [B, K, 16] int64 tensor");
    TORCH_CHECK(tokens.scalar_type() == at::kInt && tokens.is_contiguous() && tokens.numel() == requests * positions
            && nextTokens.scalar_type() == at::kInt && nextTokens.is_contiguous()
            && nextTokens.numel() == requests * (positions + 1),
        "dr draft tail: tokens / next tokens must be contiguous int32 [B, K] / [B, K + 1]");
    TORCH_CHECK(accepted.scalar_type() == at::kInt && accepted.dim() == 2 && accepted.stride(1) == 1
            && accepted.size(1) == positions + 1,
        "dr draft tail: accepted must be int32 [B, K + 1] with unit column stride");
    TORCH_CHECK(numAccepted.scalar_type() == at::kInt && numAccepted.numel() == requests
            && batchIndices.scalar_type() == at::kInt && batchIndices.numel() >= requests,
        "dr draft tail: num_accepted / batch_indices must be int32");
    TORCH_CHECK(!kvRewind
            || (kvLens.scalar_type() == at::kInt && kvLens.numel() >= requests && rewind.scalar_type() == at::kInt
                && rewind.numel() == requests),
        "dr draft tail: kv_lens / rewind must be int32");
    WalkParams p{};
    p.edges = edges.data_ptr<float>();
    p.unary = unary.data_ptr<float>();
    p.ids = reinterpret_cast<long long const*>(ids.data_ptr<int64_t>());
    p.positions = static_cast<int>(positions);
    p.tokens = tokens.data_ptr<int32_t>();
    p.nextTokens = nextTokens.data_ptr<int32_t>();
    p.accepted = accepted.data_ptr<int32_t>();
    p.acceptedStride = accepted.stride(0);
    p.numAccepted = numAccepted.data_ptr<int32_t>();
    p.batchIndices = batchIndices.data_ptr<int32_t>();
    p.kvLens = kvRewind ? kvLens.data_ptr<int32_t>() : nullptr;
    p.rewind = kvRewind ? rewind.data_ptr<int32_t>() : nullptr;
    p.kvRewind = kvRewind ? 1 : 0;
    return p;
}

void walkTailOp(at::Tensor const& edges, at::Tensor const& unary, at::Tensor const& ids, at::Tensor const& tokens,
    at::Tensor const& nextTokens, at::Tensor const& accepted, at::Tensor const& numAccepted,
    at::Tensor const& batchIndices, at::Tensor const& kvLens, at::Tensor const& rewind, bool kvRewind, bool pdl,
    int64_t variant)
{
    c10::cuda::CUDAGuard guard(edges.device());
    WalkParams const p = walkParamsOf(
        edges, unary, ids, tokens, nextTokens, accepted, numAccepted, batchIndices, kvLens, rewind, kvRewind);
    RsWalkParams const rs{};
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config = launchConfig(static_cast<int>(unary.size(0)), 32, pdl, attribute);
    TORCH_CHECK(variant >= 1 && variant <= 3, "dr draft tail: variant 1, 2 or 3");
    if (variant == 3)
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkTailKernel<3, 0>, p, rs));
    }
    else if (variant == 2)
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkTailKernel<2, 0>, p, rs));
    }
    else
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkTailKernel<1, 0>, p, rs));
    }
}

//! ``walk_tail`` with rejection sampling (variant 3, K = 7): the drafts drawn from q, the q records written. kMode 2
//! applies the request's top-k / top-p / min-p (``top_ks`` / ``top_ps`` / ``min_ps``, per token like
//! ``temperatures``) to q when given, restricted to ``filter_mask`` and off under the strict ``filter_mode``.
void walkTailRsOp(at::Tensor const& edges, at::Tensor const& unary, at::Tensor const& ids, at::Tensor const& tokens,
    at::Tensor const& nextTokens, at::Tensor const& accepted, at::Tensor const& numAccepted,
    at::Tensor const& batchIndices, at::Tensor const& kvLens, at::Tensor const& rewind, bool kvRewind, bool pdl,
    int64_t variant, at::Tensor const& records, int64_t numSlots, int64_t dummySlot, at::Tensor const& slotIds,
    at::Tensor const& seeds, at::Tensor const& offsets, at::Tensor const& temperatures, int64_t mode, int64_t transform,
    double mult, double invCap, double cap, int64_t check, at::Tensor const& diag,
    std::optional<at::Tensor> const& exportE, std::optional<at::Tensor> const& exportU,
    std::optional<at::Tensor> const& topKs, std::optional<at::Tensor> const& topPs,
    std::optional<at::Tensor> const& minPs, int64_t filterMode, int64_t filterMask)
{
    c10::cuda::CUDAGuard guard(edges.device());
    WalkParams const p = walkParamsOf(
        edges, unary, ids, tokens, nextTokens, accepted, numAccepted, batchIndices, kvLens, rewind, kvRewind);
    int64_t const requests = unary.size(0);
    int64_t const rows = requests * (unary.size(1) + 1);
    TORCH_CHECK(variant == 3, "dr draft tail: rejection sampling runs on variant 3");
    TORCH_CHECK(unary.size(1) == rs::kPos, "dr draft tail: rejection sampling needs ", rs::kPos, " positions");
    TORCH_CHECK(mode == 1 || mode == 2, "dr draft tail: rejection-sampling mode 1 (greedy chain) or 2 (sampled chain)");
    TORCH_CHECK(transform >= 0 && transform <= 2, "dr draft tail: q transform 0 (raw), 1 (score) or 2 (unary)");
    TORCH_CHECK(records.is_cuda() && records.get_device() == edges.get_device() && records.scalar_type() == at::kByte
            && records.is_contiguous() && numSlots >= 1 && records.numel() >= numSlots * rs::kRecordBytes
            && aligned(records.data_ptr(), alignof(rs::QRecord)),
        "dr draft tail: records must be a contiguous, 16-byte aligned uint8 [num_slots, ", rs::kRecordBytes,
        "] tensor");
    TORCH_CHECK(slotIds.is_cuda() && slotIds.scalar_type() == at::kLong && slotIds.is_contiguous()
            && slotIds.numel() >= requests,
        "dr draft tail: slot_ids must be a contiguous int64 tensor of at least B entries");
    TORCH_CHECK(seeds.is_cuda() && seeds.scalar_type() == at::kLong && seeds.is_contiguous() && seeds.numel() >= rows
            && offsets.is_cuda() && offsets.scalar_type() == at::kLong && offsets.is_contiguous()
            && offsets.numel() >= rows && temperatures.is_cuda() && temperatures.scalar_type() == at::kFloat
            && temperatures.is_contiguous() && temperatures.numel() >= rows,
        "dr draft tail: seeds / offsets (int64) and temperatures (fp32) must be contiguous per-token tensors of at "
        "least B (K + 1) entries");
    TORCH_CHECK(
        diag.is_cuda() && diag.scalar_type() == at::kLong && diag.is_contiguous() && diag.numel() == rs::kDiagWords,
        "dr draft tail: diag must be a contiguous int64 [", rs::kDiagWords, "] tensor");
    RsWalkParams rs{};
    rs.records = records.data_ptr<uint8_t>();
    rs.numSlots = static_cast<int>(numSlots);
    rs.dummySlot = static_cast<int>(dummySlot);
    rs.slotIds = slotIds.data_ptr<int64_t>();
    rs.seeds = seeds.data_ptr<int64_t>();
    rs.offsets = offsets.data_ptr<int64_t>();
    rs.temperatures = temperatures.data_ptr<float>();
    rs.transform = static_cast<int>(transform);
    rs.mult = static_cast<float>(mult);
    rs.invCap = static_cast<float>(invCap);
    rs.cap = static_cast<float>(cap);
    rs.check = check != 0 ? 1 : 0;
    rs.diag = reinterpret_cast<unsigned long long*>(diag.data_ptr<int64_t>());
    if (exportE.has_value())
    {
        TORCH_CHECK(exportE->is_cuda() && exportE->scalar_type() == at::kFloat && exportE->is_contiguous()
                && exportE->numel() == requests * rs::kPos * kTopK,
            "dr draft tail: export_e must be a contiguous fp32 [B, 7, 16] tensor");
        rs.exportE = exportE->data_ptr<float>();
    }
    if (exportU.has_value())
    {
        TORCH_CHECK(exportU->is_cuda() && exportU->scalar_type() == at::kLong && exportU->is_contiguous()
                && exportU->numel() == requests * rs::kPos,
            "dr draft tail: export_u must be a contiguous int64 [B, 7] tensor");
        rs.exportU = reinterpret_cast<long long*>(exportU->data_ptr<int64_t>());
    }
    bool const filters = topKs.has_value() && topPs.has_value() && minPs.has_value();
    TORCH_CHECK(filters || (!topKs.has_value() && !topPs.has_value() && !minPs.has_value()),
        "dr draft tail: top_ks, top_ps and min_ps go together");
    TORCH_CHECK(filterMode == static_cast<int64_t>(rs::FilterMode::kExact)
            || filterMode == static_cast<int64_t>(rs::FilterMode::kStrict),
        "dr draft tail: filter mode 0 (exact) or 1 (strict)");
    TORCH_CHECK(filterMask >= 0 && filterMask <= 7, "dr draft tail: filter mask bits 0 (top-k), 1 (top-p), 2 (min-p)");
    rs.filterMode = static_cast<int>(filterMode);
    if (filters && filterMask != 0)
    {
        TORCH_CHECK(topKs->is_cuda() && topKs->scalar_type() == at::kInt && topKs->is_contiguous()
                && topKs->numel() >= rows && topPs->is_cuda() && topPs->scalar_type() == at::kFloat
                && topPs->is_contiguous() && topPs->numel() >= rows && minPs->is_cuda()
                && minPs->scalar_type() == at::kFloat && minPs->is_contiguous() && minPs->numel() >= rows,
            "dr draft tail: top_ks (int32), top_ps and min_ps (fp32) must be contiguous per-token tensors of at least "
            "B (K + 1) entries");
        rs.topKs = topKs->data_ptr<int32_t>();
        rs.topPs = topPs->data_ptr<float>();
        rs.minPs = minPs->data_ptr<float>();
        rs.filterMask = static_cast<int>(filterMask);
    }
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config = launchConfig(static_cast<int>(requests), 32, pdl, attribute);
    if (mode == 2)
    {
        if (rs.exportE != nullptr || rs.exportU != nullptr)
        {
            C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkSampledKernel<true>, p, rs));
        }
        else
        {
            C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkSampledKernel<false>, p, rs));
        }
    }
    else
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkTailKernel<3, 1>, p, rs));
    }
}

//! ``cudaFuncAttributes`` of the rejection-sampling walks of kMode 1 (``drWalkTailKernel<3, 1>``), kMode 2
//! (``drWalkSampledKernel<false>``) and kMode 2 with the test exports (``drWalkSampledKernel<true>``) (int64 [3, 4]:
//! registers, local bytes, static shared bytes, max threads per block). Resolves the kernels, so that a CUDA-graph
//! capture is never their first use.
at::Tensor walkRsAttributesOp(at::Tensor const& anchor)
{
    c10::cuda::CUDAGuard guard(anchor.device());
    auto out = at::empty({3, 4}, at::TensorOptions().dtype(at::kLong));
    int64_t* const o = out.data_ptr<int64_t>();
    void const* const kernels[3] = {reinterpret_cast<void const*>(drWalkTailKernel<3, 1>),
        reinterpret_cast<void const*>(drWalkSampledKernel<false>),
        reinterpret_cast<void const*>(drWalkSampledKernel<true>)};
    for (int i = 0; i < 3; ++i)
    {
        cudaFuncAttributes a{};
        C10_CUDA_CHECK(cudaFuncGetAttributes(&a, kernels[i]));
        o[4 * i + 0] = a.numRegs;
        o[4 * i + 1] = static_cast<int64_t>(a.localSizeBytes);
        o[4 * i + 2] = static_cast<int64_t>(a.sharedSizeBytes);
        o[4 * i + 3] = a.maxThreadsPerBlock;
    }
    return out;
}

} // namespace

TORCH_LIBRARY(musespark_open_dr_tail, m)
{
    m.def(
        "select(Tensor logits, Tensor(a!) counters, int mc_area, int area_bytes, int rank, int world, bool pdl, "
        "bool dry) -> ()");
    m.def(
        "merge(Tensor(a!) counters, int uc_area, int area_bytes, int world, int shard, Tensor gate, Tensor anchor, "
        "Tensor pred_codebook, Tensor succ_codebook, Tensor(b!) ids, Tensor(c!) unary, Tensor(d!) preds, "
        "Tensor(e!) succs, bool pdl, int timeout_ns, int variant) -> ()");
    m.def(
        "walk_tail(Tensor edges, Tensor unary, Tensor ids, Tensor(a!) tokens, Tensor(b!) next_tokens, Tensor accepted, "
        "Tensor num_accepted, Tensor batch_indices, Tensor(c!) kv_lens, Tensor rewind, bool kv_rewind, bool pdl, "
        "int variant) -> ()");
    m.def(
        "walk_tail_rs(Tensor edges, Tensor unary, Tensor ids, Tensor(a!) tokens, Tensor(b!) next_tokens, "
        "Tensor accepted, Tensor num_accepted, Tensor batch_indices, Tensor(c!) kv_lens, Tensor rewind, "
        "bool kv_rewind, bool pdl, int variant, Tensor(d!) records, int num_slots, int dummy_slot, Tensor slot_ids, "
        "Tensor seeds, Tensor offsets, Tensor temperatures, int mode, int transform, float mult, float inv_cap, "
        "float cap, int check, Tensor(e!) diag, Tensor(f!)? export_e, Tensor(g!)? export_u, Tensor? top_ks=None, "
        "Tensor? top_ps=None, Tensor? min_ps=None, int filter_mode=0, int filter_mask=0) -> ()");
    m.def("walk_rs_attributes(Tensor anchor) -> Tensor");
}

TORCH_LIBRARY_IMPL(musespark_open_dr_tail, CUDA, m)
{
    m.impl("select", &selectPublishOp);
    m.impl("merge", &mergeOp);
    m.impl("walk_tail", &walkTailOp);
    m.impl("walk_tail_rs", &walkTailRsOp);
    m.impl("walk_rs_attributes", &walkRsAttributesOp);
}
