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
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

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

template <int kVariant>
__global__ void __launch_bounds__(32, 1) drWalkTailKernel(WalkParams const p)
{
    constexpr bool kRedux = kVariant >= 2;
    __shared__ float4 sEdges[kMaxPositions * kTopK * kTopK / 4];
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
    gridDependencyWait();
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
    __syncwarp();
    float const* edges = reinterpret_cast<float const*>(sEdges);
    int32_t* nextRow = p.nextTokens + static_cast<int64_t>(b) * (positions + 1);
    int predecessor = 0;
#pragma unroll
    for (int t = 0; t < kMaxPositions; ++t)
    {
        if (t < positions)
        {
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
            if (lane == 0)
            {
                p.tokens[b * positions + t] = static_cast<int32_t>(token);
                nextRow[1 + t] = static_cast<int32_t>(token);
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

void walkTailOp(at::Tensor const& edges, at::Tensor const& unary, at::Tensor const& ids, at::Tensor const& tokens,
    at::Tensor const& nextTokens, at::Tensor const& accepted, at::Tensor const& numAccepted,
    at::Tensor const& batchIndices, at::Tensor const& kvLens, at::Tensor const& rewind, bool kvRewind, bool pdl,
    int64_t variant)
{
    c10::cuda::CUDAGuard guard(edges.device());
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
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config = launchConfig(static_cast<int>(requests), 32, pdl, attribute);
    TORCH_CHECK(variant >= 1 && variant <= 3, "dr draft tail: variant 1, 2 or 3");
    if (variant == 3)
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkTailKernel<3>, p));
    }
    else if (variant == 2)
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkTailKernel<2>, p));
    }
    else
    {
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, drWalkTailKernel<1>, p));
    }
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
}

TORCH_LIBRARY_IMPL(musespark_open_dr_tail, CUDA, m)
{
    m.impl("select", &selectPublishOp);
    m.impl("merge", &mergeOp);
    m.impl("walk_tail", &walkTailOp);
}
