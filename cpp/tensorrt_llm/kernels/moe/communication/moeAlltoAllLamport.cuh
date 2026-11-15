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

// Sentinel-polled (data-as-flag) one-sided MoE all-to-all: each receive slot holds a sentinel until the sender's
// data overwrites it, and the receiver polls the slot, so the data is its own completion flag.
// The fence path (moeAlltoAllKernels.cu) separates data from synchronization: payload stores, a system-scope
// release fence, a completion flag per peer, a poll, and in the combine a flag exchange, an acquire fence and a
// remote pull. The sentinel-polled path instead uses the data as its own flag, like the one-shot all-reduce in
// allReduceFusionKernels.cu (helpers in lamportUtils.cuh): the receiver polls each slot with volatile loads until
// no word of it is the sentinel. There is no fence, no flag hop and no remote read:
//   - dispatch: the sender announces its token count to every peer at kernel start, pushes each token's payloads
//     into the target's receive slot, and adds each routed token to a per-peer tally with a remote atomic (tokens
//     routed in the high word, rows for that peer in the low word). The receiver waits until a source's tally covers
//     all its announced tokens, then polls the rows, copies them into the caller-visible receive tensors, restores
//     the sentinel, and writes the invalid expert id into unused rows (the separate sanitize kernel is fused). Rows a
//     rank sends to itself go straight to its receive tensors.
//   - combine: every rank pushes the expert outputs it computed for a peer's tokens into that peer's combine slots,
//     then reduces its own tokens, polling the slots of remote contributions and reading local ones in place. The
//     reduction keeps the fence combine's top-k order and fp32 tree, so the result is bit-identical.
//   - state that the fence path resets with prepare kernels (round parity, send counters, the announced counts and
//     the tallies) is reset by the combine for the next round, so no prepare kernel runs.
//
// Data movement: a token's payloads form one record of 16-byte chunks (each payload starts on a chunk; a payload
// whose row is not a multiple of 16 bytes is zero-padded to its last chunk). Every thread moves whole chunks with one
// 16-byte access, and all chunks of a row (and of all rows a block handles) are loaded before any is waited on, so a
// row costs one memory round trip, not one per payload.
//
// Sentinel: the all-ones 32-bit word. The receiver checks every word of a slot: a slot is complete when none of its
// words is the sentinel. The sentinel is a NaN pattern in every payload format this path carries (fp8 e4m3 x4, E8M0
// scale x4, bf16/fp16 x2, fp32, and -1 as an int32 expert id, which routing never emits), so no value is changed to
// make room for it. A source word equal to the sentinel cannot be sent; the sender traps with a message instead of
// hanging the receiver.
//
// Reuse: dispatch slots, announced counts and tallies alternate between two parities by round; the combine region is
// restored within the round that consumes it. A peer can write a slot again only after it has received a message
// this rank sends in a later kernel (the next round's dispatch tally), so every restore is complete before reuse.
// All blocks of a grid must be co-resident (each block pushes before it polls); the launcher caps the grid at the
// device's resident-block capacity and the kernels stride over tokens.
//
// This header has no #include: the including translation unit provides <cuda_runtime.h>, <cuda_bf16.h>,
// <cuda_fp16.h>, <cstdint>, <cstddef> and <cstdio> before opening its namespace. Without __CUDACC__ (a host compiler)
// it declares only the constants, the workspace layout and the parameter structs.

#ifndef TRTLLM_MOE_ALLTOALL_LAMPORT_CUH
#define TRTLLM_MOE_ALLTOALL_LAMPORT_CUH

namespace moe_a2a_lamport
{

static constexpr int kMaxRanks = 64;
static constexpr int kMaxPayloads = 4;
static constexpr int kMaxItemsPerThread = 4; // chunks a thread loads before it waits on any of them
// Token t adds to tally shard t % kTallyShards. One shard: more shards were not faster in testing, so the receiver
// polls a single word per source.
static constexpr int kTallyShards = 1;
static constexpr uint32_t kSentinel32 = 0xFFFFFFFFu;

// Per-rank workspace layout, in bytes from the rank's base pointer. [0, kZeroInitBytes) starts zeroed, the rest all
// ones (the sentinel).
static constexpr size_t kCtrlBytes = 1024;       // Ctrl, local only
static constexpr size_t kTallyOffset = 1024;     // uint64 [2][kMaxRanks][kTallyShards]: tokens << 32 | rows (zero)
static constexpr size_t kZeroInitBytes = 9216;
static constexpr size_t kExpectedOffset = 9216;  // int32 [2][kMaxRanks] per source: announced tokens (sentinel)
static constexpr size_t kDataOffset = 12288;     // dispatch slots [2][ep][capacity][record], then combine slots
static_assert(kTallyOffset + 2 * kMaxRanks * kTallyShards * sizeof(uint64_t) <= kZeroInitBytes, "tally layout");
static_assert(kExpectedOffset + 2 * kMaxRanks * sizeof(int32_t) <= kDataOffset, "count layout");

#if defined(__CUDACC__)
#define MOE_A2A_LAMPORT_HD __host__ __device__
#else
#define MOE_A2A_LAMPORT_HD
#endif

struct Ctrl
{
    uint32_t round;     // dispatch parity is round & 1; the combine advances it
    int32_t reserved[3];
    int32_t sendCounters[kMaxRanks];
    int32_t recvCounts[kMaxRanks]; // rows received from each source in the current round (read by the combine)
};

static_assert(sizeof(Ctrl) <= kCtrlBytes, "Ctrl must fit its reserved bytes");

// Capacity of one rank's workspace; identical on every rank and fixed for the workspace's lifetime.
struct Layout
{
    int epSize;
    int capacityTokens;    // per source rank
    int recordBytes;       // dispatch bytes per token slot (16-byte multiple)
    int combineSlotBytes;  // combine bytes per token slot (16-byte multiple)

    MOE_A2A_LAMPORT_HD size_t dispatchBytesPerParity() const
    {
        return static_cast<size_t>(epSize) * capacityTokens * recordBytes;
    }

    MOE_A2A_LAMPORT_HD size_t dispatchOffset(uint32_t parity) const
    {
        return kDataOffset + parity * dispatchBytesPerParity();
    }

    MOE_A2A_LAMPORT_HD size_t combineOffset() const
    {
        return kDataOffset + 2 * dispatchBytesPerParity();
    }

    MOE_A2A_LAMPORT_HD size_t totalBytes() const
    {
        return combineOffset() + static_cast<size_t>(epSize) * capacityTokens * combineSlotBytes;
    }
};

struct DispatchParams
{
    Layout layout;
    int epSize;
    int epRank;
    int numExperts;
    int localNumTokens;
    int runtimeMaxTokens;
    int numPayloads;
    void const* src[kMaxPayloads];      // [localNumTokens, bytes]
    int bytesPerToken[kMaxPayloads];
    int vecBytes[kMaxPayloads];         // element granularity of a payload row: 16, 8, 4 or 2 bytes
    int chunkBegin[kMaxPayloads + 1];   // payload q owns record chunks [chunkBegin[q], chunkBegin[q + 1])
    void* recv[kMaxPayloads];           // local [epSize, runtimeMaxTokens, bytes] receive tensors
    int32_t const* tokenSelectedExperts; // [localNumTokens, topK]
    int* topkTargetRanks;               // [localNumTokens, topK], -1 for duplicate routes
    int* topkSendIndices;               // [localNumTokens, topK], -1 for duplicate routes
    int expertIdPayload;                // payload index of the expert ids; -1: no invalid-id fill
    int32_t invalidExpertId;
    uint8_t* workspace[kMaxRanks];      // every rank's sentinel-polled workspace, mapped locally
    int64_t timeoutCycles;
};

struct CombineParams
{
    Layout layout;
    int epSize;
    int epRank;
    int localNumTokens;
    int runtimeMaxTokens;
    int elementsPerToken;
    void const* source;     // expert outputs [epSize, runtimeMaxTokens, elements], sourceStrideBytes per token
    int sourceStrideBytes;
    void* output;           // [localNumTokens, elements]
    int const* topkTargetRanks;
    int const* topkSendIndices;
    uint8_t* workspace[kMaxRanks];
    int64_t timeoutCycles;
};

// Element granularity of a payload row (the bytes per token must be even).
inline int vecBytesFor(int bytes)
{
    if (bytes % 16 == 0)
    {
        return 16;
    }
    if (bytes % 8 == 0)
    {
        return 8;
    }
    if (bytes % 4 == 0)
    {
        return 4;
    }
    if (bytes % 2 == 0)
    {
        return 2;
    }
    return 0; // odd byte counts are not supported
}

// One thread per 16-byte chunk of a row, in whole warps, at least 64 threads (routing, counts and invalid-id fill
// use the first threads of a block).
inline int blockSizeForChunks(int chunks)
{
    int b = (chunks + 31) / 32 * 32;
    return b < 64 ? 64 : (b > 1024 ? 1024 : b);
}

#if defined(__CUDACC__)

// ---------------------------------------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ int targetRankOf(int expertId, int base, int remainder)
{
    // Same contiguous ceil/floor partition as moeAlltoAllKernels.cu compute_target_rank_id.
    if (remainder == 0)
    {
        return expertId / base;
    }
    int const split = remainder * (base + 1);
    if (expertId < split)
    {
        return expertId / (base + 1);
    }
    return remainder + (expertId - split) / base;
}

__device__ __forceinline__ int loadVolatileS32(int const* ptr)
{
    int v;
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ void storeVolatileS32(int* ptr, int v)
{
    asm volatile("st.volatile.global.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ uint64_t loadVolatileU64(uint64_t const* ptr)
{
    uint64_t v;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ int* expectedWord(uint8_t* base, uint32_t parity, int source)
{
    return reinterpret_cast<int*>(base + kExpectedOffset) + parity * kMaxRanks + source;
}

__device__ __forceinline__ uint64_t* tallyWord(uint8_t* base, uint32_t parity, int source, int shard)
{
    return reinterpret_cast<uint64_t*>(base + kTallyOffset) + (parity * kMaxRanks + source) * kTallyShards + shard;
}

__device__ __forceinline__ uint4 loadVolatileChunk(void const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ bool chunkDirty(uint4 const& v)
{
    return v.x == kSentinel32 || v.y == kSentinel32 || v.z == kSentinel32 || v.w == kSentinel32;
}

__device__ __forceinline__ void storeSentinelChunk(void* p)
{
    *reinterpret_cast<uint4*>(p) = make_uint4(kSentinel32, kSentinel32, kSentinel32, kSentinel32);
}

__device__ __forceinline__ void trapWith(char const* what, int rank, int peer, int index)
{
    printf("MoE all-to-all (sentinel-polled): rank %d %s (peer %d, index %d)\n", rank, what, peer, index);
    asm volatile("trap;");
}

// Poll a 16-byte slot until none of its words is the sentinel.
__device__ __forceinline__ void pollChunk(
    uint4& v, void const* slot, int64_t timeoutCycles, char const* what, int rank, int peer, int index)
{
    if (chunkDirty(v))
    {
        long long const start = clock64();
        do
        {
            if (clock64() - start > timeoutCycles)
            {
                trapWith(what, rank, peer, index);
            }
            v = loadVolatileChunk(slot);
        } while (chunkDirty(v));
    }
}

__device__ __forceinline__ uint32_t loadU16(uint8_t const* p)
{
    return *reinterpret_cast<uint16_t const*>(p);
}

// Bytes [off, off + 16) of a payload row, accessed with the row's element granularity; zero beyond the row's end.
// Words are composed in registers (no address-taken temporaries).
__device__ __forceinline__ uint4 loadRowChunk(uint8_t const* row, int bytes, int vec, int off)
{
    uint8_t const* s = row + off;
    if (vec == 16)
    {
        return *reinterpret_cast<uint4 const*>(s);
    }
    int const n = bytes - off < 16 ? bytes - off : 16;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (vec == 8)
    {
        uint2 const lo = *reinterpret_cast<uint2 const*>(s);
        r.x = lo.x;
        r.y = lo.y;
        if (n > 8)
        {
            uint2 const hi = *reinterpret_cast<uint2 const*>(s + 8);
            r.z = hi.x;
            r.w = hi.y;
        }
    }
    else if (vec == 4)
    {
        uint32_t const* w = reinterpret_cast<uint32_t const*>(s);
        r.x = w[0];
        r.y = n > 4 ? w[1] : 0u;
        r.z = n > 8 ? w[2] : 0u;
        r.w = n > 12 ? w[3] : 0u;
    }
    else
    {
        r.x = loadU16(s) | (n > 2 ? loadU16(s + 2) << 16 : 0u);
        r.y = (n > 4 ? loadU16(s + 4) : 0u) | (n > 6 ? loadU16(s + 6) << 16 : 0u);
        r.z = (n > 8 ? loadU16(s + 8) : 0u) | (n > 10 ? loadU16(s + 10) << 16 : 0u);
        r.w = (n > 12 ? loadU16(s + 12) : 0u) | (n > 14 ? loadU16(s + 14) << 16 : 0u);
    }
    return r;
}

__device__ __forceinline__ void storeU16(uint8_t* p, uint32_t v)
{
    *reinterpret_cast<uint16_t*>(p) = static_cast<uint16_t>(v);
}

__device__ __forceinline__ void storeRowChunk(uint8_t* row, int bytes, int vec, int off, uint4 const& v)
{
    uint8_t* d = row + off;
    if (vec == 16)
    {
        *reinterpret_cast<uint4*>(d) = v;
        return;
    }
    int const n = bytes - off < 16 ? bytes - off : 16;
    if (vec == 8)
    {
        *reinterpret_cast<uint2*>(d) = make_uint2(v.x, v.y);
        if (n > 8)
        {
            *reinterpret_cast<uint2*>(d + 8) = make_uint2(v.z, v.w);
        }
    }
    else if (vec == 4)
    {
        uint32_t* w = reinterpret_cast<uint32_t*>(d);
        w[0] = v.x;
        if (n > 4)
        {
            w[1] = v.y;
        }
        if (n > 8)
        {
            w[2] = v.z;
        }
        if (n > 12)
        {
            w[3] = v.w;
        }
    }
    else
    {
        uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            if (2 * i < n)
            {
                storeU16(d + 2 * i, (i & 1) ? (words[i >> 1] >> 16) : words[i >> 1]);
            }
        }
    }
}

__device__ __forceinline__ int payloadOfChunk(DispatchParams const& p, int chunk)
{
    int q = 0;
#pragma unroll
    for (int i = 1; i < kMaxPayloads; ++i)
    {
        if (i < p.numPayloads && chunk >= p.chunkBegin[i])
        {
            q = i;
        }
    }
    return q;
}

__device__ __forceinline__ uint8_t* dispatchSlot(
    DispatchParams const& p, int owner, uint32_t parity, int source, int index)
{
    return p.workspace[owner] + p.layout.dispatchOffset(parity)
        + (static_cast<size_t>(source) * p.layout.capacityTokens + index) * p.layout.recordBytes;
}

__device__ __forceinline__ uint8_t* recvRow(DispatchParams const& p, int payload, int source, int index)
{
    return static_cast<uint8_t*>(p.recv[payload])
        + (static_cast<size_t>(source) * p.runtimeMaxTokens + index) * p.bytesPerToken[payload];
}

__device__ __forceinline__ uint8_t* combineSlot(uint8_t* ownerBase, Layout const& layout, int from, int index)
{
    return ownerBase + layout.combineOffset()
        + (static_cast<size_t>(from) * layout.capacityTokens + index) * layout.combineSlotBytes;
}

// ---------------------------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------------------------

template <int TOP_K>
__global__ void moeA2ALamportDispatchKernel(DispatchParams const p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
    static_assert(TOP_K <= 32, "warp-parallel routing requires TOP_K <= warpSize");
    constexpr int kItems = kMaxItemsPerThread;
    __shared__ int sTarget[TOP_K];
    __shared__ int sIndex[TOP_K];
    __shared__ int sCount[kMaxRanks];
    __shared__ int sSources[kMaxRanks];
    __shared__ int sNumSources;

    uint8_t* const self = p.workspace[p.epRank];
    Ctrl* const ctrl = reinterpret_cast<Ctrl*>(self);
    uint32_t const parity = ctrl->round & 1u; // written by an earlier kernel only
    int const base = p.numExperts / p.epSize;
    int const remainder = p.numExperts - base * p.epSize;
    int const recordChunks = p.chunkBegin[p.numPayloads];

    // Announce this rank's token count to every peer; it arrives while the tokens are routed.
    if (blockIdx.x == 0 && threadIdx.x < p.epSize)
    {
        storeVolatileS32(expectedWord(p.workspace[threadIdx.x], parity, p.epRank), p.localNumTokens);
    }

    // Send: one token per block iteration. The token's chunks are loaded first, so the loads overlap the routing.
    for (int t = blockIdx.x; t < p.localNumTokens; t += gridDim.x)
    {
        for (int first = 0; first < recordChunks; first += blockDim.x * kItems)
        {
            uint4 v[kItems];
#pragma unroll
            for (int i = 0; i < kItems; ++i)
            {
                int const c = first + i * blockDim.x + threadIdx.x;
                if (c < recordChunks)
                {
                    int const q = payloadOfChunk(p, c);
                    uint8_t const* row
                        = static_cast<uint8_t const*>(p.src[q]) + static_cast<size_t>(t) * p.bytesPerToken[q];
                    v[i] = loadRowChunk(row, p.bytesPerToken[q], p.vecBytes[q], (c - p.chunkBegin[q]) * 16);
                }
            }
            if (first == 0)
            {
                if (threadIdx.x < TOP_K)
                {
                    // Routing as in moeAlltoAllKernels.cu route_dispatch_token: one route per distinct target rank.
                    int const k = threadIdx.x;
                    uint32_t const laneMask = (TOP_K == 32) ? ~0U : ((1U << TOP_K) - 1U);
                    int const expertId = p.tokenSelectedExperts[t * TOP_K + k];
                    int const target = targetRankOf(expertId, base, remainder);
                    uint32_t const sameTarget = __match_any_sync(laneMask, target);
                    bool const keep = (__ffs(sameTarget) - 1) == k;
                    int const index = keep ? atomicAdd(&ctrl->sendCounters[target], 1) : -1;
                    int const route = keep ? target : -1;
                    p.topkTargetRanks[t * TOP_K + k] = route;
                    p.topkSendIndices[t * TOP_K + k] = index;
                    sTarget[k] = route;
                    sIndex[k] = index;
                }
                __syncthreads();
                if (threadIdx.x < p.epSize)
                {
                    // Tally the routed token at every peer: one token, and one row if it goes there. A source's rows
                    // are complete once its tally covers every token it announced; the low word is then the count.
                    int const owner = threadIdx.x;
                    uint64_t rows = 0;
#pragma unroll
                    for (int k = 0; k < TOP_K; ++k)
                    {
                        rows |= (sTarget[k] == owner) ? 1ull : 0ull;
                    }
                    asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(
                                     tallyWord(p.workspace[owner], parity, p.epRank, t % kTallyShards)),
                                 "l"((1ull << 32) | rows)
                                 : "memory");
                }
            }
#pragma unroll
            for (int i = 0; i < kItems; ++i)
            {
                int const c = first + i * blockDim.x + threadIdx.x;
                if (c >= recordChunks)
                {
                    continue;
                }
                if (chunkDirty(v[i]))
                {
                    trapWith("dispatch payload word equals the sentinel (NaN pattern); unset "
                             "TLLM_MOE_A2A_LAMPORT",
                        p.epRank, -1, t);
                }
                int const q = payloadOfChunk(p, c);
                int const off = (c - p.chunkBegin[q]) * 16;
#pragma unroll
                for (int k = 0; k < TOP_K; ++k)
                {
                    int const index = sIndex[k];
                    if (index < 0)
                    {
                        continue;
                    }
                    int const target = sTarget[k];
                    if (target == p.epRank)
                    {
                        // Own rows go straight to the receive tensor.
                        storeRowChunk(
                            recvRow(p, q, p.epRank, index), p.bytesPerToken[q], p.vecBytes[q], off, v[i]);
                    }
                    else
                    {
                        *reinterpret_cast<uint4*>(dispatchSlot(p, target, parity, p.epRank, index) + c * 16) = v[i];
                    }
                }
            }
        }
        __syncthreads(); // shared routing is reused by this block's next token
    }
    // Receive: each source's announced count and complete tally first, then the rows of each index this block owns.
    if (threadIdx.x < p.epSize)
    {
        int const source = threadIdx.x;
        int const* announced = expectedWord(self, parity, source);
        int count = 0;
        long long start = 0;
        for (bool first = true;; first = false)
        {
            int const expected = loadVolatileS32(announced);
            if (expected >= 0)
            {
                uint64_t seen[kTallyShards];
#pragma unroll
                for (int s = 0; s < kTallyShards; ++s)
                {
                    seen[s] = loadVolatileU64(tallyWord(self, parity, source, s));
                }
                bool complete = true;
                count = 0;
#pragma unroll
                for (int s = 0; s < kTallyShards; ++s)
                {
                    // Shard s holds the tokens t < expected with t % kTallyShards == s.
                    int const want = expected > s ? (expected - s + kTallyShards - 1) / kTallyShards : 0;
                    complete = complete && static_cast<int>(seen[s] >> 32) == want;
                    count += static_cast<int>(seen[s] & 0xFFFFFFFFull);
                }
                if (complete)
                {
                    break;
                }
            }
            if (first)
            {
                start = clock64();
            }
            else if (clock64() - start > p.timeoutCycles)
            {
                trapWith("timed out waiting for a dispatch tally", p.epRank, source, expected);
            }
        }
        sCount[threadIdx.x] = count;
        if (blockIdx.x == 0)
        {
            ctrl->recvCounts[threadIdx.x] = count;
        }
    }
    __syncthreads();
    for (int index = blockIdx.x; index < p.runtimeMaxTokens; index += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            int n = 0;
            for (int source = 0; source < p.epSize; ++source)
            {
                if (source != p.epRank && index < sCount[source])
                {
                    sSources[n++] = source;
                }
            }
            sNumSources = n;
        }
        if (p.expertIdPayload >= 0 && threadIdx.x < p.epSize * TOP_K)
        {
            // Fused moe_a2a_sanitize_expert_ids: unused rows carry the invalid expert id.
            int const source = threadIdx.x / TOP_K;
            if (index >= sCount[source])
            {
                reinterpret_cast<int32_t*>(recvRow(p, p.expertIdPayload, source, index))[threadIdx.x % TOP_K]
                    = p.invalidExpertId;
            }
        }
        __syncthreads();
        int const items = sNumSources * recordChunks;
        for (int first = 0; first < items; first += blockDim.x * kItems)
        {
            uint4 v[kItems];
#pragma unroll
            for (int i = 0; i < kItems; ++i)
            {
                int const item = first + i * blockDim.x + threadIdx.x;
                if (item < items)
                {
                    int const source = sSources[item / recordChunks];
                    int const c = item % recordChunks;
                    v[i] = loadVolatileChunk(dispatchSlot(p, p.epRank, parity, source, index) + c * 16);
                }
            }
#pragma unroll
            for (int i = 0; i < kItems; ++i)
            {
                int const item = first + i * blockDim.x + threadIdx.x;
                if (item >= items)
                {
                    continue;
                }
                int const source = sSources[item / recordChunks];
                int const c = item % recordChunks;
                uint8_t* slot = dispatchSlot(p, p.epRank, parity, source, index) + c * 16;
                pollChunk(v[i], slot, p.timeoutCycles, "timed out waiting for a dispatch row", p.epRank, source, index);
                int const q = payloadOfChunk(p, c);
                storeRowChunk(recvRow(p, q, source, index), p.bytesPerToken[q], p.vecBytes[q],
                    (c - p.chunkBegin[q]) * 16, v[i]);
                storeSentinelChunk(slot);
            }
        }
        __syncthreads(); // sSources is rebuilt for the next index
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------------------------------------------
// Combine
// ---------------------------------------------------------------------------------------------------------------

// Element j of a 16-byte chunk of bf16/fp16 values sits in word j / 2, low half first (little endian).
template <typename T>
__device__ __forceinline__ void unpackToFloat(uint4 const& raw, float (&out)[8]);

template <>
__device__ __forceinline__ void unpackToFloat<__nv_bfloat16>(uint4 const& raw, float (&out)[8])
{
    // Exact widening, as static_cast<float> (__bfloat162float) in the fence combine: bf16 is the upper half of fp32.
    uint32_t const w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        out[2 * i] = __uint_as_float(w[i] << 16);
        out[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
    }
}

template <>
__device__ __forceinline__ void unpackToFloat<half>(uint4 const& raw, float (&out)[8])
{
    uint32_t const w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        out[2 * i] = __half2float(__ushort_as_half(static_cast<unsigned short>(w[i] & 0xFFFFu)));
        out[2 * i + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(w[i] >> 16)));
    }
}

// Round-to-nearest-even pairs, the instruction __float22bfloat162_rn / __float22half2_rn emit (flashinfer
// vec_cast::cast in the fence combine's cast_store): the first value goes to the low half.
template <typename T>
__device__ __forceinline__ uint32_t packPair(float lo, float hi);

template <>
__device__ __forceinline__ uint32_t packPair<__nv_bfloat16>(float lo, float hi)
{
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

template <>
__device__ __forceinline__ uint32_t packPair<half>(float lo, float hi)
{
    uint32_t d;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

template <typename T>
__device__ __forceinline__ uint4 packFromFloat(float const (&in)[8])
{
    return make_uint4(packPair<T>(in[0], in[1]), packPair<T>(in[2], in[3]), packPair<T>(in[4], in[5]),
        packPair<T>(in[6], in[7]));
}

// The fence combine's reduction (moeAlltoAllKernels.cu vectorized_combine_impl): acc[TOP_K] in fp32, invalid routes
// as +0.0f, reduced into acc[0] by the same tree per TOP_K. Keep the two identical: this is what makes the
// sentinel-polled combine bit-exact against the fence combine.
template <int TOP_K>
__device__ __forceinline__ void reduceTopK(float (&acc)[TOP_K][8])
{
    constexpr int E = 8;
    if constexpr (TOP_K == 22)
    {
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[1][j];
            acc[2][j] += acc[3][j];
            acc[4][j] += acc[5][j];
            acc[6][j] += acc[7][j];
            acc[8][j] += acc[9][j];
            acc[10][j] += acc[11][j];
            acc[12][j] += acc[13][j];
            acc[14][j] += acc[15][j];
            acc[16][j] += acc[17][j];
            acc[18][j] += acc[19][j];
            acc[20][j] += acc[21][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[2][j];
            acc[4][j] += acc[6][j];
            acc[8][j] += acc[10][j];
            acc[12][j] += acc[14][j];
            acc[16][j] += acc[18][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[4][j];
            acc[8][j] += acc[12][j];
            acc[16][j] += acc[20][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[8][j];
            acc[0][j] += acc[16][j];
        }
    }
    else if constexpr (TOP_K == 16)
    {
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[1][j];
            acc[2][j] += acc[3][j];
            acc[4][j] += acc[5][j];
            acc[6][j] += acc[7][j];
            acc[8][j] += acc[9][j];
            acc[10][j] += acc[11][j];
            acc[12][j] += acc[13][j];
            acc[14][j] += acc[15][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[2][j];
            acc[4][j] += acc[6][j];
            acc[8][j] += acc[10][j];
            acc[12][j] += acc[14][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[4][j];
            acc[8][j] += acc[12][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[8][j];
        }
    }
    else if constexpr (TOP_K == 10)
    {
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[1][j];
            acc[2][j] += acc[3][j];
            acc[4][j] += acc[5][j];
            acc[6][j] += acc[7][j];
            acc[8][j] += acc[9][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[2][j];
            acc[4][j] += acc[6][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[4][j];
            acc[0][j] += acc[8][j];
        }
    }
    else if constexpr (TOP_K == 8)
    {
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[1][j];
            acc[2][j] += acc[3][j];
            acc[4][j] += acc[5][j];
            acc[6][j] += acc[7][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[2][j];
            acc[4][j] += acc[6][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[4][j];
        }
    }
    else if constexpr (TOP_K == 6)
    {
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[1][j];
            acc[2][j] += acc[3][j];
            acc[4][j] += acc[5][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[2][j];
            acc[0][j] += acc[4][j];
        }
    }
    else if constexpr (TOP_K == 4)
    {
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[1][j];
            acc[2][j] += acc[3][j];
        }
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[2][j];
        }
    }
    else if constexpr (TOP_K == 2)
    {
#pragma unroll
        for (int j = 0; j < E; ++j)
        {
            acc[0][j] += acc[1][j];
        }
    }
    else if constexpr (TOP_K == 1)
    {
    }
    else
    {
#pragma unroll
        for (int k = 1; k < TOP_K; ++k)
        {
#pragma unroll
            for (int j = 0; j < E; ++j)
            {
                acc[0][j] += acc[k][j];
            }
        }
    }
}

template <typename T, int TOP_K>
__global__ void moeA2ALamportCombineKernel(CombineParams const p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
    constexpr int kItems = kMaxItemsPerThread;
    __shared__ int sRecv[kMaxRanks];
    __shared__ int sOwners[kMaxRanks];
    __shared__ int sNumOwners;
    uint8_t* const self = p.workspace[p.epRank];
    Ctrl* const ctrl = reinterpret_cast<Ctrl*>(self);
    if (threadIdx.x < p.epSize)
    {
        sRecv[threadIdx.x] = ctrl->recvCounts[threadIdx.x];
    }
    __syncthreads();
    int const rowChunks = p.elementsPerToken * static_cast<int>(sizeof(T)) / 16;
    uint8_t const* source = static_cast<uint8_t const*>(p.source);

    // Push the expert outputs of every peer's tokens into that peer's combine slots.
    for (int index = blockIdx.x; index < p.runtimeMaxTokens; index += gridDim.x)
    {
        if (threadIdx.x == 0)
        {
            int n = 0;
            for (int owner = 0; owner < p.epSize; ++owner)
            {
                if (owner != p.epRank && index < sRecv[owner])
                {
                    sOwners[n++] = owner;
                }
            }
            sNumOwners = n;
        }
        __syncthreads();
        int const items = sNumOwners * rowChunks;
        for (int first = 0; first < items; first += blockDim.x * kItems)
        {
            uint4 v[kItems];
#pragma unroll
            for (int i = 0; i < kItems; ++i)
            {
                int const item = first + i * blockDim.x + threadIdx.x;
                if (item < items)
                {
                    int const owner = sOwners[item / rowChunks];
                    v[i] = *reinterpret_cast<uint4 const*>(source
                        + (static_cast<size_t>(owner) * p.runtimeMaxTokens + index) * p.sourceStrideBytes
                        + (item % rowChunks) * 16);
                }
            }
#pragma unroll
            for (int i = 0; i < kItems; ++i)
            {
                int const item = first + i * blockDim.x + threadIdx.x;
                if (item >= items)
                {
                    continue;
                }
                int const owner = sOwners[item / rowChunks];
                if (chunkDirty(v[i]))
                {
                    trapWith("combine payload word equals the sentinel (NaN pattern); unset "
                             "TLLM_MOE_A2A_LAMPORT",
                        p.epRank, owner, index);
                }
                *reinterpret_cast<uint4*>(
                    combineSlot(p.workspace[owner], p.layout, p.epRank, index) + (item % rowChunks) * 16)
                    = v[i];
            }
        }
        __syncthreads(); // sOwners is rebuilt for the next index
    }

    // Reduce this rank's tokens: own contributions in place, peer contributions from the combine slots.
    for (int t = blockIdx.x; t < p.localNumTokens; t += gridDim.x)
    {
        int target[TOP_K];
        int index[TOP_K];
#pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            target[k] = p.topkTargetRanks[t * TOP_K + k];
            index[k] = p.topkSendIndices[t * TOP_K + k];
        }
        uint8_t* out = static_cast<uint8_t*>(p.output) + static_cast<size_t>(t) * rowChunks * 16;
        for (int c = threadIdx.x; c < rowChunks; c += blockDim.x)
        {
            int const off = c * 16;
            uint4 raw[TOP_K];
#pragma unroll
            for (int k = 0; k < TOP_K; ++k)
            {
                if (index[k] < 0)
                {
                    continue;
                }
                if (target[k] == p.epRank)
                {
                    raw[k] = *reinterpret_cast<uint4 const*>(source
                        + (static_cast<size_t>(target[k]) * p.runtimeMaxTokens + index[k]) * p.sourceStrideBytes
                        + off);
                }
                else
                {
                    raw[k] = loadVolatileChunk(combineSlot(self, p.layout, target[k], index[k]) + off);
                }
            }
#pragma unroll
            for (int k = 0; k < TOP_K; ++k)
            {
                if (index[k] < 0 || target[k] == p.epRank)
                {
                    continue;
                }
                uint8_t* slot = combineSlot(self, p.layout, target[k], index[k]) + off;
                pollChunk(raw[k], slot, p.timeoutCycles, "timed out waiting for a combine row", p.epRank, target[k],
                    index[k]);
                storeSentinelChunk(slot);
            }
            float acc[TOP_K][8];
#pragma unroll
            for (int k = 0; k < TOP_K; ++k)
            {
                if (index[k] < 0)
                {
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        acc[k][j] = 0.0f;
                    }
                }
                else
                {
                    unpackToFloat<T>(raw[k], acc[k]);
                }
            }
            reduceTopK<TOP_K>(acc);
            *reinterpret_cast<uint4*>(out + off) = packFromFloat<T>(acc[0]);
        }
    }

    // Reset the dispatch-side state for the next round: this kernel follows the dispatch on the stream, and every
    // peer's tally for this parity is complete (the dispatch waited for it).
    if (blockIdx.x == 0)
    {
        if (threadIdx.x < p.epSize)
        {
            uint32_t const parity = ctrl->round & 1u;
            storeVolatileS32(expectedWord(self, parity, threadIdx.x), -1);
#pragma unroll
            for (int s = 0; s < kTallyShards; ++s)
            {
                *tallyWord(self, parity, threadIdx.x, s) = 0ull;
            }
            ctrl->sendCounters[threadIdx.x] = 0;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            ctrl->round = ctrl->round + 1u;
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ---------------------------------------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------------------------------------

template <typename KernelFn>
inline int residentGridCap(KernelFn kernel, int blockSize)
{
    int device = 0;
    int smCount = 0;
    int blocksPerSm = 0;
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device) != cudaSuccess
        || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, blockSize, 0) != cudaSuccess)
    {
        return 0;
    }
    return smCount * blocksPerSm;
}

// Resident capacity per (kernel, block size), computed once: the grid must never exceed it.
template <auto Kernel>
inline int cachedGridCap(int blockSize)
{
    static int cap[33] = {};
    int const slot = blockSize / 32;
    if (slot <= 0 || slot > 32)
    {
        return 0;
    }
    if (cap[slot] == 0)
    {
        cap[slot] = residentGridCap(Kernel, blockSize);
    }
    return cap[slot];
}

template <typename KernelFn, typename Params>
inline cudaError_t launchMaybePdl(KernelFn kernel, int grid, int block, cudaStream_t stream, bool pdl, Params const& p)
{
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid);
    config.blockDim = dim3(block);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, kernel, p);
}

inline int gridFor(int localNumTokens, int runtimeMaxTokens, int cap)
{
    int grid = localNumTokens > runtimeMaxTokens ? localNumTokens : runtimeMaxTokens;
    return grid < 1 ? 1 : (grid > cap ? cap : grid);
}

// The grid covers max(local tokens, runtime max tokens) and never exceeds what can be resident at once.
template <int TOP_K>
inline cudaError_t launchDispatch(DispatchParams const& p, cudaStream_t stream, bool pdl)
{
    if (p.epSize > kMaxRanks || p.epSize * TOP_K > 1024 || p.numPayloads < 1 || p.numPayloads > kMaxPayloads)
    {
        return cudaErrorInvalidValue;
    }
    // One thread per record chunk, and enough threads to fill the invalid ids of a row index for every source.
    int const chunks = p.chunkBegin[p.numPayloads];
    int const block = blockSizeForChunks(chunks > p.epSize * TOP_K ? chunks : p.epSize * TOP_K);
    int const cap = cachedGridCap<moeA2ALamportDispatchKernel<TOP_K>>(block);
    if (cap <= 0)
    {
        return cudaErrorInvalidConfiguration;
    }
    return launchMaybePdl(
        moeA2ALamportDispatchKernel<TOP_K>, gridFor(p.localNumTokens, p.runtimeMaxTokens, cap), block, stream, pdl, p);
}

template <typename T, int TOP_K>
inline cudaError_t launchCombine(CombineParams const& p, cudaStream_t stream, bool pdl)
{
    int const rowBytes = p.elementsPerToken * static_cast<int>(sizeof(T));
    if (p.epSize > kMaxRanks || rowBytes % 16 != 0 || rowBytes > p.layout.combineSlotBytes)
    {
        return cudaErrorInvalidValue;
    }
    int const block = blockSizeForChunks(rowBytes / 16);
    int const cap = cachedGridCap<moeA2ALamportCombineKernel<T, TOP_K>>(block);
    if (cap <= 0)
    {
        return cudaErrorInvalidConfiguration;
    }
    return launchMaybePdl(moeA2ALamportCombineKernel<T, TOP_K>, gridFor(p.localNumTokens, p.runtimeMaxTokens, cap),
        block, stream, pdl, p);
}

#endif // defined(__CUDACC__)

} // namespace moe_a2a_lamport

#endif // TRTLLM_MOE_ALLTOALL_LAMPORT_CUH
