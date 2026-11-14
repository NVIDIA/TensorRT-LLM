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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/draftTail.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace draft_tail
{

namespace
{

// ---------------------------------------------------------------------------
// Candidate keys
// ---------------------------------------------------------------------------

// Monotone map of BF16 bits to 16 bits: -0 and +0 compare equal, NaN of either
// sign ranks above +inf (as the native torch.max reduction does).
__device__ __forceinline__ uint32_t orderedBf16(uint32_t bits)
{
    uint32_t const mag = bits & 0x7FFFu;
    if (mag > 0x7F80u)
    {
        return 0xFFFFu;
    }
    if (mag == 0u)
    {
        bits = 0u;
    }
    return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
}

// Larger key == larger value, then smaller global id. Never zero.
__device__ __forceinline__ uint64_t makeKey(uint32_t bf16Bits, int64_t globalId)
{
    return (static_cast<uint64_t>(orderedBf16(bf16Bits)) << 32)
        | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(globalId));
}

__device__ __forceinline__ int32_t keyToToken(uint64_t key)
{
    return static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(key & 0xFFFFFFFFull));
}

__device__ __forceinline__ uint64_t umax64(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

__device__ __forceinline__ uint64_t warpMax64(uint64_t v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        v = umax64(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void griddepWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ uint64_t ldRelaxedSys(uint64_t const* addr)
{
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(addr) : "memory");
    return v;
}

__device__ __forceinline__ void stRelaxedSys(uint64_t* addr, uint64_t v)
{
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(addr), "l"(v) : "memory");
}

__device__ __forceinline__ void mcStoreRelaxedSys(uint64_t* mcAddr, uint64_t v)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("multimem.st.relaxed.sys.global.b64 [%0], %1;" ::"l"(mcAddr), "l"(v) : "memory");
#endif
}

__device__ __forceinline__ uint64_t evictFirstPolicy()
{
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ uint64_t evictLastPolicy()
{
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

// One 32-byte chunk = 16 BF16 values.
struct alignas(16) Chunk
{
    uint64_t v[4];
};

enum LoadKind : int
{
    kLoad32Hint = 0, // 32-byte load, L2 evict_first policy
    kLoad32 = 1,     // 32-byte load
    kLoad16x2Pf = 2, // two 16-byte loads with 256-byte L2 prefetch
};

template <int kKind>
__device__ __forceinline__ void loadChunk(Chunk& c, void const* ptr, uint64_t policy)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
    // 32-byte global loads need SM100+.
    __trap();
#else
    if constexpr (kKind == kLoad32Hint)
    {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.b64 {%0, %1, %2, %3}, [%4], %5;"
                     : "=l"(c.v[0]), "=l"(c.v[1]), "=l"(c.v[2]), "=l"(c.v[3])
                     : "l"(ptr), "l"(policy));
    }
    else if constexpr (kKind == kLoad32)
    {
        asm volatile("ld.global.nc.L1::no_allocate.v4.b64 {%0, %1, %2, %3}, [%4];"
                     : "=l"(c.v[0]), "=l"(c.v[1]), "=l"(c.v[2]), "=l"(c.v[3])
                     : "l"(ptr));
    }
    else
    {
        int4 a;
        int4 b;
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w)
                     : "l"(ptr));
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w)
                     : "l"(static_cast<char const*>(ptr) + 16));
        c.v[0] = (static_cast<uint64_t>(static_cast<uint32_t>(a.y)) << 32) | static_cast<uint32_t>(a.x);
        c.v[1] = (static_cast<uint64_t>(static_cast<uint32_t>(a.w)) << 32) | static_cast<uint32_t>(a.z);
        c.v[2] = (static_cast<uint64_t>(static_cast<uint32_t>(b.y)) << 32) | static_cast<uint32_t>(b.x);
        c.v[3] = (static_cast<uint64_t>(static_cast<uint32_t>(b.w)) << 32) | static_cast<uint32_t>(b.z);
    }
#endif
}

// acc += dot(w, h) over 16 BF16 pairs, in a fixed order.
__device__ __forceinline__ float dotChunk(Chunk const& w, Chunk const& h, float acc)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        uint32_t const wl = static_cast<uint32_t>(w.v[i]);
        uint32_t const wh = static_cast<uint32_t>(w.v[i] >> 32);
        uint32_t const hl = static_cast<uint32_t>(h.v[i]);
        uint32_t const hh = static_cast<uint32_t>(h.v[i] >> 32);
        acc = fmaf(__uint_as_float(wl << 16), __uint_as_float(hl << 16), acc);
        acc = fmaf(__uint_as_float(wl & 0xFFFF0000u), __uint_as_float(hl & 0xFFFF0000u), acc);
        acc = fmaf(__uint_as_float(wh << 16), __uint_as_float(hh << 16), acc);
        acc = fmaf(__uint_as_float(wh & 0xFFFF0000u), __uint_as_float(hh & 0xFFFF0000u), acc);
    }
    return acc;
}

constexpr int kHidden6144ChunksDev = 6144 / 512; // 32-byte chunks per lane per row

__device__ __forceinline__ uint32_t* arrivalCounter(uint64_t* scratch)
{
    return reinterpret_cast<uint32_t*>(scratch + kMaxRows);
}

__device__ __forceinline__ uint32_t* epochCounter(uint64_t* scratch)
{
    return reinterpret_cast<uint32_t*>(scratch + kMaxRows) + 1;
}

// ---------------------------------------------------------------------------
// Cross-rank exchange (one warp)
//
// Slot s of the exchange buffer holds kMaxRanks x kMaxRows keys; rank q writes
// only its own row block, every rank polls its local copy until all tpSize
// entries of each row are non-zero. Epoch e uses slot e % 3. Before publishing
// epoch e a rank clears slot (e + 1) % 3, whose last use was epoch e - 2: every
// key of that epoch was already observed here, and no peer can publish epoch
// e + 1 before it has observed this rank's epoch-e key, which the release fence
// orders after the clear (the peer's own fence before its next publish is the
// matching acquire). All ranks run the same sequence of epochs.
// ---------------------------------------------------------------------------
__device__ void exchangeAndWrite(
    DraftTailParams const& p, uint64_t const* localKeys /* smem [rows] */, uint64_t (*gathered)[kMaxRows] /* smem */)
{
    int const lane = threadIdx.x & 31;
    int const rows = p.rows;
    bool const exchange = p.mode != ExchangeMode::kNone && p.tpSize > 1;
    uint64_t result = 0;
    if (!exchange)
    {
        if (lane < rows)
        {
            result = localKeys[lane];
        }
    }
    else
    {
        uint32_t* epochPtr = epochCounter(p.scratch);
        uint32_t const epoch = *reinterpret_cast<uint32_t volatile*>(epochPtr);
        int const cur = static_cast<int>(epoch % kNumSlots);
        int const nxt = static_cast<int>((epoch + 1) % kNumSlots);
        for (int i = lane; i < kSlotWords; i += 32)
        {
            p.ucSlotsLocal[nxt * kSlotWords + i] = 0ull;
        }
        // Release: the clear above is ordered before this rank's publish below;
        // a peer's next write to the cleared slot is causally after that publish.
        asm volatile("fence.acq_rel.sys;" ::: "memory");
        __syncwarp();
        int const mine = cur * kSlotWords + p.tpRank * kMaxRows;
        if (p.mode == ExchangeMode::kMulticast)
        {
            if (lane < rows)
            {
                mcStoreRelaxedSys(p.mcSlots + mine + lane, localKeys[lane]);
            }
        }
        else
        {
            for (int i = lane; i < p.tpSize * rows; i += 32)
            {
                int const q = i / rows;
                int const r = i - q * rows;
                stRelaxedSys(p.ucSlotsPeers[q] + mine + r, localKeys[r]);
            }
        }
        long long const start = clock64();
        for (int i = lane; i < p.tpSize * rows; i += 32)
        {
            int const q = i / rows;
            int const r = i - q * rows;
            uint64_t const* addr = p.ucSlotsLocal + cur * kSlotWords + q * kMaxRows + r;
            uint64_t v = ldRelaxedSys(addr);
            while (v == 0ull)
            {
                if (p.timeoutCycles > 0 && clock64() - start > p.timeoutCycles)
                {
                    // A peer never published: the ranks disagree on the
                    // collective sequence. Record it and abort loudly.
                    atomicExch(reinterpret_cast<unsigned long long*>(p.scratch + kMaxRows + 1),
                        0x5733000000000000ull | (static_cast<unsigned long long>(epoch) << 8)
                            | static_cast<unsigned long long>(q));
                    __threadfence_system();
                    __trap();
                }
                v = ldRelaxedSys(addr);
            }
            gathered[q][r] = v;
        }
        __syncwarp();
        if (lane < rows)
        {
            for (int q = 0; q < p.tpSize; ++q)
            {
                result = umax64(result, gathered[q][lane]);
            }
        }
        if (lane == 0)
        {
            *epochPtr = epoch + 1;
        }
    }
    if (lane < rows)
    {
        p.tokens[lane] = keyToToken(result);
        if (p.keysOut != nullptr)
        {
            p.keysOut[lane] = result;
        }
    }
}

// ---------------------------------------------------------------------------
// Fused kernel: GEMV + rank-local max + exchange
//
// CTA c owns a contiguous block of vocab rows; warp w of it takes rows
// begin + w, begin + w + kWarps, ... Each row is split into kSplit units of
// kUnit 32-byte chunks per lane, double-buffered in registers so a warp always
// has the next unit in flight while it computes the current one. The first
// unit is requested before griddepcontrol.wait, i.e. while the producer of
// the hidden rows may still be running under programmatic dependent launch.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Direct exchange: every CTA publishes its partial key for each row into
// entry [slot][row][rank][cta] of every rank's buffer; CTA 0 of each rank polls
// all tpSize x numCtas entries of each row, writes the tokens, then clears
// exactly the entries it polled and advances the epoch. A slot's entries are
// written again only three epochs later, i.e. after this rank's next two
// publishes, which follow this launch; so no peer can write a cleared entry
// before the clear has completed. CTAs without vocab rows publish key 1, below
// every real key.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool isDirectMode(ExchangeMode m)
{
    return m == ExchangeMode::kMulticastDirect || m == ExchangeMode::kUnicastDirect;
}

__device__ __forceinline__ size_t directEntry(int slot, int row, int rank, int cta)
{
    return kDirectOffsetWords + static_cast<size_t>(slot) * kDirectSlotWords
        + (static_cast<size_t>(row) * kMaxRanks + rank) * kMaxDirectCtas + cta;
}

// All threads of the CTA call this once ctaKeys[0..rows) (shared) is final.
__device__ void directExchange(DraftTailParams const& p, uint64_t const* ctaKeys, int ctaIdx, int numCtas)
{
    __shared__ uint32_t sEpoch;
    __shared__ uint64_t red[32][kMaxRows];
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const rows = p.rows;
    if (tid == 0)
    {
        sEpoch = *reinterpret_cast<uint32_t volatile*>(epochCounter(p.scratch));
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    int const cur = static_cast<int>(epoch % kNumSlots);
    if (p.mode == ExchangeMode::kMulticastDirect)
    {
        if (tid < rows)
        {
            uint64_t const key = ctaKeys[tid] != 0ull ? ctaKeys[tid] : 1ull;
            mcStoreRelaxedSys(p.mcSlots + directEntry(cur, tid, p.tpRank, ctaIdx), key);
        }
    }
    else
    {
        for (int i = tid; i < p.tpSize * rows; i += blockDim.x)
        {
            int const q = i / rows;
            int const r = i - q * rows;
            uint64_t const key = ctaKeys[r] != 0ull ? ctaKeys[r] : 1ull;
            stRelaxedSys(p.ucSlotsPeers[q] + directEntry(cur, r, p.tpRank, ctaIdx), key);
        }
    }
    if (ctaIdx != 0)
    {
        return;
    }
    long long const start = clock64();
    int const perRow = p.tpSize * numCtas;
    // Each thread issues up to kPoll independent loads per round and re-polls
    // only the entries still empty, so a row costs about one memory round
    // trip after the last entry lands, however many CTAs published.
    constexpr int kPoll = 32;
    for (int r = 0; r < rows; ++r)
    {
        uint64_t m = 0ull;
        for (int base = tid; base < perRow; base += kPoll * blockDim.x)
        {
            uint64_t v[kPoll];
#pragma unroll
            for (int b = 0; b < kPoll; ++b)
            {
                int const i = base + b * blockDim.x;
                v[b] = i < perRow ? ldRelaxedSys(p.ucSlotsLocal + directEntry(cur, r, i / numCtas, i % numCtas)) : 1ull;
            }
            bool done = false;
            while (!done)
            {
                done = true;
#pragma unroll
                for (int b = 0; b < kPoll; ++b)
                {
                    if (v[b] == 0ull)
                    {
                        int const i = base + b * blockDim.x;
                        done = false;
                        v[b] = ldRelaxedSys(p.ucSlotsLocal + directEntry(cur, r, i / numCtas, i % numCtas));
                    }
                }
                if (!done && p.timeoutCycles > 0 && clock64() - start > p.timeoutCycles)
                {
                    atomicExch(reinterpret_cast<unsigned long long*>(p.scratch + kMaxRows + 1),
                        0x5734000000000000ull | (static_cast<unsigned long long>(epoch) << 16)
                            | static_cast<unsigned long long>(base));
                    __threadfence_system();
                    __trap();
                }
            }
#pragma unroll
            for (int b = 0; b < kPoll; ++b)
            {
                m = umax64(m, v[b]);
            }
        }
        m = warpMax64(m);
        if (lane == 0)
        {
            red[warp][r] = m;
        }
    }
    __syncthreads();
    if (tid < rows)
    {
        uint64_t result = 0ull;
        int const numWarps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < numWarps; ++w)
        {
            result = umax64(result, red[w][tid]);
        }
        p.tokens[tid] = keyToToken(result);
        if (p.keysOut != nullptr)
        {
            p.keysOut[tid] = result;
        }
    }
    for (int i = tid; i < rows * perRow; i += blockDim.x)
    {
        int const r = i / perRow;
        int const j = i - r * perRow;
        int const q = j / numCtas;
        int const c = j - q * numCtas;
        p.ucSlotsLocal[directEntry(cur, r, q, c)] = 0ull;
    }
    if (tid == 0)
    {
        *epochCounter(p.scratch) = epoch + 1;
    }
}

// Per-CTA epilogue of the fused kernels: CTA max of the warps' keys, one
// atomicMax per row into rank-local scratch, then the last CTA to arrive takes
// (and resets) the rank-local keys and runs the exchange.
template <int R, int kWarps>
__device__ __forceinline__ void ctaFinish(DraftTailParams const& p, uint64_t const (&best)[R])
{
    __shared__ uint64_t warpBest[kWarps][R];
    __shared__ uint64_t ctaKeys[kMaxRows];
    __shared__ uint64_t gathered[kMaxRanks][kMaxRows];
    __shared__ int isLast;
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    if (lane == 0)
    {
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            warpBest[warp][r] = best[r];
        }
    }
    __syncthreads();
    if (isDirectMode(p.mode))
    {
        if (threadIdx.x < R)
        {
            uint64_t m = 0ull;
#pragma unroll
            for (int w = 0; w < kWarps; ++w)
            {
                m = umax64(m, warpBest[w][threadIdx.x]);
            }
            ctaKeys[threadIdx.x] = m;
        }
        __syncthreads();
        directExchange(p, ctaKeys, blockIdx.x, gridDim.x);
        return;
    }
    if (threadIdx.x < R)
    {
        uint64_t m = 0ull;
#pragma unroll
        for (int w = 0; w < kWarps; ++w)
        {
            m = umax64(m, warpBest[w][threadIdx.x]);
        }
        if (m != 0ull)
        {
            atomicMax(reinterpret_cast<unsigned long long*>(p.scratch + threadIdx.x), m);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        __threadfence();
        uint32_t const prev = atomicAdd(arrivalCounter(p.scratch), 1u);
        isLast = prev == gridDim.x - 1;
    }
    __syncthreads();
    if (!isLast || warp != 0)
    {
        return;
    }
    __threadfence();
    if (lane < R)
    {
        ctaKeys[lane] = atomicExch(reinterpret_cast<unsigned long long*>(p.scratch + lane), 0ull);
    }
    if (lane == 0)
    {
        atomicExch(arrivalCounter(p.scratch), 0u);
    }
    __syncwarp();
    exchangeAndWrite(p, ctaKeys, gathered);
}

template <int kUnit, int kLoad>
__device__ __forceinline__ void loadUnit(Chunk (&buf)[kUnit], char const* base, uint64_t policy)
{
#pragma unroll
    for (int j = 0; j < kUnit; ++j)
    {
        loadChunk<kLoad>(buf[j], base + j * 32 * 32, policy);
    }
}

// Byte address of unit u of this lane: row firstRow + (u / kSplit) * kWarps,
// part u % kSplit; chunk c of a row lives at bytes [32 c, 32 c + 32) and lane l
// owns chunks part * kUnit * 32 + j * 32 + l.
template <int kWarps, int kChunks, int kSplit>
__device__ __forceinline__ char const* unitAddress(__nv_bfloat16 const* weight, int firstRow, int u, int lane)
{
    constexpr int kUnit = kChunks / kSplit;
    int const rowIdx = u / kSplit;
    int const part = u - rowIdx * kSplit;
    int64_t const row = firstRow + static_cast<int64_t>(rowIdx) * kWarps;
    return reinterpret_cast<char const*>(weight + row * (kChunks * 32 * 16)) + (part * kUnit * 32 + lane) * 32;
}

template <int R, int kWarps, int kChunks, int kSplit>
__device__ __forceinline__ void computeUnit(Chunk const (&buf)[kChunks / kSplit], Chunk const* smemH, int u,
    int firstRow, int lane, int64_t vocabOffset, float (&acc)[R], uint64_t (&best)[R])
{
    constexpr int kUnit = kChunks / kSplit;
    int const rowIdx = u / kSplit;
    int const part = u - rowIdx * kSplit;
#pragma unroll
    for (int j = 0; j < kUnit; ++j)
    {
        int const c = (part * kUnit + j) * 32 + lane;
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            acc[r] = dotChunk(buf[j], smemH[r * kChunks * 32 + c], acc[r]);
        }
    }
    if (part == kSplit - 1)
    {
        int64_t const row = firstRow + static_cast<int64_t>(rowIdx) * kWarps;
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            float s = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
            }
            uint32_t const bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(s)));
            best[r] = umax64(best[r], makeKey(bits, vocabOffset + row));
            acc[r] = 0.f;
        }
    }
}

template <int R, int kWarps, int kChunks, int kSplit, int kLoad, int kMinBlocks>
__global__ void __launch_bounds__(kWarps * 32, kMinBlocks) fusedDraftTailKernel(DraftTailParams p)
{
    constexpr int kUnit = kChunks / kSplit;
    static_assert(kChunks % kSplit == 0, "row must split evenly");

    extern __shared__ uint4 smemDyn[];
    Chunk* smemH = reinterpret_cast<Chunk*>(smemDyn); // [R][kChunks * 32]

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const numRows = p.vocabLocal;
    int const rowsPerCta = (numRows + gridDim.x - 1) / gridDim.x;
    int const rowBegin = blockIdx.x * rowsPerCta;
    int const rowEnd = min(numRows, rowBegin + rowsPerCta);
    int const firstRow = rowBegin + warp;
    int const myRows = firstRow < rowEnd ? (rowEnd - firstRow + kWarps - 1) / kWarps : 0;
    int const numUnits = myRows * kSplit;

    uint64_t const policyFirst = kLoad == kLoad32Hint ? evictFirstPolicy() : 0ull;
    uint64_t const policyLast = kLoad == kLoad32Hint ? evictLastPolicy() : 0ull;
    int const keepRows = static_cast<int>((static_cast<int64_t>(myRows) * p.keepPermille) / 1000);
    __nv_bfloat16 const* weight = p.weight;

    Chunk bufA[kUnit];
    Chunk bufB[kUnit];
    if (numUnits > 0)
    {
        loadUnit<kUnit, kLoad>(bufA, unitAddress<kWarps, kChunks, kSplit>(weight, firstRow, 0, lane),
            keepRows > 0 ? policyLast : policyFirst);
    }

    griddepWait();

    {
        int const total = R * kChunks * 32;
        for (int i = threadIdx.x; i < total; i += blockDim.x)
        {
            int const r = i / (kChunks * 32);
            int const c = i - r * (kChunks * 32);
            uint4 const* src = reinterpret_cast<uint4 const*>(p.hidden + r * p.hiddenStride) + 2 * c;
            uint4 const lo = src[0];
            uint4 const hi = src[1];
            Chunk ch;
            ch.v[0] = (static_cast<uint64_t>(lo.y) << 32) | lo.x;
            ch.v[1] = (static_cast<uint64_t>(lo.w) << 32) | lo.z;
            ch.v[2] = (static_cast<uint64_t>(hi.y) << 32) | hi.x;
            ch.v[3] = (static_cast<uint64_t>(hi.w) << 32) | hi.z;
            smemH[i] = ch;
        }
    }
    __syncthreads();

    float acc[R];
    uint64_t best[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        acc[r] = 0.f;
        best[r] = 0ull;
    }

    for (int u = 0; u < numUnits; u += 2)
    {
        if (u + 1 < numUnits)
        {
            loadUnit<kUnit, kLoad>(bufB, unitAddress<kWarps, kChunks, kSplit>(weight, firstRow, u + 1, lane),
                (u + 1) / kSplit < keepRows ? policyLast : policyFirst);
        }
        computeUnit<R, kWarps, kChunks, kSplit>(bufA, smemH, u, firstRow, lane, p.vocabOffset, acc, best);
        if (u + 1 < numUnits)
        {
            if (u + 2 < numUnits)
            {
                loadUnit<kUnit, kLoad>(bufA, unitAddress<kWarps, kChunks, kSplit>(weight, firstRow, u + 2, lane),
                    (u + 2) / kSplit < keepRows ? policyLast : policyFirst);
            }
            computeUnit<R, kWarps, kChunks, kSplit>(bufB, smemH, u + 1, firstRow, lane, p.vocabOffset, acc, best);
        }
    }

    ctaFinish<R, kWarps>(p, best);
}

// ---------------------------------------------------------------------------
// Fused kernel, bulk-copy flavour: each warp streams its rows through its own
// ring of kSlots shared-memory row buffers filled by cp.async.bulk (one 12 KB
// copy per row, completion on a per-slot mbarrier), so the bytes in flight do
// not depend on registers. Lane l reads chunks l + 32 j of a row in the same
// order as the register kernel, so both produce bitwise-identical keys.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smemAddr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t phase)
{
    uint32_t const addr = smemAddr(bar);
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "DRAFT_TAIL_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra DRAFT_TAIL_WAIT_%=;\n"
        "}\n" ::"r"(addr),
        "r"(phase)
        : "memory");
}

__device__ __forceinline__ void bulkLoad(void* dst, void const* src, uint32_t bytes, uint64_t* bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
            "r"(smemAddr(dst)),
        "l"(src), "r"(bytes), "r"(smemAddr(bar)), "l"(policy)
        : "memory");
}

template <int R, int kWarps, int kSlots>
__global__ void __launch_bounds__(kWarps * 32, 1) fusedBulkDraftTailKernel(DraftTailParams p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
    // Bulk copies and transaction mbarriers need SM90+.
    __trap();
#else
    constexpr int kChunks = kHidden6144ChunksDev;
    constexpr int kRowBytes = kChunks * 32 * 32;
    extern __shared__ uint4 smemDyn[];
    Chunk* smemH = reinterpret_cast<Chunk*>(smemDyn); // [R][kChunks * 32]
    Chunk* ring = smemH + R * kChunks * 32;           // [kWarps][kSlots][kChunks * 32]
    __shared__ __align__(8) uint64_t bars[kWarps][kSlots];

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const numRows = p.vocabLocal;
    int const rowsPerCta = (numRows + gridDim.x - 1) / gridDim.x;
    int const rowBegin = blockIdx.x * rowsPerCta;
    int const rowEnd = min(numRows, rowBegin + rowsPerCta);
    int const firstRow = rowBegin + warp;
    int const myRows = firstRow < rowEnd ? (rowEnd - firstRow + kWarps - 1) / kWarps : 0;
    int const keepRows = static_cast<int>((static_cast<int64_t>(myRows) * p.keepPermille) / 1000);
    Chunk* myRing = ring + warp * kSlots * kChunks * 32;

    uint64_t const policyFirst = evictFirstPolicy();
    uint64_t const policyLast = evictLastPolicy();
    auto rowPtr = [&](int i) -> void const*
    { return p.weight + (static_cast<int64_t>(firstRow) + static_cast<int64_t>(i) * kWarps) * (kChunks * 32 * 16); };

    if (lane == 0)
    {
#pragma unroll
        for (int s = 0; s < kSlots; ++s)
        {
            mbarInit(&bars[warp][s], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll
        for (int s = 0; s < kSlots; ++s)
        {
            if (s < myRows)
            {
                mbarExpectTx(&bars[warp][s], kRowBytes);
                bulkLoad(myRing + s * kChunks * 32, rowPtr(s), kRowBytes, &bars[warp][s],
                    s < keepRows ? policyLast : policyFirst);
            }
        }
    }

    griddepWait();
    {
        int const total = R * kChunks * 32;
        for (int i = threadIdx.x; i < total; i += blockDim.x)
        {
            int const r = i / (kChunks * 32);
            int const c = i - r * (kChunks * 32);
            uint4 const* src = reinterpret_cast<uint4 const*>(p.hidden + r * p.hiddenStride) + 2 * c;
            uint4 const lo = src[0];
            uint4 const hi = src[1];
            Chunk ch;
            ch.v[0] = (static_cast<uint64_t>(lo.y) << 32) | lo.x;
            ch.v[1] = (static_cast<uint64_t>(lo.w) << 32) | lo.z;
            ch.v[2] = (static_cast<uint64_t>(hi.y) << 32) | hi.x;
            ch.v[3] = (static_cast<uint64_t>(hi.w) << 32) | hi.z;
            smemH[i] = ch;
        }
    }
    __syncthreads();

    uint64_t best[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        best[r] = 0ull;
    }
    for (int i = 0; i < myRows; ++i)
    {
        int const slot = i % kSlots;
        mbarWait(&bars[warp][slot], static_cast<uint32_t>((i / kSlots) & 1));
        Chunk const* row = myRing + slot * kChunks * 32;
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            acc[r] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < kChunks; ++j)
        {
            int const c = j * 32 + lane;
            Chunk const w = row[c];
#pragma unroll
            for (int r = 0; r < R; ++r)
            {
                acc[r] = dotChunk(w, smemH[r * kChunks * 32 + c], acc[r]);
            }
        }
        __syncwarp();
        if (lane == 0 && i + kSlots < myRows)
        {
            mbarExpectTx(&bars[warp][slot], kRowBytes);
            bulkLoad(myRing + slot * kChunks * 32, rowPtr(i + kSlots), kRowBytes, &bars[warp][slot],
                i + kSlots < keepRows ? policyLast : policyFirst);
        }
        int64_t const rowId = firstRow + static_cast<int64_t>(i) * kWarps;
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            float sum = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
            {
                sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
            }
            uint32_t const bits = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(sum)));
            best[r] = umax64(best[r], makeKey(bits, p.vocabOffset + rowId));
        }
    }
    ctaFinish<R, kWarps>(p, best);
#endif
}

// ---------------------------------------------------------------------------
// Fused kernel, tensor-core flavour: each warp owns 16-row tiles of the vocab
// shard and computes them with mma.sync m16n8k16 (BF16 in, FP32 accumulate),
// the R <= 8 hidden rows being the n columns. Lane (g = lane / 4, t = lane % 4)
// loads 32 contiguous bytes (16 values) of rows g and g + 8 per 64-value K
// block; inside a block the K order is permuted consistently for A and B
// (values 16t + 4s .. 16t + 4s + 3 feed k-step s), so a lane's two 16-byte
// halves map directly onto the A/B fragments without shuffles. This replaces
// the CUDA-core unpack + FMA loop (instruction bound) with 4 HMMA per 2 KB
// of weight per warp.
//
// Native-order flavours (MmaOrder below): k-step q of a 64-value block takes
// the 16 consecutive values 16q .. 16q + 15, so every MMA sums the same K
// values, in the same k-step order, as a GEMM that walks K in 16-value steps.
// They differ only in how the values reach the lanes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

constexpr int kMmaBlockElems = 64;                // K values per lane-group block (4 lanes x 16)
constexpr int kMmaBlocks = 6144 / kMmaBlockElems; // 96
constexpr int kMmaHStride = 6144 + 16;            // padded smem row (elements) against bank conflicts

// Which K values lane t (= lane % 4) of a quad feeds into the 4 mma k-steps
// q = 0..3 of each 64-value block, and how they are loaded. Every lane feeds
// 4 values (one 8-byte piece) per k-step and row.
enum MmaOrder : int
{
    // Lane t: values 16 t .. 16 t + 15 (one 32-byte load), piece q into k-step
    // q; each k-step sums a K subset spread over the block (not native order).
    kOrderPermuted = 0,
    // Native orders: k-step q sums exactly values 16 q .. 16 q + 15.
    // Lane t loads piece t of every k-step: four 8-byte loads.
    kOrderNat8 = 1,
    // Lane t loads k-step t whole (one 32-byte load); three in-quad shuffles
    // hand piece (t - q) mod 4 of k-step q to lane t.
    kOrderRot32 = 2,
    // Lane t loads half (t >> 1) of k-steps 2 (t & 1) and 2 (t & 1) + 1 (two
    // 16-byte loads), keeps the first piece of each half and swaps the second
    // with lane t ^ 1.
    kOrderPair16 = 3,
};

// Byte offset of lane t's first load inside a 128-byte block row.
template <int kOrder>
__device__ __forceinline__ int mmaLaneOffset(int t)
{
    if constexpr (kOrder == kOrderNat8)
    {
        return 8 * t;
    }
    else if constexpr (kOrder == kOrderPair16)
    {
        return 64 * (t & 1) + 16 * (t >> 1);
    }
    else
    {
        return 32 * t;
    }
}

// Rot32 / Pair16: which 8-byte piece (0..3) of k-step q lane t feeds.
template <int kOrder>
__device__ __forceinline__ int mmaNativePiece(int t, int q)
{
    if constexpr (kOrder == kOrderRot32)
    {
        return (t - q) & 3;
    }
    else
    {
        return q < 2 ? t : (t ^ 1);
    }
}

template <int kLoad>
__device__ __forceinline__ void load16(uint64_t& a, uint64_t& b, void const* ptr, uint64_t policy)
{
    if constexpr (kLoad == kLoad32Hint)
    {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.b64 {%0, %1}, [%2], %3;"
                     : "=l"(a), "=l"(b)
                     : "l"(ptr), "l"(policy));
    }
    else
    {
        asm volatile("ld.global.nc.L1::no_allocate.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(ptr));
    }
}

template <int kLoad>
__device__ __forceinline__ uint64_t load8(void const* ptr, uint64_t policy)
{
    uint64_t v;
    if constexpr (kLoad == kLoad32Hint)
    {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.b64 %0, [%1], %2;" : "=l"(v) : "l"(ptr), "l"(policy));
    }
    else
    {
        asm volatile("ld.global.nc.L1::no_allocate.b64 %0, [%1];" : "=l"(v) : "l"(ptr));
    }
    return v;
}

// row0 / row1 already include the lane offset (mmaLaneOffset).
template <int kStage, int kLoad, int kOrder>
__device__ __forceinline__ void mmaLoadStage(
    Chunk (&x)[kStage], Chunk (&y)[kStage], char const* row0, char const* row1, int block, uint64_t policy)
{
#pragma unroll
    for (int j = 0; j < kStage; ++j)
    {
        int const off = (block + j) * kMmaBlockElems * 2;
        if constexpr (kOrder == kOrderNat8)
        {
#pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                x[j].v[q] = load8<kLoad>(row0 + off + 32 * q, policy);
                y[j].v[q] = load8<kLoad>(row1 + off + 32 * q, policy);
            }
        }
        else if constexpr (kOrder == kOrderPair16)
        {
            // v[0], v[1]: k-step 2 (t & 1); v[2], v[3]: k-step 2 (t & 1) + 1.
            load16<kLoad>(x[j].v[0], x[j].v[1], row0 + off, policy);
            load16<kLoad>(x[j].v[2], x[j].v[3], row0 + off + 32, policy);
            load16<kLoad>(y[j].v[0], y[j].v[1], row1 + off, policy);
            load16<kLoad>(y[j].v[2], y[j].v[3], row1 + off + 32, policy);
        }
        else
        {
            loadChunk<kLoad>(x[j], row0 + off, policy);
            loadChunk<kLoad>(y[j], row1 + off, policy);
        }
    }
}

// a[q]: the weight piece lane t feeds into k-step q, from what it loaded.
// Every lane of the warp must call this (in-quad shuffles).
template <int kOrder>
__device__ __forceinline__ void mmaWeightPieces(uint64_t (&a)[4], Chunk const& c, int t)
{
    if constexpr (kOrder == kOrderRot32)
    {
        // c = k-step t. Round r takes piece r of k-step (t - r) mod 4 from the
        // lane that loaded it, so k-step q arrives in round (t - q) mod 4.
        uint64_t s[4];
        s[0] = c.v[0];
        int const quadBase = (threadIdx.x & 31) & ~3;
#pragma unroll
        for (int r = 1; r < 4; ++r)
        {
            s[r] = __shfl_sync(0xFFFFFFFFu, c.v[r], quadBase | ((t - r) & 3));
        }
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            int const idx = (t - q) & 3;
            uint64_t const lo = (idx & 1) ? s[1] : s[0];
            uint64_t const hi = (idx & 1) ? s[3] : s[2];
            a[q] = (idx & 2) ? hi : lo;
        }
    }
    else if constexpr (kOrder == kOrderPair16)
    {
        // Even lanes hold k-steps 0, 1 and odd lanes k-steps 2, 3 (pieces
        // 2 (t >> 1) and 2 (t >> 1) + 1 of each); swap the second pieces.
        bool const odd = (t & 1) != 0;
        uint64_t const r0 = __shfl_xor_sync(0xFFFFFFFFu, c.v[1], 1);
        uint64_t const r1 = __shfl_xor_sync(0xFFFFFFFFu, c.v[3], 1);
        a[0] = odd ? r0 : c.v[0];
        a[1] = odd ? r1 : c.v[2];
        a[2] = odd ? c.v[0] : r0;
        a[3] = odd ? c.v[2] : r1;
    }
    else
    {
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            a[q] = c.v[q];
        }
    }
}

template <int R, int kStage, int kOrder>
__device__ __forceinline__ void mmaComputeStage(float (&d)[4], Chunk const (&x)[kStage], Chunk const (&y)[kStage],
    __nv_bfloat16 const* hRow, bool hLive, int block, int t)
{
#pragma unroll
    for (int j = 0; j < kStage; ++j)
    {
        uint4 h0 = make_uint4(0, 0, 0, 0);
        uint4 h1 = make_uint4(0, 0, 0, 0);
        if (hLive)
        {
            if constexpr (kOrder == kOrderNat8)
            {
                // Values 16 q + 4 t .. + 3 for q = 0..3: 8 bytes every 32.
                uint2 const* src = reinterpret_cast<uint2 const*>(hRow + (block + j) * kMmaBlockElems + 4 * t);
                uint2 const q0 = src[0];
                uint2 const q1 = src[4];
                uint2 const q2 = src[8];
                uint2 const q3 = src[12];
                h0 = make_uint4(q0.x, q0.y, q1.x, q1.y);
                h1 = make_uint4(q2.x, q2.y, q3.x, q3.y);
            }
            else
            {
                // Permuted: values 16 t .. 16 t + 15 of the block. Rot32 /
                // Pair16: the row is staged in lane order, so the same 32
                // bytes are lane t's pieces for k-steps 0..3.
                uint4 const* src = reinterpret_cast<uint4 const*>(hRow + (block + j) * kMmaBlockElems + 16 * t);
                h0 = src[0];
                h1 = src[1];
            }
        }
        uint32_t const hb[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        uint64_t xp[4];
        uint64_t yp[4];
        mmaWeightPieces<kOrder>(xp, x[j], t);
        mmaWeightPieces<kOrder>(yp, y[j], t);
#pragma unroll
        for (int sIdx = 0; sIdx < 4; ++sIdx)
        {
            // Piece s holds the pairs mapped to k positions (2t, 2t + 1) and
            // (2t + 8, 2t + 9) of k-step s, for A and B alike.
            uint64_t const xw = xp[sIdx];
            uint64_t const yw = yp[sIdx];
            mmaBf16(d, static_cast<uint32_t>(xw), static_cast<uint32_t>(yw), static_cast<uint32_t>(xw >> 32),
                static_cast<uint32_t>(yw >> 32), hb[2 * sIdx], hb[2 * sIdx + 1]);
        }
    }
}

template <int R, int kWarps, int kStage, int kLoad, int kMinBlocks, int kOrder>
__global__ void __launch_bounds__(kWarps * 32, kMinBlocks) fusedMmaDraftTailKernel(DraftTailParams p)
{
    static_assert(R >= 1 && R <= 8, "n columns of m16n8k16");
    static_assert(kMmaBlocks % (2 * kStage) == 0, "double-buffered stages must tile K");
    extern __shared__ uint4 smemDyn[];
    __nv_bfloat16* smemH = reinterpret_cast<__nv_bfloat16*>(smemDyn); // [R][kMmaHStride]

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const g = lane >> 2;
    int const t = lane & 3;
    int const numRows = p.vocabLocal;
    int const numTiles = (numRows + 15) / 16;
    int const warpStride = gridDim.x * kWarps;
    int tile = blockIdx.x * kWarps + warp;

    uint64_t const policy = kLoad == kLoad32Hint ? evictFirstPolicy() : 0ull;
    char const* weight = reinterpret_cast<char const*>(p.weight);
    // Rows beyond the shard read row 0 (valid memory); their results are dropped.
    auto rowPtr = [&](int row) -> char const*
    { return weight + static_cast<int64_t>(row < numRows ? row : 0) * (6144 * 2) + mmaLaneOffset<kOrder>(t); };

    Chunk xa[kStage], ya[kStage], xb[kStage], yb[kStage];
    if (tile < numTiles)
    {
        mmaLoadStage<kStage, kLoad, kOrder>(xa, ya, rowPtr(tile * 16 + g), rowPtr(tile * 16 + g + 8), 0, policy);
    }

    griddepWait();
    if constexpr (kOrder == kOrderRot32 || kOrder == kOrderPair16)
    {
        // Lane order: in each 64-value block, lane t's pieces for k-steps
        // 0..3 are contiguous at values 16 t .. 16 t + 15.
        for (int i = threadIdx.x; i < R * kMmaBlocks * 16; i += blockDim.x)
        {
            int const r = i / (kMmaBlocks * 16);
            int const rem = i - r * (kMmaBlocks * 16);
            int const block = rem >> 4;
            int const lt = (rem >> 2) & 3;
            int const q = rem & 3;
            reinterpret_cast<uint64_t*>(smemH + r * kMmaHStride)[rem] = reinterpret_cast<uint64_t const*>(
                p.hidden + r * p.hiddenStride)[block * 16 + q * 4 + mmaNativePiece<kOrder>(lt, q)];
        }
    }
    else
    {
        for (int i = threadIdx.x; i < R * 6144 / 8; i += blockDim.x)
        {
            int const r = i / (6144 / 8);
            int const c = i - r * (6144 / 8);
            reinterpret_cast<uint4*>(smemH + r * kMmaHStride)[c]
                = reinterpret_cast<uint4 const*>(p.hidden + r * p.hiddenStride)[c];
        }
    }
    __syncthreads();

    bool const hLive = g < R;
    __nv_bfloat16 const* hRow = smemH + (hLive ? g : 0) * kMmaHStride;
    uint64_t bestJ[2] = {0ull, 0ull};

    for (; tile < numTiles; tile += warpStride)
    {
        int const row0 = tile * 16 + g;
        int const row1 = row0 + 8;
        char const* p0 = rowPtr(row0);
        char const* p1 = rowPtr(row1);
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < kMmaBlocks; b += 2 * kStage)
        {
            mmaLoadStage<kStage, kLoad, kOrder>(xb, yb, p0, p1, b + kStage, policy);
            mmaComputeStage<R, kStage, kOrder>(d, xa, ya, hRow, hLive, b, t);
            if (b + 2 * kStage < kMmaBlocks)
            {
                mmaLoadStage<kStage, kLoad, kOrder>(xa, ya, p0, p1, b + 2 * kStage, policy);
            }
            else if (tile + warpStride < numTiles)
            {
                int const next = tile + warpStride;
                mmaLoadStage<kStage, kLoad, kOrder>(
                    xa, ya, rowPtr(next * 16 + g), rowPtr(next * 16 + g + 8), 0, policy);
            }
            mmaComputeStage<R, kStage, kOrder>(d, xb, yb, hRow, hLive, b + kStage, t);
        }
        // d[j] = (row0, column 2t + j), d[2 + j] = (row1, column 2t + j).
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            int const n = 2 * t + j;
            if (n < R)
            {
                if (row0 < numRows)
                {
                    __nv_bfloat16 const v = __float2bfloat16_rn(d[j]);
                    bestJ[j] = umax64(bestJ[j], makeKey(__bfloat16_as_ushort(v), p.vocabOffset + row0));
                    if (p.logitsOut != nullptr)
                    {
                        p.logitsOut[n * p.logitsOutStride + row0] = v;
                    }
                }
                if (row1 < numRows)
                {
                    __nv_bfloat16 const v = __float2bfloat16_rn(d[2 + j]);
                    bestJ[j] = umax64(bestJ[j], makeKey(__bfloat16_as_ushort(v), p.vocabOffset + row1));
                    if (p.logitsOut != nullptr)
                    {
                        p.logitsOut[n * p.logitsOutStride + row1] = v;
                    }
                }
            }
        }
    }
    // Reduce over g (lane bits 2..4), then lane 0 collects column n from lane n / 2.
#pragma unroll
    for (int j = 0; j < 2; ++j)
    {
#pragma unroll
        for (int off = 4; off < 32; off <<= 1)
        {
            bestJ[j] = umax64(bestJ[j], __shfl_xor_sync(0xFFFFFFFFu, bestJ[j], off));
        }
    }
    uint64_t best[R];
#pragma unroll
    for (int n = 0; n < R; ++n)
    {
        best[n] = __shfl_sync(0xFFFFFFFFu, bestJ[n & 1], n >> 1);
    }
    ctaFinish<R, kWarps>(p, best);
}

// ---------------------------------------------------------------------------
// Logits kernel: rank-local max over precomputed BF16 logits + exchange.
// ---------------------------------------------------------------------------
template <int R>
__global__ void __launch_bounds__(512, 1) fromLogitsDraftTailKernel(DraftTailParams p)
{
    __shared__ uint64_t warpBest[32][R];
    __shared__ uint64_t ctaKeys[kMaxRows];
    __shared__ uint64_t gathered[kMaxRanks][kMaxRows];

    griddepWait();
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    int const numWarps = blockDim.x >> 5;
    uint64_t best[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        best[r] = 0ull;
    }
    // Flat (row, 8-element vector) index space; each thread issues kBatch
    // independent 16-byte loads before reducing them, so a row costs about one
    // memory round trip instead of one per loop iteration.
    constexpr int kBatch = 8;
    bool vec = (p.logitsStride % 8) == 0 && (reinterpret_cast<uintptr_t>(p.logits) & 15) == 0;
    int const nVec = vec ? p.vocabLocal / 8 : 0;
    int const total = R * nVec;
    for (int base = threadIdx.x; base < total; base += kBatch * blockDim.x)
    {
        uint4 x[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b)
        {
            int const i = base + b * blockDim.x;
            if (i < total)
            {
                int const r = i / nVec;
                int const v = i - r * nVec;
                x[b] = __ldg(reinterpret_cast<uint4 const*>(p.logits + r * p.logitsStride) + v);
            }
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b)
        {
            int const i = base + b * blockDim.x;
            if (i < total)
            {
                int const r = i / nVec;
                int const v = i - r * nVec;
                uint32_t const w[4] = {x[b].x, x[b].y, x[b].z, x[b].w};
                uint64_t m = 0ull;
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    int64_t const id = p.vocabOffset + 8 * v + 2 * k;
                    m = umax64(m, makeKey(w[k] & 0xFFFFu, id));
                    m = umax64(m, makeKey(w[k] >> 16, id + 1));
                }
#pragma unroll
                for (int rr = 0; rr < R; ++rr)
                {
                    if (rr == r)
                    {
                        best[rr] = umax64(best[rr], m);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        __nv_bfloat16 const* row = p.logits + r * p.logitsStride;
        for (int i = nVec * 8 + threadIdx.x; i < p.vocabLocal; i += blockDim.x)
        {
            uint32_t const bits = __bfloat16_as_ushort(row[i]);
            best[r] = umax64(best[r], makeKey(bits, p.vocabOffset + i));
        }
        best[r] = warpMax64(best[r]);
    }
    if (lane == 0)
    {
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            warpBest[warp][r] = best[r];
        }
    }
    __syncthreads();
    if (warp == 0)
    {
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            uint64_t m = lane < numWarps ? warpBest[lane][r] : 0ull;
            m = warpMax64(m);
            if (lane == 0)
            {
                ctaKeys[r] = m;
            }
        }
    }
    __syncthreads();
    if (isDirectMode(p.mode))
    {
        directExchange(p, ctaKeys, 0, 1);
        return;
    }
    if (warp != 0)
    {
        return;
    }
    exchangeAndWrite(p, ctaKeys, gathered);
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

constexpr int kHidden6144Chunks = 6144 / 512; // 32-byte chunks per lane per row

struct VariantDesc
{
    int warps;
    int split;
    int load;
    int minBlocks;
};

// Kernel shape and load flavour, selectable at run time for tuning.
constexpr VariantDesc kVariants[] = {
    {8, 2, kLoad32Hint, 1},  // 0: default
    {8, 2, kLoad32, 1},      // 1: no L2 eviction hint
    {8, 2, kLoad16x2Pf, 1},  // 2: 16-byte loads with L2 prefetch
    {16, 3, kLoad32Hint, 1}, // 3: more warps, shorter units
    {4, 2, kLoad32Hint, 2},  // 4: two CTAs per SM
    {8, 1, kLoad32Hint, 1},  // 5: whole-row units
    {8, 0, kLoad32Hint, 1},  // 6: per-warp cp.async.bulk row ring (R <= 2; else variant 0)
    {8, 0, kLoad32Hint, 1},  // 7: tensor-core tiles, 8 warps, 2-block stages
    {4, 0, kLoad32Hint, 2},  // 8: tensor-core tiles, 4 warps, 2 CTAs/SM
    {8, 0, kLoad32, 1},      // 9: tensor-core tiles, no L2 hint
    {8, 0, kLoad32Hint, 1},  // 10: tensor-core tiles, 8 warps, 4-block stages
    {8, 0, kLoad32Hint, 1},  // 11: as 10, native K order (kOrderNat8: 8-byte lane loads)
    {4, 0, kLoad32Hint, 2},  // 12: native K order (kOrderNat8), 4 warps, 2 CTAs/SM
    {8, 0, kLoad32Hint, 1},  // 13: as 10, native K order (kOrderRot32: 32-byte loads, quad shuffles)
    {8, 0, kLoad32Hint, 1},  // 14: as 10, native K order (kOrderPair16: 16-byte loads, pair shuffles)
};
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);
// Variants from here on are tensor-core kernels; their default grid covers
// one 16-row tile per warp.
constexpr int kFirstMmaVariant = 7;

template <int R, int kWarps, int kSplit, int kLoad, int kMinBlocks>
void launchFusedShape(DraftTailParams const& p, int numCtas, bool pdl, cudaStream_t stream)
{
    auto kernel = fusedDraftTailKernel<R, kWarps, kHidden6144Chunks, kSplit, kLoad, kMinBlocks>;
    size_t const smem = static_cast<size_t>(R) * 6144 * sizeof(__nv_bfloat16);
    static bool const attrSet = [&]()
    {
        TLLM_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        return true;
    }();
    (void) attrSet;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(numCtas);
    config.blockDim = dim3(kWarps * 32);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

template <int R, int kWarps, int kSlots>
void launchBulkShape(DraftTailParams const& p, int numCtas, bool pdl, cudaStream_t stream)
{
    auto kernel = fusedBulkDraftTailKernel<R, kWarps, kSlots>;
    size_t const smem = static_cast<size_t>(R + kWarps * kSlots) * 6144 * sizeof(__nv_bfloat16);
    static bool const attrSet = [&]()
    {
        TLLM_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        return true;
    }();
    (void) attrSet;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(numCtas);
    config.blockDim = dim3(kWarps * 32);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

template <int R, int kWarps, int kStage, int kLoad, int kMinBlocks, int kOrder = kOrderPermuted>
void launchMmaShape(DraftTailParams const& p, int numCtas, bool pdl, cudaStream_t stream)
{
    auto kernel = fusedMmaDraftTailKernel<R, kWarps, kStage, kLoad, kMinBlocks, kOrder>;
    size_t const smem = static_cast<size_t>(R) * kMmaHStride * sizeof(__nv_bfloat16);
    static bool const attrSet = [&]()
    {
        TLLM_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        return true;
    }();
    (void) attrSet;
    if (numCtas <= 0)
    {
        int const numTiles = (p.vocabLocal + 15) / 16;
        numCtas = (numTiles + kWarps - 1) / kWarps;
    }
    TLLM_CHECK_WITH_INFO((p.mode != ExchangeMode::kMulticastDirect && p.mode != ExchangeMode::kUnicastDirect)
            || numCtas <= kMaxDirectCtas,
        "glm_moe_dsa_draft_tail: direct exchange supports at most %d CTAs", kMaxDirectCtas);
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(numCtas);
    config.blockDim = dim3(kWarps * 32);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

template <int R>
void launchFusedRows(DraftTailParams const& p, int variant, int numCtas, bool pdl, cudaStream_t stream)
{
    switch (variant)
    {
    case 0: launchFusedShape<R, 8, 2, kLoad32Hint, 1>(p, numCtas, pdl, stream); break;
    case 1: launchFusedShape<R, 8, 2, kLoad32, 1>(p, numCtas, pdl, stream); break;
    case 2: launchFusedShape<R, 8, 2, kLoad16x2Pf, 1>(p, numCtas, pdl, stream); break;
    case 3: launchFusedShape<R, 16, 3, kLoad32Hint, 1>(p, numCtas, pdl, stream); break;
    case 4: launchFusedShape<R, 4, 2, kLoad32Hint, 2>(p, numCtas, pdl, stream); break;
    case 5: launchFusedShape<R, 8, 1, kLoad32Hint, 1>(p, numCtas, pdl, stream); break;
    case 6:
        if constexpr (R <= 2)
        {
            launchBulkShape<R, 8, 2>(p, numCtas, pdl, stream);
        }
        else
        {
            launchFusedShape<R, 8, 2, kLoad32Hint, 1>(p, numCtas, pdl, stream);
        }
        break;
    case 7: launchMmaShape<R, 8, 2, kLoad32Hint, 1>(p, numCtas, pdl, stream); break;
    case 8: launchMmaShape<R, 4, 2, kLoad32Hint, 2>(p, numCtas, pdl, stream); break;
    case 9: launchMmaShape<R, 8, 2, kLoad32, 1>(p, numCtas, pdl, stream); break;
    case 10: launchMmaShape<R, 8, 4, kLoad32Hint, 1>(p, numCtas, pdl, stream); break;
    case 11: launchMmaShape<R, 8, 4, kLoad32Hint, 1, kOrderNat8>(p, numCtas, pdl, stream); break;
    case 12: launchMmaShape<R, 4, 4, kLoad32Hint, 2, kOrderNat8>(p, numCtas, pdl, stream); break;
    case 13: launchMmaShape<R, 8, 4, kLoad32Hint, 1, kOrderRot32>(p, numCtas, pdl, stream); break;
    case 14: launchMmaShape<R, 8, 4, kLoad32Hint, 1, kOrderPair16>(p, numCtas, pdl, stream); break;
    default: TLLM_THROW("glm_moe_dsa_draft_tail: unknown fused variant %d", variant);
    }
}

int smCount()
{
    static int const count = []()
    {
        int dev = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&dev));
        int n = 0;
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
        return n;
    }();
    return count;
}

} // namespace

int numFusedVariants()
{
    return kNumVariants;
}

void invokeDraftTailFused(DraftTailParams const& p, FusedLaunchConfig const& config, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.hiddenDim == 6144, "glm_moe_dsa_draft_tail: fused kernel supports hidden size 6144 only");
    TLLM_CHECK_WITH_INFO(p.rows >= 1 && p.rows <= kMaxRows, "glm_moe_dsa_draft_tail: rows out of range");
    TLLM_CHECK_WITH_INFO(config.variant >= 0 && config.variant < kNumVariants, "glm_moe_dsa_draft_tail: bad variant");
    TLLM_CHECK_WITH_INFO(p.logitsOut == nullptr || config.variant >= kFirstMmaVariant,
        "glm_moe_dsa_draft_tail: logits output needs a tensor-core variant");
    int numCtas = config.numCtas;
    if (numCtas <= 0 && config.variant < kFirstMmaVariant)
    {
        numCtas = smCount() * kVariants[config.variant].minBlocks;
    }
    TLLM_CHECK_WITH_INFO((p.mode != ExchangeMode::kMulticastDirect && p.mode != ExchangeMode::kUnicastDirect)
            || numCtas <= kMaxDirectCtas,
        "glm_moe_dsa_draft_tail: direct exchange supports at most %d CTAs", kMaxDirectCtas);
    switch (p.rows)
    {
    case 1: launchFusedRows<1>(p, config.variant, numCtas, config.pdl, stream); break;
    case 2: launchFusedRows<2>(p, config.variant, numCtas, config.pdl, stream); break;
    case 3: launchFusedRows<3>(p, config.variant, numCtas, config.pdl, stream); break;
    case 4: launchFusedRows<4>(p, config.variant, numCtas, config.pdl, stream); break;
    case 5: launchFusedRows<5>(p, config.variant, numCtas, config.pdl, stream); break;
    case 6: launchFusedRows<6>(p, config.variant, numCtas, config.pdl, stream); break;
    case 7: launchFusedRows<7>(p, config.variant, numCtas, config.pdl, stream); break;
    case 8: launchFusedRows<8>(p, config.variant, numCtas, config.pdl, stream); break;
    default: break;
    }
}

template <int R>
void launchFromLogitsRows(DraftTailParams const& p, bool pdl, cudaStream_t stream)
{
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(1);
    config.blockDim = dim3(512);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, fromLogitsDraftTailKernel<R>, p));
}

void invokeDraftTailFromLogits(DraftTailParams const& p, bool pdl, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(p.rows >= 1 && p.rows <= kMaxRows, "glm_moe_dsa_draft_tail: rows out of range");
    switch (p.rows)
    {
    case 1: launchFromLogitsRows<1>(p, pdl, stream); break;
    case 2: launchFromLogitsRows<2>(p, pdl, stream); break;
    case 3: launchFromLogitsRows<3>(p, pdl, stream); break;
    case 4: launchFromLogitsRows<4>(p, pdl, stream); break;
    case 5: launchFromLogitsRows<5>(p, pdl, stream); break;
    case 6: launchFromLogitsRows<6>(p, pdl, stream); break;
    case 7: launchFromLogitsRows<7>(p, pdl, stream); break;
    case 8: launchFromLogitsRows<8>(p, pdl, stream); break;
    default: break;
    }
}

} // namespace draft_tail

TRTLLM_MV2_NAMESPACE_END
