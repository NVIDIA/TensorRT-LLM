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

// Each CTA owns an equal range of the shard's vocabulary rows and walks it in tiles of 16 rows and k-chunks of KC
// elements: a producer warp issues one cp.async.bulk per row into a ring of NS shared-memory stages, and eight
// consumer warps run m16n8k16 MMAs with the weight tile as A (16 vocabulary rows) and the input rows as B (the n = 8
// token columns), each warp taking KC / 8 of the chunk; the warps' partial sums are added in warp order, and each
// logit becomes an argmax key in registers. The CTA's best key per row goes to the scratch, and the last CTA to finish
// reduces them and, with several ranks, runs the exchange.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/lmHeadArgmax/lmHeadArgmax.h"

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace lm_head_argmax
{
namespace
{

constexpr int kConsumerWarps = 8;
constexpr int kThreads = (kConsumerWarps + 1) * 32; // + the producer warp
constexpr int kTileRows = 16;
constexpr int kChunk = 1024;                        // KC: k elements per stage
constexpr int kSmemMax = 227 * 1024;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrixX4(uint32_t* r, uint32_t a)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void ldmatrixX2(uint32_t* r, uint32_t a)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void mmaBf16(float* c, uint32_t const* a, uint32_t const* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void barInit(uint64_t* b, int count)
{
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smemAddr(b)), "r"(count));
}

__device__ __forceinline__ void barExpectTx(uint64_t* b, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(smemAddr(b)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void barArrive(uint64_t* b)
{
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(smemAddr(b)) : "memory");
}

__device__ __forceinline__ void barWait(uint64_t* b, uint32_t phase)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
                     : "=r"(done)
                     : "r"(smemAddr(b)), "r"(phase)
                     : "memory");
    }
}

__device__ __forceinline__ void bulkCopy(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                     smemAddr(dst)),
                 "l"(src), "r"(bytes), "r"(smemAddr(bar))
                 : "memory");
}

//! The vocabulary rows are read once per call: an L2 evict-first policy keeps them from displacing other data.
__device__ __forceinline__ uint64_t evictFirstPolicy()
{
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ void bulkCopyEvictFirst(
    void* dst, void const* src, uint32_t bytes, uint64_t* bar, uint64_t policy)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], "
        "%4;\n" ::"r"(smemAddr(dst)),
        "l"(src), "r"(bytes), "r"(smemAddr(bar)), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void consumersSync()
{
    asm volatile("bar.sync 1, 256;\n" ::: "memory");
}

__device__ __forceinline__ void pdlWait()
{
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

__device__ __forceinline__ void pdlLaunchDependents()
{
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t argmaxKey(float v, int index)
{
    uint32_t const u = __float_as_uint(v);
    return (static_cast<uint64_t>((u & 0x80000000u) ? ~u : (u | 0x80000000u)) << 32)
        | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(index));
}

__device__ __forceinline__ int32_t keyToken(uint64_t key)
{
    return static_cast<int32_t>(0xFFFFFFFFu - static_cast<uint32_t>(key));
}

#endif // __CUDA_ARCH__ >= 900

__host__ __device__ inline size_t inputBytes(int m, int k)
{
    return (static_cast<size_t>(m) * (k * 2 + 16) + 127) / 128 * 128;
}

template <int NS>
__host__ __device__ inline size_t smemBytes(int m, int k)
{
    return inputBytes(m, k) + static_cast<size_t>(NS) * kTileRows * (kChunk * 2 + 16)
        + static_cast<size_t>(kConsumerWarps) * kTileRows * 8 * sizeof(float) + (2 * NS + 1) * sizeof(uint64_t)
        + kMaxRows * sizeof(uint64_t) + sizeof(int);
}

template <int NS>
__global__ void __launch_bounds__(kThreads, 1) lmHeadArgmaxKernel(Params const p, int rowsPerCta)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    extern __shared__ __align__(128) uint8_t smem[];
    constexpr int kTokens = 8;                   // MMA n
    constexpr int kRingStride = kChunk * 2 + 16; // bytes per ring row (padded against bank conflicts)
    constexpr int kStageBytes = kTileRows * kRingStride;
    int const inputRowBytes = p.k * 2 + 16;
    uint8_t* sInput = smem;
    uint8_t* ring = smem + inputBytes(p.m, p.k);
    float* partial = reinterpret_cast<float*>(ring + static_cast<size_t>(NS) * kStageBytes);
    uint64_t* full = reinterpret_cast<uint64_t*>(partial + kConsumerWarps * kTileRows * kTokens);
    uint64_t* empty = full + NS;
    uint64_t* inputBar = empty + NS;
    unsigned long long* sBest = reinterpret_cast<unsigned long long*>(inputBar + 1);
    int* sLast = reinterpret_cast<int*>(sBest + kMaxRows);

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const row0 = blockIdx.x * rowsPerCta;
    int const row1 = min(p.n, row0 + rowsPerCta);
    int const numTiles = row1 > row0 ? (row1 - row0 + kTileRows - 1) / kTileRows : 0;
    int const numChunks = p.k / kChunk;
    int const total = numTiles * numChunks;

    if (tid == 0)
    {
        for (int s = 0; s < NS; ++s)
        {
            barInit(&full[s], 1);
            barInit(&empty[s], kConsumerWarps);
        }
        barInit(inputBar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (tid < kMaxRows)
    {
        sBest[tid] = 0ull;
    }
    __syncthreads();

    if (warp == kConsumerWarps)
    {
        // The producer: one vocabulary row per lane. The weights are read-only, so the first stages are issued
        // before waiting for the kernel that writes the input rows.
        uint64_t const policy = evictFirstPolicy();
        auto issue = [&](int i)
        {
            int const s = i % NS;
            int const t = i / numChunks;
            int const c = i % numChunks;
            int const n0 = row0 + t * kTileRows;
            int const rows = min(kTileRows, row1 - n0);
            if (lane == 0)
            {
                barExpectTx(&full[s], static_cast<uint32_t>(rows) * kChunk * 2);
            }
            __syncwarp();
            if (lane < rows)
            {
                bulkCopyEvictFirst(ring + static_cast<size_t>(s) * kStageBytes + lane * kRingStride,
                    static_cast<__nv_bfloat16 const*>(p.weight) + static_cast<size_t>(n0 + lane) * p.weightStride
                        + static_cast<size_t>(c) * kChunk,
                    kChunk * 2, &full[s], policy);
            }
        };
        for (int i = 0; i < NS && i < total; ++i)
        {
            issue(i);
        }
        pdlWait();
        pdlLaunchDependents();
        if (lane == 0)
        {
            barExpectTx(inputBar, static_cast<uint32_t>(p.m) * p.k * 2);
        }
        __syncwarp();
        for (int m = lane; m < p.m; m += 32)
        {
            bulkCopy(sInput + static_cast<size_t>(m) * inputRowBytes,
                static_cast<__nv_bfloat16 const*>(p.input) + static_cast<size_t>(m) * p.inputStride, p.k * 2, inputBar);
        }
        for (int i = NS; i < total; ++i)
        {
            barWait(&empty[i % NS], ((i / NS) - 1) & 1);
            issue(i);
        }
        return;
    }

    pdlWait();
    pdlLaunchDependents();
    // This thread's (tile row, token) in every tile's epilogue: one logit per tile for tid < 16 * m.
    int const em = tid % p.m;
    int const er = tid / p.m;
    bool const epilogue = tid < kTileRows * p.m;
    unsigned long long best = 0ull;
    barWait(inputBar, 0);
    constexpr int kStepsPerWarp = kChunk / 16 / kConsumerWarps; // 16-element k-steps per warp and chunk
    float acc[4];
    int const tok = lane & 7;
    uint32_t const inputRow = smemAddr(sInput) + (tok < p.m ? tok : 0) * inputRowBytes + ((lane >> 3) & 1) * 16;
    for (int i = 0; i < total; ++i)
    {
        int const s = i % NS;
        int const t = i / numChunks;
        int const c = i % numChunks;
        if (c == 0)
        {
            acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
        barWait(&full[s], (i / NS) & 1);
        uint32_t const stage
            = smemAddr(ring + static_cast<size_t>(s) * kStageBytes) + (lane & 15) * kRingStride + (lane >> 4) * 16;
#pragma unroll
        for (int q = 0; q < kStepsPerWarp; ++q)
        {
            int const ks = warp * kStepsPerWarp + q;
            uint32_t a[4];
            uint32_t b[2];
            ldmatrixX4(a, stage + ks * 32);
            ldmatrixX2(b, inputRow + (c * (kChunk / 8) + ks * 2) * 16);
            mmaBf16(acc, a, b);
        }
        __syncwarp();
        if (lane == 0)
        {
            barArrive(&empty[s]);
        }
        if (c == numChunks - 1)
        {
            // The tile's logits: the warps' partial sums added in warp order, then each epilogue thread's key.
            float* mine = partial + static_cast<size_t>(warp) * kTileRows * kTokens;
            int const r = lane >> 2;
            int const col = (lane & 3) * 2;
            mine[r * kTokens + col] = acc[0];
            mine[r * kTokens + col + 1] = acc[1];
            mine[(r + 8) * kTokens + col] = acc[2];
            mine[(r + 8) * kTokens + col + 1] = acc[3];
            consumersSync();
            int const n = row0 + t * kTileRows + er;
            if (epilogue && n < row1)
            {
                float v = 0.f;
#pragma unroll
                for (int w = 0; w < kConsumerWarps; ++w)
                {
                    v += partial[static_cast<size_t>(w) * kTileRows * kTokens + er * kTokens + em];
                }
                unsigned long long const key = argmaxKey(v, p.vocabOffset + n);
                best = key > best ? key : best;
            }
            consumersSync();
        }
    }

    // The CTA's best key per row -> its scratch row; the last CTA to count reduces every CTA's row.
    uint64_t* parts = p.scratch;
    unsigned int* counter = reinterpret_cast<unsigned int*>(p.scratch + kMaxCtas * kMaxRows);
    uint32_t* state = reinterpret_cast<uint32_t*>(p.scratch + kMaxCtas * kMaxRows + 1);
    if (epilogue)
    {
        atomicMax(&sBest[em], best);
    }
    consumersSync();
    if (tid == 0)
    {
        for (int m = 0; m < p.m; ++m)
        {
            parts[static_cast<size_t>(blockIdx.x) * kMaxRows + m] = sBest[m];
        }
        unsigned int arrived;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(arrived) : "l"(counter) : "memory");
        *sLast = arrived == gridDim.x - 1;
    }
    consumersSync();
    if (!*sLast)
    {
        return;
    }
    if (tid == 0)
    {
        *counter = 0u;
    }
    unsigned long long reduced[kMaxRows];
#pragma unroll
    for (int m = 0; m < kMaxRows; ++m)
    {
        reduced[m] = 0ull;
    }
    for (int b = tid; b < static_cast<int>(gridDim.x); b += kConsumerWarps * 32)
    {
#pragma unroll
        for (int m = 0; m < kMaxRows; ++m)
        {
            if (m < p.m)
            {
                unsigned long long const x = __ldcg(
                    reinterpret_cast<unsigned long long const*>(parts) + static_cast<size_t>(b) * kMaxRows + m);
                reduced[m] = x > reduced[m] ? x : reduced[m];
            }
        }
    }
#pragma unroll
    for (int m = 0; m < kMaxRows; ++m)
    {
        if (m < p.m)
        {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                unsigned long long const y = __shfl_xor_sync(0xffffffffu, reduced[m], o);
                reduced[m] = y > reduced[m] ? y : reduced[m];
            }
            if (lane == 0 && reduced[m])
            {
                atomicMax(&sBest[m], reduced[m]);
            }
        }
    }
    consumersSync();
    unsigned long long const local = tid < p.m ? sBest[tid] : 0ull;
    if (p.ranks <= 1)
    {
        if (tid < p.m)
        {
            p.tokens[tid * p.tokensStride] = keyToken(local);
        }
        return;
    }
    // The exchange: epoch e writes every rank's keys into buffer e % 3 of each receiver, the receiver polls until all
    // ranks' keys are there (keys are never 0), and the buffer of the previous call is cleared for epoch e + 2.
    uint32_t const epoch = state[0] + 1;
    int const previousRows = static_cast<int>(state[1]);
    int const cur = epoch % 3;
    int const old = (epoch + 2) % 3;
    size_t const epochSlots = static_cast<size_t>(kMaxRanks) * kMaxRows;
    if (tid < p.m)
    {
        for (int r = 0; r < p.ranks; ++r)
        {
            *reinterpret_cast<unsigned long long volatile*>(
                p.peers[r] + cur * epochSlots + static_cast<size_t>(p.rank) * kMaxRows + tid)
                = local;
        }
        unsigned long long global = 0ull;
        for (int r = 0; r < p.ranks; ++r)
        {
            unsigned long long x;
            do
            {
                x = *reinterpret_cast<unsigned long long volatile*>(
                    p.peers[p.rank] + cur * epochSlots + static_cast<size_t>(r) * kMaxRows + tid);
            } while (x == 0ull);
            global = global > x ? global : x;
        }
        p.tokens[tid * p.tokensStride] = keyToken(global);
    }
    for (int i = tid; i < previousRows; i += kConsumerWarps * 32)
    {
        for (int r = 0; r < p.ranks; ++r)
        {
            p.peers[p.rank][old * epochSlots + static_cast<size_t>(r) * kMaxRows + i] = 0ull;
        }
    }
    consumersSync();
    if (tid == 0)
    {
        state[0] = epoch;
        state[1] = static_cast<uint32_t>(p.m);
    }
#endif // __CUDA_ARCH__ >= 900
}

template <int NS>
void launchStages(Params const& p, int ctas, cudaStream_t stream, bool enablePdl)
{
    auto kernel = lmHeadArgmaxKernel<NS>;
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemMax));
    int const rowsPerCta = (p.n + ctas - 1) / ctas;
    cudaLaunchConfig_t config{};
    cudaLaunchAttribute attrs[1];
    config.gridDim = dim3(ctas);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = smemBytes<NS>(p.m, p.k);
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p, rowsPerCta));
}

} // namespace

bool supports(int m, int n, int k)
{
    return m >= 1 && m <= kMaxRows && n >= 1 && k > 0 && k % kChunk == 0
        && smemBytes<2>(m, k) <= static_cast<size_t>(kSmemMax);
}

void launch(Params const& params, int ctas, cudaStream_t stream, bool enablePdl)
{
    TLLM_CHECK_WITH_INFO(supports(params.m, params.n, params.k), "lm_head_argmax: unsupported shape");
    TLLM_CHECK_WITH_INFO(
        params.ranks >= 1 && params.ranks <= kMaxRanks && params.rank >= 0 && params.rank < params.ranks,
        "lm_head_argmax: rank out of range");
    ctas = std::max(1, std::min({ctas, kMaxCtas, params.n}));
    if (smemBytes<6>(params.m, params.k) <= static_cast<size_t>(kSmemMax))
    {
        launchStages<6>(params, ctas, stream, enablePdl);
    }
    else if (smemBytes<4>(params.m, params.k) <= static_cast<size_t>(kSmemMax))
    {
        launchStages<4>(params, ctas, stream, enablePdl);
    }
    else
    {
        launchStages<2>(params, ctas, stream, enablePdl);
    }
}

} // namespace lm_head_argmax
} // namespace kernels

TRTLLM_NAMESPACE_END
