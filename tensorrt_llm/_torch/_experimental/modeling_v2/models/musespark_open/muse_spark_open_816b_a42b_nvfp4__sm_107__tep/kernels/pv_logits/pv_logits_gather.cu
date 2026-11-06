/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * One-shot all-gather of the target LM head's vocabulary shards into the final row layout
 * (``distributed_head_sampler``).
 *
 * Every rank holds ``local`` = its ``[rows, shard]`` BF16 shard of the logits; the result on every
 * rank is ``out`` = ``[rows, world * shard]`` with rank ``p``'s shard in columns ``p * shard ..``,
 * i.e. what the built-in all-gather followed by ``torch.cat(chunks, dim=-1)`` produces, as a copy
 * of the same bytes.
 *
 * Transport: LL pairs over the NVLink multicast fabric, no reset. Each 32-bit data word (two BF16)
 * travels as one 64-bit element ``(epoch << 32) | word``; this rank publishes its shard with
 * 16-byte stores (two elements) to the multicast address of the area, which the fabric delivers
 * to every rank's copy at the final position ``row * vocabWords + rank * shardWords + w``. Each
 * rank polls its own copy with 16-byte relaxed system-scope loads until both elements of a vector
 * carry this call's epoch, then stores the data words into ``out``. A 64-bit element is written
 * and read as one access, so a matching epoch half implies the data half of the same call.
 *
 * Epoch: CTA ``b`` keeps ``counters[b]`` (this rank's device memory, read and written only by CTA
 * ``b`` of this kernel): it uses ``counters[b] + 1`` and stores that back at its end. The grid is
 * fixed (``kCtas``), every call runs every CTA, and every rank runs the same sequence of calls, so
 * all CTAs of all ranks agree on the epoch of a call. The area starts zeroed (epoch 0, never used).
 * Area words are never reset: a stale word carries an older epoch. Two areas alternate by epoch
 * parity (call ``e`` uses area ``e & 1``), so a peer writes the area this rank reads in call ``e``
 * next in its call ``e + 2``, which it starts only after its call ``e + 1`` has seen this rank's
 * call-``e + 1`` publish, which this rank's stream orders after its call ``e`` completed. So calls
 * may follow each other with no other collective in between.
 *
 * PDL: ``griddepcontrol.wait`` precedes every global access; dependents are released after the
 * last transport access, before the output stores (a dependent reads ``out`` only after its own
 * wait, which waits for this grid's completion).
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>

namespace
{

constexpr int kCtas = 128;
constexpr int kThreads = 512;
//! Area vectors (two pairs, 16 bytes) one thread polls at most.
constexpr int kPerThread = 8;

struct Params
{
    uint2 const* local; // [rows][shardWords / 2]: this rank's shard, two data words per vector
    uint64_t mcArea;    // multicast address of the two areas, each [rows][vocabWords] 64-bit pairs
    uint64_t ucArea;    // this rank's copy of the two areas
    uint64_t areaBytes; // bytes of one area
    uint2* out;         // [rows][vocabWords / 2]
    uint32_t* counters; // [kCtas]
    int rank;
    int halfShard;      // vectors (two words) per shard row
    int halfVocab;      // vectors per full row
    int rows;
    int pollMode;       // 0: re-poll every stale vector each round; 1: spin on the first stale one, then re-poll
                        // the rest; 2: as 1 with a 32 ns back-off between spins
    unsigned long long timeoutNs;
};

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

__device__ __forceinline__ uint32_t staleBit(unsigned long long a, unsigned long long b, uint32_t epoch, int k)
{
    uint32_t const ea = static_cast<uint32_t>(a >> 32);
    uint32_t const eb = static_cast<uint32_t>(b >> 32);
    return (ea != epoch || eb != epoch) ? (1u << k) : 0u;
}

__global__ void __launch_bounds__(kThreads, 1) pvLogitsGatherKernel(Params const p)
{
    __shared__ uint32_t sEpoch;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (threadIdx.x == 0)
    {
        sEpoch = p.counters[blockIdx.x] + 1u;
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
    uint64_t const parity = static_cast<uint64_t>(epoch & 1u) * p.areaBytes;
    uint64_t const mcArea = p.mcArea + parity;
    uint64_t const ucArea = p.ucArea + parity;
    int const t = blockIdx.x * kThreads + threadIdx.x;
    int const stride = kCtas * kThreads;

    // Publish this rank's shard into every rank's copy (one multicast store per two words).
    int const own = p.rows * p.halfShard;
    for (int v = t; v < own; v += stride)
    {
        int const row = v / p.halfShard;
        int const c = v - row * p.halfShard;
        uint2 const d = p.local[v];
        int const dst = row * p.halfVocab + p.rank * p.halfShard + c;
        publishPair(mcArea + static_cast<uint64_t>(dst) * 16u, tag | d.x, tag | d.y);
    }

    // Poll this rank's copy: every vector of every row, all in flight together.
    int const total = p.rows * p.halfVocab;
    unsigned long long lo[kPerThread];
    unsigned long long hi[kPerThread];
    uint32_t stale = 0;
#pragma unroll
    for (int k = 0; k < kPerThread; ++k)
    {
        int const v = t + k * stride;
        lo[k] = 0;
        hi[k] = 0;
        if (v < total)
        {
            pollPair(ucArea + static_cast<uint64_t>(v) * 16u, lo[k], hi[k]);
            stale |= staleBit(lo[k], hi[k], epoch, k);
        }
    }
    if (stale != 0)
    {
        unsigned long long const start = globalTimer();
        while (stale != 0)
        {
            if (p.pollMode != 0)
            {
                // Wait for the first stale vector alone (the rest of a peer's shard lands with it), so the
                // polls do not keep the whole area's worth of loads in flight while the fabric writes arrive.
                uint32_t const first = stale & (0u - stale);
                uint32_t waiting = first;
                while (waiting != 0)
                {
#pragma unroll
                    for (int k = 0; k < kPerThread; ++k)
                    {
                        if ((first >> k) & 1u)
                        {
                            int const v = t + k * stride;
                            pollPair(ucArea + static_cast<uint64_t>(v) * 16u, lo[k], hi[k]);
                            waiting = staleBit(lo[k], hi[k], epoch, k);
                        }
                    }
                    if (waiting != 0)
                    {
                        if (p.pollMode == 2)
                        {
                            __nanosleep(32);
                        }
                        if (globalTimer() - start > p.timeoutNs)
                        {
                            __trap();
                        }
                    }
                }
                stale &= ~first;
                if (stale == 0)
                {
                    break;
                }
            }
            uint32_t const polled = stale;
            stale = 0;
#pragma unroll
            for (int k = 0; k < kPerThread; ++k)
            {
                if ((polled >> k) & 1u)
                {
                    int const v = t + k * stride;
                    pollPair(ucArea + static_cast<uint64_t>(v) * 16u, lo[k], hi[k]);
                    stale |= staleBit(lo[k], hi[k], epoch, k);
                }
            }
            if (stale != 0 && globalTimer() - start > p.timeoutNs)
            {
                __trap();
            }
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
#pragma unroll
    for (int k = 0; k < kPerThread; ++k)
    {
        int const v = t + k * stride;
        if (v < total)
        {
            p.out[v] = make_uint2(static_cast<uint32_t>(lo[k]), static_cast<uint32_t>(hi[k]));
        }
    }
    if (threadIdx.x == 0)
    {
        p.counters[blockIdx.x] = epoch;
    }
}

__global__ void pvLogitsDryKernel() {}

void gather(at::Tensor const& local, at::Tensor const& out, at::Tensor const& counters, int64_t mcArea, int64_t ucArea,
    int64_t rank, int64_t world, int64_t rows, int64_t shard, bool pdl, int64_t timeoutNs, int64_t pollMode, bool dry)
{
    c10::cuda::CUDAGuard guard(out.device());
    cudaLaunchConfig_t config{};
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    if (dry)
    {
        config.gridDim = dim3(1);
        C10_CUDA_CHECK(cudaLaunchKernelEx(&config, pvLogitsDryKernel));
        return;
    }
    TORCH_CHECK(local.is_cuda() && out.is_cuda() && counters.is_cuda(), "pv logits gather: CUDA tensors only");
    TORCH_CHECK(local.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
        "pv logits gather: BF16 logits only");
    TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= kCtas && counters.is_contiguous(),
        "pv logits gather: counters must be a contiguous int32 tensor of at least ", kCtas, " entries");
    TORCH_CHECK(shard % 4 == 0, "pv logits gather: the shard must be a whole number of 4-element vectors");
    TORCH_CHECK(world >= 1 && rank >= 0 && rank < world, "pv logits gather: bad rank / world");
    TORCH_CHECK(local.dim() == 2 && local.size(0) == rows && local.size(1) == shard && local.is_contiguous(),
        "pv logits gather: local must be a contiguous [rows, shard] tensor");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == rows && out.size(1) == world * shard && out.is_contiguous(),
        "pv logits gather: out must be a contiguous [rows, world * shard] tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0
            && reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0 && mcArea % 16 == 0 && ucArea % 16 == 0,
        "pv logits gather: misaligned operand");
    int64_t const halfShard = shard / 4;
    int64_t const total = rows * world * halfShard;
    TORCH_CHECK(total <= static_cast<int64_t>(kCtas) * kThreads * kPerThread, "pv logits gather: ", total,
        " vectors exceed the grid's ", static_cast<int64_t>(kCtas) * kThreads * kPerThread);
    Params p{};
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    p.mcArea = static_cast<uint64_t>(mcArea);
    p.ucArea = static_cast<uint64_t>(ucArea);
    p.out = reinterpret_cast<uint2*>(out.data_ptr());
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.rank = static_cast<int>(rank);
    p.halfShard = static_cast<int>(halfShard);
    p.halfVocab = static_cast<int>(world * halfShard);
    p.rows = static_cast<int>(rows);
    p.areaBytes = static_cast<uint64_t>(total) * 16u;
    p.timeoutNs = static_cast<unsigned long long>(timeoutNs);
    TORCH_CHECK(pollMode >= 0 && pollMode <= 2, "pv logits gather: poll mode 0, 1 or 2");
    p.pollMode = static_cast<int>(pollMode);
    config.gridDim = dim3(kCtas);
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, pvLogitsGatherKernel, p));
}

} // namespace

TORCH_LIBRARY(musespark_open_pv_logits, m)
{
    m.def(
        "gather(Tensor local, Tensor(a!) out, Tensor(b!) counters, int mc_area, int uc_area, int rank, int world, "
        "int rows, int shard, bool pdl, int timeout_ns, int poll_mode, bool dry) -> ()");
}

TORCH_LIBRARY_IMPL(musespark_open_pv_logits, CUDA, m)
{
    m.impl("gather", &gather);
}
