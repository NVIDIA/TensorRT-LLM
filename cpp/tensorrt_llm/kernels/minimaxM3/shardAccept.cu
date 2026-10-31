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

#include "tensorrt_llm/kernels/minimaxM3/shardAccept.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Fused greedy Eagle3 verification for the MiniMax-M3 target head (GB300, TP-sharded vocabulary, <= 48 verify rows).
//
// One kernel replaces the all-gather of the target logits, the argmax and the strict-acceptance ops:
//
// * Each logit of this rank's BF16 shard becomes a 64-bit key ordered(value) << 32 | ~global_index (ties pick the
//   lowest index, as torch.argmax), max-reduced per row: warp shuffles, then atomicMax.
// * The last CTA to finish pushes the rank's per-row key into every rank's exchange slot through a multicast mapping
//   (Lamport: 0 = empty, the key is never 0), polls its own slots for all ranks, takes the global argmax, re-arms the
//   slots and runs the strict acceptance. Call n uses slot half n % 2.
//
// The keys order the BF16 values exactly, so the tokens are the argmax of the gathered logits.

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace shardAccept
{
namespace
{
__device__ __forceinline__ unsigned long long make_key(float v, uint32_t index)
{
    uint32_t u = __float_as_uint(v);
    // -0.0 and +0.0 compare equal (torch.argmax takes the lower index among them): one key for both.
    u = (u == 0x80000000u) ? 0u : u;
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | static_cast<unsigned long long>(0xFFFFFFFFu - index);
}

__device__ __forceinline__ unsigned long long shfl_max(unsigned long long k, int o)
{
    unsigned long long const ok = __shfl_xor_sync(0xffffffffu, k, o);
    return ok > k ? ok : k;
}

// Greedy verification from this rank's logits shard (verify-row count <= kMaxRows): per-row argmax keys over the
// shard, the last CTA exchanges them across ranks (multicast slots [2][kMaxRows][world], Lamport 0 = empty) and runs
// the acceptance epilogue.
__global__ void __launch_bounds__(kThreads, 1) shard_accept_kernel(__nv_bfloat16 const* __restrict__ logits, int M,
    int Vs, long long ld, int vocab_offset, unsigned long long* __restrict__ keys, unsigned int* __restrict__ counters,
    unsigned long long* __restrict__ ws_uc, unsigned long long* __restrict__ ws_mc, int rank, int world, AcceptArgs acc)
{
    __shared__ bool s_last;
    __shared__ int s_tok[kMaxRows];
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const nwarps = blockDim.x >> 5;
    int const groups = (Vs + 7) / 8;
    int const g0 = static_cast<int>((static_cast<long long>(blockIdx.x) * groups) / gridDim.x);
    int const g1 = static_cast<int>((static_cast<long long>(blockIdx.x + 1) * groups) / gridDim.x);
    bool const vec = (ld % 8) == 0;
    for (int r = warp; r < M; r += nwarps)
    {
        unsigned long long best = 0ull;
        __nv_bfloat16 const* row = logits + r * ld;
        for (int gi = g0 + lane; gi < g1; gi += 32)
        {
            int const c = gi * 8;
            if (vec && c + 8 <= Vs)
            {
                uint4 const raw = *reinterpret_cast<uint4 const*>(row + c);
                __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
                for (int q = 0; q < 8; ++q)
                {
                    unsigned long long const k
                        = make_key(__bfloat162float(e[q]), static_cast<uint32_t>(vocab_offset + c + q));
                    best = k > best ? k : best;
                }
            }
            else
            {
                for (int q = 0; q < 8 && c + q < Vs; ++q)
                {
                    unsigned long long const k
                        = make_key(__bfloat162float(row[c + q]), static_cast<uint32_t>(vocab_offset + c + q));
                    best = k > best ? k : best;
                }
            }
        }
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
            best = shfl_max(best, o);
        if (lane == 0 && best != 0ull)
            atomicMax(keys + r, best);
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int const prev = atomicAdd(counters, 1u);
        s_last = prev == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last)
        return;
    __threadfence();
    unsigned int const call = *reinterpret_cast<unsigned int volatile*>(counters + 1);
    int const half = static_cast<int>(call & 1u);
    for (int t = threadIdx.x; t < M; t += blockDim.x)
    {
        unsigned long long const mine = atomicExch(keys + t, 0ull);
        unsigned long long* slot_mc = ws_mc + (half * kMaxRows + t) * world;
        unsigned long long* slot_uc = ws_uc + (half * kMaxRows + t) * world;
        slot_mc[rank] = mine;
        unsigned long long best = 0ull;
        for (int r = 0; r < world; ++r)
        {
            unsigned long long v;
            do
            {
                v = *reinterpret_cast<unsigned long long volatile*>(slot_uc + r);
            } while (v == 0ull);
            best = v > best ? v : best;
            slot_uc[r] = 0ull;
        }
        s_tok[t] = static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(best & 0xFFFFFFFFull));
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        counters[0] = 0u;
        counters[1] = call + 1u;
    }
    if (threadIdx.x < acc.nreq)
    {
        int const b = threadIdx.x;
        int const k1 = acc.draft_len + 1;
        int n = 1;
        bool run = true;
        for (int k = 0; k < acc.draft_len; ++k)
        {
            int const tt = s_tok[b * k1 + k];
            acc.acc_tokens[b * k1 + k] = tt;
            run = run && acc.draft[b * acc.draft_len + k] == tt;
            n += run ? 1 : 0;
        }
        acc.acc_tokens[b * k1 + acc.draft_len] = s_tok[b * k1 + acc.draft_len];
        if (acc.force_vec != nullptr)
            n = acc.force_vec[b];
        else if (acc.force_total > 0)
            n = acc.force_total;
        acc.num_acc[b] = n;
        acc.gather_ids[b] = static_cast<long long>(acc.batch_idx[b]) * k1 + n - 1;
        if (acc.next_new != nullptr)
            acc.next_new[b * acc.next_stride] = s_tok[acc.batch_idx[b] * k1 + n - 1];
    }
}

} // namespace

cudaError_t invokeShardAccept(__nv_bfloat16 const* logits, int M, int Vs, long long ld, int vocabOffset,
    unsigned long long* keys, unsigned int* counters, unsigned long long* wsUc, unsigned long long* wsMc, int rank,
    int world, AcceptArgs const& acc, int numCtas, cudaStream_t stream)
{
    cudaLaunchAttribute attr[1];
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(static_cast<unsigned>(numCtas));
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(
        &cfg, shard_accept_kernel, logits, M, Vs, ld, vocabOffset, keys, counters, wsUc, wsMc, rank, world, acc);
}

} // namespace shardAccept
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
