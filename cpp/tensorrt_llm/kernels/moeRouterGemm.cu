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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moeRouterGemm.h"
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Shared-memory tiled GEMM computing logits[M, N] = act[M, K] @ weight[N, K]^T.
// See moeRouterGemm.h for the precision contract.
//
// Each block computes a [BM, BN] output tile and each thread a [TM, TN]
// micro-tile. Staging tiles are stored K-major so the global activation load is
// fully coalesced. The bf16 activation is widened to fp32 in shared memory and
// accumulated in fp32.
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void moe_router_gemm_kernel(
    float* __restrict__ out, T const* __restrict__ act, float const* __restrict__ weight, int M, int N, int K)
{
    // K-major staging tiles: As[k][m], Bs[k][n].
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    constexpr int kThreads = (BM / TM) * (BN / TN);
    constexpr int kThreadsPerRow = BN / TN;

    int const block_row = blockIdx.x * BM; // token (M) offset of this tile
    int const block_col = blockIdx.y * BN; // expert (N) offset of this tile

    int const tid = threadIdx.x;
    int const thread_col = tid % kThreadsPerRow; // which N micro-column
    int const thread_row = tid / kThreadsPerRow; // which M micro-row

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Load activation tile [BM, BK] (bf16 -> fp32), stored K-major in smem.
#pragma unroll
        for (int idx = tid; idx < BM * BK; idx += kThreads)
        {
            int const r = idx / BK; // row within tile (token)
            int const c = idx % BK; // col within tile (k)
            int const gr = block_row + r;
            int const gc = k0 + c;
            float v = 0.0f;
            if (gr < M && gc < K)
            {
                v = static_cast<float>(act[static_cast<int64_t>(gr) * K + gc]);
            }
            As[c * BM + r] = v;
        }
        // Load weight tile [BN, BK] (fp32), stored K-major in smem.
#pragma unroll
        for (int idx = tid; idx < BN * BK; idx += kThreads)
        {
            int const r = idx / BK; // row within tile (expert)
            int const c = idx % BK; // col within tile (k)
            int const gr = block_col + r;
            int const gc = k0 + c;
            float v = 0.0f;
            if (gr < N && gc < K)
            {
                v = weight[static_cast<int64_t>(gr) * K + gc];
            }
            Bs[c * BN + r] = v;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            float a_frag[TM];
            float b_frag[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                a_frag[i] = As[kk * BM + thread_row * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                b_frag[j] = Bs[kk * BN + thread_col * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int const gr = block_row + thread_row * TM + i;
        if (gr >= M)
        {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int const gc = block_col + thread_col * TN + j;
            if (gc < N)
            {
                out[static_cast<int64_t>(gr) * N + gc] = acc[i][j];
            }
        }
    }
}

template <typename T>
void invokeMoeRouterGemm(float* output, T const* act, float const* weight, int num_tokens, int num_experts,
    int hidden_dim, cudaStream_t stream)
{
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int TM = 4;
    constexpr int TN = 8;
    constexpr int kThreads = (BM / TM) * (BN / TN); // 256

    dim3 const grid((num_tokens + BM - 1) / BM, (num_experts + BN - 1) / BN);
    dim3 const block(kThreads);
    moe_router_gemm_kernel<T, BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(output, act, weight, num_tokens, num_experts, hidden_dim);
}

template void invokeMoeRouterGemm<__nv_bfloat16>(
    float*, __nv_bfloat16 const*, float const*, int, int, int, cudaStream_t);

template void invokeMoeRouterGemm<half>(float*, half const*, float const*, int, int, int, cudaStream_t);

} // namespace kernels

TRTLLM_NAMESPACE_END
