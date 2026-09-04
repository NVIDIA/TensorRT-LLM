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

#include "fusedSamplingKernelsCommon.cuh"

namespace tensorrt_llm
{
namespace kernels
{
namespace fusedSampling
{

namespace
{

//! The generic entry point: no __launch_bounds__, so ptxas compiles every instantiation
//! exactly as it did before the tokens-only path existed.
template <typename T, int BLOCK, bool NEED_TOKENS, bool NEED_PROBS>
__global__ void fusedSamplingKernel(FusedSamplingParams params)
{
    __shared__ FusedSamplingShared<BLOCK> shared;
    fusedSamplingBody<T, BLOCK, NEED_TOKENS, NEED_PROBS>(params, shared);
}

//! The tokens-only narrow-block entry point, and the only one carrying an occupancy bound.
//!
//! The rejection loop costs registers: 47 at BLOCK=512 against 32 for the other output
//! shapes. 512 threads x 47 registers fits 2 blocks in an SM's 65536 where 32 fits 4, and
//! and that halving costs more than the spill avoided -- including on rows that never enter
//! the loop, since top-k rows take the descent. Naming the target trades a small spill for
//! the occupancy, which is the right way round for a kernel reading half a megabyte per
//! row.
//!
//! Only the narrow block. The wide block is the few-rows regime, where a block already owns
//! its SM and capping registers would buy nothing while the spill still cost.
template <typename T>
__global__ __launch_bounds__(kNarrowBlock, kMaxThreadsPerSm / kNarrowBlock) void fusedSamplingTokensNarrowKernel(
    FusedSamplingParams params)
{
    __shared__ FusedSamplingShared<kNarrowBlock> shared;
    fusedSamplingBody<T, kNarrowBlock, true, false>(params, shared);
}

//! Launch one (block size, output shape) specialization. Both are compile-time so the
//! kernel is emitted without the half it does not need and with cub sized correctly.
template <typename T, int BLOCK>
void launchFusedSampling(FusedSamplingParams const& params, cudaStream_t stream)
{
    dim3 const grid(params.numRows);
    dim3 const block(BLOCK);
    bool const needTokens = params.outputTokens != nullptr;
    bool const needProbs = params.outputProbs != nullptr;

    if (needTokens && needProbs)
    {
        fusedSamplingKernel<T, BLOCK, true, true><<<grid, block, 0, stream>>>(params);
    }
    else if (needTokens)
    {
        if constexpr (BLOCK == kNarrowBlock)
        {
            fusedSamplingTokensNarrowKernel<T><<<grid, block, 0, stream>>>(params);
        }
        else
        {
            fusedSamplingKernel<T, BLOCK, true, false><<<grid, block, 0, stream>>>(params);
        }
    }
    else if (needProbs)
    {
        fusedSamplingKernel<T, BLOCK, false, true><<<grid, block, 0, stream>>>(params);
    }
}

} // namespace

template <typename T>
void launchFusedSamplingSingleCta(FusedSamplingParams const& params, cudaStream_t stream)
{
    // See kNarrowBlock / kWideBlock: a small batch wants a wide block to shorten each
    // row's critical path, a large one wants narrow blocks so more fit per SM.
    if (params.numRows <= kWideBlockMaxRows)
    {
        launchFusedSampling<T, kWideBlock>(params, stream);
    }
    else
    {
        launchFusedSampling<T, kNarrowBlock>(params, stream);
    }
}

template void launchFusedSamplingSingleCta<float>(FusedSamplingParams const&, cudaStream_t);
template void launchFusedSamplingSingleCta<__half>(FusedSamplingParams const&, cudaStream_t);
template void launchFusedSamplingSingleCta<__nv_bfloat16>(FusedSamplingParams const&, cudaStream_t);

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm
